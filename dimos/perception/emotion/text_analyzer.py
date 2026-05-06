# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""NLP-based text emotion analyser with a lightweight keyword fallback.

Attempts to use a HuggingFace ``transformers`` pipeline for high-quality
predictions.  Falls back to a fast keyword-matching heuristic when
``transformers`` is not installed.

Install the optional extra to enable the transformer backend::

    pip install "dimos[emotion]"
"""

from __future__ import annotations

from dimos.core.core import rpc
from dimos.core.module import Module
from dimos.core.stream import In, Out
from dimos.perception.emotion.types import (
    EMOTION_LABELS,
    Emotion,
    EmotionResult,
)
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

# ---------------------------------------------------------------------------
# Keyword fallback tables
# ---------------------------------------------------------------------------

_KEYWORD_MAP: dict[str, list[str]] = {
    "happy": ["happy", "great", "joy", "love", "wonderful", "excited", "glad", "fantastic", "amazing", "delighted"],
    "sad": ["sad", "unhappy", "sorry", "depressed", "miss", "grief", "cry", "tears", "unfortunate", "miserable"],
    "angry": ["angry", "furious", "mad", "rage", "hate", "annoyed", "frustrated", "outraged", "irritated"],
    "fear": ["scared", "afraid", "fear", "terrified", "anxious", "nervous", "worried", "panic", "dread"],
    "surprise": ["wow", "surprised", "unexpected", "shock", "unbelievable", "really", "astonished", "amazing"],
    "neutral": ["okay", "fine", "alright", "sure", "ok", "noted", "understood", "yes", "no"],
    "disgust": ["disgusting", "gross", "nasty", "revolting", "awful", "horrible", "repulsive", "yuck"],
    "contempt": ["ridiculous", "pathetic", "worthless", "contempt", "disdain", "beneath", "inferior"],
}

# HuggingFace model for zero-shot / fine-tuned emotion classification.
_DEFAULT_MODEL = "j-hartmann/emotion-english-distilroberta-base"

# Mapping from model label → Emotion value (covers common variants).
_MODEL_LABEL_MAP: dict[str, str] = {
    "joy": "happy",
    "happiness": "happy",
    "happy": "happy",
    "sadness": "sad",
    "sad": "sad",
    "anger": "angry",
    "angry": "angry",
    "fear": "fear",
    "surprise": "surprise",
    "neutral": "neutral",
    "disgust": "disgust",
    "contempt": "contempt",
}


class TextEmotionAnalyzer(Module):
    """Classifies text into one of eight emotions.

    Tries a HuggingFace transformers pipeline first; falls back to keyword
    matching when the library is unavailable.

    Streams
    -------
    text_input : In[str]
        Raw text to classify.
    text_emotion : Out[EmotionResult]
        Classified emotion with per-class scores.
    """

    text_input: In[str]
    text_emotion: Out[EmotionResult]

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        """
        Args:
            model_name: HuggingFace model identifier for emotion classification.
                Defaults to ``"j-hartmann/emotion-english-distilroberta-base"``.
        """
        super().__init__()
        self._model_name = model_name
        self._pipeline: object = None

    @rpc
    def start(self) -> None:
        super().start()
        self._pipeline = _try_load_pipeline(self._model_name)
        self._disposables.add(
            self.text_input.subscribe(self._on_text)  # type: ignore[arg-type]
        )

    @rpc
    def stop(self) -> None:
        super().stop()

    def _on_text(self, text: str) -> None:
        result = self.classify(text)
        self.text_emotion.publish(result)

    def classify(self, text: str) -> EmotionResult:
        """Classify *text* and return an :class:`EmotionResult`.

        This method may also be called directly (without streams).
        """
        if self._pipeline is not None:
            return _classify_with_pipeline(self._pipeline, text)
        return _classify_with_keywords(text)


def _try_load_pipeline(model_name: str) -> object:
    try:
        from transformers import pipeline  # type: ignore[import-untyped]

        pipe = pipeline(
            "text-classification",
            model=model_name,
            top_k=None,
            truncation=True,
        )
        logger.info("TextEmotionAnalyzer: loaded transformer model '%s'", model_name)
        return pipe
    except Exception as exc:
        logger.info(
            "TextEmotionAnalyzer: transformer pipeline unavailable (%s). "
            "Using keyword fallback.",
            exc,
        )
        return None


def _classify_with_pipeline(pipeline: object, text: str) -> EmotionResult:  # type: ignore[type-arg]
    try:
        raw = pipeline(text)  # type: ignore[call-arg, operator]
        # top_k=None returns list[list[dict]] for a single input
        entries: list[dict[str, object]] = raw[0] if isinstance(raw[0], list) else raw
        scores: dict[str, float] = {}
        for entry in entries:
            label = str(entry.get("label", "")).lower()
            score = float(entry.get("score", 0.0))
            mapped = _MODEL_LABEL_MAP.get(label, label)
            if mapped in EMOTION_LABELS:
                scores[mapped] = scores.get(mapped, 0.0) + score

        # Fill missing labels
        for label in EMOTION_LABELS:
            scores.setdefault(label, 0.0)

        # Normalise
        total = sum(scores.values()) or 1.0
        scores = {k: v / total for k, v in scores.items()}

        dominant = max(scores, key=lambda k: scores[k])
        return EmotionResult(
            emotion=Emotion(dominant),
            scores=scores,
            confidence=scores[dominant],
            source="text",
        )
    except Exception as exc:
        logger.debug("TextEmotionAnalyzer: pipeline inference error: %s", exc)
        return EmotionResult.neutral(source="text")


def _classify_with_keywords(text: str) -> EmotionResult:
    lower = text.lower()
    scores: dict[str, float] = {label: 0.0 for label in EMOTION_LABELS}
    for emotion, keywords in _KEYWORD_MAP.items():
        for kw in keywords:
            if kw in lower:
                scores[emotion] += 1.0

    total = sum(scores.values())
    if total == 0.0:
        scores["neutral"] = 1.0
        total = 1.0

    scores = {k: v / total for k, v in scores.items()}
    dominant = max(scores, key=lambda k: scores[k])
    return EmotionResult(
        emotion=Emotion(dominant),
        scores=scores,
        confidence=scores[dominant],
        source="text",
    )


text_emotion_analyzer = TextEmotionAnalyzer.blueprint

__all__ = ["TextEmotionAnalyzer", "text_emotion_analyzer"]
