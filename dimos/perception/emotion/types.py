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

from dataclasses import dataclass, field
from enum import Enum


class Emotion(str, Enum):
    """The eight canonical emotion categories used across the engine."""

    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    FEAR = "fear"
    SURPRISE = "surprise"
    NEUTRAL = "neutral"
    DISGUST = "disgust"
    CONTEMPT = "contempt"


EMOTION_LABELS: list[str] = [e.value for e in Emotion]

# Default per-source weights used by MultimodalEmotionFusion.
DEFAULT_WEIGHTS: dict[str, float] = {
    "face": 0.5,
    "prosody": 0.3,
    "text": 0.2,
}


@dataclass
class EmotionResult:
    """Result produced by any single emotion detector or the fusion module.

    Attributes:
        emotion: The dominant emotion label.
        scores: Mapping of every emotion label to its normalised [0, 1] score.
        confidence: Confidence of the dominant emotion (0–1).
        source: Which detector produced this result: "face", "prosody", "text",
            or "fused".
    """

    emotion: Emotion
    scores: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    source: str = "unknown"

    def __post_init__(self) -> None:
        # Ensure all canonical emotions are present in scores.
        for label in EMOTION_LABELS:
            self.scores.setdefault(label, 0.0)

    @classmethod
    def neutral(cls, source: str = "unknown") -> "EmotionResult":
        """Return a flat neutral result (all scores equal)."""
        score = 1.0 / len(EMOTION_LABELS)
        return cls(
            emotion=Emotion.NEUTRAL,
            scores={label: score for label in EMOTION_LABELS},
            confidence=score,
            source=source,
        )


__all__ = ["Emotion", "EMOTION_LABELS", "DEFAULT_WEIGHTS", "EmotionResult"]
