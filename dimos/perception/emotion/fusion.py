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

"""Multimodal emotion fusion – combines face, prosody, and text signals.

Publishes a single fused :class:`~dimos.perception.emotion.types.EmotionResult`
whenever any upstream detector produces a new result, using the most recent
reading from each source combined via weighted averaging.
"""

from __future__ import annotations

from reactivex.disposable import Disposable

from dimos.core.core import rpc
from dimos.core.module import Module
from dimos.core.stream import In, Out
from dimos.perception.emotion.types import (
    DEFAULT_WEIGHTS,
    EMOTION_LABELS,
    Emotion,
    EmotionResult,
)
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


class MultimodalEmotionFusion(Module):
    """Fuses emotion signals from multiple modalities into one result.

    Each source (face, prosody, text) has a configurable weight.  Whenever any
    upstream result arrives, the module recomputes a weighted average over the
    most recent result from every active source and publishes the fused output.

    Streams
    -------
    face_emotion : In[EmotionResult]
        Optional facial emotion signal.
    prosody_emotion : In[EmotionResult]
        Optional prosody emotion signal.
    text_emotion : In[EmotionResult]
        Optional text emotion signal.
    fused_emotion : Out[EmotionResult]
        Weighted-average fused result.
    """

    face_emotion: In[EmotionResult]
    prosody_emotion: In[EmotionResult]
    text_emotion: In[EmotionResult]
    fused_emotion: Out[EmotionResult]

    def __init__(
        self,
        face_weight: float = DEFAULT_WEIGHTS["face"],
        prosody_weight: float = DEFAULT_WEIGHTS["prosody"],
        text_weight: float = DEFAULT_WEIGHTS["text"],
    ) -> None:
        """
        Args:
            face_weight: Contribution weight for the face detector (default 0.5).
            prosody_weight: Contribution weight for the prosody analyser (default 0.3).
            text_weight: Contribution weight for the text analyser (default 0.2).
        """
        super().__init__()
        self._weights: dict[str, float] = {
            "face": face_weight,
            "prosody": prosody_weight,
            "text": text_weight,
        }
        self._latest: dict[str, EmotionResult] = {}

    @rpc
    def start(self) -> None:
        super().start()
        self._disposables.add(
            Disposable(self.face_emotion.subscribe(self._on_face))  # type: ignore[arg-type]
        )
        self._disposables.add(
            Disposable(self.prosody_emotion.subscribe(self._on_prosody))  # type: ignore[arg-type]
        )
        self._disposables.add(
            Disposable(self.text_emotion.subscribe(self._on_text))  # type: ignore[arg-type]
        )

    @rpc
    def stop(self) -> None:
        super().stop()

    # ------------------------------------------------------------------
    # Internal handlers
    # ------------------------------------------------------------------

    def _on_face(self, result: EmotionResult) -> None:
        self._latest["face"] = result
        self._publish_fused()

    def _on_prosody(self, result: EmotionResult) -> None:
        self._latest["prosody"] = result
        self._publish_fused()

    def _on_text(self, result: EmotionResult) -> None:
        self._latest["text"] = result
        self._publish_fused()

    def _publish_fused(self) -> None:
        fused = fuse(self._latest, self._weights)
        self.fused_emotion.publish(fused)


# ---------------------------------------------------------------------------
# Pure fusion function (reusable without the Module)
# ---------------------------------------------------------------------------


def fuse(
    results: dict[str, EmotionResult],
    weights: dict[str, float] | None = None,
) -> EmotionResult:
    """Fuse multiple :class:`EmotionResult` objects via weighted averaging.

    Args:
        results: Mapping of source name (``"face"``, ``"prosody"``, ``"text"``)
            to the most recent :class:`EmotionResult` for that source.
        weights: Per-source weight.  Defaults to :data:`DEFAULT_WEIGHTS`.

    Returns:
        A new :class:`EmotionResult` with ``source="fused"``.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    if not results:
        return EmotionResult.neutral(source="fused")

    fused_scores: dict[str, float] = {label: 0.0 for label in EMOTION_LABELS}
    total_weight = 0.0

    for source, result in results.items():
        w = weights.get(source, 1.0)
        total_weight += w
        for label in EMOTION_LABELS:
            fused_scores[label] += w * result.scores.get(label, 0.0)

    if total_weight > 0:
        fused_scores = {k: v / total_weight for k, v in fused_scores.items()}

    dominant = max(fused_scores, key=lambda k: fused_scores[k])
    return EmotionResult(
        emotion=Emotion(dominant),
        scores=fused_scores,
        confidence=fused_scores[dominant],
        source="fused",
    )


multimodal_emotion_fusion = MultimodalEmotionFusion.blueprint

__all__ = ["MultimodalEmotionFusion", "fuse", "multimodal_emotion_fusion"]
