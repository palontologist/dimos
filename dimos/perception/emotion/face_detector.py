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

"""Real-time facial emotion detector using DeepFace (optional dependency).

Install the optional extra to enable this module::

    pip install "dimos[emotion]"
"""

from __future__ import annotations

import numpy as np

from dimos.core.core import rpc
from dimos.core.module import Module
from dimos.core.stream import In, Out
from dimos.msgs.sensor_msgs import Image
from dimos.perception.emotion.types import (
    EMOTION_LABELS,
    Emotion,
    EmotionResult,
)
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

# DeepFace label → Emotion mapping.  DeepFace uses "fear" not "scared".
_DEEPFACE_MAP: dict[str, str] = {
    "happy": "happy",
    "sad": "sad",
    "angry": "angry",
    "fear": "fear",
    "surprise": "surprise",
    "neutral": "neutral",
    "disgust": "disgust",
    "contempt": "contempt",
}


class FaceEmotionDetector(Module):
    """Detects facial emotions from a live colour image stream.

    Uses `DeepFace <https://github.com/serengil/deepface>`_ for inference.
    Install with ``pip install "dimos[emotion]"``.

    Streams
    -------
    color_image : In[Image]
        Input RGB/BGR frame.
    face_emotion : Out[EmotionResult]
        Dominant facial emotion with per-class scores.
    """

    color_image: In[Image]
    face_emotion: Out[EmotionResult]

    def __init__(self, detector_backend: str = "opencv") -> None:
        """
        Args:
            detector_backend: Face detector backend passed to DeepFace
                (``"opencv"``, ``"mtcnn"``, ``"retinaface"`` …).
        """
        super().__init__()
        self._detector_backend = detector_backend
        self._analyze: object = None  # populated in start()

    @rpc
    def start(self) -> None:
        super().start()
        try:
            from deepface import DeepFace  # type: ignore[import-untyped]

            self._analyze = DeepFace.analyze
        except ImportError as exc:
            raise ImportError(
                "deepface is required for FaceEmotionDetector. "
                "Install it with: pip install 'dimos[emotion]'"
            ) from exc

        self._disposables.add(
            self.color_image.subscribe(self._on_frame)  # type: ignore[arg-type]
        )

    @rpc
    def stop(self) -> None:
        super().stop()

    def _on_frame(self, image: Image) -> None:
        result = self._detect(image)
        if result is not None:
            self.face_emotion.publish(result)

    def _detect(self, image: Image) -> EmotionResult | None:
        if self._analyze is None:
            return None

        # Convert Image to numpy array expected by DeepFace.
        frame: np.ndarray[np.Any, np.dtype[np.uint8]]
        if hasattr(image, "data") and isinstance(image.data, (bytes, bytearray)):
            import cv2

            buf = np.frombuffer(image.data, dtype=np.uint8)
            frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        elif isinstance(image, np.ndarray):
            frame = image  # type: ignore[assignment]
        else:
            logger.warning("FaceEmotionDetector: unsupported image type %s", type(image))
            return None

        try:
            results = self._analyze(  # type: ignore[call-arg]
                img_path=frame,
                actions=["emotion"],
                enforce_detection=False,
                detector_backend=self._detector_backend,
                silent=True,
            )
        except Exception as exc:
            logger.debug("FaceEmotionDetector: DeepFace error: %s", exc)
            return None

        if not results:
            return None

        # DeepFace returns a list; take the first (most prominent) face.
        face = results[0] if isinstance(results, list) else results
        raw_scores: dict[str, float] = face.get("emotion", {})
        dominant: str = face.get("dominant_emotion", "neutral")

        scores = _normalise_scores(
            {_DEEPFACE_MAP.get(k.lower(), k.lower()): v for k, v in raw_scores.items()}
        )
        emotion = Emotion(dominant.lower()) if dominant.lower() in Emotion._value2member_map_ else Emotion.NEUTRAL

        return EmotionResult(
            emotion=emotion,
            scores=scores,
            confidence=scores.get(emotion.value, 0.0),
            source="face",
        )


def _normalise_scores(raw: dict[str, float]) -> dict[str, float]:
    """Return scores normalised to [0, 1] summing to 1."""
    total = sum(raw.values()) or 1.0
    result = {label: raw.get(label, 0.0) / total for label in EMOTION_LABELS}
    return result


face_emotion_detector = FaceEmotionDetector.blueprint

__all__ = ["FaceEmotionDetector", "face_emotion_detector"]
