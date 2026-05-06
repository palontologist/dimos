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

"""Voice prosody analyser that infers emotion from audio features using librosa.

Install the optional extra to enable this module::

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


class ProsodyAnalyzer(Module):
    """Infers emotions from raw audio waveform data via prosody features.

    Extracts pitch (fundamental frequency), energy (RMS), tempo, and spectral
    centroid from each audio chunk, then maps these heuristically onto the
    eight-emotion vocabulary.

    Install with ``pip install "dimos[emotion]"``.

    Streams
    -------
    audio_chunk : In[bytes]
        Raw PCM audio bytes (mono float32, sample rate set by ``sample_rate``).
    prosody_emotion : Out[EmotionResult]
        Inferred emotion from voice prosody features.
    """

    audio_chunk: In[bytes]
    prosody_emotion: Out[EmotionResult]

    def __init__(self, sample_rate: int = 22050) -> None:
        """
        Args:
            sample_rate: Sample rate of the incoming PCM audio in Hz.
        """
        super().__init__()
        self._sample_rate = sample_rate

    @rpc
    def start(self) -> None:
        super().start()
        try:
            import librosa as _  # noqa: F401  # validate optional dep

        except ImportError as exc:
            raise ImportError(
                "librosa is required for ProsodyAnalyzer. "
                "Install it with: pip install 'dimos[emotion]'"
            ) from exc

        self._disposables.add(
            self.audio_chunk.subscribe(self._on_audio)  # type: ignore[arg-type]
        )

    @rpc
    def stop(self) -> None:
        super().stop()

    def _on_audio(self, chunk: bytes) -> None:
        result = self._analyse(chunk)
        if result is not None:
            self.prosody_emotion.publish(result)

    def _analyse(self, chunk: bytes) -> EmotionResult | None:
        try:
            import librosa
            import numpy as np
        except ImportError:
            return None

        try:
            y = np.frombuffer(chunk, dtype=np.float32)
            if y.size == 0:
                return None

            sr = self._sample_rate

            # --- Feature extraction ---
            rms: float = float(np.sqrt(np.mean(y**2)))
            spectral_centroid: float = float(
                np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            )

            # Fundamental frequency (pitch) via pyin
            f0, voiced_flag, _ = librosa.pyin(
                y,
                fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C7"),
                sr=sr,
            )
            pitch: float = float(np.nanmean(f0[voiced_flag])) if voiced_flag.any() else 0.0

            # Tempo
            tempo_arr, _ = librosa.beat.beat_track(y=y, sr=sr)
            tempo: float = float(np.atleast_1d(tempo_arr)[0])

        except Exception as exc:
            logger.debug("ProsodyAnalyzer: feature extraction error: %s", exc)
            return None

        scores = _features_to_scores(pitch, rms, tempo, spectral_centroid)
        dominant = max(scores, key=lambda k: scores[k])

        return EmotionResult(
            emotion=Emotion(dominant),
            scores=scores,
            confidence=scores[dominant],
            source="prosody",
        )


def _features_to_scores(
    pitch: float,
    rms: float,
    tempo: float,
    spectral_centroid: float,
) -> dict[str, float]:
    """Heuristic mapping from prosody features to emotion scores.

    The rules are intentionally simple and explainable; a production system
    would train a classifier on labelled prosody data.
    """
    scores: dict[str, float] = {label: 0.0 for label in EMOTION_LABELS}

    # High pitch + high energy → happy / surprise / fear
    # Low pitch + low energy  → sad / neutral
    # High energy + low pitch → angry
    # Very low energy         → neutral / sad

    normalised_pitch = min(pitch / 400.0, 1.0) if pitch > 0 else 0.0
    normalised_energy = min(rms / 0.1, 1.0)
    normalised_tempo = min(tempo / 200.0, 1.0)
    normalised_centroid = min(spectral_centroid / 4000.0, 1.0)

    scores["happy"] = 0.4 * normalised_pitch + 0.3 * normalised_energy + 0.3 * normalised_tempo
    scores["surprise"] = (
        0.4 * normalised_pitch + 0.2 * normalised_energy + 0.4 * normalised_centroid
    )
    scores["angry"] = (
        (1.0 - normalised_pitch) * 0.3 + normalised_energy * 0.5 + normalised_centroid * 0.2
    )
    scores["fear"] = 0.5 * normalised_pitch + 0.3 * (1.0 - normalised_energy) + 0.2 * normalised_centroid
    scores["sad"] = (
        (1.0 - normalised_pitch) * 0.4
        + (1.0 - normalised_energy) * 0.4
        + (1.0 - normalised_tempo) * 0.2
    )
    scores["disgust"] = (
        (1.0 - normalised_pitch) * 0.3
        + normalised_energy * 0.3
        + (1.0 - normalised_centroid) * 0.4
    )
    scores["contempt"] = (
        (1.0 - normalised_pitch) * 0.4
        + (1.0 - normalised_energy) * 0.3
        + (1.0 - normalised_tempo) * 0.3
    )
    scores["neutral"] = 1.0 - max(
        scores["happy"],
        scores["surprise"],
        scores["angry"],
        scores["fear"],
        scores["sad"],
    )
    scores["neutral"] = max(0.0, scores["neutral"])

    # Normalise
    total = sum(scores.values()) or 1.0
    return {k: v / total for k, v in scores.items()}


prosody_analyzer = ProsodyAnalyzer.blueprint

__all__ = ["ProsodyAnalyzer", "prosody_analyzer"]
