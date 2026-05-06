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

"""Emotion perception package.

Provides real-time emotion detection from multiple modalities (face, voice
prosody, and text) plus a fusion layer that combines them into a single
emotion reading.

Modules
-------
FaceEmotionDetector
    Facial emotion recognition using DeepFace (optional dep).
ProsodyAnalyzer
    Voice prosody emotion inference using librosa (optional dep).
TextEmotionAnalyzer
    NLP-based emotion classification (transformers + keyword fallback).
MultimodalEmotionFusion
    Weighted-average fusion of the above sources.
"""

from dimos.perception.emotion.face_detector import FaceEmotionDetector, face_emotion_detector
from dimos.perception.emotion.fusion import MultimodalEmotionFusion, fuse, multimodal_emotion_fusion
from dimos.perception.emotion.prosody_analyzer import ProsodyAnalyzer, prosody_analyzer
from dimos.perception.emotion.text_analyzer import TextEmotionAnalyzer, text_emotion_analyzer
from dimos.perception.emotion.types import (
    DEFAULT_WEIGHTS,
    EMOTION_LABELS,
    Emotion,
    EmotionResult,
)

__all__ = [
    "Emotion",
    "EMOTION_LABELS",
    "DEFAULT_WEIGHTS",
    "EmotionResult",
    "FaceEmotionDetector",
    "face_emotion_detector",
    "ProsodyAnalyzer",
    "prosody_analyzer",
    "TextEmotionAnalyzer",
    "text_emotion_analyzer",
    "MultimodalEmotionFusion",
    "multimodal_emotion_fusion",
    "fuse",
]
