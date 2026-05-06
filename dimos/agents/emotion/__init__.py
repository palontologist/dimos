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

"""Emotion-aware agent package.

Provides:

* :class:`~dimos.agents.emotion.context.EmotionalState`
* :class:`~dimos.agents.emotion.context.ContextItem`
* :class:`~dimos.agents.emotion.context.SituationalContextManager`
* :class:`~dimos.agents.emotion.response.ResponseParameters`
* :func:`~dimos.agents.emotion.response.adapt_response`
* :class:`~dimos.agents.emotion.emotion_agent.EmotionAwareAgent`
"""

from dimos.agents.emotion.context import ContextItem, EmotionalState, SituationalContextManager
from dimos.agents.emotion.emotion_agent import (
    EmotionAwareAgent,
    EmotionAwareAgentConfig,
    emotion_aware_agent,
)
from dimos.agents.emotion.response import (
    ActionProperties,
    ResponseParameters,
    ResponseStrategy,
    ResponseStyle,
    VocalProperties,
    adapt_response,
)

__all__ = [
    "EmotionalState",
    "ContextItem",
    "SituationalContextManager",
    "ResponseStyle",
    "ResponseStrategy",
    "VocalProperties",
    "ActionProperties",
    "ResponseParameters",
    "adapt_response",
    "EmotionAwareAgentConfig",
    "EmotionAwareAgent",
    "emotion_aware_agent",
]
