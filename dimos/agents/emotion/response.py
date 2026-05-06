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

"""Response adaptation layer for emotion-aware agents.

Defines the vocabulary of response styles, strategies, and physical/vocal
properties, and provides a factory that selects appropriate parameters based
on the detected emotion and context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from dimos.perception.emotion.types import Emotion, EmotionResult


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ResponseStyle(str, Enum):
    """High-level tone / register for generated responses."""

    PROFESSIONAL = "professional"
    CASUAL = "casual"
    HUMOROUS = "humorous"
    EMPATHETIC = "empathetic"
    ENCOURAGING = "encouraging"
    CALM = "calm"
    DIRECT = "direct"


class ResponseStrategy(str, Enum):
    """Action strategy that shapes *how* the robot responds."""

    MATCH_ENERGY = "match_energy"  # Mirror the human's emotional energy.
    CALM = "calm"                   # De-escalate; speak slowly and softly.
    ENCOURAGE = "encourage"         # Uplift and motivate.
    CLARIFY = "clarify"             # Seek clarification when confused.
    HUMOR = "humor"                 # Use light humour to ease tension.
    EMPATHIZE = "empathize"         # Acknowledge and validate feelings.


# ---------------------------------------------------------------------------
# Physical / vocal property descriptors
# ---------------------------------------------------------------------------


@dataclass
class VocalProperties:
    """Target vocal characteristics for the TTS / speech synthesizer.

    Attributes:
        pitch: Relative pitch adjustment (-1 = very low, 0 = normal, +1 = high).
        speed: Speech rate multiplier (1.0 = normal).
        energy: Loudness/energy level (0 = whisper, 1 = normal, 2 = loud).
        tone: Qualitative tonal description passed to the TTS prompt.
    """

    pitch: float = 0.0
    speed: float = 1.0
    energy: float = 1.0
    tone: str = "neutral"


@dataclass
class ActionProperties:
    """Suggested physical action properties for embodied robots.

    Attributes:
        gesture: Recommended gesture category (e.g. ``"wave"``, ``"nod"``).
        duration: Suggested gesture/action duration in seconds.
        intensity: Action intensity level (0–1).
    """

    gesture: str = "none"
    duration: float = 1.0
    intensity: float = 0.5


# ---------------------------------------------------------------------------
# Response parameters bundle
# ---------------------------------------------------------------------------


@dataclass
class ResponseParameters:
    """Full set of adaptation parameters for a single interaction turn.

    Attributes:
        style: Selected response style.
        strategy: Selected response strategy.
        vocal: Target vocal properties.
        action: Target action properties.
        system_prompt_addon: Optional text appended to the LLM system prompt
            to steer the response toward the desired style/strategy.
        metadata: Arbitrary extra fields for extensibility.
    """

    style: ResponseStyle = ResponseStyle.PROFESSIONAL
    strategy: ResponseStrategy = ResponseStrategy.MATCH_ENERGY
    vocal: VocalProperties = field(default_factory=VocalProperties)
    action: ActionProperties = field(default_factory=ActionProperties)
    system_prompt_addon: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Emotion → response mapping tables
# ---------------------------------------------------------------------------

# Maps each emotion to its default (style, strategy, vocal, action) parameters.
_EMOTION_DEFAULTS: dict[str, tuple[ResponseStyle, ResponseStrategy, VocalProperties, ActionProperties]] = {
    Emotion.HAPPY.value: (
        ResponseStyle.CASUAL,
        ResponseStrategy.MATCH_ENERGY,
        VocalProperties(pitch=0.2, speed=1.1, energy=1.2, tone="cheerful"),
        ActionProperties(gesture="wave", duration=1.0, intensity=0.7),
    ),
    Emotion.SAD.value: (
        ResponseStyle.EMPATHETIC,
        ResponseStrategy.EMPATHIZE,
        VocalProperties(pitch=-0.2, speed=0.9, energy=0.7, tone="gentle"),
        ActionProperties(gesture="nod", duration=1.5, intensity=0.3),
    ),
    Emotion.ANGRY.value: (
        ResponseStyle.CALM,
        ResponseStrategy.CALM,
        VocalProperties(pitch=-0.1, speed=0.85, energy=0.8, tone="soothing"),
        ActionProperties(gesture="open_hands", duration=2.0, intensity=0.3),
    ),
    Emotion.FEAR.value: (
        ResponseStyle.CALM,
        ResponseStrategy.ENCOURAGE,
        VocalProperties(pitch=0.0, speed=0.9, energy=0.8, tone="reassuring"),
        ActionProperties(gesture="nod", duration=1.5, intensity=0.4),
    ),
    Emotion.SURPRISE.value: (
        ResponseStyle.CASUAL,
        ResponseStrategy.CLARIFY,
        VocalProperties(pitch=0.1, speed=1.0, energy=1.0, tone="curious"),
        ActionProperties(gesture="tilt_head", duration=1.0, intensity=0.5),
    ),
    Emotion.NEUTRAL.value: (
        ResponseStyle.PROFESSIONAL,
        ResponseStrategy.MATCH_ENERGY,
        VocalProperties(pitch=0.0, speed=1.0, energy=1.0, tone="neutral"),
        ActionProperties(gesture="none", duration=0.5, intensity=0.5),
    ),
    Emotion.DISGUST.value: (
        ResponseStyle.DIRECT,
        ResponseStrategy.CLARIFY,
        VocalProperties(pitch=-0.1, speed=0.95, energy=0.9, tone="measured"),
        ActionProperties(gesture="none", duration=0.5, intensity=0.4),
    ),
    Emotion.CONTEMPT.value: (
        ResponseStyle.PROFESSIONAL,
        ResponseStrategy.CLARIFY,
        VocalProperties(pitch=-0.1, speed=0.9, energy=0.85, tone="measured"),
        ActionProperties(gesture="none", duration=0.5, intensity=0.4),
    ),
}

# Style → system-prompt sentence fragment.
_STYLE_PROMPTS: dict[ResponseStyle, str] = {
    ResponseStyle.PROFESSIONAL: "Respond in a clear, professional manner.",
    ResponseStyle.CASUAL: "Respond in a warm, conversational tone.",
    ResponseStyle.HUMOROUS: "Use light, appropriate humour to keep things fun.",
    ResponseStyle.EMPATHETIC: "Acknowledge the human's feelings with genuine empathy.",
    ResponseStyle.ENCOURAGING: "Be uplifting and motivating in your response.",
    ResponseStyle.CALM: "Use a calm, measured tone to help de-escalate tension.",
    ResponseStyle.DIRECT: "Be concise and factual.",
}

# Strategy → system-prompt sentence fragment.
_STRATEGY_PROMPTS: dict[ResponseStrategy, str] = {
    ResponseStrategy.MATCH_ENERGY: "Mirror the human's energy level appropriately.",
    ResponseStrategy.CALM: "Speak slowly and gently to help the human calm down.",
    ResponseStrategy.ENCOURAGE: "Motivate and encourage the human.",
    ResponseStrategy.CLARIFY: "Ask a clarifying question if something is unclear.",
    ResponseStrategy.HUMOR: "Lighten the mood with a tasteful joke if appropriate.",
    ResponseStrategy.EMPATHIZE: "Validate the human's emotional experience before responding.",
}


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def adapt_response(
    emotion_result: EmotionResult,
    context: dict[str, Any] | None = None,
) -> ResponseParameters:
    """Select :class:`ResponseParameters` appropriate for *emotion_result*.

    Args:
        emotion_result: The current fused or per-modality emotion reading.
        context: Optional additional context (e.g. from
            :class:`~dimos.agents.emotion.context.SituationalContextManager`)
            that may influence the adaptation.

    Returns:
        A populated :class:`ResponseParameters` instance.
    """
    emotion_key = emotion_result.emotion.value
    style, strategy, vocal, action = _EMOTION_DEFAULTS.get(
        emotion_key,
        _EMOTION_DEFAULTS[Emotion.NEUTRAL.value],
    )

    # Allow context to override strategy (e.g. user explicitly requests help).
    if context:
        if context.get("user_needs_help"):
            strategy = ResponseStrategy.CLARIFY
        if context.get("high_stress"):
            strategy = ResponseStrategy.CALM

    system_prompt_addon = (
        f"{_STYLE_PROMPTS[style]} {_STRATEGY_PROMPTS[strategy]} "
        f"(Detected emotion: {emotion_key}, confidence: {emotion_result.confidence:.2f})"
    )

    return ResponseParameters(
        style=style,
        strategy=strategy,
        vocal=vocal,
        action=action,
        system_prompt_addon=system_prompt_addon,
        metadata={"emotion": emotion_key, "confidence": emotion_result.confidence},
    )


__all__ = [
    "ResponseStyle",
    "ResponseStrategy",
    "VocalProperties",
    "ActionProperties",
    "ResponseParameters",
    "adapt_response",
]
