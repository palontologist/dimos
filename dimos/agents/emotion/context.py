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

"""Situational context manager for emotion-aware agents.

Provides:

* :class:`EmotionalState` – persistent emotion representation with exponential
  decay over time so stale readings lose influence.
* :class:`ContextItem` – a typed key/value store entry with TTL and importance.
* :class:`SituationalContextManager` – thread-safe container that tracks the
  current emotional state, arbitrary context items, and interaction history.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from dimos.perception.emotion.types import Emotion, EmotionResult


# ---------------------------------------------------------------------------
# EmotionalState
# ---------------------------------------------------------------------------


@dataclass
class EmotionalState:
    """Tracks a smoothed emotional state that decays toward neutral over time.

    Attributes:
        emotion: Current dominant emotion.
        scores: Per-emotion confidence scores (sum ≈ 1).
        confidence: Confidence of the dominant emotion.
        last_updated: Unix timestamp of the most recent update.
        decay_rate: Exponential decay rate per second.  A value of 0.1 halves
            influence after ~7 seconds.
    """

    emotion: Emotion = Emotion.NEUTRAL
    scores: dict[str, float] = field(default_factory=lambda: {e.value: 1.0 / 8 for e in Emotion})
    confidence: float = 0.125
    last_updated: float = field(default_factory=time.time)
    decay_rate: float = 0.05

    def update(self, result: EmotionResult) -> None:
        """Blend *result* into the current state using exponential smoothing."""
        now = time.time()
        elapsed = now - self.last_updated
        decay = math.exp(-self.decay_rate * elapsed)

        flat = 1.0 / len(self.scores)
        decayed_scores = {
            k: v * decay + flat * (1.0 - decay) for k, v in self.scores.items()
        }
        # Blend with the new result
        alpha = result.confidence
        for k in self.scores:
            new_val = result.scores.get(k, flat)
            self.scores[k] = (1.0 - alpha) * decayed_scores[k] + alpha * new_val

        # Normalise
        total = sum(self.scores.values()) or 1.0
        self.scores = {k: v / total for k, v in self.scores.items()}

        self.emotion = Emotion(max(self.scores, key=lambda k: self.scores[k]))
        self.confidence = self.scores[self.emotion.value]
        self.last_updated = now

    def current(self) -> EmotionResult:
        """Return an :class:`EmotionResult` representing the current state."""
        return EmotionResult(
            emotion=self.emotion,
            scores=dict(self.scores),
            confidence=self.confidence,
            source="state",
        )


# ---------------------------------------------------------------------------
# ContextItem
# ---------------------------------------------------------------------------


@dataclass
class ContextItem:
    """A key/value context entry with an optional TTL and importance weight.

    Attributes:
        key: Unique identifier for the context entry.
        value: Arbitrary data.
        importance: 0–1 weight controlling retention during pruning.
        ttl: Time-to-live in seconds.  ``None`` means the item never expires.
        created_at: Unix timestamp when the item was created.
    """

    key: str
    value: Any
    importance: float = 0.5
    ttl: float | None = None
    created_at: float = field(default_factory=time.time)

    def is_expired(self) -> bool:
        """Return ``True`` if this item has exceeded its TTL."""
        if self.ttl is None:
            return False
        return (time.time() - self.created_at) > self.ttl


# ---------------------------------------------------------------------------
# SituationalContextManager
# ---------------------------------------------------------------------------


class SituationalContextManager:
    """Thread-safe manager for emotional state, context items, and history.

    Args:
        history_maxlen: Maximum number of past emotion readings to retain.
        max_context_items: Hard cap on stored context items; least-important
            items are pruned when the limit is reached.
        decay_rate: Decay rate forwarded to :class:`EmotionalState`.
    """

    def __init__(
        self,
        history_maxlen: int = 100,
        max_context_items: int = 200,
        decay_rate: float = 0.05,
    ) -> None:
        self._lock = threading.RLock()
        self._state = EmotionalState(decay_rate=decay_rate)
        self._context: dict[str, ContextItem] = {}
        self._history: deque[EmotionResult] = deque(maxlen=history_maxlen)
        self._max_context_items = max_context_items

    # ------------------------------------------------------------------
    # Emotional state
    # ------------------------------------------------------------------

    def update_emotion(self, result: EmotionResult) -> EmotionalState:
        """Update the emotional state with a new detection result.

        Returns the updated :class:`EmotionalState`.
        """
        with self._lock:
            self._state.update(result)
            self._history.append(result)
            return self.get_state()

    def get_state(self) -> EmotionalState:
        """Return a copy-safe snapshot of the current emotional state."""
        with self._lock:
            return EmotionalState(
                emotion=self._state.emotion,
                scores=dict(self._state.scores),
                confidence=self._state.confidence,
                last_updated=self._state.last_updated,
                decay_rate=self._state.decay_rate,
            )

    def get_history(self, n: int | None = None) -> list[EmotionResult]:
        """Return the last *n* emotion readings (or all if *n* is ``None``)."""
        with self._lock:
            history = list(self._history)
            return history[-n:] if n is not None else history

    # ------------------------------------------------------------------
    # Context items
    # ------------------------------------------------------------------

    def set_context(
        self,
        key: str,
        value: Any,
        importance: float = 0.5,
        ttl: float | None = None,
    ) -> None:
        """Insert or update a context item.

        Args:
            key: Identifier.
            value: Arbitrary value to store.
            importance: 0–1 retention priority (higher = kept longer).
            ttl: Optional time-to-live in seconds.
        """
        with self._lock:
            self._context[key] = ContextItem(
                key=key, value=value, importance=importance, ttl=ttl
            )
            self._prune()

    def get_context(self, key: str, default: Any = None) -> Any:
        """Retrieve the value of a context item by *key*."""
        with self._lock:
            item = self._context.get(key)
            if item is None or item.is_expired():
                return default
            return item.value

    def remove_context(self, key: str) -> None:
        """Remove a context item."""
        with self._lock:
            self._context.pop(key, None)

    def all_context(self) -> dict[str, Any]:
        """Return all non-expired context items as a plain ``{key: value}`` dict."""
        with self._lock:
            self._prune_expired()
            return {k: v.value for k, v in self._context.items()}

    # ------------------------------------------------------------------
    # Pruning
    # ------------------------------------------------------------------

    def _prune_expired(self) -> None:
        expired = [k for k, v in self._context.items() if v.is_expired()]
        for k in expired:
            del self._context[k]

    def _prune(self) -> None:
        self._prune_expired()
        if len(self._context) > self._max_context_items:
            # Remove lowest-importance items until within limit.
            sorted_keys = sorted(self._context, key=lambda k: self._context[k].importance)
            excess = len(self._context) - self._max_context_items
            for k in sorted_keys[:excess]:
                del self._context[k]


__all__ = [
    "EmotionalState",
    "ContextItem",
    "SituationalContextManager",
]
