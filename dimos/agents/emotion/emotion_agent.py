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

"""Emotion-aware agent that extends :class:`~dimos.agents.agent.Agent`.

The agent subscribes to fused emotion readings, maintains a
:class:`~dimos.agents.emotion.context.SituationalContextManager`, augments
every incoming human message with emotion metadata, and dynamically adjusts
the system prompt based on detected emotion and response strategy.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from threading import RLock
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.messages.base import BaseMessage
from reactivex.disposable import Disposable

from dimos.agents.agent import Agent, AgentConfig
from dimos.agents.emotion.context import EmotionalState, SituationalContextManager
from dimos.agents.emotion.response import ResponseParameters, adapt_response
from dimos.agents.system_prompt import SYSTEM_PROMPT
from dimos.core.core import rpc
from dimos.core.stream import In
from dimos.perception.emotion.types import EmotionResult
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class EmotionAwareAgentConfig(AgentConfig):
    """Configuration for :class:`EmotionAwareAgent`.

    Attributes:
        emotion_augment_messages: When ``True``, inject emotion metadata into
            each human message before passing it to the LLM.
        context_decay_rate: Exponential decay rate for the emotional state.
        context_history_maxlen: Number of past emotion readings to retain.
    """

    emotion_augment_messages: bool = True
    context_decay_rate: float = 0.05
    context_history_maxlen: int = 100


# ---------------------------------------------------------------------------
# EmotionAwareAgent
# ---------------------------------------------------------------------------


class EmotionAwareAgent(Agent):
    """An :class:`~dimos.agents.agent.Agent` with emotion-awareness.

    Additional streams
    ------------------
    fused_emotion : In[EmotionResult]
        Accepts emotion readings from
        :class:`~dimos.perception.emotion.fusion.MultimodalEmotionFusion` or
        any individual detector.

    Features
    --------
    * Maintains a :class:`SituationalContextManager` with exponential decay.
    * Augments each human message with a brief emotion summary when
      ``emotion_augment_messages`` is ``True``.
    * Adjusts the *dynamic* system prompt for every turn based on the
      current :class:`~dimos.agents.emotion.response.ResponseParameters`.
    * Fires optional callbacks whenever the emotional state changes.
    """

    default_config = EmotionAwareAgentConfig

    fused_emotion: In[EmotionResult]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._emotion_lock = RLock()
        self._context_mgr = SituationalContextManager(
            history_maxlen=self.config.context_history_maxlen,
            decay_rate=self.config.context_decay_rate,
        )
        self._current_response_params: ResponseParameters | None = None
        # Callbacks: fn(EmotionalState) -> None
        self._emotion_callbacks: list[Callable[[EmotionalState], None]] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @rpc
    def start(self) -> None:
        super().start()
        self._disposables.add(
            Disposable(self.fused_emotion.subscribe(self._on_emotion))  # type: ignore[arg-type]
        )

    @rpc
    def stop(self) -> None:
        super().stop()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_emotion_callback(self, fn: Callable[[EmotionalState], None]) -> None:
        """Register a callback invoked whenever the emotional state updates.

        Args:
            fn: Callable receiving the new :class:`EmotionalState`.
        """
        with self._emotion_lock:
            self._emotion_callbacks.append(fn)

    def get_emotional_state(self) -> EmotionalState:
        """Return the current emotional state snapshot."""
        return self._context_mgr.get_state()

    def get_emotion_history(self, n: int | None = None) -> list[EmotionResult]:
        """Return the last *n* emotion readings."""
        return self._context_mgr.get_history(n)

    def set_context(
        self,
        key: str,
        value: Any,
        importance: float = 0.5,
        ttl: float | None = None,
    ) -> None:
        """Store an arbitrary context item (thread-safe)."""
        self._context_mgr.set_context(key, value, importance=importance, ttl=ttl)

    def get_context(self, key: str, default: Any = None) -> Any:
        """Retrieve a stored context item by key."""
        return self._context_mgr.get_context(key, default)

    # ------------------------------------------------------------------
    # Internal handlers
    # ------------------------------------------------------------------

    def _on_emotion(self, result: EmotionResult) -> None:
        with self._emotion_lock:
            state = self._context_mgr.update_emotion(result)
            context = self._context_mgr.all_context()
            self._current_response_params = adapt_response(result, context)
            callbacks = list(self._emotion_callbacks)

        logger.debug(
            "EmotionAwareAgent: emotion=%s confidence=%.2f",
            state.emotion.value,
            state.confidence,
        )

        for cb in callbacks:
            try:
                cb(state)
            except Exception as exc:
                logger.warning("EmotionAwareAgent: callback error: %s", exc)

    def _process_message(
        self,
        state_graph: Any,
        message: BaseMessage,
    ) -> None:
        """Override to inject emotion metadata and adapt system prompt."""
        augmented = self._augment_message(message)
        dynamic_system = self._build_dynamic_system_message()

        # Prepend a fresh system message reflecting the current emotional context.
        if dynamic_system is not None:
            # Replace the first SystemMessage in history if present, else prepend.
            history_with_system = _inject_system(self._history, dynamic_system)
        else:
            history_with_system = self._history

        self.agent_idle.publish(False)
        self._history.append(augmented)
        history_with_system.append(augmented)

        from dimos.agents.utils import pretty_print_langchain_message

        pretty_print_langchain_message(augmented)
        self.agent.publish(augmented)

        # Stream using the dynamically modified history.
        for update in state_graph.stream(
            {"messages": history_with_system}, stream_mode="updates"
        ):
            for node_output in update.values():
                for msg in node_output.get("messages", []):
                    self._history.append(msg)
                    pretty_print_langchain_message(msg)
                    self.agent.publish(msg)

        if self._message_queue.empty():
            self.agent_idle.publish(True)

    def _augment_message(self, message: BaseMessage) -> BaseMessage:
        """Inject a brief emotion summary into a human text message."""
        if not self.config.emotion_augment_messages:
            return message
        if not isinstance(message, HumanMessage):
            return message
        if not isinstance(message.content, str):
            return message

        with self._emotion_lock:
            state = self._context_mgr.get_state()

        suffix = (
            f" [Detected emotion: {state.emotion.value}, "
            f"confidence: {state.confidence:.2f}]"
        )
        return HumanMessage(content=message.content + suffix)

    def _build_dynamic_system_message(self) -> SystemMessage | None:
        """Build a system message that encodes the current response strategy."""
        with self._emotion_lock:
            params = self._current_response_params

        if params is None:
            return None

        base = self.config.system_prompt or SYSTEM_PROMPT or ""
        addon = params.system_prompt_addon
        return SystemMessage(content=f"{base}\n\n{addon}".strip())


def _inject_system(
    history: list[BaseMessage], system_msg: SystemMessage
) -> list[BaseMessage]:
    """Return a copy of *history* with the first SystemMessage replaced.

    If no SystemMessage is found, prepends *system_msg*.
    """
    result: list[BaseMessage] = []
    injected = False
    for msg in history:
        if isinstance(msg, SystemMessage) and not injected:
            result.append(system_msg)
            injected = True
        else:
            result.append(msg)
    if not injected:
        result = [system_msg] + result
    return result


emotion_aware_agent = EmotionAwareAgent.blueprint

__all__ = [
    "EmotionAwareAgentConfig",
    "EmotionAwareAgent",
    "emotion_aware_agent",
]
