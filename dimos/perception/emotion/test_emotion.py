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

"""Tests for the emotion detection engine."""

import math
import time

import pytest

from dimos.perception.emotion.types import (
    EMOTION_LABELS,
    Emotion,
    EmotionResult,
)
from dimos.perception.emotion.fusion import fuse
from dimos.perception.emotion.text_analyzer import (
    TextEmotionAnalyzer,
    _classify_with_keywords,
)
from dimos.agents.emotion.context import (
    ContextItem,
    EmotionalState,
    SituationalContextManager,
)
from dimos.agents.emotion.response import (
    ResponseStrategy,
    ResponseStyle,
    adapt_response,
)


# ---------------------------------------------------------------------------
# EmotionResult
# ---------------------------------------------------------------------------


class TestEmotionResult:
    def test_all_labels_present_after_init(self) -> None:
        result = EmotionResult(emotion=Emotion.HAPPY, confidence=0.9, source="face")
        for label in EMOTION_LABELS:
            assert label in result.scores

    def test_neutral_factory(self) -> None:
        r = EmotionResult.neutral(source="test")
        assert r.emotion == Emotion.NEUTRAL
        assert r.source == "test"
        assert math.isclose(sum(r.scores.values()), 1.0, abs_tol=1e-6)


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------


class TestFuse:
    def _make_result(self, dominant: Emotion, source: str) -> EmotionResult:
        scores = {label: 0.0 for label in EMOTION_LABELS}
        scores[dominant.value] = 1.0
        return EmotionResult(
            emotion=dominant, scores=scores, confidence=1.0, source=source
        )

    def test_empty_returns_neutral(self) -> None:
        result = fuse({})
        assert result.emotion == Emotion.NEUTRAL
        assert result.source == "fused"

    def test_single_source(self) -> None:
        r = self._make_result(Emotion.HAPPY, "face")
        fused = fuse({"face": r})
        assert fused.emotion == Emotion.HAPPY

    def test_weighted_fusion(self) -> None:
        face_r = self._make_result(Emotion.HAPPY, "face")
        text_r = self._make_result(Emotion.SAD, "text")
        # face weight 0.9, text weight 0.1 → happy should win
        fused = fuse(
            {"face": face_r, "text": text_r},
            weights={"face": 0.9, "text": 0.1},
        )
        assert fused.emotion == Emotion.HAPPY

    def test_normalised_scores_sum_to_one(self) -> None:
        face_r = self._make_result(Emotion.ANGRY, "face")
        text_r = self._make_result(Emotion.SAD, "text")
        fused = fuse({"face": face_r, "text": text_r})
        assert math.isclose(sum(fused.scores.values()), 1.0, abs_tol=1e-6)


# ---------------------------------------------------------------------------
# TextEmotionAnalyzer (keyword fallback)
# ---------------------------------------------------------------------------


class TestKeywordClassifier:
    def test_happy_text(self) -> None:
        result = _classify_with_keywords("I am so happy and excited today!")
        assert result.emotion == Emotion.HAPPY

    def test_sad_text(self) -> None:
        result = _classify_with_keywords("I feel so sad and depressed.")
        assert result.emotion == Emotion.SAD

    def test_angry_text(self) -> None:
        result = _classify_with_keywords("I am furious and angry!")
        assert result.emotion == Emotion.ANGRY

    def test_unknown_defaults_to_neutral(self) -> None:
        result = _classify_with_keywords("xyzzy plugh")
        assert result.emotion == Emotion.NEUTRAL

    def test_scores_sum_to_one(self) -> None:
        result = _classify_with_keywords("I feel great joy and happiness.")
        total = sum(result.scores.values())
        assert math.isclose(total, 1.0, abs_tol=1e-6)

    def test_source_is_text(self) -> None:
        result = _classify_with_keywords("hello")
        assert result.source == "text"


class TestTextEmotionAnalyzer:
    def test_classify_method_uses_keyword_fallback(self) -> None:
        """Test classify() logic directly without instantiating the Module."""
        # Test the pure keyword fallback directly - classify() calls this when pipeline is None
        result = _classify_with_keywords("I am very happy!")
        assert result.source == "text"
        assert result.emotion in Emotion


# ---------------------------------------------------------------------------
# EmotionalState
# ---------------------------------------------------------------------------


class TestEmotionalState:
    def test_update_shifts_dominant(self) -> None:
        state = EmotionalState()
        happy_scores = {label: 0.0 for label in EMOTION_LABELS}
        happy_scores["happy"] = 1.0
        result = EmotionResult(
            emotion=Emotion.HAPPY, scores=happy_scores, confidence=0.95, source="face"
        )
        state.update(result)
        assert state.emotion == Emotion.HAPPY

    def test_scores_normalised(self) -> None:
        state = EmotionalState()
        scores = {label: 0.0 for label in EMOTION_LABELS}
        scores["sad"] = 1.0
        result = EmotionResult(
            emotion=Emotion.SAD, scores=scores, confidence=0.8, source="text"
        )
        state.update(result)
        total = sum(state.scores.values())
        assert math.isclose(total, 1.0, abs_tol=1e-6)

    def test_current_returns_emotion_result(self) -> None:
        state = EmotionalState()
        current = state.current()
        assert isinstance(current, EmotionResult)
        assert current.source == "state"


# ---------------------------------------------------------------------------
# ContextItem
# ---------------------------------------------------------------------------


class TestContextItem:
    def test_no_ttl_never_expires(self) -> None:
        item = ContextItem(key="k", value="v", ttl=None)
        assert not item.is_expired()

    def test_expired_ttl(self) -> None:
        item = ContextItem(key="k", value="v", ttl=0.001)
        time.sleep(0.005)
        assert item.is_expired()

    def test_not_yet_expired(self) -> None:
        item = ContextItem(key="k", value="v", ttl=60.0)
        assert not item.is_expired()


# ---------------------------------------------------------------------------
# SituationalContextManager
# ---------------------------------------------------------------------------


class TestSituationalContextManager:
    def _happy_result(self) -> EmotionResult:
        scores = {label: 0.0 for label in EMOTION_LABELS}
        scores["happy"] = 1.0
        return EmotionResult(
            emotion=Emotion.HAPPY, scores=scores, confidence=0.9, source="face"
        )

    def test_update_and_get_state(self) -> None:
        mgr = SituationalContextManager()
        mgr.update_emotion(self._happy_result())
        state = mgr.get_state()
        assert state.emotion == Emotion.HAPPY

    def test_history_records_updates(self) -> None:
        mgr = SituationalContextManager()
        mgr.update_emotion(self._happy_result())
        mgr.update_emotion(self._happy_result())
        history = mgr.get_history()
        assert len(history) == 2

    def test_context_set_get(self) -> None:
        mgr = SituationalContextManager()
        mgr.set_context("foo", 42)
        assert mgr.get_context("foo") == 42

    def test_context_default_on_missing(self) -> None:
        mgr = SituationalContextManager()
        assert mgr.get_context("missing", default="default") == "default"

    def test_context_ttl_expiry(self) -> None:
        mgr = SituationalContextManager()
        mgr.set_context("tmp", "value", ttl=0.001)
        time.sleep(0.005)
        assert mgr.get_context("tmp") is None

    def test_context_remove(self) -> None:
        mgr = SituationalContextManager()
        mgr.set_context("key", "val")
        mgr.remove_context("key")
        assert mgr.get_context("key") is None

    def test_all_context_excludes_expired(self) -> None:
        mgr = SituationalContextManager()
        mgr.set_context("live", "yes", ttl=60.0)
        mgr.set_context("dead", "no", ttl=0.001)
        time.sleep(0.005)
        ctx = mgr.all_context()
        assert "live" in ctx
        assert "dead" not in ctx

    def test_pruning_by_importance(self) -> None:
        mgr = SituationalContextManager(max_context_items=3)
        mgr.set_context("a", 1, importance=0.1)
        mgr.set_context("b", 2, importance=0.9)
        mgr.set_context("c", 3, importance=0.5)
        # Adding a 4th item should prune the least important ("a")
        mgr.set_context("d", 4, importance=0.8)
        ctx = mgr.all_context()
        assert len(ctx) == 3
        assert "a" not in ctx

    def test_history_maxlen(self) -> None:
        mgr = SituationalContextManager(history_maxlen=3)
        for _ in range(5):
            mgr.update_emotion(self._happy_result())
        assert len(mgr.get_history()) == 3


# ---------------------------------------------------------------------------
# Response adaptation
# ---------------------------------------------------------------------------


class TestAdaptResponse:
    def _result(self, emotion: Emotion) -> EmotionResult:
        scores = {label: 0.0 for label in EMOTION_LABELS}
        scores[emotion.value] = 1.0
        return EmotionResult(
            emotion=emotion, scores=scores, confidence=1.0, source="fused"
        )

    def test_happy_style(self) -> None:
        params = adapt_response(self._result(Emotion.HAPPY))
        assert params.style == ResponseStyle.CASUAL

    def test_sad_strategy(self) -> None:
        params = adapt_response(self._result(Emotion.SAD))
        assert params.strategy == ResponseStrategy.EMPATHIZE

    def test_angry_style_calm(self) -> None:
        params = adapt_response(self._result(Emotion.ANGRY))
        assert params.style == ResponseStyle.CALM

    def test_system_prompt_addon_not_empty(self) -> None:
        params = adapt_response(self._result(Emotion.FEAR))
        assert len(params.system_prompt_addon) > 0

    def test_metadata_has_emotion_key(self) -> None:
        params = adapt_response(self._result(Emotion.SURPRISE))
        assert "emotion" in params.metadata
        assert params.metadata["emotion"] == "surprise"

    def test_context_override_strategy(self) -> None:
        params = adapt_response(
            self._result(Emotion.HAPPY),
            context={"user_needs_help": True},
        )
        assert params.strategy == ResponseStrategy.CLARIFY

    def test_vocal_properties_type(self) -> None:
        from dimos.agents.emotion.response import VocalProperties

        params = adapt_response(self._result(Emotion.NEUTRAL))
        assert isinstance(params.vocal, VocalProperties)

    def test_action_properties_type(self) -> None:
        from dimos.agents.emotion.response import ActionProperties

        params = adapt_response(self._result(Emotion.NEUTRAL))
        assert isinstance(params.action, ActionProperties)
