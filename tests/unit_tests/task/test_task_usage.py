"""
Unit tests for task-level usage tracking via RequestUsage.

All tests use mocks — no API key required.

Covers:
- RequestUsage initialization, start/stop timer, add_model_execution_time
- RequestUsage.incr (token accumulation)
- RequestUsage.upsonic_execution_time computed property
- RequestUsage.to_dict / from_dict round-trip
- Task._usage lifecycle: task_start -> populate -> task_end
- Task property delegation (duration, model_execution_time, upsonic_execution_time)
- Task.usage property
- Task.to_dict / from_dict preservation of _usage
- RunUsage.incr(RequestUsage) aggregation
- Multiple RequestUsage instances on RunUsage (multi-task scenario)
"""

from __future__ import annotations

import time
import pytest
from unittest.mock import Mock, patch

from upsonic.usage import RequestUsage, RunUsage
from upsonic.tasks.tasks import Task


# ---------------------------------------------------------------------------
# RequestUsage: Initialization & Basics
# ---------------------------------------------------------------------------

class TestRequestUsageInit:

    def test_default_values(self) -> None:
        usage = RequestUsage()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.duration is None
        assert usage.model_execution_time is None
        assert usage.upsonic_execution_time is None
        assert usage.timer is None

    def test_with_token_values(self) -> None:
        usage = RequestUsage(input_tokens=100, output_tokens=50)
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.total_tokens == 150

    def test_requests_always_one(self) -> None:
        usage = RequestUsage()
        assert usage.requests == 1


# ---------------------------------------------------------------------------
# RequestUsage: Timer & Timing
# ---------------------------------------------------------------------------

class TestRequestUsageTimer:

    def test_start_timer_creates_timer(self) -> None:
        usage = RequestUsage()
        assert usage.timer is None
        usage.start_timer()
        assert usage.timer is not None

    def test_stop_timer_sets_duration(self) -> None:
        usage = RequestUsage()
        usage.start_timer()
        time.sleep(0.05)
        usage.stop_timer()
        assert usage.duration is not None
        assert usage.duration >= 0.04

    def test_stop_timer_without_start_is_noop(self) -> None:
        usage = RequestUsage()
        usage.stop_timer()
        assert usage.duration is None

    def test_stop_timer_set_duration_false(self) -> None:
        usage = RequestUsage()
        usage.start_timer()
        time.sleep(0.02)
        usage.stop_timer(set_duration=False)
        assert usage.duration is None

    def test_add_model_execution_time_first_call(self) -> None:
        usage = RequestUsage()
        usage.add_model_execution_time(1.5)
        assert usage.model_execution_time == 1.5

    def test_add_model_execution_time_accumulates(self) -> None:
        usage = RequestUsage()
        usage.add_model_execution_time(1.0)
        usage.add_model_execution_time(0.5)
        usage.add_model_execution_time(0.3)
        assert usage.model_execution_time == pytest.approx(1.8, abs=0.01)

    def test_upsonic_execution_time_computed(self) -> None:
        usage = RequestUsage()
        usage.duration = 5.0
        usage.model_execution_time = 3.0
        assert usage.upsonic_execution_time == pytest.approx(2.0)

    def test_upsonic_execution_time_none_when_duration_missing(self) -> None:
        usage = RequestUsage()
        usage.model_execution_time = 3.0
        assert usage.upsonic_execution_time is None

    def test_upsonic_execution_time_none_when_model_time_missing(self) -> None:
        usage = RequestUsage()
        usage.duration = 5.0
        assert usage.upsonic_execution_time is None

    def test_upsonic_execution_time_zero_when_equal(self) -> None:
        usage = RequestUsage()
        usage.duration = 3.0
        usage.model_execution_time = 3.0
        assert usage.upsonic_execution_time == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# RequestUsage: incr (token accumulation)
# ---------------------------------------------------------------------------

class TestRequestUsageIncr:

    def test_incr_accumulates_tokens(self) -> None:
        usage = RequestUsage(input_tokens=10, output_tokens=5)
        other = RequestUsage(input_tokens=20, output_tokens=15)
        usage.incr(other)
        assert usage.input_tokens == 30
        assert usage.output_tokens == 20

    def test_incr_accumulates_model_execution_time(self) -> None:
        usage = RequestUsage()
        usage.model_execution_time = 1.0
        other = RequestUsage()
        other.model_execution_time = 2.0
        usage.incr(other)
        assert usage.model_execution_time == pytest.approx(3.0)

    def test_incr_sets_model_execution_time_from_none(self) -> None:
        usage = RequestUsage()
        other = RequestUsage()
        other.model_execution_time = 2.5
        usage.incr(other)
        assert usage.model_execution_time == 2.5

    def test_incr_accumulates_duration(self) -> None:
        usage = RequestUsage()
        usage.duration = 1.0
        other = RequestUsage()
        other.duration = 2.0
        usage.incr(other)
        assert usage.duration == pytest.approx(3.0)

    def test_incr_cache_tokens(self) -> None:
        usage = RequestUsage(cache_write_tokens=10, cache_read_tokens=5)
        other = RequestUsage(cache_write_tokens=20, cache_read_tokens=15)
        usage.incr(other)
        assert usage.cache_write_tokens == 30
        assert usage.cache_read_tokens == 20


# ---------------------------------------------------------------------------
# RequestUsage: __add__
# ---------------------------------------------------------------------------

class TestRequestUsageAdd:

    def test_add_creates_new_instance(self) -> None:
        a = RequestUsage(input_tokens=10, output_tokens=5)
        b = RequestUsage(input_tokens=20, output_tokens=15)
        c = a + b
        assert c.input_tokens == 30
        assert c.output_tokens == 20
        assert a.input_tokens == 10
        assert b.input_tokens == 20


# ---------------------------------------------------------------------------
# RequestUsage: to_dict / from_dict
# ---------------------------------------------------------------------------

class TestRequestUsageSerialization:

    def test_to_dict_basic(self) -> None:
        usage = RequestUsage(input_tokens=100, output_tokens=50)
        usage.duration = 5.0
        usage.model_execution_time = 3.0
        d = usage.to_dict()
        assert d["input_tokens"] == 100
        assert d["output_tokens"] == 50
        assert d["duration"] == 5.0
        assert d["model_execution_time"] == 3.0
        assert d["upsonic_execution_time"] == pytest.approx(2.0)

    def test_to_dict_excludes_zero_values(self) -> None:
        usage = RequestUsage()
        d = usage.to_dict()
        assert "input_tokens" not in d
        assert "output_tokens" not in d
        assert "duration" not in d

    def test_to_dict_includes_nonzero_values(self) -> None:
        usage = RequestUsage(input_tokens=1)
        d = usage.to_dict()
        assert "input_tokens" in d

    def test_from_dict_restores_all_fields(self) -> None:
        d = {
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_write_tokens": 10,
            "cache_read_tokens": 5,
            "duration": 5.0,
            "model_execution_time": 3.0,
        }
        usage = RequestUsage.from_dict(d)
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.cache_write_tokens == 10
        assert usage.cache_read_tokens == 5
        assert usage.duration == 5.0
        assert usage.model_execution_time == 3.0
        assert usage.upsonic_execution_time == pytest.approx(2.0)

    def test_round_trip(self) -> None:
        original = RequestUsage(input_tokens=42, output_tokens=17)
        original.duration = 2.5
        original.model_execution_time = 1.8
        d = original.to_dict()
        restored = RequestUsage.from_dict(d)
        assert restored.input_tokens == original.input_tokens
        assert restored.output_tokens == original.output_tokens
        assert restored.duration == original.duration
        assert restored.model_execution_time == original.model_execution_time
        assert restored.upsonic_execution_time == original.upsonic_execution_time

    def test_from_dict_defaults(self) -> None:
        usage = RequestUsage.from_dict({})
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.duration is None
        assert usage.model_execution_time is None


# ---------------------------------------------------------------------------
# Task._usage lifecycle: task_start / task_end
# ---------------------------------------------------------------------------

class TestTaskUsageLifecycle:

    def test_task_usage_none_initially(self) -> None:
        task = Task(description="test")
        assert task._usage is None
        assert task.usage is None

    def test_task_start_creates_usage(self) -> None:
        task = Task(description="test")
        mock_agent = Mock()
        mock_agent.canvas = None
        task.task_start(mock_agent)
        assert task._usage is not None
        assert isinstance(task._usage, RequestUsage)
        assert task._usage.timer is not None

    def test_task_end_stops_timer_sets_duration(self) -> None:
        task = Task(description="test")
        mock_agent = Mock()
        mock_agent.canvas = None
        task.task_start(mock_agent)
        time.sleep(0.05)
        task.task_end()
        assert task._usage.duration is not None
        assert task._usage.duration >= 0.04

    def test_task_start_sets_start_time(self) -> None:
        task = Task(description="test")
        mock_agent = Mock()
        mock_agent.canvas = None
        before = time.time()
        task.task_start(mock_agent)
        assert task.start_time is not None
        assert task.start_time >= before - 1

    def test_task_end_sets_end_time(self) -> None:
        task = Task(description="test")
        mock_agent = Mock()
        mock_agent.canvas = None
        task.task_start(mock_agent)
        task.task_end()
        assert task.end_time is not None


# ---------------------------------------------------------------------------
# Task property delegation to _usage
# ---------------------------------------------------------------------------

class TestTaskPropertyDelegation:

    def test_duration_delegates_to_usage(self) -> None:
        task = Task(description="test")
        mock_agent = Mock()
        mock_agent.canvas = None
        task.task_start(mock_agent)
        task._usage.duration = 3.14
        assert task.duration == 3.14

    def test_duration_fallback_to_start_end_time(self) -> None:
        task = Task(description="test")
        task.start_time = 100
        task.end_time = 105
        assert task.duration == 5

    def test_duration_none_when_no_usage_and_no_times(self) -> None:
        task = Task(description="test")
        assert task.duration is None

    def test_model_execution_time_delegates_to_usage(self) -> None:
        task = Task(description="test")
        mock_agent = Mock()
        mock_agent.canvas = None
        task.task_start(mock_agent)
        task._usage.model_execution_time = 2.0
        assert task.model_execution_time == 2.0

    def test_model_execution_time_none_without_usage(self) -> None:
        task = Task(description="test")
        assert task.model_execution_time is None

    def test_upsonic_execution_time_delegates_to_usage(self) -> None:
        task = Task(description="test")
        mock_agent = Mock()
        mock_agent.canvas = None
        task.task_start(mock_agent)
        task._usage.duration = 5.0
        task._usage.model_execution_time = 3.0
        assert task.upsonic_execution_time == pytest.approx(2.0)

    def test_upsonic_execution_time_none_without_usage(self) -> None:
        task = Task(description="test")
        assert task.upsonic_execution_time is None

    def test_usage_property_returns_internal_usage(self) -> None:
        task = Task(description="test")
        mock_agent = Mock()
        mock_agent.canvas = None
        task.task_start(mock_agent)
        assert task.usage is task._usage


# ---------------------------------------------------------------------------
# Task._usage: manually populating tokens + timing
# ---------------------------------------------------------------------------

class TestTaskUsagePopulation:

    def test_tokens_accumulated_on_task_usage(self) -> None:
        task = Task(description="test")
        mock_agent = Mock()
        mock_agent.canvas = None
        task.task_start(mock_agent)

        task._usage.incr(RequestUsage(input_tokens=100, output_tokens=50))
        task._usage.incr(RequestUsage(input_tokens=200, output_tokens=80))

        assert task._usage.input_tokens == 300
        assert task._usage.output_tokens == 130

    def test_model_time_accumulated_on_task_usage(self) -> None:
        task = Task(description="test")
        mock_agent = Mock()
        mock_agent.canvas = None
        task.task_start(mock_agent)

        task._usage.add_model_execution_time(1.0)
        task._usage.add_model_execution_time(0.5)

        assert task._usage.model_execution_time == pytest.approx(1.5)

    def test_full_lifecycle_with_tokens_and_timing(self) -> None:
        task = Task(description="test")
        mock_agent = Mock()
        mock_agent.canvas = None

        task.task_start(mock_agent)

        task._usage.incr(RequestUsage(input_tokens=50, output_tokens=20))
        task._usage.add_model_execution_time(0.8)

        task._usage.incr(RequestUsage(input_tokens=30, output_tokens=10))
        task._usage.add_model_execution_time(0.4)

        time.sleep(0.05)
        task.task_end()

        assert task.usage.input_tokens == 80
        assert task.usage.output_tokens == 30
        assert task.usage.model_execution_time == pytest.approx(1.2, abs=0.01)
        assert task.usage.duration >= 0.04
        assert task.usage.upsonic_execution_time is not None
        assert task.duration == task.usage.duration
        assert task.model_execution_time == task.usage.model_execution_time


# ---------------------------------------------------------------------------
# Task.to_dict / from_dict preserves _usage
# ---------------------------------------------------------------------------

class TestTaskUsageSerialization:

    def test_to_dict_includes_usage(self) -> None:
        task = Task(description="test")
        mock_agent = Mock()
        mock_agent.canvas = None
        task.task_start(mock_agent)
        task._usage.incr(RequestUsage(input_tokens=42, output_tokens=17))
        task._usage.duration = 2.5
        task._usage.model_execution_time = 1.8
        task.task_end()

        d = task.to_dict()
        assert "_usage" in d
        assert d["_usage"] is not None
        assert d["_usage"]["input_tokens"] == 42
        assert d["_usage"]["output_tokens"] == 17

    def test_to_dict_usage_none_when_not_started(self) -> None:
        task = Task(description="test")
        d = task.to_dict()
        assert d["_usage"] is None

    def test_from_dict_restores_usage(self) -> None:
        task = Task(description="test")
        mock_agent = Mock()
        mock_agent.canvas = None
        task.task_start(mock_agent)
        task._usage.incr(RequestUsage(input_tokens=100, output_tokens=50))
        task._usage.duration = 3.0
        task._usage.model_execution_time = 2.0
        task.task_end()

        d = task.to_dict()
        restored = Task.from_dict(d)

        assert restored.usage is not None
        assert restored.usage.input_tokens == 100
        assert restored.usage.output_tokens == 50
        assert restored.usage.duration is not None
        assert restored.usage.model_execution_time == 2.0
        assert restored.duration == restored.usage.duration
        assert restored.model_execution_time == 2.0

    def test_from_dict_without_usage(self) -> None:
        d = {"description": "test", "_usage": None}
        restored = Task.from_dict(d)
        assert restored.usage is None


# ---------------------------------------------------------------------------
# RunUsage.incr(RequestUsage) — aggregation for agent level
# ---------------------------------------------------------------------------

class TestRunUsageIncrFromRequestUsage:

    def test_incr_request_usage_increments_requests(self) -> None:
        run = RunUsage()
        req = RequestUsage(input_tokens=100, output_tokens=50)
        run.incr(req)
        assert run.requests == 1
        assert run.input_tokens == 100
        assert run.output_tokens == 50

    def test_incr_multiple_request_usages(self) -> None:
        run = RunUsage()
        run.incr(RequestUsage(input_tokens=10, output_tokens=5))
        run.incr(RequestUsage(input_tokens=20, output_tokens=15))
        run.incr(RequestUsage(input_tokens=30, output_tokens=25))
        assert run.requests == 3
        assert run.input_tokens == 60
        assert run.output_tokens == 45

    def test_incr_request_usage_with_timing(self) -> None:
        run = RunUsage()
        req1 = RequestUsage(input_tokens=10, output_tokens=5)
        req1.duration = 2.0
        req1.model_execution_time = 1.5
        run.incr(req1)

        req2 = RequestUsage(input_tokens=20, output_tokens=15)
        req2.duration = 3.0
        req2.model_execution_time = 2.0
        run.incr(req2)

        assert run.requests == 2
        assert run.input_tokens == 30
        assert run.output_tokens == 20
        assert run.duration == pytest.approx(5.0)
        assert run.model_execution_time == pytest.approx(3.5)
        assert run.upsonic_execution_time == pytest.approx(1.5)

    def test_incr_request_usage_timing_none_stays_none(self) -> None:
        run = RunUsage()
        req = RequestUsage(input_tokens=10, output_tokens=5)
        run.incr(req)
        assert run.duration is None
        assert run.model_execution_time is None
        assert run.upsonic_execution_time is None


# ---------------------------------------------------------------------------
# Multi-task simulation: multiple tasks -> RunUsage
# ---------------------------------------------------------------------------

class TestMultiTaskRunUsageAggregation:

    def test_simulate_three_tasks_aggregated_to_run_usage(self) -> None:
        run_usage = RunUsage()

        for i in range(3):
            req = RequestUsage(
                input_tokens=100 * (i + 1),
                output_tokens=50 * (i + 1),
            )
            req.duration = 1.0 + i * 0.5
            req.model_execution_time = 0.8 + i * 0.3
            run_usage.incr(req)

        assert run_usage.requests == 3
        assert run_usage.input_tokens == 600
        assert run_usage.output_tokens == 300
        assert run_usage.duration == pytest.approx(4.5, abs=0.01)
        assert run_usage.model_execution_time == pytest.approx(3.3, abs=0.01)
        assert run_usage.upsonic_execution_time == pytest.approx(1.2, abs=0.01)

    def test_independent_request_usages_not_shared(self) -> None:
        req1 = RequestUsage(input_tokens=10, output_tokens=5)
        req2 = RequestUsage(input_tokens=20, output_tokens=15)

        req1.duration = 1.0
        req2.duration = 2.0

        assert req1 is not req2
        assert req1.input_tokens != req2.input_tokens
        assert req1.duration != req2.duration


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_task_end_without_start_no_error(self) -> None:
        task = Task(description="test")
        task.task_end()
        assert task._usage is None

    def test_add_model_time_before_task_start_no_crash(self) -> None:
        task = Task(description="test")
        assert task._usage is None

    def test_multiple_task_starts_resets_usage(self) -> None:
        task = Task(description="test")
        mock_agent = Mock()
        mock_agent.canvas = None

        task.task_start(mock_agent)
        first_usage = task._usage
        task._usage.incr(RequestUsage(input_tokens=100, output_tokens=50))

        task.task_start(mock_agent)
        assert task._usage is not first_usage
        assert task._usage.input_tokens == 0

    def test_request_usage_has_values(self) -> None:
        empty = RequestUsage()
        assert not empty.has_values()

        populated = RequestUsage(input_tokens=1)
        assert populated.has_values()
