from __future__ import annotations

import pytest

from spectral_packet_engine.interfaces import inspect_service_status
from spectral_packet_engine.service_status import (
    _LOGGER,
    configure_service_logging,
    inspect_service_status as inspect_service_status_runtime,
    mark_service_task_failed,
    track_service_task,
)


def test_service_status_tracks_running_and_completed_tasks() -> None:
    before = inspect_service_status()

    with track_service_task(
        "unit-test-task",
        interface="python",
        workflow_id="unit-test-workflow",
        surface_action="python:unit-test-task",
        metadata={"kind": "unit"},
    ):
        active = inspect_service_status()
        assert active.counters.active_task_count >= 1
        assert any(task.name == "unit-test-task" and task.status == "running" for task in active.active_tasks)
        assert any(task.workflow_id == "unit-test-workflow" for task in active.active_tasks)

    after = inspect_service_status()
    assert after.counters.total_started >= before.counters.total_started + 1
    assert after.counters.total_completed >= before.counters.total_completed + 1
    assert any(task.name == "unit-test-task" and task.status == "completed" for task in after.recent_tasks)
    assert any(task.surface_action == "python:unit-test-task" for task in after.recent_tasks)


def test_service_status_tracks_failed_tasks() -> None:
    before = inspect_service_status()

    with pytest.raises(RuntimeError, match="boom"):
        with track_service_task("failing-test-task", interface="python"):
            raise RuntimeError("boom")

    after = inspect_service_status()
    assert after.counters.total_started >= before.counters.total_started + 1
    assert after.counters.total_failed >= before.counters.total_failed + 1
    assert any(task.name == "failing-test-task" and task.status == "failed" for task in after.recent_tasks)


def test_service_status_can_mark_failed_without_raising() -> None:
    before = inspect_service_status_runtime()

    with track_service_task("handled-failure-task", interface="python") as task_id:
        mark_service_task_failed(task_id, RuntimeError("handled boom"))

    after = inspect_service_status_runtime()
    assert after.counters.total_failed >= before.counters.total_failed + 1
    assert after.counters.total_completed == before.counters.total_completed
    assert any(task.name == "handled-failure-task" and task.status == "failed" for task in after.recent_tasks)


def test_service_logger_does_not_propagate_to_root_by_default() -> None:
    assert _LOGGER.propagate is False


def test_configured_service_logging_uses_stderr_not_stdout(capsys) -> None:
    configure_service_logging("INFO", force=True)

    with track_service_task("stderr-test-task", interface="python"):
        pass

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "service_task_completed" in captured.err
