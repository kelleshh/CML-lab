"""Concurrent services must not recover each other's live calculations."""

import multiprocessing

import pytest

from cml_lab.contexts.execution.domain import Run
from cml_lab.infrastructure.execution import ProcessScheduler
from cml_lab.infrastructure.runtime_lease import RuntimeLease
from cml_lab.infrastructure.storage import SqliteRunRepository
from cml_lab.shared.domain import ConflictError


def _hold_lease(path, ready):
    lease = RuntimeLease(path)
    ready.set()
    # The parent terminates this process to verify OS-level lock release.
    multiprocessing.Event().wait(30)
    lease.close()


def test_second_scheduler_rejected_without_interrupting_live_runs(tmp_path):
    runs = SqliteRunRepository(tmp_path)
    first = ProcessScheduler(tmp_path, runs)
    try:
        queued = runs.create(Run.create({"task": "regression"}))
        running = runs.create(Run.create({"task": "classification"}))
        running = runs.update(running.transition("running"))
        with pytest.raises(ConflictError, match="Папка данных уже используется"):
            ProcessScheduler(tmp_path, SqliteRunRepository(tmp_path))
        assert runs.get(queued.id).status == "queued"
        assert runs.get(running.id).status == "running"
    finally:
        first.close()


def test_distinct_roots_allow_independent_schedulers(tmp_path):
    roots = [tmp_path / "first", tmp_path / "second"]
    schedulers = []
    try:
        for root in roots:
            schedulers.append(ProcessScheduler(root, SqliteRunRepository(root)))
        assert len(schedulers) == 2
    finally:
        for scheduler in schedulers:
            scheduler.close()


def test_close_releases_owner_and_repeated_close_is_safe(tmp_path):
    runs = SqliteRunRepository(tmp_path)
    first = ProcessScheduler(tmp_path, runs)
    abandoned = runs.create(Run.create({"task": "regression"}))
    first.close()
    second = ProcessScheduler(tmp_path, runs)
    try:
        assert runs.get(abandoned.id).status == "interrupted"
        first.close()
        with pytest.raises(ConflictError):
            ProcessScheduler(tmp_path, runs)
    finally:
        second.close()
    third = ProcessScheduler(tmp_path, runs)
    third.close()


def test_failed_recovery_releases_runtime_owner(tmp_path):
    class BrokenRepository:
        def list(self):
            raise RuntimeError("Ошибка чтения запусков")

    with pytest.raises(RuntimeError, match="Ошибка чтения"):
        ProcessScheduler(tmp_path, BrokenRepository())
    scheduler = ProcessScheduler(tmp_path, SqliteRunRepository(tmp_path))
    scheduler.close()


def test_termination_releases_runtime_lease_between_processes(tmp_path):
    path = str(tmp_path / "runtime.sqlite3")
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    child = context.Process(target=_hold_lease, args=(path, ready))
    child.start()
    try:
        assert ready.wait(10)
        with pytest.raises(ConflictError):
            RuntimeLease(path)
    finally:
        child.terminate()
        child.join(10)
        assert not child.is_alive()
    reopened = RuntimeLease(path)
    reopened.close()
