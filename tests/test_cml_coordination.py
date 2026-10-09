"""A dataset cannot vanish between source validation and queued run persistence."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading

import pytest

from cml_lab.contexts.data.application import DataApplication
from cml_lab.contexts.execution.application import ExecutionService
from cml_lab.infrastructure.coordination import SqliteMutationGuard
from cml_lab.infrastructure.references import DatasetReferences
from cml_lab.infrastructure.storage import SqliteExperimentRepository, SqliteRunRepository
from cml_lab.shared.domain import ConflictError, NotFoundError

DATA_ID = "a" * 32


class DatasetFile:
    def __init__(self, root: Path):
        self.path = root / f"{DATA_ID}.json"
        self.path.write_text("{}", encoding="utf-8")

    def describe(self, dataset_id):
        assert dataset_id == DATA_ID
        if not self.path.exists():
            raise NotFoundError("Данные удалены.")
        return {"id": dataset_id}

    def delete(self, dataset_id):
        self.describe(dataset_id)
        self.path.unlink()
        return {"id": dataset_id, "deleted": True}


class Scheduler:
    def __init__(self):
        self.started = []

    def start(self, run):
        self.started.append(run.id)


@pytest.mark.parametrize("first", ["start", "delete"])
def test_two_independent_guards_preserve_dataset_start_reference(tmp_path, first):
    runs = SqliteRunRepository(tmp_path)
    experiments = SqliteExperimentRepository(tmp_path)
    gateway, scheduler = DatasetFile(tmp_path), Scheduler()
    inside_first, release_first, second_entered = threading.Event(), threading.Event(), threading.Event()
    first_references = DatasetReferences(runs, experiments)

    def validate(spec):
        if first == "start":
            inside_first.set()
            assert release_first.wait(5)
        else:
            second_entered.set()
        gateway.describe(spec["dataset_id"])

    class References:
        def dataset_in_use(self, dataset_id):
            if first == "delete":
                inside_first.set()
                assert release_first.wait(5)
            else:
                second_entered.set()
            return first_references.dataset_in_use(dataset_id)

    # Separate instances and connections share only the lock database path.
    data = DataApplication(gateway, References(), SqliteMutationGuard(tmp_path / "coordination.sqlite3"))
    execution = ExecutionService(runs, scheduler,
        mutation_guard=SqliteMutationGuard(tmp_path / "coordination.sqlite3"), validate_input=validate)
    spec = {"task": "regression", "dataset_id": DATA_ID}
    actions = {"start": lambda: execution.start(spec), "delete": lambda: data.delete(DATA_ID)}
    with ThreadPoolExecutor(max_workers=2) as pool:
        first_future = pool.submit(actions[first])
        assert inside_first.wait(5)
        second_future = pool.submit(actions["delete" if first == "start" else "start"])
        try:
            assert not second_entered.wait(0.1)
        finally:
            release_first.set()
        first_result = first_future.result(timeout=5)
        if first == "start":
            with pytest.raises(ConflictError, match="используется"):
                second_future.result(timeout=5)
            assert gateway.path.exists()
            assert runs.get(first_result.id).status == "queued"
            assert scheduler.started == [first_result.id]
        else:
            with pytest.raises(NotFoundError):
                second_future.result(timeout=5)
            assert first_result["deleted"] is True
            assert not gateway.path.exists()
            assert runs.list() == [] and scheduler.started == []


def test_failed_mutation_releases_guard_for_another_instance(tmp_path):
    path = tmp_path / "coordination.sqlite3"
    with pytest.raises(RuntimeError):
        with SqliteMutationGuard(path).hold():
            raise RuntimeError("Отказ до сохранения")
    with SqliteMutationGuard(path).hold():
        assert path.exists()


def test_scheduler_runs_after_guard_release(tmp_path):
    class CheckingScheduler:
        def start(self, run):
            with SqliteMutationGuard(tmp_path / "coordination.sqlite3").hold():
                assert runs.get(run.id).status == "queued"

    runs = SqliteRunRepository(tmp_path)
    execution = ExecutionService(runs, CheckingScheduler(),
        mutation_guard=SqliteMutationGuard(tmp_path / "coordination.sqlite3"))
    assert execution.start({"task": "regression", "dataset_id": DATA_ID}).status == "queued"
