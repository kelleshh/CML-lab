"""Durable CRUD, revision conflicts and worker-process storage boundaries."""

from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import sqlite3
import threading

import pytest

from cml_lab.contexts.execution.domain import Run
from cml_lab.contexts.recipes.domain import Recipe
from cml_lab.infrastructure.storage import SqliteExperimentRepository, SqliteRecipeRepository, SqliteRunRepository
from cml_lab.shared.domain import ConflictError, NotFoundError, ValidationError


def _model(name="Модель", alpha=1):
    return Recipe.create("model", {"name": name, "config": {"algorithm_id": "ridge", "params": {"alpha": alpha}}})


def _process_revision(path, dto, barrier, queue):
    repository = SqliteRecipeRepository(path)
    changed = Recipe.from_dict(dto).revise({"name": "Из процесса"})
    barrier.wait()
    try:
        repository.update(changed, expected_revision=1)
        queue.put("saved")
    except ConflictError:
        queue.put("conflict")


def test_recipe_versions_survive_reopen_deletion_and_read_by_revision(tmp_path):
    first = _model()
    repository = SqliteRecipeRepository(tmp_path)
    repository.create(first)
    second = repository.update(first.revise({"name": "Вторая", "config": {"algorithm_id": "ridge", "params": {"alpha": 9}}}), 1)
    reloaded = SqliteRecipeRepository(tmp_path)
    assert reloaded.get(first.id).config["params"]["alpha"] == 9
    assert reloaded.get(first.id, revision=1).to_dict() == first.to_dict()
    reloaded.delete(first.id, "model")
    assert reloaded.list("model") == []
    with pytest.raises(NotFoundError):
        reloaded.get(first.id)
    assert reloaded.get(first.id, revision=2).to_dict() == second.to_dict()
    assert reloaded.get(first.id, revision=3).deleted
    with pytest.raises(ConflictError):
        reloaded.create(first)
    with pytest.raises(ConflictError):
        reloaded.update(second.revise({"name": "Вернуть"}), 2)


def test_recipe_wrong_kind_and_russian_casefold_search(tmp_path):
    repository = SqliteRecipeRepository(tmp_path)
    recipe = repository.create(_model("Регуляризация"))
    assert repository.list("model", "РЕГУЛЯРИЗАЦИЯ")[0].id == recipe.id
    assert repository.list("preprocessor") == []
    with pytest.raises(NotFoundError):
        repository.get(recipe.id, "preprocessor")
    with pytest.raises(NotFoundError):
        repository.delete(recipe.id, "preprocessor")
    assert repository.get(recipe.id).id == recipe.id
    with pytest.raises(ValidationError):
        repository.list("bad")


def test_concurrent_recipe_writers_one_revision_wins(tmp_path):
    repository = SqliteRecipeRepository(tmp_path)
    recipe = repository.create(_model())
    changes = [recipe.revise({"name": f"Автор {index}"}) for index in range(8)]
    def save(change):
        try:
            return SqliteRecipeRepository(tmp_path).update(change, 1).name
        except ConflictError:
            return None
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(save, changes))
    assert len([value for value in results if value]) == 1
    assert repository.get(recipe.id).revision == 2
    assert repository.get(recipe.id).name in results


def test_recipe_lost_updates_rejected_between_spawned_processes(tmp_path):
    repository = SqliteRecipeRepository(tmp_path)
    recipe = repository.create(_model())
    context = multiprocessing.get_context("spawn")
    barrier, queue = context.Barrier(2), context.Queue()
    children = [context.Process(target=_process_revision, args=(str(tmp_path), recipe.to_dict(), barrier, queue)) for _ in range(2)]
    for child in children:
        child.start()
    for child in children:
        child.join(20)
        assert child.exitcode == 0
    assert sorted(queue.get(timeout=2) for _ in range(2)) == ["conflict", "saved"]
    assert repository.get(recipe.id).revision == 2
    queue.close()


def test_failed_write_rolls_back_revision_and_head_atomically(tmp_path):
    repository = SqliteRecipeRepository(tmp_path)
    recipe = repository.create(_model())
    # Simulate a disk-side write failure after the use case begins its transaction.
    with sqlite3.connect(repository.path) as connection:
        connection.execute("CREATE TRIGGER stop_revision BEFORE INSERT ON recipe_revisions WHEN NEW.revision=2 BEGIN SELECT RAISE(ABORT, 'write failed'); END")
    with pytest.raises(sqlite3.IntegrityError):
        repository.update(recipe.revise({"name": "Нельзя сохранить"}), 1)
    assert repository.get(recipe.id).to_dict() == recipe.to_dict()
    with pytest.raises(NotFoundError):
        repository.get(recipe.id, revision=2)


def test_run_state_and_same_status_metadata_conflicts(tmp_path):
    repository = SqliteRunRepository(tmp_path)
    run = repository.create(Run.create({"dataset_id": "a" * 32, "task": "classification"}))
    running = repository.update(run.transition("running", progress=0.2), expected_status="queued")
    with pytest.raises(ConflictError):
        repository.update(run.transition("cancelled"), expected_status="queued")
    result = {"evaluations": {"train": {"metrics": {"accuracy": 1}}, "test": {"metrics": {"accuracy": 0.25}}}}
    completed = repository.update(running.transition("completed", result=result), expected_status="running")
    assert completed.public_dict()["result"]["evaluations"]["test"] == {"hidden": True}
    assert SqliteRunRepository(tmp_path).get(run.id).result == result
    renamed = completed.transition("completed", name="Сохранить название")
    revealed = completed.transition("completed", test_revealed=True)
    repository.update(renamed, "completed")
    with pytest.raises(ConflictError):
        repository.update(revealed, "completed")
    fresh = repository.get(run.id)
    repository.update(fresh.transition("completed", test_revealed=True), "completed")
    assert repository.get(run.id).public_dict()["result"]["evaluations"]["test"]["metrics"]["accuracy"] == 0.25


def test_run_cancellation_cannot_be_overwritten_by_late_worker(tmp_path):
    repository = SqliteRunRepository(tmp_path)
    queued = repository.create(Run.create({"task": "regression"}))
    with pytest.raises(ConflictError):
        repository.delete(queued.id)
    running = repository.update(queued.transition("running"), "queued")
    cancelled = repository.update(running.transition("cancelled"), "running")
    with pytest.raises(ConflictError):
        repository.update(running.transition("completed", result={}), "running")
    assert repository.get(cancelled.id).status == "cancelled"
    repository.delete(cancelled.id)
    with pytest.raises(NotFoundError):
        repository.get(cancelled.id)


def test_run_rejects_changed_immutable_spec(tmp_path):
    repository = SqliteRunRepository(tmp_path)
    run = repository.create(Run.create({"seed": 42}))
    dto = run.transition("running").to_dict()
    dto["spec"]["seed"] = 7
    with pytest.raises(ValidationError):
        repository.update(Run.from_dict(dto), "queued")
    assert repository.get(run.id).spec == {"seed": 42}


def test_experiment_snapshot_visibility_crud_and_revision_conflicts(tmp_path):
    repository = SqliteExperimentRepository(tmp_path)
    runs = SqliteRunRepository(tmp_path)
    spec = {"dataset_id": "a" * 32, "params": {"alpha": 1}}
    result = {"evaluations": {"test": {"metrics": {"rmse": 10}}, "validation": {"metrics": {"rmse": 2}}}}
    run = runs.create(Run.create(spec))
    running = runs.update(run.transition("running"))
    completed = runs.update(running.transition("completed", result=result))
    saved = repository.save("Первый", spec, result, completed.id)
    assert saved["result"]["evaluations"]["test"] == {"hidden": True}
    spec["params"]["alpha"] = 100
    result["evaluations"]["validation"]["metrics"]["rmse"] = 100
    changed = repository.update(saved["id"], {"name": "Новое имя", "description": "Описание", "expected_revision": 1})
    assert changed["spec"]["params"]["alpha"] == 1
    assert changed["result"]["evaluations"]["validation"]["metrics"]["rmse"] == 2
    assert changed["revision"] == 2 and not changed["test_revealed"]
    with pytest.raises(ConflictError):
        repository.update(saved["id"], {"name": "Потерянная запись", "expected_revision": 1})
    with pytest.raises(ValidationError):
        repository.update(saved["id"], {"spec": {"seed": 8}, "expected_revision": 2})
    assert SqliteExperimentRepository(tmp_path).get(saved["id"])["name"] == "Новое имя"
    revealed = repository.save("Открытый тест", {}, dict(result, test_hidden=False), completed.id)
    assert revealed["result"]["evaluations"]["test"]["metrics"]["rmse"] == 10
    repository.delete(saved["id"])
    assert [item["id"] for item in repository.list()] == [revealed["id"]]
    with pytest.raises(NotFoundError):
        repository.get(saved["id"])


def test_run_repository_guards_result_even_when_transition_is_bypassed(tmp_path):
    runs = SqliteRunRepository(tmp_path)
    run = runs.create(Run.create({"task": "classification"}))
    running = runs.update(run.transition("running"))
    completed = runs.update(running.transition("completed", result={"accuracy": 0.2}))
    forged = completed.transition("completed", name="Новое имя").to_dict()
    forged["result"]["accuracy"] = 1
    with pytest.raises(ConflictError, match="неизменен"):
        runs.update(Run.from_dict(forged))
    assert runs.get(run.id).result == {"accuracy": 0.2}


def test_experiment_save_requires_existing_completed_run(tmp_path):
    experiments = SqliteExperimentRepository(tmp_path)
    runs = SqliteRunRepository(tmp_path)
    with pytest.raises(NotFoundError):
        experiments.save("Нет запуска", {}, {}, "a" * 32)
    run = runs.create(Run.create({"task": "regression"}))
    with pytest.raises(ConflictError, match="успешного завершения"):
        experiments.save("Ожидает", run.spec, {}, run.id)
    assert experiments.list() == []


def test_atomic_run_delete_blocks_active_snapshot_but_allows_deleted_snapshot(tmp_path):
    experiments = SqliteExperimentRepository(tmp_path)
    runs = SqliteRunRepository(tmp_path)
    run = runs.create(Run.create({"task": "regression"}))
    running = runs.update(run.transition("running"))
    completed = runs.update(running.transition("completed", result={"metric": 3}))
    saved = experiments.save("Сохраненный", completed.spec, completed.public_dict()["result"], run.id)
    with pytest.raises(ConflictError, match="сохраненным экспериментом"):
        runs.delete(run.id)
    assert runs.get(run.id).status == "completed"
    assert experiments.get(saved["id"])["result"]["metric"] == 3
    experiments.delete(saved["id"])
    runs.delete(run.id)
    with pytest.raises(NotFoundError):
        runs.get(run.id)


def test_save_delete_race_never_creates_active_orphan_snapshot(tmp_path):
    experiments = SqliteExperimentRepository(tmp_path)
    runs = SqliteRunRepository(tmp_path)
    run = runs.create(Run.create({"task": "regression"}))
    running = runs.update(run.transition("running"))
    completed = runs.update(running.transition("completed", result={"metric": 3}))
    barrier = threading.Barrier(2)

    def save():
        barrier.wait()
        try:
            experiments.save("Гонка", completed.spec, completed.public_dict()["result"], run.id)
            return "saved"
        except NotFoundError:
            return "save_rejected"

    def delete():
        barrier.wait()
        try:
            runs.delete(run.id)
            return "deleted"
        except ConflictError:
            return "delete_rejected"

    with ThreadPoolExecutor(max_workers=2) as pool:
        save_future, delete_future = pool.submit(save), pool.submit(delete)
        outcome = save_future.result(), delete_future.result()
    assert outcome in {("saved", "delete_rejected"), ("save_rejected", "deleted")}
    if experiments.list():
        assert runs.get(run.id).status == "completed"
    else:
        with pytest.raises(NotFoundError):
            runs.get(run.id)


@pytest.mark.parametrize("bad", ["../other", "a" * 31, "g" * 32, None, 1])
def test_storage_identifiers_are_not_filesystem_paths(tmp_path, bad):
    for repository in (SqliteRecipeRepository(tmp_path), SqliteRunRepository(tmp_path), SqliteExperimentRepository(tmp_path)):
        with pytest.raises(ValidationError):
            repository.get(bad)
