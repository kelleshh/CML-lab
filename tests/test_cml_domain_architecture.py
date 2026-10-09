"""Independent domain invariants and application port contracts for CML-lab."""

from __future__ import annotations

import ast
from copy import deepcopy
from pathlib import Path
import subprocess
import sys

import pytest

from cml_lab.contexts.data.application import DataApplication
from cml_lab.contexts.data.domain import Dataset
from cml_lab.contexts.experiments.application import ExperimentResolver
from cml_lab.contexts.experiments.domain import DatasetSelection, Experiment
from cml_lab.contexts.execution.application import ExecutionService
from cml_lab.contexts.execution.domain import Run
from cml_lab.contexts.recipes.application import RecipeService
from cml_lab.contexts.recipes.domain import Recipe
from cml_lab.shared.domain import ConflictError, NotFoundError, TaskKind, ValidationError

DATA_ID = "1" * 32
ROOT = Path(__file__).resolve().parents[1]


class MemoryRecipes:
    def __init__(self):
        self.heads = {}
        self.revisions = {}

    def list(self, kind, query=""):
        return [item for item in self.heads.values() if item.kind == kind and query in item.name]

    def get(self, identifier, kind=None, revision=None):
        try:
            item = self.heads[identifier] if revision is None else self.revisions[identifier, revision]
        except KeyError as exc:
            raise NotFoundError(identifier) from exc
        if kind is not None and item.kind != kind:
            raise NotFoundError(identifier)
        return item

    def create(self, recipe):
        self.heads[recipe.id] = recipe
        self.revisions[recipe.id, recipe.revision] = recipe
        return recipe

    def update(self, recipe, expected_revision):
        if self.get(recipe.id).revision != expected_revision:
            raise ConflictError("revision")
        return self.create(recipe)

    def delete(self, identifier, kind=None):
        self.get(identifier, kind)
        del self.heads[identifier]


class MemoryRuns:
    def __init__(self):
        self.items = {}

    def create(self, run):
        self.items[run.id] = run
        return run

    def get(self, identifier):
        return self.items[identifier]

    def list(self):
        return list(self.items.values())

    def update(self, run, expected_status=None):
        current = self.get(run.id)
        if run.revision != current.revision + 1 or expected_status is not None and current.status != expected_status:
            raise ConflictError("stale run")
        self.items[run.id] = run
        return run

    def delete(self, identifier):
        del self.items[identifier]


class Scheduler:
    def __init__(self):
        self.started = []
        self.cancelled = []
        self.closed = False

    def start(self, run):
        self.started.append(run.id)

    def cancel(self, identifier):
        self.cancelled.append(identifier)

    def close(self):
        self.closed = True


class Data:
    def __init__(self):
        self.metadata = {"id": DATA_ID, "name": "Контрольные данные", "rows": 40,
                         "columns": [{"name": name, "numeric": name != "class", "missing": 0}
                                     for name in ("x", "y", "class", "time", "entity", "query", "weight")],
                         "default_target": "y", "tasks": ["regression"]}
        self.deleted = []

    def describe(self, identifier):
        assert identifier == DATA_ID
        return deepcopy(self.metadata)

    def delete(self, identifier):
        self.deleted.append(identifier)
        return {"id": identifier, "deleted": True}


class Algorithms:
    def spec(self, algorithm_id):
        return {"id": algorithm_id, "available": True, "tasks": [item.value for item in TaskKind],
                "params": [{"key": "alpha"}]}


def resolver():
    return ExperimentResolver(Data(), MemoryRecipes(), Algorithms())


def test_recipe_revisions_and_saved_config_are_independent():
    initial = {"algorithm_id": "ridge", "params": {"alpha": 2, "nested": {"a": [1]}}}
    recipe = Recipe.create("model", {"name": "Мой Ridge", "config": initial})
    initial["params"]["nested"]["a"].append(999)
    returned = recipe.config
    returned["params"]["alpha"] = 400
    revised = recipe.revise({"config": {"algorithm_id": "ridge", "params": {"alpha": 9}}, "expected_revision": 1})
    assert recipe.config["params"] == {"alpha": 2, "nested": {"a": [1]}}
    assert revised.revision == 2 and revised.id == recipe.id
    assert revised.config["params"]["alpha"] == 9
    assert Recipe.from_dict(recipe.to_dict()).to_dict() == recipe.to_dict()


def test_recipe_service_rejects_stale_update_without_changing_head():
    repository = MemoryRecipes()
    service = RecipeService(repository)
    saved = service.create("model", {"name": "Модель", "config": {"algorithm_id": "ridge"}})
    service.update("model", saved["id"], {"name": "Обновленная", "expected_revision": 1})
    with pytest.raises(ConflictError):
        service.update("model", saved["id"], {"name": "Потерянное обновление", "expected_revision": 1})
    assert service.get("model", saved["id"])["name"] == "Обновленная"
    assert service.get("model", saved["id"], revision=1)["name"] == "Модель"


def test_recipe_snapshot_survives_recipe_update_and_deletion():
    data, recipes = Data(), MemoryRecipes()
    recipe = recipes.create(Recipe.create("model", {"name": "Сохраненный", "config": {"algorithm_id": "ridge", "params": {"alpha": 3}}}))
    resolve = ExperimentResolver(data, recipes, Algorithms())
    spec = resolve.resolve({"dataset_id": DATA_ID, "model_recipe_id": recipe.id, "features": ["x"], "target": "y"})
    recipes.update(recipe.revise({"config": {"algorithm_id": "ridge", "params": {"alpha": 100}}}), 1)
    recipes.delete(recipe.id)
    assert spec["params"] == {"alpha": 3}
    assert spec["recipe_snapshots"]["model"]["revision"] == 1
    assert spec["recipe_snapshots"]["model"]["config"]["params"] == {"alpha": 3}


@pytest.mark.parametrize("task,roles", [("ranking", {"query_column": "query"}), ("forecasting", {"time_column": "time"}),
                                          ("panel", {"time_column": "time", "entity_column": "entity"})])
def test_task_roles_are_required_and_excluded_from_features(task, roles):
    with pytest.raises(ValidationError):
        resolver().resolve({"task": task, "dataset_id": DATA_ID, "target": "y", "features": ["x"]})
    requested = ["x", "y", *roles.values(), "weight"]
    spec = resolver().resolve({"task": task, "dataset_id": DATA_ID, "target": "y", "features": requested,
                               "roles": dict(roles, weight_column="weight")})
    assert spec["features"] == ["x"]
    if task in {"forecasting", "panel"}:
        assert spec["split"]["shuffle"] is False


def test_classification_accepts_categorical_target_and_regression_rejects_it():
    spec = resolver().resolve({"task": "classification", "dataset_id": DATA_ID, "target": "class", "features": ["x", "class"]})
    assert spec["target"] == "class" and spec["features"] == ["x"]
    with pytest.raises(ValidationError, match="числовая"):
        resolver().resolve({"task": "regression", "dataset_id": DATA_ID, "target": "class", "features": ["x"]})


def test_clustering_reference_labels_never_enter_fitted_features():
    spec = resolver().resolve({"task": "clustering", "dataset_id": DATA_ID, "target": "class", "features": ["x", "class"]})
    assert spec["target"] is None
    assert spec["roles"]["reference_target"] == "class"
    assert spec["features"] == ["x"]


@pytest.mark.parametrize("changes", [{"artifact_path": "/tmp/attacker.joblib"}, {"seed": True}, {"n_jobs": 32},
                                     {"params": {"unknown": 5}}, {"split": {"train": .8, "validation": .2, "test": .2}}])
def test_resolver_rejects_unsafe_or_inconsistent_configuration(changes):
    with pytest.raises(ValidationError):
        resolver().resolve({"dataset_id": DATA_ID, "features": ["x"], "target": "y", **changes})


def test_temporal_validation_and_sampling_are_constrained():
    request = {"task": "forecasting", "dataset_id": DATA_ID, "features": ["x"], "roles": {"time_column": "time"}}
    with pytest.raises(ValidationError, match="хронологическую"):
        resolver().resolve(dict(request, validation={"strategy": "kfold"}))
    with pytest.raises(ValidationError, match="Пересэмплирование"):
        resolver().resolve(dict(request, preprocessing={"steps": [], "resampling": {"method": "random_over"}}))
    spec = resolver().resolve(dict(request, validation={"strategy": "timeseries", "folds": 3}, temporal={"lags": [2], "rolling_windows": [], "horizon": 1}))
    assert spec["temporal"] == {"lags": [2], "rolling_windows": [], "horizon": 1}


@pytest.mark.parametrize("task,roles", [("forecasting", {"time_column": "time"}),
                                       ("panel", {"time_column": "time", "entity_column": "entity"})])
def test_autoregression_can_create_lags_without_external_features(task, roles):
    spec = resolver().resolve({"task": task, "dataset_id": DATA_ID, "target": "y", "features": [], "roles": roles})
    assert spec["features"] == [] and spec["temporal"]["lags"] == [1, 2, 3]
    with pytest.raises(ValidationError):
        resolver().resolve({"task": "regression", "dataset_id": DATA_ID, "target": "y", "features": []})


def test_saved_experiment_revision_cannot_edit_original_spec_or_results():
    experiment = Experiment.create("Контрольный расчет", {"params": {"alpha": 2}}, {"evaluations": {}}, "2" * 32)
    edited = experiment.revise_metadata({"name": "Понятное название", "expected_revision": 1})
    edited.spec["params"]["alpha"] = 900
    assert experiment.name == "Контрольный расчет" and edited.name == "Понятное название"
    assert edited.spec == experiment.spec == {"params": {"alpha": 2}}
    with pytest.raises(ValidationError):
        edited.revise_metadata({"spec": {"params": {"alpha": 900}}})
    with pytest.raises(ConflictError):
        edited.revise_metadata({"expected_revision": 1, "name": "Устарело"})
    assert Experiment.from_dict(edited.to_dict()).to_dict() == edited.to_dict()


def test_cancelled_run_cannot_be_overwritten_by_completion_or_stale_progress():
    repository, scheduler = MemoryRuns(), Scheduler()
    service = ExecutionService(repository, scheduler)
    started = service.start({"task": "regression", "dataset_id": DATA_ID})
    running = repository.update(started.transition("running", progress=.4))
    cancelled = service.cancel(started.id)
    assert cancelled["status"] == "cancelled" and scheduler.cancelled == [started.id]
    with pytest.raises(ConflictError):
        repository.get(started.id).transition("completed")
    with pytest.raises(ConflictError):
        repository.update(running.transition("running", progress=.7))
    assert service.cancel(started.id)["status"] == "cancelled"
    assert scheduler.cancelled == [started.id]


def test_one_visibility_policy_hides_entire_test_evaluation_then_reveals_it():
    result = {"task": "classification", "evaluations": {"train": {"metrics": {"accuracy": 1}},
              "validation": {"metrics": {"accuracy": .8}}, "test": {"metrics": {"accuracy": .1},
              "rows": [{"actual": "secret", "predicted": "other"}], "plots": {"confusion": [[2, 9]]}}}}
    repository, scheduler = MemoryRuns(), Scheduler()
    service = ExecutionService(repository, scheduler)
    run = service.start({"task": "classification"})
    running = repository.update(run.transition("running"))
    repository.update(running.transition("completed", result=result))
    hidden = service.get(run.id)
    assert hidden["result"]["evaluations"]["test"] == {"hidden": True}
    assert "secret" not in str(hidden)
    assert repository.get(run.id).result["evaluations"]["test"]["rows"][0]["actual"] == "secret"
    public = service.reveal_test(run.id)
    assert public["result"]["evaluations"]["test"] == result["evaluations"]["test"]
    assert public["result"]["test_hidden"] is False
    public["result"]["evaluations"]["test"]["metrics"]["accuracy"] = 999
    assert repository.get(run.id).result["evaluations"]["test"]["metrics"]["accuracy"] == .1


def test_unfinished_run_cannot_reveal_test_or_be_deleted():
    repository, scheduler = MemoryRuns(), Scheduler()
    service = ExecutionService(repository, scheduler)
    run = service.start({"task": "clustering"})
    with pytest.raises(ConflictError):
        service.reveal_test(run.id)
    with pytest.raises(ConflictError):
        service.delete(run.id)
    service.cancel(run.id)
    assert service.delete(run.id)["deleted"] is True


def test_direct_execution_deletion_preserves_referenced_model_and_artifact():
    class References:
        def run_in_use(self, identifier):
            return True

    repository, scheduler, cleaned = MemoryRuns(), Scheduler(), []
    service = ExecutionService(repository, scheduler, References(), cleaned.append)
    run = service.start({"task": "regression"})
    running = repository.update(run.transition("running"))
    completed = repository.update(running.transition("completed", result={"value": 42}))
    with pytest.raises(ConflictError, match="сохраненным экспериментом"):
        service.delete(run.id)
    assert repository.get(run.id) == completed and cleaned == []


def test_finished_results_and_revealed_test_are_immutable():
    completed = Run.create({"task": "classification"}).transition("running").transition(
        "completed", result={"evaluations": {"test": {"metrics": {"accuracy": 0.7}}}}
    )
    with pytest.raises(ConflictError, match="неизменен"):
        completed.transition("completed", result={"invented": True})
    revealed = completed.transition("completed", test_revealed=True)
    with pytest.raises(ConflictError, match="скрыть"):
        revealed.transition("completed", test_revealed=False)
    assert revealed.transition("completed", name="Новое описание").result == completed.result


def test_dataset_deletion_respects_saved_experiment_references():
    class References:
        def dataset_in_use(self, identifier):
            return True
    data = Data()
    with pytest.raises(ConflictError):
        DataApplication(data, References()).delete(DATA_ID)
    assert data.deleted == []
    assert DataApplication(data).delete(DATA_ID)["deleted"] is True


def test_clean_layers_have_no_technical_imports_or_outer_layer_dependencies():
    forbidden = {"numpy", "pandas", "sklearn", "fastapi", "joblib", "sqlite3", "scipy", "linear_lab"}
    paths = list((ROOT / "cml_lab" / "shared").glob("*.py"))
    paths += [path for path in (ROOT / "cml_lab" / "contexts").rglob("*.py")
              if path.name in {"domain.py", "application.py", "ports.py"}
              or path.name.endswith("_application.py") or path.name.startswith("ports_")]
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module]
        imports += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
        assert all(name.split(".")[0] not in forbidden for name in imports), (path, imports)
        assert all(not name.startswith(("cml_lab.infrastructure", "cml_lab.presentation")) for name in imports), (path, imports)


def test_domain_and_application_import_without_any_ml_or_http_library():
    script = '''
import importlib, importlib.abc, pathlib, sys
sys.path.insert(0, sys.argv[1])
class DenyLibraries(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'numpy','pandas','sklearn','fastapi','joblib','sqlite3','scipy','linear_lab'}:
            raise RuntimeError('Clean layer imported technical library: ' + fullname)
sys.meta_path.insert(0, DenyLibraries())
root = pathlib.Path(sys.argv[1]) / 'cml_lab'
files = [*root.glob('shared/*.py')]
files += [file for file in root.glob('contexts/**/*.py') if file.name in {'domain.py','application.py','ports.py'} or file.name.endswith('_application.py') or file.name.startswith('ports_')]
for file in files:
    importlib.import_module('.'.join(file.relative_to(root.parent).with_suffix('').parts))
'''
    result = subprocess.run([sys.executable, "-I", "-c", script, str(ROOT)], text=True, capture_output=True, timeout=20)
    assert result.returncode == 0, result.stderr
