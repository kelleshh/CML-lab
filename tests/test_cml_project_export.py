"""Execute exported projects against isolated snapshots and real estimators."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_squared_error

from cml_lab.bootstrap import build_services
from cml_lab.contexts.experiments.application import ExperimentResolver
from cml_lab.infrastructure.datasets import LegacyDataGateway
from cml_lab.infrastructure.ml.catalogue import AlgorithmCatalogue
from cml_lab.infrastructure.ml.engine import ExperimentEngine
from cml_lab.infrastructure.ml.splitting import holdout
from cml_lab.infrastructure.project_export import LocalProjectExporter
from cml_lab.infrastructure.storage import SqliteRecipeRepository
from cml_lab.presentation.http.project_exports import project_exports


@pytest.fixture
def laboratory(tmp_path):
    data = LegacyDataGateway(tmp_path / "datasets")
    catalogue = AlgorithmCatalogue()
    resolver = ExperimentResolver(data, SqliteRecipeRepository(tmp_path), catalogue)
    return data, catalogue, resolver, LocalProjectExporter(data, catalogue)


def execute_module(payload):
    namespace = {"__name__": "exported_cml_project"}
    exec(compile(payload.data.decode("utf-8"), payload.filename, "exec"), namespace)
    return namespace


def selection(laboratory, task="regression", algorithm="ridge", **options):
    data, catalogue, resolver, exporter = laboratory
    dataset = data.load({"name": "linear" if task == "regression" else task,
                         "kind": "synthetic", "params": {"n_samples": 120, "seed": 7}})
    return dataset, resolver.resolve({"task": task, "dataset_id": dataset["id"],
                                      "algorithm_id": algorithm, "roles": dataset.get("roles", {}),
                                      "seed": 7, "validation": {"strategy": "none"}, **options})


@pytest.mark.parametrize("task,algorithm", [
    ("regression", "ridge"), ("classification", "logistic_regression"),
    ("clustering", "kmeans"), ("ranking", "xgboost_ranker"),
    ("forecasting", "ridge"), ("panel", "ridge"),
    ("anomaly", "isolation_forest"), ("reduction", "pca"),
])
def test_python_export_executes_all_task_contracts(laboratory, tmp_path, task, algorithm):
    data, catalogue, resolver, exporter = laboratory
    params = {"n_estimators": 5} if task in {"ranking", "anomaly"} else {}
    dataset, spec = selection(laboratory, task, algorithm, params=params)
    original = deepcopy(spec)
    exported = exporter.export(spec)
    namespace = execute_module(exported)
    pd.testing.assert_frame_equal(namespace["load_dataset"](), data.frame(dataset["id"]))
    outputs = namespace["run_project"](tmp_path / "results")
    assert spec == original
    result = outputs["result"]
    assert result["effective_spec"] == {key: value for key, value in spec.items() if key != "recipe_snapshots"}
    assert result["task"] == task
    artifact = joblib.load(tmp_path / "results" / "artifact.joblib")
    X, y, context = ExperimentEngine(data, catalogue).prepare_data(spec)
    if task in {"clustering", "anomaly", "reduction"}:
        assert set(result["evaluations"]) == {"train"}
        assert result["evaluations"]["train"]["n_rows"] == len(X)
        if task == "reduction":
            assert artifact.reduce(X.iloc[:4]).shape == (4, 2)
        else:
            assert len(artifact.predict(X.iloc[:4])) == 4
    else:
        assert result["evaluations"]["test"] == {"hidden": True}
        assert result["evaluations"]["validation"]["rows"]
        assert len(artifact.predict(X.iloc[:4])) == 4
        if task == "classification":
            assert set(artifact.predict(X)).issubset(set(data.frame(dataset["id"])[spec["target"]]))
        if task in {"forecasting", "panel"}:
            train = result["evaluations"]["train"]["rows"]
            valid = result["evaluations"]["validation"]["rows"]
            assert max(row["time"] for row in train) < min(row["origin"] for row in valid)


def test_exported_tuning_and_scaler_fit_only_training_rows(laboratory, tmp_path):
    data, catalogue, resolver, exporter = laboratory
    dataset, spec = selection(laboratory, search={"method": "grid", "trials": 3,
        "metric": "rmse", "param_space": {"alpha": [.01, 1., 50.]}},
        validation={"strategy": "kfold", "folds": 3})
    frame = data.frame(dataset["id"])
    parts = holdout(spec, len(frame), frame[spec["target"]].to_numpy())
    frame.loc[np.r_[parts["validation"], parts["test"]], spec["features"]] = 1e6
    changed = data.load({"name": "Heldout extremes", "kind": "custom", "rows": frame.to_dict("records"),
                         "target": spec["target"]})
    spec = resolver.resolve({**spec, "dataset_id": changed["id"]})
    namespace = execute_module(exporter.export(spec))
    outputs = namespace["run_project"](tmp_path / "search")
    result = outputs["result"]
    search = result["diagnostics"]["search"]
    assert search["best_score"] == min(item["score"] for item in search["trials"])
    assert len(search["trials"]) == 3
    artifact = joblib.load(tmp_path / "search" / "artifact.joblib")
    scaler = artifact.preprocessing.named_steps["columns"].named_transformers_["numeric0"].named_steps["scale"]
    np.testing.assert_allclose(scaler.mean_, frame.iloc[parts["train"]][spec["features"]].mean())
    expected = np.sqrt(mean_squared_error(frame.iloc[parts["validation"]][spec["target"]],
                                         artifact.predict(frame.iloc[parts["validation"]][spec["features"]])))
    assert result["evaluations"]["validation"]["metrics"]["rmse"] == pytest.approx(expected)


def test_snapshot_and_python_quoting_are_data_not_executed_code(laboratory, tmp_path):
    data, catalogue, resolver, exporter = laboratory
    marker = tmp_path / "never-create"
    hostile = f"__import__('pathlib').Path({str(marker)!r}).touch()"
    dataset = data.load({"name": "source-like data", "kind": "custom", "target": "target",
                         "rows": [{"value": i, "text": hostile if i % 2 else "", "flag": bool(i % 2),
                                   "target": float(i)} for i in range(30)]})
    spec = resolver.resolve({"dataset_id": dataset["id"], "features": ["value"],
                             "artifact_path_is_not_an_api_field": "unused"})
    spec["artifact_path"] = str(marker)
    spec["recipe_snapshots"] = {"secret_local": {"artifact_root": str(marker)}}
    module = execute_module(exporter.export(spec, name=hostile))
    assert not marker.exists()
    assert "artifact_path" not in module["SPEC"]
    assert "recipe_snapshots" not in module["SPEC"]
    pd.testing.assert_frame_equal(module["load_dataset"](), data.frame(dataset["id"]))
    data.delete(dataset["id"])
    assert len(module["load_dataset"]()) == 30


def test_notebook_cells_execute_in_order_and_results_are_not_overwritten(laboratory, tmp_path, monkeypatch):
    data, catalogue, resolver, exporter = laboratory
    dataset, spec = selection(laboratory)
    payload = exporter.export(spec, "ipynb")
    notebook = json.loads(payload.data)
    assert notebook["nbformat"] == 4 and notebook["nbformat_minor"] == 5
    assert len({cell["id"] for cell in notebook["cells"]}) == len(notebook["cells"])
    namespace = {"__name__": "exported_cml_notebook"}
    monkeypatch.chdir(tmp_path)
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None
            exec(compile(cell["source"], payload.filename, "exec"), namespace)
    output = Path(namespace["outputs"]["output_dir"])
    previous = (output / "artifact.joblib").read_bytes()
    with pytest.raises(FileExistsError, match="another output directory"):
        namespace["run_project"](output)
    assert (output / "artifact.joblib").read_bytes() == previous
    revealed = namespace["run_project"](tmp_path / "explicit-test", reveal_test=True)
    assert revealed["result"]["evaluations"]["test"]["n_rows"] > 0
    assert namespace["outputs"]["result"]["evaluations"]["test"] == {"hidden": True}


def test_generated_python_cli_executes_from_release_source(laboratory, tmp_path):
    dataset, spec = selection(laboratory)
    exporter = laboratory[-1]
    source = tmp_path / "project with spaces.py"
    source.write_bytes(exporter.export(spec).data)
    output = tmp_path / "output with spaces"
    completed = subprocess.run([sys.executable, str(source), "--output", str(output)],
                               capture_output=True, text=True, timeout=35,
                               env={**__import__("os").environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])})
    assert completed.returncode == 0, completed.stderr
    assert joblib.load(output / "artifact.joblib").task == "regression"
    assert json.loads((output / "result.json").read_text())["evaluations"]["test"] == {"hidden": True}


def test_http_exports_saved_revision_and_draft_using_real_resolver(tmp_path):
    services = build_services(tmp_path)
    app = FastAPI()

    @app.exception_handler(ValueError)
    async def invalid(request: Request, error: ValueError):
        return JSONResponse({"detail": str(error)}, status_code=422)

    app.include_router(project_exports(services, LocalProjectExporter(services.data.gateway)))
    try:
        dataset = services.data.load({"name": "linear", "params": {"n_samples": 30}})
        config = {"dataset_id": dataset["id"], "algorithm_id": "ridge", "params": {"alpha": .5}}
        project = services.recipes.create("project", {"name": "Original", "config": config})
        services.recipes.update("project", project["id"], {"name": "Revised", "config": {**config, "params": {"alpha": 3.}}})
        with TestClient(app) as client:
            original = client.get(f"/api/cml/projects/{project['id']}/export?revision=1")
            assert original.status_code == 200, original.text
            exported = type("Download", (), {"data": original.content, "filename": "project.py"})
            namespace = execute_module(exported)
            assert namespace["SPEC"]["params"]["alpha"] == .5
            assert namespace["PROJECT_METADATA"]["origin"]["revision"] == 1
            notebook = client.post("/api/cml/project-exports?format=ipynb", json={"spec": config})
            assert notebook.status_code == 200 and json.loads(notebook.content)["nbformat"] == 4
            invalid = client.post("/api/cml/project-exports?format=exe", json={"spec": config})
            assert invalid.status_code == 422
    finally:
        services.close()


def test_nested_declarative_pipeline_survives_python_and_notebook_export(laboratory, tmp_path, monkeypatch):
    from sklearn.compose import ColumnTransformer, make_column_selector
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from cml_lab.infrastructure.ml.pipelines import pipeline_catalogue

    data, catalogue, resolver, exporter = laboratory
    rows = [{"x": None if i % 9 == 0 else float(i), "category": "A" if i % 2 else "B",
             "target": 3. * i + (5 if i % 2 else 0)} for i in range(60)]
    dataset = data.load({"name": "Nested", "kind": "custom", "rows": rows, "target": "target"})
    source = pipeline_catalogue()["templates"][0]["source"]
    spec = resolver.resolve({"dataset_id": dataset["id"], "features": ["x", "category"],
        "algorithm_id": "ridge", "params": {"alpha": .7},
        "preprocessing": {"declarative_pipeline": {"format": "cml.pipeline", "version": 1, "source": source}}})
    module = execute_module(exporter.export(spec))
    assert module["SPEC"]["preprocessing"]["declarative_pipeline"]["source"] == source
    output = module["run_project"](tmp_path / "nested-python")
    artifact = joblib.load(tmp_path / "nested-python" / "artifact.joblib")
    frame = data.frame(dataset["id"])
    X, y = frame[spec["features"]], frame[spec["target"]].to_numpy()
    parts = holdout(spec, len(X), y)
    independent = Pipeline([("columns", ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                              ("scaler", StandardScaler())]), make_column_selector(dtype_include="number")),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
                                  ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), make_column_selector(dtype_exclude="number")),
    ], remainder="drop", verbose_feature_names_out=True)), ("model", Ridge(alpha=.7))])
    independent.fit(X.iloc[parts["train"]], y[parts["train"]])
    np.testing.assert_allclose(artifact.predict(X), independent.predict(X), atol=1e-10)
    notebook = json.loads(exporter.export(spec, "ipynb").data)
    monkeypatch.chdir(tmp_path)
    namespace = {"__name__": "cml_nested_notebook"}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            exec(compile(cell["source"], "nested.ipynb", "exec"), namespace)
    assert isinstance(namespace["preprocessing_pipeline"], ColumnTransformer)
    assert namespace["outputs"]["result"]["evaluations"]["validation"]["metrics"] == output["result"]["evaluations"]["validation"]["metrics"]
