"""Own fitted exports retain raw-input behavior and the hidden-test boundary."""

from __future__ import annotations

import csv
from io import BytesIO, StringIO
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
from zipfile import ZipFile

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import LabelEncoder

from cml_lab.contexts.execution.artifact_application import ArtifactApplication
from cml_lab.contexts.execution.domain import Run
from cml_lab.infrastructure.ml.artifacts import FittedArtifact
from cml_lab.infrastructure.ml.catalogue import AlgorithmCatalogue
from cml_lab.infrastructure.ml.preparation import build_preprocessor
from cml_lab.infrastructure.model_artifacts import LocalArtifactGateway
from cml_lab.infrastructure.storage import SqliteRunRepository
from cml_lab.shared.domain import ConflictError, NotFoundError, ValidationError


class Data:
    def __init__(self, frame):
        self.saved = frame.copy()

    def frame(self, identifier):
        return self.saved.copy()


def fitted(tmp_path, task="regression", config=None, frame=None, algorithm_id=None):
    rng = np.random.default_rng(9)
    X = frame if frame is not None else pd.DataFrame({"a": rng.normal(size=40), "b": rng.normal(size=40)})
    numeric = pd.to_numeric(X["a"], errors="coerce").fillna(0).to_numpy()
    y = np.where(numeric > 0, "да", "нет") if task == "classification" else 7 + 2 * numeric
    encoder = LabelEncoder().fit(y) if task == "classification" else None
    encoded = encoder.transform(y) if encoder is not None else y
    prep = build_preprocessor(X, config or {"steps": []}, task=task)
    values = prep.fit_transform(X, encoded)
    estimator = AlgorithmCatalogue().build(task, algorithm_id) if algorithm_id else (
        LogisticRegression(C=3.0) if task == "classification" else Ridge(alpha=0.1))
    estimator.fit(values, encoded)
    artifact = FittedArtifact(task, prep, estimator, list(X.columns), encoder)
    repository = SqliteRunRepository(tmp_path)
    run = repository.create(Run.create({"task": task, "dataset_id": "d" * 32,
                                        "algorithm_id": algorithm_id or "ridge", "features": list(X.columns)}))
    run = repository.update(run.transition("running"))
    result = {"evaluations": {
        "train": {"metrics": {"mae": 0.1}, "rows": [{"index": 0, "actual": 1, "predicted": 1.1}]},
        "validation": {"metrics": {"mae": 0.2}, "rows": [{"index": 1, "actual": 2, "predicted": 2.2}]},
        "test": {"metrics": {"mae": 888.888}, "rows": [{"index": 2, "actual": "SEALED_TEST_VALUE", "predicted": 3}]},
    }}
    run = repository.update(run.transition("completed", result=result))
    directory = tmp_path / "artifacts"
    directory.mkdir(exist_ok=True)
    joblib.dump(artifact, directory / f"{run.id}.joblib")
    gateway = LocalArtifactGateway(tmp_path, data_gateway=Data(X))
    return ArtifactApplication(repository, gateway), repository, gateway, run, artifact, X


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_python_formats_keep_raw_predictions_and_label_decoding(tmp_path, task):
    service, _, _, run, artifact, X = fitted(tmp_path, task)
    rows = X.iloc[[0, 5, 10]].to_dict(orient="records")
    expected = artifact.predict(rows)
    for format in ("joblib", "pickle"):
        payload = service.export(run.id, format)
        reloaded = joblib.load(BytesIO(payload.data)) if format == "joblib" else pickle.loads(payload.data)
        if task == "classification":
            np.testing.assert_array_equal(reloaded.predict(rows), expected)
            assert expected.dtype.kind in {"O", "U"}
        else:
            np.testing.assert_allclose(reloaded.predict(rows), expected, rtol=0, atol=0)
    predicted = service.predict(run.id, rows)
    assert predicted["predictions"] == expected.tolist()
    if task == "classification":
        assert predicted["classes"] == artifact.classes_.tolist()
        np.testing.assert_allclose(predicted["probabilities"], artifact.predict_proba(rows))


def test_passport_csv_and_bundle_use_one_visibility_policy(tmp_path):
    service, repository, _, run, _, _ = fitted(tmp_path)
    for format in ("passport", "json", "csv"):
        assert b"SEALED_TEST_VALUE" not in service.export(run.id, format).data
    passport = json.loads(service.export(run.id, "passport").data)
    assert passport["run"]["result"]["evaluations"]["test"] == {"hidden": True}
    assert passport["test_visibility"] == "hidden"
    assert passport["raw_input"]["features"] == ["a", "b"]
    exported = service.export(run.id, "bundle")
    with ZipFile(BytesIO(exported.data)) as archive:
        assert b"SEALED_TEST_VALUE" not in archive.read("passport.json")
        names = archive.namelist()
        assert "cml_lab/infrastructure/ml/artifacts.py" in names
        assert "linear_lab/_vendor/smogn-LICENSE" in names
        assert "linear_lab/_vendor/PROVENANCE.md" in names
        assert not any(name.endswith(".csv") or "/.env" in name or "__pycache__" in name for name in names)
    current = repository.get(run.id)
    repository.update(current.transition("completed", test_revealed=True))
    assert b"SEALED_TEST_VALUE" in service.export(run.id, "passport").data
    assert b"SEALED_TEST_VALUE" in service.export(run.id, "csv").data
    rows = list(csv.DictReader(StringIO(service.export(run.id, "csv").data.decode("utf-8-sig"))))
    assert {row["split"] for row in rows} == {"train", "validation", "test"}


def test_bundle_runs_in_fresh_python_process_with_its_own_source(tmp_path):
    service, _, _, run, artifact, X = fitted(tmp_path)
    directory = tmp_path / "bundle"
    directory.mkdir()
    with ZipFile(BytesIO(service.export(run.id, "bundle").data)) as archive:
        archive.extractall(directory)
    rows = X.iloc[[2, 6, 9]]
    rows.to_csv(directory / "rows.csv", index=False)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(directory)
    completed = subprocess.run([sys.executable, "inference.py", "--input", "rows.csv", "--output", "pred.csv"],
                               cwd=directory, env=environment, capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
    predictions = pd.read_csv(directory / "pred.csv")["prediction"].to_numpy()
    np.testing.assert_allclose(predictions, artifact.predict(rows), rtol=1e-13, atol=1e-13)


def test_skops_requires_explicit_review_of_custom_types(tmp_path):
    import skops.io as sio
    service, _, _, run, artifact, X = fitted(tmp_path)
    payload = service.export(run.id, "skops")
    unknown = sio.get_untrusted_types(data=payload.data)
    assert "cml_lab.infrastructure.ml.artifacts.FittedArtifact" in unknown
    with pytest.raises(Exception):
        sio.loads(payload.data)
    loaded = sio.loads(payload.data, trusted=unknown)
    np.testing.assert_allclose(loaded.predict(X.iloc[:3]), artifact.predict(X.iloc[:3]))
    passport = json.loads(service.export(run.id, "passport").data)
    assert passport["skops_untrusted_types"] == unknown


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_onnx_runs_complete_numeric_recipe_and_decoded_classes(tmp_path, task):
    import onnxruntime as ort
    X = pd.DataFrame({"a": [-3., -2., -1., 0., 1., 2., 3., 4.] * 5,
                      "b": [None, 4., 3., 2., 1., -1., -2., -3.] * 5})
    config = {"steps": [
        {"adapter_id": "numeric.impute", "columns": [], "params": {"strategy": "median"}},
        {"adapter_id": "numeric.power", "columns": [], "params": {"method": "log1p"}},
        {"adapter_id": "numeric.scale", "columns": [], "params": {"method": "standard"}},
    ]}
    service, _, _, run, artifact, _ = fitted(tmp_path, task, config, X)
    caps = {item["format"]: item for item in service.capabilities(run.id)["formats"]}
    assert caps["onnx"]["available"], caps["onnx"]["reason"]
    payload = service.export(run.id, "onnx")
    session = ort.InferenceSession(payload.data, providers=["CPUExecutionProvider"])
    rows = X.iloc[[0, 1, 20, 39]].copy()
    rows.loc[rows.index[1], "b"] = np.inf
    feed = {name: rows[name].to_numpy(dtype=np.float32).reshape(-1, 1) for name in rows.columns}
    result = session.run(None, feed)
    if task == "classification":
        np.testing.assert_array_equal(result[0].reshape(-1), artifact.predict(rows))
        np.testing.assert_allclose(result[1], artifact.predict_proba(rows), rtol=1e-4, atol=1e-5)
    else:
        np.testing.assert_allclose(result[0].reshape(-1), artifact.predict(rows), rtol=1e-4, atol=1e-4)


def test_categorical_onnx_handles_unknown_categories_and_documents_null_limit(tmp_path):
    import onnxruntime as ort
    X = pd.DataFrame({"a": np.arange(40, dtype=float), "color": ["red", "blue"] * 20})
    service, _, _, run, artifact, _ = fitted(tmp_path, frame=X)
    caps = {item["format"]: item for item in service.capabilities(run.id)["formats"]}
    assert caps["onnx"]["available"], caps["onnx"]["reason"]
    session = ort.InferenceSession(service.export(run.id, "onnx").data, providers=["CPUExecutionProvider"])
    rows = pd.DataFrame({"a": [2., 6.], "color": ["blue", "new"]})
    outputs = session.run(None, {"a": rows.a.to_numpy(dtype=np.float32).reshape(-1, 1),
                                "color": rows.color.to_numpy(dtype=object).reshape(-1, 1)})
    np.testing.assert_allclose(outputs[0].reshape(-1), artifact.predict(rows), rtol=1e-4, atol=1e-4)
    schema = json.loads(service.export(run.id, "passport").data)["onnx_input_schema"]
    assert next(field for field in schema["inputs"] if field["name"] == "color")["nullable"] is False


def test_unconvertible_pipeline_has_precise_native_fallback(tmp_path):
    X = pd.DataFrame({"a": np.arange(40, dtype=float), "date": pd.date_range("2026-01-01", periods=40)})
    service, _, _, run, _, _ = fitted(tmp_path, frame=X)
    caps = {item["format"]: item for item in service.capabilities(run.id)["formats"]}
    assert caps["joblib"]["available"] is True
    assert caps["onnx"]["available"] is False
    assert "DatetimeExtractor" in caps["onnx"]["reason"]
    with pytest.raises(ValidationError, match="DatetimeExtractor"):
        service.export(run.id, "onnx")


def test_artifacts_reject_external_paths_symlinks_and_unfinished_runs(tmp_path):
    service, repository, gateway, run, _, _ = fitted(tmp_path)
    with pytest.raises(ValidationError):
        gateway.predict("../../evil.pkl", [{"a": 1, "b": 2}])
    path = tmp_path / "artifacts" / f"{run.id}.joblib"
    outside = tmp_path / "outside.joblib"
    path.rename(outside)
    path.symlink_to(outside)
    with pytest.raises(ValidationError, match="собственном"):
        service.export(run.id, "joblib")
    queued = repository.create(Run.create({"task": "regression"}))
    with pytest.raises(ConflictError, match="завершения"):
        service.predict(queued.id, [{"a": 1, "b": 2}])
    with pytest.raises(NotFoundError):
        service.export("f" * 32, "joblib")
    with pytest.raises(ValidationError):
        service.export(run.id, "arbitrary-format")


def test_saved_temporal_artifact_recurses_from_history_without_future_target(tmp_path):
    from cml_lab.infrastructure.ml.temporal import build_temporal_frame
    dates = pd.date_range("2026-01-01", periods=20)
    history = pd.DataFrame({"time": dates, "target": 10.0 + 2 * np.arange(20)})
    prepared = build_temporal_frame(history, "target", "time", lags=[1], rolling=[], features=[])
    preprocessing = build_preprocessor(prepared.X, {"steps": []}, task="forecasting")
    values = preprocessing.fit_transform(prepared.X, prepared.y)
    model = AlgorithmCatalogue().build("forecasting", "ols").fit(values, prepared.y)
    artifact = FittedArtifact("forecasting", preprocessing, model, list(prepared.X.columns),
                              temporal=prepared.history_config)
    repository = SqliteRunRepository(tmp_path)
    run = repository.create(Run.create({"task": "forecasting", "dataset_id": "d" * 32}))
    run = repository.update(run.transition("running"))
    run = repository.update(run.transition("completed", result={"evaluations": {}}))
    (tmp_path / "artifacts").mkdir()
    joblib.dump(artifact, tmp_path / "artifacts" / f"{run.id}.joblib")
    service = ArtifactApplication(repository, LocalArtifactGateway(tmp_path))
    future = [{"time": "2026-01-21"}, {"time": "2026-01-22"}]
    history_rows = history.assign(time=history.time.astype(str)).to_dict(orient="records")
    result = service.forecast(run.id, history_rows, future)
    np.testing.assert_allclose([row["predicted"] for row in result["rows"]], [50, 52], atol=1e-12)
    assert all("target" not in row for row in future)
    with pytest.raises(ValidationError, match="историю"):
        service.predict(run.id, [{"target__lag_1": 48}])
    caps = {item["format"]: item for item in service.capabilities(run.id)["formats"]}
    assert caps["onnx"]["available"] is False
    assert "истории" in caps["onnx"]["reason"]


def test_pca_artifact_projects_new_rows_and_transductive_methods_explain_refusal(tmp_path):
    X = pd.DataFrame(np.random.default_rng(8).normal(size=(25, 3)), columns=["a", "b", "c"])
    preprocessing = build_preprocessor(X, {"steps": []}, task="reduction")
    values = preprocessing.fit_transform(X)
    model = AlgorithmCatalogue().build("reduction", "pca").fit(values)
    artifact = FittedArtifact("reduction", preprocessing, model, list(X.columns))
    repository = SqliteRunRepository(tmp_path)
    run = repository.create(Run.create({"task": "reduction"}))
    run = repository.update(run.transition("running"))
    run = repository.update(run.transition("completed", result={"evaluations": {}}))
    (tmp_path / "artifacts").mkdir()
    joblib.dump(artifact, tmp_path / "artifacts" / f"{run.id}.joblib")
    service = ArtifactApplication(repository, LocalArtifactGateway(tmp_path))
    rows = X.iloc[:2].to_dict(orient="records")
    result = service.predict(run.id, rows)
    np.testing.assert_allclose(result["coordinates"], model.transform(preprocessing.transform(pd.DataFrame(rows))))
    cluster = AlgorithmCatalogue().build("clustering", "dbscan").fit(values)
    artifact = FittedArtifact("clustering", preprocessing, cluster, list(X.columns))
    joblib.dump(artifact, tmp_path / "artifacts" / f"{run.id}.joblib")
    with pytest.raises(ValueError, match="обучающую таблицу"):
        service.predict(run.id, rows)
