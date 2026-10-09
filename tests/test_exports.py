"""Exported models receive raw rows, and never bypass fitted preprocessing."""

from io import BytesIO
import json
import pickle
import subprocess
from pathlib import Path
from zipfile import ZipFile

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from linear_lab.exports import ModelExportService
from linear_lab.pipeline import RegressionPipeline
from linear_lab.preprocessing import build_preprocessor


JOB_ID = "a" * 32


def make_job(tmp_path, *, config=None, categorical=False, custom_pipeline=False):
    X = pd.DataFrame({"distance": [-2., -1., 0., 1., 2., np.nan], "weight": [1., 3., 2., 5., 8., 6.]})
    y = np.asarray([-5., -2., 0., 4., 8., 6.])
    if categorical:
        X["zone"] = ["east", "west", "east", "east", "west", "west"]
    preprocessor, _, _ = build_preprocessor(X, config or {})
    cls = RegressionPipeline if custom_pipeline else Pipeline
    pipeline = cls([("preprocessing", preprocessor), ("model", Ridge(alpha=.1))]).fit(X, y)
    directory = tmp_path / "jobs" / JOB_ID
    directory.mkdir(parents=True)
    joblib.dump(pipeline, directory / "model.joblib")
    request = {"model": "ridge", "preprocessing": config or {}, "artifact_path": "/private/internal/model.joblib"}
    (directory / "request.json").write_text(json.dumps(request))
    result = {"model": "ridge", "model_name": "Ridge", "target_name": "price", "metrics": {"validation": {"r2": .95}}, "feature_names": preprocessor.get_feature_names_out().tolist()}
    return ModelExportService(tmp_path), pipeline, X, result, request


def onnx_prediction(data, X):
    ort = pytest.importorskip("onnxruntime")
    session = ort.InferenceSession(data, providers=["CPUExecutionProvider"])
    feeds = {item.name: X[[item.name]].to_numpy(dtype=object if item.type == "tensor(string)" else np.float64) for item in session.get_inputs()}
    return np.asarray(session.run(None, feeds)[0]).reshape(-1)


@pytest.mark.parametrize("format", ["joblib", "pickle"])
def test_python_export_preserves_raw_input_preprocessing(tmp_path, format):
    service, original, X, result, request = make_job(tmp_path, categorical=True, custom_pipeline=True)
    data, filename, media = service.export(JOB_ID, format, result, request)
    restored = joblib.load(BytesIO(data)) if format == "joblib" else pickle.loads(data)
    assert isinstance(restored, RegressionPipeline)
    np.testing.assert_allclose(restored.predict(X), original.predict(X), rtol=0, atol=0)
    assert filename.endswith(".joblib" if format == "joblib" else ".pkl")
    assert media == "application/octet-stream"


@pytest.mark.parametrize("config", [
    {}, {"degree": 2}, {"numeric_transform": "log1p"},
    {"numeric_transform": "sqrt", "clip_quantiles": [.1, .9]},
    {"selection": "f_regression", "max_features": 1},
    {"variance_threshold": 0, "scaler": "minmax"},
])
def test_onnx_matches_raw_ridge_pipeline_including_nonfinite(tmp_path, config):
    pytest.importorskip("skl2onnx")
    service, original, X, result, request = make_job(tmp_path, config=config)
    capabilities = {item["format"]: item for item in service.capabilities(JOB_ID)}
    assert capabilities["onnx"]["available"], capabilities["onnx"]["reason"]
    data, filename, _ = service.export(JOB_ID, "onnx", result, request)
    probes = pd.concat([X, X.iloc[[0]].assign(distance=np.inf), X.iloc[[0]].assign(distance=-np.inf)], ignore_index=True)
    np.testing.assert_allclose(onnx_prediction(data, probes), original.predict(probes), rtol=2e-5, atol=2e-5)
    assert filename.endswith(".onnx")
    import onnx
    metadata = {item.key: item.value for item in onnx.load_model_from_string(data).metadata_props}
    assert json.loads(metadata["linear_lab_schema"])["raw_input_pipeline"] is True


def test_onnx_categorical_input_is_explicit_nonnullable_string(tmp_path):
    pytest.importorskip("skl2onnx")
    service, original, X, result, request = make_job(tmp_path, categorical=True)
    capabilities = {item["format"]: item for item in service.capabilities(JOB_ID)}
    assert capabilities["onnx"]["available"], capabilities["onnx"]["reason"]
    assert "null/NaN" in capabilities["onnx"]["reason"]
    data, _, _ = service.export(JOB_ID, "onnx", result, request)
    probes = pd.concat([X, X.iloc[[0]].assign(zone="never seen")], ignore_index=True)
    np.testing.assert_allclose(onnx_prediction(data, probes), original.predict(probes), rtol=2e-5, atol=2e-5)
    passport, _, _ = service.export(JOB_ID, "passport", result, request)
    schema = json.loads(passport)["onnx"]["input_schema"]
    assert next(item for item in schema if item["name"] == "zone")["nullable"] is False


def test_unsupported_onnx_preprocessing_returns_reason_without_partial_model(tmp_path):
    service, _, _, result, request = make_job(tmp_path, config={"imputation": "knn"})
    capabilities = {item["format"]: item for item in service.capabilities(JOB_ID)}
    assert capabilities["joblib"]["available"]
    assert not capabilities["onnx"]["available"]
    assert "KNN" in capabilities["onnx"]["reason"]
    with pytest.raises(ValueError, match="KNN"):
        service.export(JOB_ID, "onnx", result, request)


def test_skops_lists_required_types_and_roundtrip_requires_explicit_trust(tmp_path):
    sio = pytest.importorskip("skops.io")
    service, original, X, result, request = make_job(tmp_path, custom_pipeline=True)
    data, filename, _ = service.export(JOB_ID, "skops", result, request)
    required = sio.get_untrusted_types(data=data)
    assert "linear_lab.preprocessing.FeatureEngineeringTransformer" in required
    assert "linear_lab.pipeline.RegressionPipeline" in required
    with pytest.raises(Exception):
        sio.loads(data)
    # The fixture produces known project code; trust is explicitly established.
    restored = sio.loads(data, trusted=required)
    np.testing.assert_allclose(restored.predict(X), original.predict(X))
    passport, _, _ = service.export(JOB_ID, "passport", result, request)
    assert json.loads(passport)["skops_required_trusted_types"] == sorted(required)
    assert filename.endswith(".skops")


def test_bundle_contains_runnable_full_pipeline_and_no_raw_rows(tmp_path):
    service, original, X, result, request = make_job(tmp_path, custom_pipeline=True)
    data, filename, media = service.export(JOB_ID, "bundle", result, request)
    bundle = tmp_path / "bundle"
    with ZipFile(BytesIO(data)) as archive:
        assert {"model.joblib", "passport.json", "requirements.txt", "predict.py", "linear_lab/preprocessing.py", "linear_lab/models.py", "README.md"}.issubset(archive.namelist())
        assert not any(name.endswith(".csv") or "datasets/" in name for name in archive.namelist())
        passport = json.loads(archive.read("passport.json"))
        assert "artifact_path" not in passport["configuration"]
        assert passport["inputs"][0]["role"] == "feature"
        assert "scikit-learn==" in archive.read("requirements.txt").decode()
        archive.extractall(bundle)
    X.to_csv(bundle / "rows.csv", index=False)
    import sys
    process = subprocess.run([sys.executable, "predict.py", "rows.csv", "predictions.csv"], cwd=bundle, text=True, capture_output=True, timeout=30)
    assert process.returncode == 0, process.stderr
    np.testing.assert_allclose(pd.read_csv(bundle / "predictions.csv")["prediction"], original.predict(X))
    assert filename.endswith(".zip") and media == "application/zip"


@pytest.mark.parametrize("identifier", ["../outside", "b" * 31, "b" * 33, "A" * 32, None])
def test_invalid_identifiers_cannot_load_external_model(tmp_path, identifier):
    service = ModelExportService(tmp_path)
    with pytest.raises(FileNotFoundError):
        service.capabilities(identifier)


def test_artifact_symlinks_are_rejected_before_joblib_load(tmp_path):
    directory = tmp_path / "jobs" / JOB_ID
    directory.mkdir(parents=True)
    target = tmp_path / "external.joblib"
    target.write_bytes(b"must never be unpickled")
    (directory / "model.joblib").symlink_to(target)
    with pytest.raises(FileNotFoundError):
        ModelExportService(tmp_path).export(JOB_ID, "joblib", {}, {})


def test_unknown_format_is_not_accepted(tmp_path):
    service, _, _, result, request = make_job(tmp_path)
    with pytest.raises(ValueError, match="формат"):
        service.export(JOB_ID, "python", result, request)
