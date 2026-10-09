"""Independent exports audit: real workers, raw-input inference, honest formats."""

from __future__ import annotations

from io import BytesIO
import json
import os
import pickle
from pathlib import Path
import subprocess
import sys
import time
from zipfile import ZipFile

from fastapi.testclient import TestClient
import joblib
import numpy as np
import pandas as pd
import pytest

from linear_lab.app import create_app
from linear_lab.exports import ModelExportService


def wait_for_job(client, identifier, timeout=80):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/api/jobs/{identifier}")
        assert response.status_code == 200, response.text
        state = response.json()
        if state["status"] != "running":
            assert state["status"] == "completed", state
            return state
        time.sleep(.03)
    pytest.fail(f"Real worker did not finish: {identifier}")


@pytest.fixture(scope="module")
def trained_api(tmp_path_factory):
    root = tmp_path_factory.mktemp("export-audit-service")
    rows = []
    for index in range(160):
        x = index / 11 - 7
        weight = 1 + (index * 17 % 51) / 3
        zone = ("east", "west", "north")[index % 3]
        y = 1.5 + 4 * x - .2 * weight + (index % 3)
        rows.append({"distance": None if index % 29 == 0 else x,
                     "weight": weight, "zone": zone, "flag": bool(index % 2), "price": y})
    with TestClient(create_app(root)) as client:
        loaded = client.post("/api/datasets/load", json={"kind": "custom", "rows": rows, "target": "price"})
        assert loaded.status_code == 200, loaded.text
        dataset_id = loaded.json()["id"]
        request = {
            "dataset_id": dataset_id, "target": "price",
            "features": ["distance", "weight", "zone", "flag"],
            "model": "ridge", "params": {"alpha": 400., "fit_intercept": True}, "seed": 31,
            "split": {"train": .6, "validation": .2, "test": .2, "shuffle": True},
            "preprocessing": {"degree": 2, "numeric_transform": "log1p", "clip_quantiles": [.03, .97],
                              "scaler": "robust", "selection": "f_regression", "max_features": 5},
            "metrics": ["mse", "r2"], "regularization_path": False,
            "search": {"method": "grid", "trials": 3, "metric": "mse", "n_jobs": 1,
                       "param_space": {"alpha": [.001, .3, 30.]}},
            "cv_config": {"strategy": "kfold", "folds": 3, "shuffle": True},
        }
        started = client.post("/api/jobs", json=request)
        assert started.status_code == 200, started.text
        identifier = started.json()["id"]
        state = wait_for_job(client, identifier)
        yield client, root, dataset_id, identifier, state


def download(client, identifier, format):
    response = client.get(f"/api/jobs/{identifier}/export", params={"format": format})
    assert response.status_code == 200, response.text[:500]
    assert response.headers["content-disposition"].startswith('attachment; filename="linear-lab-')
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.content
    return response


def probes():
    return pd.DataFrame([
        {"distance": None, "weight": 4., "zone": None, "flag": True},
        {"distance": -3.25, "weight": 10000., "zone": "never observed", "flag": False},
        {"distance": 1.2, "weight": 2., "zone": "east", "flag": True},
    ])


def test_real_api_export_capabilities_holdout_and_actual_search_winner(trained_api):
    client, root, _, identifier, state = trained_api
    capabilities = client.get(f"/api/jobs/{identifier}/export-capabilities")
    assert capabilities.status_code == 200, capabilities.text
    formats = {item["format"]: item for item in capabilities.json()["formats"]}
    assert set(formats) == {"joblib", "pickle", "skops", "onnx", "bundle", "passport"}
    assert all(item["available"] for item in formats.values()), formats
    assert "null/NaN" in formats["onnx"]["reason"]

    passport = download(client, identifier, "passport").json()
    winner = state["result"]["search"]["best_params"]
    actual = joblib.load(root / "jobs" / identifier / "model.joblib")
    assert passport["configuration"]["params"] == winner
    assert passport["configuration"]["params"]["alpha"] != 400
    assert actual.named_steps["model"].get_params()["alpha"] == winner["alpha"]
    assert passport["configuration"]["params"]["fit_intercept"] is True
    assert len(passport["transformed_features"]) == 5
    assert "artifact_path" not in passport["configuration"]
    assert str(root) not in json.dumps(passport)
    assert passport["metrics"]["test"] == {}
    assert passport["metrics"]["validation"]["mse"] >= 0
    bundle = download(client, identifier, "bundle")
    with ZipFile(BytesIO(bundle.content)) as archive:
        zipped_passport = json.loads(archive.read("passport.json"))
        assert zipped_passport["metrics"]["test"] == {}
        assert zipped_passport["configuration"]["params"] == winner
        assert not any(name.endswith((".csv", ".parquet", ".npz")) for name in archive.namelist())
        assert "request.json" not in archive.namelist()
        package = Path(__file__).resolve().parents[1] / "linear_lab"
        for name in ("smogn-LICENSE", "ImbalancedLearningRegression-LICENSE", "PROVENANCE.md", "upstream-sha256.json"):
            assert archive.read(f"linear_lab/_vendor/{name}") == (package / "_vendor" / name).read_bytes()
        assert archive.read("THIRD_PARTY_NOTICES.md") == (package.parent / "THIRD_PARTY_NOTICES.md").read_bytes()


@pytest.mark.parametrize("format", ["joblib", "pickle"])
def test_native_download_uses_same_raw_input_pipeline_in_fresh_process(trained_api, tmp_path, format):
    client, _, _, identifier, _ = trained_api
    raw = probes()
    response = client.post("/api/predict", json={"job_id": identifier, "rows": json.loads(raw.to_json(orient="records"))})
    assert response.status_code == 200, response.text
    expected = np.asarray(response.json()["predictions"])
    artifact = download(client, identifier, format)
    restored = joblib.load(BytesIO(artifact.content)) if format == "joblib" else pickle.loads(artifact.content)
    np.testing.assert_allclose(restored.predict(raw), expected, rtol=0, atol=0)

    # No access to the service directory or original source installation is
    # needed: execute using only the code shipped with the prediction bundle.
    bundle = download(client, identifier, "bundle")
    with ZipFile(BytesIO(bundle.content)) as archive:
        archive.extractall(tmp_path)
    model_name = "download.joblib" if format == "joblib" else "download.pkl"
    (tmp_path / model_name).write_bytes(artifact.content)
    raw.to_json(tmp_path / "rows.json", orient="records")
    script = """import json, pickle, joblib, pandas as pd
from pathlib import Path
path=Path(__import__('sys').argv[1])
model=joblib.load(path) if path.suffix=='.joblib' else pickle.loads(path.read_bytes())
rows=pd.DataFrame(json.loads(Path('rows.json').read_text()))
print(json.dumps(model.predict(rows[rows.columns[::-1]]).tolist()))
"""
    (tmp_path / "fresh.py").write_text(script)
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment["OPENBLAS_NUM_THREADS"] = environment["OMP_NUM_THREADS"] = "1"
    process = subprocess.run([sys.executable, "fresh.py", model_name], cwd=tmp_path,
                             env=environment, capture_output=True, text=True, timeout=40)
    assert process.returncode == 0, process.stderr
    np.testing.assert_allclose(json.loads(process.stdout), expected, rtol=0, atol=0)
    assert restored.named_steps["preprocessing"].selection_steps_[-1].score_func.__name__ == "stable_f_regression"


def test_onnx_runtime_preserves_numeric_missing_unknown_categories_and_contract(trained_api):
    import onnx
    import onnxruntime as ort
    client, root, _, identifier, _ = trained_api
    pipeline = joblib.load(root / "jobs" / identifier / "model.joblib")
    rows = probes().drop(index=0).reset_index(drop=True)
    rows["flag"] = rows["flag"].astype(str)
    extras = [rows.iloc[[0]].assign(distance=value) for value in [np.nan, np.inf, -np.inf]]
    rows = pd.concat([rows, *extras], ignore_index=True)
    response = download(client, identifier, "onnx")
    assert response.headers["content-type"] == "application/octet-stream"
    graph = onnx.load_model_from_string(response.content)
    metadata = {item.key: item.value for item in graph.metadata_props}
    schema = json.loads(metadata["linear_lab_schema"])
    assert schema["raw_input_pipeline"] is True
    types = {item["name"]: item for item in schema["input_schema"]}
    assert set(types) == {"distance", "weight", "zone", "flag"}
    assert types["distance"]["type"] == "float64" and types["distance"]["nullable"] is True
    assert types["zone"]["type"] == "string" and types["zone"]["nullable"] is False
    assert "null/NaN" in schema["categorical_contract"]
    options = ort.SessionOptions()
    options.inter_op_num_threads = options.intra_op_num_threads = 1
    session = ort.InferenceSession(response.content, sess_options=options, providers=["CPUExecutionProvider"])
    feeds = {item.name: rows[[item.name]].to_numpy(dtype=object if item.type == "tensor(string)" else np.float64)
             for item in session.get_inputs()}
    actual = np.asarray(session.run(None, feeds)[0]).reshape(-1)
    np.testing.assert_allclose(actual, pipeline.predict(rows), atol=2e-5, rtol=2e-5)
    # Categorical nulls are handled by native exports, and explicitly unsupported
    # in string-tensor ONNX inputs rather than claimed to be equivalent.
    assert np.isfinite(pipeline.predict(probes())).all()
    passport = download(client, identifier, "passport").json()
    assert passport["onnx"]["parity_check"]["rows"] >= 1


def test_skops_download_does_not_automatically_trust_custom_project_types(trained_api):
    import skops.io as sio
    client, root, _, identifier, _ = trained_api
    response = download(client, identifier, "skops")
    required = sio.get_untrusted_types(data=response.content)
    assert "linear_lab.preprocessing.FeatureEngineeringTransformer" in required
    assert "linear_lab.preprocessing.stable_f_regression" in required
    with pytest.raises(Exception, match="[Tt]rust"):
        sio.loads(response.content)
    passport = download(client, identifier, "passport").json()
    assert passport["skops_required_trusted_types"] == sorted(required)
    # For this generated, known source package only, audit the exact trusted type
    # names before granting them. The API never accepts a foreign model upload.
    assert all(name.startswith(("linear_lab.", "scipy.", "numpy.", "sklearn.")) for name in required), required
    original = joblib.load(root / "jobs" / identifier / "model.joblib")
    restored = sio.loads(response.content, trusted=required)
    np.testing.assert_allclose(restored.predict(probes()), original.predict(probes()), rtol=0, atol=0)
    uploaded = client.post("/api/predict", json={"job_id": identifier,
        "model": response.content.hex(), "rows": probes().iloc[[1]].to_dict(orient="records")})
    assert uploaded.status_code == 200
    np.testing.assert_allclose(uploaded.json()["predictions"], original.predict(probes().iloc[[1]]))


@pytest.mark.parametrize("config,reason", [
    ({"imputation": "knn", "scaler": "standard"}, "KNN"),
    ({"numeric_transform": "yeo_johnson", "scaler": "standard"}, "ONNX"),
])
def test_unsupported_onnx_is_disabled_but_native_full_pipeline_remains_usable(trained_api, config, reason):
    client, root, dataset_id, _, _ = trained_api
    request = {"dataset_id": dataset_id, "target": "price", "features": ["distance", "weight"],
               "model": "ridge", "params": {"alpha": .7}, "preprocessing": config,
               "regularization_path": False, "cv": 0}
    started = client.post("/api/jobs", json=request)
    assert started.status_code == 200, started.text
    identifier = started.json()["id"]
    wait_for_job(client, identifier)
    response = client.get(f"/api/jobs/{identifier}/export-capabilities")
    assert response.status_code == 200, response.text
    formats = {item["format"]: item for item in response.json()["formats"]}
    assert not formats["onnx"]["available"]
    assert reason in formats["onnx"]["reason"]
    assert formats["joblib"]["available"] and formats["pickle"]["available"] and formats["bundle"]["available"]
    unavailable = client.get(f"/api/jobs/{identifier}/export?format=onnx")
    assert unavailable.status_code == 422
    assert reason in unavailable.json()["detail"]
    native = joblib.load(BytesIO(download(client, identifier, "joblib").content))
    original = joblib.load(root / "jobs" / identifier / "model.joblib")
    rows = probes()[["distance", "weight"]]
    np.testing.assert_allclose(native.predict(rows), original.predict(rows), rtol=0, atol=0)


def test_bundle_recursively_copies_package_code_and_licenses_only(trained_api, tmp_path, monkeypatch):
    from linear_lab import exports
    client, root, _, identifier, state = trained_api
    package = tmp_path / "linear_lab"
    nested = package / "_vendor" / "sampler"
    nested.mkdir(parents=True)
    (package / "__init__.py").write_text("# package\n")
    (package / "exports.py").write_text("# export module\n")
    (nested / "__init__.py").write_text("# nested sampler\n")
    (nested / "algorithm.py").write_text("def run(): return 1\n")
    (nested / "LICENSE").write_text("Test licence\n")
    (nested / "rows.csv").write_text("private,training,rows\n")
    (nested / "model.joblib").write_bytes(b"private model")
    (nested / ".secret.py").write_text("private = 1\n")
    (nested / "__pycache__").mkdir()
    (nested / "__pycache__" / "not_source.py").write_text("cache\n")
    monkeypatch.setattr(exports, "__file__", str(package / "exports.py"))
    request = json.loads((root / "jobs" / identifier / "request.json").read_text())
    service = ModelExportService(root)
    data, _, _ = service.export(identifier, "bundle", state["result"], request)
    with ZipFile(BytesIO(data)) as archive:
        files = set(archive.namelist())
        assert {"linear_lab/_vendor/sampler/__init__.py", "linear_lab/_vendor/sampler/algorithm.py",
                "linear_lab/_vendor/sampler/LICENSE"} <= files
        assert not any(name.endswith("rows.csv") or "secret" in name or "__pycache__" in name for name in files)
        assert "linear_lab/_vendor/sampler/model.joblib" not in files


def test_passport_reveal_changes_metrics_without_changing_the_fitted_model(trained_api):
    client, root, _, identifier, _ = trained_api
    before = download(client, identifier, "joblib").content
    assert download(client, identifier, "passport").json()["metrics"]["test"] == {}
    revealed = client.post(f"/api/jobs/{identifier}/reveal-test")
    assert revealed.status_code == 200, revealed.text
    assert revealed.json()["result"]["metrics"]["test"]["mse"] >= 0
    assert download(client, identifier, "passport").json()["metrics"]["test"]["mse"] >= 0
    with ZipFile(BytesIO(download(client, identifier, "bundle").content)) as archive:
        assert json.loads(archive.read("passport.json"))["metrics"]["test"]["mse"] >= 0
    assert download(client, identifier, "joblib").content == before
