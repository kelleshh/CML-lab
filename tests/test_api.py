"""Real HTTP, worker, artifact and persistence integration in isolated directories."""

from __future__ import annotations

import csv
import io
import time

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient

from linear_lab.app import create_app


def wait_for_terminal_state(client: TestClient, identifier: str, timeout: float = 45):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/api/jobs/{identifier}")
        assert response.status_code == 200, response.text
        state = response.json()
        if state["status"] in {"completed", "failed", "error", "cancelled"}:
            return state
        time.sleep(0.03)
    pytest.fail(f"Job {identifier} did not finish within {timeout} seconds")


def test_real_training_holdout_reveal_exports_and_experiment_survive_restart(tmp_path):
    rows = [{"x": i / 10, "y": 2 + 3 * i / 10} for i in range(80)]
    with TestClient(create_app(tmp_path)) as client:
        loaded = client.post("/api/datasets/load", json={"kind": "custom", "rows": rows, "target": "y"})
        assert loaded.status_code == 200, loaded.text
        dataset_id = loaded.json()["id"]
        started = client.post("/api/jobs", json={
            "dataset_id": dataset_id,
            "target": "y",
            "features": ["x"],
            "model": "ols",
            "params": {},
            "seed": 19,
            "split": {"train": 0.6, "validation": 0.2, "test": 0.2, "shuffle": True},
            "preprocessing": {"scale": True, "degree": 1, "impute": True},
            "metrics": ["mse", "r2"],
            "regularization_path": False,
            "cv": 0,
        })
        assert started.status_code == 200, started.text
        identifier = started.json()["id"]
        state = wait_for_terminal_state(client, identifier)
        assert state["status"] == "completed", state
        hidden = state["result"]
        assert not hidden["metrics"].get("test")
        assert "test" not in hidden["predictions"]["split"]
        assert "test" not in hidden["plot_data"]["split"]
        assert hidden["metrics"]["validation"]["mse"] < 1e-20
        assert all(len(frame["predicted"]) == len(hidden["predictions"]["actual"]) for frame in hidden["trace"])

        before_reveal_export = client.get(f"/api/jobs/{identifier}/export?format=json")
        assert before_reveal_export.status_code == 200, before_reveal_export.text
        assert not before_reveal_export.json()["metrics"].get("test")
        assert "test" not in before_reveal_export.json()["predictions"]["split"]
        before_reveal_csv = client.get(f"/api/jobs/{identifier}/export?format=csv")
        assert before_reveal_csv.status_code == 200
        before_reveal_rows = list(csv.DictReader(io.StringIO(before_reveal_csv.text.lstrip("\ufeff"))))
        assert "test" not in {row["split"] for row in before_reveal_rows}

        reveal = client.post(f"/api/jobs/{identifier}/reveal-test")
        assert reveal.status_code == 200, reveal.text
        revealed = client.get(f"/api/jobs/{identifier}").json()["result"]
        assert revealed["metrics"]["test"]["mse"] < 1e-20
        assert revealed["metrics"]["test"]["r2"] == pytest.approx(1)
        assert len(revealed["predictions"]["actual"]) == len(rows)

        json_export = client.get(f"/api/jobs/{identifier}/export?format=json")
        assert json_export.status_code == 200, json_export.text
        assert json_export.json()["metrics"]["test"]["mse"] < 1e-20

        csv_export = client.get(f"/api/jobs/{identifier}/export?format=csv")
        assert csv_export.status_code == 200, csv_export.text
        exported_rows = list(csv.DictReader(io.StringIO(csv_export.text.lstrip("\ufeff"))))
        assert len(exported_rows) == len(rows)
        assert {row["split"] for row in exported_rows} == {"train", "validation", "test"}
        for row in exported_rows:
            assert float(row["actual"]) == pytest.approx(float(row["predicted"]), abs=1e-8)

        model_export = client.get(f"/api/jobs/{identifier}/export?format=model")
        assert model_export.status_code == 200, model_export.text
        artifact = joblib.load(io.BytesIO(model_export.content))
        model = artifact.get("pipeline", artifact.get("model")) if isinstance(artifact, dict) else artifact
        import pandas as pd
        np.testing.assert_allclose(model.predict(pd.DataFrame({"x": [1, 2]})), [5, 8], atol=1e-8)

        saved = client.post("/api/experiments", json={"job_id": identifier, "name": "Known exact relation"})
        assert saved.status_code == 200, saved.text
        experiment_id = saved.json()["id"]

    with TestClient(create_app(tmp_path)) as restarted:
        experiment = restarted.get(f"/api/experiments/{experiment_id}")
        assert experiment.status_code == 200, experiment.text
        assert experiment.json()["name"] == "Known exact relation"
        assert any(item["id"] == experiment_id for item in restarted.get("/api/experiments").json())
        reloaded_dataset = restarted.get(f"/api/datasets/{dataset_id}")
        assert reloaded_dataset.status_code == 200
        assert reloaded_dataset.json()["rows"] == len(rows)


def test_custom_metric_computes_requested_error_and_rejects_python_execution(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        valid = client.post("/api/metrics/preview", json={
            "expression": "mean(abs(error))", "actual": [1, 2, 4], "predicted": [0, 4, 4],
        })
        assert valid.status_code == 200, valid.text
        assert valid.json()["value"] == pytest.approx(1)
        for expression in ("y.__class__", "[x for x in y]", "__import__('math').sqrt(4)", "(lambda: 1)()"):
            invalid = client.post("/api/metrics/preview", json={
                "expression": expression, "actual": [1, 2], "predicted": [1, 1],
            })
            assert invalid.status_code == 422, (expression, invalid.text)
            assert invalid.json()["detail"]


def test_failed_job_explains_invalid_positive_target_and_cannot_export(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        loaded = client.post("/api/datasets/load", json={
            "kind": "custom", "rows": [{"x": i, "y": -1} for i in range(40)], "target": "y",
        })
        assert loaded.status_code == 200, loaded.text
        started = client.post("/api/jobs", json={
            "dataset_id": loaded.json()["id"], "model": "gamma", "target": "y",
            "features": ["x"], "regularization_path": False,
        })
        assert started.status_code == 200, started.text
        state = wait_for_terminal_state(client, started.json()["id"])
        assert state["status"] in {"failed", "error"}, state
        assert state["error"]
        assert client.get(f"/api/jobs/{state['id']}/export?format=model").status_code == 422


def test_running_job_can_be_cancelled_without_exporting_partial_model(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        loaded = client.post("/api/datasets/load", json={
            "kind": "synthetic", "name": "linear",
            "params": {"n_samples": 1000, "n_features": 3, "noise": 3, "seed": 42},
        })
        assert loaded.status_code == 200, loaded.text
        started = client.post("/api/jobs", json={
            "dataset_id": loaded.json()["id"], "model": "sgd", "epochs": 1000,
            "regularization_path": False,
        })
        assert started.status_code == 200, started.text
        identifier = started.json()["id"]
        cancelled = client.delete(f"/api/jobs/{identifier}")
        assert cancelled.status_code == 200, cancelled.text
        assert cancelled.json()["status"] == "cancelled"
        assert client.get(f"/api/jobs/{identifier}").json()["status"] == "cancelled"
        assert client.get(f"/api/jobs/{identifier}/export?format=model").status_code == 422


def test_uploaded_russian_csv_trains_and_predicts_through_saved_pipeline(tmp_path):
    content = "расстояние,цель\n" + "\n".join(f"{i},{2 + 3 * i}" for i in range(30))
    with TestClient(create_app(tmp_path)) as client:
        uploaded = client.post("/api/datasets/upload", files={
            "file": ("данные.csv", content.encode("utf-8-sig"), "text/csv"),
        })
        assert uploaded.status_code == 200, uploaded.text
        assert uploaded.json()["rows"] == 30
        started = client.post("/api/jobs", json={
            "dataset_id": uploaded.json()["id"], "target": "цель", "features": ["расстояние"],
            "model": "ols", "regularization_path": False,
        })
        assert started.status_code == 200, started.text
        identifier = started.json()["id"]
        state = wait_for_terminal_state(client, identifier)
        assert state["status"] == "completed", state
        prediction = client.post("/api/predict", json={
            "job_id": identifier, "rows": [{"расстояние": 1}, {"расстояние": 5}],
        })
        assert prediction.status_code == 200, prediction.text
        np.testing.assert_allclose(prediction.json()["predictions"], [5, 17], atol=1e-8)
