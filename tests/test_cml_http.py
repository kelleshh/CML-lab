"""Real HTTP journeys with isolated storage, spawn workers and saved artifacts."""
from io import BytesIO
import time

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient

from cml_lab.presentation.http.api import create_app


def completed(client, run_id):
    deadline = time.monotonic() + 40
    while time.monotonic() < deadline:
        response = client.get(f"/api/cml/runs/{run_id}")
        assert response.status_code == 200, response.text
        run = response.json()
        if run["status"] in {"completed", "error", "interrupted", "cancelled"}:
            assert run["status"] == "completed", run
            return run
        time.sleep(.03)
    pytest.fail("The isolated spawn worker did not finish within 40 seconds")


@pytest.mark.parametrize("task,algorithm", [
    ("regression", "ridge"), ("classification", "logistic_regression"),
    ("clustering", "kmeans"), ("ranking", "xgboost_ranker"),
    ("forecasting", "ridge"), ("panel", "ridge"),
    ("anomaly", "isolation_forest"), ("reduction", "pca"),
])
def test_http_train_export_reload_and_delete_journey(tmp_path, task, algorithm):
    """A saved experiment protects the dataset/model and seals its test snapshot."""
    run_id = None
    with TestClient(create_app(tmp_path)) as client:
        health = client.get("/api/health").json()
        assert health["name"] == "CML-lab" and health["status"] == "ok"
        response = client.post("/api/datasets/load", json={
            "name": "linear" if task == "regression" else task,
            "kind": "synthetic", "params": {"n_samples": 180, "seed": 13},
        })
        assert response.status_code == 200, response.text
        dataset = response.json()
        rows = client.get(f"/api/datasets/{dataset['id']}/rows?limit=500").json()["rows"]
        # Roles and target come from the public catalogue; the browser need not
        # know the generator's internal column names.
        response = client.post("/api/cml/runs", json={
            "dataset_id": dataset["id"], "task": task,
            "algorithm_id": algorithm, "roles": dataset.get("roles", {}),
            "validation": {"strategy": "none"}, "seed": 13,
            "params": {"n_estimators": 8} if task in {"ranking", "anomaly"} else {},
        })
        assert response.status_code == 200, response.text
        run_id = response.json()["id"]
        run = completed(client, run_id)
        assert run["spec"]["task"] == task
        if task not in {"clustering", "anomaly", "reduction"}:
            assert run["result"]["evaluations"]["test"] == {"hidden": True}
        saved = client.post(f"/api/cml/runs/{run_id}/save", json={"name": "Мой опыт"})
        assert saved.status_code == 200, saved.text
        experiment_id = saved.json()["id"]
        assert client.delete(f"/api/cml/models/{run_id}").status_code == 409
        assert client.delete(f"/api/datasets/{dataset['id']}").status_code == 409
        exported = client.get(f"/api/cml/runs/{run_id}/export?format=joblib")
        assert exported.status_code == 200, exported.text
        artifact = joblib.load(BytesIO(exported.content))
        inputs = [{key: value for key, value in row.items() if key in run["spec"]["features"]} for row in rows[:3]]
        if task in {"forecasting", "panel"}:
            assert client.post("/api/cml/predict", json={"run_id": run_id, "rows": inputs}).status_code == 422
            role = run["spec"]["roles"].get("entity_column")
            groups = {}
            for row in rows:
                groups.setdefault(row[role] if role else "series", []).append(row)
            history = [row for group in groups.values() for row in group[:-2]]
            future = [{key: value for key, value in row.items() if key != run["spec"]["target"]}
                      for group in groups.values() for row in group[-2:]]
            response = client.post("/api/cml/forecast", json={"run_id": run_id, "history": history, "future": future})
            assert response.status_code == 200, response.text
            assert len(response.json()["rows"]) == len(future)
        else:
            response = client.post("/api/cml/predict", json={"run_id": run_id, "rows": inputs})
            assert response.status_code == 200, response.text
            result = response.json()
            if task == "reduction":
                np.testing.assert_allclose(result["coordinates"], artifact.reduce(inputs))
            elif task == "classification":
                assert result["predictions"] == artifact.predict(inputs).tolist()
                np.testing.assert_allclose(result["probabilities"], artifact.predict_proba(inputs))
            else:
                np.testing.assert_allclose(result["predictions"], artifact.predict(inputs))
        if task not in {"clustering", "anomaly", "reduction"}:
            opened = client.post(f"/api/cml/runs/{run_id}/reveal-test").json()
            assert opened["result"]["evaluations"]["test"]["n_rows"] > 0
            # The archived evaluation is an immutable snapshot of what was
            # visible when saved, even after opening the current run's test.
            snapshot = client.get(f"/api/cml/experiments/{experiment_id}").json()
            assert snapshot["result"]["evaluations"]["test"] == {"hidden": True}

    with TestClient(create_app(tmp_path)) as restored:
        assert restored.get(f"/api/cml/models/{run_id}").status_code == 200
        assert restored.get(f"/api/cml/experiments/{experiment_id}").status_code == 200
        assert restored.delete(f"/api/cml/experiments/{experiment_id}").status_code == 200
        assert restored.delete(f"/api/cml/models/{run_id}").status_code == 200
        assert restored.get(f"/api/cml/models/{run_id}").status_code == 404
        assert not (tmp_path / "artifacts" / f"{run_id}.joblib").exists()
        assert restored.delete(f"/api/datasets/{dataset['id']}").status_code == 200


def test_http_recipe_revision_and_dataset_snapshot(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        original = client.post("/api/datasets/load", json={"name": "linear"}).json()
        renamed = client.patch(f"/api/datasets/{original['id']}/metadata", json={"name": "Новая редакция"})
        assert renamed.status_code == 200, renamed.text
        assert renamed.json()["id"] != original["id"]
        assert client.get(f"/api/datasets/{original['id']}").json()["name"] == original["name"]
        recipe = client.post("/api/cml/model-recipes", json={
            "name": "Свой лес", "config": {"algorithm_id": "random_forest_regressor", "params": {"max_depth": 3}},
        }).json()
        revised = client.patch(f"/api/cml/model-recipes/{recipe['id']}", json={"name": "Переименованный", "expected_revision": recipe["revision"]})
        assert revised.status_code == 200, revised.text
        assert revised.json()["config"] == recipe["config"]
        stale = client.patch(f"/api/cml/model-recipes/{recipe['id']}", json={"name": "Устаревшая форма", "expected_revision": recipe["revision"]})
        assert stale.status_code == 409
        assert client.get(f"/api/cml/model-recipes/{recipe['id']}").json()["name"] == "Переименованный"


def test_http_diagnostic_flags_reject_truthy_strings_before_starting(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        dataset = client.post("/api/datasets/load", json={"name": "linear"}).json()
        for flag in ("regularization_path", "learning_curve", "permutation_importance"):
            response = client.post("/api/cml/runs", json={"dataset_id": dataset["id"], flag: "false"})
            assert response.status_code == 422, response.text
        assert client.get("/api/cml/runs").json()["total"] == 0
