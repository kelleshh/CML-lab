"""Independent numerical and real-worker acceptance checks for parameter search.

The reference scores use sklearn primitives and raw train indexes, rather than
the application's CV evaluator. Search does not get access to outer holdouts.
"""

from __future__ import annotations

from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
import pickle
import time

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

from linear_lab.app import create_app
from linear_lab.datasets import DataService
from linear_lab.models import ModelRegistry
from linear_lab.search import SearchService
from linear_lab.validation import build_cv, group_values, model_features, split_holdout


SEED = 31


def rows_for_audit(size=96, *, groups=False):
    rng = np.random.default_rng(15)
    features = rng.normal(size=(size, 2))
    features[:, 1] += .7 * features[:, 0]
    y = 4 + 2.5 * features[:, 0] - .8 * features[:, 1] + rng.normal(0, 2, size)
    return [dict(x=float(x), z=float(z), answer=float(target), **({"subject": f"person-{index // 6}"} if groups else {}))
            for index, ((x, z), target) in enumerate(zip(features, y))]


def search_request(dataset_id, artifact):
    return {
        "dataset_id": dataset_id, "target": "answer", "features": ["x", "z"],
        "model": "ridge", "params": {"fit_intercept": True}, "seed": SEED,
        "preprocessing": {"degree": 1, "scaler": "standard", "imputation": "median"},
        "resampling": {"method": "none"},
        "split": {"train": .6, "validation": .2, "test": .2, "shuffle": True},
        "cv_config": {"strategy": "kfold", "folds": 3, "shuffle": True},
        "regularization_path": False, "metrics": ["custom", "rmse"],
        "custom_metric": "-mean(error**2)", "artifact_path": str(artifact),
        "search": {"method": "grid", "trials": 3, "metric": "custom", "direction": "max", "n_jobs": 2,
                   "param_space": {"alpha": [0., 2., 100.]}},
    }


@pytest.fixture
def numerical_lab(tmp_path):
    data = DataService(tmp_path / "datasets")
    rows = rows_for_audit()
    dataset = data.load({"kind": "custom", "rows": rows, "target": "answer"})
    return data, ModelRegistry(), rows, search_request(dataset["id"], tmp_path / "winner.joblib")


def independent_outer_indices(size, split, seed=SEED):
    """Use sklearn splitting directly, preserving train order for real SGD."""
    train, rest = train_test_split(np.arange(size), train_size=split["train"], random_state=seed)
    valid, test = train_test_split(rest, train_size=split["validation"] / (1 - split["train"]), random_state=seed + 1)
    return train, valid, test


def independent_fold_scores(rows, request, params):
    table = pd.DataFrame(rows)
    train, _, _ = independent_outer_indices(len(rows), request["split"], request["seed"])
    X = table.loc[train, ["x", "z"]].to_numpy()
    y = table.loc[train, "answer"].to_numpy()
    scores = []
    for fit, check in KFold(n_splits=3, shuffle=True, random_state=request["seed"]).split(X):
        scaler = StandardScaler().fit(X[fit])
        if request["model"] == "sgd":
            model = SGDRegressor(random_state=request["seed"], **params)
            for _ in range(request["epochs"]):
                model.partial_fit(scaler.transform(X[fit]), y[fit])
        else:
            model = Ridge(**params).fit(scaler.transform(X[fit]), y[fit])
        scores.append(-float(np.mean((y[check] - model.predict(scaler.transform(X[check]))) ** 2)))
    return scores


def test_custom_max_scores_match_independent_cv_and_native_refit(numerical_lab):
    data, registry, rows, request = numerical_lab
    result = SearchService(data, registry).run(request)
    expected = {}
    for trial in result["search"]["trials"]:
        params = {**request["params"], **trial["params"]}
        scores = independent_fold_scores(rows, request, params)
        np.testing.assert_allclose(trial["fold_scores"], scores, atol=1e-12, rtol=1e-12)
        assert trial["score"] == pytest.approx(np.mean(scores), abs=1e-12)
        expected[params["alpha"]] = np.mean(scores)
    winning_alpha = max(expected, key=expected.get)
    assert result["search"]["best_params"]["alpha"] == winning_alpha
    assert result["search"]["best_score"] == pytest.approx(expected[winning_alpha])
    assert result["effective_request"]["params"] == result["search"]["best_params"]
    assert "artifact_path" not in result["effective_request"]

    table = pd.DataFrame(rows)
    train, _, _ = independent_outer_indices(len(rows), request["split"])
    scaler = StandardScaler().fit(table.loc[train, ["x", "z"]])
    X = scaler.transform(table.loc[train, ["x", "z"]])
    y = table.loc[train, "answer"].to_numpy()
    centered_y = y - y.mean()
    # Ridge solves (X'X + alpha I)w = X'(y - mean(y)); scaling is train-only.
    closed_form = np.linalg.solve(X.T @ X + winning_alpha * np.eye(2), X.T @ centered_y)
    artifact = joblib.load(request["artifact_path"])
    np.testing.assert_allclose(artifact.named_steps["model"].coef_, closed_form, atol=1e-12)
    assert float(np.asarray(artifact.named_steps["model"].intercept_).reshape(-1)[0]) == pytest.approx(y.mean())
    np.testing.assert_allclose(artifact.named_steps["preprocessing"].named_transformers_["numeric"].named_steps["scale"].mean_, scaler.mean_, atol=1e-12)
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("changed_column", ["answer", "x", "z"])
def test_each_outer_holdout_column_is_independent_of_search_and_artifact(numerical_lab, changed_column):
    data, registry, rows, request = numerical_lab
    first = SearchService(data, registry).run(request)
    first_model = joblib.load(request["artifact_path"])
    changed = deepcopy(rows)
    _, valid, test = independent_outer_indices(len(rows), request["split"])
    for index in np.concatenate([valid, test]):
        changed[index][changed_column] += 100_000 + int(index)
    dataset = data.load({"kind": "custom", "rows": changed, "target": "answer"})
    request["dataset_id"] = dataset["id"]
    second = SearchService(data, registry).run(request)
    second_model = joblib.load(request["artifact_path"])
    assert first["search"]["best_params"] == second["search"]["best_params"]
    np.testing.assert_allclose([item["score"] for item in first["search"]["trials"]],
                               [item["score"] for item in second["search"]["trials"]], atol=0, rtol=0)
    probe = pd.DataFrame(rows)[["x", "z"]].iloc[:10]
    np.testing.assert_allclose(first_model.predict(probe), second_model.predict(probe), atol=0, rtol=0)
    np.testing.assert_allclose(first_model.named_steps["model"].coef_, second_model.named_steps["model"].coef_, atol=0, rtol=0)


def test_sgd_search_and_refit_use_seven_real_partial_fit_epochs(numerical_lab):
    data, registry, rows, request = numerical_lab
    request.update(model="sgd", epochs=7, params={"learning_rate": "constant", "eta0": .003, "penalty": "l2", "loss": "squared_error", "fit_intercept": True})
    request["search"]["param_space"] = {"alpha": [.001, .2]}
    request["search"]["trials"] = 2
    result = SearchService(data, registry).run(request)
    for trial in result["search"]["trials"]:
        reference = independent_fold_scores(rows, request, {**request["params"], **trial["params"]})
        np.testing.assert_allclose(trial["fold_scores"], reference, atol=1e-12, rtol=1e-12)
    table = pd.DataFrame(rows)
    train, _, _ = independent_outer_indices(len(rows), request["split"])
    scaler = StandardScaler().fit(table.loc[train, ["x", "z"]])
    reference = SGDRegressor(random_state=SEED, **result["search"]["best_params"])
    for _ in range(7):
        reference.partial_fit(scaler.transform(table.loc[train, ["x", "z"]]), table.loc[train, "answer"])
    artifact = joblib.load(request["artifact_path"])
    assert artifact.named_steps["model"].t_ == 7 * len(train) + 1
    assert artifact.named_steps["model"].max_iter == 7
    assert len(result["loss_history"]) == 7
    np.testing.assert_allclose(artifact.named_steps["model"].coef_, reference.coef_, atol=1e-12)


def test_seeded_optuna_tpe_sequential_proposals_are_reproducible(numerical_lab):
    data, registry, _, request = numerical_lab
    request["search"].update(method="optuna_tpe", trials=9, param_space={"alpha": {"type": "float", "low": 1e-5, "high": 1e3, "log": True}})
    first = SearchService(data, registry).run(request)
    second = SearchService(data, registry).run(request)
    assert len(first["search"]["trials"]) == 9
    assert first["search"]["best_params"] == second["search"]["best_params"]
    assert [(item["params"], item["score"], item["fold_scores"], item["status"]) for item in first["search"]["trials"]] == [(item["params"], item["score"], item["fold_scores"], item["status"]) for item in second["search"]["trials"]]


def test_time_order_and_inner_gap_are_preserved_by_search_refit(numerical_lab):
    data, registry, rows, request = numerical_lab
    request["cv_config"] = {"strategy": "timeseries", "folds": 3, "gap": 4, "test_size": 8, "max_train_size": 20}
    result = SearchService(data, registry).run(request)
    parts = split_holdout(len(rows), request["split"], SEED, cv_config=request["cv_config"])
    assert parts["train"].max() < parts["validation"].min() < parts["test"].min()
    # The documented gap is an inner CV parameter; outer holdouts stay adjacent.
    assert parts["train"].max() + 1 == parts["validation"].min()
    plan = build_cv(request["cv_config"], len(parts["train"]), seed=SEED)
    for train, check in plan.splits:
        assert check.min() - train.max() - 1 == 4
        assert len(train) <= 20 and len(check) == 8
    assert result["effective_request"]["split"]["shuffle"] is False
    pipeline = joblib.load(request["artifact_path"])
    expected_mean = pd.DataFrame(rows).loc[parts["train"], ["x", "z"]].mean().to_numpy()
    np.testing.assert_allclose(pipeline.named_steps["preprocessing"].named_transformers_["numeric"].named_steps["scale"].mean_, expected_mean)


def test_groups_are_split_only_identifiers_at_outer_and_inner_boundaries(tmp_path):
    data = DataService(tmp_path / "datasets")
    rows = rows_for_audit(120, groups=True)
    dataset = data.load({"kind": "custom", "rows": rows, "target": "answer"})
    request = search_request(dataset["id"], tmp_path / "group.joblib")
    request["features"].append("subject")
    request["cv_config"] = {"strategy": "group_kfold", "group_column": "subject", "folds": 3}
    groups = group_values(data, request)
    parts = split_holdout(len(rows), request["split"], SEED, groups, request["cv_config"])
    for first, second in (("train", "validation"), ("train", "test"), ("validation", "test")):
        assert not set(groups[parts[first]]).intersection(groups[parts[second]])
    inner_groups = groups[parts["train"]]
    plan = build_cv(request["cv_config"], len(inner_groups), groups=inner_groups, seed=SEED)
    assert all(not set(inner_groups[train]).intersection(inner_groups[check]) for train, check in plan.splits)
    assert model_features(data, request) == ["x", "z"]
    result = SearchService(data, ModelRegistry()).run(request)
    artifact = joblib.load(request["artifact_path"])
    assert list(artifact.feature_names_in_) == ["x", "z"]
    assert result["effective_request"]["features"] == ["x", "z"]


@pytest.mark.parametrize("method", ["halving_grid", "halving_random"])
@pytest.mark.parametrize("strategy", ["timeseries", "group_kfold"])
def test_halving_rejects_time_and_group_subsampling_before_any_artifact(tmp_path, method, strategy):
    data = DataService(tmp_path / "datasets")
    dataset = data.load({"kind": "custom", "rows": rows_for_audit(120, groups=True), "target": "answer"})
    request = search_request(dataset["id"], tmp_path / "forbidden.joblib")
    request["cv_config"] = {"strategy": strategy, "folds": 3, **({"group_column": "subject"} if strategy == "group_kfold" else {})}
    request["search"]["method"] = method
    with pytest.raises(ValueError, match="Сокращение по числу строк"):
        SearchService(data, ModelRegistry()).run(request)
    assert not (tmp_path / "forbidden.joblib").exists()


@pytest.mark.parametrize("change,match", [
    ({"trials": 101}, "от 1 до 100"),
    ({"trials": True}, "от 1 до 100"),
    ({"n_jobs": 5}, "от 1 до 4"),
    ({"metric": "all"}, "одна главная"),
    ({"param_space": {"alpha": {"type": "unknown", "low": 0, "high": 2}}}, "тип поиска"),
    ({"param_space": {"fit_intercept": {"type": "float", "low": 0, "high": 1}}}, "true или false"),
    ({"param_space": {"invented_parameter": [1]}}, "не поддерживает"),
    ({"param_space": {"alpha": {"type": "float", "low": 0, "high": 1, "log": True}}}, "строго положительным"),
])
def test_invalid_search_settings_cannot_start_fits(numerical_lab, change, match):
    data, registry, _, request = numerical_lab
    request["search"].update(change)
    with pytest.raises(ValueError, match=match):
        SearchService(data, registry).run(request)
    assert not Path(request["artifact_path"]).exists()


def test_search_fit_cap_is_enforced_before_work(numerical_lab):
    data, registry, _, request = numerical_lab
    request["search"].update(method="optuna_tpe", trials=100)
    request["cv_config"]["folds"] = 6
    with pytest.raises(ValueError, match="600 обучений; предел 500"):
        SearchService(data, registry).run(request)
    assert not Path(request["artifact_path"]).exists()
    SearchService._budget(500)
    with pytest.raises(ValueError, match="предел 500"):
        SearchService._budget(501)


def test_huge_grid_is_rejected_from_cardinality_without_materializing(numerical_lab, monkeypatch):
    data, registry, _, request = numerical_lab
    request.update(model="sgd", epochs=7, params={})
    # Eight supported dimensions with 100 values imply 10**16 candidates.
    # Evaluating cardinality is safe; constructing the candidate list is not.
    request["search"]["param_space"] = {
        parameter["key"]: [parameter["default"]] * 100
        for parameter in registry.spec("sgd")["params"]
        if parameter["key"] not in {"max_iter", "tol"}
    }
    from sklearn.model_selection import ParameterGrid
    def materializing_is_forbidden(self):
        pytest.fail("ParameterGrid iteration occurred before rejecting its budget")
    monkeypatch.setattr(ParameterGrid, "__iter__", materializing_is_forbidden)
    with pytest.raises(ValueError, match="Сетка содержит .* комбинаций"):
        SearchService(data, registry).run(request)
    assert not Path(request["artifact_path"]).exists()


def poll_job(client, identifier, *, timeout=60, first_trial=False):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/api/jobs/{identifier}")
        assert response.status_code == 200, response.text
        state = response.json()
        if first_trial and any("search_trial" in event for event in state["events"]):
            return state
        if state["status"] != "running":
            return state
        time.sleep(.03)
    pytest.fail(f"Worker did not reach requested state: {identifier}")


def test_search_real_worker_winner_save_and_all_available_native_formats(tmp_path):
    rows = rows_for_audit()
    with TestClient(create_app(tmp_path)) as client:
        response = client.post("/api/datasets/load", json={"kind": "custom", "rows": rows, "target": "answer"})
        assert response.status_code == 200, response.text
        request = search_request(response.json()["id"], tmp_path / "client-cannot-choose.joblib")
        started = client.post("/api/jobs", json=request)
        assert started.status_code == 200, started.text
        identifier = started.json()["id"]
        state = poll_job(client, identifier)
        assert state["status"] == "completed", state
        result = state["result"]
        assert "artifact_path" not in result["effective_request"]
        assert not result["metrics"].get("test")
        assert "test" not in result["predictions"]["split"]
        assert not (tmp_path / "client-cannot-choose.joblib").exists()

        saved = client.post("/api/experiments", json={"job_id": identifier, "name": "Search audit winner"})
        assert saved.status_code == 200, saved.text
        experiment = saved.json()
        assert experiment["request"]["params"] == result["search"]["best_params"]
        # API configurations must not disclose an internal worker artifact path.
        assert "artifact_path" not in experiment["request"]
        assert client.get(f"/api/experiments/{experiment['id']}").json()["request"]["params"] == result["search"]["best_params"]

        capabilities = client.get(f"/api/jobs/{identifier}/export-capabilities")
        assert capabilities.status_code == 200, capabilities.text
        native = {item["format"]: item for item in capabilities.json()["formats"]}
        assert native["joblib"]["available"] and native["pickle"]["available"]
        probe = pd.DataFrame(rows).iloc[:5][["x", "z"]]
        prediction = client.post("/api/predict", json={"job_id": identifier, "rows": probe.to_dict(orient="records")})
        assert prediction.status_code == 200, prediction.text
        expected = prediction.json()["predictions"]
        for format in ("joblib", "pickle", "skops", "onnx"):
            if not native[format]["available"]:
                continue
            exported = client.get(f"/api/jobs/{identifier}/export?format={format}")
            assert exported.status_code == 200, (format, exported.text)
            if format == "joblib":
                loaded = joblib.load(BytesIO(exported.content))
                actual = loaded.predict(probe)
                assert loaded.named_steps["model"].alpha == result["search"]["best_params"]["alpha"]
            elif format == "pickle":
                actual = pickle.loads(exported.content).predict(probe)
            elif format == "skops":
                import skops.io as sio
                trusted = sio.get_untrusted_types(data=exported.content)
                actual = sio.loads(exported.content, trusted=trusted).predict(probe)
            else:
                import onnxruntime as ort
                session = ort.InferenceSession(exported.content, providers=["CPUExecutionProvider"])
                feed = {item.name: probe[[item.name]].to_numpy(dtype=np.float64) for item in session.get_inputs()}
                actual = np.asarray(session.run(None, feed)[0]).reshape(-1)
            np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
        passport = client.get(f"/api/jobs/{identifier}/export?format=passport")
        assert passport.status_code == 200, passport.text
        assert passport.json()["configuration"]["params"] == result["search"]["best_params"]
        assert "artifact_path" not in passport.json()["configuration"]


def test_real_optuna_worker_cancellation_keeps_completed_trials_and_no_model(tmp_path):
    app = create_app(tmp_path)
    with TestClient(app) as client:
        loaded = client.post("/api/datasets/load", json={"kind": "custom", "rows": rows_for_audit(2000), "target": "answer"})
        assert loaded.status_code == 200, loaded.text
        request = search_request(loaded.json()["id"], tmp_path / "forbidden-partial.joblib")
        request.update(model="sgd", epochs=1000, params={"eta0": .001, "learning_rate": "constant", "penalty": "l2", "loss": "squared_error"})
        request["search"].update(method="optuna_tpe", trials=60, n_jobs=2, param_space={"alpha": {"type": "float", "low": 1e-6, "high": .1, "log": True}})
        started = client.post("/api/jobs", json=request)
        assert started.status_code == 200, started.text
        identifier = started.json()["id"]
        pending = poll_job(client, identifier, first_trial=True)
        assert pending["status"] == "running", pending
        assert 0 < len([event for event in pending["events"] if "search_trial" in event]) < 60
        cancelled = client.delete(f"/api/jobs/{identifier}")
        assert cancelled.status_code == 200, cancelled.text
        assert cancelled.json()["status"] == "cancelled"
        assert "result" not in cancelled.json()
        assert not app.state.jobs._processes[identifier].is_alive()
        directory = tmp_path / "jobs" / identifier
        assert not (directory / "model.joblib").exists()
        assert not (directory / "model.joblib.tmp").exists()
        assert not (directory / "result.json").exists()
        assert client.get(f"/api/jobs/{identifier}/export?format=model").status_code == 422
