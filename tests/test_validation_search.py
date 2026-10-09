"""CV boundaries, real search selection and regression metrics, not UI mocks."""

from __future__ import annotations

from copy import deepcopy
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from linear_lab.datasets import DataService
from linear_lab.models import ModelRegistry
from linear_lab.pipeline import RegressionPipeline
from linear_lab.search import SearchService
from linear_lab.training import TrainingCancelled
from linear_lab.validation import build_cv, evaluate_cv, split_holdout


@pytest.mark.parametrize("strategy,extra,count", [
    ("kfold", {}, 3), ("repeated_kfold", {"repeats": 2}, 6),
    ("shuffle_split", {}, 3), ("timeseries", {"gap": 2}, 3),
    ("group_kfold", {}, 3), ("group_shuffle_split", {}, 3),
    ("leave_one_group_out", {}, 6), ("leave_one_out", {}, 30),
    ("stratified_bins", {"bins": 3}, 3),
])
def test_every_cv_strategy_builds_disjoint_valid_parts(strategy, extra, count):
    y = np.arange(30, dtype=float)
    groups = np.repeat(np.arange(6), 5)
    plan = build_cv({"strategy": strategy, "folds": 3, **extra}, 30, y, groups, seed=9)
    assert plan.n_splits == count
    for train, valid in plan.splits:
        assert len(train) >= 2 and len(valid) >= 1
        assert not set(train).intersection(valid)
        assert np.all(train < 30) and np.all(valid < 30)
        if "group" in strategy:
            assert not set(groups[train]).intersection(groups[valid])
        if strategy == "timeseries":
            assert train.max() + 2 < valid.min()


def test_time_holdout_and_cv_never_train_on_future():
    parts = split_holdout(100, {"shuffle": True}, 7, cv_config={"strategy": "timeseries"})
    assert parts["train"].max() < parts["validation"].min()
    assert parts["validation"].max() < parts["test"].min()
    plan = build_cv({"strategy": "timeseries", "folds": 4, "gap": 3, "max_train_size": 12, "test_size": 6}, len(parts["train"]))
    for train, valid in plan.splits:
        assert len(train) <= 12 and len(valid) == 6
        assert train.max() + 3 < valid.min()


def test_group_holdout_and_group_cv_are_disjoint_at_both_levels():
    groups = np.repeat(np.arange(15), 8)
    parts = split_holdout(120, {}, 12, groups, {"strategy": "group_kfold"})
    group_sets = [set(groups[indices]) for indices in parts.values()]
    for first in range(3):
        for second in range(first + 1, 3):
            assert not group_sets[first].intersection(group_sets[second])
    train_groups = groups[parts["train"]]
    plan = build_cv({"strategy": "group_kfold", "folds": 3}, len(train_groups), groups=train_groups)
    assert all(not set(train_groups[a]).intersection(train_groups[b]) for a, b in plan.splits)


@pytest.mark.parametrize("config,n,match", [
    ({"strategy": "leave_one_out"}, 101, "предел"),
    ({"strategy": "repeated_kfold", "folds": 20, "repeats": 10}, 100, "предел"),
    ({"strategy": "timeseries", "gap": 99}, 100, "Нельзя"),
    ({"strategy": "invented"}, 100, "Неизвестная"),
    ({"strategy": "kfold", "folds": True}, 30, "целое"),
])
def test_cv_work_and_invalid_settings_are_rejected(config, n, match):
    with pytest.raises(ValueError, match=match):
        build_cv(config, n)


class RecordingScale(TransformerMixin, BaseEstimator):
    fits = []

    def fit(self, X, y=None):
        self.mean_ = np.asarray(X).mean(axis=0)
        self.__class__.fits.append((np.asarray(X).copy(), np.asarray(y).copy()))
        return self

    def transform(self, X):
        return np.asarray(X) - self.mean_


def test_parallel_cv_clones_whole_unfitted_pipeline_for_each_fold():
    X = pd.DataFrame({"x": np.arange(40.)})
    y = 3 * X.x.to_numpy() + 1
    RecordingScale.fits = []
    pipeline = Pipeline([("scale", RecordingScale()), ("model", Ridge(alpha=0))])
    plan = build_cv({"folds": 4}, 40, y, seed=17)
    result = evaluate_cv(pipeline, X, y, {"folds": 4}, ["mse", "r2"], seed=17, n_jobs=2, plan=plan)
    assert result["summary"]["mse"]["mean"] < 1e-20
    assert len(RecordingScale.fits) == 4
    assert not hasattr(pipeline.named_steps["scale"], "mean_")
    trained_sets = {tuple(np.sort(values[:, 0]).tolist()) for values, _ in RecordingScale.fits}
    assert trained_sets == {tuple(np.sort(X.x.iloc[indices]).tolist()) for indices, _ in plan.splits}


def test_cv_resampling_does_not_change_validation_rows():
    from linear_lab.preprocessing import build_preprocessor
    X = pd.DataFrame({"x": np.arange(80.)})
    y = X.x.to_numpy() * 2 + 1
    preprocessing, _, _ = build_preprocessor(X, {"scaler": "none"})
    pipeline = RegressionPipeline([("preprocessing", preprocessing), ("model", Ridge(alpha=0))], resampling={"method": "none"})
    results = evaluate_cv(pipeline, X, y, {"folds": 4}, ["rmse"])
    assert all(fold["validation_size"] == 20 for fold in results["folds"])
    assert results["summary"]["rmse"]["mean"] < 1e-10


@pytest.fixture
def lab(tmp_path):
    return DataService(tmp_path / "data"), ModelRegistry(), tmp_path


def request_for(data, root, y=None, groups=False):
    rng = np.random.default_rng(17)
    X = rng.normal(size=(120, 2))
    y = 3 * X[:, 0] - 2 * X[:, 1] + 5 if y is None else y
    rows = [{"x": float(row[0]), "z": float(row[1]), "y": float(target), **({"group": str(i // 8)} if groups else {})} for i, (row, target) in enumerate(zip(X, y))]
    dataset_id = data.load({"kind": "custom", "rows": rows, "target": "y"})["id"]
    return {"dataset_id": dataset_id, "target": "y", "features": ["x", "z", *(["group"] if groups else [])], "model": "ridge", "params": {}, "seed": 42,
            "split": {"train": .6, "validation": .2, "test": .2, "shuffle": True}, "preprocessing": {"scale": True, "degree": 1, "impute": True},
            "metrics": ["rmse", "mae", "r2"], "regularization_path": False, "artifact_path": str(root / "model.joblib"),
            "cv_config": {"strategy": "kfold", "folds": 3}, "search": {"method": "grid", "trials": 3, "param_space": {"alpha": [0., 1., 100.]}, "metric": "rmse", "n_jobs": 2}}


def test_grid_selection_and_exported_refit_are_train_only(lab):
    data, registry, root = lab
    request = request_for(data, root)
    events = []
    result = SearchService(data, registry).run(request, events.append)
    assert result["search"]["best_params"]["alpha"] == 0
    assert len(result["search"]["trials"]) == 3
    assert result["search"]["best_score"] < 1e-10
    assert len([event for event in events if "search_trial" in event]) == 3
    assert all(0 <= event["progress"] <= 1 for event in events)
    assert result["effective_request"]["params"]["alpha"] == 0
    assert "artifact_path" not in result["effective_request"]
    pipeline = joblib.load(request["artifact_path"])
    bundle = data.resolve(request["dataset_id"], "y", ["x", "z"])
    parts = split_holdout(len(bundle.y), request["split"], request["seed"])
    np.testing.assert_allclose(pipeline.named_steps["preprocessing"].named_transformers_["numeric"].named_steps["scale"].mean_, bundle.X.iloc[parts["train"]].mean())
    json.dumps(result, allow_nan=False)


def test_search_never_uses_external_validation_or_test_to_pick(lab):
    data, registry, root = lab
    request = request_for(data, root)
    first = SearchService(data, registry).run(request)
    bundle = data.resolve(request["dataset_id"], "y", ["x", "z"])
    parts = split_holdout(len(bundle.y), request["split"], request["seed"])
    changed_y = bundle.y.copy()
    changed_y[np.concatenate([parts["validation"], parts["test"]])] += 1e7
    other = request_for(data, root, changed_y)
    second = SearchService(data, registry).run(other)
    assert first["search"]["best_params"] == second["search"]["best_params"]
    np.testing.assert_allclose([trial["score"] for trial in first["search"]["trials"]], [trial["score"] for trial in second["search"]["trials"]])
    np.testing.assert_allclose(first["coefficients"], second["coefficients"])


@pytest.mark.parametrize("method", ["optuna_tpe", "optuna_random", "random"])
def test_real_seeded_sampling_and_custom_maximization(lab, monkeypatch, method):
    import optuna
    data, registry, root = lab
    request = request_for(data, root)
    request["search"] = {"method": method, "trials": 5, "param_space": {"alpha": {"type": "float", "low": 1e-5, "high": 10., "log": True}}, "metric": "custom", "direction": "max", "n_jobs": 2}
    request["custom_metric"] = "-mean(error**2)"
    studies = []
    actual_create = optuna.create_study

    def capture(*args, **kwargs):
        study = actual_create(*args, **kwargs)
        studies.append(study)
        return study

    monkeypatch.setattr(optuna, "create_study", capture)
    first = SearchService(data, registry).run(request)
    second = SearchService(data, registry).run(request)
    assert len(first["search"]["trials"]) == 5
    assert first["search"]["best_score"] == max(trial["score"] for trial in first["search"]["trials"])
    assert first["search"]["best_params"] == second["search"]["best_params"]
    assert [t["params"] for t in first["search"]["trials"]] == [t["params"] for t in second["search"]["trials"]]
    if method.startswith("optuna"):
        assert len(studies) == 2
        assert all(len(study.trials) == 5 for study in studies)
        assert first["search"]["best_score"] == studies[0].best_value
        assert first["search"]["best_params"]["alpha"] == studies[0].best_params["alpha"]


def test_group_search_refit_excludes_group_identifier(lab):
    data, registry, root = lab
    request = request_for(data, root, groups=True)
    request["cv_config"] = {"strategy": "group_kfold", "folds": 3, "group_column": "group"}
    result = SearchService(data, registry).run(request)
    assert result["search"]["best_params"]["alpha"] == 0
    assert "group" not in result["feature_names"]
    assert result["effective_request"]["features"] == ["x", "z"]
    pipeline = joblib.load(request["artifact_path"])
    assert "group" not in pipeline.feature_names_in_


@pytest.mark.parametrize("method", ["halving_grid", "halving_random"])
def test_official_halving_selects_only_final_resource_rung(lab, method):
    data, registry, root = lab
    request = request_for(data, root)
    request["search"]["method"] = method
    result = SearchService(data, registry).run(request)
    records = result["search"]["trials"]
    final = max(record["iteration"] for record in records)
    eligible = [record for record in records if record["iteration"] == final and record["score"] is not None]
    assert result["search"]["best_trial"] in [record["trial"] for record in eligible]
    assert result["search"]["best_score"] == min(record["score"] for record in eligible)
    assert max(record["resources"] for record in records) > min(record["resources"] for record in records)


def test_search_cancel_before_first_fit(lab):
    data, registry, root = lab
    with pytest.raises(TrainingCancelled):
        SearchService(data, registry).run(request_for(data, root), cancelled=lambda: True)
    assert not (root / "model.joblib").exists()


def test_search_invalid_metric_and_work_budget_raise_actionable_error(lab):
    data, registry, root = lab
    request = request_for(data, root, np.ones(120))
    request["search"]["metric"] = "r2"
    with pytest.raises(ValueError, match="Ни один кандидат"):
        SearchService(data, registry).run(request)
    request["search"] = {"method": "optuna_random", "trials": 100, "param_space": {"alpha": [0., 1.]}}
    request["cv_config"]["folds"] = 10
    with pytest.raises(ValueError, match="предел 500"):
        SearchService(data, registry).run(request)


@pytest.mark.parametrize("method", ["random_over", "random_under"])
def test_actual_regression_sampling_is_fold_local(method):
    from linear_lab.preprocessing import build_preprocessor
    targets = np.concatenate([np.linspace(0, 1, 100), np.linspace(5, 15, 20)])
    X = pd.DataFrame({"x": targets})
    preprocessing, _, _ = build_preprocessor(X, {"scale": False})
    pipeline = RegressionPipeline([("preprocessing", preprocessing), ("model", Ridge(alpha=0))], resampling={"method": method, "focus": "high"})
    result = evaluate_cv(pipeline, X, targets, {"folds": 4}, ["rmse"], n_jobs=2)
    assert result["summary"]["rmse"]["mean"] < 1e-9
    for fold in result["folds"]:
        assert fold["validation_size"] == 30
        assert fold["resampling"]["before"] == 90
        assert fold["resampling"]["fitted_on"] == "train"
        if method == "random_over":
            assert fold["resampling"]["after"] > 90
        else:
            assert fold["resampling"]["after"] < 90


def test_search_with_disabled_normal_cv_gets_three_inner_folds(lab):
    data, registry, root = lab
    request = request_for(data, root)
    request["cv_config"] = {"strategy": "none"}
    result = SearchService(data, registry).run(request)
    assert result["search"]["folds"] == 3
    assert result["search"]["cv_config"]["strategy"] == "kfold"
    assert result["cv"] is None
