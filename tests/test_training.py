"""Numerical behavior checks using known truth and independent estimators."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from linear_lab.datasets import DataService
from linear_lab.models import ModelRegistry
from linear_lab.metrics import MetricRegistry
from linear_lab.training import TrainingCancelled, TrainingService


@pytest.fixture
def laboratory(tmp_path: Path):
    data = DataService(tmp_path / "data")
    return data, TrainingService(data, ModelRegistry()), tmp_path


def load_table(data: DataService, X: np.ndarray, y: np.ndarray) -> str:
    columns = [f"x{i + 1}" for i in range(X.shape[1])]
    rows = [
        dict(zip(columns, row, strict=True), y=float(target))
        for row, target in zip(X.tolist(), y, strict=True)
    ]
    metadata = data.load({"kind": "custom", "rows": rows, "target": "y"})
    return metadata["id"]


def fit(laboratory, X, y, model="ols", params=None, **changes):
    data, trainer, root = laboratory
    dataset_id = load_table(data, X, y)
    request = {
        "dataset_id": dataset_id,
        "target": "y",
        "features": [f"x{i + 1}" for i in range(X.shape[1])],
        "model": model,
        "params": params or {},
        "seed": 17,
        "split": {"train": 0.6, "validation": 0.2, "test": 0.2, "shuffle": True},
        "preprocessing": {"scale": False, "degree": 1, "impute": True},
        "metrics": ["mse", "rmse", "mae", "r2"],
        "epochs": 80,
        "cv": 0,
        "regularization_path": False,
        "artifact_path": str(root / f"{model}.joblib"),
    }
    request.update(changes)
    events = []
    result = trainer.run(request, events.append, lambda: False)
    return result, events, request


def test_recovers_known_linear_function_and_exports_json_safe_result(laboratory):
    rng = np.random.default_rng(73)
    X = rng.normal(size=(150, 2))
    y = 5 + X @ np.array([3.0, -2.0])

    result, events, request = fit(laboratory, X, y)

    np.testing.assert_allclose(result["coefficients"], [3, -2], atol=1e-9)
    assert result["intercept"] == pytest.approx(5, abs=1e-9)
    for partition in ("train", "validation", "test"):
        assert result["metrics"][partition]["mse"] < 1e-20
        assert result["metrics"][partition]["r2"] == pytest.approx(1)
    assert Path(request["artifact_path"]).is_file()
    assert events and all(0 <= event["progress"] <= 1 for event in events)
    json.dumps(result, allow_nan=False)


def test_split_is_disjoint_complete_and_scaling_uses_training_only(laboratory):
    rng = np.random.default_rng(2)
    X = rng.normal(size=(100, 2))
    X[-20:, 0] += 30
    X[-10:, 1] *= 25
    y = 2 + X @ np.array([3.0, -0.8]) + rng.normal(size=100)

    result, _, _ = fit(
        laboratory,
        X,
        y,
        model="ridge",
        params={"alpha": 8.0},
        preprocessing={"scale": True, "degree": 1, "impute": True},
    )

    predictions = result["predictions"]
    indices = np.asarray(predictions["indices"], dtype=int)
    partitions = np.asarray(predictions["split"])
    assert len(indices) == len(X)
    assert len(set(indices)) == len(X)
    assert set(indices) == set(range(len(X)))
    assert {name: int(np.sum(partitions == name)) for name in ("train", "validation", "test")} == {
        "train": 60, "validation": 20, "test": 20,
    }
    np.testing.assert_allclose(predictions["actual"], y[indices])

    train_indices = indices[partitions == "train"]
    reference = make_pipeline(StandardScaler(), Ridge(alpha=8.0))
    reference.fit(X[train_indices], y[train_indices])
    np.testing.assert_allclose(
        predictions["predicted"], reference.predict(X[indices]), rtol=1e-8, atol=1e-8
    )


def test_ridge_shrinks_coefficients_and_lasso_removes_nuisance_features(laboratory):
    rng = np.random.default_rng(91)
    X = rng.normal(size=(180, 6))
    y = 1 + 3 * X[:, 0] + rng.normal(scale=0.03, size=len(X))

    ordinary, _, _ = fit(laboratory, X, y)
    ridge, _, _ = fit(laboratory, X, y, model="ridge", params={"alpha": 100.0})
    lasso, _, _ = fit(laboratory, X, y, model="lasso", params={"alpha": 0.1})

    assert np.linalg.norm(ridge["coefficients"]) < np.linalg.norm(ordinary["coefficients"])
    assert lasso["coefficients"][0] > 2.5
    np.testing.assert_allclose(lasso["coefficients"][1:], np.zeros(5), atol=1e-10)


def test_holdout_answers_do_not_change_fit_regularization_path_or_cross_validation(laboratory):
    rng = np.random.default_rng(33)
    X = rng.normal(size=(100, 2))
    y = 2 + X @ [1.5, -0.4] + rng.normal(scale=0.03, size=100)
    settings = {
        "split": {"train": 0.6, "validation": 0.2, "test": 0.2, "shuffle": False},
        "preprocessing": {"scale": True, "degree": 1, "impute": True},
        "regularization_path": True,
        "cv": 3,
    }
    original, _, _ = fit(laboratory, X, y, "ridge", {"alpha": 2.0}, **settings)
    altered_y = y.copy()
    altered_y[60:] += 10000
    altered, _, _ = fit(laboratory, X, altered_y, "ridge", {"alpha": 2.0}, **settings)

    np.testing.assert_allclose(original["coefficients"], altered["coefficients"], atol=1e-12)
    assert original["intercept"] == pytest.approx(altered["intercept"], abs=1e-12)
    assert original["regularization_path"] == altered["regularization_path"]
    # Fold durations describe real work and naturally differ between runs. All
    # scores, selected rows and sampling diagnostics must remain identical.
    def without_fold_durations(value):
        return {
            **value,
            "folds": [{key: item for key, item in fold.items() if key != "seconds"}
                      for fold in value["folds"]],
        }

    assert without_fold_durations(original["cv"]) == without_fold_durations(altered["cv"])
    assert altered["metrics"]["test"]["mse"] > original["metrics"]["test"]["mse"] + 1e7


def test_permutation_importance_identifies_signal_and_ignores_known_nuisance(laboratory):
    rng = np.random.default_rng(62)
    X = rng.normal(size=(180, 2))
    y = 3 + 4 * X[:, 0]
    result, _, _ = fit(laboratory, X, y, permutation_importance=True)

    importance = result["diagnostics"]["permutation_importance"]
    assert importance["fitted_on"] == "validation"
    assert importance["feature_names"] == ["x1", "x2"]
    assert importance["mean"][0] > 5
    assert abs(importance["mean"][1]) < 1e-10


def test_sgd_records_actual_epochs_and_learns_known_relation(laboratory):
    rng = np.random.default_rng(11)
    X = rng.uniform(-1, 1, size=(120, 2))
    y = 1 + 2 * X[:, 0] - X[:, 1]

    result, events, _ = fit(
        laboratory,
        X,
        y,
        model="sgd",
        params={"eta0": 0.02, "learning_rate": "constant", "alpha": 0.0001},
        epochs=50,
    )

    trace = result["trace"]
    assert result["trace_kind"] == "epochs"
    assert len(trace) >= 10
    assert len({frame["step"] for frame in trace}) == len(trace)
    assert all(np.isfinite(frame["train_loss"]) for frame in trace)
    assert trace[-1]["train_loss"] < trace[0]["train_loss"] * 0.05
    assert result["metrics"]["test"]["mse"] < 0.01
    assert any("frame" in event for event in events)


@pytest.mark.parametrize("model", ["lars", "omp"])
def test_active_set_frames_match_their_actual_linear_predictions(laboratory, model):
    rng = np.random.default_rng(47)
    X = rng.normal(size=(120, 3))
    y = 4 + X @ [2.0, -0.8, 0.0]
    result, _, _ = fit(laboratory, X, y, model, {"n_nonzero_coefs": 2})

    assert result["trace_kind"] == "active_set"
    assert len(result["trace"]) >= 2
    train = np.asarray(result["split"]["indices"]["train"], dtype=int)
    shown = np.asarray(result["predictions"]["indices"], dtype=int)
    for frame in result["trace"]:
        predicted = X @ np.asarray(frame["coef"]) + frame["intercept"]
        np.testing.assert_allclose(frame["predicted"], predicted[shown], atol=1e-9)
        assert frame["train_loss"] == pytest.approx(np.mean((y[train] - predicted[train]) ** 2), abs=1e-12)
    assert result["metrics"]["test"]["mse"] < 1e-20


@pytest.mark.parametrize("model", ["huber", "ransac", "theilsen"])
def test_robust_models_recover_clean_relation_despite_large_response_outliers(laboratory, model):
    rng = np.random.default_rng(19)
    X = rng.uniform(-2, 2, size=(180, 1))
    y = 1 + 2 * X[:, 0] + rng.normal(scale=0.02, size=180)
    y[:18] += 40
    ordinary, _, _ = fit(laboratory, X, y)
    robust, _, _ = fit(laboratory, X, y, model)

    clean_x = np.linspace(-2, 2, 50)
    truth = 1 + 2 * clean_x
    ordinary_error = np.mean(np.abs(truth - (ordinary["intercept"] + ordinary["coefficients"][0] * clean_x)))
    robust_error = np.mean(np.abs(truth - (robust["intercept"] + robust["coefficients"][0] * clean_x)))
    assert robust_error < 0.15
    assert robust_error < ordinary_error * 0.1


@pytest.mark.parametrize("model", ["gamma", "poisson"])
def test_positive_target_model_rejects_invalid_data_explicitly(laboratory, model):
    X = np.arange(50, dtype=float).reshape(-1, 1)
    y = np.full(50, -1.0)
    with pytest.raises(ValueError, match="(?i)(target|positive|positiv|negative|y|цел|полож|отриц)"):
        fit(laboratory, X, y, model=model)


def test_immediate_cancellation_does_not_create_model_artifact(laboratory):
    data, trainer, root = laboratory
    X = np.arange(40, dtype=float).reshape(-1, 1)
    dataset_id = load_table(data, X, X[:, 0])
    artifact = root / "cancelled.joblib"
    request = {"dataset_id": dataset_id, "model": "ridge", "artifact_path": str(artifact)}

    with pytest.raises(TrainingCancelled):
        trainer.run(request, lambda event: None, lambda: True)

    assert not artifact.exists()


def test_cancellation_during_sgd_stops_before_all_epochs_and_saves_no_artifact(laboratory):
    data, trainer, root = laboratory
    rng = np.random.default_rng(8)
    X = rng.normal(size=(80, 2))
    dataset_id = load_table(data, X, 1 + X @ [2, -1])
    artifact = root / "partially-trained.joblib"
    frames = []

    def record_and_cancel(event):
        if "frame" in event:
            frames.append(event["frame"])

    with pytest.raises(TrainingCancelled):
        trainer.run(
            {"dataset_id": dataset_id, "model": "sgd", "epochs": 20, "artifact_path": str(artifact)},
            record_and_cancel,
            lambda: bool(frames),
        )

    assert 0 < len(frames) < 20
    assert not artifact.exists()


def test_registry_rejects_unknown_and_invalid_model_parameters():
    registry = ModelRegistry()
    with pytest.raises(ValueError):
        registry.create("ridge", {"not_an_estimator_parameter": 5}, 42)
    with pytest.raises(ValueError):
        registry.create("ridge", {"alpha": -1}, 42)


def test_all_metrics_report_invalid_domains_as_null_with_explanation():
    values, details = MetricRegistry().evaluate([0, -2, 3], [0, 1, -1], selection="all")

    assert values["mse"] == pytest.approx(25 / 3)
    for metric in ("mape", "msle", "rmsle", "poisson_deviance", "gamma_deviance"):
        assert values[metric] is None
        assert details[metric]["reason"]
    json.dumps({"values": values, "details": details}, allow_nan=False)


def test_constant_target_has_no_fabricated_r_squared():
    values, details = MetricRegistry().evaluate([2, 2, 2], [2, 2, 2], ["mse", "r2"])
    assert values["mse"] == 0
    assert values["r2"] is None
    assert details["r2"]["reason"]


def test_tweedie_power_zero_accepts_real_targets_and_matches_mse():
    actual, predicted = [-2, 0, 3], [-1, -1, 2]
    values, details = MetricRegistry().evaluate(
        actual, predicted, ["tweedie_deviance"], params={"power": 0}
    )
    assert values["tweedie_deviance"] == pytest.approx(1)
    assert details["tweedie_deviance"]["reason"] is None
