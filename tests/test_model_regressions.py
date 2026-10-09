"""Regression-model behavior after the expanded preprocessing pipeline.

These checks use real saved estimators. Numerical reference fits use sklearn
directly, and small domain-correct datasets keep every supported solver useful.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from threadpoolctl import threadpool_limits

from linear_lab.datasets import DataService
from linear_lab.models import ModelRegistry
from linear_lab.training import TrainingService


MODEL_IDS = (
    "ols", "ridge", "lasso", "elasticnet", "lars", "lassolars", "omp", "sgd",
    "ransac", "theilsen", "huber", "bayesian_ridge", "ard", "quantile",
    "poisson", "gamma", "tweedie", "linear_svr", "nonnegative", "weighted_lasso",
    "adaptive_lasso", "group_lasso", "sparse_group_lasso", "fused_lasso",
    "tikhonov", "l0", "scad", "mcp", "sqrt_lasso",
)

NATIVE_CLASSES = {
    "ols": "LinearRegression", "ridge": "Ridge", "lasso": "Lasso",
    "elasticnet": "ElasticNet", "lars": "Lars", "lassolars": "LassoLars",
    "omp": "OrthogonalMatchingPursuit", "sgd": "SGDRegressor",
    "ransac": "RANSACRegressor", "theilsen": "TheilSenRegressor",
    "huber": "HuberRegressor", "bayesian_ridge": "BayesianRidge",
    "ard": "ARDRegression", "quantile": "QuantileRegressor",
    "poisson": "PoissonRegressor", "gamma": "GammaRegressor",
    "tweedie": "TweedieRegressor", "linear_svr": "LinearSVR",
    "nonnegative": "LinearRegression", "weighted_lasso": "WeightedLassoRegressor",
    "adaptive_lasso": "AdaptiveLassoRegressor", "group_lasso": "GroupLassoRegressor",
    "sparse_group_lasso": "SparseGroupLassoRegressor", "fused_lasso": "FusedLassoRegressor",
    "tikhonov": "TikhonovRegressor", "l0": "BestSubsetRegressor",
    "scad": "SCADRegressor", "mcp": "MCPRegression", "sqrt_lasso": "SqrtLasso",
}


@pytest.fixture
def laboratory(tmp_path: Path):
    data = DataService(tmp_path / "data")
    registry = ModelRegistry()
    with threadpool_limits(limits=1):
        yield data, registry, TrainingService(data, registry), tmp_path


def request_for(laboratory, X, y, model, **changes):
    data, _, _, root = laboratory
    names = [f"x{index + 1}" for index in range(X.shape[1])]
    rows = [dict(zip(names, row, strict=True), y=float(target))
            for row, target in zip(X.tolist(), y, strict=True)]
    identifier = data.load({"kind": "custom", "rows": rows, "target": "y"})["id"]
    request = {
        "dataset_id": identifier, "target": "y", "features": names,
        "model": model, "params": {}, "seed": 17,
        "split": {"train": .6, "validation": .2, "test": .2, "shuffle": True},
        "preprocessing": {}, "metrics": ["mse", "rmse", "r2", "custom"],
        "custom_metric": "sqrt(mean(error**2))", "epochs": 50,
        "cv": 0, "regularization_path": False,
        "artifact_path": str(root / f"{identifier}.joblib"),
    }
    request.update(changes)
    return request


@pytest.mark.parametrize("model", MODEL_IDS)
def test_each_model_retains_real_predictions_and_saved_inference_after_feature_expansion(laboratory, model):
    """A trained model must be useful, serializable, and numerically honest.

    Both old defaults and degree-two engineering are exercised. In particular,
    renamed estimators, a dummy constant fit, post-fit transform changes, and
    confusing a GLM's linear predictor with its target prediction fail here.
    """
    data, registry, trainer, _ = laboratory
    assert {item["id"] for item in registry.catalogue()} == set(MODEL_IDS)
    rng = np.random.default_rng(184)
    X = rng.uniform(-1, 1, size=(80, 3))
    if model == "poisson":
        y = np.round(np.exp(2 + .35 * X[:, 0] + .15 * X[:, 1]))
    elif model in {"gamma", "tweedie"}:
        y = np.exp(1 + .6 * X[:, 0] + .25 * X[:, 1])
    else:
        y = 8 + X @ [3.0, 1.5, .2] + rng.normal(scale=.03, size=len(X))

    for preprocessing in ({}, {"degree": 2, "scaler": "standard", "imputation": "mean"}):
        request = request_for(laboratory, X, y, model, preprocessing=preprocessing)
        result = trainer.run(request)
        saved = joblib.load(request["artifact_path"])
        bundle = data.resolve(request["dataset_id"], "y", request["features"])
        prediction = saved.predict(bundle.X)
        assert result["model"] == model
        assert saved.named_steps["model"].__class__.__name__ == NATIVE_CLASSES[model]
        assert np.isfinite(prediction).all()
        assert np.std(prediction) > 1e-6, "The actual signal must not become a dummy constant model."
        if model in {"poisson", "gamma", "tweedie"}:
            assert np.all(prediction > 0)
        if model == "nonnegative":
            assert np.all(np.asarray(result["coefficients"]) >= -1e-12)
        shown = np.asarray(result["predictions"]["indices"], dtype=int)
        np.testing.assert_allclose(result["predictions"]["predicted"], prediction[shown], rtol=1e-12, atol=1e-12)
        assert len(result["coefficients"]) == len(result["feature_names"])
        assert saved.named_steps["preprocessing"].get_feature_names_out().tolist() == result["feature_names"]
        for split, indices in result["split"]["indices"].items():
            mse = np.mean((y[indices] - prediction[indices]) ** 2)
            assert result["metrics"][split]["mse"] == pytest.approx(mse, rel=1e-11, abs=1e-11)
            assert result["metrics"][split]["custom"] == pytest.approx(np.sqrt(mse), rel=1e-11, abs=1e-11)
        last = result["trace"][-1]
        np.testing.assert_allclose(last["predicted"], prediction[shown], rtol=1e-9, atol=1e-9)
        if model in {"scad", "mcp"}:
            # Nonconvex solvers can find a local solution; a finite stated
            # objective is still required, without promising a global minimum.
            assert last["objective"] is not None and np.isfinite(last["objective"])
        if last.get("grid") is not None:
            np.testing.assert_allclose(last["grid"]["z"], result["prediction_grid"]["z"], rtol=1e-9, atol=1e-9)
        json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("model,params,reference", [
    ("ols", {}, lambda: LinearRegression()),
    ("ridge", {"alpha": 4.0}, lambda: Ridge(alpha=4.0)),
    ("lasso", {"alpha": .08, "tol": 1e-10}, lambda: Lasso(alpha=.08, tol=1e-10)),
    ("elasticnet", {"alpha": .12, "l1_ratio": .35, "tol": 1e-10},
     lambda: ElasticNet(alpha=.12, l1_ratio=.35, tol=1e-10)),
])
def test_native_objectives_keep_sklearn_semantics_on_engineered_features(laboratory, model, params, reference):
    data, _, trainer, _ = laboratory
    rng = np.random.default_rng(278)
    X = rng.normal(size=(120, 2))
    # Known polynomial truth checks actual engineering, rather than relying on
    # prediction parity between two instances of our own implementation.
    y = 3 + 2 * X[:, 0] - X[:, 1] + .7 * X[:, 0] ** 2 + .3 * X[:, 0] * X[:, 1]
    request = request_for(laboratory, X, y, model, params=params,
                          preprocessing={"degree": 2, "scaler": "standard", "imputation": "median"})
    result = trainer.run(request)
    saved = joblib.load(request["artifact_path"])
    train = np.asarray(result["split"]["indices"]["train"], dtype=int)
    independent = make_pipeline(PolynomialFeatures(degree=2, include_bias=False), StandardScaler(), reference())
    independent.fit(X[train], y[train])
    expected = independent.predict(X)
    bundle = data.resolve(request["dataset_id"], "y", request["features"])
    np.testing.assert_allclose(saved.predict(bundle.X), expected, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(result["coefficients"], independent.steps[-1][1].coef_, rtol=1e-9, atol=1e-9)
    if model == "ols":
        np.testing.assert_allclose(expected, y, rtol=1e-12, atol=1e-12)
    beta = independent.steps[-1][1].coef_
    residual = y[train] - expected[train]
    if model == "ols":
        objective = residual @ residual
    elif model == "ridge":
        objective = residual @ residual + params["alpha"] * (beta @ beta)
    else:
        ratio = 1 if model == "lasso" else params["l1_ratio"]
        objective = np.mean(residual ** 2) / 2 + params["alpha"] * (ratio * np.abs(beta).sum() + (1 - ratio) * (beta @ beta) / 2)
    assert result["trace"][-1]["objective"] == pytest.approx(objective, rel=1e-9, abs=1e-9)


@pytest.mark.parametrize("model", ["huber", "ransac", "theilsen"])
def test_robust_response_models_still_reject_outliers_with_quadratic_engineering(laboratory, model):
    _, _, trainer, _ = laboratory
    rng = np.random.default_rng(933)
    X = rng.uniform(-2, 2, size=(150, 1))
    y = 2 + 3 * X[:, 0] + .4 * X[:, 0] ** 2 + rng.normal(scale=.015, size=len(X))
    y[:15] += 40
    configuration = {"degree": 2, "scaler": "robust"}
    ordinary_request = request_for(laboratory, X, y, "ols", preprocessing=configuration)
    robust_request = request_for(laboratory, X, y, model, preprocessing=configuration)
    trainer.run(ordinary_request)
    trainer.run(robust_request)
    ordinary = joblib.load(ordinary_request["artifact_path"])
    robust = joblib.load(robust_request["artifact_path"])
    clean_x = np.linspace(-1, 1, 51)
    clean = pd.DataFrame({"x1": clean_x})
    truth = 2 + 3 * clean_x + .4 * clean_x ** 2
    ordinary_error = np.mean(np.abs(ordinary.predict(clean) - truth))
    robust_error = np.mean(np.abs(robust.predict(clean) - truth))
    assert robust_error < .15
    assert robust_error < .15 * ordinary_error


@pytest.mark.parametrize("model,width,config,params,explanation", [
    ("l0", 4, {"degree": 2}, {}, "12"),
    ("theilsen", 8, {"degree": 2}, {}, "30"),
    ("ols", 60, {"degree": 3}, {}, "2000"),
    ("omp", 3, {}, {"n_nonzero_coefs": 4}, "OMP"),
])
def test_model_dimension_limits_are_explained_and_never_export_an_untrained_model(laboratory, model, width, config, params, explanation):
    _, _, trainer, _ = laboratory
    rng = np.random.default_rng(172)
    X = rng.normal(size=(30, width))
    request = request_for(laboratory, X, X[:, 0] + 1, model,
                          preprocessing=config, params=params)
    with pytest.raises(ValueError, match=explanation):
        trainer.run(request)
    artifact = Path(request["artifact_path"])
    assert not artifact.exists()
    assert not artifact.with_name(artifact.name + ".tmp").exists()
