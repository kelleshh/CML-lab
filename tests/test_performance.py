"""Parallel execution preserves the fitted experiment and numerical results."""

from __future__ import annotations

import multiprocessing as mp

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from linear_lab.models import ModelRegistry
from linear_lab.training import TrainingCancelled, TrainingService


def _samples():
    rng = np.random.default_rng(417)
    X = rng.normal(size=(180, 5))
    X[:, 4] = .8 * X[:, 0] + .2 * X[:, 4]
    y = X @ np.array([3., -2., .4, 0., 1.]) + rng.normal(scale=.1, size=len(X))
    return X, y


def _path(trainer, estimator, model_id, X, y, n_jobs, progress=lambda event: None, cancelled=lambda: False):
    notes = []
    result = trainer._regularization_path(
        {"regularization_path": True, "n_jobs": n_jobs}, model_id, estimator,
        X, y, [f"x{i}" for i in range(X.shape[1])], progress, cancelled, notes,
    )
    return result, notes


@pytest.mark.parametrize("model_id,estimator", [("ridge", Ridge(alpha=2)), ("lasso", Lasso(alpha=.03, tol=1e-10, max_iter=10000))])
def test_parallel_alpha_path_matches_serial_and_keeps_fitted_model(model_id, estimator):
    X, y = _samples()
    estimator.fit(X, y)
    coefficients = estimator.coef_.copy()
    original_params = estimator.get_params().copy()
    trainer = TrainingService(None, ModelRegistry())
    serial, serial_notes = _path(trainer, estimator, model_id, X, y, 1)
    events = []
    parallel, parallel_notes = _path(trainer, estimator, model_id, X, y, 2, events.append)

    assert parallel["n_jobs"] == 2
    assert len(parallel["alphas"]) == len(events) == 18
    assert np.all(np.diff(parallel["alphas"]) > 0)
    np.testing.assert_array_equal(parallel["alphas"], serial["alphas"])
    np.testing.assert_allclose(parallel["coefficients"], serial["coefficients"], rtol=1e-12, atol=1e-12)
    assert not serial_notes and not parallel_notes
    assert events[-1]["progress"] == pytest.approx(.83)
    np.testing.assert_array_equal(estimator.coef_, coefficients)
    assert estimator.get_params() == original_params


@pytest.mark.parametrize("n_jobs", [0, -1, 5, True, "2"])
def test_alpha_path_rejects_unbounded_worker_counts(n_jobs):
    X, y = _samples()
    with pytest.raises(ValueError, match="целое число от 1 до 4"):
        _path(TrainingService(None, ModelRegistry()), Ridge(), "ridge", X, y, n_jobs)


def test_parallel_alpha_path_stops_after_cancelled_progress():
    X, y = _samples()
    events = []
    def progress(event):
        events.append(event)
    with pytest.raises(TrainingCancelled):
        _path(TrainingService(None, ModelRegistry()), Ridge(), "ridge", X, y, 2, progress, lambda: len(events) >= 4)
    assert 0 < len(events) <= 4


def test_parallel_importance_matches_serial_without_changing_pipeline():
    X, y = _samples()
    frame = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    pipeline = make_pipeline(StandardScaler(), LinearRegression(n_jobs=3)).fit(frame.iloc[:120], y[:120])
    before = pipeline.predict(frame.iloc[120:])
    trainer = TrainingService(None, ModelRegistry())
    def importance(n_jobs):
        return trainer._importance({"permutation_importance": True, "n_jobs": n_jobs, "seed": 417}, frame.iloc[120:], y[120:], pipeline, ["rmse"], None, {}, lambda: False)
    serial, parallel = importance(1), importance(2)
    assert parallel["n_jobs"] == 2 and parallel["fitted_on"] == "validation"
    np.testing.assert_allclose(parallel["mean"], serial["mean"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(parallel["std"], serial["std"], rtol=1e-12, atol=1e-12)
    assert pipeline.named_steps["linearregression"].n_jobs == 3
    np.testing.assert_array_equal(pipeline.predict(frame.iloc[120:]), before)


def test_importance_checks_cancellation_between_real_scores():
    X, y = _samples()
    frame = pd.DataFrame(X)
    pipeline = make_pipeline(StandardScaler(), Ridge()).fit(frame, y)
    checks = 0
    def cancelled():
        nonlocal checks
        checks += 1
        return checks >= 3
    with pytest.raises(TrainingCancelled):
        TrainingService(None, ModelRegistry())._importance(
            {"permutation_importance": True, "n_jobs": 2}, frame, y,
            pipeline, ["mse"], None, {}, cancelled,
        )


def _daemon_path(queue):
    try:
        X, y = _samples()
        trainer = TrainingService(None, ModelRegistry())
        result, notes = _path(trainer, Ridge(), "ridge", X, y, 2)
        frame = pd.DataFrame(X)
        pipeline = make_pipeline(StandardScaler(), Ridge()).fit(frame, y)
        importance = trainer._importance({"permutation_importance": True, "n_jobs": 2}, frame, y, pipeline, ["mse"], None, {}, lambda: False)
        queue.put((len(result["alphas"]), notes, len(importance["mean"])))
    except BaseException as exc:
        queue.put((type(exc).__name__, str(exc)))


def test_parallel_path_runs_inside_daemon_worker_without_nested_processes():
    context = mp.get_context("spawn")
    queue = context.Queue()
    worker = context.Process(target=_daemon_path, args=(queue,), daemon=True)
    worker.start()
    try:
        result = queue.get(timeout=30)
        worker.join(timeout=10)
        assert not worker.is_alive()
        assert worker.exitcode == 0
        assert result == (18, [], 5)
    finally:
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=5)
        queue.close()
        queue.join_thread()
