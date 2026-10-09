"""Numerical diagnostics must use visible data and leave the saved model intact."""
from copy import deepcopy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_squared_error, log_loss
from sklearn.preprocessing import LabelEncoder

from cml_lab.infrastructure.ml.catalogue import AlgorithmCatalogue
from cml_lab.infrastructure.ml.diagnostics import build_diagnostics
from cml_lab.infrastructure.ml.engine import ExperimentEngine
from cml_lab.infrastructure.ml.fitting import fit_artifact
from cml_lab.infrastructure.ml.metrics import TaskMetrics
from cml_lab.infrastructure.ml.splitting import holdout

FLAGS = {"regularization_path": True, "learning_curve": True, "permutation_importance": True}


@pytest.fixture
def example():
    rng = np.random.default_rng(24)
    X = pd.DataFrame({"signal": rng.normal(size=100), "noise": rng.normal(size=100)})
    y = 4 * X.signal.to_numpy() + .05 * rng.normal(size=len(X))
    spec = {"task": "regression", "algorithm_id": "ridge", "params": {"alpha": 1.},
            "dataset_id": "data", "target": "target", "features": list(X.columns),
            "metrics": ["mse"], "seed": 42, "n_jobs": 2, **FLAGS}
    return X, y, spec, AlgorithmCatalogue(), TaskMetrics()


def calculate(example, spec=None, **kwargs):
    X, y, default, catalogue, metrics = example
    spec = default if spec is None else spec
    artifact, _, _ = fit_artifact(spec, X.iloc[:70], y[:70], catalogue)
    output, warnings = build_diagnostics(spec, artifact, X.iloc[:70], y[:70], X.iloc[70:], y[70:],
                                          catalogue, metrics, **kwargs)
    return artifact, output, warnings


def test_regularization_points_equal_independent_complete_pipeline_fits(example):
    X, y, spec, catalogue, _ = example
    artifact, diagnostics, warnings = calculate(example)
    assert not warnings
    path = diagnostics["regularization_path"]
    assert len(path["alphas"]) == 9
    assert path["scope"]["test_used"] is False
    for position in [0, 4, 8]:
        alpha = path["alphas"][position]
        independent, _, _ = fit_artifact({**spec, "params": {"alpha": alpha}}, X.iloc[:70], y[:70], catalogue)
        expected = mean_squared_error(y[70:], independent.predict(X.iloc[70:]))
        assert path["points"][position]["validation_score"] == pytest.approx(expected)
        np.testing.assert_allclose(path["coefficients"][position], independent.estimator.coef_)
    assert np.linalg.norm(path["coefficients"][-1]) < np.linalg.norm(path["coefficients"][0])
    assert diagnostics["permutation_importance"]["mean"][0] > diagnostics["permutation_importance"]["mean"][1]
    assert diagnostics["permutation_importance"]["mean"][0] > 0


def test_learning_curve_uses_fresh_train_prefixes_and_frozen_validation(example):
    X, y, spec, catalogue, _ = example
    _, diagnostics, _ = calculate(example)
    curve = diagnostics["learning_curve"]
    assert len(curve["points"]) == 5
    order = np.random.default_rng(spec["seed"]).permutation(70)
    for point in curve["points"]:
        indices = order[:point["train_size"]]
        independent, _, _ = fit_artifact(spec, X.iloc[indices], y[indices], catalogue)
        assert point["train_score"] == pytest.approx(mean_squared_error(y[indices], independent.predict(X.iloc[indices])))
        assert point["validation_score"] == pytest.approx(mean_squared_error(y[70:], independent.predict(X.iloc[70:])))
    assert curve["points"][-1]["train_size"] == 70


def test_permutation_importance_never_fits_and_keeps_original_artifact(example, monkeypatch):
    import cml_lab.infrastructure.ml.diagnostics as module
    X, y, spec, catalogue, metrics = example
    spec = {**spec, "regularization_path": False, "learning_curve": False}
    artifact, _, _ = fit_artifact(spec, X.iloc[:70], y[:70], catalogue)
    expected = artifact.predict(X).copy()
    coefficients = artifact.estimator.coef_.copy()
    monkeypatch.setattr(module, "fit_artifact", lambda *args, **kwargs: pytest.fail("Permutation importance must not fit"))
    diagnostics, warnings = build_diagnostics(spec, artifact, X.iloc[:70], y[:70], X.iloc[70:], y[70:],
                                               catalogue, metrics)
    assert not warnings
    assert diagnostics["permutation_importance"]["metric"] == "mse"
    assert diagnostics["permutation_importance"]["direction"] == "min"
    assert diagnostics["permutation_importance"]["mean"][0] > 0
    np.testing.assert_array_equal(artifact.predict(X), expected)
    np.testing.assert_array_equal(artifact.estimator.coef_, coefficients)


def test_classification_probability_metric_uses_correct_quality_direction():
    rng = np.random.default_rng(8)
    X = pd.DataFrame({"signal": rng.normal(size=120), "noise": rng.normal(size=120)})
    y = (X.signal.to_numpy() > 0).astype(int)
    spec = {"task": "classification", "algorithm_id": "sgd_classifier",
            "params": {"loss": "log_loss"}, "epochs": 4, "metrics": ["log_loss"], "seed": 42, **FLAGS}
    catalogue, metrics = AlgorithmCatalogue(), TaskMetrics()
    artifact, _, _ = fit_artifact(spec, X.iloc[:80], y[:80], catalogue,
                                  label_encoder=LabelEncoder().fit(["no", "yes"]))
    diagnostics, warnings = build_diagnostics(spec, artifact, X.iloc[:80], y[:80], X.iloc[80:], y[80:], catalogue, metrics)
    assert not warnings
    assert diagnostics["regularization_path"]["metric"] == "log_loss"
    assert diagnostics["permutation_importance"]["direction"] == "min"
    assert diagnostics["permutation_importance"]["mean"][0] > 0
    assert all(point["validation_score"] is not None for point in diagnostics["learning_curve"]["points"])
    alpha = diagnostics["regularization_path"]["alphas"][4]
    independent, _, _ = fit_artifact({**spec, "params": {**spec["params"], "alpha": alpha}},
                                      X.iloc[:80], y[:80], catalogue, label_encoder=artifact.label_encoder)
    assert diagnostics["regularization_path"]["points"][4]["validation_score"] == pytest.approx(
        log_loss(y[80:], independent.predict_proba(X.iloc[80:]), labels=[0, 1]))


def test_changing_test_data_cannot_change_diagnostics_or_saved_model(example, tmp_path):
    X, y, spec, catalogue, _ = example
    frame = X.assign(target=y)
    parts = holdout(spec, len(X), y)
    class Data:
        def __init__(self, frame): self.table = frame
        def frame(self, dataset_id): return self.table.copy()
    paths = [tmp_path/"first.joblib", tmp_path/"second.joblib"]
    baseline = ExperimentEngine(Data(frame), catalogue).run({**spec, "artifact_path": str(paths[0])})
    changed = frame.copy()
    changed.iloc[parts["test"], :2] = 1e9
    changed.loc[parts["test"], "target"] = -1e9
    second = ExperimentEngine(Data(changed), catalogue).run({**spec, "artifact_path": str(paths[1])})
    for name in FLAGS:
        assert baseline["diagnostics"][name] == second["diagnostics"][name]
    np.testing.assert_array_equal(joblib.load(paths[0]).predict(X), joblib.load(paths[1]).predict(X))
    no_diagnostics = {**spec, **{name: False for name in FLAGS}, "artifact_path": str(tmp_path/"without.joblib")}
    ExperimentEngine(Data(frame), catalogue).run(no_diagnostics)
    np.testing.assert_array_equal(joblib.load(paths[0]).predict(X), joblib.load(tmp_path/"without.joblib").predict(X))


@pytest.mark.parametrize("task,groups", [("forecasting", None), ("panel", None), ("ranking", None),
                                        ("clustering", None), ("regression", np.repeat(np.arange(10), 10))])
def test_unsupported_context_flags_get_explicit_reason_and_warnings(example, task, groups):
    X, y, spec, catalogue, metrics = example
    diagnostics, warnings = build_diagnostics({**spec, "task": task}, None, X, y, X, y, catalogue, metrics, groups=groups)
    assert len(warnings) == 3
    assert set(diagnostics) == set(FLAGS)
    assert all(record["reason"] for record in diagnostics.values())


def test_regularization_flag_on_tree_returns_reason(example):
    _, _, spec, _, _ = example
    _, diagnostics, warnings = calculate(example, {**spec, "algorithm_id": "decision_tree_regressor", "params": {},
                                                    "learning_curve": False, "permutation_importance": False})
    assert "alpha" in diagnostics["regularization_path"]["reason"]
    assert warnings


def test_diagnostics_cancel_before_fit(example):
    X, y, spec, catalogue, metrics = example
    with pytest.raises(InterruptedError, match="отменена"):
        build_diagnostics(spec, None, X, y, X, y, catalogue, metrics, cancelled=lambda: True)
