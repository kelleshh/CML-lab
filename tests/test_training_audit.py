"""Independent numeric checks across fitting, preview, CV and saved inference.

Expected values come from the documented split, sklearn reference estimators,
and direct loss formulas. No application worker, sampler or estimator is mocked.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.base import clone
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from linear_lab.app import create_app
from linear_lab.datasets import DataService
from linear_lab.models import ModelRegistry
from linear_lab.preprocessing import ResamplingService
from linear_lab.training import TrainingService


@pytest.fixture
def lab(tmp_path):
    data = DataService(tmp_path / "data")
    return data, TrainingService(data, ModelRegistry()), tmp_path


def _request(data, root, rows, **changes):
    identifier = data.load({"kind": "custom", "rows": rows, "target": "y"})["id"]
    result = {
        "dataset_id": identifier, "target": "y",
        "features": [key for key in rows[0] if key != "y"],
        "model": "ridge", "params": {"alpha": 3.0}, "seed": 17,
        "split": {"train": .6, "validation": .2, "test": .2, "shuffle": True},
        "preprocessing": {"scale": True, "impute": True, "degree": 1},
        "metrics": ["mse", "mae"], "cv": 0, "regularization_path": False,
        "artifact_path": str(root / f"{identifier}.joblib"),
    }
    result.update(changes)
    return result


def _rare_rows():
    rng = np.random.default_rng(21)
    X = rng.normal(size=(180, 3))
    y = np.r_[np.linspace(0, 1, 150), np.linspace(10, 25, 30)]
    rows = [{"x": float(a), "z": float(b), "w": float(c), "y": float(target)}
            for (a, b, c), target in zip(X, y, strict=True)]
    return rows, X, y


def _stable_cv(value):
    return {**value, "folds": [{key: item for key, item in fold.items() if key != "seconds"}
                               for fold in value["folds"]]}


@pytest.mark.parametrize("method", ["random_over", "random_under", "smoter", "smogn"])
def test_sampled_fit_matches_reference_while_original_train_and_holdout_stay_separate(lab, method):
    data, trainer, root = lab
    rows, original_X, original_y = _rare_rows()
    sampling = {"method": method, "focus": "high", "neighbors": 3}
    request = _request(data, root, rows, resampling=sampling, cv=3)
    result = trainer.run(request)
    saved = joblib.load(request["artifact_path"])
    bundle = data.resolve(request["dataset_id"], "y", ["x", "z", "w"])
    X, y = bundle.X.to_numpy(), bundle.y
    np.testing.assert_allclose(X, original_X, atol=1e-14, rtol=1e-14)
    np.testing.assert_allclose(y, original_y, atol=1e-14, rtol=1e-14)

    # This independent sklearn split does not consult the application's result.
    train, held_out = train_test_split(np.arange(len(y)), train_size=.6, random_state=17)
    np.testing.assert_array_equal(result["split"]["indices"]["train"], train)
    scaler = StandardScaler().fit(X[train])
    sampled_X, sampled_y, summary = ResamplingService().fit_resample(
        scaler.transform(X[train]), y[train], sampling, seed=17,
    )
    reference = Ridge(alpha=3.0).fit(sampled_X, sampled_y)
    reference_prediction = reference.predict(scaler.transform(X))
    np.testing.assert_allclose(saved.predict(pd.DataFrame(X, columns=["x", "z", "w"])),
                               reference_prediction, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result["coefficients"], reference.coef_, rtol=1e-12, atol=1e-12)
    assert result["resampling"] == summary
    if method in {"random_over", "random_under"}:
        assert (summary["after"] > summary["before"]) == (method == "random_over")
    else:
        original_points = set(tuple(row) for row in np.c_[scaler.transform(X[train]), y[train]])
        assert any(tuple(row) not in original_points for row in np.c_[sampled_X, sampled_y])
    final = result["trace"][-1]
    original_mse = np.mean((y[train] - reference_prediction[train]) ** 2)
    fit_squared_error = (sampled_y - reference.predict(sampled_X)) ** 2
    assert final["train_loss"] == pytest.approx(original_mse, rel=1e-12)
    assert result["metrics"]["train"]["mse"] == pytest.approx(original_mse, rel=1e-12)
    assert final["fit_loss"] == pytest.approx(np.mean(fit_squared_error), rel=1e-12)
    assert final["objective"] == pytest.approx(
        np.sum(fit_squared_error) + 3 * np.sum(reference.coef_ ** 2), rel=1e-12,
    )
    assert abs(final["fit_loss"] - final["train_loss"]) > .1

    # Relevance curves, samplers, scalers, estimators and CV must never learn
    # from either of the two outer held-out partitions, even very extreme y.
    changed = deepcopy(rows)
    for index in held_out:
        changed[index]["y"] += 1e8
    other_request = _request(data, root, changed, resampling=sampling, cv=3)
    other = trainer.run(other_request)
    other_saved = joblib.load(other_request["artifact_path"])
    np.testing.assert_array_equal(saved.named_steps["preprocessing"].named_transformers_["numeric"].named_steps["scale"].mean_,
                                  other_saved.named_steps["preprocessing"].named_transformers_["numeric"].named_steps["scale"].mean_)
    np.testing.assert_allclose(result["coefficients"], other["coefficients"], rtol=1e-12, atol=1e-12)
    assert final["objective"] == pytest.approx(other["trace"][-1]["objective"], rel=1e-12, abs=1e-12)
    assert final["fit_loss"] == pytest.approx(other["trace"][-1]["fit_loss"], rel=1e-12, abs=1e-12)
    assert result["resampling"] == other["resampling"]
    assert _stable_cv(result["cv"]) == _stable_cv(other["cv"])
    assert other["metrics"]["test"]["mse"] > 1e15


@pytest.mark.parametrize("time_setting", ["ordered_split", "timeseries_cv"])
def test_resampling_refuses_ordered_observations_instead_of_mixing_the_past_and_future(lab, time_setting):
    data, trainer, root = lab
    rows, _, _ = _rare_rows()
    request = _request(data, root, rows, resampling={"method": "random_over"})
    if time_setting == "ordered_split":
        request["split"]["shuffle"] = False
    else:
        request["cv_config"] = {"strategy": "timeseries", "folds": 3}
    with pytest.raises(ValueError, match="хронологию"):
        trainer.run(request)
    assert not Path(request["artifact_path"]).exists()


def test_smoter_interpolation_preserves_signed_affine_feature_relationships():
    # Any interpolation between (a, -a) and (b, -b) stays on z=-x. These are
    # ordinary signed, standardized coordinates, not a nonnegative domain.
    y = np.r_[np.linspace(0, 1, 100), np.linspace(10, 25, 20)]
    scaler = StandardScaler().fit(np.c_[y - 5, 5 - y])
    X = scaler.transform(np.c_[y - 5, 5 - y])
    sampled_X, sampled_y, summary = ResamplingService().fit_resample(
        X, y, {"method": "smoter", "focus": "high", "neighbors": 3}, seed=8,
    )
    assert summary["synthetic"] is True
    np.testing.assert_allclose(sampled_X[:, 0] + sampled_X[:, 1], 0, atol=1e-12)
    assert np.any(sampled_X[:, 1] < -1)
    # Inverse-distance target interpolation also preserves y=x_original+5.
    np.testing.assert_allclose(sampled_y, scaler.inverse_transform(sampled_X)[:, 0] + 5,
                               atol=1e-12, rtol=1e-12)


def test_smogn_seed_is_reproducible_after_unrelated_numpy_allocations():
    _, X, y = _rare_rows()
    X = StandardScaler().fit_transform(X)
    sampling = {"method": "smogn", "focus": "high", "neighbors": 3}
    expected_X, expected_y, expected_summary = ResamplingService().fit_resample(X, y, sampling, seed=19)
    for value in (-1000., 1e6, 0.):
        # Heap contents must never become an input to a seeded sampler. These
        # allocations exercise real NumPy memory reuse, without mocking a backend.
        unrelated = [np.full((rows, 4), value) for rows in range(2, 150)]
        del unrelated
        actual_X, actual_y, actual_summary = ResamplingService().fit_resample(X, y, sampling, seed=19)
        np.testing.assert_allclose(actual_X, expected_X, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(actual_y, expected_y, atol=1e-12, rtol=1e-12)
        assert actual_summary == expected_summary


def test_selected_features_and_mixed_json_categories_survive_pipeline_clone_and_export(lab):
    data, trainer, root = lab
    rng = np.random.default_rng(103)
    X = rng.normal(size=(160, 2))
    categories = [True, False, 1, "red", None]
    rows = [{"x": float(x), "noise": float(noise), "constant": 9.0,
             "category": categories[index % len(categories)], "y": float(4 + 7 * x)}
            for index, (x, noise) in enumerate(X)]
    request = _request(data, root, rows, model="ols", params={},
                       preprocessing={"scaler": "standard", "imputation": "mean",
                                      "variance_threshold": 0., "selection": "f_regression", "max_features": 1})
    result = trainer.run(request)
    saved = joblib.load(request["artifact_path"])
    bundle = data.resolve(request["dataset_id"], "y", request["features"])
    train = result["split"]["indices"]["train"]
    assert result["feature_names"] == ["x"]
    assert saved.named_steps["preprocessing"].get_feature_names_out().tolist() == ["x"]
    copied = clone(saved).fit(bundle.X.iloc[train], bundle.y[train])
    np.testing.assert_allclose(copied.predict(bundle.X), saved.predict(bundle.X), atol=1e-12)
    np.testing.assert_allclose(saved.predict(bundle.X), bundle.y, atol=1e-12)
    json_rows = pd.DataFrame([
        {"x": -.3, "noise": 100., "constant": 9., "category": True},
        {"x": 2., "noise": 0., "constant": 9., "category": "new category"},
        {"x": 1., "noise": -3., "constant": 9., "category": None},
    ])
    np.testing.assert_allclose(saved.predict(json_rows), [1.9, 18., 11.], atol=1e-12)


def test_sgd_main_fit_and_each_cv_fold_use_the_requested_partial_fit_epoch_budget(lab):
    data, trainer, root = lab
    rng = np.random.default_rng(24)
    X = rng.normal(size=(100, 2))
    y = 2 + X @ [3., -1.] + rng.normal(scale=.1, size=len(X))
    rows = [{"x": float(x), "z": float(z), "y": float(target)} for (x, z), target in zip(X, y, strict=True)]
    params = {"alpha": .001, "eta0": .025, "learning_rate": "constant", "penalty": "l2"}
    request = _request(data, root, rows, model="sgd", params=params, epochs=7, cv=3)
    result = trainer.run(request)
    saved = joblib.load(request["artifact_path"])
    outer_train = np.asarray(result["split"]["indices"]["train"])
    train_X, train_y = X[outer_train], y[outer_train]
    assert saved.named_steps["model"].max_iter == 7
    assert result["trace"][-1]["step"] == 7

    def fitted_reference(indices):
        scaler = StandardScaler().fit(train_X[indices])
        candidate = SGDRegressor(random_state=17, max_iter=7, tol=None, **params)
        transformed = scaler.transform(train_X[indices])
        for _ in range(7):
            candidate.partial_fit(transformed, train_y[indices])
        return scaler, candidate

    scaler, candidate = fitted_reference(np.arange(len(outer_train)))
    np.testing.assert_allclose(saved.named_steps["model"].coef_, candidate.coef_, rtol=1e-12, atol=1e-12)
    for fold in result["cv"]["folds"]:
        valid = np.asarray(fold["validation_indices"])
        scaler, candidate = fitted_reference(np.asarray(fold["train_indices"]))
        errors = train_y[valid] - candidate.predict(scaler.transform(train_X[valid]))
        assert fold["metrics"]["mse"] == pytest.approx(np.mean(errors ** 2), rel=1e-12, abs=1e-12)
        assert fold["metrics"]["mae"] == pytest.approx(np.mean(np.abs(errors)), rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("strategy", ["group_kfold", "timeseries"])
def test_holdout_and_arbitrary_prediction_grid_use_the_same_group_or_time_context(lab, strategy):
    data, trainer, root = lab
    rows = [{"x": float(index / 10), "z": float(index % 11),
             **({"group": f"patient-{index // 8}"} if strategy == "group_kfold" else {}),
             "y": float(1 + 2 * index / 10 - .4 * (index % 11))}
            for index in range(120)]
    config = {"strategy": strategy, "folds": 3}
    if strategy == "group_kfold":
        config["group_column"] = "group"
    request = _request(data, root, rows, cv_config=config)
    result = trainer.run(request)
    saved = joblib.load(request["artifact_path"])
    train = np.asarray(result["split"]["indices"]["train"])
    if strategy == "group_kfold":
        group_sets = [set(rows[index]["group"] for index in indices)
                      for indices in result["split"]["indices"].values()]
        assert all(group_sets[a].isdisjoint(group_sets[b]) for a, b in ((0, 1), (0, 2), (1, 2)))
        assert saved.feature_names_in_.tolist() == ["x", "z"]
        assert "group" not in result["preprocessing"]["categorical_features"]
    else:
        assert result["split"]["shuffle"] is False
        np.testing.assert_array_equal(train, np.arange(72))
        assert train.max() < min(result["split"]["indices"]["validation"])
    grid = trainer.prediction_grid(request, saved, "z")
    train_median = np.median([rows[index]["x"] for index in train])
    grid_rows = pd.DataFrame({"x": np.full(len(grid["x"]), train_median), "z": grid["x"]})
    np.testing.assert_allclose(grid["z"], saved.predict(grid_rows), rtol=1e-12, atol=1e-12)
    assert grid["features"] == ["z"]
    with pytest.raises(ValueError, match="числовых признака"):
        trainer.prediction_grid(request, saved, "group" if strategy == "group_kfold" else "y")


def test_time_learning_curve_keeps_chronological_subsets_even_if_ui_shuffle_is_true(lab):
    data, trainer, root = lab
    x = np.arange(100, dtype=float) / 10
    y = 3 * x + .2 * x ** 2
    rows = [{"x": float(a), "y": float(b)} for a, b in zip(x, y, strict=True)]
    request = _request(data, root, rows, learning_curve=True,
                       cv_config={"strategy": "timeseries", "folds": 3})
    result = trainer.run(request)
    first = result["diagnostics"]["learning_curve"]["points"][0]
    selected = np.asarray(result["split"]["indices"]["train"][:first["train_size"]])
    validation = np.asarray(result["split"]["indices"]["validation"])
    reference = make_pipeline(StandardScaler(), Ridge(alpha=3.0)).fit(x[selected, None], y[selected])
    assert first["train_mse"] == pytest.approx(np.mean((y[selected] - reference.predict(x[selected, None])) ** 2), rel=1e-12)
    assert first["validation_mse"] == pytest.approx(np.mean((y[validation] - reference.predict(x[validation, None])) ** 2), rel=1e-12)


def test_preprocessing_preview_aligns_original_rows_targets_transforms_and_sampled_histogram(tmp_path):
    rows, X, y = _rare_rows()
    train, held_out = train_test_split(np.arange(len(y)), train_size=.6, random_state=17)
    settings = {"target": "y", "features": ["x", "z", "w"], "seed": 17,
                "split": {"train": .6, "validation": .2, "test": .2, "shuffle": True},
                "preprocessing": {"scaler": "standard", "imputation": "mean", "degree": 2,
                                  "interaction_only": True, "selection": "f_regression", "max_features": 2},
                "resampling": {"method": "random_over", "focus": "high"}}
    app = create_app(tmp_path)
    with TestClient(app) as client:
        def preview(values):
            loaded = client.post("/api/datasets/load", json={"kind": "custom", "rows": values, "target": "y"})
            assert loaded.status_code == 200, loaded.text
            identifier = loaded.json()["id"]
            response = client.post(f"/api/datasets/{identifier}/preprocessing-preview", json=settings)
            assert response.status_code == 200, response.text
            return response.json(), app.state.datasets.resolve(identifier, "y", ["x", "z", "w"])

        actual, bundle = preview(rows)
        reference = make_pipeline(SimpleImputer(strategy="mean", keep_empty_features=True),
                                  PolynomialFeatures(degree=2, interaction_only=True, include_bias=False),
                                  StandardScaler(), SelectKBest(f_regression, k=2))
        transformed = reference.fit_transform(bundle.X.iloc[train], bundle.y[train])
        sampled, sampled_y, summary = ResamplingService().fit_resample(
            transformed, bundle.y[train], settings["resampling"], seed=17,
        )
        assert actual["fitted_on"] == "train"
        np.testing.assert_array_equal(actual["indices"], train[:100])
        assert actual["original_rows"] == bundle.X.iloc[train[:100]].to_dict(orient="records")
        np.testing.assert_allclose(pd.DataFrame(actual["original_rows"])[["x", "z", "w"]], X[train[:100]], atol=1e-14, rtol=1e-14)
        np.testing.assert_array_equal(actual["target"]["values"], bundle.y[train[:100]])
        assert actual["features"] == reference.get_feature_names_out().tolist()
        np.testing.assert_allclose(actual["rows"], transformed[:100], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(actual["means"], transformed.mean(axis=0), atol=1e-12)
        np.testing.assert_allclose(actual["std"], transformed.std(axis=0), atol=1e-12)
        np.testing.assert_array_equal(actual["sampling"]["target_before"], bundle.y[train])
        np.testing.assert_array_equal(actual["sampling"]["target_after"], sampled_y)
        np.testing.assert_allclose(actual["sampling"]["rows"], sampled[:100], rtol=1e-12, atol=1e-12)
        assert actual["sampling"]["after"] == len(sampled_y) > len(train)
        assert actual["sampling"]["before"] == len(train)
        assert summary["after"] == actual["sampling"]["after"]
        changed = deepcopy(rows)
        for index in held_out:
            changed[index]["y"] += 1e8
            changed[index]["x"] += 1e9
        assert preview(changed)[0] == actual


@pytest.mark.parametrize("model,params,target", [
    ("tweedie", {"power": 0.}, np.linspace(-5, 5, 50)),
    ("tweedie", {"power": 1.5}, np.linspace(-1, 5, 50)),
    ("tweedie", {"power": 2.}, np.linspace(0, 5, 50)),
])
def test_tweedie_target_domain_tracks_power_instead_of_blanket_positive_rule(lab, model, params, target):
    data, trainer, root = lab
    rows = [{"x": float(i), "y": float(y)} for i, y in enumerate(target)]
    request = _request(data, root, rows, model=model, params=params,
                       split={"train": .6, "validation": .2, "test": .2, "shuffle": False})
    if params["power"] == 0:
        result = trainer.run(request)
        assert np.isfinite(result["metrics"]["validation"]["mse"])
        assert result["trace"][-1]["objective"] is not None
    else:
        with pytest.raises(ValueError, match="(неотрицательная|положительная)"):
            trainer.run(request)
        assert not Path(request["artifact_path"]).exists()


@pytest.mark.parametrize("model", ["gamma", "poisson"])
def test_glm_learns_only_positive_train_and_keeps_link_correct_in_saved_grid(lab, model):
    data, trainer, root = lab
    x = np.linspace(-1, 1, 100)
    y = np.exp(1 + .4 * x)
    y[60:] = -3.  # Invalid external answers must not be fitted or silently removed.
    rows = [{"x": float(a), "y": float(b)} for a, b in zip(x, y, strict=True)]
    request = _request(data, root, rows, model=model, params={"alpha": .1},
                       metrics=["mse", f"{model}_deviance"],
                       split={"train": .6, "validation": .2, "test": .2, "shuffle": False})
    result = trainer.run(request)
    saved = joblib.load(request["artifact_path"])
    assert result["metrics"]["train"][f"{model}_deviance"] is not None
    assert result["metrics"]["validation"][f"{model}_deviance"] is None
    assert result["metric_details"]["validation"][f"{model}_deviance"]["reason"]
    assert result["split"]["counts"] == {"train": 60, "validation": 20, "test": 20}
    grid = result["prediction_grid"]
    np.testing.assert_allclose(grid["z"], saved.predict(pd.DataFrame({"x": grid["x"]})), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result["trace"][-1]["grid"]["z"], grid["z"], rtol=1e-12, atol=1e-12)
    assert np.all(np.asarray(grid["z"]) > 0)
