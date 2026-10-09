"""Feature engineering invariants and real regression-sampler behavior."""

import importlib.util

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from linear_lab.preprocessing import QuantileClipper, ResamplingService, build_preprocessor


def test_training_statistics_are_unchanged_by_heldout_rows():
    train = pd.DataFrame({"a": [0.0, 1, 2, 3, 100, np.nan], "b": [1.0, 2, 3, 4, 5, 6]})
    preprocessor, numeric, categories = build_preprocessor(train, {"clip_quantiles": [0.2, 0.8], "imputation": "mean", "scaler": "standard"})
    assert isinstance(preprocessor, ColumnTransformer)
    assert numeric == ["a", "b"] and categories == []
    transformed = preprocessor.fit_transform(train)
    fitted = preprocessor.named_transformers_["numeric"]
    np.testing.assert_allclose(fitted.named_steps["clip"].bounds_[:, 0], [0.8, 22.4])
    expected_a = [0.8, 1, 2, 3, 22.4]
    assert fitted.named_steps["impute"].statistics_[0] == pytest.approx(np.mean(expected_a))
    statistics = fitted.named_steps["impute"].statistics_.copy()
    scale_mean = fitted.named_steps["scale"].mean_.copy()
    bounds = fitted.named_steps["clip"].bounds_.copy()
    predictions = preprocessor.transform(pd.DataFrame({"a": [-1e9, 1e9], "b": [-1e12, 1e12]}))
    assert np.isfinite(predictions).all()
    np.testing.assert_array_equal(fitted.named_steps["impute"].statistics_, statistics)
    np.testing.assert_array_equal(fitted.named_steps["scale"].mean_, scale_mean)
    np.testing.assert_array_equal(fitted.named_steps["clip"].bounds_, bounds)
    np.testing.assert_allclose(transformed.mean(axis=0), 0, atol=1e-12)


@pytest.mark.parametrize("strategy", ["median", "mean", "most_frequent", "constant", "knn"])
def test_imputation_retains_empty_columns_and_names(strategy):
    X = pd.DataFrame({"a": [1.0, np.nan, 3, 4], "empty": [np.nan] * 4})
    preprocessing, _, _ = build_preprocessor(X, {"imputation": strategy, "scaler": "none", "missing_indicator": True, "fill_value": 9})
    fitted = preprocessing.fit_transform(X)
    assert fitted.shape == (4, 4)
    assert len(preprocessing.get_feature_names_out()) == fitted.shape[1]
    assert np.isfinite(fitted).all()
    assert "missingindicator_a" in preprocessing.get_feature_names_out()
    assert "empty" in preprocessing.get_feature_names_out()


@pytest.mark.parametrize("transform", ["none", "log1p", "sqrt", "yeo_johnson", "quantile_normal", "quantile_uniform"])
@pytest.mark.parametrize("scaler", ["standard", "robust", "minmax", "maxabs", "none"])
def test_numeric_transformations_accept_negative_values_and_serialize(transform, scaler, tmp_path):
    X = pd.DataFrame({"negative": np.linspace(-30, 15, 50), "constant": np.zeros(50)})
    preprocessing, _, _ = build_preprocessor(X, {"numeric_transform": transform, "scaler": scaler, "degree": 2, "interaction_only": True})
    fitted = preprocessing.fit_transform(X)
    assert fitted.shape == (50, 3)
    assert np.isfinite(fitted).all()
    assert list(preprocessing.get_feature_names_out()) == ["negative", "constant", "negative constant"]
    path = tmp_path / "preprocessor.joblib"
    joblib.dump(preprocessing, path)
    np.testing.assert_allclose(joblib.load(path).transform(X), fitted)
    assert clone(preprocessing).get_params()["selection"] == "none"


def test_mixed_json_categories_and_unknown_values_survive_pipeline_export(tmp_path):
    X = pd.DataFrame({"number": [1.0, 2, np.nan, 4, 5, 6], "category": [True, False, None, "blue", True, False]})
    y = [1, 2, 3, 4, 5, 6]
    preprocessing, _, categorical = build_preprocessor(X, {})
    assert categorical == ["category"]
    pipeline = Pipeline([("preprocessing", preprocessing), ("model", LinearRegression())]).fit(X, y)
    predictions = pd.DataFrame([{"number": 2.5, "category": True}, {"number": None, "category": "unknown"}])
    path = tmp_path / "pipeline.joblib"
    joblib.dump(pipeline, path)
    np.testing.assert_allclose(joblib.load(path).predict(predictions), pipeline.predict(predictions))
    assert np.isfinite(pipeline.predict(predictions)).all()
    assert "category_True" in preprocessing.get_feature_names_out()


@pytest.mark.parametrize("selection", ["f_regression", "mutual_info"])
def test_global_selection_uses_train_target_and_keeps_real_names(selection):
    rng = np.random.default_rng(11)
    X = pd.DataFrame({"signal": rng.normal(size=300), "noise": rng.normal(size=300), "constant": np.ones(300)})
    y = 5 * X["signal"].to_numpy()
    preprocessor, _, _ = build_preprocessor(X, {"selection": selection, "max_features": 1, "variance_threshold": 0})
    transformed = preprocessor.fit_transform(X, y)
    assert transformed.shape == (300, 1)
    assert list(preprocessor.get_feature_names_out()) == ["signal"]
    names_before = preprocessor.get_feature_names_out().copy()
    heldout = X.copy()
    heldout["noise"] *= 1e9
    assert np.isfinite(preprocessor.transform(heldout)).all()
    np.testing.assert_array_equal(preprocessor.get_feature_names_out(), names_before)
    assert clone(preprocessor).fit_transform(X, -y).shape == transformed.shape


def test_selection_without_training_target_has_actionable_error():
    X = pd.DataFrame({"x": [1.0, 2, 3]})
    preprocessor, _, _ = build_preprocessor(X, {"selection": "f_regression", "max_features": 1})
    with pytest.raises(ValueError, match="X_train, y_train"):
        preprocessor.fit_transform(X)


@pytest.mark.parametrize("config,match", [
    ({"degree": 6}, "Степень"),
    ({"degree": True}, "Степень"),
    ({"scaler": "magic"}, "Масштабирование"),
    ({"imputation": "guess"}, "Заполнение"),
    ({"clip_quantiles": [0.9, 0.1]}, "квантиль"),
    ({"clip_quantiles": "bad"}, "квантилями"),
    ({"selection": "none", "max_features": 2}, "метод отбора"),
    ({"variance_threshold": -1}, "Порог дисперсии"),
    ({"missing_indicator": True, "imputation": "none"}, "Индикаторы"),
    ({"interaction_only": "yes"}, "выбери да или нет"),
])
def test_invalid_settings_fail_before_fitting(config, match):
    X = pd.DataFrame({"x": [1.0, 2, 3]})
    with pytest.raises(ValueError, match=match):
        build_preprocessor(X, config)


def test_feature_expansion_is_guarded_before_allocation():
    X = pd.DataFrame(np.zeros((10, 20)))
    with pytest.raises(ValueError, match="2000"):
        build_preprocessor(X, {"degree": 5})


def test_no_imputation_does_not_silently_pass_missing_values_to_model():
    X = pd.DataFrame({"x": [1.0, np.nan, 3]})
    preprocessing, _, _ = build_preprocessor(X, {"impute": False, "scale": False})
    with pytest.raises(ValueError, match="пропуски"):
        preprocessing.fit_transform(X)


def test_clipper_learns_no_global_statistics_at_construction():
    clipper = QuantileClipper()
    assert not hasattr(clipper, "bounds_")
    clone(clipper).fit([[0.0], [1.0], [2.0]])
    assert not hasattr(clipper, "bounds_")


def _rare_regression():
    rng = np.random.default_rng(15)
    y = np.r_[rng.normal(size=100), rng.normal(20, 1, size=20)]
    X = np.c_[y + rng.normal(size=len(y)), rng.normal(size=len(y))]
    return X, y


@pytest.mark.skipif(importlib.util.find_spec("ImbalancedLearningRegression") is None or importlib.util.find_spec("smogn") is None, reason="optional research packages not installed")
@pytest.mark.parametrize("method", ["random_over", "random_under", "smoter", "smogn"])
def test_real_regression_sampling_is_reproducible_and_does_not_modify_inputs_or_rng(method):
    X, y = _rare_regression()
    original_X, original_y = X.copy(), y.copy()
    settings = {"method": method, "neighbors": 3}
    np.random.seed(321)
    expected = np.random.random(4)
    np.random.seed(321)
    sampler = ResamplingService()
    sampled_X, sampled_y, summary = sampler.fit_resample(X, y, settings, seed=8)
    np.testing.assert_array_equal(np.random.random(4), expected)
    again_X, again_y, again_summary = sampler.fit_resample(X, y, settings, seed=8)
    np.testing.assert_allclose(sampled_X, again_X, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(sampled_y, again_y, rtol=1e-14, atol=1e-14)
    np.testing.assert_array_equal(X, original_X)
    np.testing.assert_array_equal(y, original_y)
    assert summary == again_summary
    assert summary["before"] == 120 and summary["after"] == len(sampled_y)
    assert summary["fitted_on"] == "train"
    if method == "random_under":
        assert len(sampled_y) < len(y)
    elif method == "random_over":
        assert len(sampled_y) > len(y)
    else:
        original = set(tuple(row) for row in np.c_[X, y])
        assert any(tuple(row) not in original for row in np.c_[sampled_X, sampled_y])


@pytest.mark.parametrize("method", ["smoter", "smogn"])
def test_synthetic_sampling_rejects_fractional_one_hot_categories(method):
    X, y = _rare_regression()
    with pytest.raises(ValueError, match="дробные категории"):
        ResamplingService().fit_resample(X, y, {"method": method}, categorical=True)


def test_none_resampling_preserves_original_rows_exactly():
    X, y = _rare_regression()
    result_X, result_y, summary = ResamplingService().fit_resample(X, y)
    np.testing.assert_array_equal(X, result_X)
    np.testing.assert_array_equal(y, result_y)
    assert summary == {"method": "none", "before": 120, "after": 120, "fitted_on": "train", "synthetic": False}
