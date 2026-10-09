"""Independent slice geometry and held-out isolation, using real estimators."""
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder

from cml_lab.infrastructure.ml.catalogue import AlgorithmCatalogue
from cml_lab.infrastructure.ml.fitting import fit_artifact
from cml_lab.infrastructure.ml.prediction_views import build_prediction_view


def test_plane_and_fixed_values_match_known_function_without_test_rows():
    rng = np.random.default_rng(20)
    X = pd.DataFrame(rng.normal(size=(100, 3)), columns=["a", "b", "c"])
    y = 2 * X.a.to_numpy() - 3 * X.b.to_numpy() + 4 * X.c.to_numpy() + 5
    train, validation = np.arange(60), np.arange(60, 80)
    artifact, _, _ = fit_artifact({"task": "regression", "algorithm_id": "ols"}, X.iloc[train], y[train], AlgorithmCatalogue())
    before = artifact.predict(X)
    view = build_prediction_view(artifact, X, y, train, validation, ["a", "b"])
    xx, yy = np.meshgrid(*view["axes"])
    expected = 2 * xx - 3 * yy + 4 * np.median(X.c.iloc[train]) + 5
    np.testing.assert_allclose(view["predictions"], expected, atol=1e-12)
    assert len(view["points"]["actual"]) == 80
    assert view["fixed_features"] == {"c": np.median(X.c.iloc[train])}
    changed = X.copy()
    changed.iloc[80:] = 1e12
    changed_y = y.copy()
    changed_y[80:] = -1e12
    assert view == build_prediction_view(artifact, changed, changed_y, train, validation, ["a", "b"])
    np.testing.assert_allclose(before, artifact.predict(X), rtol=0, atol=0)


def test_classification_regions_return_real_decoded_class_indices():
    X = pd.DataFrame({"x": np.tile(np.arange(-10, 10), 3), "category": ["a", "b", "a"] * 20})
    y = (X.x > 0).astype(int).to_numpy()
    encoder = LabelEncoder().fit(["left", "right"])
    artifact, _, _ = fit_artifact({"task": "classification", "algorithm_id": "decision_tree_classifier"}, X.iloc[:40], y[:40], AlgorithmCatalogue(), label_encoder=encoder)
    view = build_prediction_view(artifact, X, y, np.arange(40), np.arange(40, 50))
    assert view["classes"] == ["left", "right"]
    assert view["features"] == ["x"]
    assert view["fixed_features"] == {"category": "a"}
    expected = (np.asarray(view["axes"][0]) > .5).astype(int)
    np.testing.assert_array_equal(view["predictions"], expected)


def test_slice_refuses_categorical_axis_and_handles_nullable_numeric():
    X = pd.DataFrame({"x": pd.Series(range(50), dtype="Float64"), "label": ["a", "b"] * 25})
    X.loc[2, "x"] = pd.NA
    y = np.arange(50)
    artifact, _, _ = fit_artifact({"task": "regression", "algorithm_id": "ridge"}, X.iloc[:30], y[:30], AlgorithmCatalogue())
    view = build_prediction_view(artifact, X, y, np.arange(30), np.arange(30, 40))
    assert len(view["points"]["actual"]) == 39
    assert np.isfinite(view["predictions"]).all()
    with pytest.raises(ValueError, match="числов"):
        build_prediction_view(artifact, X, y, np.arange(30), np.arange(30, 40), ["label"])
