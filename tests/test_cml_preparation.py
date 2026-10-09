"""Preparation scenarios: saved recipes, leakage, real features and samplers."""

import pickle

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.base import clone

from cml_lab.infrastructure.ml.preparation import PreparationCatalogue, build_preprocessor, fit_resample


def step(adapter_id, columns=None, **params):
    return {"adapter_id": adapter_id, "columns": columns or [], "params": params}


def config(*steps):
    return {"steps": list(steps)}


def dense(values):
    return values.toarray() if sparse.issparse(values) else values


def test_train_statistics_and_unknown_categories_remain_frozen():
    train = pd.DataFrame({"x": [0., 2., np.nan], "category": ["red", "blue", "red"]})
    prep = build_preprocessor(train, config())
    transformed = dense(prep.fit_transform(train))
    np.testing.assert_allclose(transformed[:, 0], [-np.sqrt(1.5), np.sqrt(1.5), 0])
    names = prep.get_feature_names_out().tolist()
    heldout = dense(prep.transform(pd.DataFrame({"x": [1000., np.nan], "category": ["unseen", "red"]})))
    assert heldout.shape[1] == len(names)
    assert "unseen" not in " ".join(names)
    np.testing.assert_allclose(heldout[1, 0], 0)
    np.testing.assert_allclose(heldout[0, 1:], 0)
    np.testing.assert_allclose(dense(prep.transform(train)), transformed)


def test_disjoint_numeric_branches_honour_column_selection_and_drop():
    train = pd.DataFrame({"first": [0., 10.], "second": [10., 20.], "id": [123., 999.]})
    prep = build_preprocessor(train, config(step("columns.drop", ["id"]), step("numeric.scale", ["first"], method="minmax"), step("numeric.scale", ["second"], method="none")))
    values = dense(prep.fit_transform(train))
    np.testing.assert_allclose(values, [[0., 10.], [1., 20.]])
    assert all("id" not in name for name in prep.get_feature_names_out())


def test_explicit_stage_order_changes_polynomial_geometry():
    train = pd.DataFrame({"x": [0., 1., 2.]})
    before = build_preprocessor(train, config(step("numeric.scale"), step("numeric.polynomial", degree=2)))
    after = build_preprocessor(train, config(step("numeric.polynomial", degree=2), step("numeric.scale")))
    first = dense(before.fit_transform(train))
    second = dense(after.fit_transform(train))
    np.testing.assert_allclose(first[:, 1], [1.5, 0., 1.5])
    np.testing.assert_allclose(np.mean(second, axis=0), [0., 0.], atol=1e-12)
    assert not np.allclose(first, second)


def test_clone_and_serialization_keep_prediction_preparation():
    train = pd.DataFrame({"x": [1., np.nan, 4., 8.], "category": ["a", "a", "b", "c"]})
    original = build_preprocessor(train, config(step("numeric.impute", strategy="knn", n_neighbors=2, add_indicator=True), step("categorical.frequency")))
    replica = clone(original)
    expected = dense(original.fit_transform(train))
    np.testing.assert_allclose(dense(replica.fit_transform(train)), expected)
    restored = pickle.loads(pickle.dumps(original))
    np.testing.assert_allclose(dense(restored.transform(train)), expected)
    assert len(original.get_feature_names_out()) == expected.shape[1]


def test_categorical_frequency_uses_train_only_and_unknown_zero():
    train = pd.DataFrame({"kind": ["a", "a", "b"]})
    prep = build_preprocessor(train, config(step("categorical.frequency")))
    np.testing.assert_allclose(dense(prep.fit_transform(train)).ravel(), [2/3, 2/3, 1/3])
    np.testing.assert_allclose(dense(prep.transform(pd.DataFrame({"kind": ["unseen", "a"]}))).ravel(), [0., 2/3])


def test_ordinal_given_order_and_unknown_are_real_codes():
    train = pd.DataFrame({"size": ["large", "small", "medium"]})
    prep = build_preprocessor(train, config(step("categorical.ordinal", categories=[["small", "medium", "large"]])))
    np.testing.assert_allclose(dense(prep.fit_transform(train)).ravel(), [2., 0., 1.])
    assert dense(prep.transform(pd.DataFrame({"size": ["new"]})))[0, 0] == -1


def test_target_encoder_training_is_cross_fitted_not_own_label():
    train = pd.DataFrame({"id_like": [f"unique_{i}" for i in range(12)]})
    y = np.arange(12, dtype=float) ** 2
    prep = build_preprocessor(train, config(step("categorical.target", smooth=0., cv=3)), task="regression")
    cross_fitted = dense(prep.fit_transform(train, y)).ravel()
    memorized = dense(prep.transform(train)).ravel()
    np.testing.assert_allclose(memorized, y)
    assert not np.allclose(cross_fitted, y)
    grouped = config(step("categorical.target"))
    grouped["split_kind"] = "group_kfold"
    with pytest.raises(ValueError, match="crossfit"):
        build_preprocessor(train, grouped)


@pytest.mark.parametrize("method", ["tfidf", "count", "hash"])
def test_text_vocabulary_names_sparse_and_unknown_tokens(method):
    train = pd.DataFrame({"text": ["кошка спит", "кот спит", "кошка ест"]})
    prep = build_preprocessor(train, config(step(f"text.{method}", ["text"])))
    values = prep.fit_transform(train)
    assert sparse.issparse(values)
    assert values.shape[1] == len(prep.get_feature_names_out())
    unknown = prep.transform(pd.DataFrame({"text": ["телескоп"]}))
    assert unknown.shape[1] == values.shape[1]
    if method != "hash":
        assert unknown.nnz == 0
        assert "телескоп" not in " ".join(prep.get_feature_names_out())
    assert np.all(values.data >= 0)


def test_datetime_invalid_is_missing_and_cyclic_month_wraps():
    train = pd.DataFrame({"date": ["2020-01-01", "2020-12-01", "bad date"]})
    prep = build_preprocessor(train, config(step("datetime.extract", ["date"], parts=["month"], cyclic=True)))
    values = dense(prep.fit_transform(train))
    np.testing.assert_allclose(values[0], [1., 0., 1.], atol=1e-12)
    np.testing.assert_allclose(values[1], [12., -0.5, np.sqrt(3)/2], atol=1e-12)
    np.testing.assert_allclose(values[2], np.median(values[:2], axis=0))
    assert values.shape[1] == len(prep.get_feature_names_out())


@pytest.mark.parametrize("method", ["pca", "svd", "ica"])
def test_reduction_components_fitted_on_train_and_named(method):
    rng = np.random.default_rng(91)
    train = pd.DataFrame(rng.normal(size=(40, 3)), columns=["a", "b", "c"])
    prep = build_preprocessor(train, config(step(f"reduction.{method}", n_components=2)))
    fit_values = dense(prep.fit_transform(train))
    assert fit_values.shape == (40, 2)
    assert len(prep.get_feature_names_out()) == 2
    np.testing.assert_allclose(dense(prep.transform(train)), fit_values)
    copied = clone(prep).fit(train)
    np.testing.assert_allclose(dense(copied.transform(train)), fit_values)


def test_svd_preserves_sparse_text_path_without_densifying_input():
    train = pd.DataFrame({"text": ["red green", "red blue", "green yellow", "blue yellow"]})
    prep = build_preprocessor(train, config(step("text.tfidf", ["text"]), step("reduction.svd", n_components=2)))
    values = prep.fit_transform(train)
    assert values.shape == (4, 2)
    assert np.isfinite(values).all()


def test_classification_selector_retains_known_discriminative_feature():
    y = np.repeat(["left", "right"], 20)
    X = pd.DataFrame({"signal": np.r_[np.arange(20)/100, 10+np.arange(20)/100], "noise": np.tile(np.arange(20), 2)})
    prep = build_preprocessor(X, config(step("selection.univariate", method="f", k=1)), task="classification")
    values = prep.fit_transform(X, y)
    assert values.shape == (40, 1)
    assert "signal" in prep.get_feature_names_out()[0]


@pytest.mark.parametrize("stages,match", [
    ([step("numeric.power", method="bogus")], "выберите"),
    ([step("numeric.scale", hacked=True)], "неизвестные"),
    ([step("numeric.scale", ["x"]), step("numeric.polynomial", ["x", "other"])], "selector"),
    ([step("numeric.scale"), step("numeric.impute")], "первым"),
    ([step("numeric"), step("numeric.scale")], "Составной"),
    ([step("columns.drop")], "явно"),
    ([step("selection.univariate", ["x"])], "Глобальный"),
])
def test_invalid_recipes_rejected_before_fit(stages, match):
    with pytest.raises(ValueError, match=match):
        build_preprocessor(pd.DataFrame({"x": [1., 2.], "other": [3., 4.]}), {"steps": stages})


def test_clustering_cannot_select_features_from_a_target():
    with pytest.raises(ValueError, match="несовместим"):
        build_preprocessor(pd.DataFrame({"x": [1., 2.]}), config(step("selection.univariate")), task="clustering")


def test_polynomial_size_guard_runs_before_materialization():
    X = pd.DataFrame(np.zeros((20, 80)), columns=[f"x{i}" for i in range(80)])
    prep = build_preprocessor(X, config(step("numeric.polynomial", degree=5)))
    with pytest.raises(ValueError, match="Полином превысит"):
        prep.fit_transform(X)


def test_negative_chi_square_refused_and_counts_supported():
    X = pd.DataFrame({"x": [-1., 2., 3., 4.], "z": [0., 1., 0., 1.]})
    y = [0, 0, 1, 1]
    prep = build_preprocessor(X, config(step("selection.univariate", method="chi2", k=1)), task="classification")
    with pytest.raises(ValueError, match="неотрицательные"):
        prep.fit_transform(X, y)
    nonnegative = build_preprocessor(X, config(step("numeric.scale", method="minmax", clip=True), step("selection.univariate", method="chi2", k=1)), task="classification")
    assert nonnegative.fit_transform(X, y).shape == (4, 1)


def test_catalogue_metadata_is_independent_and_every_stage_has_instruction():
    catalogue = PreparationCatalogue()
    first = catalogue.stages()
    assert {"text.tfidf", "categorical.target", "reduction.pca", "numeric", "selection.univariate"}.issubset({stage["id"] for stage in first})
    first[0]["name"] = "mutated"
    assert catalogue.stages()[0]["name"] != "mutated"
    assert all(item["lesson_id"] and item["help_key"] and item["source_urls"] for item in catalogue.stages())
    assert all(field["lesson_id"] and field["help_key"] for item in catalogue.stages() for field in item["params"])


@pytest.fixture
def class_data():
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=80, n_features=3, n_informative=2, n_redundant=0, weights=[0.8, 0.2], random_state=18)
    return X, np.where(y, "rare", "common")


@pytest.mark.parametrize("method", ["random_over", "random_under", "smote", "adasyn", "borderline_smote", "svm_smote", "kmeans_smote", "tomek", "enn", "near_miss", "smoteenn", "smotetomek"])
def test_real_class_samplers_preserve_class_domain_and_seed(method, class_data):
    pytest.importorskip("imblearn")
    X, y = class_data
    options = {"method": method, "neighbors": 2}
    first_X, first_y, summary = fit_resample(X, y, options, task="classification", seed=77)
    second_X, second_y, _ = fit_resample(X, y, options, task="classification", seed=77)
    assert set(first_y) == {"rare", "common"}
    assert np.isfinite(first_X).all()
    assert summary["before"] == 80 and summary["after"] == len(first_y)
    np.testing.assert_allclose(first_X, second_X)
    np.testing.assert_array_equal(first_y, second_y)
    if method in {"random_over", "smote"}:
        counts = pd.Series(first_y).value_counts()
        assert counts.iloc[0] == counts.iloc[1]


def test_sampler_rejects_temporal_category_interpolation_and_small_class(class_data):
    X, y = class_data
    for task in ("forecasting", "panel", "ranking", "clustering"):
        with pytest.raises(ValueError, match="протокол"):
            fit_resample(X, y, {"method": "random_over"}, task=task)
    with pytest.raises(ValueError, match="исходные числовые"):
        fit_resample(X, y, {"method": "smote"}, task="classification", categorical=True)
    with pytest.raises(ValueError, match="число соседей"):
        fit_resample(X, y, {"method": "smote", "neighbors": 100}, task="classification")
    with pytest.raises(ValueError, match="непрерывной цели"):
        fit_resample(X, np.arange(len(y)), {"method": "smote"}, task="regression")


def test_compatible_numeric_recipe_retains_corrected_regression_backend():
    X = pd.DataFrame({"x": np.arange(10, dtype=float)})
    prep = build_preprocessor(X, config(step("numeric", degree=2, scaler="none")))
    result = prep.fit_transform(X)
    np.testing.assert_allclose(result[:, 1], X["x"] ** 2)
    assert len(prep.get_feature_names_out()) == 2


def test_class_sampler_rejects_huge_requested_output_before_allocation(class_data):
    X, y = class_data
    with pytest.raises(ValueError, match="создаст больше"):
        fit_resample(X, y, {"method": "random_over", "sampling_strategy": {"rare": 100_000_000}}, task="classification")


def test_dense_onehot_budget_checked_before_output_matrix_is_created():
    X = pd.DataFrame({"kind": [f"category_{i}" for i in range(15000)]})
    prep = build_preprocessor(X, config(step("categorical.onehot", max_categories=2000)))
    with pytest.raises(ValueError, match="15 млн плотных"):
        prep.fit_transform(X)


@pytest.mark.parametrize("adapter_id", [item["id"] for item in PreparationCatalogue().stages()])
def test_every_enabled_stage_executes_actual_estimator_and_exposes_names(adapter_id):
    rng = np.random.default_rng(50)
    X = pd.DataFrame(rng.uniform(1, 3, (40, 3)), columns=["x", "z", "other"])
    y = X["x"].to_numpy() * 2 + X["z"].to_numpy() ** 2
    stages = []
    if adapter_id.startswith("categorical."):
        X["kind"] = np.tile(["a", "b", "c", "d"], 10)
        stages = [step(adapter_id, ["kind"])]
    elif adapter_id.startswith("text."):
        X["document"] = np.tile(["кошка спит", "собака ест", "кошка ест", "собака спит"], 10)
        stages = [step(adapter_id, ["document"])]
    elif adapter_id == "datetime.extract":
        X["date"] = pd.date_range("2020-01-01", periods=40)
        stages = [step(adapter_id, ["date"])]
    elif adapter_id == "columns.drop":
        stages = [step(adapter_id, ["other"])]
    elif adapter_id == "selection.univariate":
        stages = [step(adapter_id, k=2)]
    elif adapter_id == "selection.model":
        stages = [step(adapter_id, max_features=2)]
    elif adapter_id == "reduction.nmf":
        stages = [step("numeric.scale", method="minmax", clip=True), step(adapter_id, n_components=2)]
    else:
        stages = [step(adapter_id)]
    prep = build_preprocessor(X, config(*stages))
    values = prep.fit_transform(X, y)
    matrix = values.data if sparse.issparse(values) else values
    assert np.isfinite(matrix).all()
    assert values.shape[1] == len(prep.get_feature_names_out())
    assert prep.transform(X.iloc[:2]).shape == (2, values.shape[1])


def test_iterative_imputation_learns_feature_relation_only_from_train():
    train = pd.DataFrame({"x": np.arange(12, dtype=float), "double": np.arange(12, dtype=float) * 2})
    train.loc[[3, 7], "double"] = np.nan
    train["empty"] = np.nan
    prep = build_preprocessor(train, config(step("numeric.impute", strategy="iterative", n_nearest_features=2)))
    values = dense(prep.fit_transform(train))
    np.testing.assert_allclose(values[[3, 7], 1], [6., 14.], atol=1e-4)
    np.testing.assert_allclose(values[:, 2], 0.)
    assert len(prep.get_feature_names_out()) == 3
    heldout = pd.DataFrame({"x": [20., 10000.], "double": [np.nan, 50000.], "empty": [np.nan, np.nan]})
    first = dense(prep.transform(heldout))
    heldout.loc[1, "double"] = -1000000.
    second = dense(prep.transform(heldout))
    np.testing.assert_allclose(first[0], second[0])
    np.testing.assert_allclose(first[0, 1], 40., atol=1e-4)
    replica = clone(prep).fit(train)
    np.testing.assert_allclose(replica.transform(train), values)


def test_iterative_imputation_rejects_excessive_feature_count_before_fit():
    train = pd.DataFrame(np.zeros((2, 101)), columns=[f"x{i}" for i in range(101)])
    with pytest.raises(ValueError, match="Итеративное заполнение"):
        build_preprocessor(train, config(step("numeric.impute", strategy="iterative")))


def test_spline_basis_matches_library_and_keeps_training_knots():
    from sklearn.preprocessing import SplineTransformer
    train = pd.DataFrame({"x": np.linspace(0., 10., 30)})
    prep = build_preprocessor(train, config(step("numeric.spline", n_knots=4, degree=3, knots="uniform")))
    values = dense(prep.fit_transform(train))
    reference = SplineTransformer(n_knots=4, degree=3, knots="uniform", include_bias=False).fit(train)
    np.testing.assert_allclose(values, reference.transform(train))
    heldout = pd.DataFrame({"x": [-100., 5., 100.]})
    np.testing.assert_allclose(prep.transform(heldout), reference.transform(heldout))
    assert values.shape == (30, 5)
    assert len(prep.get_feature_names_out()) == 5
    restored = pickle.loads(pickle.dumps(prep))
    np.testing.assert_allclose(restored.transform(heldout), prep.transform(heldout))
    np.testing.assert_allclose(clone(prep).fit_transform(train), values)


def test_spline_and_bin_check_output_width_before_materialization():
    X = pd.DataFrame(np.tile(np.arange(100)[:, None], (1, 100)), columns=[f"x{i}" for i in range(100)])
    spline = build_preprocessor(X, config(step("numeric.spline", n_knots=30, degree=3)))
    with pytest.raises(ValueError, match="Сплайны превысят"):
        spline.fit_transform(X)
    bins = build_preprocessor(X, config(step("numeric.bin", n_bins=50)))
    with pytest.raises(ValueError, match="Интервалы превысят"):
        bins.fit_transform(X)


def test_mixed_sparse_mutual_information_does_not_treat_measurements_as_discrete():
    X = pd.DataFrame({"measurement": np.linspace(0., 1., 40), "kind": [f"id_{i}" for i in range(40)]})
    y = X["measurement"] ** 2
    mixed = config(step("categorical.onehot", sparse_output=True), step("selection.univariate", method="mutual_info", k=1))
    with pytest.raises(ValueError, match="MI.*плотную"):
        build_preprocessor(X, mixed).fit_transform(X, y)
    dense_config = config(step("categorical.onehot", sparse_output=False), step("selection.univariate", method="mutual_info", k=1, discrete_features="continuous"))
    prep = build_preprocessor(X, dense_config)
    prep.fit_transform(X, y)
    assert "measurement" in prep.get_feature_names_out()[0]


def test_sparse_text_mi_remains_available_for_discrete_counts():
    X = pd.DataFrame({"text": ["cat cat" if i % 2 else "dog" for i in range(40)]})
    y = np.arange(40) % 2
    prep = build_preprocessor(X, config(step("text.count", ["text"]), step("selection.univariate", method="mutual_info", k=1)), task="classification")
    result = prep.fit_transform(X, y)
    assert result.shape == (40, 1)
    assert set(np.unique(result.toarray())).issubset({0., 1., 2.})


def test_ordinal_numeric_categories_use_same_stringification_as_column_values():
    X = pd.DataFrame({"rank": pd.Categorical([10, 20, 30, 10])})
    prep = build_preprocessor(X, config(step("categorical.ordinal", categories=[[10, 20, 30]])))
    np.testing.assert_allclose(prep.fit_transform(X).ravel(), [0., 1., 2., 0.])
    with pytest.raises(ValueError, match="повторяющиеся"):
        build_preprocessor(X, config(step("categorical.ordinal", categories=[[10, "10", 20]])))


def test_expansion_guards_also_apply_to_inference_batches(monkeypatch):
    import cml_lab.infrastructure.ml.preparation as module
    train = pd.DataFrame({"x": [1., 2., 3., 4.]})
    polynomial = build_preprocessor(train, config(step("numeric.polynomial", degree=2))).fit(train)
    spline = build_preprocessor(train, config(step("numeric.spline", n_knots=3, degree=2))).fit(train)
    bins = build_preprocessor(train, config(step("numeric.bin", n_bins=3))).fit(train)
    reducer = build_preprocessor(train, config(step("reduction.pca", n_components=1))).fit(train)
    monkeypatch.setattr(module, "MAX_DENSE_CELLS", 8)
    larger = pd.DataFrame({"x": np.linspace(1., 4., 10)})
    for prep, message in [(polynomial, "Полином прогноза"), (spline, "Сплайны прогноза"), (bins, "Интервалы превысят"), (reducer, "Компоненты прогноза")]:
        with pytest.raises(ValueError, match=message):
            prep.transform(larger)


def test_datetime_expansion_has_allocation_guard(monkeypatch):
    import cml_lab.infrastructure.ml.preparation as module
    X = pd.DataFrame({"first": pd.date_range("2020-01-01", periods=4), "second": pd.date_range("2021-01-01", periods=4)})
    prep = build_preprocessor(X, config(step("datetime.extract", parts=["year", "month", "weekday"], cyclic=True)))
    monkeypatch.setattr(module, "MAX_DENSE_FEATURES", 10)
    with pytest.raises(ValueError, match="Календарные признаки превысят"):
        prep.fit_transform(X)
