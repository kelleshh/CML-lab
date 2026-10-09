"""Factory contracts: real estimators, bounded forms, and truthful capabilities."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import make_blobs, make_classification
from sklearn.metrics import mean_squared_error
from threadpoolctl import threadpool_limits

from cml_lab.infrastructure.ml.catalogue import AlgorithmCatalogue


@pytest.fixture(scope="module")
def catalogue():
    return AlgorithmCatalogue()


def test_catalogue_preserves_linear_ids_and_separates_task_from_family(catalogue):
    all_specs = catalogue.list()
    ids = [item["id"] for item in all_specs]
    assert len(ids) == len(set(ids))
    assert {"ridge", "lasso", "elasticnet", "sgd", "ransac", "theilsen", "huber", "sqrt_lasso"} <= set(ids)
    assert {"random_forest_classifier", "xgboost_ranker", "tsne", "local_outlier_factor"} <= set(ids)
    assert "forecasting" in catalogue.spec("random_forest_regressor")["tasks"]
    assert catalogue.spec("random_forest_regressor")["family"] == "forest"
    assert all(item["tasks"] == ["classification"] for item in catalogue.list("classification"))
    assert all("ranking" in item["tasks"] for item in catalogue.list("ranking"))
    assert catalogue.spec("ridge") == catalogue.descriptor("ridge")


def test_metadata_is_detached_and_every_setting_has_contextual_help(catalogue):
    first = catalogue.spec("svc")
    first["params"][0]["default"] = 999999
    assert catalogue.spec("svc")["params"][0]["default"] == 1.0
    keys = set()
    for spec in catalogue.list():
        assert spec["source"].startswith("https://")
        assert spec["description"] and spec["lesson_id"]
        for field in spec["params"]:
            assert field["label"] and field["help"] and field["lesson_id"]
            assert field["help_key"] == f"model.{spec['id']}.{field['key']}"
            keys.add(field["help_key"])
    assert "model.svc.C" in keys
    assert "model.logistic_regression.C" in keys
    assert "model.scad.gamma" in keys
    assert "model.svc.gamma" in keys


@pytest.mark.parametrize("algorithm_id", [
    spec["id"] for spec in AlgorithmCatalogue().list()
])
def test_every_available_definition_builds_cloneable_estimator(algorithm_id, catalogue):
    spec = catalogue.spec(algorithm_id)
    if not spec["available"]:
        pytest.skip(spec["reason"])
    estimator = catalogue.build(spec["tasks"][0], algorithm_id, seed=17, n_jobs=2)
    copied = clone(estimator)
    assert type(copied) is type(estimator)
    assert copied.get_params(deep=False).keys() == estimator.get_params(deep=False).keys()
    if hasattr(estimator, "n_jobs"):
        assert estimator.n_jobs == 2


@pytest.mark.parametrize("algorithm_id", [
    spec["id"] for spec in AlgorithmCatalogue().list("classification")
])
def test_classification_factories_learn_real_three_class_data(algorithm_id, catalogue):
    spec = catalogue.spec(algorithm_id)
    if not spec["available"]:
        pytest.skip(spec["reason"])
    X, y = make_classification(n_samples=90, n_features=4, n_informative=3, n_redundant=0,
                               n_classes=3, n_clusters_per_class=1, class_sep=2.0, random_state=11)
    if spec["capabilities"]["input_domain"] == "nonnegative":
        X = X - X.min(axis=0) + 0.1
    estimator = catalogue.build("classification", algorithm_id, seed=21)
    with threadpool_limits(limits=1):
        estimator.fit(X, y)
        predicted = np.asarray(estimator.predict(X[:13])).reshape(-1)
    assert predicted.shape == (13,)
    assert set(predicted.tolist()) <= set(y.tolist())
    if hasattr(estimator, "predict_proba"):
        probabilities = estimator.predict_proba(X[:13])
        assert probabilities.shape == (13, 3)
        assert np.isfinite(probabilities).all()
        tolerance = max(1e-12, 4 * np.finfo(probabilities.dtype).eps)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=tolerance, rtol=0)


@pytest.mark.parametrize("algorithm_id", [
    spec["id"] for spec in AlgorithmCatalogue().list("regression")
])
def test_regression_factories_predict_finite_numbers(algorithm_id, catalogue):
    spec = catalogue.spec(algorithm_id)
    if not spec["available"]:
        pytest.skip(spec["reason"])
    rng = np.random.default_rng(15)
    X = rng.normal(size=(70, 4))
    y = np.exp(0.2 * X[:, 0] - 0.1 * X[:, 1] + 1.5)
    estimator = catalogue.build("regression", algorithm_id, seed=21)
    with threadpool_limits(limits=1):
        estimator.fit(X, y)
        predicted = np.asarray(estimator.predict(X[:13])).reshape(-1)
    assert predicted.shape == (13,)
    assert np.isfinite(predicted).all()


@pytest.mark.parametrize("algorithm_id", [
    spec["id"] for spec in AlgorithmCatalogue().list("clustering")
])
def test_cluster_capabilities_do_not_invent_new_row_prediction(algorithm_id, catalogue):
    X, _ = make_blobs(n_samples=80, centers=[[-4, -4, 0], [0, 4, 0], [4, -4, 0]],
                       cluster_std=0.2, random_state=11)
    spec = catalogue.spec(algorithm_id)
    estimator = catalogue.build("clustering", algorithm_id)
    with threadpool_limits(limits=1):
        labels = estimator.fit_predict(X)
    assert np.asarray(labels).shape == (80,)
    assert 2 <= len(np.unique(labels)) <= 80
    assert hasattr(estimator, "predict") == spec["capabilities"]["predict"]
    if spec["capabilities"]["predict"]:
        assert np.asarray(estimator.predict(X[:7])).shape == (7,)


@pytest.mark.parametrize("algorithm_id", [
    spec["id"] for spec in AlgorithmCatalogue().list("anomaly")
])
def test_anomaly_factories_identify_observations_without_target(algorithm_id, catalogue):
    rng = np.random.default_rng(31)
    X = np.concatenate([rng.normal(scale=0.2, size=(80, 4)), rng.normal(loc=9, size=(10, 4))])
    estimator = catalogue.build("anomaly", algorithm_id, seed=17)
    with threadpool_limits(limits=1):
        labels = estimator.fit_predict(X)
    assert np.asarray(labels).shape == (90,)
    assert set(labels) == {-1, 1}
    if algorithm_id == "local_outlier_factor":
        assert not hasattr(estimator, "predict")
        assert catalogue.spec(algorithm_id)["capabilities"]["transductive"] is True


@pytest.mark.parametrize("algorithm_id", [
    spec["id"] for spec in AlgorithmCatalogue().list("reduction")
])
def test_reducers_have_real_two_coordinate_output(algorithm_id, catalogue):
    X = np.random.default_rng(51).uniform(0.1, 3.0, size=(80, 5))
    spec = catalogue.spec(algorithm_id)
    estimator = catalogue.build("reduction", algorithm_id, seed=17)
    with threadpool_limits(limits=1):
        transformed = estimator.fit_transform(X)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    assert np.asarray(transformed).shape == (80, 2)
    assert np.isfinite(transformed).all()
    assert hasattr(estimator, "transform") == spec["capabilities"]["transform"]


def test_linear_fit_matches_known_function_and_temporal_build_keeps_same_estimator(catalogue):
    X = np.asarray([[0, 0], [1, 0], [0, 1], [2, 3], [4, 1]], dtype=float)
    y = 7 + X @ np.asarray([2, -3])
    ordinary = catalogue.build("regression", "ols").fit(X, y)
    forecasting = catalogue.build("forecasting", "ols").fit(X, y)
    np.testing.assert_allclose(ordinary.coef_, [2, -3], atol=1e-12)
    assert ordinary.intercept_ == pytest.approx(7)
    assert mean_squared_error(y, ordinary.predict(X)) < 1e-25
    np.testing.assert_allclose(forecasting.predict(X), ordinary.predict(X))


@pytest.mark.parametrize("task,algorithm_id,params", [
    ("classification", "ridge", {}),
    ("regression", "svc", {}),
    ("classification", "logistic_regression", {"__class__": "exec"}),
    ("classification", "random_forest_classifier", {"n_estimators": True}),
    ("classification", "random_forest_classifier", {"n_estimators": 0}),
    ("regression", "svr", {"C": float("inf")}),
    ("regression", "svr", {"kernel": "arbitrary-python"}),
    ("classification", "knn_classifier", {"p": True}),
    ("classification", "knn_classifier", {"p": "2"}),
    ("regression", "ridge", {"alpha": float("nan")}),
    ("clustering", "kmeans", {"n_clusters": 3.5}),
    ("reduction", "pca", []),
    ("unknown", "ridge", {}),
])
def test_invalid_task_and_unsafe_parameter_values_are_rejected(task, algorithm_id, params, catalogue):
    with pytest.raises(ValueError):
        catalogue.build(task, algorithm_id, params)


def test_ensemble_components_are_customizable_without_importing_user_code(catalogue):
    requested = {"components": [
        {"algorithm_id": "decision_tree_classifier", "params": {"max_depth": 2}},
        {"algorithm_id": "logistic_regression", "params": {"C": 0.25}},
    ], "voting": "soft"}
    original = deepcopy(requested)
    estimator = catalogue.build("classification", "voting_classifier", requested, seed=2, n_jobs=4)
    assert estimator.estimators[0][1].max_depth == 2
    assert estimator.estimators[1][1].C == 0.25
    assert estimator.n_jobs == 4
    assert requested == original
    with pytest.raises(ValueError, match="разрешен"):
        catalogue.build("classification", "voting_classifier", {"components": [
            {"algorithm_id": "os.system", "params": {}}, {"algorithm_id": "gaussian_nb"}]})
    with pytest.raises(ValueError, match="8"):
        catalogue.build("classification", "voting_classifier", {"components": [
            {"algorithm_id": "gaussian_nb"}] * 9})
    with pytest.raises(ValueError, match="две"):
        catalogue.build("classification", "voting_classifier", {"components": [
            {"algorithm_id": "gaussian_nb"}]})


def test_nested_ensembles_have_bounded_depth_and_reject_probability_mismatch(catalogue):
    base = {"components": [{"algorithm_id": "gaussian_nb"}, {"algorithm_id": "logistic_regression"}]}
    nested = {"components": [{"algorithm_id": "voting_classifier", "params": base},
                              {"algorithm_id": "gaussian_nb"}]}
    estimator = catalogue.build("classification", "voting_classifier", nested)
    assert estimator.estimators[0][1].estimators[0][1].__class__.__name__ == "GaussianNB"
    too_deep = base
    for _ in range(3):
        too_deep = {"components": [{"algorithm_id": "voting_classifier", "params": too_deep},
                                    {"algorithm_id": "gaussian_nb"}]}
    with pytest.raises(ValueError, match="три уровня"):
        catalogue.build("classification", "voting_classifier", too_deep)
    hard_child = {"components": [{"algorithm_id": "voting_classifier", "params": {"voting": "hard"}},
                                  {"algorithm_id": "gaussian_nb"}], "voting": "soft"}
    with pytest.raises(ValueError, match="вероятностей"):
        catalogue.build("classification", "voting_classifier", hard_child)


@pytest.mark.parametrize("library,algorithm_id", [
    ("xgboost", "xgboost_ranker"), ("lightgbm", "lightgbm_ranker"), ("catboost", "catboost_ranker")
])
def test_real_rankers_learn_query_grouped_relevance(library, algorithm_id, catalogue):
    spec = catalogue.spec(algorithm_id)
    if not spec["available"]:
        pytest.skip(spec["reason"])
    groups = np.repeat(np.arange(16), 5)
    relevance = np.tile(np.arange(5), 16)
    X = np.column_stack([relevance / 4, np.sin(groups), np.cos(groups)])
    params = {"iterations" if library == "catboost" else "n_estimators": 40}
    estimator = catalogue.build("ranking", algorithm_id, params, seed=4)
    metadata = ({"qid": groups} if library == "xgboost" else
                {"group": [5] * 16} if library == "lightgbm" else {"group_id": groups})
    with threadpool_limits(limits=1):
        estimator.fit(X, relevance, **metadata)
        scores = estimator.predict(X)
    assert scores.shape == (80,)
    assert np.isfinite(scores).all()
    assert np.mean(scores[relevance == 4]) > np.mean(scores[relevance == 0])


def test_missing_native_library_is_explained_and_does_not_break_other_catalogue(monkeypatch):
    from cml_lab.infrastructure.ml import catalogue as module
    original = module.import_module

    def broken(name):
        if name == "lightgbm":
            raise OSError("libomp unavailable")
        return original(name)

    monkeypatch.setattr(module, "import_module", broken)
    cat = AlgorithmCatalogue()
    broken_spec = cat.spec("lightgbm_classifier")
    assert broken_spec["available"] is False
    assert "libomp" in broken_spec["reason"]
    assert cat.spec("hist_gradient_boosting_classifier")["available"] is True
    with pytest.raises(ValueError, match="libomp"):
        cat.build("classification", "lightgbm_classifier")


@pytest.mark.parametrize("seed,n_jobs", [(True, 1), (-1, 1), (2**32, 1), (42, True), (42, 0), (42, 9)])
def test_randomness_and_parallelism_are_bounded(seed, n_jobs, catalogue):
    with pytest.raises(ValueError):
        catalogue.build("regression", "ridge", seed=seed, n_jobs=n_jobs)


def test_stacking_catalogue_only_advertises_tasks_its_adapter_supports(catalogue):
    assert catalogue.spec("stacking_regressor")["tasks"] == ["regression"]
    assert catalogue.spec("stacking_classifier")["tasks"] == ["classification"]
    assert "stacking_regressor" not in {item["id"] for item in catalogue.list("forecasting")}
    assert "stacking_regressor" not in {item["id"] for item in catalogue.list("panel")}
    for task in ("forecasting", "panel"):
        with pytest.raises(ValueError, match="не поддерживает"):
            catalogue.build(task, "stacking_regressor")
