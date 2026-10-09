"""Regression contracts at data boundaries, not implementation mirroring."""

from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn import datasets
from sklearn.utils import Bunch

from linear_lab.datasets import DataService


@pytest.fixture
def service(tmp_path: Path) -> DataService:
    return DataService(tmp_path)


def test_dataset_survives_new_service_and_preserves_data(service):
    meta = service.load({"kind": "synthetic", "name": "linear", "params": {"seed": 7, "n_samples": 40, "n_features": 3}})
    first = service.resolve(meta["id"])
    second = DataService(service.root).resolve(meta["id"])
    pd.testing.assert_frame_equal(first.X, second.X)
    np.testing.assert_array_equal(first.y, second.y)
    assert second.feature_names == ["x1", "x2", "x3"]
    assert second.meta["preview"][0]["y"] == second.y[0]


def test_reproducible_synthetic_signal_and_noise(service):
    spec = {"kind": "synthetic", "name": "linear", "params": {"noise": 0, "n_features": 3, "scale_spread": 4}}
    one = service.resolve(service.load(spec)["id"])
    two = service.resolve(service.load(spec)["id"])
    np.testing.assert_allclose(one.y, one.X @ one.meta["true_coefficients"] + one.meta["true_intercept"])
    np.testing.assert_array_equal(one.y, two.y)
    np.testing.assert_array_equal(one.X, two.X)


@pytest.mark.parametrize("name", ["linear", "correlated", "sparse", "nonlinear", "heteroscedastic", "positive", "counts", "grouped", "fused"])
def test_all_presets_resolve_to_finite_tabular_regression(service, name):
    meta = service.load({"kind": "synthetic", "name": name, "params": {"n_features": 8, "outliers": 0.1}})
    bundle = service.resolve(meta["id"])
    assert bundle.X.shape == (180, 8)
    assert np.isfinite(bundle.y).all()
    assert len(meta["outlier_indices"]) == 18
    if name == "positive":
        assert (bundle.y > 0).all()
    if name == "counts":
        assert (bundle.y >= 0).all()
        np.testing.assert_array_equal(bundle.y, np.round(bundle.y))


def test_sparsity_and_negative_correlation_validation(service):
    meta = service.load({"kind": "synthetic", "name": "sparse", "params": {"n_features": 10, "sparsity": 0.8}})
    assert np.count_nonzero(meta["true_coefficients"]) == 2
    with pytest.raises(ValueError, match="корреляция"):
        service.load({"kind": "synthetic", "name": "linear", "params": {"n_features": 5, "correlation": -0.5}})


def test_builtin_diabetes_and_multi_target_dataset(service):
    diabetes = service.resolve(service.load({"kind": "builtin", "name": "diabetes"})["id"])
    assert diabetes.X.shape == (442, 10)
    np.testing.assert_array_equal(diabetes.y, datasets.load_diabetes().target)
    linnerud = service.load({"kind": "builtin", "name": "linnerud"})
    assert {"Weight", "Waist", "Pulse"}.issubset(linnerud["targets"])
    assert service.resolve(linnerud["id"], "Pulse", ["Chins", "Situps", "Jumps"]).X.shape == (20, 3)


def test_classification_labels_never_silently_become_regression_target(service):
    meta = service.load({"kind": "builtin", "name": "iris"})
    assert meta["default_target"] is None
    assert "class_label" not in meta["targets"]
    with pytest.raises(ValueError, match="Выберите"):
        service.resolve(meta["id"])
    with pytest.raises(ValueError, match="Метки классов"):
        service.resolve(meta["id"], "class_label")
    selected = service.resolve(meta["id"], "sepal length (cm)")
    assert "class_label" not in selected.feature_names
    assert selected.X.shape == (150, 3)


def test_catalogue_contains_installed_sklearn_surface(service):
    catalogue = service.catalogue()
    available = {entry["name"] for entry in catalogue}
    expected = {name for name in dir(datasets) if name.startswith(("load_", "fetch_", "make_")) and callable(getattr(datasets, name))}
    assert expected <= available
    texts = next(entry for entry in catalogue if entry["name"] == "fetch_20newsgroups")
    assert not texts["supported"]
    assert texts["reason"]


def test_csv_import_keeps_categorical_features_and_numeric_target(service):
    meta = service.import_bytes(b"x,city,y\n1,A,4\n2,B,6\n3,A,8\n4,,10\n", "table.csv")
    bundle = DataService(service.root).resolve(meta["id"])
    assert bundle.feature_names == ["x", "city"]
    assert bundle.X["city"].iloc[0] == "A"
    assert pd.isna(bundle.X["city"].iloc[3])
    np.testing.assert_array_equal(bundle.y, [4, 6, 8, 10])


def test_russian_decimal_comma_csv_and_excel(service):
    data = "признак;цель\n1,5;3,0\n2,5;5,0\n3,5;7,0\n".encode("utf-8-sig")
    meta = service.import_bytes(data, "table.csv")
    np.testing.assert_array_equal(service.resolve(meta["id"]).y, [3, 5, 7])
    file = BytesIO()
    pd.DataFrame({"x": [1, 2, 3], "y": [5, 7, 9]}).to_excel(file, index=False)
    excel = service.import_bytes(file.getvalue(), "example.xlsx")
    assert service.resolve(excel["id"]).X.shape == (3, 1)


@pytest.mark.parametrize("payload,filename", [(b"", "x.csv"), (b"not a table", "x.csv"), (b"x,x\n1,2\n3,4\n5,6", "x.csv"), (b"x,y\n1,2", "x.csv"), (b"x,y\n1,2\n3,4\n5,6", "x.exe"), (b"x,y\n1,inf\n2,3\n4,5", "x.csv"), (b"x,y\nA,B\nC,D\nE,F", "x.csv"), (b"not excel", "x.xlsx")])
def test_malformed_uploads_fail_with_user_facing_error(service, payload, filename):
    with pytest.raises(ValueError):
        service.import_bytes(payload, filename)


def test_target_validation_prevents_leakage_and_missing_labels(service):
    meta = service.import_bytes(b"x,y\n1,2\n2,\n3,4\n", "table.csv")
    with pytest.raises(ValueError, match="пропуски"):
        service.resolve(meta["id"])
    other = service.load({"kind": "synthetic", "name": "linear"})
    with pytest.raises(ValueError, match="утечка"):
        service.resolve(other["id"], features=["y"])
    with pytest.raises(ValueError, match="несколько"):
        service.resolve(other["id"], features=["x1", "x1"])
    with pytest.raises(ValueError, match="нет признаков"):
        service.resolve(other["id"], features=["absent"])


def test_custom_edits_persist_and_reject_nested_cells(service):
    meta = service.load({"kind": "custom", "name": "edited", "rows": [{"x": 0, "y": 1}, {"x": 1, "y": 4}, {"x": 2, "y": 7}], "target": "y"})
    np.testing.assert_array_equal(service.resolve(meta["id"]).y, [1, 4, 7])
    with pytest.raises(ValueError, match="вложенные"):
        service.load({"kind": "custom", "rows": [{"x": [1], "y": 1}, {"x": 2, "y": 2}, {"x": 3, "y": 3}], "target": "y"})


def test_large_category_feature_is_reported_before_training(service):
    rows = [{"identifier": f"id-{index}", "x": index, "y": index * 3} for index in range(520)]
    meta = service.load({"kind": "custom", "rows": rows, "target": "y"})
    assert meta["warnings"]
    with pytest.raises(ValueError, match="категорий"):
        service.resolve(meta["id"])
    assert service.resolve(meta["id"], features=["x"]).X.shape == (520, 1)


@pytest.mark.parametrize("identifier", ["../../secret", "/tmp/secret", "abc", "0" * 32 + "/.."])
def test_client_identifiers_cannot_address_files(service, identifier):
    with pytest.raises(ValueError, match="идентификатор"):
        service.describe(identifier)


def test_fetch_failure_explains_network_and_manual_upload(service, monkeypatch):
    def offline(**kwargs):
        raise OSError("offline")
    monkeypatch.setattr(datasets, "fetch_california_housing", offline)
    with pytest.raises(ValueError, match="CSV вручную"):
        service.load({"kind": "fetch", "name": "california_housing"})


def test_openml_classification_and_sampling(service, monkeypatch):
    def fetch_openml(*, data_id, as_frame=True, data_home=None, n_retries=1, delay=0.1, parser="auto"):
        assert data_id == 123
        return Bunch(data=pd.DataFrame({"x": np.arange(20), "group": pd.Categorical(["a", "b"] * 10)}), target=pd.Series(pd.Categorical(["0", "1"] * 10), name="label"), details={"name": "demo", "id": "123"}, DESCR="demo")
    monkeypatch.setattr(datasets, "fetch_openml", fetch_openml)
    meta = service.load({"kind": "openml", "params": {"data_id": 123, "sample_limit": 12}})
    assert meta["task"] == "classification"
    assert meta["default_target"] is None
    assert meta["rows"] == 12
    assert meta["original_rows"] == 20
    assert "class_label" not in meta["targets"]
    assert service.resolve(meta["id"], "x", ["group"]).X.shape == (12, 1)


@pytest.mark.parametrize("name", ["make_regression", "make_friedman1", "make_friedman2", "make_friedman3", "make_sparse_uncorrelated", "make_classification", "make_blobs", "make_circles", "make_moons", "make_gaussian_quantiles", "make_low_rank_matrix"])
def test_supported_sklearn_generators_can_create_table(service, name):
    meta = service.load({"kind": "synthetic", "name": name, "params": {"n_features": 5, "n_samples": 40}})
    assert meta["rows"] == 40
    assert meta["targets"]
    if name == "make_regression":
        assert service.resolve(meta["id"]).X.shape == (40, 5)


def test_original_rows_are_paginated_without_dropping_columns(service):
    meta = service.load({"kind": "builtin", "name": "iris"})
    page = service.rows(meta["id"], offset=100, limit=20)
    assert page["total"] == 150
    assert page["offset"] == 100
    assert len(page["rows"]) == 20
    assert page["rows"][0]["class_label"] == "класс: 2"
    assert "class_label" in {column["name"] for column in page["columns"]}
    assert len(service.rows(meta["id"], offset=150)["rows"]) == 0
    with pytest.raises(ValueError):
        service.rows(meta["id"], limit=501)
    with pytest.raises(ValueError):
        service.rows(meta["id"], offset=-1)


def test_patch_preserves_unedited_rows_original_version_and_metadata(service):
    old = service.load({"kind": "builtin", "name": "iris"})
    original = service.rows(old["id"], limit=150)["rows"]
    modified = service.update_rows(old["id"], [{"index": 120, "values": {"sepal length (cm)": 99}}])
    current = service.rows(modified["id"], limit=150)["rows"]
    assert modified["rows"] == 150
    assert modified["parent_id"] == old["id"]
    assert modified["source"] == old["source"]
    assert modified["default_target"] is None
    assert "class_label" not in modified["targets"]
    assert original[0] == current[0]
    assert current[120]["sepal length (cm)"] == 99
    assert original[121:] == current[121:]
    assert service.rows(old["id"], offset=120, limit=1)["rows"][0]["sepal length (cm)"] != 99


def test_delete_and_append_indices_refer_to_original_dataset(service):
    old = service.load({"kind": "custom", "rows": [{"x": index, "y": index * 2} for index in range(10)], "target": "y"})
    new = service.update_rows(old["id"], [{"index": 5, "values": {"y": 99}}], additions=[{"x": 10, "y": 20}], deletes=[0, 3])
    bundle = service.resolve(new["id"])
    assert len(bundle.y) == 9
    assert bundle.target_name == "y"
    assert bundle.X["x"].iloc[3] == 5
    assert bundle.y[3] == 99
    assert bundle.y[-1] == 20
    assert service.describe(old["id"])["rows"] == 10


@pytest.mark.parametrize("changes,additions,deletes", [
    ([{"index": -1, "values": {"y": 1}}], [], []),
    ([{"index": 999, "values": {"y": 1}}], [], []),
    ([{"index": True, "values": {"y": 1}}], [], []),
    ([{"index": 0, "values": {"wrong": 1}}], [], []),
    ([{"index": 0, "values": {"y": "invalid target"}}], [], []),
    ([{"index": 0, "values": {"y": 3}}], [], [0]),
    ([], [{"x1": 3}], []),
    ([], [], [1000]),
    ([], [], [1, 1]),
    ([], [], []),
])
def test_invalid_patches_cannot_mutate_original(service, changes, additions, deletes):
    old = service.load({"kind": "synthetic", "name": "linear"})
    before = service.resolve(old["id"]).y.copy()
    with pytest.raises(ValueError):
        service.update_rows(old["id"], changes, additions, deletes)
    np.testing.assert_array_equal(before, service.resolve(old["id"]).y)


def test_patch_volume_limits(service):
    old = service.load({"kind": "synthetic", "name": "linear"})
    with pytest.raises(ValueError, match="500"):
        service.update_rows(old["id"], [{"index": 0, "values": {"y": 1}}] * 501)
    with pytest.raises(ValueError, match="100"):
        service.update_rows(old["id"], [], [{"x1": 1, "x2": 2, "y": 3}] * 101)
