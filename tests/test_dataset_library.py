"""User-facing dataset library and pre-training exploration contracts."""

from io import BytesIO
import json

import numpy as np
import pandas as pd
import pytest

from linear_lab.datasets import DataService


@pytest.fixture
def service(tmp_path):
    return DataService(tmp_path)


def make_mixed(service):
    return service.import_bytes(b"x,city,y\n1,A,2\n2,B,4\n3,A,6\n4,,8\n", "cities.csv")


def test_library_persists_summary_and_combined_filters(service):
    first = make_mixed(service)
    renamed = service.update_metadata(first["id"], {"name": "Стоимость города", "description": "Учебный пример", "tags": ["ТЕСТ", "города", "тест"]})
    service.load({"kind": "builtin", "name": "iris"})
    reloaded = DataService(service.root)
    result = reloaded.list_library(query="ГОРОДА", task="regression", tags=["тест"], source="upload")
    assert result["total"] == 1
    item = result["items"][0]
    assert item["id"] == renamed["id"]
    assert item["tags"] == ["тест", "города"]
    assert "preview" not in item
    assert item["stats"]["numeric_columns"] == 2
    assert item["stats"]["categorical_columns"] == 1
    assert item["stats"]["missing_cells"] == 1
    assert item["stats"]["missing_fraction"] == pytest.approx(1 / 12)
    assert item["stats"]["target_quantiles"]["q50"] == 5
    assert item["stats"]["memory_bytes"] > 0
    assert reloaded.list_library(task="classification")["total"] == 1
    assert reloaded.list_library(offset=1, limit=1)["total"] == 3
    assert len(reloaded.list_library(offset=1, limit=1)["items"]) == 1


def test_metadata_edit_makes_independent_snapshot_and_new_target(service):
    original = make_mixed(service)
    before = service.rows(original["id"])["rows"]
    child = service.update_metadata(original["id"], {"name": "New", "default_target": "x", "task": "time_series", "task_target": "x"})
    assert child["id"] != original["id"]
    assert child["parent_id"] == original["id"]
    assert service.describe(original["id"])["name"] == "cities.csv"
    assert service.describe(original["id"])["default_target"] == "y"
    assert service.rows(child["id"])["rows"] == before
    assert child["stats"]["target"] == "x"
    assert child["stats"]["target_quantiles"]["q50"] == 2.5
    assert service.list_library(task="time_series")["items"][0]["id"] == child["id"]


def test_listing_new_library_does_not_load_full_tables(service, monkeypatch):
    metadata = make_mixed(service)
    def must_not_load_cells(*args):
        raise AssertionError("Library listing should read metadata sidecars only")
    monkeypatch.setattr(service, "_read", must_not_load_cells)
    assert service.list_library()["items"][0]["id"] == metadata["id"]


def test_class_label_stays_class_and_colours_raw_exploration(service):
    metadata = service.load({"kind": "builtin", "name": "iris"})
    assert metadata["task"] == "classification"
    assert metadata["task_target"] == "class_label"
    assert metadata["default_target"] is None
    explored = service.explore(metadata["id"], x="sepal length (cm)", y="petal length (cm)", z="sepal width (cm)", color="class_label")
    assert explored["points"]["color_kind"] == "categorical"
    assert len(explored["points"]["color_categories"]) == 3
    assert set(explored["points"]["color_codes"]) == {0, 1, 2}
    assert explored["points"]["color"][0].startswith("класс:")
    assert explored["axes"]["target"] == "class_label"
    with pytest.raises(ValueError, match="Метки классов"):
        service.resolve(metadata["id"], "class_label")


def test_task_taxonomy_does_not_confuse_task_with_image_modality(service):
    catalogue = {item["name"]: item for item in service.catalogue()}
    assert catalogue["load_digits"]["task"] == "classification"
    assert catalogue["load_digits"]["modality"] == "image"
    assert catalogue["make_blobs"]["tasks"] == ["clustering", "classification"]
    blobs = service.load({"kind": "synthetic", "name": "make_blobs"})
    assert blobs["task"] == "clustering"
    assert service.list_library(task="classification")["total"] == 1
    overridden = service.update_metadata(blobs["id"], {"task": "classification"})
    assert overridden["task"] == "classification"
    assert overridden["tasks"] == ["classification"]


def test_exploration_sample_keeps_original_indices_and_full_profiles(service):
    metadata = service.load({"kind": "synthetic", "name": "linear", "params": {"n_samples": 2500, "n_features": 3}})
    request = {"x": "x3", "y": "y", "z": "x1", "color": "x2", "sample_size": 117, "seed": 9}
    first = service.explore(metadata["id"], **request)
    second = service.explore(metadata["id"], **request)
    assert first["points"] == second["points"]
    assert first["sampled"] and first["sample_rows"] == 117
    raw = service.rows(metadata["id"], offset=0, limit=500)
    bundle = service.resolve(metadata["id"])
    indices = first["points"]["indices"]
    np.testing.assert_array_equal(first["points"]["x"], bundle.X["x3"].iloc[indices])
    np.testing.assert_array_equal(first["points"]["y"], bundle.y[indices])
    assert first["points"]["color_kind"] == "numeric"
    assert first["profile"]["stats"]["rows"] == 2500
    assert first["profile"]["computed_on"] == "all_rows"
    assert sum(first["histograms"]["y"]["counts"]) == 2500
    corr = first["correlation"]
    matrix = np.asarray(corr["values"])
    expected = pd.concat([bundle.X, pd.Series(bundle.y, name="y")], axis=1)[corr["columns"]].corr()
    np.testing.assert_allclose(matrix, expected, atol=1e-12)
    assert raw["total"] == first["total_rows"]


def test_missing_values_and_constant_features_stay_visible(service):
    metadata = service.import_bytes(b"x,city,y\n1,A,2\n1,,4\n1,B,6\n1,A,8\n", "missing.csv")
    explored = service.explore(metadata["id"], x="x", y="city", color="city")
    assert explored["points"]["y"] == ["A", None, "B", "A"]
    assert explored["points"]["color_codes"] == [0, None, 1, 0]
    assert explored["histograms"]["x"]["counts"] == [4]
    assert explored["histograms"]["city"]["missing"] == 1
    assert explored["correlation"]["values"][0][0] is None
    json.dumps(explored, allow_nan=False)


def test_read_column_does_not_require_feature_selection_and_returns_copy(service):
    metadata = make_mixed(service)
    column = service.read_column(metadata["id"], "city")
    column.iloc[0] = "changed locally"
    assert service.read_column(metadata["id"], "city").iloc[0] == "A"
    with pytest.raises(ValueError, match="Колонка"):
        service.read_column(metadata["id"], "missing")


def test_csv_export_roundtrip_and_delete_only_selected_snapshot(service):
    parent = make_mixed(service)
    child = service.update_rows(parent["id"], [{"index": 2, "values": {"y": 100}}])
    exported = service.export_csv(child["id"])
    assert exported.startswith(b"\xef\xbb\xbf")
    frame = pd.read_csv(BytesIO(exported))
    assert frame["city"].iloc[0] == "A" and pd.isna(frame["city"].iloc[3])
    assert frame["y"].tolist() == [2, 4, 100, 8]
    assert service.delete(child["id"]) == {"id": child["id"], "deleted": True}
    assert service.list_library()["total"] == 1
    assert service.describe(parent["id"])["id"] == parent["id"]
    with pytest.raises(FileNotFoundError):
        service.export_csv(child["id"])


def test_legacy_saved_dataset_enriches_without_overwriting_original(service):
    metadata = make_mixed(service)
    path = service.root / f"{metadata['id']}.json"
    saved = json.loads(path.read_text())
    for key in ["stats", "task", "tasks", "task_label", "task_target", "tags", "created_at"]:
        saved["metadata"].pop(key, None)
    path.write_text(json.dumps(saved))
    (service.root / f"{metadata['id']}.meta.json").unlink()
    original = path.read_bytes()
    item = DataService(service.root).list_library()["items"][0]
    assert item["task"] == "regression"
    assert item["stats"]["rows"] == 4
    assert path.read_bytes() == original


@pytest.mark.parametrize("patch", [{"task": []}, {"task": "invalid"}, {"tags": "x"}, {"tags": [""]}, {"name": ""}, {"task_target": "absent"}, {"default_target": "city"}, {"rows": []}])
def test_bad_library_metadata_never_creates_snapshot(service, patch):
    metadata = make_mixed(service)
    with pytest.raises(ValueError):
        service.update_metadata(metadata["id"], patch)
    assert service.list_library()["total"] == 1


@pytest.mark.parametrize("parameters", [{"sample_size": 2001}, {"x": "absent"}, {"color": ["x"]}, {"seed": -1}])
def test_exploration_rejects_unbounded_or_invalid_requests(service, parameters):
    metadata = make_mixed(service)
    with pytest.raises(ValueError):
        service.explore(metadata["id"], **parameters)
