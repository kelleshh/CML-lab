"""Dataset-source contracts grounded in sklearn and OpenML public formats.

The six built-in tables are actually loaded. Network fetches use small Bunch
fixtures at the external boundary; this suite makes no claim about availability
of the OpenML service or about a successful remote download.
"""

from io import BytesIO
import json

import numpy as np
import pandas as pd
import pytest
from sklearn import datasets
from sklearn.utils import Bunch

from linear_lab.datasets import DataService


@pytest.fixture
def service(tmp_path):
    return DataService(tmp_path)


def test_all_six_builtin_tables_preserve_real_measurements_and_task_roles(service):
    # Counts and task roles are described in sklearn's Toy datasets guide.
    expected = {
        "iris": (150, 4, "classification", "tabular"),
        "wine": (178, 13, "classification", "tabular"),
        "breast_cancer": (569, 30, "classification", "tabular"),
        "digits": (1797, 64, "classification", "image"),
        "diabetes": (442, 10, "regression", "tabular"),
        "linnerud": (20, 3, "regression", "tabular"),
    }
    for name, (rows, features, task, modality) in expected.items():
        kwargs = {"as_frame": True, **({"scaled": False} if name == "diabetes" else {})}
        original = getattr(datasets, f"load_{name}")(**kwargs)
        metadata = service.load({"kind": "builtin", "name": name})
        assert (metadata["rows"], metadata["task"], metadata["modality"]) == (rows, task, modality)
        assert len(original.data.columns) == features
        raw = service.rows(metadata["id"], limit=3)["rows"]
        for column in original.data.columns:
            np.testing.assert_allclose([row[column] for row in raw], original.data[column].iloc[:3])
        if task == "classification":
            assert metadata["default_target"] is None
            assert metadata["task_target"] == "class_label"
            assert "class_label" in metadata["excluded_features"]
            assert "class_label" not in metadata["targets"]
            with pytest.raises(ValueError, match="Метки классов"):
                service.resolve(metadata["id"], "class_label")
            measurement = original.data.columns[0]
            bundle = service.resolve(metadata["id"], measurement)
            assert "class_label" not in bundle.feature_names
        else:
            bundle = service.resolve(metadata["id"])
            assert len(bundle.y) == rows
            assert len(bundle.feature_names) == features
            if name == "linnerud":
                assert {"Weight", "Waist", "Pulse"}.issubset(metadata["targets"])
                assert {"Weight", "Waist", "Pulse"}.isdisjoint(bundle.feature_names)


def test_catalogue_lists_every_installed_loader_and_blocks_unsupported_adapters(service, monkeypatch):
    installed = {
        name for name in dir(datasets)
        if name.startswith(("load_", "fetch_", "make_")) and callable(getattr(datasets, name))
    }
    catalogue = {entry["name"]: entry for entry in service.catalogue()}
    assert installed <= catalogue.keys()
    assert all(catalogue[name]["task"] in {"regression", "classification", "clustering", "time_series", "other"} for name in installed)
    unsupported = [catalogue[name] for name in installed if not catalogue[name]["supported"]]
    assert {"fetch_20newsgroups", "fetch_lfw_people", "load_svmlight_file"} <= {entry["name"] for entry in unsupported}

    def unexpected_call(*args, **kwargs):
        raise AssertionError("An unsupported source must be rejected before reading files or using the network")

    for entry in unsupported:
        assert entry["reason"]
        monkeypatch.setattr(datasets, entry["name"], unexpected_call)
        with pytest.raises(ValueError):
            service.load({"kind": entry["kind"], "name": entry["name"]})
    assert service.list_library()["total"] == 0


def test_task_and_modality_agree_between_source_card_and_persisted_generator(service):
    catalogue = {entry["name"]: entry for entry in service.catalogue()}
    for name in ("make_moons", "make_circles", "make_blobs", "make_low_rank_matrix"):
        metadata = service.load({"kind": "synthetic", "name": name, "params": {"n_samples": 24, "n_features": 3}})
        card = catalogue[name]
        assert (metadata["task"], metadata["tasks"], metadata["modality"]) == (card["task"], card["tasks"], card["modality"])
        for task in card["tasks"]:
            assert metadata["id"] in {item["id"] for item in service.list_library(task=task)["items"]}
    for name in ("make_spd_matrix", "make_sparse_spd_matrix", "make_sparse_coded_signal"):
        assert catalogue[name]["task"] == "other"
        assert catalogue[name]["modality"] == "matrix"
    for name in ("make_s_curve", "make_swiss_roll"):
        assert catalogue[name]["task"] == "other"
    for name in ("fetch_lfw_people", "fetch_lfw_pairs", "fetch_olivetti_faces"):
        assert catalogue[name]["task"] == "classification"
        assert catalogue[name]["modality"] == "image"


def test_csv_and_xlsx_import_preserve_mixed_values_and_allow_explicit_task_labels(service):
    table = pd.DataFrame({"area": [10.0, 20.0, 30.0, 40.0], "city": ["Москва", "Казань", None, "Москва"], "class": ["A", "B", "A", "B"], "y": [100, 200, 300, 400]})
    excel = BytesIO()
    table.to_excel(excel, index=False)
    payloads = [(table.to_csv(index=False).encode("utf-8"), "mixed.csv"), (excel.getvalue(), "mixed.xlsx")]
    for data, filename in payloads:
        metadata = service.import_bytes(data, filename)
        assert metadata["source"] == "upload"
        assert metadata["default_target"] == "y"
        assert metadata["task"] == "regression"
        assert metadata["stats"]["missing_cells"] == 1
        classification = service.update_metadata(metadata["id"], {"task": "classification", "task_target": "class", "default_target": None})
        assert classification["task"] == "classification"
        assert classification["task_target"] == "class"
        assert classification["default_target"] is None
        raw = service.explore(classification["id"], x="area", y="class", color="class")
        assert raw["points"]["color_kind"] == "categorical"
        assert raw["points"]["color_categories"] == ["A", "B"]
        assert raw["data"]["class"] == ["A", "B", "A", "B"]
        assert service.describe(metadata["id"])["task"] == "regression"


def test_openml_id_and_name_version_preserve_source_license_and_categorical_target(service, monkeypatch):
    calls = []

    def fetch_openml(*, data_id=None, name=None, version="active", target_column="default-target", as_frame=True, data_home=None, n_retries=3, delay=1.0, parser="auto"):
        calls.append({"data_id": data_id, "name": name, "version": version, "target_column": target_column, "as_frame": as_frame})
        return Bunch(data=pd.DataFrame({"measurement": np.arange(18.0), "city": pd.Categorical(["A", "B"] * 9)}), target=pd.Series(pd.Categorical(["0", "1"] * 9), name="diagnosis"), details={"id": "900", "name": "audit-mixed", "version": "2", "licence": "CC0", "url": "https://www.openml.org/d/900"}, DESCR="Small fake boundary response")

    monkeypatch.setattr(datasets, "fetch_openml", fetch_openml)
    by_id = service.load({"kind": "openml", "params": {"data_id": 900, "sample_limit": 12}})
    by_name = service.load({"kind": "openml", "params": {"name": "audit-mixed", "version": 2, "sample_limit": 12}})
    assert calls == [
        {"data_id": 900, "name": None, "version": "active", "target_column": "default-target", "as_frame": True},
        {"data_id": None, "name": "audit-mixed", "version": 2, "target_column": "default-target", "as_frame": True},
    ]
    for metadata in (by_id, by_name):
        assert metadata["source"] == "openml"
        assert metadata["openml"]["licence"] == "CC0"
        assert metadata["openml"]["version"] == "2"
        assert metadata["task"] == "classification"
        assert metadata["default_target"] is None
        assert metadata["rows"] == 12 and metadata["original_rows"] == 18
        assert metadata["task_target"] == "class_label"
    # v1 stored OpenML under the generic sklearn source. Its existing cells and
    # metadata sidecar must still be discoverable through the new OpenML filter.
    dataset_file = service.root / f"{by_id['id']}.json"
    saved = json.loads(dataset_file.read_text())
    saved["metadata"]["source"] = "sklearn"
    dataset_file.write_text(json.dumps(saved))
    sidecar = service.root / f"{by_id['id']}.meta.json"
    summary = json.loads(sidecar.read_text())
    summary["source"] = "sklearn"
    sidecar.write_text(json.dumps(summary))
    assert service.describe(by_id["id"])["source"] == "openml"
    assert {item["id"] for item in service.list_library(source="openml")["items"]} == {by_id["id"], by_name["id"]}


def test_openml_without_target_remains_other_and_numeric_target_can_be_selected(service, monkeypatch):
    def fetch_openml(*, data_id=None, target_column="default-target", as_frame=True, data_home=None, n_retries=3, delay=1.0, parser="auto"):
        assert target_column is None
        return Bunch(data=pd.DataFrame({"floor": [1, 2, 3, 4], "price": [110, 220, 330, 440]}), target=None, details={"id": "901", "name": "without-target", "version": "1"}, DESCR="No declared target")

    monkeypatch.setattr(datasets, "fetch_openml", fetch_openml)
    metadata = service.load({"kind": "openml", "params": {"data_id": 901, "target_column": None}})
    assert metadata["task"] == "other"
    assert metadata["task_target"] is None
    assert metadata["default_target"] is None
    with pytest.raises(ValueError, match="Выберите числовую"):
        service.resolve(metadata["id"])
    declared = service.update_metadata(metadata["id"], {"task": "regression", "default_target": "price"})
    bundle = service.resolve(declared["id"])
    assert bundle.feature_names == ["floor"]
    np.testing.assert_array_equal(bundle.y, [110, 220, 330, 440])
