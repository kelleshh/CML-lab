"""Task-aware original tables, public boundaries and immutable dataset CRUD."""

import numpy as np
import pandas as pd
import pytest

from cml_lab.infrastructure.datasets import LegacyDataGateway
from cml_lab.shared.domain import ValidationError


@pytest.fixture
def gateway(tmp_path):
    return LegacyDataGateway(tmp_path / "datasets")


@pytest.mark.parametrize("name,task,target,roles", [
    ("linear", "regression", "y", {}),
    ("classification", "classification", "target", {}),
    ("clustering", "clustering", "class_label", {"reference_target": "class_label"}),
    ("ranking", "ranking", "relevance", {"query_column": "query_id"}),
    ("forecasting", "forecasting", "y", {"time_column": "time"}),
    ("panel", "panel", "y", {"time_column": "time", "entity_column": "entity"}),
    ("anomaly", "anomaly", "is_anomaly", {"reference_target": "is_anomaly"}),
    ("reduction", "reduction", None, {}),
])
def test_generators_cover_eight_tasks_reproducibly(gateway, name, task, target, roles):
    spec = {"kind": "synthetic", "name": name, "params": {"n_samples": 180, "seed": 7}}
    first = gateway.load(spec)
    second = gateway.load(spec)
    assert first["id"] != second["id"]
    assert first["task"] == task and first["task_target"] == target
    if name != "linear":
        assert first["roles"] == roles
    pd.testing.assert_frame_equal(gateway.frame(first["id"]), gateway.frame(second["id"]))
    assert gateway.list_library(task=task)["total"] == 2
    assert {first["id"], second["id"]} == {item["id"] for item in gateway.list_library(task=task)["items"]}


def test_classification_keeps_categories_and_does_not_select_them_as_regression(gateway):
    meta = gateway.load({"kind": "synthetic", "name": "classification"})
    frame = gateway.frame(meta["id"])
    assert set(frame.target.unique()) == {"класс 0", "класс 1"}
    assert meta["default_target"] is None and "target" not in meta["targets"]
    assert "target" not in meta["feature_names"]
    explored = gateway.explore(meta["id"], x="x1", y="x2", color="target", target="target")
    assert explored["points"]["color_kind"] == "categorical"
    assert explored["points"]["color_categories"] == ["класс 0", "класс 1"]


def test_categorical_only_user_dataset_is_supported_and_read_is_detached(gateway):
    rows = [{"colour": "red", "target": "yes"}, {"colour": "blue", "target": "no"}, {"colour": "red", "target": "yes"}]
    meta = gateway.load({"kind": "custom", "name": "Категории", "task": "classification", "target": "target", "rows": rows})
    assert meta["task_target"] == "target" and meta["default_target"] is None
    assert meta["targets"] == []
    frame = gateway.frame(meta["id"])
    assert frame.to_dict(orient="records") == rows
    frame.iloc[0, 0] = "changed"
    assert gateway.frame(meta["id"]).iloc[0, 0] == "red"
    assert "target" not in meta["feature_names"]


def test_gateway_calls_public_frame_boundary_once(gateway, monkeypatch):
    meta = gateway.load({"name": "linear"})
    calls = []
    original = gateway.legacy.read_frame
    def public_read(identifier):
        calls.append(identifier)
        return original(identifier)
    monkeypatch.setattr(gateway.legacy, "read_frame", public_read)
    frame = gateway.frame(meta["id"])
    assert len(frame) == meta["rows"] and calls == [meta["id"]]


def test_ranking_groups_have_comparable_integer_relevance(gateway):
    meta = gateway.load({"name": "ranking", "params": {"n_samples": 240, "n_queries": 12, "seed": 42}})
    frame = gateway.frame(meta["id"])
    assert frame.query_id.nunique() == 12
    assert np.array_equal(frame.relevance, frame.relevance.astype(int))
    assert set(frame.relevance.unique()) == {0, 1, 2, 3, 4}
    assert set(frame.groupby("query_id").relevance.nunique()) == {5}
    assert "query_id" not in meta["feature_names"]


def test_panel_time_is_unique_per_entity_and_n_rows_is_total(gateway):
    meta = gateway.load({"name": "panel", "params": {"n_samples": 181, "n_entities": 5, "seed": 1}})
    frame = gateway.frame(meta["id"])
    assert len(frame) == 181 and frame.entity.nunique() == 5
    assert not frame.duplicated(["entity", "time"]).any()
    assert all(pd.to_datetime(part.time).is_monotonic_increasing for _, part in frame.groupby("entity"))
    assert "entity" not in meta["feature_names"] and "time" not in meta["feature_names"]


def test_rows_and_metadata_updates_leave_parent_snapshot_unchanged(gateway):
    first = gateway.load({"name": "ranking"})
    original = gateway.frame(first["id"])
    second = gateway.update_metadata(first["id"], {"name": "Мой рейтинг", "task": "ranking", "task_target": "relevance", "tags": ["Тест"]})
    third = gateway.update_rows(second["id"], [{"index": 0, "values": {"relevance": 4}}])
    assert first["id"] != second["id"] != third["id"]
    assert gateway.describe(first["id"])["name"] == first["name"]
    assert gateway.describe(second["id"])["parent_id"] == first["id"]
    assert gateway.describe(third["id"])["parent_id"] == second["id"]
    pd.testing.assert_frame_equal(gateway.frame(first["id"]), original)
    assert gateway.frame(third["id"]).iloc[0].relevance == 4
    gateway.delete(third["id"])
    assert gateway.describe(second["id"])["id"] == second["id"]


def test_sklearn_alias_and_old_time_series_labels(gateway):
    iris = gateway.load({"kind": "sklearn", "name": "load_iris"})
    assert iris["task"] == "classification" and len(gateway.frame(iris["id"])) == 150
    moons = gateway.load({"kind": "sklearn", "name": "make_moons"})
    assert "classification" in moons["tasks"]
    old = gateway.legacy.load({"name": "linear"})
    labelled = gateway.legacy.update_metadata(old["id"], {"task": "time_series"})
    assert gateway.describe(labelled["id"])["task"] == "forecasting"
    assert labelled["id"] in {item["id"] for item in gateway.list_library(task="forecasting")["items"]}


@pytest.mark.parametrize("kwargs", [{"offset": True}, {"offset": -1}, {"limit": 0}, {"limit": 501}, {"task": "invalid"}, {"tags": "bad"}])
def test_library_filters_and_pagination_are_bounded(gateway, kwargs):
    with pytest.raises(ValueError):
        gateway.list_library(**kwargs)


@pytest.mark.parametrize("spec", [
    {"name": "classification", "params": {"seed": True}},
    {"name": "classification", "params": {"n_classes": 10, "n_features": 2}},
    {"name": "ranking", "params": {"n_queries": 1}},
    {"name": "panel", "params": {"n_entities": 100, "n_samples": 12}},
    {"name": "panel", "params": {"noise": float("nan")}},
    {"kind": "custom", "task": [], "rows": [{"x": 1, "y": 2}] * 3},
])
def test_generator_requests_fail_with_explanations(gateway, spec):
    with pytest.raises(ValueError):
        gateway.load(spec)


@pytest.mark.parametrize("metadata", [
    {"excluded_features": None},
    {"excluded_targets": [[]]},
    {"excluded_features": ["absent"]},
    {"excluded_targets": ["absent"]},
    {"excluded_features": ["x", "x"]},
    {"warnings": None},
    {"warnings": [{}]},
    {"roles": []},
    {"roles": {"time_column": "absent"}},
    {"roles": {"query_column": []}},
    {"task_target": []},
    {"source": []},
    {"generator": []},
    {"description": {}},
    {"params": []},
])
def test_invalid_custom_metadata_returns_http_422_without_creating_snapshot(tmp_path, metadata):
    from fastapi.testclient import TestClient
    from cml_lab.presentation.http.api import create_app
    rows = [{"x": 1., "y": 2.}, {"x": 2., "y": 4.}, {"x": 3., "y": 6.}]
    with TestClient(create_app(tmp_path), raise_server_exceptions=False) as client:
        before = client.get("/api/datasets/library").json()["total"]
        response = client.post("/api/datasets/load", json={"kind": "custom", "rows": rows, "metadata": metadata})
        assert response.status_code == 422
        assert response.json()["detail"]
        assert client.get("/api/datasets/library").json()["total"] == before


def test_valid_custom_metadata_survives_detached_reads_and_new_snapshot(gateway):
    rows = [{"x": i, "query": f"q{i % 2}", "y": i % 5} for i in range(12)]
    details = {"task": "ranking", "tasks": ["ranking"], "task_target": "y", "roles": {"query_column": "query"}, "excluded_features": ["query"], "excluded_targets": ["query"], "warnings": ["Учебные данные"], "params": {"seed": 42}, "source": "custom"}
    first = gateway.load({"kind": "custom", "rows": rows, "metadata": details})
    second = gateway.update_metadata(first["id"], {"name": "Новая версия"})
    assert first["id"] != second["id"]
    for key in details:
        assert second[key] == first[key] == details[key]
    details["roles"]["query_column"] = "changed"
    assert gateway.describe(first["id"])["roles"] == {"query_column": "query"}
    pd.testing.assert_frame_equal(gateway.frame(first["id"]), gateway.frame(second["id"]))


def test_import_numeric_string_policy_is_explicitly_preserved(gateway):
    metadata = gateway.import_bytes(b"id,label,mixed\n001,01,01\n002,1,1\n010,10,a\n", "codes.csv")
    frame = gateway.frame(metadata["id"])
    assert frame["id"].tolist() == [1, 2, 10]
    assert frame["label"].tolist() == [1, 1, 10]
    assert frame["mixed"].tolist() == ["01", "1", "a"]
