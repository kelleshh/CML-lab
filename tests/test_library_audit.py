"""Independent HTTP checks for the library and raw-data UX promises.

Expected points and summaries come from short, explicit CSV tables rather than
from another DataService call. No estimator or preprocessing is involved in
the exploration checks.
"""

from __future__ import annotations

import csv
import io
import time

import pytest
from fastapi.testclient import TestClient

from linear_lab.app import create_app


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(tmp_path)) as value:
        yield value


def upload(client, content: str, name="audit.csv"):
    response = client.post(
        "/api/datasets/upload",
        files={"file": (name, content.encode("utf-8"), "text/csv")},
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_library_metadata_snapshot_survives_restart_and_combines_filters(client, tmp_path):
    original = upload(client, "area,city,y\n10,A,100\n20,B,200\n30,A,300\n40,,400\n")
    response = client.patch(f"/api/datasets/{original['id']}/metadata", json={
        "name": "Цены квартир", "description": "Наблюдения по городам",
        "tags": ["ЖИЛЬЁ", "Учебный", " жильё "],
        "task": "regression", "default_target": "area", "task_target": "area",
    })
    assert response.status_code == 200, response.text
    edited = response.json()
    assert edited["id"] != original["id"]
    assert edited["parent_id"] == original["id"]
    assert client.get(f"/api/datasets/{original['id']}").json()["default_target"] == "y"
    with TestClient(create_app(tmp_path)) as reopened:
        result = reopened.get("/api/datasets/library", params={
            "query": "ЖИЛЬЁ", "source": "upload", "task": "regression",
        }).json()
        assert result["total"] == 1
        summary = result["items"][0]
        assert summary["id"] == edited["id"]
        assert summary["tags"] == ["жильё", "учебный"]
        assert summary["stats"]["missing_cells"] == 1
        assert summary["stats"]["missing_fraction"] == pytest.approx(1 / 12)
        assert summary["stats"]["target"] == "area"
        assert summary["stats"]["target_quantiles"]["q50"] == 25
        assert "preview" not in summary
        assert summary["stats"]["memory_bytes"] > 0


def test_explore_returns_original_2d_3d_coordinates_without_training(client):
    metadata = upload(client, "area,floor,y\n10,1,100\n20,2,240\n30,3,290\n40,4,450\n")
    response = client.get(f"/api/datasets/{metadata['id']}/explore", params={
        "x": "area", "y": "floor", "z": "y", "target": "y", "color": "y",
    })
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["points"]["indices"] == [0, 1, 2, 3]
    assert result["points"]["x"] == [10, 20, 30, 40]
    assert result["points"]["y"] == [1, 2, 3, 4]
    assert result["points"]["z"] == [100, 240, 290, 450]
    assert result["points"]["color"] == [100, 240, 290, 450]
    assert result["points"]["color_kind"] == "numeric"
    assert result["axes"]["target"] == "y"
    assert result["profile"]["target_profile"]["quantiles"]["q50"] == 265
    assert client.get("/api/experiments").json() == []
    assert not list(client.app.state.jobs.directory.iterdir())


def test_iris_is_classification_and_categorical_colours_are_not_regression_labels(client):
    response = client.post("/api/datasets/load", json={"kind": "builtin", "name": "iris"})
    assert response.status_code == 200, response.text
    metadata = response.json()
    assert metadata["task"] == "classification"
    assert metadata["task_target"] == "class_label"
    assert metadata["default_target"] is None
    assert "class_label" not in metadata["targets"]
    explored = client.get(f"/api/datasets/{metadata['id']}/explore", params={
        "x": "sepal length (cm)", "y": "petal length (cm)",
        "z": "sepal width (cm)", "color": "class_label",
    }).json()
    assert explored["points"]["x"][0] == 5.1
    assert explored["points"]["y"][0] == 1.4
    assert explored["points"]["z"][0] == 3.5
    assert explored["points"]["color_kind"] == "categorical"
    assert len(explored["points"]["color_categories"]) == 3
    assert [explored["points"]["color_codes"].count(i) for i in range(3)] == [50, 50, 50]
    with pytest.raises(ValueError, match="Метки классов"):
        client.app.state.datasets.resolve(metadata["id"], target="class_label")


def test_one_dataset_can_belong_to_clustering_and_classification(client):
    metadata = client.post("/api/datasets/load", json={
        "kind": "synthetic", "name": "make_blobs", "params": {"n_samples": 30},
    }).json()
    assert metadata["task"] == "clustering"
    for task in ("clustering", "classification"):
        response = client.get("/api/datasets/library", params={"task": task})
        assert response.status_code == 200, response.text
        assert [item["id"] for item in response.json()["items"]] == [metadata["id"]]
    catalogue = {item["name"]: item for item in client.get("/api/catalogue").json()["datasets"]}
    assert catalogue["make_blobs"]["tasks"] == ["clustering", "classification"]
    assert catalogue["load_digits"]["task"] == "classification"
    assert catalogue["load_digits"]["modality"] == "image"


def test_profiles_histograms_keep_missing_and_constant_values_visible(client):
    metadata = upload(client, "constant,city,y\n7,A,10\n7,,20\n7,B,30\n7,A,40\n")
    response = client.get(f"/api/datasets/{metadata['id']}/explore", params={
        "x": "constant", "y": "city", "color": "city",
    })
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["points"]["y"] == ["A", None, "B", "A"]
    assert result["points"]["color_codes"] == [0, None, 1, 0]
    assert result["histograms"]["constant"]["counts"] == [4]
    assert result["histograms"]["constant"]["edges"] == [7, 7]
    assert result["histograms"]["city"]["missing"] == 1
    assert sum(result["histograms"]["city"]["counts"]) == 3
    columns = {item["name"]: item for item in result["profile"]["columns"]}
    assert columns["constant"]["std"] == 0
    assert columns["constant"]["quantiles"]["q50"] == 7
    assert columns["city"]["unique"] == 2
    assert columns["city"]["missing"] == 1
    assert result["correlation"]["values"][0][0] is None


def test_original_group_column_can_be_read_without_selecting_it_as_a_feature(client):
    metadata = upload(client, "x,patient,y\n1,A,3\n2,A,5\n3,B,7\n4,B,9\n")
    service = client.app.state.datasets
    selected = service.resolve(metadata["id"], features=["x"])
    assert selected.feature_names == ["x"]
    groups = service.read_column(metadata["id"], "patient")
    assert groups.tolist() == ["A", "A", "B", "B"]
    groups.iloc[0] = "edited copy"
    assert service.read_column(metadata["id"], "patient").tolist() == ["A", "A", "B", "B"]
    with pytest.raises(ValueError, match="Колонка"):
        service.read_column(metadata["id"], "missing")


def test_csv_roundtrip_preserves_unicode_quotes_missing_and_snapshot_parent(client):
    original = upload(client, 'x,city,y\n1,"Москва, центр",3\n2,"Дом ""А""",5\n3,,7\n4,Казань,9\n')
    response = client.patch(f"/api/datasets/{original['id']}/rows", json={
        "changes": [{"index": 2, "values": {"y": 99}}],
    })
    assert response.status_code == 200, response.text
    child = response.json()
    exported = client.get(f"/api/datasets/{child['id']}/export")
    assert exported.status_code == 200, exported.text
    assert exported.content.startswith(b"\xef\xbb\xbf")
    records = list(csv.DictReader(io.StringIO(exported.content.decode("utf-8-sig"))))
    assert [row["city"] for row in records] == ["Москва, центр", 'Дом "А"', "", "Казань"]
    reimported = upload(client, exported.content.decode("utf-8-sig"), "roundtrip.csv")
    assert [row["y"] for row in reimported["preview"]] == [3, 5, 99, 9]
    assert reimported["preview"][2]["city"] is None
    assert client.delete(f"/api/datasets/{child['id']}").status_code == 200
    assert client.get(f"/api/datasets/{child['id']}").status_code == 404
    assert [row["y"] for row in client.get(f"/api/datasets/{original['id']}").json()["preview"]] == [3, 5, 7, 9]


def test_invalid_metadata_and_coordinates_never_create_partial_datasets(client):
    metadata = upload(client, "x,city,y\n1,A,3\n2,B,5\n3,A,7\n")
    for patch in ({"task": "unknown"}, {"default_target": "city"}, {"task_target": "absent"},
                  {"tags": "not a list"}, {"name": ""}, {"columns": []}):
        response = client.patch(f"/api/datasets/{metadata['id']}/metadata", json=patch)
        assert response.status_code == 422, (patch, response.text)
    for params in ({"x": "absent"}, {"color": "absent"}, {"target": "absent"},
                   {"sample_size": 2001}, {"sample_size": 0}, {"seed": -1}):
        response = client.get(f"/api/datasets/{metadata['id']}/explore", params=params)
        assert response.status_code == 422, (params, response.text)
    assert client.get("/api/datasets/library").json()["total"] == 1
    assert not list(client.app.state.datasets.root.glob("*.tmp"))


def test_identifiers_upload_columns_and_paging_are_bounded(client, tmp_path):
    metadata = upload(client, "x,y\n1,3\n2,5\n3,7\n")
    before = set(client.app.state.datasets.root.iterdir())
    for identifier in ("not-a-dataset", "..%2F..%2Fsecret", "0" * 32):
        response = client.get(f"/api/datasets/{identifier}")
        assert response.status_code in (404, 422), response.text
    for content in ("x,x\n1,3\n2,5\n3,7\n", "x, x \n1,3\n2,5\n3,7\n", ",y\n1,3\n2,5\n3,7\n"):
        response = client.post("/api/datasets/upload", files={"file": ("bad.csv", content)})
        assert response.status_code == 422, response.text
    for params in ({"offset": -1}, {"limit": 501}, {"limit": 0}):
        response = client.get(f"/api/datasets/{metadata['id']}/rows", params=params)
        assert response.status_code == 422, response.text
    assert set(client.app.state.datasets.root.iterdir()) == before
    assert not (tmp_path.parent / "secret").exists()


def test_library_page_does_not_materialize_stored_tables(client, monkeypatch):
    upload(client, "x,y\n1,3\n2,5\n3,7\n", "first.csv")
    upload(client, "x,y\n10,30\n20,50\n30,70\n", "second.csv")

    def forbid_full_table_read(*_args, **_kwargs):
        raise AssertionError("Browsing metadata must not read dataset cells")

    monkeypatch.setattr(client.app.state.datasets, "_read", forbid_full_table_read)
    response = client.get("/api/datasets/library", params={"offset": 1, "limit": 1})
    assert response.status_code == 200, response.text
    assert response.json()["total"] == 2
    assert len(response.json()["items"]) == 1
    assert response.json()["items"][0]["name"] == "first.csv"
    assert "preview" not in response.json()["items"][0]


def test_saved_experiment_protects_its_dataset_until_experiment_is_deleted(client):
    rows = [{"x": index / 10, "y": 2 + 3 * index / 10} for index in range(30)]
    metadata = client.post("/api/datasets/load", json={
        "kind": "custom", "rows": rows, "target": "y", "name": "Known relation",
    }).json()
    started = client.post("/api/jobs", json={
        "dataset_id": metadata["id"], "target": "y", "features": ["x"],
        "model": "ols", "params": {}, "seed": 19,
        "split": {"train": 0.6, "validation": 0.2, "test": 0.2},
        "preprocessing": {"scale": True, "degree": 1, "impute": True},
        "metrics": ["mse"], "regularization_path": False, "cv": 0,
    })
    assert started.status_code == 200, started.text
    identifier = started.json()["id"]
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        state = client.get(f"/api/jobs/{identifier}").json()
        if state["status"] != "running":
            break
        time.sleep(0.03)
    assert state["status"] == "completed", state
    assert state["result"]["metrics"]["validation"]["mse"] < 1e-20
    saved = client.post("/api/experiments", json={"job_id": identifier, "name": "Must remain reproducible"})
    assert saved.status_code == 200, saved.text
    saved_id = saved.json()["id"]
    blocked = client.delete(f"/api/datasets/{metadata['id']}")
    assert blocked.status_code == 422, blocked.text
    assert "эксперимент" in blocked.json()["detail"]
    assert client.get(f"/api/datasets/{metadata['id']}").status_code == 200
    assert client.get(f"/api/experiments/{saved_id}").status_code == 200
    assert client.delete(f"/api/experiments/{saved_id}").status_code == 200
    assert client.delete(f"/api/datasets/{metadata['id']}").status_code == 200
