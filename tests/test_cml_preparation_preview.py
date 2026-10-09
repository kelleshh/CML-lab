"""Fresh previews share the real split and numerical compiler used by runs."""

import json

import numpy as np
import pandas as pd
import pytest

from cml_lab.contexts.data.preparation_application import PreparationApplication
from cml_lab.infrastructure.preparation_preview import PreparationPreviewGateway
from cml_lab.infrastructure.ml.splitting import holdout


class TableGateway:
    def __init__(self, frame):
        self.table = frame.copy()

    def frame(self, dataset_id):
        assert dataset_id == "data"
        return self.table.copy()


def application(frame):
    return PreparationApplication(PreparationPreviewGateway(TableGateway(frame), None))


def request(task="regression", **kwargs):
    return {"dataset_id": "data", "task": task, "target": "y", "features": ["x"], "seed": 42,
            "preprocessing": {"steps": []}, "split": {"shuffle": False}, **kwargs}


def test_supervised_preview_fits_only_train_and_preserves_row_pairing():
    frame = pd.DataFrame({"x": np.arange(40, dtype=float), "y": np.arange(40, dtype=float) * 3})
    spec = request()
    expected = holdout(spec, len(frame))["train"]
    baseline = application(frame).preview(spec)
    modified = frame.copy()
    modified.loc[24:, "x"] = 1e6
    after = application(modified).preview(spec)
    assert baseline["fitted_on"] == "train"
    assert baseline["train_rows"] == 24
    assert baseline["indices"] == expected.tolist()
    assert baseline["rows"] == after["rows"]
    assert baseline["means"] == after["means"]
    for row, index, y in zip(baseline["original_rows"], baseline["indices"], baseline["target"]["values"]):
        assert row["x"] == frame.iloc[index]["x"]
        assert y == frame.iloc[index]["y"]
    assert sum(baseline["histograms"]["before"]["x"]["counts"]) == 24
    assert len(baseline["features"]) == len(baseline["rows"][0])
    json.dumps(baseline, allow_nan=False)


def test_classification_preview_keeps_original_labels_after_train_resampling():
    frame = pd.DataFrame({"x": np.arange(80, dtype=float), "y": ["обычный"] * 60 + ["редкий"] * 20})
    spec = request("classification", split={"shuffle": True}, preprocessing={"steps": [], "resampling": {"method": "random_over"}})
    result = application(frame).preview(spec)
    assert set(result["target"]["values"]) == {"обычный", "редкий"}
    assert set(result["sampling"]["target_after"]) == {"обычный", "редкий"}
    counts = pd.Series(result["sampling"]["target_after"]).value_counts()
    assert counts.iloc[0] == counts.iloc[1]
    assert result["sampling"]["summary"]["before"] == result["train_rows"]
    assert len(result["rows"]) == result["train_rows"]
    assert result["sampling"]["summary"]["after"] > result["train_rows"]


def test_unsupervised_preview_scope_is_explicit_full_working_dataset():
    frame = pd.DataFrame({"x": np.arange(24, dtype=float), "y": np.arange(24)})
    result = application(frame).preview(request("clustering", target=None))
    assert result["fitted_on"] == "working_dataset"
    assert result["train_rows"] == 24
    assert result["target"]["values"] == []
    assert "не оценка" in result["note"]


def test_temporal_preview_uses_purged_train_and_original_indices():
    frame = pd.DataFrame({"time": pd.date_range("2020-01-01", periods=50, freq="D"), "y": np.arange(50, dtype=float)})
    result = application(frame).preview(request("forecasting", features=[], roles={"time_column": "time"}, temporal={"lags": [1], "rolling_windows": [], "horizon": 3}))
    assert result["fitted_on"] == "train"
    assert result["original_features"] == ["y__lag_1"]
    assert result["indices"][0] == 1
    assert result["train_rows"] < int(47 * .6)
    assert result["original_rows"][0]["y__lag_1"] == 0.
    assert result["target"]["values"][0] == 3.


def test_sparse_preview_caps_display_and_records_full_matrix_width():
    frame = pd.DataFrame({"text": ["кошка спит" if i % 2 else "собака ест" for i in range(40)], "y": np.arange(40, dtype=float)})
    spec = request(features=["text"], preprocessing={"steps": [{"adapter_id": "text.hash", "columns": ["text"], "params": {"n_features": 300}}]})
    result = application(frame).preview(spec)
    assert result["feature_count"] == 300
    assert len(result["all_features"]) == 300
    assert len(result["features"]) == 100
    assert len(result["rows"][0]) == 100
    assert "100 из 300" in result["note"]


def test_preview_missing_numeric_and_categorical_values_serialize_as_null():
    frame = pd.DataFrame({"x": [np.nan if i % 3 else float(i) for i in range(40)], "category": pd.Series([pd.NA if i % 2 else "a" for i in range(40)], dtype="string"), "y": np.arange(40, dtype=float)})
    result = application(frame).preview(request(features=["x", "category"]))
    json.dumps(result, allow_nan=False)
    assert any(row["category"] is None for row in result["original_rows"])


def test_group_target_encoding_preview_is_blocked_before_it_can_leak():
    frame = pd.DataFrame({"x": [f"cat_{i}" for i in range(60)], "group": [f"g_{i % 10}" for i in range(60)], "y": np.arange(60, dtype=float)})
    spec = request(validation={"strategy": "group_kfold", "group_column": "group"}, preprocessing={"steps": [{"adapter_id": "categorical.target", "params": {}}]})
    with pytest.raises(ValueError, match="crossfit"):
        application(frame).preview(spec)


def test_stage_catalogue_filters_tasks_and_rejects_invalid_stored_recipe():
    app = application(pd.DataFrame({"x": np.arange(12), "y": np.arange(12)}))
    stages = app.stages("clustering")
    assert stages["total"] == len(stages["items"])
    assert not any(stage["id"] == "categorical.target" for stage in stages["items"])
    assert [sampler["id"] for sampler in stages["resampling"]] == ["none"]
    with pytest.raises(ValueError, match="неизвестные настройки"):
        app.validate_config({"steps": [{"adapter_id": "numeric.scale", "params": {"surprise": 1}}]})
    with pytest.raises(ValueError, match="сохранённый набор"):
        app.preview({"task": "regression"})
