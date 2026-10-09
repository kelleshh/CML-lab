"""Raw dataset analyses: independent counts, boundaries and bounded scope."""

import json

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
import pytest

from cml_lab.infrastructure.analysis import DatasetAnalysis, GRAPH_TYPES
from cml_lab.infrastructure.datasets import LegacyDataGateway
from cml_lab.presentation.http.analysis_routes import analysis_routes


@pytest.fixture
def laboratory(tmp_path):
    data = LegacyDataGateway(tmp_path)
    rows = [{"x": None if index % 7 == 0 else index / 10, "y": index % 11 + index / 2,
             "z": index % 5, "time": str(pd.Timestamp("2024-01-01") + pd.Timedelta(days=index)),
             "group": "A" if index % 3 else "B", "category": f"c{index % 8}",
             "empty": None, "constant": 1, "flag": bool(index % 2)} for index in range(96)]
    dataset = data.load({"kind": "custom", "name": "EDA", "rows": rows, "target": "y"})
    return data, dataset, DatasetAnalysis(data)


COLUMNS = {
    "histogram": ["x"], "box": ["x"], "violin": ["x"], "ecdf": ["x"], "qq": ["x"], "kde": ["x"],
    "scatter": ["x", "y"], "scatter3d": ["x", "y", "z"], "histogram2d": ["x", "y"],
    "correlation": ["x", "y", "constant"], "missing_bar": ["x", "empty"], "missing_matrix": ["x", "empty"],
    "category_counts": ["category"], "category_crosstab": ["group", "category"], "group_box": ["group", "y"],
    "group_mean": ["group", "y"], "time_line": ["time", "y"], "rolling_mean": ["time", "y"],
    "autocorrelation": ["time", "y"], "outliers": ["x"],
}


@pytest.mark.parametrize("kind", list(COLUMNS))
def test_every_catalogued_graph_uses_real_data_without_modifying_snapshot(laboratory, kind):
    data, dataset, analysis = laboratory
    original = data.frame(dataset["id"])
    result = analysis.analyze({"dataset_id": dataset["id"], "kind": kind, "columns": COLUMNS[kind],
                               "options": {"sample_size": 40}})
    assert result["kind"] == kind and result["traces"]
    assert result["sample_info"]["total_rows"] == 96
    assert result["sample_info"]["used_rows"] <= 96
    assert result["lesson_id"] == "dataset-analysis"
    assert result["explanation"]
    json.dumps(result, allow_nan=False)
    pd.testing.assert_frame_equal(data.frame(dataset["id"]), original)


def test_overview_totals_and_lazy_catalogue_are_consistent(laboratory):
    data, dataset, analysis = laboratory
    overview = analysis.analyze({"dataset_id": dataset["id"]})
    assert overview["summary"] == {"rows": 96, "columns": 9, "numeric_columns": 4,
                                   "categorical_columns": 5, "missing_cells": 110, "duplicate_rows": 0}
    assert [plot["kind"] for plot in overview["plots"]] == ["missing_bar", "histogram", "category_counts"]
    assert {item["id"] for item in overview["graph_types"]} == set(COLUMNS)
    assert len(overview["plots"]) == 3
    assert all(item["explanation"] and item["lesson_id"] for item in analysis.catalogue()["items"])


def test_full_frequency_counts_and_ecdf_denominator_include_ties_not_missing(tmp_path):
    gateway = LegacyDataGateway(tmp_path)
    dataset = gateway.load({"name": "Known values", "kind": "custom", "rows": [{"x": value, "target": 1.} for value in [1., 1., 2., 4., None]], "target": "target"})
    analysis = DatasetAnalysis(gateway)
    ecdf = analysis.analyze({"dataset_id": dataset["id"], "kind": "ecdf", "columns": ["x"]})
    assert ecdf["traces"][0]["x"] == [1., 2., 4.]
    assert ecdf["traces"][0]["y"] == [.5, .75, 1.]
    assert ecdf["sample_info"]["used_rows"] == 4
    histogram = analysis.analyze({"dataset_id": dataset["id"], "kind": "histogram", "columns": ["x"], "options": {"bins": 3, "sample_size": 1}})
    assert histogram["statistics"]["counts"] == [2, 1, 1]
    assert histogram["sample_info"]["sampled"] is False
    missing = analysis.analyze({"dataset_id": dataset["id"], "kind": "missing_bar", "columns": ["x"]})
    assert missing["statistics"]["counts"] == {"x": 1}
    assert missing["statistics"]["fractions"] == {"x": .2}


def test_pairwise_correlation_and_missing_pairs_have_explicit_counts(tmp_path):
    gateway = LegacyDataGateway(tmp_path)
    rows = [{"x": x, "y": y, "constant": 5} for x, y in [(1., 2.), (2., None), (None, 4.), (4., 8.)]]
    dataset = gateway.load({"name": "Pairs", "kind": "custom", "rows": rows, "target": "constant"})
    result = DatasetAnalysis(gateway).analyze({"dataset_id": dataset["id"], "kind": "correlation", "columns": ["x", "y", "constant"]})
    assert result["statistics"]["pair_counts"] == [[3, 2, 3], [2, 3, 3], [3, 3, 4]]
    assert result["traces"][0]["z"][0][1] == pytest.approx(1.)
    assert result["traces"][0]["z"][2] == [None, None, None]


def test_group_means_show_population_spread_and_category_counts_reconcile(laboratory):
    data, dataset, analysis = laboratory
    result = analysis.analyze({"dataset_id": dataset["id"], "kind": "group_mean", "columns": ["group", "y"]})
    frame = data.frame(dataset["id"])
    trace = result["traces"][0]
    for index, group in enumerate(trace["x"]):
        values = frame.loc[frame.group == group, "y"].to_numpy()
        assert trace["y"][index] == pytest.approx(sum(values) / len(values))
        assert trace["error_y"]["array"][index] == pytest.approx(np.std(values, ddof=0))
        assert trace["customdata"][index] == len(values)
    counts = analysis.analyze({"dataset_id": dataset["id"], "kind": "category_counts", "columns": ["category"], "options": {"top_k": 2}})
    assert sum(counts["traces"][0]["y"]) == 96
    assert counts["statistics"]["other_rows"] == 72
    cross = analysis.analyze({"dataset_id": dataset["id"], "kind": "category_crosstab", "columns": ["group", "category"], "options": {"top_k": 2}})
    assert np.asarray(cross["traces"][0]["z"]).sum() + cross["statistics"]["excluded_rare_rows"] == 96


def test_ordered_rolling_mean_preserves_original_values_and_never_uses_future(tmp_path):
    gateway = LegacyDataGateway(tmp_path)
    rows = [{"time": time, "__rolling_mean": value} for time, value in [(3, 9.), (1, 1.), (2, 5.), (4, 13.)]]
    dataset = gateway.load({"name": "Ordered", "kind": "custom", "rows": rows, "target": "__rolling_mean"})
    analysis = DatasetAnalysis(gateway)
    result = analysis.analyze({"dataset_id": dataset["id"], "kind": "rolling_mean", "columns": ["time", "__rolling_mean"], "options": {"window": 2}})
    assert result["traces"][0]["x"] == [1, 2, 3, 4]
    assert result["traces"][0]["y"] == [1., 5., 9., 13.]
    assert result["traces"][1]["y"] == [1., 3., 7., 11.]
    assert gateway.frame(dataset["id"])["__rolling_mean"].to_list() == [9., 1., 5., 13.]


def test_autocorrelation_known_four_point_sequence_and_irregular_rejection(tmp_path):
    gateway = LegacyDataGateway(tmp_path)
    first = gateway.load({"name": "Sequence", "kind": "custom", "rows": [{"t": i, "x": float(i + 1)} for i in range(4)], "target": "x"})
    analysis = DatasetAnalysis(gateway)
    result = analysis.analyze({"dataset_id": first["id"], "kind": "autocorrelation", "columns": ["t", "x"], "options": {"max_lag": 3}})
    assert result["traces"][0]["y"] == pytest.approx([1., .25, -.3, -.45])
    second = gateway.load({"name": "Irregular", "kind": "custom", "rows": [{"t": t, "x": float(i)} for i, t in enumerate([0, 1, 3, 4])], "target": "x"})
    with pytest.raises(ValueError, match="равномерными"):
        analysis.analyze({"dataset_id": second["id"], "kind": "autocorrelation", "columns": ["t", "x"]})


def test_outlier_flags_use_full_quantiles_but_leave_snapshot_unchanged(tmp_path):
    gateway = LegacyDataGateway(tmp_path)
    values = [0.] * 25 + [1.] * 25 + [2.] * 25 + [3.] * 24 + [1000.]
    dataset = gateway.load({"name": "Flags", "kind": "custom", "rows": [{"index": index, "x": x} for index, x in enumerate(values)], "target": "x"})
    result = DatasetAnalysis(gateway).analyze({"dataset_id": dataset["id"], "kind": "outliers", "columns": ["x"], "options": {"sample_size": 20}})
    assert result["statistics"]["flagged_rows"] == 1
    q1, q3 = np.percentile(values, [25, 75])
    assert result["statistics"]["lower_fence"] == q1 - 1.5 * (q3 - q1)
    assert result["statistics"]["upper_fence"] == q3 + 1.5 * (q3 - q1)
    assert gateway.frame(dataset["id"])["x"].to_list() == values


def test_point_sampling_reproducible_and_all_dimensions_follow_same_rows(laboratory):
    data, dataset, analysis = laboratory
    request = {"dataset_id": dataset["id"], "kind": "scatter3d", "columns": ["x", "y", "z"], "options": {"sample_size": 15, "seed": 6}}
    first = analysis.analyze(request)
    assert analysis.analyze(request) == first
    trace = first["traces"][0]
    selected = data.frame(dataset["id"]).loc[trace["customdata"]]
    assert len(trace["x"]) == len(trace["y"]) == len(trace["z"]) == 15
    assert trace["x"] == selected.x.to_list() and trace["y"] == selected.y.to_list() and trace["z"] == selected.z.to_list()
    assert first["sample_info"]["sampled"] is True


@pytest.mark.parametrize("patch,match", [
    ({"kind": "unknown"}, "каталога"), ({"kind": "histogram", "columns": ["group"]}, "числовой"),
    ({"kind": "scatter", "columns": ["x"]}, "столбцов"), ({"kind": "scatter", "columns": ["x", "x"]}, "неповторяющихся"),
    ({"options": {"sample_size": 5001}}, "sample_size"), ({"options": {"bins": True}}, "bins"),
    ({"options": {"corr_method": "kendall"}}, "corr_method"), ({"options": {"execute": "evil"}}, "Неизвестный"),
    ({"kind": "histogram", "columns": ["flag"]}, "bool"),
])
def test_invalid_or_unbounded_requests_fail_before_plotting(laboratory, patch, match):
    data, dataset, analysis = laboratory
    with pytest.raises(ValueError, match=match):
        analysis.analyze({"dataset_id": dataset["id"], **patch})


def test_real_http_route_returns_json_plot_and_validation_status(laboratory):
    data, dataset, analysis = laboratory
    app = FastAPI()

    @app.exception_handler(ValueError)
    async def invalid(request: Request, error: ValueError):
        return JSONResponse({"detail": str(error)}, status_code=422)

    app.include_router(analysis_routes(analysis))
    with TestClient(app) as client:
        assert len(client.get("/api/cml/analysis/kinds").json()["items"]) == 20
        response = client.post("/api/cml/analysis", json={"dataset_id": dataset["id"], "kind": "missing_bar", "columns": ["empty"]})
        assert response.status_code == 200 and response.json()["traces"][0]["y"] == [96]
        invalid = client.post("/api/cml/analysis", json={"dataset_id": dataset["id"], "kind": "correlation", "columns": ["category"]})
        assert invalid.status_code == 422
