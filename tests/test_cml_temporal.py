"""Forecast-origin, panel isolation and recursive future leakage checks."""

import numpy as np
import pandas as pd
import pytest

from cml_lab.infrastructure.ml.temporal import TemporalArtifact, TemporalDataset, build_temporal_frame, forecast_features


def single():
    return pd.DataFrame({"time": pd.date_range("2020-01-01", periods=12, freq="D"), "y": np.arange(12, dtype=float), "known": np.arange(12, dtype=float) * 2})


def test_origin_horizon_and_shifted_rolling_match_known_sequence():
    frame = single()
    data = build_temporal_frame(frame, "y", "time", lags=[1, 2], rolling=[3], horizon=2, features=["known"])
    np.testing.assert_allclose(data.X.iloc[0], [6., 2., 1., 1., 1., 0., 2.])
    assert data.y.iloc[0] == 4
    assert data.origins.iloc[0] == pd.Timestamp("2020-01-04", tz="UTC")
    assert data.targets.iloc[0] == pd.Timestamp("2020-01-05", tz="UTC")
    assert data.row_indices[0] == 3
    assert data.history_config["origin_semantics"] == "first_unknown"
    assert (data.targets >= data.origins).all()


def test_panel_lags_never_cross_an_entity_and_output_sorts_global_origins():
    frame = pd.concat([single().assign(entity="a"), single().assign(entity="b", y=np.arange(12) + 100)], ignore_index=True)
    frame = frame.sample(frac=1, random_state=40).reset_index(drop=True)
    data = TemporalDataset.from_frame(frame, "y", "time", "entity", lags=[1], rolling=[], features=[])
    for name, group in data.X.assign(entity=data.entities, y=data.y).groupby("entity"):
        np.testing.assert_allclose(group["y"] - group["y__lag_1"], 1.)
        if name == "b":
            assert group["y__lag_1"].min() >= 100
    assert data.origins.is_monotonic_increasing
    assert set(data.entities) == {"a", "b"}
    assert "entity" not in data.X and "time" not in data.X
    for index, position in enumerate(data.row_indices):
        assert frame.iloc[position]["entity"] == data.entities.iloc[index]


def test_future_perturbation_does_not_change_earlier_features():
    frame = single()
    baseline = build_temporal_frame(frame, "y", "time", lags=[1, 2], rolling=[3], features=[])
    changed = frame.copy()
    changed.loc[8:, "y"] = 100000.
    perturbed = build_temporal_frame(changed, "y", "time", lags=[1, 2], rolling=[3], features=[])
    mask = baseline.origins <= pd.Timestamp("2020-01-09", tz="UTC")
    np.testing.assert_allclose(baseline.X.loc[mask], perturbed.X.loc[mask])


@pytest.mark.parametrize("mutation,match", [
    (lambda frame: pd.concat([frame, frame.iloc[[0]]]), "повторяющаяся дата"),
    (lambda frame: frame.drop(index=4), "нерегулярные"),
    (lambda frame: frame.assign(time="bad date"), "разобрать"),
])
def test_invalid_time_grid_rejected_instead_of_silent_row_lags(mutation, match):
    with pytest.raises(ValueError, match=match):
        build_temporal_frame(mutation(single()), "y", "time")


def test_regular_calendar_months_accept_different_day_lengths():
    frame = pd.DataFrame({"time": pd.date_range("2020-01-01", periods=8, freq="MS"), "y": np.arange(8.)})
    result = build_temporal_frame(frame, "y", "time", lags=[1], rolling=[], features=[])
    assert len(result.y) == 7
    assert result.history_config["frequency"]["single"] == "MS"


@pytest.mark.parametrize("kwargs,match", [
    ({"lags": [0]}, "от 1"),
    ({"lags": [1, 1]}, "повторяться"),
    ({"lags": [], "rolling": []}, "хотя бы"),
    ({"horizon": 0}, "Горизонт"),
    ({"features": ["y"]}, "не содержат цель"),
    ({"lags": [1000], "rolling": []}, "не осталось"),
])
def test_invalid_feature_policy_has_clear_error(kwargs, match):
    with pytest.raises(ValueError, match=match):
        build_temporal_frame(single(), "y", "time", **kwargs)


class PreviousPlusOne:
    def predict(self, frame):
        return frame["y__lag_1"].to_numpy() + 1


def artifact():
    data = build_temporal_frame(single(), "y", "time", lags=[1], rolling=[], features=[])
    return TemporalArtifact(PreviousPlusOne(), data.history_config)


def test_future_targets_are_ignored_and_recursive_predictions_feed_history():
    history = single().iloc[:5]
    future = single().iloc[5:8].copy()
    future["y"] = [20000., -40000., 90000.]
    first = artifact().forecast(history, future)
    np.testing.assert_allclose(first["predicted"], [5., 6., 7.])
    future["y"] = -999999.
    second = artifact().forecast(history, future)
    np.testing.assert_allclose(first["predicted"], second["predicted"])
    assert list(first["time"]) == list(pd.date_range("2020-01-06", periods=3, tz="UTC"))


def test_recursive_panel_predictions_remain_entity_specific():
    history = pd.concat([single().iloc[:5].assign(entity="a"), single().iloc[:5].assign(entity="b", y=np.arange(5.) + 100)], ignore_index=True)
    future = pd.concat([single().iloc[5:8].assign(entity="a"), single().iloc[5:8].assign(entity="b")], ignore_index=True)
    data = build_temporal_frame(history, "y", "time", "entity", lags=[1], rolling=[], features=[])
    result = TemporalArtifact(PreviousPlusOne(), data.history_config).forecast(history, future)
    np.testing.assert_allclose(result.loc[result["entity"] == "a", "predicted"], [5., 6., 7.])
    np.testing.assert_allclose(result.loc[result["entity"] == "b", "predicted"], [105., 106., 107.])


def test_missing_future_exogenous_input_is_not_fabricated():
    history = single().iloc[:5]
    with pytest.raises(ValueError, match="внешних признаков"):
        forecast_features(history, pd.DataFrame({"time": ["2020-01-06"]}), "y", "time", lags=[1], rolling=[], features=["known"])


def test_forecast_cannot_reuse_a_past_timestamp_or_read_an_unknown_entity():
    history = single().iloc[:5]
    with pytest.raises(ValueError, match="после последнего"):
        forecast_features(history, pd.DataFrame({"time": ["2020-01-04"]}), "y", "time", lags=[1], rolling=[], features=[])
    panel = history.assign(entity="a")
    with pytest.raises(ValueError, match="нет сохранённой истории"):
        forecast_features(panel, pd.DataFrame({"time": ["2020-01-06"], "entity": ["b"]}), "y", "time", "entity", lags=[1], rolling=[], features=[])


def test_direct_horizon_cannot_be_silently_used_recursively():
    data = build_temporal_frame(single(), "y", "time", lags=[1], rolling=[], horizon=2, features=[])
    model = TemporalArtifact(PreviousPlusOne(), data.history_config)
    with pytest.raises(ValueError, match="horizon=1"):
        model.forecast(single().iloc[:5], single().iloc[5:7])


def test_direct_forecast_records_label_time_separately_from_origin():
    data = build_temporal_frame(single(), "y", "time", lags=[1], rolling=[], horizon=3, features=[])
    result = TemporalArtifact(PreviousPlusOne(), data.history_config).forecast(single().iloc[:5], single().iloc[5:6])
    assert result.iloc[0]["origin"] == pd.Timestamp("2020-01-06", tz="UTC")
    assert result.iloc[0]["time"] == pd.Timestamp("2020-01-08", tz="UTC")


def test_missing_future_steps_are_rejected_instead_of_relabelling_lags():
    with pytest.raises(ValueError, match="следующей точки"):
        artifact().forecast(single().iloc[:5], single().iloc[7:8])


@pytest.mark.parametrize("strategy", ["timeseries", "time_series", "rolling_origin", "expanding_window"])
def test_temporal_cv_aliases_purge_horizon_labels_and_keep_whole_dates(strategy):
    from cml_lab.infrastructure.ml.splitting import folds
    frame = pd.concat([
        pd.DataFrame({"time": np.arange(60), "y": np.arange(60.), "entity": "a"}),
        pd.DataFrame({"time": np.arange(60), "y": np.arange(60.) + 100, "entity": "b"}),
    ], ignore_index=True)
    prepared = build_temporal_frame(frame, "y", "time", "entity", lags=[1], rolling=[], horizon=4, features=[])
    plans = folds({"task": "panel", "validation": {"strategy": strategy, "folds": 3}},
                  prepared.y.to_numpy(), origins=prepared.origins.to_numpy(), target_times=prepared.targets.to_numpy())
    for train, validation in plans:
        boundary = prepared.origins.iloc[validation].min()
        assert prepared.targets.iloc[train].max() < boundary
        assert not set(prepared.origins.iloc[train]) & set(prepared.origins.iloc[validation])
        for timestamp in prepared.origins.iloc[validation].unique():
            date_positions = np.flatnonzero(prepared.origins.to_numpy() == timestamp)
            assert set(date_positions).issubset(set(validation))


def test_temporal_holdout_purges_horizon_crossings_at_both_boundaries():
    from cml_lab.infrastructure.ml.splitting import holdout
    frame = pd.DataFrame({"time": np.arange(60), "y": np.arange(60.)})
    prepared = build_temporal_frame(frame, "y", "time", lags=[1], rolling=[], horizon=5, features=[])
    parts = holdout({"task": "forecasting"}, len(prepared.y), prepared.y.to_numpy(),
                    origins=prepared.origins.to_numpy(), target_times=prepared.targets.to_numpy())
    assert prepared.targets.iloc[parts["train"]].max() < prepared.origins.iloc[parts["validation"]].min()
    assert prepared.targets.iloc[parts["validation"]].max() < prepared.origins.iloc[parts["test"]].min()
    assert len(set(np.concatenate(list(parts.values())))) == sum(map(len, parts.values()))
