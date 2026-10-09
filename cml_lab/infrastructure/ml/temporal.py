"""History-aware features with explicit forecast origins and panel boundaries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def _positive_values(values, label, maximum_count, minimum=1):
    if not isinstance(values, (list, tuple)) or len(values) > maximum_count:
        raise ValueError(f"{label}: нужен список не более {maximum_count} значений.")
    if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not minimum <= value <= 1000 for value in values):
        raise ValueError(f"{label}: нужны целые числа от {minimum} до 1000.")
    if len(set(values)) != len(values):
        raise ValueError(f"{label}: значения не должны повторяться.")
    return list(values)


def _horizon(value):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not 1 <= value <= 1000:
        raise ValueError("Горизонт прогноза: целое число от 1 до 1000.")
    return int(value)


def _normalise_time(values):
    if pd.api.types.is_numeric_dtype(values.dtype):
        result = pd.to_numeric(values, errors="raise")
        if not np.isfinite(result).all():
            raise ValueError("Время не может содержать пропуски и бесконечности.")
        return result
    result = pd.to_datetime(values, errors="coerce", utc=True, format="mixed")
    if result.isna().any():
        raise ValueError("Не удалось разобрать все значения временного столбца. Исправьте даты и пропуски.")
    return result


def _regular_frequency(times):
    if len(times) < 2:
        return None
    if pd.api.types.is_numeric_dtype(times.dtype):
        differences = np.diff(times.to_numpy(dtype=float))
        if not np.allclose(differences, differences[0]) or differences[0] <= 0:
            raise ValueError("Лаги считаются по наблюдениям: для этого рецепта нужна регулярная временная сетка внутри каждой серии.")
        return float(differences[0])
    if len(times) == 2:
        return pd.tseries.frequencies.to_offset(times.iloc[1] - times.iloc[0]).freqstr
    frequency = pd.infer_freq(pd.DatetimeIndex(times))
    if frequency is None:
        raise ValueError("Внутри серии нерегулярные даты. Сначала приведите данные к одной частоте; пропущенные периоды не считаются соседними.")
    return frequency


def _advance_time(timestamp, frequency, steps=1):
    if isinstance(frequency, (int, float)):
        return timestamp + frequency * steps
    return timestamp + pd.tseries.frequencies.to_offset(frequency) * steps


def _roles(frame, target, time, entity, features):
    if not isinstance(frame, pd.DataFrame) or frame.columns.has_duplicates:
        raise ValueError("Для временной подготовки нужна таблица с уникальными столбцами.")
    required = [target, time] + ([entity] if entity else [])
    if len(set(required)) != len(required) or any(name not in frame for name in required):
        raise ValueError("Выберите разные существующие столбцы цели, времени и объекта панели.")
    features = [name for name in frame if name not in required] if features is None else features
    if not isinstance(features, (list, tuple)) or len(features) != len(set(features)) or any(name not in frame or name in required for name in features):
        raise ValueError("Входы ряда не содержат цель, время и ID объекта; выберите существующие неповторяющиеся признаки.")
    if entity and frame[entity].isna().any():
        raise ValueError("ID объекта панели не может быть пропущен.")
    return list(features)


def _prepare(frame, target, time, entity, features):
    features = _roles(frame, target, time, entity, features)
    table = frame.copy()
    table[time] = _normalise_time(table[time])
    table[target] = pd.to_numeric(table[target], errors="coerce")
    table["__cml_row_index__"] = np.arange(len(table))
    if "__cml_entity__" in frame or "__cml_row_index__" in frame:
        raise ValueError("Имена __cml_entity__/__cml_row_index__ зарезервированы временным адаптером.")
    key = entity or "__cml_entity__"
    if entity is None:
        table[key] = "single"
    if table.duplicated([key, time]).any():
        raise ValueError("У объекта есть повторяющаяся дата: каждая пара объект/время должна быть уникальной.")
    # Sort within groups; heterogeneous entity IDs do not need to be comparable.
    groups = [group.sort_values(time, kind="mergesort").reset_index(drop=True) for _, group in table.groupby(key, sort=False, dropna=False)]
    frequencies = {}
    for group in groups:
        frequencies[str(group[key].iloc[0])] = _regular_frequency(group[time])
    return groups, key, features, frequencies


@dataclass(frozen=True)
class TemporalDataset:
    X: pd.DataFrame
    y: pd.Series
    origins: pd.Series
    targets: pd.Series
    entities: pd.Series
    row_indices: np.ndarray
    history_config: dict

    @classmethod
    def from_frame(cls, frame, target, time, entity=None, lags=(1, 2, 3), rolling=(3,), horizon=1, features=None):
        return build_temporal_frame(frame, target, time, entity, lags, rolling, horizon, features)


def _feature_names(target, lags, rolling):
    return [f"{target}__lag_{lag}" for lag in lags] + [f"{target}__rolling_{method}_{window}" for window in rolling for method in ("mean", "std", "min", "max")]


def build_temporal_frame(frame, target, time, entity=None, lags=(1, 2, 3), rolling=(3,), horizon=1, features=None, *, rolling_windows=None):
    """Create labels and past-only features without choosing a train/test split.

    Origin t is the first unknown point: lag1 is y(t-1). Horizon h targets
    y(t+h-1). All rolling windows first shift y by one observation. The caller
    must purge training labels whose target time reaches the next split origin.
    """
    if rolling_windows is not None:
        if rolling != (3,) and list(rolling) != list(rolling_windows):
            raise ValueError("Укажите rolling либо rolling_windows, не два разных набора окон.")
        rolling = rolling_windows
    lags = _positive_values(lags, "Лаги", 64)
    rolling = _positive_values(rolling, "Окна", 32, minimum=2)
    horizon = _horizon(horizon)
    if not lags and not rolling:
        raise ValueError("Для прогноза выберите хотя бы один лаг или окно истории.")
    groups, key, features, frequencies = _prepare(frame, target, time, entity, features)
    if not groups:
        raise ValueError("В ряде нет наблюдений.")
    generated = _feature_names(target, lags, rolling)
    if set(generated).intersection(frame.columns):
        raise ValueError("Имя создаваемого временного признака совпало с исходным столбцом.")
    outputs = []
    for group in groups:
        X = group[features].copy()
        target_values = group[target]
        past = target_values.shift(1)
        for lag in lags:
            X[f"{target}__lag_{lag}"] = target_values.shift(lag)
        for window in rolling:
            roll = past.rolling(window=window, min_periods=window)
            for method in ("mean", "std", "min", "max"):
                X[f"{target}__rolling_{method}_{window}"] = getattr(roll, method)()
        labels = target_values.shift(-(horizon - 1))
        target_times = group[time].shift(-(horizon - 1))
        history_valid = np.isfinite(X[generated].to_numpy(dtype=float)).all(axis=1)
        mask = history_valid & np.isfinite(labels.to_numpy(dtype=float)) & target_times.notna().to_numpy()
        if mask.any():
            outputs.append((X.loc[mask], labels.loc[mask], group.loc[mask, time], target_times.loc[mask], group.loc[mask, key], group.loc[mask, "__cml_row_index__"]))
    if not outputs:
        raise ValueError("После лагов/окон/горизонта не осталось строк. Уменьшите историю или загрузите более длинный ряд.")
    combined = [pd.concat([output[index] for output in outputs], ignore_index=True) for index in range(6)]
    order = combined[2].sort_values(kind="mergesort").index
    X, y, origins, targets, entities, row_indices = [values.loc[order].reset_index(drop=True) for values in combined]
    y.name = target
    history_config = {"target": target, "time": time, "entity": entity, "lags": lags, "rolling": rolling,
                      "rolling_windows": rolling, "horizon": horizon, "features": features,
                      "origin_semantics": "first_unknown", "frequency": frequencies,
                      "evaluation_protocol": "rolling_origin_observed_history"}
    return TemporalDataset(X, y, origins, targets, entities, row_indices.to_numpy(dtype=int), history_config)


def forecast_features(history, future, target, time, entity=None, lags=(1, 2, 3), rolling=(3,), features=None):
    """Build one future row per entity from explicitly available past history.

    Future target values, if present, are ignored. Selected exogenous inputs
    must be supplied in future; their future availability is caller policy.
    """
    lags = _positive_values(lags, "Лаги", 64)
    rolling = _positive_values(rolling, "Окна", 32, minimum=2)
    if not lags and not rolling:
        raise ValueError("Для прогноза выберите хотя бы один лаг или окно истории.")
    groups, key, features, _ = _prepare(history, target, time, entity, features)
    if not isinstance(future, pd.DataFrame) or time not in future or any(name not in future for name in features):
        raise ValueError("Для прогноза передайте дату и значения выбранных внешних признаков будущего.")
    future = future.copy()
    future[time] = _normalise_time(future[time])
    if entity:
        if entity not in future or future[entity].isna().any() or future[entity].duplicated().any():
            raise ValueError("Один шаг прогноза требует по одной строке на каждый выбранный объект панели.")
    elif len(future) != 1:
        raise ValueError("Для одного ряда передайте одну будущую дату; несколько шагов выполняются рекурсивно.")
    result = []
    lookup = {group[key].iloc[0]: group for group in groups}
    required = max(lags + rolling)
    for _, row in future.iterrows():
        value = row[entity] if entity else "single"
        if value not in lookup:
            raise ValueError("Для объекта прогноза нет сохранённой истории.")
        group = lookup[value]
        if row[time] <= group[time].iloc[-1]:
            raise ValueError("Дата прогноза должна идти после последнего доступного наблюдения.")
        frequency = _regular_frequency(group[time])
        if frequency is None or row[time] != _advance_time(group[time].iloc[-1], frequency):
            raise ValueError("Прогноз должен начинаться со следующей точки регулярной сетки; пропуск будущих шагов меняет смысл лагов.")
        values = group[target].to_numpy(dtype=float)
        if len(values) < required or not np.isfinite(values[-required:]).all():
            raise ValueError("Для прогноза не хватает конечных прошлых значений цели.")
        record = {name: row[name] for name in features}
        for lag in lags:
            record[f"{target}__lag_{lag}"] = values[-lag]
        for window in rolling:
            selected = values[-window:]
            record.update({f"{target}__rolling_mean_{window}": float(np.mean(selected)),
                           f"{target}__rolling_std_{window}": float(np.std(selected, ddof=1)),
                           f"{target}__rolling_min_{window}": float(np.min(selected)),
                           f"{target}__rolling_max_{window}": float(np.max(selected))})
        result.append(record)
    return pd.DataFrame(result, columns=list(features) + _feature_names(target, lags, rolling))


@dataclass
class TemporalArtifact:
    """Serializable estimator plus a recorded history contract for forecasting."""

    estimator: object
    history_config: dict

    def forecast(self, history, future):
        """Forecast at most 20 regular future steps, feeding predictions as history.

        This recursive operation is valid only for a horizon-one fitted model.
        Direct models with h>1 forecast exactly one target at a supplied origin.
        """
        config = self.history_config
        target, time, entity = config["target"], config["time"], config.get("entity")
        if not isinstance(future, pd.DataFrame) or time not in future or not 1 <= len(future) <= 2000:
            raise ValueError("Будущее задаётся таблицей от 1 до 2000 строк.")
        future = future.copy()
        future[time] = _normalise_time(future[time])
        key = entity or "__cml_entity__"
        if entity is None:
            future[key] = "single"
        if key not in future or future[key].isna().any() or future.duplicated([key, time]).any():
            raise ValueError("Будущие объект/время должны быть уникальны и заполнены.")
        if future.groupby(key).size().max() > 20:
            raise ValueError("Прогноз ограничен 20 будущими шагами на объект.")
        if config.get("horizon", 1) != 1 and future.groupby(key).size().max() > 1:
            raise ValueError("Рекурсивный прогноз требует модель с horizon=1; для прямого h>1 передайте один origin на объект.")
        available = history.copy()
        groups, group_key, _, _ = _prepare(history, target, time, entity, config["features"])
        frequencies = {group[group_key].iloc[0]: _regular_frequency(group[time]) for group in groups}
        records = []
        for timestamp, batch in future.sort_values(time, kind="mergesort").groupby(time, sort=True):
            X = forecast_features(available, batch, target, time, entity, config["lags"], config["rolling"], config["features"])
            predictions = np.asarray(self.estimator.predict(X), dtype=float).reshape(-1)
            if len(predictions) != len(batch) or not np.isfinite(predictions).all():
                raise ValueError("Модель вернула некорректный прогноз.")
            additions = batch.copy()
            additions[target] = predictions
            for position, (_, row) in enumerate(batch.iterrows()):
                entity_value = row[entity] if entity else "single"
                target_time = _advance_time(timestamp, frequencies[entity_value], config.get("horizon", 1) - 1)
                result = {time: target_time, "origin": timestamp, "predicted": float(predictions[position])}
                if entity:
                    result[entity] = row[entity]
                records.append(result)
            if config.get("horizon", 1) == 1:
                available = pd.concat([available, additions[[name for name in available.columns if name in additions]]], ignore_index=True)
        return pd.DataFrame(records)
