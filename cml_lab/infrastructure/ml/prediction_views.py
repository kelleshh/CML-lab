"""Model slices use training ranges and fixed training values, never test rows."""
from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_bool_dtype


def build_prediction_view(artifact, X, y, train, validation, features=None, grid_size=25):
    if artifact.task not in {"regression", "classification"}:
        return None
    training = X.iloc[train]
    eligible = []
    for name in X.columns:
        if not is_numeric_dtype(training[name]) or is_bool_dtype(training[name]):
            continue
        values = training[name].to_numpy(dtype=float, na_value=np.nan)
        finite = values[np.isfinite(values)]
        if len(finite) and np.unique(finite).size > 1:
            eligible.append(name)
    chosen = list(features or eligible[:2])
    if not chosen:
        return {"available": False, "reason": "Для среза нужен хотя бы один изменяющийся числовой признак."}
    if len(chosen) > 2 or len(set(chosen)) != len(chosen) or any(name not in eligible for name in chosen):
        raise ValueError("Для среза модели выберите один или два изменяющихся числовых признака.")
    grid_size = int(grid_size)
    if not 5 <= grid_size <= 50:
        raise ValueError("Сетка среза ограничена 5–50 значениями на ось.")
    fixed = {}
    for name in X.columns:
        column = training[name]
        if is_numeric_dtype(column) and not is_bool_dtype(column):
            values = column.to_numpy(dtype=float, na_value=np.nan)
            finite = values[np.isfinite(values)]
            fixed[name] = float(np.median(finite)) if len(finite) else None
        else:
            modes = column.dropna().mode()
            fixed[name] = modes.iloc[0] if len(modes) else None
    axes = []
    for name in chosen:
        values = training[name].to_numpy(dtype=float, na_value=np.nan)
        finite = values[np.isfinite(values)]
        axes.append(np.linspace(float(finite.min()), float(finite.max()), grid_size))
    coordinates = np.meshgrid(*axes, indexing="xy") if len(axes) == 2 else [axes[0]]
    grid = pd.DataFrame([fixed] * coordinates[0].size, columns=X.columns)
    for name, values in zip(chosen, coordinates):
        grid[name] = values.ravel()
    predicted = artifact.predict(grid)
    classes = None
    if artifact.task == "classification":
        classes = artifact.classes_.tolist()
        lookup = {label: index for index, label in enumerate(classes)}
        predicted = np.asarray([lookup[label] for label in predicted], dtype=int)
    values = np.asarray(predicted).reshape(coordinates[0].shape).tolist()
    visible = np.r_[train, validation]
    if len(visible) > 1000:
        visible = visible[np.linspace(0, len(visible) - 1, 1000, dtype=int)]
    visible_X = X.iloc[visible]
    mask = np.ones(len(visible), dtype=bool)
    for name in chosen:
        mask &= np.isfinite(visible_X[name].to_numpy(dtype=float, na_value=np.nan))
    visible = visible[mask]
    training_set = set(map(int, train))
    return {
        "available": True, "task": artifact.task, "features": chosen,
        "axes": [axis.tolist() for axis in axes], "predictions": values,
        "classes": classes,
        "fixed_features": {name: value for name, value in fixed.items() if name not in chosen},
        "points": {"coordinates": [X.iloc[visible][name].astype(float).tolist() for name in chosen],
                   "actual": np.asarray(y)[visible].tolist(),
                   "split": ["train" if int(index) in training_set else "validation" for index in visible]},
        "description": "Срез модели: выбранные признаки меняются в диапазоне обучения; остальные закреплены на медианах или наиболее частых значениях обучающей части. Точки показывают обучение и выбор настроек. Это срез многомерной модели, поэтому отдельные точки могут не лежать на поверхности.",
    }
