"""Bounded exploratory plots from raw, immutable dataset snapshots."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
from scipy import stats

from cml_lab.shared.domain import ValidationError, identifier, integer, json_object


MAX_POINTS = 5000
MAX_COLUMNS = 20
MAX_CELLS = 5_000_000
COLOR = "#803748"


def _definition(id, name, category, column_types, explanation):
    return {"id": id, "name": name, "category": category, "column_types": column_types,
            "explanation": explanation, "lesson_id": "dataset-analysis"}


GRAPH_TYPES = [
    _definition("histogram", "Histogram", "Distribution", ["numeric"], "Столбец показывает число значений внутри интервала. bins меняет подробность; вид гистограммы не доказывает нормальность."),
    _definition("box", "Box plot", "Distribution", ["numeric"], "Коробка содержит средние 50% значений, линия внутри — медиана. Точки за усами требуют проверки и не удаляются автоматически."),
    _definition("violin", "Violin plot", "Distribution", ["numeric"], "Ширина отражает сглаженную плотность значений. Форма зависит от сглаживания и размера показанной выборки."),
    _definition("ecdf", "ECDF", "Distribution", ["numeric"], "Высота ECDF в точке x — доля непустых значений, которые не больше x. Пропуски не входят в знаменатель."),
    _definition("qq", "Normal Q-Q plot", "Distribution", ["numeric"], "Сравнивает упорядоченные значения с квантилями нормального распределения. Изгибы и хвосты показывают различие; это график, а не автоматическое решение о нормальности."),
    _definition("kde", "Kernel density estimate", "Distribution", ["numeric"], "Гауссовские ядра дают гладкую оценку плотности. Площадь плотности примерно 1, высота не является числом строк; многомодальные данные могут сглаживаться."),
    _definition("scatter", "Scatter plot", "Relations", ["numeric", "numeric"], "Каждая точка — одна строка с обеими координатами. График помогает увидеть форму связи, группы и редкие точки; связь не устанавливает причину."),
    _definition("scatter3d", "3D scatter plot", "Relations", ["numeric", "numeric", "numeric"], "Три координаты показывают выбранные числовые столбцы. Поворачивайте график и проверяйте 2D-срезы: проекция может скрывать точки."),
    _definition("histogram2d", "2D histogram", "Relations", ["numeric", "numeric"], "Цвет клетки показывает число строк в двух интервалах одновременно. Используются все строки с обеими координатами, без случайной выборки точек."),
    _definition("correlation", "Correlation matrix", "Relations", [], "Pearson измеряет линейную связь, Spearman — монотонную связь рангов. Каждая пара использует свои непустые строки. null означает, что связь не определена; корреляция не устанавливает причину."),
    _definition("missing_bar", "Missing values by column", "Missing data", [], "Количество пропусков считается по всей таблице. Большая доля пропусков — повод выяснить причину, а не автоматически заполнить столбец средним."),
    _definition("missing_matrix", "Missingness matrix", "Missing data", [], "1 означает пропущенную ячейку, 0 — известную. Совпадающие полосы помогают заметить совместное отсутствие данных; номера строк принадлежат исходной таблице."),
    _definition("category_counts", "Category counts", "Categorical", ["any"], "Количество строк каждой категории считается по всей таблице. Показаны top_k категорий и отдельная сумма остальных; пропуски описаны отдельно."),
    _definition("category_crosstab", "Category cross-tabulation", "Categorical", ["any", "any"], "Клетка показывает число строк с двумя категориями. Редкие категории вне top_k исключены из этой матрицы и явно перечислены в счетчике."),
    _definition("group_box", "Box plot by group", "Groups", ["any", "numeric"], "Сравнивает распределения числового значения внутри категорий. Размеры групп могут различаться; график не доказывает влияние категории."),
    _definition("group_mean", "Group mean", "Groups", ["any", "numeric"], "Столбец — среднее значение группы. Полоса ошибки показывает стандартное отклонение значений, а не доверительный интервал среднего. Количество строк группы указано отдельно."),
    _definition("time_line", "Time series plot", "Time", ["time", "numeric"], "Значения упорядочены по времени. Линия не заполняет пропущенные даты; повторяющиеся даты могут давать несколько точек. Это анализ исходных наблюдений."),
    _definition("rolling_mean", "Rolling mean", "Time", ["time", "numeric"], "Среднее вычисляется по последним window наблюдениям после сортировки, без будущих строк. Это описательный график; для признака прогноза нужна отдельная задержка относительно цели."),
    _definition("autocorrelation", "Autocorrelation", "Time", ["time", "numeric"], "Сравнивает ряд с ним же при сдвиге на lag наблюдений. Нужны равномерные уникальные моменты и одна последовательность. Тренд может создавать большую корреляцию без сезонности."),
    _definition("outliers", "IQR outlier flags", "Quality", ["numeric"], "Порог 1.5×IQR отмечает значения ниже Q1−1.5×IQR и выше Q3+1.5×IQR. Это эвристическая метка для просмотра; строки не меняются и не удаляются."),
]
DEFINITIONS = {item["id"]: item for item in GRAPH_TYPES}
_OPTIONS = {"sample_size", "seed", "bins", "top_k", "window", "max_lag", "corr_method"}


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _numeric(series):
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def _sample(frame, count, seed, *, ordered=False):
    if len(frame) <= count:
        return frame
    if ordered:
        return frame.iloc[np.unique(np.linspace(0, len(frame) - 1, count, dtype=int))]
    return frame.sample(count, random_state=seed).sort_index()


def _trace(type, **values):
    return {"type": type, **values}


def _layout(x=None, y=None, **values):
    return {"xaxis": {"title": x or ""}, "yaxis": {"title": y or ""},
            "margin": {"t": 25, "b": 65, "l": 65, "r": 25}, **values}


class DatasetAnalysis:
    def __init__(self, data_gateway):
        self.data = data_gateway

    @staticmethod
    def catalogue():
        return {"items": deepcopy(GRAPH_TYPES), "max_points": MAX_POINTS, "max_columns": MAX_COLUMNS}

    def analyze(self, payload):
        request = json_object(payload, "Настройки анализа")
        frame = self.data.frame(identifier(request.get("dataset_id")))
        if not len(frame) or frame.size > MAX_CELLS:
            raise ValidationError("Анализ требует непустую таблицу не более пяти миллионов ячеек.")
        kind = request.get("kind", "overview")
        options = json_object(request.get("options", {}), "Параметры графика")
        if set(options) - _OPTIONS:
            raise ValidationError("Неизвестный параметр анализа: " + ", ".join(sorted(set(options) - _OPTIONS)))
        settings = {
            "sample_size": integer(options.get("sample_size", 2000), "sample_size", 1, MAX_POINTS),
            "seed": integer(options.get("seed", 42), "seed", 0, 2**32 - 1),
            "bins": integer(options.get("bins", 30), "bins", 2, 100),
            "top_k": integer(options.get("top_k", 15), "top_k", 2, MAX_COLUMNS),
            "window": integer(options.get("window", 10), "window", 2, 1000),
            "max_lag": integer(options.get("max_lag", 30), "max_lag", 1, 100),
            "corr_method": options.get("corr_method", "pearson"),
        }
        if settings["corr_method"] not in {"pearson", "spearman"}:
            raise ValidationError("corr_method: выберите pearson или spearman.")
        columns = request.get("columns", [])
        if not isinstance(columns, list) or len(columns) > MAX_COLUMNS or any(
                not isinstance(name, str) or name not in frame for name in columns) or len(set(columns)) != len(columns):
            raise ValidationError("Выберите не более 20 неповторяющихся столбцов исходной таблицы.")
        if kind == "overview":
            return self._overview(frame, settings)
        if kind not in DEFINITIONS:
            raise ValidationError("Выберите вид анализа из каталога.")
        definition = DEFINITIONS[kind]
        types = definition["column_types"]
        if types and len(columns) != len(types):
            raise ValidationError(f"Для {definition['name']} нужно столбцов: {len(types)}.")
        for name, type in zip(columns, types):
            if type == "numeric" and not _numeric(frame[name]):
                raise ValidationError(f"{name}: выберите числовой столбец; bool и категории не являются числовой координатой.")
        if not types and not columns:
            columns = [name for name in frame if _numeric(frame[name])][:MAX_COLUMNS] if kind == "correlation" else list(frame.columns)[:MAX_COLUMNS]
        if not columns:
            raise ValidationError("Для графика не осталось подходящих столбцов.")
        return self._graph(frame, kind, columns, settings)

    def _overview(self, frame, settings):
        numeric = [name for name in frame if _numeric(frame[name])]
        categorical = [name for name in frame if name not in numeric]
        plots = [self._graph(frame, "missing_bar", list(frame.columns)[:MAX_COLUMNS], settings)]
        if numeric:
            plots.append(self._graph(frame, "histogram", [numeric[0]], settings))
        if categorical:
            plots.append(self._graph(frame, "category_counts", [categorical[0]], settings))
        return {"kind": "overview", "title": "Dataset overview", "traces": [], "layout": {},
                "explanation": "Обзор описывает исходную таблицу до подготовки и обучения. Дополнительные графики создаются только по кнопке. Не выбирайте параметры модели по отдельному итоговому test.",
                "sample_info": {"total_rows": len(frame), "used_rows": len(frame), "sampled": False, "scope": "full dataset"},
                "summary": {"rows": len(frame), "columns": len(frame.columns), "numeric_columns": len(numeric),
                            "categorical_columns": len(categorical), "missing_cells": int(frame.isna().sum().sum()),
                            "duplicate_rows": int(frame.duplicated().sum())},
                "columns": [{"name": name, "numeric": _numeric(frame[name]), "dtype": str(frame[name].dtype),
                             "missing": int(frame[name].isna().sum()), "unique": int(frame[name].nunique(dropna=True))}
                            for name in frame], "graph_types": deepcopy(GRAPH_TYPES), "plots": plots}

    def _graph(self, frame, kind, columns, settings):
        definition = DEFINITIONS[kind]
        result = {"kind": kind, "title": definition["name"] + (": " + ", ".join(columns) if columns else ""),
                  "explanation": definition["explanation"], "lesson_id": definition["lesson_id"],
                  "traces": [], "layout": _layout(), "statistics": {},
                  "sample_info": {"total_rows": len(frame), "eligible_rows": len(frame), "used_rows": len(frame),
                                  "sampled": False, "scope": "full dataset", "seed": settings["seed"]}}
        data = frame[columns]
        points = settings["sample_size"]

        def used(data, *, sampled=False):
            result["sample_info"].update(used_rows=len(data), sampled=sampled,
                                         scope="sample of eligible rows" if sampled else "all eligible rows")

        if kind in {"missing_bar", "missing_matrix"}:
            counts = data.isna().sum()
            result["statistics"] = {"counts": counts.to_dict(), "fractions": (counts / len(frame)).to_dict(),
                                    "columns_truncated": len(columns) < len(frame.columns)}
            if kind == "missing_bar":
                result["traces"] = [_trace("bar", x=columns, y=counts.to_numpy(), marker={"color": COLOR})]
                result["layout"] = _layout("Column", "Missing rows")
            else:
                selected = _sample(data, min(points, 1000), settings["seed"])
                result["traces"] = [_trace("heatmap", x=columns, y=selected.index.to_list(), z=selected.isna().astype(int).to_numpy(),
                                            zmin=0, zmax=1, colorscale=[[0, "#f4efe3"], [1, COLOR]], showscale=False)]
                result["layout"] = _layout("Column", "Original row index")
                used(selected, sampled=len(selected) < len(data))
        elif kind == "correlation":
            if any(not _numeric(data[name]) for name in columns):
                raise ValidationError("Correlation matrix использует только числовые столбцы.")
            correlations = data.corr(method=settings["corr_method"])
            presence = data.notna().astype(np.int64)
            result["statistics"] = {"pair_counts": (presence.T @ presence).to_numpy(), "method": settings["corr_method"]}
            result["traces"] = [_trace("heatmap", x=columns, y=columns, z=correlations.to_numpy(), zmin=-1, zmax=1,
                                        colorscale=[[0, "#315e54"], [.5, "#f4efe3"], [1, COLOR]], colorbar={"title": settings["corr_method"]})]
        elif kind == "category_counts":
            values = data.iloc[:, 0].dropna().astype(str).value_counts()
            top = values.head(settings["top_k"])
            labels, counts = top.index.to_list(), top.to_list()
            other = int(values.iloc[len(top):].sum())
            if other:
                label = "(other categories)"
                while label in labels:
                    label += "*"
                labels.append(label); counts.append(other)
            result["traces"] = [_trace("bar", x=labels, y=counts, marker={"color": COLOR})]
            result["layout"] = _layout(columns[0], "Rows")
            result["statistics"] = {"missing": int(data.iloc[:, 0].isna().sum()), "other_rows": other}
        elif kind == "category_crosstab":
            clean = data.dropna().astype(str)
            if not len(clean):
                raise ValidationError("Category cross-tabulation требует строки с обеими известными категориями.")
            x = clean.iloc[:, 0].value_counts().head(settings["top_k"]).index
            y = clean.iloc[:, 1].value_counts().head(settings["top_k"]).index
            selected = clean[clean.iloc[:, 0].isin(x) & clean.iloc[:, 1].isin(y)]
            table = pd.crosstab(selected.iloc[:, 1], selected.iloc[:, 0]).reindex(index=y, columns=x, fill_value=0)
            result["traces"] = [_trace("heatmap", x=x.to_list(), y=y.to_list(), z=table.to_numpy(), colorscale=[[0, "#f4efe3"], [1, COLOR]])]
            result["layout"] = _layout(columns[0], columns[1])
            result["statistics"] = {"missing_rows": len(data) - len(clean), "excluded_rare_rows": len(clean) - len(selected)}
            result["sample_info"]["eligible_rows"] = len(clean)
            used(selected)
        elif kind in {"group_box", "group_mean"}:
            clean = data.dropna().copy()
            if not len(clean):
                raise ValidationError("График групп требует известные категорию и числовое значение в одной строке.")
            clean[columns[0]] = clean[columns[0]].astype(str)
            groups = clean[columns[0]].value_counts().head(settings["top_k"]).index.to_list()
            eligible_rows = len(clean)
            clean = clean[clean[columns[0]].isin(groups)]
            result["statistics"] = {"group_counts": clean.groupby(columns[0], sort=False).size().to_dict(),
                                    "missing_rows": len(data) - eligible_rows,
                                    "excluded_rare_rows": eligible_rows - len(clean)}
            if kind == "group_mean":
                group = clean.groupby(columns[0])[columns[1]]
                result["traces"] = [_trace("bar", x=groups, y=group.mean().reindex(groups).to_numpy(),
                                            error_y={"type": "data", "array": group.std(ddof=0).reindex(groups).to_numpy()},
                                            customdata=group.size().reindex(groups).to_numpy(), marker={"color": COLOR},
                                            hovertemplate="%{x}: mean=%{y}; rows=%{customdata}<extra></extra>")]
                used(clean)
            else:
                selected = _sample(clean, points, settings["seed"])
                result["traces"] = [_trace("box", name=name, y=selected.loc[selected[columns[0]] == name, columns[1]].to_numpy(), boxpoints="outliers") for name in groups]
                used(selected, sampled=len(selected) < len(clean))
            result["sample_info"]["eligible_rows"] = eligible_rows
            result["layout"] = _layout(columns[0], columns[1])
        elif kind in {"time_line", "rolling_mean", "autocorrelation"}:
            clean = data.copy()
            if _numeric(clean[columns[0]]):
                clean[columns[0]] = pd.to_numeric(clean[columns[0]], errors="coerce")
            else:
                clean[columns[0]] = pd.to_datetime(clean[columns[0]], utc=True, errors="coerce", format="mixed")
            clean = clean.dropna().sort_values(columns[0], kind="stable")
            if len(clean) < 2:
                raise ValidationError("Для временного графика нужны минимум два известных значения и корректных момента.")
            result["sample_info"]["eligible_rows"] = len(clean)
            if kind == "autocorrelation":
                times = clean[columns[0]].astype(np.int64).to_numpy() if not _numeric(clean[columns[0]]) else clean[columns[0]].to_numpy(dtype=float)
                gaps = np.diff(times)
                if np.any(gaps <= 0) or not np.allclose(gaps, gaps[0], rtol=1e-6, atol=0):
                    raise ValidationError("Autocorrelation требует одну последовательность с уникальными равномерными моментами. Выберите один объект и заполните сетку времени.")
                values = clean[columns[1]].to_numpy(dtype=float)
                centered = values - values.mean()
                denominator = np.dot(centered, centered)
                if not denominator:
                    raise ValidationError("Autocorrelation постоянного ряда не определена.")
                maximum = min(settings["max_lag"], len(values) - 1)
                acf = [1.] + [float(np.dot(centered[lag:], centered[:-lag]) / denominator) for lag in range(1, maximum + 1)]
                result["traces"] = [_trace("bar", x=list(range(maximum + 1)), y=acf, marker={"color": COLOR})]
                result["layout"] = _layout("Lag (observations)", "Autocorrelation")
                result["statistics"] = {"normalization": "sum of centered lag products / full centered sum of squares", "time_step": str(gaps[0])}
                used(clean)
            else:
                rolling_values = clean[columns[1]].rolling(settings["window"], min_periods=1).mean() if kind == "rolling_mean" else None
                selected = _sample(clean, points, settings["seed"], ordered=True)
                result["traces"] = [_trace("scatter", mode="lines", x=selected[columns[0]].to_list(), y=selected[columns[1]].to_numpy(), name=columns[1], line={"color": "#315e54"})]
                if kind == "rolling_mean":
                    result["traces"].append(_trace("scatter", mode="lines", x=selected[columns[0]].to_list(), y=rolling_values.loc[selected.index].to_numpy(), name=f"rolling mean ({settings['window']})", line={"color": COLOR}))
                result["layout"] = _layout(columns[0], columns[1])
                used(clean)
                result["sample_info"].update(rendered_rows=len(selected), sampled=len(selected) < len(clean), scope="all ordered rows; bounded displayed points")
        else:
            clean = data.dropna()
            result["sample_info"]["eligible_rows"] = len(clean)
            if not len(clean):
                raise ValidationError("Выбранные столбцы не содержат совместно известных значений.")
            selected = _sample(clean, points, settings["seed"])
            values = selected.iloc[:, 0].to_numpy(dtype=float)
            used(selected, sampled=len(selected) < len(clean))
            if kind == "histogram":
                counts, edges = np.histogram(clean.iloc[:, 0].to_numpy(dtype=float), bins=settings["bins"])
                result["traces"] = [_trace("bar", x=((edges[:-1] + edges[1:]) / 2), y=counts,
                                            width=np.diff(edges), marker={"color": COLOR})]
                result["statistics"] = {"edges": edges, "counts": counts}
                result["layout"] = _layout(columns[0], "Rows")
                used(clean)
            elif kind in {"box", "violin"}:
                result["traces"] = [_trace(kind, y=values, name=columns[0], boxpoints="outliers") if kind == "box"
                                    else _trace("violin", y=values, name=columns[0], box={"visible": True}, meanline={"visible": True})]
                result["layout"] = _layout(None, columns[0])
            elif kind == "ecdf":
                unique, counts = np.unique(clean.iloc[:, 0].to_numpy(dtype=float), return_counts=True)
                cumulative = np.cumsum(counts) / counts.sum()
                indices = np.unique(np.linspace(0, len(unique) - 1, min(points, len(unique)), dtype=int))
                result["traces"] = [_trace("scatter", mode="lines", x=unique[indices], y=cumulative[indices], line={"shape": "hv", "color": COLOR})]
                result["layout"] = _layout(columns[0], "Fraction ≤ x", yaxis={"title": "Fraction ≤ x", "range": [0, 1]})
                used(clean)
                result["sample_info"].update(rendered_points=len(indices), output_thinned=len(indices) < len(unique))
            elif kind == "qq":
                if len(values) < 3:
                    raise ValidationError("Normal Q-Q plot требует минимум три непустых значения.")
                (theoretical, observed), (slope, intercept, r) = stats.probplot(values, dist="norm", fit=True)
                result["traces"] = [_trace("scatter", mode="markers", x=theoretical, y=observed, name="Observed quantiles", marker={"color": COLOR}),
                                    _trace("scatter", mode="lines", x=theoretical, y=slope * theoretical + intercept, name="Reference fit")]
                result["layout"] = _layout("Standard normal quantiles", columns[0])
                result["statistics"] = {"reference_slope": slope, "reference_intercept": intercept, "reference_r": r}
            elif kind == "kde":
                if len(values) < 3 or np.ptp(values) == 0:
                    raise ValidationError("Kernel density estimate требует минимум три значения с ненулевым разбросом.")
                estimate = stats.gaussian_kde(values)
                grid = np.linspace(values.min(), values.max(), 200)
                result["traces"] = [_trace("scatter", mode="lines", x=grid, y=estimate(grid), line={"color": COLOR})]
                result["layout"] = _layout(columns[0], "Estimated density")
                result["statistics"] = {"bandwidth_factor": estimate.factor, "bandwidth_method": "Scott"}
            elif kind in {"scatter", "scatter3d"}:
                trace = _trace("scatter3d" if kind == "scatter3d" else "scatter", mode="markers",
                               x=selected.iloc[:, 0].to_numpy(), y=selected.iloc[:, 1].to_numpy(),
                               customdata=selected.index.to_numpy(), marker={"color": COLOR, "size": 4, "opacity": .65})
                if kind == "scatter3d":
                    trace["z"] = selected.iloc[:, 2].to_numpy()
                    result["layout"] = {"scene": {"xaxis": {"title": columns[0]}, "yaxis": {"title": columns[1]}, "zaxis": {"title": columns[2]}}}
                else:
                    result["layout"] = _layout(*columns)
                result["traces"] = [trace]
            elif kind == "histogram2d":
                counts, xedges, yedges = np.histogram2d(clean.iloc[:, 0], clean.iloc[:, 1], bins=settings["bins"])
                result["traces"] = [_trace("heatmap", x=(xedges[:-1] + xedges[1:]) / 2,
                                            y=(yedges[:-1] + yedges[1:]) / 2, z=counts.T, colorscale=[[0, "#f4efe3"], [1, COLOR]])]
                result["layout"] = _layout(*columns)
                result["statistics"] = {"xedges": xedges, "yedges": yedges, "counts": counts.T}
                used(clean)
            elif kind == "outliers":
                q1, q3 = clean.iloc[:, 0].quantile([.25, .75]).to_numpy()
                low, high = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
                flags = (selected.iloc[:, 0] < low) | (selected.iloc[:, 0] > high)
                result["traces"] = [_trace("scatter", mode="markers", x=selected.index[~flags].to_numpy(), y=selected.loc[~flags].iloc[:, 0].to_numpy(), name="Inside IQR fences", marker={"color": "#315e54", "size": 4}),
                                    _trace("scatter", mode="markers", x=selected.index[flags].to_numpy(), y=selected.loc[flags].iloc[:, 0].to_numpy(), name="Flagged; inspect before changing", marker={"color": COLOR, "size": 7})]
                result["layout"] = _layout("Original row index", columns[0], shapes=[{"type": "line", "xref": "paper", "x0": 0, "x1": 1, "y0": fence, "y1": fence, "line": {"dash": "dot", "color": COLOR}} for fence in (low, high)])
                result["statistics"] = {"q1": q1, "q3": q3, "lower_fence": low, "upper_fence": high,
                                        "flagged_rows": int(((clean.iloc[:, 0] < low) | (clean.iloc[:, 0] > high)).sum())}
        return _plain(result)
