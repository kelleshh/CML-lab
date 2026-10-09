"""Numerical adapter for fresh, train-scoped recipe previews."""

from copy import deepcopy

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import LabelEncoder

from .ml.engine import ExperimentEngine, plain
from .ml.preparation import PreparationCatalogue, build_preprocessor, fit_resample, validate_preparation_config
from .ml.splitting import holdout


MAX_PREVIEW_ROWS = 100
MAX_PREVIEW_FEATURES = 100


def _histogram(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"counts": [], "centers": [], "edges": []}
    counts, edges = np.histogram(values, bins=min(20, max(1, int(np.sqrt(len(values))))))
    return {"counts": counts.tolist(), "centers": ((edges[:-1] + edges[1:]) / 2).tolist(), "edges": edges.tolist()}


def _matrix_rows(values, names):
    shown = values[:MAX_PREVIEW_ROWS, :len(names)]
    if sparse.issparse(shown):
        shown = shown.toarray()
    return [{name: float(value) for name, value in zip(names, row)} for row in np.asarray(shown)]


def _statistics(values, names):
    shown = values[:, :len(names)]
    if sparse.issparse(shown):
        means = np.asarray(shown.mean(axis=0)).ravel()
        squares = np.asarray(shown.multiply(shown).mean(axis=0)).ravel()
        std = np.sqrt(np.maximum(0, squares - means**2))
    else:
        means = np.mean(shown, axis=0)
        std = np.std(shown, axis=0)
    return means, std


class PreparationPreviewGateway:
    def __init__(self, data_gateway, algorithm_catalogue):
        self._engine = ExperimentEngine(data_gateway, algorithm_catalogue)
        self._catalogue = PreparationCatalogue()

    def stages(self):
        return self._catalogue.stages()

    def samplers(self):
        return self._catalogue.samplers()

    def validate_config(self, config, task=None):
        return validate_preparation_config(config, task)

    def preview(self, resolved_spec):
        spec = deepcopy(resolved_spec)
        spec.setdefault("seed", 42)
        X, y, context = self._engine.prepare_data(spec)
        if len(X) * len(X.columns) > 5_000_000:
            raise ValueError("Предпросмотр ограничен пятью миллионами исходных ячеек.")
        unsupervised = spec["task"] in {"clustering", "anomaly", "reduction"}
        if unsupervised:
            train = np.arange(len(X))
            scope = "working_dataset"
        else:
            train = holdout(spec, len(X), y, context.get("groups"), context.get("origins"), context.get("target_times"))["train"]
            scope = "train"
        Xtrain = X.iloc[train].copy()
        original_target = None if y is None else np.asarray(y)[train]
        fit_target = original_target
        classes = None
        if spec["task"] == "classification":
            encoder = LabelEncoder().fit(original_target)
            fit_target = encoder.transform(original_target)
            classes = encoder.classes_.tolist()
        config = deepcopy(spec.get("preprocessing") or {})
        if context.get("groups") is not None:
            config["split_kind"] = "group"
        elif spec["task"] in {"forecasting", "panel"}:
            config["split_kind"] = "timeseries"
        transformer = build_preprocessor(Xtrain, config, task=spec["task"], seed=spec["seed"])
        Xt = transformer.fit_transform(Xtrain, fit_target)
        from .ml.feature_names import feature_names
        names = feature_names(transformer, Xt.shape[1])
        if len(names) != Xt.shape[1]:
            raise ValueError("Имена подготовленных признаков не совпадают с фактической матрицей.")
        values = Xt.data if sparse.issparse(Xt) else np.asarray(Xt)
        if not np.isfinite(values).all():
            raise ValueError("Подготовка оставила пропуски/бесконечности. Заполните пропуски либо выберите другой рецепт.")
        sampling = config.get("resampling") or spec.get("resampling") or {"method": "none"}
        if unsupervised:
            if sampling.get("method", "none") != "none":
                raise ValueError("Классификационные/регрессионные пересэмплировщики не применяются к этой задаче.")
            Xsampled, ysampled = Xt, None
            summary = {"method": "none", "before": len(train), "after": len(train), "fitted_on": scope, "synthetic": False}
        else:
            if context.get("weights") is not None and sampling.get("method", "none") != "none":
                raise ValueError("Веса и пересэмплирование требуют отдельного правила переноса весов.")
            Xsampled, ysampled, summary = fit_resample(Xt, fit_target, sampling, task=spec["task"], seed=spec["seed"],
                categorical=any(not pd.api.types.is_numeric_dtype(dtype) for dtype in Xtrain.dtypes),
                temporal=spec["task"] in {"forecasting", "panel"})
        shown_names = names[:MAX_PREVIEW_FEATURES]
        means, std = _statistics(Xt, shown_names)
        original_preview = Xtrain.head(MAX_PREVIEW_ROWS).astype(object)
        original_rows = original_preview.where(pd.notna(original_preview), None).to_dict(orient="records")
        raw_histograms = {name: _histogram(Xtrain[name]) for name in Xtrain.columns if pd.api.types.is_numeric_dtype(Xtrain[name])}
        # Limit histogram materialization for sparse tables to at most 2000x100.
        histogram_matrix = Xt[:2000, :len(shown_names)]
        if sparse.issparse(histogram_matrix):
            histogram_matrix = histogram_matrix.toarray()
        prepared_histograms = {name: _histogram(np.asarray(histogram_matrix)[:, index]) for index, name in enumerate(shown_names)}
        paired_target = [] if original_target is None else original_target[:MAX_PREVIEW_ROWS].tolist()
        if classes is not None and ysampled is not None:
            sampled_target = [classes[int(value)] for value in np.asarray(ysampled)[:2000]]
        else:
            sampled_target = [] if ysampled is None else np.asarray(ysampled)[:2000].tolist()
        note = "Статистики и преобразования обучены заново только на train; validation/test не изменяются. Строки до/после имеют одинаковые исходные индексы."
        if unsupervised:
            note = "Предпросмотр обучения без учителя использует всю выбранную рабочую таблицу. Это не оценка новых наблюдений на holdout."
        elif spec["task"] in {"forecasting", "panel"}:
            note += " Столбцы до обычной подготовки уже содержат past-only лаги и окна; строки с недостаточной историей и пересекающиеся с границей цели исключены."
        if len(names) > len(shown_names):
            note += f" Показаны первые {len(shown_names)} из {len(names)} признаков; обученный рецепт сохраняет все."
        result = {"fitted_on": scope, "train_rows": len(train), "original_features": list(Xtrain.columns),
            "features": shown_names, "all_features": names, "feature_count": len(names),
            "indices": np.asarray(context["indices"])[train[:MAX_PREVIEW_ROWS]].tolist(),
            "original_rows": original_rows, "rows": _matrix_rows(Xt, shown_names),
            "target": {"name": spec.get("target"), "values": paired_target, "classes": classes},
            "means": means.tolist(), "std": std.tolist(),
            "statistics": [{"name": name, "mean": float(mean), "std": float(deviation)} for name, mean, deviation in zip(shown_names, means, std)],
            "histograms": {"before": raw_histograms, "after": prepared_histograms},
            "sampling": {"summary": summary, "target_before": [] if original_target is None else original_target[:2000].tolist(),
                "target_after": sampled_target, "rows": _matrix_rows(Xsampled, shown_names), "features": shown_names},
            "note": note, "warnings": context.get("warnings", [])}
        return plain(result)
