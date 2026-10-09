"""Leakage-safe regression experiments and real algorithm trajectories."""

from __future__ import annotations

import math
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import orthogonal_mp
from sklearn.model_selection import KFold, TimeSeriesSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, PolynomialFeatures, StandardScaler

from .metrics import MetricRegistry


class TrainingCancelled(RuntimeError):
    """The user cancelled between algorithm steps."""


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return value


def _string_categories(values):
    """Normalize categories inside the exported pipeline, including JSON booleans."""
    result = np.asarray(values, dtype=object).copy()
    missing = pd.isna(result)
    result[missing] = np.nan
    if np.any(~missing):
        result[~missing] = [str(value) for value in result[~missing]]
    return result


class TrainingService:
    """Coordinate data, preprocessing, estimators, metrics and diagnostics."""

    def __init__(self, data_service, model_registry):
        self.data_service = data_service
        self.model_registry = model_registry
        self.metric_registry = MetricRegistry()

    @staticmethod
    def _check(cancelled):
        if cancelled():
            raise TrainingCancelled("Обучение отменено пользователем.")

    @staticmethod
    def _split(n, config, seed):
        fractions = {key: float(config.get(key, default)) for key, default in (("train", 0.6), ("validation", 0.2), ("test", 0.2))}
        if not all(np.isfinite(v) and 0 < v < 1 for v in fractions.values()) or not math.isclose(sum(fractions.values()), 1, abs_tol=1e-8):
            raise ValueError("Доли обучения, проверки настроек и теста должны быть положительными и в сумме давать 1.")
        if n < 10:
            raise ValueError("Нужно хотя бы 10 наблюдений для трех отдельных выборок.")
        indices = np.arange(n)
        shuffle = config.get("shuffle", True)
        if not isinstance(shuffle, bool):
            raise ValueError("shuffle должен быть true или false.")
        if not shuffle:
            end_train = max(2, int(n * fractions["train"]))
            end_validation = end_train + max(1, int(n * fractions["validation"]))
            if end_validation >= n:
                raise ValueError("При выбранных долях в тестовой части не осталось наблюдений.")
            parts = {"train": indices[:end_train], "validation": indices[end_train:end_validation], "test": indices[end_validation:]}
        else:
            # Split the train first, then split the remainder into validation/test.
            train, rest = train_test_split(indices, train_size=fractions["train"], random_state=seed, shuffle=True)
            validation, test = train_test_split(rest, train_size=fractions["validation"] / (1 - fractions["train"]), random_state=seed + 1, shuffle=True)
            parts = {"train": train, "validation": validation, "test": test}
        if len(parts["train"]) < 2 or len(parts["validation"]) < 1 or len(parts["test"]) < 1:
            raise ValueError("Недостаточно наблюдений для выбранного разделения.")
        return parts

    @staticmethod
    def _preprocessor(X, config):
        from .preprocessing import build_preprocessor
        return build_preprocessor(X, config)

    def _dataset_context(self, request):
        bundle = self.data_service.resolve(request["dataset_id"], request.get("target"), request.get("features"))
        group_column = request.get("cv_config", {}).get("group_column") or request.get("split", {}).get("group_column")
        groups = None
        X = bundle.X
        if group_column:
            if group_column == bundle.target_name:
                raise ValueError("Цель нельзя использовать как идентификатор группы.")
            groups = np.asarray(self.data_service.read_column(request["dataset_id"], group_column))
            if pd.isna(groups).any():
                raise ValueError("Идентификаторы групп не должны содержать пропуски.")
            X = X.drop(columns=[group_column], errors="ignore")
        X, numeric = self._clean_features(X)
        return bundle, X, numeric, np.asarray(bundle.y, dtype=float).reshape(-1), groups

    def _partition(self, X, y, groups, request):
        from .validation import split_holdout
        return split_holdout(len(X), request.get("split", {}), int(request.get("seed", 42)),
                             groups=groups, cv_config=request.get("cv_config", {}))

    @staticmethod
    def _linear_parameters(estimator, p):
        inner = getattr(estimator, "estimator_", estimator)
        coef = getattr(inner, "coef_", None)
        if coef is None or np.asarray(coef).size != p:
            return None, None
        return np.asarray(coef, dtype=float).reshape(-1), float(np.asarray(getattr(inner, "intercept_", 0)).reshape(-1)[0])

    def _objective(self, model_id, estimator, X, y):
        objective = getattr(self.model_registry, "objective", None)
        if objective is None:
            return None
        try:
            value = objective(model_id, estimator, X, y)
            return float(value) if value is not None and np.isfinite(value) else None
        except (ValueError, AttributeError, TypeError, ArithmeticError):
            return None

    @staticmethod
    def _clean_features(X):
        X = X.copy()
        numeric = X.select_dtypes(include=[np.number]).columns.tolist()
        for column in numeric:
            X[column] = X[column].replace([np.inf, -np.inf], np.nan)
        for column in X.columns:
            if column not in numeric:
                X[column] = X[column].astype(object).where(X[column].notna(), np.nan)
                present = X[column].notna()
                X.loc[present, column] = X.loc[present, column].astype(str)
        return X, numeric

    def prediction_grid(self, request, pipeline, x_feature, y_feature=None):
        """Predict an arbitrary original-feature slice using an already fitted model."""
        bundle, X, numeric, y, groups = self._dataset_context(request)
        selected = [x_feature] + ([y_feature] if y_feature is not None else [])
        if len(set(selected)) != len(selected) or any(name not in numeric for name in selected):
            raise ValueError("Для сетки выбери один или два разных числовых признака из обученного набора.")
        parts = self._partition(X, y, groups, request)
        grid, _ = self._grid(X, numeric, parts, pipeline, selected)
        if grid is None:
            raise ValueError("Выбранный признак полностью состоит из пропусков в обучающей части.")
        return json_safe(grid)

    @staticmethod
    def _grid(X, numeric, parts, pipeline, selected_features=None):
        if not numeric:
            return None, None
        train = X.iloc[parts["train"]]
        features = selected_features or numeric[:2]
        axes = []
        for name in features:
            column = pd.to_numeric(train[name], errors="coerce").dropna()
            if len(column) == 0:
                return None, None
            low, high = float(column.min()), float(column.max())
            spread = max(high - low, abs(low) * 0.1, 1e-3)
            axes.append(np.linspace(low - 0.06 * spread, high + 0.06 * spread, 36 if len(features) == 2 else 100))
        if len(features) == 2:
            gx, gy = np.meshgrid(*axes)
            values = {features[0]: gx.ravel(), features[1]: gy.ravel()}
            size = gx.size
        else:
            values = {features[0]: axes[0]}
            size = len(axes[0])
        grid = pd.DataFrame(index=np.arange(size))
        for name in X.columns:
            if name in values:
                grid[name] = values[name]
            elif name in numeric:
                median = train[name].median()
                grid[name] = float(median) if pd.notna(median) else 0
            else:
                mode = train[name].dropna().mode()
                grid[name] = mode.iloc[0] if len(mode) else ""
        transformed = pipeline.named_steps["preprocessing"].transform(grid[X.columns])
        predictions = np.asarray(pipeline.named_steps["model"].predict(transformed), dtype=float)
        z = predictions.reshape(len(axes[1]), len(axes[0])) if len(features) == 2 else predictions
        return {"x": axes[0], "y": axes[1] if len(features) == 2 else None, "z": z, "features": features, "note": "Остальные признаки зафиксированы на медианах и наиболее частых категориях обучающей части."}, transformed

    def _surface(self, model_id, estimator, Xt, yt, coef, intercept, names):
        supported = {"ols", "nonnegative", "ridge", "lasso", "elasticnet", "sgd"}
        if coef is None or len(coef) == 0:
            return None, None
        params = estimator.get_params()
        if len(coef) == 1:
            return self._general_surface(model_id, estimator, Xt, yt, coef, intercept, names), self._penalty_profile(model_id, estimator, coef)
        if model_id not in supported or (model_id == "sgd" and params.get("loss", "squared_error") != "squared_error"):
            return self._general_surface(model_id, estimator, Xt, yt, coef, intercept, names), self._penalty_profile(model_id, estimator, coef)
        alpha = float(params.get("alpha", 0))
        penalty = "none"
        ratio = 0.5
        factor = 1.0
        if model_id == "ridge":
            penalty, factor = "l2", float(len(yt))
        elif model_id == "lasso":
            penalty, factor = "l1", 0.5
        elif model_id == "elasticnet":
            penalty, factor, ratio = "elasticnet", 0.5, float(params.get("l1_ratio", 0.5))
        elif model_id == "sgd":
            penalty, factor, ratio = params.get("penalty") or "none", 0.5, float(params.get("l1_ratio", 0.15))
        else:
            alpha = 0
            factor = float(len(yt))
        residual = Xt @ coef + intercept - yt
        base = float(np.mean(residual**2))
        gram = Xt[:, :2].T @ Xt[:, :2] / len(yt)
        linear = Xt[:, :2].T @ residual / len(yt)
        radii = np.maximum(np.abs(coef[:2]) * 0.8, np.maximum(np.std(yt) / np.maximum(np.std(Xt[:, :2], axis=0), 1e-8) * 0.25, 0.5))
        axes = [np.linspace(coef[j] - radii[j], coef[j] + radii[j], 41) for j in range(2)]
        mesh_x, mesh_y = np.meshgrid(*axes)
        dx, dy = mesh_x - coef[0], mesh_y - coef[1]
        loss = factor * (base + 2 * linear[0] * dx + 2 * linear[1] * dy + gram[0, 0] * dx**2 + 2 * gram[0, 1] * dx * dy + gram[1, 1] * dy**2)
        absolute = np.sum(np.abs(coef[2:])) + np.abs(mesh_x) + np.abs(mesh_y)
        squares = np.sum(coef[2:]**2) + mesh_x**2 + mesh_y**2
        if penalty == "l1":
            loss += alpha * absolute
        elif penalty == "l2":
            loss += alpha * squares * (1 if model_id == "ridge" else 0.5)
        elif penalty == "elasticnet":
            loss += alpha * (ratio * absolute + (1 - ratio) * squares / 2)
        if model_id == "nonnegative":
            loss[(mesh_x < 0) | (mesh_y < 0)] = np.nan
        labels = {"ridge": "Сумма квадратов ошибок + α·Σβ²", "lasso": "MSE/2 + α·Σ|β|", "elasticnet": "MSE/2 + α·(r·Σ|β| + (1−r)·Σβ²/2)", "sgd": "MSE/2 + штраф выбранной SGD-модели"}
        location = np.unravel_index(np.nanargmin(loss), loss.shape)
        surface = {"x": axes[0], "y": axes[1], "z": loss, "coef_names": names[:2], "current": {"x": coef[0], "y": coef[1]}, "minimum": {"x": axes[0][location[1]], "y": axes[1][location[0]], "z": loss[location]}, "exact": True, "label": labels.get(model_id, "Сумма квадратов ошибок"), "note": "Точный срез по двум коэффициентам: свободный член и остальные коэффициенты зафиксированы. Отмечен минимум сетки, не решение всей задачи."}
        geometry = {"kind": penalty, "alpha": alpha, "l1_ratio": ratio, "note": "Линия равного штрафа в пространстве двух коэффициентов; радиус иллюстрации равен 1."}
        return surface, geometry

    def _general_surface(self, model_id, estimator, Xt, yt, coef, intercept, names):
        """Use registry's exact objective while caching the unchanged linear terms."""
        univariate = len(coef) == 1
        if univariate and not estimator.get_params().get("fit_intercept", True):
            return None
        centers = np.array([coef[0], intercept]) if univariate else coef[:2]
        columns = np.column_stack((Xt[:, 0], np.ones(len(Xt)))) if univariate else Xt[:, :2]
        radii = np.std(yt) / np.maximum(np.std(columns, axis=0), 1e-8) * 0.25
        if univariate:
            radii[1] = max(np.std(yt) * 0.25, 0.5)
        radius = np.maximum(np.abs(centers) * 0.8, np.maximum(radii, 0.5))
        axes = [np.linspace(centers[j] - radius[j], centers[j] + radius[j], 25) for j in range(2)]
        baseline = Xt @ coef + intercept
        params = estimator.get_params()
        log_link = model_id in {"poisson", "gamma"} or (model_id == "tweedie" and params.get("link", "auto") == "log")
        if model_id == "tweedie" and estimator.get_params().get("link", "auto") == "auto":
            log_link = estimator.get_params().get("power", 0) > 0

        class ObjectiveView:
            def __init__(self, source):
                self.source = source
                self.coef_ = coef.copy()
                self.prediction = baseline
                self.intercept_ = getattr(source, "intercept_", intercept)

            def __getattr__(self, name):
                return getattr(self.source, name)

            def predict(self, ignored):
                return self.prediction

        view = ObjectiveView(estimator)
        exact = self._objective(model_id, estimator, Xt, yt) is not None
        surface = np.full((len(axes[1]), len(axes[0])), np.nan)
        with np.errstate(over="ignore", invalid="ignore"):
            for j, beta_y in enumerate(axes[1]):
                for i, beta_x in enumerate(axes[0]):
                    if univariate:
                        view.coef_[0] = beta_x
                        view.intercept_ = np.array([beta_y]) if np.ndim(getattr(estimator, "intercept_", intercept)) else beta_y
                    else:
                        view.coef_[:2] = [beta_x, beta_y]
                    linear = baseline + (beta_x - centers[0]) * columns[:, 0] + (beta_y - centers[1]) * columns[:, 1]
                    view.prediction = np.exp(linear) if log_link else linear
                    objective = self._objective(model_id, view, Xt, yt) if exact else float(np.mean((yt - view.prediction)**2))
                    if model_id == "nonnegative" and (beta_x < 0 or (not univariate and beta_y < 0)):
                        continue
                    if objective is not None and np.isfinite(objective):
                        surface[j, i] = objective
        if not np.any(np.isfinite(surface)):
            return None
        minimum = np.unravel_index(np.nanargmin(surface), surface.shape)
        note = "Свободный член, остальные коэффициенты и вспомогательные параметры (например масштаб Huber и веса Adaptive Lasso) зафиксированы. Отмечен минимум сетки этого среза."
        if univariate:
            note = "Меняются единственный коэффициент и свободный член. Вспомогательные параметры модели зафиксированы. Отмечен минимум сетки."
        if not exact:
            note = "Показана диагностическая MSE. Это не функция, которую оптимизирует данный алгоритм. " + note
        return {"x": axes[0], "y": axes[1], "z": surface, "coef_names": [names[0], "Свободный член β₀"] if univariate else names[:2], "current": {"x": centers[0], "y": centers[1]}, "minimum": {"x": axes[0][minimum[1]], "y": axes[1][minimum[0]], "z": surface[minimum]}, "exact": exact, "label": "Функция оптимизации модели: точный срез" if exact else "Диагностическая MSE", "note": note}

    def _penalty_profile(self, model_id, estimator, coef):
        supported = {"l0", "scad", "mcp", "weighted_lasso", "adaptive_lasso", "group_lasso", "sparse_group_lasso", "fused_lasso", "tikhonov", "sqrt_lasso", "lasso", "lassolars", "elasticnet", "ridge", "quantile", "sgd"}
        if model_id not in supported:
            return None
        alpha = float(getattr(estimator, "alpha", 0))
        radius = min(max(abs(coef[0]) * 1.3, alpha * float(getattr(estimator, "gamma", 3)) * 1.2, 1), 10000)
        axis = np.linspace(-radius, radius, 201)
        zero_X = np.zeros((1, len(coef)))
        zero_y = np.zeros(1)

        class PenaltyView:
            def __init__(self, source):
                self.source = source
                self.coef_ = np.zeros_like(coef)

            def __getattr__(self, name):
                return getattr(self.source, name)

            def predict(self, ignored):
                return zero_y

        view = PenaltyView(estimator)
        baseline = self._objective(model_id, view, zero_X, zero_y)
        if baseline is None:
            return None
        values = []
        for beta in axis:
            view.coef_[0] = beta
            value = self._objective(model_id, view, zero_X, zero_y)
            values.append(None if value is None else value - baseline)
        return {"kind": "profile", "x": axis, "y": values, "label": "Штраф при изменении одного коэффициента", "note": "Первый коэффициент меняется, все остальные равны нулю. Веса и группы взяты из обученной модели. Показан сам штраф; ошибки предсказаний здесь нет."}

    def run(self, request, progress=lambda event: None, cancelled=lambda: False):
        started = time.perf_counter()
        self._check(cancelled)
        progress({"progress": 0.01, "message": "Читаем данные и проверяем настройки."})
        bundle, X, numeric_columns, y, groups = self._dataset_context(request)
        if len(X) != len(y) or not np.all(np.isfinite(y)):
            raise ValueError("Целевая переменная должна быть числовой, без пропусков и бесконечностей.")
        if len(X) > 100000:
            raise ValueError("Интерактивное обучение ограничено 100000 строками. Загрузить описание набора можно полностью; для опыта выбери меньшую выборку.")
        if not X.columns.is_unique:
            raise ValueError("Названия признаков должны быть уникальными.")
        seed = int(request.get("seed", 42))
        if not 0 <= seed < 2**32 - 1:
            raise ValueError("seed должен быть целым числом от 0 до 4294967294.")
        parts = self._partition(X, y, groups, request)
        preprocessor, numeric, categorical = self._preprocessor(X, request.get("preprocessing", {}))
        if request.get("preprocessing", {}).get("imputation", "median" if request.get("preprocessing", {}).get("impute", True) else "none") == "none" and X.isna().any().any():
            raise ValueError("В признаках есть пропуски. Включи заполнение пропусков или исправь данные.")
        self._check(cancelled)
        train_X, train_y = X.iloc[parts["train"]], y[parts["train"]]
        degree = int(request.get("preprocessing", {}).get("degree", 1))
        estimated_width = math.comb(len(numeric) + degree, degree) - 1 if numeric else 0
        estimated_width += sum(max(1, train_X[column].nunique(dropna=True)) for column in categorical)
        if estimated_width > 2000 or estimated_width * len(train_X) > 15000000:
            raise ValueError("Преобразования создадут более 2000 признаков или 15 миллионов обучающих ячеек. Сократи категории, признаки либо степень полинома.")
        Xt = preprocessor.fit_transform(train_X, train_y)
        if Xt.shape[1] > 2000 or Xt.size > 15000000:
            raise ValueError("После преобразований превышен предел 2000 признаков или 15 миллионов ячеек. Сократи признаки, категории или степень полинома.")
        feature_names = preprocessor.get_feature_names_out().tolist()
        transformed = {name: (Xt if name == "train" else preprocessor.transform(X.iloc[indices])) for name, indices in parts.items()}
        from .preprocessing import ResamplingService
        original_train_y = train_y.copy()
        if request.get("resampling", {}).get("method", "none") != "none" and (request.get("cv_config", {}).get("strategy") == "timeseries" or request.get("split", {}).get("shuffle") is False):
            raise ValueError("Для временных рядов пересэмплирование выключено: оно меняет хронологию наблюдений.")
        Xt, train_y, resampling = ResamplingService().fit_resample(
            Xt, train_y, request.get("resampling", {}), seed=seed, categorical=bool(categorical))
        model_id = request.get("model", "ridge")
        params = request.get("params", {})
        estimator = self.model_registry.create(model_id, params, seed)
        if model_id == "l0" and Xt.shape[1] > 12:
            raise ValueError("Точный L₀ перебирает подмножества: допустимо не более 12 преобразованных признаков.")
        if model_id == "theilsen" and (Xt.shape[1] > 30 or len(train_y) > 15000):
            raise ValueError("Theil–Sen ограничен 30 преобразованными признаками и 15000 обучающими строками. Уменьши набор для интерактивного опыта.")
        if model_id == "poisson" and (np.any(train_y < 0) or not np.any(train_y > 0)):
            raise ValueError("Регрессии Пуассона нужна неотрицательная обучающая цель и хотя бы одно положительное значение.")
        if model_id == "gamma" and np.any(train_y <= 0):
            raise ValueError("Гамма-регрессии нужна строго положительная обучающая цель.")
        if model_id == "tweedie":
            power = float(estimator.get_params().get("power", 0))
            if power >= 2 and np.any(train_y <= 0):
                raise ValueError("При power≥2 регрессии Твиди нужна строго положительная обучающая цель.")
            if 1 <= power < 2 and (np.any(train_y < 0) or not np.any(train_y > 0)):
                raise ValueError("При 1≤power<2 регрессии Твиди нужна неотрицательная цель и хотя бы одно положительное значение.")
        display_indices = np.arange(len(X))
        if len(X) > 600:
            display_indices = np.sort(np.random.default_rng(seed).choice(len(X), 600, replace=False))
        display_Xt = preprocessor.transform(X.iloc[display_indices])
        trace = []
        loss_history = []
        captured_warnings = []
        trace_kind, trace_label = "final", "Итог обучения: промежуточные коэффициенты реализация не публикует."

        def frame(step, coef=None, intercept=None, use_estimator=True):
            if coef is None:
                coef, intercept = self._linear_parameters(estimator, Xt.shape[1])
            if use_estimator:
                train_pred = estimator.predict(transformed["train"])
                validation_pred = estimator.predict(transformed["validation"])
                display_pred = estimator.predict(display_Xt)
                objective = self._objective(model_id, estimator, Xt, train_y)
            else:
                train_pred = transformed["train"] @ coef + intercept
                validation_pred = transformed["validation"] @ coef + intercept
                display_pred = display_Xt @ coef + intercept
                objective = None
            item = {"step": int(step), "coef": coef.tolist() if coef is not None else [], "intercept": intercept, "train_loss": float(np.mean((original_train_y - train_pred)**2)), "validation_loss": float(np.mean((y[parts["validation"]] - validation_pred)**2)), "objective": objective, "predicted": display_pred.tolist()}
            fit_pred = estimator.predict(Xt) if use_estimator else Xt @ coef + intercept
            item["fit_loss"] = float(np.mean((train_y - fit_pred)**2))
            return item

        progress({"progress": 0.08, "message": "Преобразования обучены только на обучающей части. Запускаем модель."})
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            if model_id == "sgd":
                epochs = int(request.get("epochs", 100))
                if not 1 <= epochs <= 1000:
                    raise ValueError("Число эпох SGD должно быть от 1 до 1000.")
                estimator.set_params(max_iter=epochs, tol=None)
                trace_kind, trace_label = "epochs", "Настоящие эпохи SGD: каждый кадр записан после partial_fit на обучающей части."
                stride = max(1, math.ceil(epochs / 160))
                for epoch in range(epochs):
                    self._check(cancelled)
                    # partial_fit performs one real epoch, keeping optimizer state.
                    estimator.partial_fit(Xt, train_y)
                    if not np.all(np.isfinite(estimator.coef_)) or not np.all(np.isfinite(estimator.intercept_)):
                        raise ValueError("SGD разошелся: коэффициенты стали бесконечными. Уменьши шаг eta0 и включи масштабирование.")
                    item = frame(epoch + 1)
                    loss_history.append({key: item[key] for key in ("step", "train_loss", "validation_loss", "objective")})
                    record_frame = epoch % stride == 0 or epoch == epochs - 1
                    if record_frame:
                        trace.append(item)
                    event = {"progress": 0.08 + 0.57 * (epoch + 1) / epochs, "message": f"SGD: эпоха {epoch + 1} из {epochs}"}
                    if record_frame:
                        event["frame"] = json_safe(item)
                    progress(event)
            elif model_id in {"lars", "lassolars"}:
                self._check(cancelled)
                estimator.fit(Xt, train_y)
                path = np.asarray(getattr(estimator, "coef_path_", []))
                if path.ndim == 2 and path.size:
                    trace_kind, trace_label = "active_set", "Настоящий путь Lars: узлы изменения активного набора признаков, а не градиентные эпохи."
                    if model_id == "lassolars":
                        trace_label = "Настоящий путь LassoLars по α: узлы изменения активного набора. Это решения с разными штрафами, а не эпохи одной задачи."
                    center_X = Xt.mean(axis=0) if estimator.get_params().get("fit_intercept", True) else np.zeros(Xt.shape[1])
                    center_y = train_y.mean() if estimator.get_params().get("fit_intercept", True) else 0
                    for step in range(path.shape[1]):
                        self._check(cancelled)
                        coef = path[:, step]
                        item = frame(step, coef, float(center_y - center_X @ coef), False)
                        alphas = getattr(estimator, "alphas_", [])
                        if len(alphas) > step:
                            item["alpha"] = float(alphas[step])
                            if model_id == "lassolars":
                                item["objective"] = item["fit_loss"] / 2 + item["alpha"] * float(np.abs(coef).sum())
                        trace.append(item)
                        loss_history.append({key: item[key] for key in ("step", "train_loss", "validation_loss", "objective")})
            elif model_id == "ransac":
                trace_kind, trace_label = "candidates", "Настоящие пробные модели RANSAC и выбранная итоговая модель. Ошибка кандидатов может расти; RANSAC ищет согласованную группу, а не убывающую MSE."
                candidate_count = 0
                stride = max(1, math.ceil(estimator.get_params().get("max_trials", 100) / 150))
                original_callback = estimator.get_params().get("is_model_valid")

                def capture_candidate(candidate, subset_X, subset_y):
                    nonlocal candidate_count
                    self._check(cancelled)
                    candidate_count += 1
                    valid = original_callback(candidate, subset_X, subset_y) if original_callback else True
                    if valid and candidate_count % stride == 0:
                        candidate_coef, candidate_intercept = self._linear_parameters(candidate, Xt.shape[1])
                        if candidate_coef is not None:
                            item = frame(candidate_count, candidate_coef, candidate_intercept, False)
                            item["kind"] = "candidate"
                            trace.append(item)
                            progress({"progress": min(0.6, 0.08 + 0.52 * candidate_count / estimator.get_params().get("max_trials", 100)), "message": f"RANSAC: настоящий кандидат {candidate_count}", "frame": json_safe(item)})
                    return valid

                estimator.set_params(is_model_valid=capture_candidate)
                try:
                    estimator.fit(Xt, train_y)
                finally:
                    # Do not leave a local callback inside the exported estimator.
                    estimator.set_params(is_model_valid=original_callback)
                selected = frame(candidate_count + 1)
                selected["kind"] = "selected"
                trace.append(selected)
                loss_history = [{key: item[key] for key in ("step", "train_loss", "validation_loss", "objective")} for item in trace]
            elif model_id == "omp":
                self._check(cancelled)
                fit_intercept = estimator.get_params().get("fit_intercept", True)
                center_X = Xt.mean(axis=0) if fit_intercept else np.zeros(Xt.shape[1])
                center_y = train_y.mean() if fit_intercept else 0
                omp_params = estimator.get_params()
                n_nonzero = omp_params.get("n_nonzero_coefs")
                if n_nonzero is not None and n_nonzero > Xt.shape[1]:
                    raise ValueError("Количество ненулевых коэффициентов OMP не может превышать число преобразованных признаков.")
                path = np.asarray(orthogonal_mp(Xt - center_X, train_y - center_y, n_nonzero_coefs=n_nonzero, tol=omp_params.get("tol"), precompute=omp_params.get("precompute", "auto"), return_path=True))
                estimator.fit(Xt, train_y)
                if path.ndim == 1:
                    path = path[:, None]
                if path.ndim == 2 and path.shape[0] == Xt.shape[1]:
                    trace_kind, trace_label = "active_set", "Настоящие шаги OMP: добавление признака и пересчет выбранных коэффициентов."
                    trace.append(frame(0, np.zeros(Xt.shape[1]), float(center_y), False))
                    for step in range(path.shape[1]):
                        self._check(cancelled)
                        coef = path[:, step]
                        trace.append(frame(step + 1, coef, float(center_y - center_X @ coef), False))
                    loss_history = [{key: item[key] for key in ("step", "train_loss", "validation_loss", "objective")} for item in trace]
            else:
                self._check(cancelled)
                estimator.fit(Xt, train_y)
            self._check(cancelled)
            if not trace:
                trace.append(frame(1))
                loss_history = [{key: trace[0][key] for key in ("step", "train_loss", "validation_loss", "objective")}]
            captured_warnings.extend(str(w.message) for w in recorded)
        pipeline = Pipeline([("preprocessing", preprocessor), ("model", estimator)])
        selected_metrics = request.get("metrics", ["mse", "rmse", "mae", "r2"])
        expression = request.get("custom_metric")
        metric_params = request.get("metric_params", {})
        metrics, metric_details = {}, {}
        all_predictions = np.empty(len(y))
        labels = np.empty(len(y), dtype=object)
        for name, indices in parts.items():
            prediction = np.asarray(estimator.predict(transformed[name]), dtype=float)
            if not np.all(np.isfinite(prediction)):
                raise ValueError("Модель выдала бесконечные предсказания. Проверь масштаб признаков, шаг обучения и силу регуляризации.")
            all_predictions[indices], labels[indices] = prediction, name
            metrics[name], metric_details[name] = self.metric_registry.evaluate(y[indices], prediction, selected_metrics, expression, metric_params)
        progress({"progress": 0.69, "message": "Считаем метрики и диагностические графики."})
        coef, intercept = self._linear_parameters(estimator, Xt.shape[1])
        surface, geometry = self._surface(model_id, estimator, Xt, train_y, coef, intercept, feature_names)
        self._check(cancelled)
        grid, transformed_grid = self._grid(X, numeric, parts, pipeline)
        if grid is not None and Xt.shape[1] <= 300:
            for item in trace:
                self._check(cancelled)
                if len(item["coef"]) != Xt.shape[1] or item["intercept"] is None:
                    continue
                grid_pred = transformed_grid @ np.asarray(item["coef"]) + item["intercept"]
                if model_id in {"poisson", "gamma"} or (model_id == "tweedie" and (estimator.get_params().get("link", "auto") == "log" or (estimator.get_params().get("link", "auto") == "auto" and estimator.get_params().get("power", 0) > 0))):
                    with np.errstate(over="ignore"):
                        grid_pred = np.exp(grid_pred)
                shape = (len(grid["y"]), len(grid["x"])) if grid["y"] is not None else (len(grid["x"]),)
                item["grid"] = dict(grid, z=grid_pred.reshape(shape), note=grid["note"] + " Предсказания соответствуют коэффициентам этого настоящего кадра.")
        regularization_path = self._regularization_path(request, model_id, estimator, Xt, train_y, feature_names, progress, cancelled, captured_warnings)
        cv = self._cross_validation(request, X.iloc[parts["train"]], original_train_y, preprocessor, estimator, selected_metrics, expression, metric_params, progress, cancelled, groups[parts["train"]] if groups is not None else None)
        learning_curve = self._learning_curve(request, X, y, parts, preprocessor, estimator, cancelled)
        importance = self._importance(request, X.iloc[parts["validation"]], y[parts["validation"]], pipeline, selected_metrics, expression, metric_params, cancelled)
        progress({"progress": 0.98, "message": "Сохраняем воспроизводимую модель и результаты."})
        self._check(cancelled)
        if request.get("artifact_path"):
            path = Path(request["artifact_path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + ".tmp")
            try:
                joblib.dump(pipeline, temporary)
                self._check(cancelled)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        correlation = train_X[numeric].corr() if numeric else pd.DataFrame()
        model_spec = self.model_registry.spec(model_id)
        inliers = None
        if hasattr(estimator, "inlier_mask_"):
            global_inliers = np.empty(len(X), dtype=object)
            global_inliers[:] = None
            if len(estimator.inlier_mask_) == len(parts["train"]) and resampling.get("method", "none") == "none":
                global_inliers[parts["train"]] = np.asarray(estimator.inlier_mask_, dtype=bool)
            inliers = global_inliers[display_indices].tolist()
        if len(display_indices) < len(X):
            captured_warnings.append(f"Графики показывают воспроизводимую выборку из {len(display_indices)} строк. Модель и метрики рассчитаны по всем {len(X)} строкам.")
        if categorical:
            captured_warnings.append("Неизвестные категории проверки и теста кодируются нулями; словарь категорий создан только на обучающей части.")
        if coef is not None and request.get("preprocessing", {}).get("scale", True):
            captured_warnings.append("Коэффициенты относятся к преобразованным, стандартизованным признакам. Плоскость предсказаний построена в исходных единицах.")
        if surface is None:
            captured_warnings.append("Для этой модели точная поверхность ее функции оптимизации не построена. MSE на графике ошибок остается диагностической метрикой.")
        result = {
            "model": model_id, "model_name": model_spec.get("name", model_id), "dataset_name": bundle.name, "target_name": bundle.target_name,
            "metrics": metrics, "metric_details": metric_details, "feature_names": feature_names,
            "coefficients": coef.tolist() if coef is not None else [], "intercept": intercept,
            "trace": trace, "loss_history": loss_history, "trace_kind": trace_kind, "trace_label": trace_label, "trace_loss_label": "MSE · средняя квадратичная ошибка",
            "warnings": list(dict.fromkeys(captured_warnings)),
            "predictions": {"actual": y[display_indices], "predicted": all_predictions[display_indices], "residual": y[display_indices] - all_predictions[display_indices], "split": labels[display_indices], "indices": display_indices},
            "plot_data": {"X": X.iloc[display_indices][numeric].to_numpy(dtype=float), "y": y[display_indices], "split": labels[display_indices], "feature_names": numeric, "sampled": len(display_indices) < len(X)},
            "prediction_grid": grid, "regularization_path": regularization_path,
            "objective_surface": surface, "penalty_geometry": geometry,
            "diagnostics": {"correlation": {"feature_names": numeric, "matrix": correlation.to_numpy()}, "condition_number": float(np.linalg.cond(Xt)) if Xt.size < 3000000 and Xt.shape[1] < 200 else None, "rank": int(np.linalg.matrix_rank(Xt)) if Xt.size < 3000000 and Xt.shape[1] < 200 else None, "bayesian_evidence": getattr(estimator, "scores_", None), "learning_curve": learning_curve, "permutation_importance": importance, "ransac_inliers": inliers},
            "split": {"counts": {name: len(indices) for name, indices in parts.items()}, "indices": {name: indices for name, indices in parts.items()}, "shuffle": False if request.get("cv_config", {}).get("strategy") == "timeseries" else request.get("split", {}).get("shuffle", True)},
            "resampling": resampling, "cv": cv, "timing": {"seconds": time.perf_counter() - started},
            "preprocessing": {"scale": request.get("preprocessing", {}).get("scale", True), "degree": request.get("preprocessing", {}).get("degree", 1), "impute": request.get("preprocessing", {}).get("impute", True), "numeric_features": numeric, "categorical_features": categorical, "fitted_on": "train", "transformed_features": feature_names, "config": request.get("preprocessing", {})},
        }
        progress({"progress": 1.0, "message": "Эксперимент готов."})
        return json_safe(result)

    def _regularization_path(self, request, model_id, estimator, Xt, y, names, progress, cancelled, output_warnings):
        if not request.get("regularization_path", True):
            return None
        supported = {"ridge", "lasso", "elasticnet", "lassolars", "adaptive_lasso", "group_lasso", "sparse_group_lasso", "fused_lasso", "tikhonov", "scad", "mcp", "sqrt_lasso", "weighted_lasso"}
        if model_id not in supported or "alpha" not in estimator.get_params():
            return None
        if model_id not in {"ridge", "lasso", "elasticnet", "lassolars"} and (len(y) > 5000 or Xt.shape[1] > 100):
            output_warnings.append("Путь регуляризации пропущен: для экспериментальных штрафов предел 5000 обучающих строк и 100 признаков.")
            return None
        current = max(float(estimator.get_params().get("alpha", 1)), 1e-5)
        count = 18 if model_id in {"ridge", "lasso", "elasticnet", "lassolars"} else 8
        alphas = np.geomspace(current * 0.01, current * 100, count)
        coefficients = []
        valid_alphas = []
        from .validation import bounded_jobs, single_threaded
        from threadpoolctl import threadpool_limits

        n_jobs = bounded_jobs(request.get("n_jobs", 1))

        def fit_alpha(alpha):
            self._check(cancelled)
            candidate = single_threaded(estimator).set_params(alpha=float(alpha))
            try:
                candidate.fit(Xt, y)
                self._check(cancelled)
                coef, _ = self._linear_parameters(candidate, Xt.shape[1])
                return float(alpha), coef, None
            except TrainingCancelled:
                raise
            except (ValueError, RuntimeError, ArithmeticError) as exc:
                return float(alpha), None, str(exc)

        # One native-thread limit covers all workers. Nested BLAS pools would
        # otherwise multiply n_jobs; threads also work inside daemon job workers.
        with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            with joblib.Parallel(n_jobs=n_jobs, backend="threading", return_as="generator", pre_dispatch=n_jobs) as parallel:
                completed = parallel(joblib.delayed(fit_alpha)(alpha) for alpha in alphas)
                try:
                    for j, (alpha, coef, error) in enumerate(completed):
                        self._check(cancelled)
                        if coef is not None and np.all(np.isfinite(coef)):
                            coefficients.append(coef.tolist())
                            valid_alphas.append(alpha)
                        if error:
                            output_warnings.append(f"Путь α: значение {alpha:.3g} пропущено ({error}).")
                        progress({"progress": 0.7 + 0.13 * (j + 1) / len(alphas), "message": f"Отдельный опыт с α: {j + 1} из {len(alphas)}; используется только обучение."})
                finally:
                    completed.close()
            output_warnings.extend(str(w.message) for w in recorded)
        return {"alphas": valid_alphas, "coefficients": coefficients, "feature_names": names, "fitted_on": "train", "n_jobs": n_jobs, "note": "Каждая точка — новая задача с другим α. Это не история обучения одной модели."} if valid_alphas else None

    def _cross_validation(self, request, X, y, preprocessor, estimator, selection, expression, metric_params, progress, cancelled, groups=None):
        from .validation import evaluate_cv
        from .pipeline import RegressionPipeline
        config = request.get("cv_config")
        if not config:
            folds = int(request.get("cv", 0))
            if folds == 0:
                return None
            config = {"strategy": "kfold" if request.get("split", {}).get("shuffle", True) else "timeseries", "folds": folds}
        if config.get("strategy", "none") == "none" or config.get("enabled") is False:
            return None
        pipeline = RegressionPipeline([("preprocessing", clone(preprocessor)), ("model", clone(estimator))],
                                      resampling=request.get("resampling", {}), seed=int(request.get("seed", 42)))
        def cv_progress(event):
            progress(dict(event, progress=.84 + .12 * float(event.get("progress", 0))))
        return evaluate_cv(pipeline, X, y, config, selection, expression, metric_params,
                           groups=groups, seed=int(request.get("seed", 42)), progress=cv_progress,
                           cancelled=cancelled, n_jobs=request.get("n_jobs", 1))

    def _learning_curve(self, request, X, y, parts, preprocessor, estimator, cancelled):
        if not request.get("learning_curve", False):
            return None
        train = parts["train"]
        validation = parts["validation"]
        order = np.random.default_rng(int(request.get("seed", 42))).permutation(train) if request.get("split", {}).get("shuffle", True) and request.get("cv_config", {}).get("strategy") != "timeseries" else train
        sizes = np.unique(np.linspace(max(10, len(train) // 5), len(train), 5).astype(int))
        output = []
        for size in sizes:
            if size > len(train) or size < 2:
                continue
            self._check(cancelled)
            selected = order[:size]
            from .pipeline import RegressionPipeline
            candidate = RegressionPipeline([("preprocessing", clone(preprocessor)), ("model", clone(estimator))], resampling=request.get("resampling", {}), seed=int(request.get("seed", 42)))
            try:
                candidate.fit(X.iloc[selected], y[selected])
                train_pred = candidate.predict(X.iloc[selected])
                validation_pred = candidate.predict(X.iloc[validation])
                output.append({"train_size": int(size), "train_mse": float(np.mean((y[selected] - train_pred)**2)), "validation_mse": float(np.mean((y[validation] - validation_pred)**2))})
            except (ValueError, RuntimeError) as exc:
                output.append({"train_size": int(size), "train_mse": None, "validation_mse": None, "reason": str(exc)})
        return {"points": output, "note": "Отдельные модели на вложенных частях train. Преобразования обучаются заново; общий validation используется для диагностики, test не используется."}

    def _importance(self, request, X, y, pipeline, selection, expression, metric_params, cancelled):
        if not request.get("permutation_importance", False):
            return None
        self._check(cancelled)
        if X.shape[1] > 40 or len(X) > 15000:
            return {"reason": "Перестановочная важность ограничена 40 признаками и 15000 строками validation."}
        from sklearn.inspection import permutation_importance
        from threadpoolctl import threadpool_limits
        from .validation import bounded_jobs

        n_jobs = bounded_jobs(request.get("n_jobs", 1))
        workers = {name: 1 for name, value in pipeline.get_params(deep=True).items()
                   if (name == "n_jobs" or name.endswith("__n_jobs")) and value != 1}
        if workers:
            # clone() would discard the fitted preprocessing/model. A copy keeps
            # learned parameters and leaves the exported fitted pipeline intact.
            from copy import deepcopy
            pipeline = deepcopy(pipeline).set_params(**workers)

        selected = [selection] if isinstance(selection, str) else list(selection or ["mse"])
        metric = next((name for name in selected if name not in {"all", "custom"}), "mse")
        directions = {entry["id"]: entry["direction"] for entry in self.metric_registry.catalogue()}
        direction = directions.get(metric, "min")

        def score(estimator, samples, target):
            self._check(cancelled)
            values, details = self.metric_registry.evaluate(target, estimator.predict(samples), [metric], expression, metric_params)
            self._check(cancelled)
            value = values[metric]
            if value is None:
                raise ValueError(details[metric]["reason"])
            return value if direction == "max" else -value

        try:
            with threadpool_limits(limits=1), joblib.parallel_backend("threading"):
                result = permutation_importance(pipeline, X, y, scoring=score, n_repeats=5, random_state=int(request.get("seed", 42)), n_jobs=n_jobs)
            self._check(cancelled)
            return {"feature_names": X.columns.tolist(), "mean": result.importances_mean, "std": result.importances_std, "metric": metric, "fitted_on": "validation", "n_jobs": n_jobs, "note": "Падение качества при перемешивании исходного признака в validation. Коррелированные признаки могут взаимно заменяться; отрицательная важность возможна."}
        except ValueError as exc:
            return {"reason": str(exc)}
