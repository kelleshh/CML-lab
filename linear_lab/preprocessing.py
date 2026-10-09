"""Train-only feature engineering and regression-specific resampling.

All learned transformations implement the sklearn estimator contract. Creating a
preprocessor never computes dataset statistics: only ``fit`` does, so callers can
clone the same object independently inside cross-validation folds.
"""

from __future__ import annotations

from contextlib import contextmanager
from functools import partial
import importlib
import math
import random
import threading

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, VarianceThreshold, r_regression, mutual_info_regression
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer, MaxAbsScaler, MinMaxScaler, OneHotEncoder,
    PolynomialFeatures, PowerTransformer, QuantileTransformer, RobustScaler,
    StandardScaler,
)
from sklearn.utils.validation import check_is_fitted


def _string_categories(values):
    """Use identical category normalization for upload, CV and JSON prediction."""
    result = np.asarray(values, dtype=object).copy()
    missing = pd.isna(result)
    result[missing] = np.nan
    if np.any(~missing):
        result[~missing] = [str(value) for value in result[~missing]]
    return result


def _finite_numeric(values):
    result = np.asarray(values, dtype=float).copy()
    result[~np.isfinite(result)] = np.nan
    return result


def _require_finite(values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("В числовых признаках остались пропуски или бесконечности. Включи заполнение пропусков.")
    return values


class PreservingKNNImputer(KNNImputer):
    """Keep names aligned with sklearn's retained all-empty numeric columns.

    sklearn 1.8 KNNImputer retains an empty column with keep_empty_features=True
    but its get_feature_names_out still omits that column. This uses the public
    fitted feature count and MissingIndicator indices to describe actual output.
    """

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "n_features_in_")
        if not self.keep_empty_features:
            return super().get_feature_names_out(input_features)
        if input_features is not None:
            names = np.asarray(input_features, dtype=object)
        else:
            names = getattr(self, "feature_names_in_", np.asarray([f"x{i}" for i in range(self.n_features_in_)], dtype=object))
        if len(names) != self.n_features_in_:
            raise ValueError("Число имён не совпадает с числом входных признаков KNN.")
        if self.add_indicator:
            indicators = [f"missingindicator_{names[index]}" for index in self.indicator_.features_]
            names = np.r_[names, np.asarray(indicators, dtype=object)]
        return names


def _signed_log1p(values):
    values = np.asarray(values, dtype=float)
    return np.sign(values) * np.log1p(np.abs(values))


def _signed_sqrt(values):
    values = np.asarray(values, dtype=float)
    return np.sign(values) * np.sqrt(np.abs(values))


def _integer(value, label, low, high):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label}: укажи целое число от {low} до {high}.")
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label}: укажи целое число от {low} до {high}.") from error
    if integer != value or not low <= integer <= high:
        raise ValueError(f"{label}: укажи целое число от {low} до {high}.")
    return integer


def _boolean(value, label):
    if not isinstance(value, bool):
        raise ValueError(f"{label}: выбери да или нет.")
    return value


def _choice(value, allowed, label):
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"{label}: допустимы {', '.join(allowed)}.")
    return value


class QuantileClipper(TransformerMixin, BaseEstimator):
    """Winsorize each input feature at quantiles learned from this training fold."""

    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        if not (np.isfinite(self.lower) and np.isfinite(self.upper) and 0 <= self.lower < self.upper <= 1):
            raise ValueError("Границы обрезки: 0 ≤ нижний квантиль < верхний квантиль ≤ 1.")
        values = np.asarray(X, dtype=float)
        if values.ndim != 2:
            raise ValueError("Для обрезки нужна таблица числовых признаков.")
        self.n_features_in_ = values.shape[1]
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        bounds = []
        for column in values.T:
            finite = column[np.isfinite(column)]
            bounds.append(np.quantile(finite, [self.lower, self.upper]) if len(finite) else [-np.inf, np.inf])
        self.bounds_ = np.asarray(bounds, dtype=float).T
        return self

    def transform(self, X):
        check_is_fitted(self, "bounds_")
        values = np.asarray(X, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.n_features_in_:
            raise ValueError("Число признаков при обрезке отличается от обучающей таблицы.")
        return np.clip(values, self.bounds_[0], self.bounds_[1])

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "bounds_")
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        return getattr(self, "feature_names_in_", np.asarray([f"x{i}" for i in range(self.n_features_in_)], dtype=object))


def stable_f_regression(X, y):
    """Use sklearn's Pearson scores, bounding floating-point roundoff at |r|=1.

    Without this bound, an exactly predictive feature may get a negative F score
    when the computed correlation exceeds one by a rounding error.
    """
    from scipy.stats import f
    if len(y) < 3:
        raise ValueError('Отбор по F-статистике требует хотя бы три обучающие строки.')
    correlation = r_regression(X, y, force_finite=True)
    squared = np.clip(correlation ** 2, 0.0, 1.0)
    degrees = len(y) - 2
    with np.errstate(divide='ignore', invalid='ignore'):
        scores = squared / (1 - squared) * degrees
    scores = np.nan_to_num(scores, nan=0.0, posinf=np.finfo(float).max)
    return scores, f.sf(scores, 1, degrees)


class FeatureEngineeringTransformer(ColumnTransformer):
    """ColumnTransformer with global, train-only feature selection afterward.

    Keeping selection after concatenation lets it compare numeric, polynomial and
    one-hot features together. This remains a cloneable ColumnTransformer with
    the ordinary ``transform`` and ``get_feature_names_out`` interfaces.
    """

    def __init__(self, transformers, *, variance_threshold=None, selection="none", max_features=None, random_state=42, n_jobs=None):
        self.variance_threshold = variance_threshold
        self.selection = selection
        self.max_features = max_features
        self.random_state = random_state
        super().__init__(transformers, verbose_feature_names_out=False, sparse_threshold=0, n_jobs=n_jobs)

    def fit_transform(self, X, y=None, **params):
        values = super().fit_transform(X, y, **params)
        if values.shape[1] > 2000 or values.shape[0] * values.shape[1] > 15_000_000:
            raise ValueError("После преобразований получилось слишком много данных: предел — 2000 признаков и 15 млн ячеек. Уменьши степень полинома или число категорий.")
        if not np.isfinite(values).all():
            raise ValueError("После преобразований остались пропуски или бесконечности. Включи заполнение пропусков и проверь значения признаков.")
        self.selection_steps_ = []
        if self.variance_threshold is not None:
            selector = VarianceThreshold(threshold=self.variance_threshold)
            try:
                values = selector.fit_transform(values)
            except ValueError as error:
                raise ValueError("Порог дисперсии удалил все признаки. Уменьши порог или отключи удаление постоянных признаков.") from error
            self.selection_steps_.append(selector)
        if self.selection != "none":
            if y is None:
                raise ValueError("Отбор признаков по связи с таргетом требует y только обучающей части: вызови fit_transform(X_train, y_train).")
            k = "all" if self.max_features is None else min(self.max_features, values.shape[1])
            score_func = stable_f_regression if self.selection == "f_regression" else partial(mutual_info_regression, random_state=self.random_state)
            selector = SelectKBest(score_func=score_func, k=k)
            values = selector.fit_transform(values, y)
            self.selection_steps_.append(selector)
        return values

    def transform(self, X, **params):
        check_is_fitted(self, "selection_steps_")
        values = super().transform(X, **params)
        for selector in self.selection_steps_:
            values = selector.transform(values)
        if not np.isfinite(values).all():
            raise ValueError("Новые данные содержат незаполненные пропуски или бесконечности.")
        return values

    def get_feature_names_out(self, input_features=None):
        names = super().get_feature_names_out(input_features)
        for selector in getattr(self, "selection_steps_", []):
            names = selector.get_feature_names_out(names)
        return names


def build_preprocessor(X: pd.DataFrame, config: dict | None):
    """Build an unfitted preprocessor, plus original numeric/category names.

    Legacy ``scale``/``impute`` flags still work. Explicit ``scaler`` and
    ``imputation`` settings take precedence. Missing indicators enter the numeric
    expansion alongside filled values; all output names describe the real columns.
    """
    config = config or {}
    if not isinstance(config, dict):
        raise ValueError("Настройки подготовки данных должны быть объектом.")
    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [name for name in X.columns if name not in numeric]
    if not numeric and not categorical:
        raise ValueError("Не выбраны признаки для обучения.")
    degree = _integer(config.get("degree", 1), "Степень полинома", 1, 5)
    interaction = _boolean(config.get("interaction_only", False), "Только взаимодействия")
    indicator = _boolean(config.get("missing_indicator", False), "Индикаторы пропусков")
    seed = _integer(config.get("seed", 42), "Случайное зерно", 0, 2**32 - 1)
    impute_flag = _boolean(config.get("impute", True), "Заполнение пропусков")
    scale_flag = _boolean(config.get("scale", True), "Масштабирование")
    imputation = _choice(config.get("imputation", "median" if impute_flag else "none"), ("median", "mean", "most_frequent", "constant", "knn", "none"), "Заполнение пропусков")
    scaler = _choice(config.get("scaler", "standard" if scale_flag else "none"), ("standard", "robust", "minmax", "maxabs", "none"), "Масштабирование")
    transform = _choice(config.get("numeric_transform", "none"), ("none", "log1p", "sqrt", "yeo_johnson", "quantile_normal", "quantile_uniform"), "Преобразование числовых признаков")
    if indicator and imputation == "none":
        raise ValueError("Индикаторы пропусков требуют включённого заполнения пропусков.")
    variance = config.get("variance_threshold")
    if variance is not None:
        try:
            variance = float(variance)
        except (TypeError, ValueError) as error:
            raise ValueError("Порог дисперсии должен быть неотрицательным числом.") from error
        if not np.isfinite(variance) or variance < 0:
            raise ValueError("Порог дисперсии должен быть неотрицательным числом.")
    selection = _choice(config.get("selection", "none"), ("none", "f_regression", "mutual_info"), "Отбор признаков")
    maximum = config.get("max_features")
    if maximum is not None:
        maximum = _integer(maximum, "Число отобранных признаков", 1, 2000)
        if selection == "none":
            raise ValueError("Чтобы ограничить число признаков, выбери метод отбора по таргету.")
    clip = config.get("clip_quantiles")
    if clip is not None:
        try:
            if len(clip) != 2:
                raise ValueError
            clip = [float(value) for value in clip]
        except (TypeError, ValueError) as error:
            raise ValueError("Границы обрезки задай двумя квантилями, например [0.01, 0.99].") from error
        if not (all(np.isfinite(clip)) and 0 <= clip[0] < clip[1] <= 1):
            raise ValueError("Границы обрезки: 0 ≤ нижний квантиль < верхний квантиль ≤ 1.")
    # Count potential missing indicators without inspecting their values. The
    # bound is deliberately conservative, before polynomial allocation.
    count = len(numeric) * (2 if indicator else 1)
    generated = sum(math.comb(count, power) for power in range(1, min(degree, count) + 1)) if interaction else math.comb(count + degree, degree) - 1
    if numeric and (generated > 2000 or generated * len(X) > 15_000_000):
        raise ValueError("Полиномиальное расширение превысит предел 2000 признаков или 15 млн ячеек. Уменьши степень, число признаков или индикаторы пропусков.")
    transformers = []
    if numeric:
        steps = [("finite", FunctionTransformer(_finite_numeric, feature_names_out="one-to-one"))]
        if clip is not None:
            steps.append(("clip", QuantileClipper(*clip)))
        if imputation == "knn":
            neighbours = _integer(config.get("knn_neighbors", 5), "Соседи для заполнения пропусков", 1, 100)
            if len(X) > 20_000:
                raise ValueError("KNN-заполнение требует не более 20 000 строк. Для большого набора выбери медиану или среднее.")
            imputer = PreservingKNNImputer(n_neighbors=neighbours, keep_empty_features=True, add_indicator=indicator)
        elif imputation != "none":
            fill = config.get("fill_value", 0)
            try:
                fill = float(fill)
            except (TypeError, ValueError) as error:
                raise ValueError("Константа для числовых пропусков должна быть конечным числом.") from error
            if not np.isfinite(fill):
                raise ValueError("Константа для числовых пропусков должна быть конечным числом.")
            imputer = SimpleImputer(strategy=imputation, fill_value=fill, keep_empty_features=True, add_indicator=indicator)
        else:
            imputer = None
        if imputer is not None:
            steps.append(("impute", imputer))
        steps.append(("validate", FunctionTransformer(_require_finite, feature_names_out="one-to-one")))
        if transform in {"log1p", "sqrt"}:
            function = _signed_log1p if transform == "log1p" else _signed_sqrt
            steps.append(("transform", FunctionTransformer(function, feature_names_out="one-to-one")))
        elif transform == "yeo_johnson":
            steps.append(("transform", PowerTransformer(method="yeo-johnson", standardize=False)))
        elif transform.startswith("quantile_"):
            steps.append(("transform", QuantileTransformer(n_quantiles=min(len(X), 1000), output_distribution=transform.removeprefix("quantile_"), random_state=seed)))
        steps.append(("polynomial", PolynomialFeatures(degree=degree, include_bias=False, interaction_only=interaction)))
        scalers = {"standard": StandardScaler, "robust": RobustScaler, "minmax": MinMaxScaler, "maxabs": MaxAbsScaler}
        if scaler != "none":
            steps.append(("scale", scalers[scaler]()))
        transformers.append(("numeric", Pipeline(steps), numeric))
    if categorical:
        steps = [("stringify", FunctionTransformer(_string_categories, feature_names_out="one-to-one"))]
        if imputation != "none":
            strategy = "constant" if imputation == "constant" else "most_frequent"
            steps.append(("impute", SimpleImputer(strategy=strategy, fill_value="__missing__", keep_empty_features=True)))
        steps.append(("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)))
        transformers.append(("categorical", Pipeline(steps), categorical))
    return FeatureEngineeringTransformer(transformers, variance_threshold=variance, selection=selection, max_features=maximum, random_state=seed), numeric, categorical


# Third-party research samplers use NumPy's and Python's global RNG. Each job is
# a separate process; this lock also makes threaded calls within one job safe.
_sampling_lock = threading.RLock()


@contextmanager
def _sampling_seed(seed):
    with _sampling_lock:
        numpy_state, python_state = np.random.get_state(), random.getstate()
        np.random.seed(seed)
        random.seed(seed)
        try:
            yield
        finally:
            np.random.set_state(numpy_state)
            random.setstate(python_state)


class ResamplingService:
    """Call actual regression samplers on transformed training rows only.

    SMOTER and SMOGN are restricted to wholly numeric original features. Linear
    interpolation of one-hot category columns would invent fractional categories.
    Random sampling preserves complete rows and is also allowed with categories.
    """

    def fit_resample(self, X, y, config=None, *, seed=42, categorical=False):
        config = config or {}
        if not isinstance(config, dict):
            raise ValueError("Настройки пересэмплирования должны быть объектом.")
        method = _choice(config.get("method", "none"), ("none", "random_over", "random_under", "smoter", "smogn"), "Пересэмплирование регрессии")
        values, targets = np.asarray(X, dtype=float), np.asarray(y, dtype=float).reshape(-1)
        if values.ndim != 2 or len(values) != len(targets) or len(targets) < 2:
            raise ValueError("Для пересэмплирования нужны согласованные обучающие X и y минимум из двух строк.")
        if not np.isfinite(values).all() or not np.isfinite(targets).all():
            raise ValueError("Перед пересэмплированием заполни пропуски: X и y должны быть конечными числами.")
        summary = {"method": method, "before": len(targets), "after": len(targets), "fitted_on": "train", "synthetic": method in {"smoter", "smogn"}}
        if method == "none":
            return values, targets, summary
        if np.unique(targets).size < 3:
            raise ValueError("Пересэмплирование регрессии требует хотя бы три разных значения таргета. Для категорий нужна задача классификации.")
        if categorical and method in {"smoter", "smogn"}:
            raise ValueError("SMOTER/SMOGN здесь работают только с исходными числовыми признаками. Интерполяция one-hot категорий создаёт неверные дробные категории; выбери случайное увеличение или уменьшение.")
        if len(targets) > 10_000 or values.shape[1] > 200 or values.size > 1_000_000:
            raise ValueError("Для пересэмплирования выбери не более 10 000 обучающих строк, 200 преобразованных признаков и 1 млн ячеек.")
        seed = _integer(seed, "Случайное зерно", 0, 2**32 - 1)
        try:
            threshold = float(config.get("relevance_threshold", 0.5))
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("Порог важности редких значений таргета должен быть числом между 0 и 1.") from error
        if not np.isfinite(threshold) or not 0 < threshold < 1:
            raise ValueError("Порог важности редких значений таргета должен быть между 0 и 1.")
        focus = _choice(config.get("focus", "both"), ("both", "high", "low"), "Редкие значения таргета")
        sampling = _choice(config.get("sampling", "balance"), ("balance", "extreme"), "Интенсивность пересэмплирования")
        relevance = _choice(config.get("relevance", "auto"), ("auto", "manual"), "Способ задания важности таргета")
        control = config.get("control_points")
        if relevance == "manual":
            try:
                points = np.asarray(control, dtype=float)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError("Ручная важность: задай числовые точки [таргет, важность 0…1, производная].") from error
            if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 3 or not np.isfinite(points).all() or not np.all(np.diff(points[:, 0]) > 0) or np.any((points[:, 1] < 0) | (points[:, 1] > 1)):
                raise ValueError("Ручная важность: минимум три точки [таргет, важность 0…1, производная] с возрастающим таргетом.")
            control = points.tolist()
        table = pd.DataFrame(values.copy(), columns=[f"x{i}" for i in range(values.shape[1])])
        table["__target__"] = targets
        common = {"data": table, "y": "__target__", "samp_method": sampling, "drop_na_col": False, "drop_na_row": False, "rel_thres": threshold, "rel_method": relevance, "rel_xtrm_type": focus, "rel_ctrl_pts_rg": control}
        try:
            package = importlib.import_module("smogn" if method == "smogn" else "ImbalancedLearningRegression")
        except ImportError as error:
            raise ValueError("Не установлен пакет пересэмплирования. Перезапусти через run.py для установки зависимостей.") from error
        with _sampling_seed(seed):
            try:
                if method in {"smoter", "smogn"}:
                    k = _integer(config.get("neighbors", 5), "Соседи для SMOTER/SMOGN", 1, min(100, len(targets) - 1))
                    common["k"] = k
                    if method == "smogn":
                        perturbation = float(config.get("perturbation", 0.02))
                        if not np.isfinite(perturbation) or not 0 < perturbation <= 1:
                            raise ValueError("Доля шума SMOGN должна быть больше 0 и не больше 1.")
                        from ._vendor.smogn_smoter import smoter
                        output = smoter(**common, pert=perturbation, under_samp=config.get("undersample", True))
                    else:
                        from ._vendor.iblr_smote import smote
                        output = smote(**common)
                elif method == "random_over":
                    output = package.ro(**common)
                else:
                    output = package.random_under(**common)
            except (ValueError, IndexError, ZeroDivisionError, TypeError, KeyError) as error:
                raise ValueError(f"{method.upper()} не смог пересэмплировать обучающую часть: {error}. Проверь наличие редких значений таргета, их важность и число соседей.") from error
        columns = [f"x{i}" for i in range(values.shape[1])]
        result_X = output[columns].to_numpy(dtype=float)
        result_y = output["__target__"].to_numpy(dtype=float)
        if len(result_y) < 2 or len(result_y) > 100_000 or result_X.size > 15_000_000 or not np.isfinite(result_X).all() or not np.isfinite(result_y).all():
            raise ValueError("Пакет пересэмплирования вернул пустой, слишком большой или некорректный результат. Уменьши интенсивность либо выбери другой метод.")
        summary.update(after=len(result_y), relevance_threshold=threshold, focus=focus, sampling=sampling, backend="smogn" if method == "smogn" else "ImbalancedLearningRegression", corrected_backend=method in {"smoter", "smogn"})
        return result_X, result_y, summary


class RegressionSampler(BaseEstimator):
    """Sampler step compatible with ``imblearn.pipeline.Pipeline``.

    It has deliberately no transform: held-out rows and inference bypass sampling.
    Each pipeline clone therefore samples only the rows received by its fit call.
    """

    def __init__(self, config=None, random_state=42, categorical=False):
        self.config = config
        self.random_state = random_state
        self.categorical = categorical

    def fit_resample(self, X, y):
        X, y, self.summary_ = ResamplingService().fit_resample(X, y, self.config, seed=self.random_state, categorical=self.categorical)
        return X, y
