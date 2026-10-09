"""Compile stored preparation recipes into unfitted, cloneable estimators.

Only this adapter imports numerical libraries. Constructors never compute fitted
statistics; the experiment engine fits a new pipeline within each training fold.
"""

from __future__ import annotations

import math
from collections import defaultdict
from copy import deepcopy

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import FastICA, NMF, PCA, TruncatedSVD
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer
from sklearn.feature_selection import SelectFromModel, SelectKBest, VarianceThreshold, chi2, f_classif, mutual_info_classif, mutual_info_regression
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, MaxAbsScaler, MinMaxScaler, Normalizer, OneHotEncoder, OrdinalEncoder, PolynomialFeatures, PowerTransformer, QuantileTransformer, RobustScaler, SplineTransformer, StandardScaler, TargetEncoder
from sklearn.utils.validation import check_is_fitted

from linear_lab.preprocessing import PreservingKNNImputer, QuantileClipper, ResamplingService, build_preprocessor as build_legacy_preprocessor, stable_f_regression
from .preparation_schema import ALL_TASKS, PreparationCatalogue


MAX_DENSE_CELLS = 15_000_000
MAX_DENSE_FEATURES = 2000
MAX_SPARSE_FEATURES = 20_000
MAX_SPARSE_NNZ = 15_000_000


def _integer(value, label, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError(f"{label}: нужно целое число от {low} до {high}.")
    return int(value)


def _number(value, label, low=None, high=None):
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)) or not np.isfinite(value):
        raise ValueError(f"{label}: нужно конечное число.")
    if low is not None and value < low or high is not None and value > high:
        raise ValueError(f"{label}: число вне допустимого диапазона.")
    return float(value)


def _bool(value, label):
    if not isinstance(value, bool):
        raise ValueError(f"{label}: выберите да или нет.")
    return value


def _choice(value, choices, label):
    if value not in choices:
        raise ValueError(f"{label}: выберите {', '.join(choices)}.")
    return value


def _check_size(values):
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("Подготовка не оставила признаков для модели.")
    if sparse.issparse(values):
        if values.shape[1] > MAX_SPARSE_FEATURES or values.nnz > MAX_SPARSE_NNZ:
            raise ValueError("Разреженная подготовка превышает 20 000 признаков или 15 млн ненулевых значений.")
    elif values.shape[1] > MAX_DENSE_FEATURES or values.size > MAX_DENSE_CELLS:
        raise ValueError("Плотная подготовка превышает 2000 признаков или 15 млн ячеек.")


class FiniteNumeric(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def transform(self, X):
        values = np.asarray(X, dtype=float).copy()
        values[~np.isfinite(values)] = np.nan
        return values

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features if input_features is not None else [f"x{i}" for i in range(self.n_features_in_)], dtype=object)


class StringCategories(FiniteNumeric):
    def transform(self, X):
        values = np.asarray(X, dtype=object).copy()
        missing = pd.isna(values)
        values[missing] = np.nan
        values[~missing] = [str(value) for value in values[~missing]]
        return values


class SignedPower(FiniteNumeric):
    def __init__(self, method="log1p"):
        self.method = method

    def transform(self, X):
        values = np.asarray(X, dtype=float)
        function = np.log1p if self.method == "log1p" else np.sqrt
        return np.sign(values) * function(np.abs(values))


class BoundedPolynomial(PolynomialFeatures):
    def fit(self, X, y=None):
        count = np.asarray(X).shape[1]
        generated = sum(math.comb(count, power) for power in range(1, min(self.degree, count) + 1)) if self.interaction_only else math.comb(count + self.degree, self.degree) - 1
        if generated > MAX_DENSE_FEATURES or generated * len(X) > MAX_DENSE_CELLS:
            raise ValueError("Полином превысит 2000 признаков или 15 млн ячеек. Уменьшите степень или число выбранных признаков.")
        return super().fit(X, y)

    def transform(self, X):
        check_is_fitted(self, "n_output_features_")
        if len(X) * self.n_output_features_ > MAX_DENSE_CELLS:
            raise ValueError("Полином прогноза превысит 15 млн ячеек; уменьшите размер запроса.")
        return super().transform(X)


class BoundedOneHotEncoder(OneHotEncoder):
    def fit(self, X, y=None):
        result = super().fit(X, y)
        count = len(self.get_feature_names_out())
        maximum = MAX_SPARSE_FEATURES if self.sparse_output else MAX_DENSE_FEATURES
        if count > maximum:
            raise ValueError("One-hot создаст слишком много столбцов. Объедините редкие категории или уменьшите max_categories.")
        if not self.sparse_output and count * len(X) > MAX_DENSE_CELLS:
            raise ValueError("One-hot превысит 15 млн плотных ячеек. Выберите разреженную матрицу или объедините категории.")
        return result

    def transform(self, X):
        if not self.sparse_output and len(self.get_feature_names_out()) * len(X) > MAX_DENSE_CELLS:
            raise ValueError("One-hot прогноз превышает 15 млн плотных ячеек; уменьшите размер запроса.")
        return super().transform(X)


class BoundedBins(KBinsDiscretizer):
    def fit(self, X, y=None, sample_weight=None):
        result = super().fit(X, y, sample_weight=sample_weight)
        self._check_output(len(X))
        return result

    def _check_output(self, rows):
        count = int(np.sum(self.n_bins_)) if self.encode == "onehot-dense" else self.n_features_in_
        if count > MAX_DENSE_FEATURES or rows * count > MAX_DENSE_CELLS:
            raise ValueError("Интервалы превысят 2000 признаков или 15 млн ячеек. Выберите ordinal или меньше интервалов.")

    def transform(self, X):
        check_is_fitted(self, "n_bins_")
        self._check_output(len(X))
        return super().transform(X)


class BoundedSpline(SplineTransformer):
    def fit(self, X, y=None, sample_weight=None):
        bases = self.n_knots - 1 if self.extrapolation == "periodic" else self.n_knots + self.degree - 1
        count = np.asarray(X).shape[1] * (bases - int(not self.include_bias))
        if count > MAX_DENSE_FEATURES or len(X) * count > MAX_DENSE_CELLS:
            raise ValueError("Сплайны превысят 2000 признаков или 15 млн ячеек. Уменьшите число узлов/признаков.")
        return super().fit(X, y, sample_weight=sample_weight)

    def transform(self, X):
        check_is_fitted(self, "n_features_out_")
        if len(X) * self.n_features_out_ > MAX_DENSE_CELLS:
            raise ValueError("Сплайны прогноза превысят 15 млн ячеек; уменьшите размер запроса.")
        return super().transform(X)


class FrequencyEncoder(FiniteNumeric):
    def __init__(self, normalize=True):
        self.normalize = normalize

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=object)
        self.n_features_in_ = values.shape[1]
        self.counts_ = [pd.Series(column).value_counts(normalize=self.normalize).to_dict() for column in values.T]
        return self

    def transform(self, X):
        check_is_fitted(self, "counts_")
        values = np.asarray(X, dtype=object)
        return np.column_stack([[counts.get(value, 0.0) for value in column] for counts, column in zip(self.counts_, values.T)])


class CategoryHasher(TransformerMixin, BaseEstimator):
    def __init__(self, n_features=128, alternate_sign=False):
        self.n_features = n_features
        self.alternate_sign = alternate_sign

    def fit(self, X, y=None):
        self.n_features_in_ = np.asarray(X).shape[1]
        self.hasher_ = FeatureHasher(n_features=self.n_features, alternate_sign=self.alternate_sign)
        return self

    def transform(self, X):
        check_is_fitted(self, "hasher_")
        records = [{f"c{index}={value}": 1.0 for index, value in enumerate(row)} for row in np.asarray(X, dtype=object)]
        return self.hasher_.transform(records)

    def get_feature_names_out(self, input_features=None):
        return np.asarray([f"hash_{i}" for i in range(self.n_features)], dtype=object)


class TextJoiner(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        values = np.asarray(X, dtype=object)
        return np.asarray([" ".join("" if pd.isna(value) else str(value) for value in row) for row in values], dtype=object)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(["joined_text"], dtype=object)


class NamedHashingVectorizer(HashingVectorizer):
    def get_feature_names_out(self, input_features=None):
        return np.asarray([f"hash_{i}" for i in range(self.n_features)], dtype=object)


class DatetimeExtractor(TransformerMixin, BaseEstimator):
    def __init__(self, parts=("year", "month", "day", "weekday"), cyclic=False, timezone="UTC"):
        self.parts = parts
        self.cyclic = cyclic
        self.timezone = timezone

    def fit(self, X, y=None):
        self.n_features_in_ = np.asarray(X).shape[1]
        # Fail an unknown timezone during fit instead of after a successful job.
        pd.Timestamp("2000-01-01", tz="UTC").tz_convert(self.timezone)
        return self

    def transform(self, X):
        check_is_fitted(self, "n_features_in_")
        width = len(self.get_feature_names_out())
        if width > MAX_DENSE_FEATURES or len(X) * width > MAX_DENSE_CELLS:
            raise ValueError("Календарные признаки превысят 2000 признаков или 15 млн ячеек. Выберите меньше частей даты/столбцов.")
        result = []
        for values in np.asarray(X, dtype=object).T:
            date = pd.Series(pd.to_datetime(values, errors="coerce", utc=True, format="mixed")).dt.tz_convert(self.timezone)
            for part in self.parts:
                values = (date.dt.dayofweek >= 5).astype(float).mask(date.isna()) if part == "weekend" else getattr(date.dt, "dayofweek" if part == "weekday" else part).astype(float)
                result.append(values.to_numpy())
                if self.cyclic and part in {"month", "weekday", "hour", "minute", "dayofyear"}:
                    period = {"month": 12, "weekday": 7, "hour": 24, "minute": 60, "dayofyear": 366}[part]
                    offset = 1 if part in {"month", "dayofyear"} else 0
                    phase = 2 * np.pi * (values.to_numpy() - offset) / period
                    result.extend([np.sin(phase), np.cos(phase)])
        return np.column_stack(result)

    def get_feature_names_out(self, input_features=None):
        names = input_features if input_features is not None else [f"date{i}" for i in range(self.n_features_in_)]
        result = []
        for name in names:
            for part in self.parts:
                result.append(f"{name}_{part}")
                if self.cyclic and part in {"month", "weekday", "hour", "minute", "dayofyear"}:
                    result.extend([f"{name}_{part}_sin", f"{name}_{part}_cos"])
        return np.asarray(result, dtype=object)


class BoundedReducer(TransformerMixin, BaseEstimator):
    def __init__(self, method="pca", n_components=2, random_state=42):
        self.method = method
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None):
        if self.n_components > min(X.shape):
            raise ValueError("Число компонент не может превышать число строк и входных признаков train.")
        if self.method != "svd" and sparse.issparse(X):
            raise ValueError("Этот метод требует плотную матрицу. Выберите SVD для текста или разреженного one-hot.")
        if X.shape[0] * X.shape[1] > MAX_DENSE_CELLS and self.method != "svd":
            raise ValueError("Матрица разложения превышает 15 млн ячеек.")
        factories = {"pca": PCA, "svd": TruncatedSVD, "ica": FastICA, "nmf": NMF}
        params = {"n_components": self.n_components, "random_state": self.random_state}
        if self.method in {"ica", "nmf"}:
            params["max_iter"] = 500
        if self.method == "nmf" and np.min(X) < 0:
            raise ValueError("NMF требует неотрицательные признаки. Выберите иной масштаб или SVD.")
        self.reducer_ = factories[self.method](**params).fit(X, y)
        return self

    def transform(self, X):
        check_is_fitted(self, "reducer_")
        if X.shape[0] * self.n_components > MAX_DENSE_CELLS:
            raise ValueError("Компоненты прогноза превысят 15 млн ячеек; уменьшите размер запроса.")
        return self.reducer_.transform(X)

    def get_feature_names_out(self, input_features=None):
        return np.asarray([f"{self.method}_{index + 1}" for index in range(self.n_components)], dtype=object)


class CheckedSelector(TransformerMixin, BaseEstimator):
    def __init__(self, method="auto", k=10, task="regression", random_state=42, discrete_features="auto", has_continuous_inputs=False):
        self.method = method
        self.k = k
        self.task = task
        self.random_state = random_state
        self.discrete_features = discrete_features
        self.has_continuous_inputs = has_continuous_inputs

    def fit(self, X, y):
        if y is None:
            raise ValueError("Отбор по цели требует обучающую цель.")
        if self.k > X.shape[1]:
            raise ValueError(f"Запрошено {self.k} признаков, после подготовки есть {X.shape[1]}.")
        classification = self.task == "classification"
        if self.method == "chi2":
            if not classification:
                raise ValueError("chi2 применяется к классификации.")
            values = X.data if sparse.issparse(X) else X
            if np.min(values, initial=0) < 0:
                raise ValueError("chi2 требует неотрицательные признаки, например количества слов.")
            score = chi2
        elif self.method == "mutual_info":
            from functools import partial
            if sparse.issparse(X) and (self.discrete_features == "continuous" or self.discrete_features == "auto" and self.has_continuous_inputs):
                raise ValueError("MI для непрерывных признаков требует плотную матрицу. Отключите sparse_output категорий или добавьте SVD перед отбором. Для дискретных счетчиков текста выберите discrete_features=discrete.")
            discrete = {"auto": "auto", "continuous": False, "discrete": True}[self.discrete_features]
            score = partial(mutual_info_classif if classification else mutual_info_regression, random_state=self.random_state, discrete_features=discrete)
        else:
            score = f_classif if classification else stable_f_regression
        self.selector_ = SelectKBest(score, k=self.k).fit(X, y)
        return self

    def transform(self, X):
        check_is_fitted(self, "selector_")
        return self.selector_.transform(X)

    def get_feature_names_out(self, input_features=None):
        return self.selector_.get_feature_names_out(input_features)


class PreparedFeatures(Pipeline):
    """Ordinary sklearn Pipeline with explicit allocation bounds and metadata."""

    def fit_transform(self, X, y=None, **params):
        result = super().fit_transform(X, y, **params)
        _check_size(result)
        return result

    def transform(self, X, **params):
        result = super().transform(X, **params)
        _check_size(result)
        return result


def _numeric_factory(adapter, params, task, seed):
    if adapter == "numeric.impute":
        strategy = params.get("strategy", "median")
        indicator = params.get("add_indicator", False)
        if strategy == "knn":
            return PreservingKNNImputer(n_neighbors=params.get("n_neighbors", 5), add_indicator=indicator, keep_empty_features=True)
        if strategy == "iterative":
            return IterativeImputer(max_iter=params.get("max_iter", 10), n_nearest_features=params.get("n_nearest_features", 10), initial_strategy="median", random_state=seed, add_indicator=indicator, keep_empty_features=True, skip_complete=True)
        return SimpleImputer(strategy=strategy, fill_value=params.get("fill_value", 0), add_indicator=indicator, keep_empty_features=True)
    if adapter == "numeric.scale":
        method = params.get("method", "standard")
        if method == "none":
            return "passthrough"
        if method == "minmax":
            return MinMaxScaler(clip=params.get("clip", False))
        return {"standard": StandardScaler, "robust": RobustScaler, "maxabs": MaxAbsScaler, "normalize": Normalizer}[method]()
    if adapter == "numeric.power":
        method = params.get("method", "yeo_johnson")
        if method in {"log1p", "sqrt"}:
            return SignedPower(method)
        if method in {"yeo_johnson", "box_cox"}:
            return PowerTransformer(method=method.replace("_", "-"), standardize=False)
        return QuantileTransformer(n_quantiles=params.get("n_quantiles", 100), output_distribution=method.removeprefix("quantile_"), random_state=seed)
    if adapter == "numeric.clip":
        lower, upper = params.get("lower", 0.01), params.get("upper", 0.99)
        if lower >= upper:
            raise ValueError("Нижний квантиль обрезки должен быть меньше верхнего.")
        return QuantileClipper(lower, upper)
    if adapter == "numeric.polynomial":
        return BoundedPolynomial(degree=params.get("degree", 2), interaction_only=params.get("interaction_only", False), include_bias=False)
    if adapter == "numeric.bin":
        return BoundedBins(n_bins=params.get("n_bins", 5), strategy=params.get("strategy", "quantile"), encode=params.get("encode", "onehot-dense"), quantile_method="linear", random_state=seed)
    if adapter == "numeric.spline":
        return BoundedSpline(n_knots=params.get("n_knots", 5), degree=params.get("degree", 3), knots=params.get("knots", "quantile"), extrapolation=params.get("extrapolation", "constant"), include_bias=False)
    raise ValueError(f"Неизвестный числовой этап: {adapter}.")


def _categorical_factory(adapter, params, seed, task):
    if adapter == "categorical.onehot":
        return BoundedOneHotEncoder(handle_unknown=params.get("handle_unknown", "ignore"), min_frequency=params.get("min_frequency", 1), max_categories=params.get("max_categories", 100), sparse_output=params.get("sparse_output", False))
    if adapter == "categorical.ordinal":
        categories = params.get("categories")
        if categories is not None:
            categories = [[str(value) for value in column] for column in categories]
            if any(len(column) != len(set(column)) for column in categories):
                raise ValueError("Порядок категорий содержит повторяющиеся значения после строкового представления.")
        return OrdinalEncoder(categories="auto" if categories is None else categories, handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-2)
    if adapter == "categorical.frequency":
        return FrequencyEncoder(params.get("normalize", True))
    if adapter == "categorical.hash":
        return CategoryHasher(params.get("n_features", 128), params.get("alternate_sign", False))
    if adapter == "categorical.target":
        return TargetEncoder(target_type="continuous" if task == "regression" else "auto", smooth=params.get("smooth", 10.0), cv=params.get("cv", 3), random_state=seed)
    raise ValueError(f"Неизвестный категориальный этап: {adapter}.")


def _validate_params(stage, params):
    allowed = {field["key"] for field in stage["params"]}
    if stage["id"] == "numeric":
        # The compatible public adapter validates the complete legacy schema.
        allowed |= {"scale", "impute", "fill_value", "knn_neighbors", "seed", "clip_quantiles", "variance_threshold", "selection", "max_features"}
    unknown = set(params) - allowed
    if unknown:
        raise ValueError(f"{stage['name']}: неизвестные настройки {', '.join(sorted(unknown))}.")
    for field in stage["params"]:
        if field["key"] not in params:
            continue
        value = params[field["key"]]
        label = field["label"]
        if field["type"] == "int":
            _integer(value, label, field.get("min", -(2**31)), field.get("max", 2**31 - 1))
        elif field["type"] == "float":
            _number(value, label, field.get("min"), field.get("max"))
        elif field["type"] == "bool":
            _bool(value, label)
        elif field["type"] == "select":
            _choice(value, field["options"], label)
        elif field["type"] == "multiselect":
            if not isinstance(value, list) or not value or any(item not in field["options"] for item in value) or len(set(value)) != len(value):
                raise ValueError(f"{label}: выберите неповторяющиеся значения из списка.")
        elif field["type"] == "text" and (not isinstance(value, str) or len(value) > 100):
            raise ValueError(f"{label}: нужна строка не более 100 символов.")
        elif field["type"] == "json" and value is not None:
            if not isinstance(value, list) or len(value) > MAX_DENSE_FEATURES or any(not isinstance(items, list) or not items or len(items) > MAX_DENSE_FEATURES for items in value):
                raise ValueError(f"{label}: нужен список непустых списков категорий.")
            if any(not isinstance(category, (str, int, float, bool)) or isinstance(category, float) and not math.isfinite(category) for items in value for category in items):
                raise ValueError(f"{label}: категории должны быть строками или конечными числами.")


def _column_type(series):
    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series.dtype):
        return "numeric"
    return "categorical"


def validate_preparation_config(config, task=None):
    """Validate portable stored settings without fitting or reading a table."""
    if not isinstance(config, dict):
        raise ValueError("Рецепт подготовки должен быть объектом.")
    result = deepcopy(config)
    if task is not None and task not in ALL_TASKS:
        raise ValueError("Неизвестная задача подготовки данных.")
    catalogue = PreparationCatalogue()
    if "steps" not in result:
        legacy = {key: value for key, value in result.items() if key not in {"resampling", "split_kind"}}
        _validate_params(catalogue.get("numeric"), legacy)
        if task not in {None, "regression"} and legacy.get("selection", "none") != "none":
            raise ValueError("Старый отбор только для регрессии. Используйте selection.univariate.")
    else:
        if set(result) - {"steps", "resampling", "split_kind"}:
            raise ValueError("Настройки новой подготовки: steps и resampling.")
        if not isinstance(result["steps"], list) or len(result["steps"]) > 32:
            raise ValueError("Рецепт содержит список не более 32 этапов.")
        for spec in result["steps"]:
            if not isinstance(spec, dict) or set(spec) - {"adapter_id", "columns", "params", "enabled", "id"}:
                raise ValueError("Этап содержит adapter_id, columns, params и enabled.")
            descriptor = catalogue.get(spec.get("adapter_id"))
            enabled = _bool(spec.get("enabled", True), "Включение этапа")
            columns = spec.get("columns", [])
            if not isinstance(columns, list) or any(not isinstance(value, str) for value in columns) or len(columns) != len(set(columns)):
                raise ValueError("Столбцы этапа: список неповторяющихся строковых имён.")
            params = spec.get("params", {})
            if not isinstance(params, dict):
                raise ValueError("Параметры этапа должны быть объектом.")
            _validate_params(descriptor, params)
            if enabled and task is not None and task not in descriptor["allowed_tasks"]:
                raise ValueError(f"{descriptor['name']} несовместим с задачей {task}.")
    sampling = result.get("resampling", {"method": "none"})
    if not isinstance(sampling, dict):
        raise ValueError("Пересэмплирование должно быть объектом.")
    method = sampling.get("method", "none")
    descriptors = {sampler["id"]: sampler for sampler in catalogue.samplers()}
    if method not in descriptors:
        raise ValueError("Неизвестный пересэмплировщик.")
    if task is not None and task not in descriptors[method]["allowed_tasks"]:
        if not (task == "regression" and method in {"random_over", "random_under"}):
            raise ValueError("Пересэмплировщик несовместим с задачей.")
    if method != "none" and not descriptors[method]["available"]:
        raise ValueError(descriptors[method]["reason"])
    allowed = {"method", "neighbors", "sampling_strategy", "replacement"} if task == "classification" else {"method", "focus", "sampling", "relevance_threshold", "neighbors", "perturbation", "undersample", "relevance", "control_points"}
    if task is None:
        allowed |= {"sampling_strategy", "replacement"}
    if set(sampling) - allowed:
        raise ValueError("Неизвестные параметры пересэмплировщика.")
    _validate_params({"name": "Пересэмплирование", "id": "sampler", "params": descriptors[method]["params"]}, {key: value for key, value in sampling.items() if key in {field["key"] for field in descriptors[method]["params"]}})
    return result


def build_preprocessor(X, config=None, task="regression", seed=42):
    """Build an unfitted recipe. Empty columns selects the adapter's input type.

    Column stages sharing a selector execute in their stored order. Different
    selectors must be disjoint; global selection/reduction runs after concatenation.
    Unknown settings and incompatible overlaps are rejected, never ignored.
    """
    if not isinstance(X, pd.DataFrame) or X.columns.has_duplicates or any(not isinstance(name, str) for name in X.columns):
        raise ValueError("Подготовке нужна таблица с уникальными строковыми именами столбцов.")
    if task not in ALL_TASKS:
        raise ValueError("Неизвестная задача подготовки данных.")
    seed = _integer(seed, "Случайное зерно", 0, 2**32 - 1)
    config = deepcopy(config or {})
    if not isinstance(config, dict):
        raise ValueError("Рецепт подготовки должен быть объектом.")
    if "steps" not in config:
        legacy = dict(config)
        legacy.pop("resampling", None)
        if task != "regression" and legacy.get("selection", "none") != "none":
            raise ValueError("Старый отбор признаков предназначен для регрессии. Используйте selection.univariate новой схемы.")
        return build_legacy_preprocessor(X, legacy)[0]
    if set(config) - {"steps", "resampling", "split_kind"}:
        raise ValueError("Настройки новой подготовки: steps и resampling. Старые параметры поместите внутрь этапа numeric.")
    specs = config["steps"]
    if not isinstance(specs, list) or len(specs) > 32:
        raise ValueError("Рецепт содержит список не более 32 этапов.")
    catalogue = PreparationCatalogue()
    parsed = []
    dropped = set()
    for spec in specs:
        if not isinstance(spec, dict) or set(spec) - {"adapter_id", "columns", "params", "enabled", "id"}:
            raise ValueError("Этап содержит adapter_id, columns, params и enabled.")
        if not _bool(spec.get("enabled", True), "Включение этапа"):
            continue
        descriptor = catalogue.get(spec.get("adapter_id"))
        if task not in descriptor["allowed_tasks"]:
            raise ValueError(f"{descriptor['name']} несовместим с задачей {task}.")
        columns = spec.get("columns", [])
        params = spec.get("params", {})
        if not isinstance(columns, list) or any(not isinstance(name, str) for name in columns) or len(columns) != len(set(columns)) or any(name not in X.columns for name in columns):
            raise ValueError("Выберите существующие неповторяющиеся столбцы этапа.")
        if not isinstance(params, dict):
            raise ValueError("Параметры этапа должны быть объектом.")
        _validate_params(descriptor, params)
        if descriptor["id"] == "columns.drop":
            if not columns:
                raise ValueError("Для удаления явно выберите столбцы.")
            dropped.update(columns)
        else:
            parsed.append((descriptor, columns, params))
    types = {name: _column_type(X[name]) for name in X.columns if name not in dropped}
    if not types:
        raise ValueError("После удаления не осталось признаков.")
    groups = defaultdict(list)
    owners = {}
    global_steps = []
    for descriptor, columns, params in parsed:
        adapter = descriptor["id"]
        if descriptor["placement"] == "global":
            if columns:
                raise ValueError("Глобальный отбор/разложение работает со всеми подготовленными признаками; selector columns должен быть пуст.")
            global_steps.append((adapter, params))
            continue
        if any(name in dropped for name in columns):
            raise ValueError("Удалённый столбец нельзя использовать в последующем этапе.")
        if not columns:
            preferred = "categorical" if adapter.startswith("categorical.") else "datetime" if adapter.startswith("datetime.") else "numeric"
            if adapter.startswith("text."):
                raise ValueError("Для текстового этапа явно выберите текстовые столбцы.")
            columns = [name for name, kind in types.items() if kind == preferred]
        if not columns:
            raise ValueError(f"{descriptor['name']}: нет подходящих столбцов.")
        if any(types[name] not in descriptor["accepted_column_types"] for name in columns):
            raise ValueError(f"{descriptor['name']}: выбранный столбец имеет неподходящий тип.")
        key = tuple(columns)
        family = "numeric" if adapter.startswith("numeric") else "categorical" if adapter.startswith("categorical.") else "text" if adapter.startswith("text.") else "datetime"
        for name in columns:
            previous = owners.get(name)
            if previous is not None and previous != (key, family):
                raise ValueError("Последовательные этапы одного столбца должны использовать одинаковый selector; разные ветки не пересекаются.")
            owners[name] = (key, family)
        groups[(key, family)].append((adapter, params))
    # Unselected columns remain model inputs with the same safe automatic recipe.
    for kind in ("numeric", "categorical", "datetime"):
        columns = tuple(name for name, detected in types.items() if detected == kind and name not in owners)
        if columns:
            groups[(columns, kind)] = []
    transformers = []
    for index, ((columns, family), stages) in enumerate(groups.items()):
        ids = [adapter for adapter, _ in stages]
        if len(ids) != len(set(ids)):
            raise ValueError("Один и тот же этап не повторяется в одной ветке. Измените его параметры.")
        if family == "numeric" and "numeric" in ids:
            if len(stages) != 1:
                raise ValueError("Составной numeric уже задаёт числовую подготовку; используйте его либо отдельные числовые этапы.")
            params = stages[0][1]
            if task != "regression" and params.get("selection", "none") != "none":
                raise ValueError("Отбор внутри совместимого numeric только для регрессии; добавьте selection.univariate.")
            transformer = build_legacy_preprocessor(X.loc[:, list(columns)], params)[0]
        elif family == "numeric":
            chain = [("finite", FiniteNumeric())]
            if "numeric.impute" not in ids:
                chain.append(("impute", SimpleImputer(strategy="median", keep_empty_features=True)))
            for position, (adapter, params) in enumerate(stages):
                if adapter == "numeric.impute" and position != 0:
                    raise ValueError("Заполнение числовых пропусков должно быть первым этапом ветки.")
                if adapter == "numeric.impute" and params.get("strategy") == "knn" and len(X) > 20_000:
                    raise ValueError("KNN заполнение допускает не более 20 000 строк.")
                if adapter == "numeric.impute" and params.get("strategy") == "iterative" and (len(X) > 10_000 or len(columns) > 100 or len(X) * len(columns) > 500_000):
                    raise ValueError("Итеративное заполнение: не более 10 000 строк, 100 признаков и 500 000 ячеек.")
                chain.append((f"step{position}", _numeric_factory(adapter, params, task, seed)))
            if not stages:
                chain.append(("scale", StandardScaler()))
            transformer = Pipeline(chain)
        elif family == "categorical":
            if len(stages) > 1:
                raise ValueError("Для категориальной ветки выберите один способ кодирования.")
            adapter, params = stages[0] if stages else ("categorical.onehot", {})
            if adapter == "categorical.target" and config.get("split_kind") in {"group", "group_kfold", "group_shuffle_split", "leave_one_group_out", "timeseries"}:
                raise ValueError("Target encoding отключён для группового/временного разделения: внутренний crossfit должен соблюдать те же ограничения.")
            transformer = Pipeline([("stringify", StringCategories()), ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)), ("encode", _categorical_factory(adapter, params, seed, task))])
        elif family == "text":
            if len(stages) != 1:
                raise ValueError("Текстовая ветка содержит один vectorizer; снижение размерности задайте отдельным глобальным этапом.")
            adapter, params = stages[0]
            kwargs = {"analyzer": params.get("analyzer", "word"), "ngram_range": (1, params.get("ngram_max", 1)), "lowercase": params.get("lowercase", True)}
            if adapter == "text.hash":
                vectorizer = NamedHashingVectorizer(n_features=params.get("n_features", 1000), alternate_sign=params.get("alternate_sign", False), **kwargs)
            else:
                cls = TfidfVectorizer if adapter == "text.tfidf" else CountVectorizer
                vectorizer = cls(max_features=params.get("max_features", 1000), **kwargs)
            transformer = Pipeline([("join", TextJoiner()), ("vectorizer", vectorizer)])
        else:
            if len(stages) > 1:
                raise ValueError("Ветка даты содержит один этап извлечения.")
            params = stages[0][1] if stages else {}
            transformer = Pipeline([("extract", DatetimeExtractor(parts=params.get("parts", ["year", "month", "day", "weekday"]), cyclic=params.get("cyclic", False), timezone=params.get("timezone", "UTC"))), ("impute", SimpleImputer(strategy="median", keep_empty_features=True))])
        transformers.append((f"{family}{index}", transformer, list(columns)))
    pipeline_steps = [("columns", ColumnTransformer(transformers, remainder="drop", sparse_threshold=1.0))]
    for index, (adapter, params) in enumerate(global_steps):
        if adapter == "selection.variance":
            transformer = VarianceThreshold(threshold=params.get("threshold", 0.0))
        elif adapter == "selection.univariate":
            continuous = any(family in {"numeric", "datetime"} or any(stage in {"categorical.frequency", "categorical.target"} for stage, _ in stages) for (_, family), stages in groups.items())
            # Any preceding decomposition supplies dense continuous components.
            continuous = continuous or any(stage.startswith("reduction.") for stage, _ in global_steps[:index])
            transformer = CheckedSelector(method=params.get("method", "auto"), k=params.get("k", 10), task=task, random_state=seed, discrete_features=params.get("discrete_features", "auto"), has_continuous_inputs=continuous)
        elif adapter == "selection.model":
            estimator = (ExtraTreesClassifier if task == "classification" else ExtraTreesRegressor)(n_estimators=30, max_depth=6, random_state=seed, n_jobs=1)
            transformer = SelectFromModel(estimator, threshold=params.get("threshold", "median"), max_features=params.get("max_features", 20))
        else:
            transformer = BoundedReducer(method=adapter.removeprefix("reduction."), n_components=params.get("n_components", 2), random_state=seed)
        pipeline_steps.append((f"global{index}", transformer))
    return PreparedFeatures(pipeline_steps)


def fit_resample(X, y, config=None, task="regression", seed=42, categorical=False, temporal=False):
    """Resample only a caller-provided training matrix; never own the split."""
    config = config or {}
    if not isinstance(config, dict):
        raise ValueError("Настройки пересэмплирования должны быть объектом.")
    method = config.get("method", "none")
    if method == "none":
        return X, y, {"method": "none", "before": len(y), "after": len(y), "fitted_on": "train", "synthetic": False}
    if temporal or task in {"forecasting", "panel", "ranking", "clustering", "anomaly", "reduction"}:
        raise ValueError("Этот пересэмплировщик не сохраняет временной/групповой протокол задачи. Отключите пересэмплирование.")
    if task == "regression":
        regression_config = dict(config)
        regression_config["method"] = {"regression_over": "random_over", "regression_under": "random_under"}.get(method, method)
        if regression_config["method"] not in {"random_over", "random_under", "smoter", "smogn"}:
            raise ValueError("Для непрерывной цели используйте regression_over/regression_under, SMOTER или SMOGN; SMOTE работает с классами.")
        if sparse.issparse(X):
            raise ValueError("Пересэмплирование регрессии требует ограниченную плотную числовую матрицу.")
        return ResamplingService().fit_resample(X, y, regression_config, seed=seed, categorical=categorical)
    if task != "classification":
        raise ValueError("Неизвестная задача пересэмплирования.")
    if method in {"smotenc", "smoten"}:
        raise ValueError("SMOTENC/SMOTEN требуют отдельный план, сохраняющий исходные категории; готовую one-hot матрицу использовать нельзя.")
    if method not in {"random_over", "random_under", "smote", "adasyn", "borderline_smote", "svm_smote", "kmeans_smote", "tomek", "enn", "near_miss", "smoteenn", "smotetomek"}:
        raise ValueError("Неизвестный пересэмплировщик классификации.")
    if categorical and method not in {"random_over", "random_under"}:
        raise ValueError("Этот метод требует исходные числовые признаки. Для закодированных категорий выберите случайное увеличение/уменьшение.")
    if set(config) - {"method", "neighbors", "sampling_strategy", "replacement"}:
        raise ValueError("Неизвестные настройки пересэмплировщика классификации.")
    if len(y) > 10_000 or X.shape[1] > 200 or X.shape[0] * X.shape[1] > 1_000_000:
        raise ValueError("Пересэмплирование: не более 10 000 строк, 200 признаков и 1 млн ячеек.")
    seed = _integer(seed, "Случайное зерно", 0, 2**32 - 1)
    values = X.data if sparse.issparse(X) else np.asarray(X, dtype=float)
    if not np.isfinite(values).all() or len(y) != X.shape[0] or pd.isna(y).any():
        raise ValueError("Перед пересэмплированием заполните пропуски и согласуйте строки X/y.")
    counts = pd.Series(y).value_counts()
    if len(counts) < 2:
        raise ValueError("Для пересэмплирования классификации нужны минимум два класса.")
    neighbors = _integer(config.get("neighbors", 5), "Соседи пересэмплирования", 1, 100)
    strategy = config.get("sampling_strategy", "auto")
    if not isinstance(strategy, (str, dict, float)):
        raise ValueError("sampling_strategy должен быть auto, правилом классов либо долей для двух классов.")
    if isinstance(strategy, float) and (not np.isfinite(strategy) or not 0 < strategy <= 1 or len(counts) != 2):
        raise ValueError("Доля sampling_strategy строго между 0 и 1 включительно и только для двух классов.")
    try:
        from imblearn.combine import SMOTEENN, SMOTETomek
        from imblearn.over_sampling import ADASYN, BorderlineSMOTE, KMeansSMOTE, RandomOverSampler, SMOTE, SVMSMOTE
        from imblearn.under_sampling import EditedNearestNeighbours, NearMiss, RandomUnderSampler, TomekLinks
        from imblearn.utils import check_sampling_strategy
    except ImportError as error:
        raise ValueError("Не установлен imbalanced-learn. Запустите установку зависимостей CML-lab.") from error
    synthetic = method in {"smote", "adasyn", "borderline_smote", "svm_smote", "kmeans_smote", "smoteenn", "smotetomek"}
    if synthetic and counts.min() <= neighbors:
        raise ValueError(f"В наименьшем классе train {counts.min()} строк: число соседей должно быть меньше. Уменьшите neighbors.")
    # Check the public library's class plan before allocating synthetic arrays.
    # A small input can otherwise request millions of duplicates via a dict.
    if method not in {"tomek", "enn"}:
        sampling_type = "under-sampling" if method in {"random_under", "near_miss"} else "over-sampling"
        planned = check_sampling_strategy(strategy, y, sampling_type)
        if sampling_type == "over-sampling":
            projected = len(y) + sum(planned.values())
        else:
            projected = sum(planned.get(label, count) for label, count in counts.items())
        if projected > 100_000 or projected * X.shape[1] > MAX_DENSE_CELLS:
            raise ValueError("Запрошенный пересэмплировщик создаст больше 100 000 строк или 15 млн ячеек. Уменьшите sampling_strategy.")
    if method == "random_over":
        sampler = RandomOverSampler(sampling_strategy=strategy, random_state=seed)
    elif method == "random_under":
        sampler = RandomUnderSampler(sampling_strategy=strategy, random_state=seed, replacement=_bool(config.get("replacement", False), "Выбор с возвращением"))
    elif method in {"tomek", "enn", "near_miss"}:
        cls = {"tomek": TomekLinks, "enn": EditedNearestNeighbours, "near_miss": NearMiss}[method]
        kwargs = {"sampling_strategy": strategy, "n_jobs": 1}
        if method != "tomek":
            kwargs["n_neighbors"] = neighbors
        sampler = cls(**kwargs)
    elif method in {"smoteenn", "smotetomek"}:
        cls = SMOTEENN if method == "smoteenn" else SMOTETomek
        sampler = cls(sampling_strategy=strategy, random_state=seed, smote=SMOTE(sampling_strategy=strategy, k_neighbors=neighbors, random_state=seed), n_jobs=1)
    else:
        cls = {"smote": SMOTE, "adasyn": ADASYN, "borderline_smote": BorderlineSMOTE, "svm_smote": SVMSMOTE, "kmeans_smote": KMeansSMOTE}[method]
        kwargs = {"sampling_strategy": strategy, "random_state": seed, "n_neighbors" if method == "adasyn" else "k_neighbors": neighbors}
        sampler = cls(**kwargs)
    result_X, result_y = sampler.fit_resample(X, y)
    if len(result_y) > 100_000 or result_X.shape[0] * result_X.shape[1] > MAX_DENSE_CELLS:
        raise ValueError("Результат пересэмплирования превышает 100 000 строк или 15 млн ячеек.")
    summary = {"method": method, "before": len(y), "after": len(result_y), "fitted_on": "train", "synthetic": synthetic,
               "class_counts_before": {str(key): int(value) for key, value in counts.items()},
               "class_counts_after": {str(key): int(value) for key, value in pd.Series(result_y).value_counts().items()}}
    return result_X, result_y, summary
