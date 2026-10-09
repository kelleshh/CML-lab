"""Bounded cross-validation plans and fold-local regression evaluation.

Strategies are scikit-learn splitters. Target-bin stratification is an explicit
engineering convenience for regression, never evidence that rows are independent.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.model_selection import (
    GroupKFold, GroupShuffleSplit, KFold, LeaveOneGroupOut, LeaveOneOut,
    RepeatedKFold, ShuffleSplit, StratifiedKFold, TimeSeriesSplit,
)
from threadpoolctl import threadpool_limits

from .metrics import MetricRegistry

MAX_CV_FITS = 100
MAX_SEARCH_FITS = 500
GROUP_STRATEGIES = frozenset({"group_kfold", "group_shuffle_split", "leave_one_group_out"})
CV_STRATEGIES = (
    {"id": "kfold", "name": "KFold · по очереди проверять каждую часть", "groups": False},
    {"id": "repeated_kfold", "name": "Repeated KFold · несколько новых разбиений", "groups": False},
    {"id": "shuffle_split", "name": "ShuffleSplit · случайные проверочные части", "groups": False},
    {"id": "timeseries", "name": "TimeSeriesSplit · прошлое → будущее", "groups": False},
    {"id": "group_kfold", "name": "Group KFold · целые группы отдельно", "groups": True},
    {"id": "group_shuffle_split", "name": "Group ShuffleSplit · случайные целые группы", "groups": True},
    {"id": "leave_one_group_out", "name": "Leave One Group Out · одна группа на проверку", "groups": True},
    {"id": "leave_one_out", "name": "Leave One Out · одна строка на проверку", "groups": False},
    {"id": "stratified_bins", "name": "По диапазонам цели · экспериментальная стратификация", "groups": False},
)


@dataclass(frozen=True)
class CVPlan:
    strategy: str
    splits: tuple[tuple[np.ndarray, np.ndarray], ...]
    note: str

    @property
    def n_splits(self):
        return len(self.splits)


def _integer(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError(f"{name}: нужно целое число от {low} до {high}.")
    return int(value)


def bounded_jobs(value=1):
    return _integer(value, "Число параллельных работников", 1, 4)


def _groups(groups, n_samples):
    if groups is None:
        raise ValueError("Для групповой проверки выбери колонку с идентификатором группы.")
    result = np.asarray(groups).reshape(-1)
    if len(result) != n_samples or pd.isna(result).any():
        raise ValueError("Группы должны быть заданы для каждой строки без пропусков.")
    # Mixed numbers/strings cannot be sorted by sklearn's numpy.unique.
    return np.asarray([str(value) for value in result])


def group_values(data_service, request):
    """Read a split-only column through the dataset service's public resolver."""
    config = request.get("cv_config") or {}
    column = config.get("group_column") or request.get("split", {}).get("group_column")
    if not column:
        return None
    if column == request.get("target"):
        raise ValueError("Целевая колонка не может служить идентификатором группы.")
    target = request.get("target") or data_service.describe(request["dataset_id"]).get("default_target")
    if column == target:
        raise ValueError("Целевая колонка не может служить идентификатором группы.")
    values = data_service.read_column(request["dataset_id"], column)
    return _groups(values, len(values))


def model_features(data_service, request):
    """Exclude a group identifier from model features even if the UI selected it."""
    group_column = (request.get("cv_config") or {}).get("group_column") or request.get("split", {}).get("group_column")
    features = request.get("features")
    if group_column:
        if features is None:
            bundle = data_service.resolve(request["dataset_id"], request.get("target"))
            features = bundle.feature_names
        features = [name for name in features if name != group_column]
        if not features:
            raise ValueError("После удаления идентификатора группы не осталось признаков.")
    return features


def split_holdout(n_samples, config, seed=42, groups=None, cv_config=None):
    """Create disjoint train/validation/test, respecting groups or chronology."""
    # Keep the established split validation and defaults in one place.
    from .training import TrainingService
    config = dict(config or {})
    strategy = (cv_config or {}).get("strategy", "kfold")
    if strategy == "timeseries":
        config["shuffle"] = False
    ordinary = TrainingService._split(n_samples, config, seed)
    if groups is None:
        if strategy in GROUP_STRATEGIES:
            raise ValueError("Выбрана групповая проверка, но колонка группы не указана.")
        return ordinary
    if strategy == "timeseries" or config.get("shuffle") is False:
        raise ValueError("Хронологическое и групповое разделение нельзя включать одновременно. Выбери одно правило.")
    groups = _groups(groups, n_samples)
    if len(np.unique(groups)) < 3:
        raise ValueError("Для трех отдельных выборок нужны хотя бы три группы.")
    train_fraction = float(config.get("train", .6))
    val_fraction = float(config.get("validation", .2))
    first = GroupShuffleSplit(n_splits=1, train_size=train_fraction, random_state=seed)
    train, rest = next(first.split(np.arange(n_samples), groups=groups))
    if len(np.unique(groups[rest])) < 2:
        raise ValueError("В оставшейся части меньше двух групп. Уменьши обучающую долю.")
    second = GroupShuffleSplit(n_splits=1, train_size=val_fraction / (1 - train_fraction), random_state=seed + 1)
    val_local, test_local = next(second.split(rest, groups=groups[rest]))
    parts = {"train": train, "validation": rest[val_local], "test": rest[test_local]}
    if len(train) < 2:
        raise ValueError("Обучающая группа содержит меньше двух строк.")
    return parts


def build_cv(config, n_samples, y=None, groups=None, seed=42, max_fits=MAX_CV_FITS):
    """Materialize a validated, bounded plan with indexes local to outer train."""
    config = dict(config or {})
    if n_samples < 3:
        raise ValueError("Перекрестной проверке нужны хотя бы три обучающие строки.")
    strategy = config.get("strategy", "kfold")
    if strategy not in {item["id"] for item in CV_STRATEGIES}:
        raise ValueError(f"Неизвестная стратегия перекрестной проверки: {strategy}.")
    folds = _integer(config.get("folds", config.get("n_splits", 5)), "Число частей", 2, 20)
    repeats = _integer(config.get("repeats", 2), "Число повторов", 1, 10)
    shuffle = config.get("shuffle", True)
    if not isinstance(shuffle, bool):
        raise ValueError("Перемешивание в CV должно быть true или false.")
    dummy = np.arange(n_samples)
    target = y
    note = "Проверка проводится только внутри внешней обучающей части. На каждой части преобразования обучаются заново."
    if strategy == "kfold":
        splitter = KFold(folds, shuffle=shuffle, random_state=seed if shuffle else None)
    elif strategy == "repeated_kfold":
        splitter = RepeatedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)
    elif strategy in {"shuffle_split", "group_shuffle_split"}:
        fraction = float(config.get("test_size", .2))
        if not np.isfinite(fraction) or not 0 < fraction < 1:
            raise ValueError("Доля проверочной части в CV должна быть между 0 и 1.")
        cls = GroupShuffleSplit if strategy == "group_shuffle_split" else ShuffleSplit
        splitter = cls(n_splits=folds, test_size=fraction, random_state=seed)
        note += " Проверочные части разных повторов могут пересекаться."
    elif strategy == "timeseries":
        gap = _integer(config.get("gap", 0), "Разрыв по времени", 0, n_samples)
        max_train = config.get("max_train_size")
        if max_train is not None:
            max_train = _integer(max_train, "Максимальный размер обучения", 2, n_samples)
        test_size = config.get("test_size")
        if test_size is not None:
            test_size = _integer(test_size, "Число строк проверки по времени", 1, n_samples)
        splitter = TimeSeriesSplit(folds, gap=gap, max_train_size=max_train, test_size=test_size)
        note += " Строки должны быть заранее упорядочены по времени; равные интервалы времени нужны для сопоставимых горизонтов. Разрыв gap действует только внутри CV; между внешними train, validation и test дополнительного разрыва нет."
    elif strategy == "group_kfold":
        splitter = GroupKFold(folds, shuffle=shuffle, random_state=seed if shuffle else None)
    elif strategy == "leave_one_group_out":
        splitter = LeaveOneGroupOut()
    elif strategy == "leave_one_out":
        if n_samples > max_fits:
            raise ValueError(f"Leave One Out потребует {n_samples} обучений; предел {max_fits}. Уменьши выборку.")
        splitter = LeaveOneOut()
        note += " На одной проверочной строке R² и объясненная дисперсия не определены; используй MAE или MSE."
    else:
        if y is None or len(y) != n_samples or not np.all(np.isfinite(y)):
            raise ValueError("Стратификации по диапазонам нужна конечная числовая цель для всех обучающих строк.")
        bins = _integer(config.get("bins", 5), "Число диапазонов цели", 2, 20)
        edges = np.unique(np.quantile(np.asarray(y), np.linspace(0, 1, bins + 1)))
        if len(edges) < 3:
            raise ValueError("Цель содержит слишком мало разных значений для стратификации по диапазонам.")
        target = np.digitize(y, edges[1:-1], right=True)
        if np.bincount(target).min() < folds:
            raise ValueError("В одном диапазоне цели меньше строк, чем частей CV. Уменьши число диапазонов или частей.")
        splitter = StratifiedKFold(folds, shuffle=shuffle, random_state=seed if shuffle else None)
        note += " Диапазоны вычислены только по внешнему train. Это приближенное выравнивание распределения цели, а не стандартная стратификация для регрессии; оно не устраняет зависимость групп или времени."
    if strategy in GROUP_STRATEGIES:
        groups = _groups(groups, n_samples)
        if strategy == "leave_one_group_out" and len(np.unique(groups)) > max_fits:
            raise ValueError(f"Число групп превышает предел {max_fits} обучений.")
    try:
        expected = splitter.get_n_splits(dummy, target, groups if strategy in GROUP_STRATEGIES else None)
        if expected > max_fits:
            raise ValueError(f"Проверка требует {expected} обучений; предел {max_fits}.")
        splits = tuple((np.asarray(a, dtype=int), np.asarray(b, dtype=int)) for a, b in splitter.split(dummy, target, groups if strategy in GROUP_STRATEGIES else None))
    except ValueError as exc:
        raise ValueError(f"Нельзя построить выбранную перекрестную проверку: {exc}") from exc
    if any(len(a) < 2 or len(b) < 1 for a, b in splits):
        raise ValueError("Каждой части нужны хотя бы две обучающие строки и одна проверочная.")
    if sum(len(a) for a, _ in splits) > 3_000_000:
        raise ValueError("Перекрестная проверка обработает более 3 миллионов обучающих строк. Уменьши число повторов или размер набора.")
    return CVPlan(strategy, splits, note)


def _check(cancelled):
    if cancelled():
        from .training import TrainingCancelled
        raise TrainingCancelled("Обучение отменено пользователем.")


def single_threaded(estimator):
    """Cap nested joblib workers; BLAS is capped once around parallel execution."""
    candidate = clone(estimator)
    params = candidate.get_params(deep=True)
    workers = {name: 1 for name in params if name == "n_jobs" or name.endswith("__n_jobs")}
    if workers:
        candidate.set_params(**workers)
    return candidate


def _fold(pipeline, X, y, train, valid, number, selection, expression, metric_params, cancelled):
    _check(cancelled)
    candidate = single_threaded(pipeline)
    started = time.perf_counter()
    candidate.fit(X.iloc[train] if hasattr(X, "iloc") else X[train], y[train])
    _check(cancelled)
    prediction = candidate.predict(X.iloc[valid] if hasattr(X, "iloc") else X[valid])
    values, details = MetricRegistry().evaluate(y[valid], prediction, selection, expression, metric_params)
    return {"fold": number, "train_size": len(train), "validation_size": len(valid), "metrics": values, "metric_details": details, "seconds": time.perf_counter() - started,
            "train_indices": train.tolist() if len(train) <= 1000 else None,
            "validation_indices": valid.tolist() if len(valid) <= 1000 else None,
            "resampling": getattr(candidate, "resampling_summary_", None)}


def summarize(folds):
    summary = {}
    for name in folds[0]["metrics"] if folds else []:
        values = [fold["metrics"][name] for fold in folds if fold["metrics"].get(name) is not None]
        summary[name] = {"mean": float(np.mean(values)) if values else None, "std": float(np.std(values)) if values else None, "valid_folds": len(values)}
    return summary


def evaluate_cv(pipeline, X, y, config, selection=None, expression=None, metric_params=None,
                groups=None, seed=42, progress=lambda event: None, cancelled=lambda: False,
                n_jobs=1, plan=None):
    """Clone the whole unfitted pipeline per fold; return real fold scores."""
    n_jobs = bounded_jobs(n_jobs)
    y = np.asarray(y, dtype=float)
    plan = plan or build_cv(config, len(y), y, groups, seed)
    results = []
    tasks = (delayed(_fold)(pipeline, X, y, train, valid, i + 1, selection, expression, metric_params, cancelled)
             for i, (train, valid) in enumerate(plan.splits))
    with threadpool_limits(limits=1):
        completed = Parallel(n_jobs=n_jobs, prefer="threads", return_as="generator")(tasks)
        for result in completed:
            _check(cancelled)
            results.append(result)
            progress({"progress": len(results) / plan.n_splits, "message": f"Перекрестная проверка: {len(results)} из {plan.n_splits}.", "cv_fold": result})
    return {"folds": results, "summary": summarize(results), "kind": plan.strategy, "fitted_on": "train", "note": plan.note, "n_jobs": n_jobs}
