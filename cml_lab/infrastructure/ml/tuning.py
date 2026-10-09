"""Library-backed search on supplied training-only fold plans.

Successive halving delegates resource allocation and elimination to sklearn.
Its random row subsampling is unsuitable for temporal or query/group splits.
https://scikit-learn.org/1.8/modules/generated/sklearn.model_selection.HalvingGridSearchCV.html
https://optuna.readthedocs.io/en/stable/faq.html#how-can-i-obtain-reproducible-optimization-results
"""

from __future__ import annotations

from copy import deepcopy
import math
import sys
import time

from joblib import Parallel, delayed, parallel_backend
import numpy as np
from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator
from sklearn.model_selection import ParameterGrid, ParameterSampler
from threadpoolctl import threadpool_limits

from .fitting import fit_artifact

MAX_TRIALS = 100
MAX_FITS = 500
METHODS = ("grid", "random", "optuna_tpe", "optuna_random", "halving_grid", "halving_random")
_ALIASES = {"optuna": "optuna_tpe", "tpe": "optuna_tpe", "halving": "halving_grid"}
_SUPPORTED_TASKS = {"regression", "classification", "ranking", "forecasting", "panel"}
_PROBABILITY_METRICS = {"log_loss", "roc_auc", "average_precision", "brier"}


def _check(cancelled):
    if cancelled():
        raise InterruptedError("Подбор параметров отменен.")


def _integer(value, label, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError(f"{label}: нужно целое число от {low} до {high}.")
    return int(value)


def _params(values):
    return {key: value.item() if isinstance(value, np.generic) else deepcopy(value) for key, value in values.items()}


def _settings(spec, metrics):
    config = spec.get("search")
    if not isinstance(config, dict):
        raise ValueError("Настройки подбора должны быть объектом search.")
    config = deepcopy(config)
    method = _ALIASES.get(config.get("method", "optuna_tpe"), config.get("method", "optuna_tpe"))
    if method not in METHODS:
        raise ValueError("Выберите сетку, случайный поиск, Optuna или последовательное сокращение.")
    if spec["task"] not in _SUPPORTED_TASKS:
        raise ValueError("Этот поиск оценивает новые ответы по известной цели. Для обучения без учителя нужен отдельный критерий.")
    trials = _integer(config.get("trials", 20), "Число проб", 1, MAX_TRIALS)
    default = "accuracy" if spec["task"] == "classification" else "ndcg" if spec["task"] == "ranking" else "rmse"
    metric = config.get("metric", default)
    definitions = {item["id"]: item for item in metrics.catalogue(spec["task"])}
    if metric not in definitions or metric == "all":
        raise ValueError("Для подбора выберите одну метрику выбранной задачи.")
    if definitions[metric].get("optimizable") is False:
        raise ValueError("Эта величина описывает результат, но не оценивает качество и не может быть целью подбора.")
    if metric == "custom" and not spec.get("custom_metric"):
        raise ValueError("Введите формулу собственной метрики custom_metric.")
    direction = config.get("direction", "auto")
    if direction in {None, "auto"}:
        direction = definitions[metric]["direction"]
    if direction not in {"min", "max"}:
        raise ValueError("Направление подбора: min, max или auto.")
    config.update(method=method, trials=trials, metric=metric, direction=direction,
                  n_jobs=_integer(config.get("n_jobs", spec.get("n_jobs", 1)), "Параллельных частей CV", 1, 4))
    return config


def _space(spec, config, catalogue):
    schema = {item["key"]: item for item in catalogue.descriptor(spec["algorithm_id"])["params"]}
    raw = config.get("param_space", config.get("space"))
    if not isinstance(raw, dict) or not raw or len(raw) > 12:
        raise ValueError("Выберите от 1 до 12 параметров и задайте варианты или диапазоны поиска.")
    unknown = set(raw) - set(schema)
    if unknown:
        raise ValueError("Неизвестные параметры поиска: " + ", ".join(sorted(map(str, unknown))))
    base = spec.get("params") or {}
    catalogue.build(spec["task"], spec["algorithm_id"], base, spec.get("seed", 42), 1)
    result = {}
    for name, definition in raw.items():
        if isinstance(definition, list):
            item = {"type": "categorical", "choices": definition}
        elif isinstance(definition, dict):
            kind = definition.get("type", "float")
            if kind in {"categorical", "select", "bool"}:
                item = {"type": "categorical", "choices": definition.get("choices", definition.get("values"))}
            elif kind in {"int", "float"}:
                low, high = definition.get("low"), definition.get("high")
                if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in (low, high)) or low >= high:
                    raise ValueError(f"{name}: задайте конечные границы low < high.")
                if kind == "int" and (int(low) != low or int(high) != high):
                    raise ValueError(f"{name}: границы целочисленного диапазона должны быть целыми.")
                logarithmic = definition.get("log", False)
                if not isinstance(logarithmic, bool) or logarithmic and low <= 0:
                    raise ValueError(f"{name}: логарифмический диапазон должен быть положительным.")
                if "values" in definition or "choices" in definition:
                    raise ValueError(f"{name}: задайте либо список, либо диапазон.")
                item = {"type": kind, "low": int(low) if kind == "int" else float(low),
                        "high": int(high) if kind == "int" else float(high), "log": logarithmic}
            else:
                raise ValueError(f"{name}: доступны типы int, float или categorical.")
        else:
            raise ValueError(f"{name}: нужны список вариантов или объект диапазона.")
        if item["type"] == "categorical":
            choices = item["choices"]
            if not isinstance(choices, list) or not 1 <= len(choices) <= 100:
                raise ValueError(f"{name}: задайте от 1 до 100 вариантов.")
            if config["method"].startswith("optuna") and any(value is not None and not isinstance(value, (str, bool, int, float)) for value in choices):
                raise ValueError("Optuna categorical принимает строки, числа, bool и null; составные варианты задайте сеткой.")
            probes = choices
        else:
            probes = [item["low"], item["high"]]
        for value in probes:
            catalogue.build(spec["task"], spec["algorithm_id"], {**base, name: value}, spec.get("seed", 42), 1)
        result[name] = item
    return result


def _grid(space):
    if any(item["type"] != "categorical" for item in space.values()):
        raise ValueError("Для сетки перечислите точные варианты каждого параметра.")
    return {name: item["choices"] for name, item in space.items()}


def _distributions(space):
    result = {}
    for name, item in space.items():
        if item["type"] == "categorical":
            result[name] = item["choices"]
        elif item["type"] == "int":
            if item["log"]:
                raise ValueError("Логарифмический целочисленный диапазон доступен в Optuna; здесь используйте список вариантов.")
            result[name] = randint(item["low"], item["high"] + 1)
        else:
            result[name] = loguniform(item["low"], item["high"]) if item["log"] else uniform(item["low"], item["high"] - item["low"])
    return result


def _plans(plans, n, groups):
    result = []
    for train, valid in plans:
        train, valid = np.asarray(train), np.asarray(valid)
        if any(part.ndim != 1 or not np.issubdtype(part.dtype, np.integer) or len(part) < 1 or np.any(part < 0) or np.any(part >= n) or len(np.unique(part)) != len(part) for part in (train, valid)):
            raise ValueError("План CV содержит неверные индексы обучающих строк.")
        if np.intersect1d(train, valid).size:
            raise ValueError("Обучение и проверка одной части CV не должны пересекаться.")
        if groups is not None and np.intersect1d(np.asarray(groups)[train], np.asarray(groups)[valid]).size:
            raise ValueError("Группы запросов или объектов не должны пересекаться между обучением и проверкой CV.")
        result.append((train, valid))
    if not 1 <= len(result) <= 100:
        raise ValueError("Передайте от 1 до 100 обучающих частей CV.")
    return result


def _budget(fits):
    if fits > MAX_FITS:
        raise ValueError(f"Поиск потребует до {fits} обучений; предел {MAX_FITS}. Сократите пробы или части CV.")


def _score(artifact, X, y, spec, metrics, config, label_encoder, groups):
    probabilities = artifact.predict_proba(X) if config["metric"] in _PROBABILITY_METRICS else None
    classes = len(label_encoder.classes_) if label_encoder is not None else len(np.unique(y)) if spec["task"] == "classification" else None
    values, details = metrics.evaluate(spec["task"], y, artifact.predict_encoded(X), selection=[config["metric"]],
                                      probabilities=probabilities, n_classes=classes, groups=groups,
                                      expression=spec.get("custom_metric"), metric_params=spec.get("metric_params"))
    value = values.get(config["metric"])
    if value is None or not np.isfinite(value):
        reason = details.get(config["metric"], {}).get("reason")
        raise ValueError(reason or "Главная метрика не определена на этой части CV.")
    return float(value)


def _evaluate(candidate, spec, X, y, catalogue, metrics, plans, config, label_encoder, groups, weights, cancelled):
    started = time.perf_counter()
    trial_spec = dict(spec, params={**(spec.get("params") or {}), **_params(candidate)}, n_jobs=1, search=None)
    def fold(train, valid):
        _check(cancelled)
        try:
            artifact, _, _ = fit_artifact(trial_spec, X.iloc[train], y[train], catalogue, label_encoder=label_encoder,
                                          groups=None if groups is None else groups[train], weights=None if weights is None else weights[train], cancelled=cancelled)
            _check(cancelled)
            return _score(artifact, X.iloc[valid], y[valid], trial_spec, metrics, config, label_encoder,
                          None if groups is None else groups[valid]), None
        except (ValueError, TypeError, ArithmeticError, RuntimeError) as error:
            return None, str(error)
    results = Parallel(n_jobs=config["n_jobs"], prefer="threads")(delayed(fold)(train, valid) for train, valid in plans)
    scores = [value for value, _ in results]
    errors = [error for _, error in results if error]
    return {"params": trial_spec["params"], "score": None if errors else float(np.mean(scores)),
            "std": None if errors else float(np.std(scores)), "fold_scores": scores,
            "status": "failed" if errors else "complete", "reason": errors[0] if errors else None,
            "seconds": time.perf_counter() - started, "fits": len(plans)}


class _CandidateDistribution:
    def __init__(self, distributions):
        self.distributions = distributions

    def rvs(self, random_state=None):
        return _params(next(iter(ParameterSampler(self.distributions, n_iter=1, random_state=random_state))))


class _FoldEstimator(BaseEstimator):
    """Sklearn search estimator whose fit compiles a fresh complete fold artifact."""

    def __init__(self, spec, catalogue, label_encoder=None, weights=None, cancelled=None, candidate=None):
        self.spec = spec
        self.catalogue = catalogue
        self.label_encoder = label_encoder
        self.weights = weights
        self.cancelled = cancelled
        self.candidate = candidate

    def fit(self, X, y):
        spec = dict(self.spec, params={**(self.spec.get("params") or {}), **(self.candidate or {})}, n_jobs=1, search=None)
        weights = None if self.weights is None else np.asarray(self.weights)[X.index.to_numpy()]
        self.artifact_, _, _ = fit_artifact(spec, X, np.asarray(y), self.catalogue, label_encoder=self.label_encoder,
                                          weights=weights, cancelled=self.cancelled)
        return self


def _halving(spec, X, y, catalogue, metrics, plans, config, space, label_encoder, groups, weights, progress, cancelled):
    if spec["task"] in {"ranking", "forecasting", "panel"} or groups is not None or spec.get("split", {}).get("shuffle") is False:
        raise ValueError("Сокращение по случайным строкам не сохраняет группы или хронологию. Для них используйте grid, random или Optuna.")
    from sklearn.experimental import enable_halving_search_cv  # noqa: F401
    from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV
    factor = _integer(config.get("factor", 3), "Фактор сокращения", 2, 5)
    minimum = max(10, len(plans) * 4 * (len(label_encoder.classes_) if label_encoder is not None else 1))
    minimum = _integer(config.get("min_resources", minimum), "Начальный бюджет строк", minimum, len(X))
    levels = 1 + int(math.log(max(len(X) / minimum, 1), factor))
    _budget(config["candidate_count"] * len(plans) * levels)
    sign = -1 if config["direction"] == "min" else 1
    estimator = _FoldEstimator(spec, catalogue, label_encoder, weights, cancelled)
    def scorer(estimator, valid_X, valid_y):
        _check(cancelled)
        return sign * _score(estimator.artifact_, valid_X, valid_y, spec, metrics, config, label_encoder, None)
    common = dict(factor=factor, resource="n_samples", min_resources=minimum, max_resources=len(X),
                  scoring=scorer, cv=plans, refit=False, n_jobs=config["n_jobs"], random_state=spec.get("seed", 42),
                  error_score=np.nan, return_train_score=False)
    if config["method"] == "halving_grid":
        candidates = list(ParameterGrid(_grid(space)))
        searcher = HalvingGridSearchCV(estimator, {"candidate": candidates}, **common)
    else:
        if all(item["type"] == "categorical" for item in space.values()):
            candidates = list(ParameterSampler(_grid(space), n_iter=config["candidate_count"], random_state=spec.get("seed", 42)))
            distribution = candidates
        else:
            distribution = _CandidateDistribution(_distributions(space))
        searcher = HalvingRandomSearchCV(estimator, {"candidate": distribution},
                                        n_candidates=config["candidate_count"], **common)
    progress({"type": "search", "progress": .12, "message": "Sklearn распределяет бюджеты строк между кандидатами"})
    with parallel_backend("threading"):
        searcher.fit(X, y)
    _check(cancelled)
    raw = searcher.cv_results_
    records = []
    for index, params in enumerate(raw["params"]):
        scores = [float(raw[f"split{fold}_test_score"][index]) * sign for fold in range(len(plans))]
        valid = np.isfinite(scores).all()
        records.append({"trial": index + 1, "params": {**(spec.get("params") or {}), **_params(params["candidate"])},
                        "score": float(raw["mean_test_score"][index]) * sign if valid else None,
                        "std": float(raw["std_test_score"][index]) if valid else None,
                        "fold_scores": [value if np.isfinite(value) else None for value in scores],
                        "status": "complete" if valid else "failed", "iteration": int(raw["iter"][index]),
                        "resources": int(raw["n_resources"][index]), "fits": len(plans),
                        "seconds": float(raw["mean_fit_time"][index]) * len(plans)})
    last = max(record["iteration"] for record in records)
    return records, [record for record in records if record["iteration"] == last], "sklearn.model_selection.HalvingSearchCV"


def tune(spec, X, y, catalogue, metrics, plans, label_encoder=None, groups=None, weights=None,
         progress=lambda event: None, cancelled=lambda: False):
    """Return winning full parameters and genuine trial scores without refitting.

    The caller supplies only the outer training table and its fold plans. The
    winner is refitted by the experiment runner, not by the search adapter.
    """
    _check(cancelled)
    spec = deepcopy(spec)
    config = _settings(spec, metrics)
    y = np.asarray(y)
    groups = None if groups is None else np.asarray(groups)
    weights = None if weights is None else np.asarray(weights)
    if len(X) != len(y) or any(value is not None and len(value) != len(X) for value in (groups, weights)):
        raise ValueError("Длины обучающей таблицы, цели, групп и весов должны совпадать.")
    plans = _plans(plans, len(X), groups)
    descriptor = catalogue.descriptor(spec["algorithm_id"])
    if config["metric"] in _PROBABILITY_METRICS and not descriptor.get("capabilities", {}).get("predict_proba"):
        raise ValueError("Для этой метрики поиска нужен алгоритм, возвращающий вероятности классов.")
    space = _space(spec, config, catalogue)
    method = config["method"]
    count = config["trials"]
    if method in {"grid", "halving_grid"}:
        cardinality = math.prod(len(values) for values in _grid(space).values())
        if cardinality > count:
            raise ValueError(f"Сетка содержит {cardinality} комбинаций; бюджет {count}. Сократите сетку или выберите случайный поиск.")
        count = cardinality
    elif method in {"random", "halving_random"} and all(item["type"] == "categorical" for item in space.values()):
        cardinality = math.prod(len(item["choices"]) for item in space.values())
        if cardinality > sys.maxsize:
            raise ValueError("Дискретное пространство слишком велико для индексации sklearn. Сократите варианты или задайте числовые диапазоны.")
        count = min(count, cardinality)
    _budget(count * len(plans))
    config["candidate_count"] = count
    X = X.reset_index(drop=True)
    records = []
    def evaluate(candidate):
        _check(cancelled)
        record = _evaluate(candidate, spec, X, y, catalogue, metrics, plans, config, label_encoder, groups, weights, cancelled)
        record["trial"] = len(records) + 1
        records.append(record)
        progress({"type": "search", "progress": .12 + .28 * len(records) / max(count, 1),
                  "message": f"Подбор: проба {len(records)} из {count}", "trial": deepcopy(record)})
        return record
    with threadpool_limits(limits=1):
        if method.startswith("halving"):
            records, eligible, backend = _halving(spec, X, y, catalogue, metrics, plans, config, space, label_encoder, groups, weights, progress, cancelled)
            for record in records:
                progress({"type": "search", "progress": .4, "message": "Завершен этап sklearn halving", "trial": deepcopy(record)})
        elif method in {"grid", "random"}:
            candidates = ParameterGrid(_grid(space)) if method == "grid" else ParameterSampler(_distributions(space), n_iter=count, random_state=spec.get("seed", 42))
            for candidate in candidates:
                evaluate(candidate)
            eligible, backend = records, "sklearn.model_selection.ParameterGrid" if method == "grid" else "sklearn.model_selection.ParameterSampler"
        else:
            import optuna
            seed = spec.get("seed", 42)
            sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=min(10, max(2, count // 3))) if method == "optuna_tpe" else optuna.samplers.RandomSampler(seed=seed)
            study = optuna.create_study(direction="minimize" if config["direction"] == "min" else "maximize", sampler=sampler)
            class FailedTrial(ValueError):
                pass
            def objective(trial):
                candidate = {}
                for name, item in space.items():
                    if item["type"] == "categorical": candidate[name] = trial.suggest_categorical(name, item["choices"])
                    elif item["type"] == "int": candidate[name] = trial.suggest_int(name, item["low"], item["high"], log=item["log"])
                    else: candidate[name] = trial.suggest_float(name, item["low"], item["high"], log=item["log"])
                record = evaluate(candidate)
                record["optuna_number"] = trial.number
                if record["score"] is None:
                    raise FailedTrial(record.get("reason") or "Метрика не определена.")
                return record["score"]
            # Parallelize folds; preserve seeded sequential sampler suggestions.
            study.optimize(objective, n_trials=count, n_jobs=1, catch=(FailedTrial,), show_progress_bar=False)
            eligible, backend = records, f"optuna.{type(sampler).__name__}"
    _check(cancelled)
    complete = [record for record in eligible if record.get("score") is not None and np.isfinite(record["score"])]
    if not complete:
        raise ValueError("Ни одна проба не дала конечной метрики. " + str(next((record.get("reason") for record in records if record.get("reason")), "Проверьте диапазоны и размеры частей.")))
    winner = (min if config["direction"] == "min" else max)(complete, key=lambda record: record["score"])
    summary = {"method": method, "backend": backend, "metric": config["metric"], "direction": config["direction"],
               "best_params": deepcopy(winner["params"]), "best_score": winner["score"], "trials": records,
               "n_trials": len(records), "n_fits": sum(record["fits"] for record in records), "n_splits": len(plans),
               "scope": "Только внутренние части внешнего обучения; преобразования и пересэмплирование обучены заново в каждой части.",
               "refit": False}
    return deepcopy(winner["params"]), summary
