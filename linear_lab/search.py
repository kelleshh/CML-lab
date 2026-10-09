"""Leakage-safe bounded parameter search using sklearn and Optuna samplers.

Only outer-training rows participate in selection. Grid/random candidates come
from sklearn; adaptive suggestions come from an actual Optuna Study. The winning
model is refitted on outer train by the same service as ordinary experiments.
"""

from __future__ import annotations

from copy import deepcopy
import math
import time

import numpy as np
from joblib import parallel_backend
from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import ParameterGrid, ParameterSampler
from threadpoolctl import threadpool_limits

from .metrics import MetricRegistry
from .pipeline import RegressionPipeline
from .validation import (
    GROUP_STRATEGIES, MAX_SEARCH_FITS, bounded_jobs, build_cv, evaluate_cv,
    group_values, model_features, single_threaded, split_holdout,
)

SEARCH_METHODS = (
    {"id": "grid", "name": "Перебор сетки", "description": "Проверить каждую заданную комбинацию."},
    {"id": "random", "name": "Случайный поиск", "description": "Проверить случайные комбинации в заданных диапазонах."},
    {"id": "optuna_tpe", "name": "Optuna TPE", "description": "Следующие настройки выбираются по результатам предыдущих попыток."},
    {"id": "optuna_random", "name": "Optuna Random", "description": "Случайные предложения в настоящем Optuna Study."},
    {"id": "halving_grid", "name": "Последовательное сокращение сетки", "description": "Начать с небольших подвыборок и дать больше строк лучшим кандидатам."},
    {"id": "halving_random", "name": "Последовательное сокращение случайных кандидатов", "description": "Случайно выбрать кандидатов и постепенно увеличить бюджет лучших."},
)


class SearchService:
    def __init__(self, data_service, model_registry):
        self.data_service = data_service
        self.model_registry = model_registry
        self.metrics = MetricRegistry()

    @staticmethod
    def _check(cancelled):
        from .training import TrainingService
        TrainingService._check(cancelled)

    def _settings(self, request):
        config = request.get("search")
        if not isinstance(config, dict):
            raise ValueError("Настройки подбора должны быть объектом search.")
        config = deepcopy(config)
        method = config.get("method", "optuna_tpe")
        method = {"optuna": "optuna_tpe", "tpe": "optuna_tpe", "halving": "halving_grid"}.get(method, method)
        if method not in {item["id"] for item in SEARCH_METHODS}:
            raise ValueError("Неизвестный метод подбора гиперпараметров.")
        trials = config.get("trials", 20)
        if isinstance(trials, bool) or not isinstance(trials, int) or not 1 <= trials <= 100:
            raise ValueError("Число попыток подбора должно быть целым от 1 до 100.")
        metric = config.get("metric", "rmse")
        metric_specs = {item["id"]: item for item in self.metrics.catalogue()}
        if metric == "all":
            raise ValueError("Для выбора победителя нужна одна главная метрика. Все остальные можно одновременно показывать в результатах.")
        if metric not in metric_specs:
            raise ValueError("Неизвестная метрика подбора.")
        direction = config.get("direction") or metric_specs[metric]["direction"]
        if direction not in {"min", "max"}:
            raise ValueError("Направление подбора: min (меньше лучше) или max (больше лучше).")
        if metric == "custom" and not request.get("custom_metric"):
            raise ValueError("Для подбора по собственной метрике нужна формула custom_metric.")
        config.update(method=method, trials=trials, metric=metric, direction=direction, n_jobs=bounded_jobs(config.get("n_jobs", request.get("n_jobs", 1))))
        return config

    def _space(self, model_id, base_params, config, seed):
        schema = {p["key"]: p for p in self.model_registry.spec(model_id)["params"]}
        raw = config.get("param_space")
        if raw is None:
            if "alpha" not in schema:
                raise ValueError("У этой модели нет силы штрафа alpha. Укажи параметры для подбора явно.")
            if config["method"] in {"grid", "halving_grid"}:
                raw = {"alpha": np.logspace(-4, 3, min(config["trials"], 9)).tolist()}
            else:
                raw = {"alpha": {"type": "float", "low": max(1e-6, schema["alpha"].get("min", 0)), "high": min(1e3, schema["alpha"].get("max", 1e3)), "log": True}}
        if not isinstance(raw, dict) or not raw or len(raw) > 12:
            raise ValueError("Укажи от 1 до 12 параметров для поиска.")
        unknown = set(raw) - set(schema)
        if unknown:
            raise ValueError("Модель не поддерживает параметры поиска: " + ", ".join(sorted(unknown)))
        space = {}
        for name, definition in raw.items():
            if isinstance(definition, list):
                if not definition or len(definition) > 100:
                    raise ValueError(f"{name}: список вариантов должен содержать от 1 до 100 значений.")
                values = definition
                kind = "categorical"
                item = {"type": kind, "choices": values}
            elif isinstance(definition, dict):
                item = dict(definition)
                kind = item.get("type", "float")
                if kind in {"categorical", "select", "bool"}:
                    values = item.get("choices", item.get("values"))
                    if not isinstance(values, list) or not values or len(values) > 100:
                        raise ValueError(f"{name}: choices должен быть непустым списком до 100 вариантов.")
                    item = {"type": "categorical", "choices": values}
                elif kind in {"float", "int"}:
                    low, high = item.get("low"), item.get("high")
                    if isinstance(low, bool) or isinstance(high, bool) or not isinstance(low, (int, float)) or not isinstance(high, (int, float)) or not np.isfinite([low, high]).all() or low >= high:
                        raise ValueError(f"{name}: нужны конечные границы low < high.")
                    if kind == "int" and (int(low) != low or int(high) != high):
                        raise ValueError(f"{name}: целочисленный диапазон требует целых границ.")
                    if not isinstance(item.get("log", False), bool) or (item.get("log", False) and low <= 0):
                        raise ValueError(f"{name}: логарифмический диапазон должен быть строго положительным.")
                    values = [int(low), int(high)] if kind == "int" else [float(low), float(high)]
                    item = {"type": kind, "low": values[0], "high": values[1], "log": item.get("log", False)}
                    if "values" in definition:
                        raise ValueError(f"{name}: используй список вариантов либо диапазон low/high, не оба.")
                else:
                    raise ValueError(f"{name}: тип поиска должен быть float, int или categorical.")
            else:
                raise ValueError(f"{name}: укажи список вариантов или объект диапазона.")
            for value in values:
                self.model_registry.create(model_id, {**base_params, name: value}, seed)
            space[name] = item
        return space

    @staticmethod
    def _grid(space):
        values = {}
        for name, item in space.items():
            if item["type"] != "categorical":
                raise ValueError(f"Для перебора сетки {name} должен быть задан списком конкретных вариантов.")
            values[name] = item["choices"]
        return values

    @staticmethod
    def _distributions(space):
        result = {}
        for name, item in space.items():
            if item["type"] == "categorical":
                result[name] = item["choices"]
            elif item["type"] == "int":
                if item["log"]:
                    raise ValueError(f"Логарифмический целочисленный диапазон {name} доступен в Optuna; для случайного sklearn-поиска используй список вариантов.")
                result[name] = randint(item["low"], item["high"] + 1)
            else:
                result[name] = loguniform(item["low"], item["high"]) if item["log"] else uniform(item["low"], item["high"] - item["low"])
        return result

    def run(self, request, progress=lambda event: None, cancelled=lambda: False):
        from .training import TrainingService, json_safe
        self._check(cancelled)
        started = time.perf_counter()
        config = self._settings(request)
        seed = int(request.get("seed", 42))
        if not 0 <= seed < 2**32 - 1:
            raise ValueError("seed должен быть от 0 до 4294967294.")
        model_id = request.get("model", "ridge")
        base_params = request.get("params", {})
        estimator = self.model_registry.create(model_id, base_params, seed)
        space = self._space(model_id, base_params, config, seed)
        epochs = request.get("epochs", 100)
        if model_id == "sgd":
            if isinstance(epochs, bool) or not isinstance(epochs, int) or not 1 <= epochs <= 1000:
                raise ValueError("Число эпох SGD должно быть целым от 1 до 1000.")
            if {"max_iter", "tol"}.intersection(space):
                raise ValueError("В интерактивном SGD бюджет задается числом эпох. Настрой epochs отдельно; max_iter и tol не участвуют в подборе, поскольку обучение записывает все эпохи.")
            estimator.set_params(max_iter=epochs, tol=None)
        effective = deepcopy(request)
        effective["features"] = model_features(self.data_service, request)
        bundle = self.data_service.resolve(request["dataset_id"], request.get("target"), effective.get("features"))
        X, _ = TrainingService._clean_features(bundle.X)
        y = np.asarray(bundle.y, dtype=float)
        if not np.isfinite(y).all() or len(y) > 100000:
            raise ValueError("Подбору нужны до 100000 строк с конечной числовой целью.")
        groups = group_values(self.data_service, request)
        cv_config = dict(request.get("cv_config") or {"strategy": "kfold" if request.get("split", {}).get("shuffle", True) else "timeseries", "folds": request.get("cv") or 3})
        if cv_config.get("strategy") in {None, "none"}:
            cv_config = {**cv_config, "strategy": "kfold" if request.get("split", {}).get("shuffle", True) else "timeseries", "folds": 3}
        if cv_config.get("strategy") == "timeseries" and request.get("resampling", {}).get("method", "none") != "none":
            raise ValueError("Подбор для временных рядов требует выключить пересэмплирование, чтобы сохранить хронологию.")
        parts = split_holdout(len(y), request.get("split", {}), seed, groups, cv_config)
        train_indices = parts["train"]
        train_X, train_y = X.iloc[train_indices], y[train_indices]
        train_groups = groups[train_indices] if groups is not None else None
        plan = build_cv(cv_config, len(train_y), train_y, train_groups, seed)
        preprocessor, _, _ = TrainingService._preprocessor(train_X, request.get("preprocessing", {}))
        pipeline = RegressionPipeline([("preprocessing", preprocessor), ("model", estimator)], resampling=request.get("resampling", {}), seed=seed)
        selected = request.get("metrics", ["rmse", "mae", "r2"])
        if selected != "all":
            selected = [selected] if isinstance(selected, str) else list(selected)
            if config["metric"] not in selected and "all" not in selected:
                selected.append(config["metric"])
        expression = request.get("custom_metric")
        metric_params = request.get("metric_params", {})
        trials = []
        best = None
        progress({"progress": .01, "message": "Подбор использует только внешнюю обучающую часть. Проверка и тест отложены."})

        def emit(record, budget):
            nonlocal best
            trials.append(record)
            score = record.get("score")
            if "iteration" in record and trials[:-1] and record["iteration"] != trials[-2].get("iteration"):
                best = None
            if score is not None and (best is None or (score < best["score"] if config["direction"] == "min" else score > best["score"])):
                best = record
            curve = [{"trial": item["trial"], "score": item.get("score"), "best_score": item.get("best_score"), "resources": item.get("resources"), "iteration": item.get("iteration")} for item in trials[:-1]]
            record["best_score"] = best["score"] if best else None
            curve.append({"trial": record["trial"], "score": score, "best_score": record["best_score"], "resources": record.get("resources"), "iteration": record.get("iteration")})
            progress({"progress": .04 + .72 * min(len(trials) / max(budget, 1), 1), "message": f"Подбор: завершено {len(trials)} из {budget} попыток.", "search_trial": json_safe(record), "search_curve": json_safe(curve)})

        def evaluate(params, number, budget):
            self._check(cancelled)
            begin = time.perf_counter()
            try:
                candidate = self.model_registry.create(model_id, {**base_params, **params}, seed)
                if model_id == "sgd":
                    candidate.set_params(max_iter=epochs, tol=None)
                candidate_pipeline = RegressionPipeline([("preprocessing", preprocessor), ("model", candidate)], resampling=request.get("resampling", {}), seed=seed)
                report = evaluate_cv(candidate_pipeline, train_X, train_y, cv_config, selected, expression, metric_params,
                                     train_groups, seed, cancelled=cancelled, n_jobs=config["n_jobs"], plan=plan)
                score_report = report["summary"].get(config["metric"], {})
                # A missing fold must not make an invalid candidate win by scoring
                # only its easy folds. All planned folds must produce the metric.
                score = score_report.get("mean") if score_report.get("valid_folds") == plan.n_splits else None
                record = {"trial": number, "params": params, "score": score, "std": score_report.get("std"), "metrics": report["summary"], "fold_scores": [fold["metrics"].get(config["metric"]) for fold in report["folds"]], "status": "complete" if score is not None else "invalid_metric", "seconds": time.perf_counter() - begin}
                if score is None:
                    record["error"] = next((fold["metric_details"].get(config["metric"], {}).get("reason") for fold in report["folds"] if fold["metrics"].get(config["metric"]) is None), "Метрика не определена на всех частях.")
            except (ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
                record = {"trial": number, "params": params, "score": None, "status": "failed", "error": str(exc), "seconds": time.perf_counter() - begin}
            emit(record, budget)
            return record

        method = config["method"]
        if method in {"grid", "random"}:
            if method == "grid":
                grid = ParameterGrid(self._grid(space))
                if len(grid) > config["trials"]:
                    raise ValueError(f"Сетка содержит {len(grid)} комбинаций, а бюджет {config['trials']}. Увеличь число попыток или сократи варианты.")
                candidates = list(grid)
            else:
                candidates = list(ParameterSampler(self._distributions(space), n_iter=config["trials"], random_state=seed))
            self._budget(len(candidates) * plan.n_splits)
            for index, params in enumerate(candidates, 1):
                evaluate(params, index, len(candidates))
        elif method.startswith("optuna"):
            self._budget(config["trials"] * plan.n_splits)
            try:
                import optuna
            except ImportError as exc:
                raise ValueError("Optuna не установлена. Повтори запуск приложения для установки зависимостей.") from exc
            sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=min(10, max(2, config["trials"] // 3))) if method == "optuna_tpe" else optuna.samplers.RandomSampler(seed=seed)
            study = optuna.create_study(direction="minimize" if config["direction"] == "min" else "maximize", sampler=sampler)

            def objective(trial):
                params = {}
                for name, item in space.items():
                    if item["type"] == "categorical":
                        params[name] = trial.suggest_categorical(name, item["choices"])
                    elif item["type"] == "int":
                        params[name] = trial.suggest_int(name, item["low"], item["high"], log=item["log"])
                    else:
                        params[name] = trial.suggest_float(name, item["low"], item["high"], log=item["log"])
                record = evaluate(params, trial.number + 1, config["trials"])
                if record["score"] is None:
                    raise ValueError(record.get("error", "Непригодная метрика."))
                trial.set_user_attr("fold_scores", record["fold_scores"])
                return record["score"]

            # Sequential proposals make seeded Optuna reproducible. Independent
            # fold fits run in parallel. Never parallelize both levels at once.
            study.optimize(objective, n_trials=config["trials"], n_jobs=1, catch=(ValueError,), show_progress_bar=False)
        else:
            records = self._halving(pipeline, train_X, train_y, space, config, plan, expression, metric_params, seed, cancelled)
            for record in records:
                emit(record, len(records))
        self._check(cancelled)
        if best is None:
            errors = list(dict.fromkeys(record.get("error", "Метрика не определена.") for record in trials))
            raise ValueError("Ни один кандидат не дал пригодную метрику: " + "; ".join(errors[:3]))
        # Successive halving scores use different resource budgets. Only compare
        # candidates from the final rung when choosing the winner.
        if method.startswith("halving"):
            final_iteration = max(record["iteration"] for record in trials)
            eligible = [record for record in trials if record["iteration"] == final_iteration and record["score"] is not None]
            if not eligible:
                raise ValueError("На последнем этапе сокращения нет пригодных кандидатов.")
            best = min(eligible, key=lambda record: record["score"]) if config["direction"] == "min" else max(eligible, key=lambda record: record["score"])
        winner_params = {**base_params, **best["params"]}
        effective["params"] = winner_params
        effective.pop("search", None)
        effective["cv"] = 0
        effective.pop("cv_config", None)
        # Preserve the exact split rule for ordinary refit. CV configuration is
        # also needed to exclude groups and enforce chronology in TrainingService.
        effective["cv_config"] = {**cv_config, "enabled": False}
        if cv_config.get("strategy") == "timeseries":
            effective["split"] = {**request.get("split", {}), "shuffle": False}

        def refit_progress(event):
            event = dict(event)
            event["progress"] = .78 + .22 * float(event.get("progress", 0))
            progress(event)

        result = TrainingService(self.data_service, self.model_registry).run(effective, refit_progress, cancelled)
        result["search"] = {"method": method, "metric": config["metric"], "direction": config["direction"], "trials": trials, "best_params": winner_params, "best_score": best["score"], "best_trial": best["trial"], "cv_config": cv_config, "folds": plan.n_splits, "n_jobs": config["n_jobs"], "seconds": time.perf_counter() - started, "fitted_on": "train", "note": "Все кандидаты проверяются внутри внешнего train; победитель переобучен только на train. Внешние validation и test не выбирают параметры. " + plan.note}
        result["effective_request"] = {key: value for key, value in effective.items() if key != "artifact_path"}
        return json_safe(result)

    @staticmethod
    def _budget(fits):
        if fits > MAX_SEARCH_FITS:
            raise ValueError(f"Подбор потребует {fits} обучений; предел {MAX_SEARCH_FITS}. Сократи число попыток или частей CV.")

    def _halving(self, pipeline, X, y, space, config, plan, expression, metric_params, seed, cancelled):
        from sklearn.experimental import enable_halving_search_cv  # noqa: F401
        from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV
        if plan.strategy in GROUP_STRATEGIES or plan.strategy in {"timeseries", "leave_one_out", "stratified_bins"}:
            raise ValueError("Сокращение по числу строк доступно для KFold, Repeated KFold и ShuffleSplit. Для времени, групп и диапазонов цели используй сетку, случайный поиск или Optuna.")
        factor = config.get("factor", 3)
        if isinstance(factor, bool) or not isinstance(factor, int) or not 2 <= factor <= 5:
            raise ValueError("Фактор сокращения должен быть целым от 2 до 5.")
        minimum = max(10, plan.n_splits * 4)
        if len(y) < minimum:
            raise ValueError("Для последовательного сокращения слишком мало обучающих строк.")
        sign = -1 if config["direction"] == "min" else 1

        def scorer(estimator, check_X, check_y):
            self._check(cancelled)
            pred = estimator.predict(check_X)
            values, _ = self.metrics.evaluate(check_y, pred, [config["metric"]], expression, metric_params)
            value = values.get(config["metric"])
            return sign * value if value is not None else np.nan

        common = dict(factor=factor, resource="n_samples", min_resources=minimum, max_resources=len(y), scoring=scorer, cv=list(plan.splits), refit=False, n_jobs=config["n_jobs"], random_state=seed, error_score=np.nan, return_train_score=False)
        if config["method"] == "halving_grid":
            grid = self._grid(space)
            count = len(ParameterGrid(grid))
            if count > config["trials"]:
                raise ValueError(f"Сетка содержит {count} комбинаций, бюджет {config['trials']}.")
            searcher = HalvingGridSearchCV(single_threaded(pipeline), {"model__" + k: v for k, v in grid.items()}, **common)
        else:
            count = config["trials"]
            searcher = HalvingRandomSearchCV(single_threaded(pipeline), {"model__" + k: v for k, v in self._distributions(space).items()}, n_candidates=count, **common)
        levels = 1 + int(math.log(max(len(y) / minimum, 1), factor))
        self._budget(count * plan.n_splits * levels)
        with threadpool_limits(limits=1), parallel_backend("threading"):
            searcher.fit(X, y)
        self._check(cancelled)
        results = searcher.cv_results_
        records = []
        for i, params in enumerate(results["params"]):
            score = float(results["mean_test_score"][i]) * sign
            fold_scores = [float(results[f"split{fold}_test_score"][i]) * sign for fold in range(plan.n_splits)]
            valid = np.isfinite(score) and np.isfinite(fold_scores).all()
            records.append({"trial": i + 1, "params": {key.removeprefix("model__"): value for key, value in params.items()}, "score": score if valid else None, "std": float(results["std_test_score"][i]) if valid else None, "fold_scores": fold_scores if valid else [None if not np.isfinite(value) else value for value in fold_scores], "status": "complete" if valid else "invalid_metric", "iteration": int(results["iter"][i]), "resources": int(results["n_resources"][i]), "seconds": float(results["mean_fit_time"][i]) * plan.n_splits})
        return records
