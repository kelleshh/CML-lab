"""Real train/validation diagnostics; test rows are never passed to these adapters."""
from __future__ import annotations
from copy import deepcopy
import numpy as np
from joblib import Parallel, delayed, parallel_backend
from sklearn.base import BaseEstimator
from sklearn.inspection import permutation_importance
from threadpoolctl import threadpool_limits
from .fitting import fit_artifact

FLAGS = ("regularization_path", "learning_curve", "permutation_importance")
_PROBABILITY_METRICS = {"log_loss", "roc_auc", "average_precision", "brier"}
_LABELS = {"regularization_path": "Путь регуляризации", "learning_curve": "Кривая обучения",
           "permutation_importance": "Перестановочная важность"}


def _check(cancelled):
    if cancelled():
        raise InterruptedError("Дополнительная диагностика отменена.")


def _criterion(spec, metrics):
    selected = spec.get("metrics", "all")
    names = [selected] if isinstance(selected, str) else list(selected or [])
    metric = (spec.get("search") or {}).get("metric") or next((name for name in names if name != "all"), None)
    metric = metric or ("accuracy" if spec["task"] == "classification" else "rmse")
    definitions = {item["id"]: item for item in metrics.catalogue(spec["task"])}
    if metric not in definitions or definitions[metric].get("optimizable") is False:
        raise ValueError("Для диагностики выберите метрику качества выбранной задачи.")
    direction = definitions[metric]["direction"]
    if metric == "custom":
        explicit = (spec.get("search") or {}).get("direction")
        direction = explicit if explicit in {"min", "max"} else direction
    return metric, direction


def _score(artifact, X, y, spec, metrics, metric, cancelled):
    _check(cancelled)
    proba = artifact.predict_proba(X) if metric in _PROBABILITY_METRICS else None
    values, details = metrics.evaluate(spec["task"], y, artifact.predict_encoded(X), selection=[metric],
                                      probabilities=proba,
                                      n_classes=len(artifact.label_encoder.classes_) if artifact.label_encoder is not None else None,
                                      expression=spec.get("custom_metric"), metric_params=spec.get("metric_params"))
    _check(cancelled)
    if values.get(metric) is None:
        raise ValueError(details[metric]["reason"] or "Метрика не определена на этой части.")
    return float(values[metric])


class _FrozenPredictor(BaseEstimator):
    """Meet sklearn's estimator contract while prohibiting accidental fit."""
    def __init__(self, artifact):
        self.artifact = artifact

    def fit(self, X, y=None):
        raise RuntimeError("Перестановочная важность не обучает модель заново.")

    def predict(self, X):
        return self.artifact.predict_encoded(X)


def _frozen_copy(artifact):
    frozen = deepcopy(artifact)
    workers = {key: 1 for key in frozen.estimator.get_params(deep=True)
               if key == "n_jobs" or key.endswith("__n_jobs") or key == "thread_count" or key.endswith("__thread_count")}
    if workers:
        frozen.estimator.set_params(**workers)
    return frozen


def build_diagnostics(spec, artifact, X_train, y_train, X_validation, y_validation,
                      catalogue, metrics, *, groups=None, weights=None,
                      progress=lambda event: None, cancelled=lambda: False):
    """Return requested real diagnostic records plus explicit limitations."""
    output, warnings = {}, []
    enabled = [name for name in FLAGS if spec.get(name, False)]
    if not enabled:
        return output, warnings
    _check(cancelled)

    def skipped(name, reason):
        output[name] = {"reason": reason}
        warnings.append(f"{_LABELS[name]}: {reason}")

    dependent = groups is not None or spec["task"] not in {"regression", "classification"} or (spec.get("split") or {}).get("shuffle") is False
    if dependent:
        for name in enabled:
            skipped(name, "Доступна для регрессии и классификации с независимыми строками. Временные, групповые и упорядоченные данные требуют отдельного протокола; тест не используется.")
        return output, warnings
    try:
        metric, direction = _criterion(spec, metrics)
    except ValueError as error:
        for name in enabled:
            skipped(name, str(error))
        return output, warnings
    n_jobs = min(4, spec.get("n_jobs", 1))
    info = {"metric": metric, "direction": direction, "scope": {"fitted_on": "train", "scored_on": "validation", "test_used": False}, "n_jobs": n_jobs}
    fold_spec = {**spec, "n_jobs": 1, "search": None, **{name: False for name in FLAGS}}

    def fit_point(indices, params):
        _check(cancelled)
        fitted, _, _ = fit_artifact({**fold_spec, "params": params}, X_train.iloc[indices], np.asarray(y_train)[indices],
                                    catalogue, label_encoder=artifact.label_encoder,
                                    weights=None if weights is None else np.asarray(weights)[indices], cancelled=cancelled)
        train_score = _score(fitted, X_train.iloc[indices], np.asarray(y_train)[indices], spec, metrics, metric, cancelled)
        validation_score = _score(fitted, X_validation, y_validation, spec, metrics, metric, cancelled)
        coefficients = np.asarray(fitted.estimator.coef_).tolist() if hasattr(fitted.estimator, "coef_") else None
        return {"train_score": train_score, "validation_score": validation_score}, coefficients

    def calculate_points(name, entries, parameter):
        def calculate(entry):
            _check(cancelled)
            try:
                values, coefficients = fit_point(*parameter(entry))
                return {**entry, **values}, coefficients
            except (ValueError, TypeError, ArithmeticError, RuntimeError) as error:
                return {**entry, "train_score": None, "validation_score": None, "reason": str(error)}, None
        with threadpool_limits(limits=1):
            results = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(calculate)(entry) for entry in entries)
        _check(cancelled)
        progress({"type": "progress", "progress": .88, "message": f"{_LABELS[name]}: рассчитаны отдельные обучающие опыты"})
        return results

    fit_budget_reason = None
    if len(X_train) > 5000 or X_train.shape[1] > 200 or X_train.size > 500_000:
        fit_budget_reason = "Дополнительные обучения ограничены 5000 строками, 200 исходными признаками и 500000 ячеек train."
    model_params = artifact.estimator.get_params(deep=False)
    if max(model_params.get("n_estimators", 1), model_params.get("iterations", 1) or 1) > 500:
        fit_budget_reason = "Для дополнительных кривых используйте не больше 500 деревьев/итераций бустинга."

    if "regularization_path" in enabled:
        fields = {field["key"]: field for field in catalogue.descriptor(spec["algorithm_id"])["params"]}
        field = fields.get("alpha")
        if fit_budget_reason:
            skipped("regularization_path", fit_budget_reason)
        elif field is None or spec["algorithm_id"] == "gaussian_process_regressor":
            skipped("regularization_path", "Модель должна иметь параметр alpha, который задаёт штраф регуляризации. Дисперсия шума гауссовского процесса таким штрафом не является.")
        else:
            current = max(float(model_params.get("alpha", field["default"])), 1e-5)
            low, high = max(float(field.get("min", 0)), current * .01, 1e-8), min(float(field.get("max", 1e8)), current * 100)
            alphas = np.geomspace(low, high, 9).tolist()
            entries = [{"alpha": alpha} for alpha in alphas]
            results = calculate_points("regularization_path", entries,
                                       lambda entry: (np.arange(len(X_train)), {**(spec.get("params") or {}), "alpha": entry["alpha"]}))
            output["regularization_path"] = {**info, "alphas": alphas, "points": [point for point, _ in results],
                    "coefficients": [coef for _, coef in results], "feature_names": artifact.feature_names,
                    "note": "Каждая точка — новый полный конвейер с другим alpha, не шаг обучения. Коэффициенты относятся к подготовленным признакам; у нелинейных моделей коэффициентов нет."}

    if "learning_curve" in enabled:
        if fit_budget_reason:
            skipped("learning_curve", fit_budget_reason)
        else:
            order = np.random.default_rng(spec.get("seed", 42)).permutation(len(X_train))
            if spec["task"] == "classification":
                _, first = np.unique(np.asarray(y_train)[order], return_index=True)
                anchors = order[np.sort(first)]
                order = np.r_[anchors, order[~np.isin(order, anchors)]]
            minimum = min(len(X_train), max(10, len(np.unique(y_train)) if spec["task"] == "classification" else 2))
            sizes = np.unique(np.linspace(minimum, len(X_train), 5).astype(int))
            entries = [{"train_size": int(size)} for size in sizes]
            results = calculate_points("learning_curve", entries,
                                       lambda entry: (order[:entry["train_size"]], spec.get("params") or {}))
            output["learning_curve"] = {**info, "points": [point for point, _ in results],
                    "note": "Каждая точка обучает новый полный конвейер на вложенной части train. Все точки проверяются на одном validation; тест не используется."}

    if "permutation_importance" in enabled:
        if X_validation.shape[1] > 40 or len(X_validation) > 15000 or X_validation.size > 500_000:
            skipped("permutation_importance", "Перемешивание ограничено 40 признаками, 15000 строками и 500000 ячеек validation.")
        else:
            frozen = _FrozenPredictor(_frozen_copy(artifact))
            def scorer(estimator, X, y):
                value = _score(estimator.artifact, X, y, spec, metrics, metric, cancelled)
                return value if direction == "max" else -value
            try:
                with threadpool_limits(limits=1), parallel_backend("threading"):
                    importance = permutation_importance(frozen, X_validation, y_validation, scoring=scorer,
                                                        n_repeats=5, random_state=spec.get("seed", 42), n_jobs=n_jobs)
                _check(cancelled)
                output["permutation_importance"] = {**info, "scope": "validation", "test_used": False,
                        "feature_names": list(X_validation.columns), "mean": importance.importances_mean.tolist(),
                        "std": importance.importances_std.tolist(), "importances": importance.importances.tolist(),
                        "note": "Падение качества при перемешивании исходного признака в validation. Модель не обучается; положительное значение означает ухудшение. Связанные признаки могут заменять друг друга."}
            except (ValueError, TypeError, ArithmeticError, RuntimeError) as error:
                skipped("permutation_importance", str(error))
    return output, warnings
