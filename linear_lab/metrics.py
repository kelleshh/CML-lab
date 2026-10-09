"""Regression metrics with explicit domains and a small expression interpreter."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn import metrics as skmetrics


class MetricExpressionError(ValueError):
    """A custom metric is outside the supported numerical language."""


def evaluate_expression(expression: str, actual, predicted) -> float:
    """Interpret a bounded arithmetic AST; never execute user supplied Python."""
    if not isinstance(expression, str) or not expression.strip() or len(expression) > 1200:
        raise MetricExpressionError("Формула должна содержать от 1 до 1200 символов.")
    try:
        actual = np.asarray(actual, dtype=float).reshape(-1)
        predicted = np.asarray(predicted, dtype=float).reshape(-1)
    except (TypeError, ValueError, OverflowError) as exc:
        raise MetricExpressionError('Передай числовые массивы actual и predicted без объектов и текста.') from exc
    if actual.size != predicted.size or actual.size == 0:
        raise MetricExpressionError("Нужны непустые массивы actual и predicted одинаковой длины.")
    if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(predicted)):
        raise MetricExpressionError("Входные значения должны быть конечными числами.")
    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, RecursionError) as exc:
        raise MetricExpressionError("Не удалось разобрать числовую формулу.") from exc
    if sum(1 for _ in ast.walk(tree)) > 160:
        raise MetricExpressionError("Формула слишком сложная: максимум 160 элементов.")
    variables = {"y": actual, "pred": predicted, "error": actual - predicted}
    functions = {
        "abs": (np.abs, 1), "sqrt": (np.sqrt, 1), "log": (np.log, 1),
        "exp": (np.exp, 1), "mean": (np.mean, 1), "sum": (np.sum, 1),
        "min": (np.min, 1), "max": (np.max, 1), "clip": (np.clip, 3),
    }

    def visit(node, depth=0):
        if depth > 25:
            raise MetricExpressionError("Слишком много вложенных операций.")
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = float(node.value)
            if not np.isfinite(value) or abs(value) > 1e12:
                raise MetricExpressionError("Константа должна быть конечной и не превышать 10¹² по модулю.")
            return value
        if isinstance(node, ast.Name) and node.id in variables:
            return variables[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand, depth + 1)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp):
            left, right = visit(node.left, depth + 1), visit(node.right, depth + 1)
            if isinstance(node.op, ast.Pow):
                if np.ndim(right) or not np.isfinite(right) or abs(right) > 12:
                    raise MetricExpressionError("Степень должна быть числовой константой от −12 до 12.")
                return np.power(left, right)
            operators = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply, ast.Div: np.divide}
            function = operators.get(type(node.op))
            if function:
                return function(left, right)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            entry = functions.get(node.func.id)
            if entry and not node.keywords and len(node.args) == entry[1]:
                return entry[0](*(visit(arg, depth + 1) for arg in node.args))
        raise MetricExpressionError("Разрешены y, pred, error, числа, + − * / ** и abs/sqrt/log/exp/mean/sum/min/max/clip.")

    try:
        with np.errstate(all="raise"):
            result = visit(tree.body)
            if np.ndim(result) != 0 or not np.isfinite(result):
                raise MetricExpressionError("Метрика должна вернуть одно конечное число. Добавь mean(...) или sum(...).")
            return float(result)
    except MetricExpressionError:
        raise
    except (ArithmeticError, ValueError, TypeError, OverflowError) as exc:
        raise MetricExpressionError("Формула не определена на этих данных: проверь деление, логарифм и корень.") from exc


@dataclass(frozen=True)
class MetricSpec:
    id: str
    name: str
    description: str
    direction: str
    function: Callable
    domain: str = "finite"
    formula: str = ""


def _smape(y, p):
    denominator = np.abs(y) + np.abs(p)
    return float(200 * np.mean(np.divide(np.abs(y - p), denominator, out=np.zeros_like(y), where=denominator != 0)))


class MetricRegistry:
    """Own metric definitions and validity checks, independent of training."""

    def __init__(self):
        self._specs = [
            MetricSpec("mse", "MSE · средняя квадратичная ошибка", "Большие ошибки получают больший вес; единицы цели в квадрате.", "min", skmetrics.mean_squared_error, formula="mean((y-pred)^2)"),
            MetricSpec("rmse", "RMSE · корень из MSE", "Ошибка в единицах целевой величины.", "min", lambda y, p: np.sqrt(skmetrics.mean_squared_error(y, p)), formula="sqrt(mean((y-pred)^2))"),
            MetricSpec("mae", "MAE · средняя абсолютная ошибка", "Средняя величина промаха, в единицах цели.", "min", skmetrics.mean_absolute_error, formula="mean(abs(y-pred))"),
            MetricSpec("medae", "MedAE · медианная абсолютная ошибка", "Медиана промахов; менее чувствительна к отдельным выбросам.", "min", skmetrics.median_absolute_error),
            MetricSpec("r2", "R² · коэффициент детерминации", "1 — точное предсказание; 0 — уровень постоянного среднего; отрицательное значение возможно. Для постоянной цели не определен.", "max", lambda y, p: skmetrics.r2_score(y, p, force_finite=False), "nonconstant"),
            MetricSpec("explained_variance", "Объясненная дисперсия", "Оценивает разброс ошибок, но не штрафует постоянное смещение так же, как R².", "max", lambda y, p: skmetrics.explained_variance_score(y, p, force_finite=False), "nonconstant"),
            MetricSpec("max_error", "Максимальная абсолютная ошибка", "Самый большой промах в этой выборке.", "min", skmetrics.max_error),
            MetricSpec("mape", "MAPE · средняя относительная ошибка", "Относительная ошибка в долях; 0.1 означает 10%. Не определена при нулевой цели.", "min", skmetrics.mean_absolute_percentage_error, "nonzero"),
            MetricSpec("smape", "SMAPE · симметричная относительная ошибка", "Проценты от 0 до 200; пара нулей дает нулевой вклад.", "min", _smape),
            MetricSpec("msle", "MSLE · квадратичная ошибка логарифмов", "Сравнивает log(1+y) и log(1+pred); нужны неотрицательные числа.", "min", skmetrics.mean_squared_log_error, "nonnegative"),
            MetricSpec("rmsle", "RMSLE · корень из MSLE", "Корень из средней квадратичной ошибки логарифмов.", "min", lambda y, p: np.sqrt(skmetrics.mean_squared_log_error(y, p)), "nonnegative"),
            MetricSpec("pinball", "Квантильная ошибка", "Асимметричный штраф: quantile=0.5 оценивает медиану.", "min", lambda y, p: skmetrics.mean_pinball_loss(y, p, alpha=0.5)),
            MetricSpec("poisson_deviance", "Девианс Пуассона", "Для неотрицательной цели и строго положительных предсказаний.", "min", skmetrics.mean_poisson_deviance, "poisson"),
            MetricSpec("gamma_deviance", "Гамма-девианс", "Для строго положительной цели и строго положительных предсказаний.", "min", skmetrics.mean_gamma_deviance, "positive"),
            MetricSpec("tweedie_deviance", "Девианс Твиди", "Допустимые значения зависят от параметра power; по умолчанию power=1.5.", "min", lambda y, p: skmetrics.mean_tweedie_deviance(y, p, power=1.5), "poisson"),
        ]
        self._index = {spec.id: spec for spec in self._specs}

    def catalogue(self):
        result = [{"id": s.id, "name": s.name, "description": s.description, "direction": s.direction, "domain": s.domain, "formula": s.formula} for s in self._specs]
        result.append({"id": "custom", "name": "Своя числовая формула", "description": "Например mean(abs(error)) или sqrt(mean(error**2)).", "direction": "min", "formula": "mean(abs(error))"})
        return result

    @staticmethod
    def _reason(domain, y, p):
        if y.size == 0:
            return "В этой части нет наблюдений."
        if not np.all(np.isfinite(y)) or not np.all(np.isfinite(p)):
            return "В данных или предсказаниях есть бесконечность либо пропуск."
        if domain == "nonconstant" and (y.size < 2 or np.ptp(y) == 0):
            return "Нужны хотя бы два наблюдения и непостоянная целевая переменная."
        if domain == "nonzero" and np.any(y == 0):
            return "MAPE не определена: целевая переменная содержит ноль."
        if domain == "nonnegative" and (np.any(y < 0) or np.any(p < 0)):
            return "Нужны неотрицательные цель и предсказания."
        if domain == "poisson" and (np.any(y < 0) or np.any(p <= 0)):
            return "Нужна неотрицательная цель и строго положительные предсказания."
        if domain == "positive" and (np.any(y <= 0) or np.any(p <= 0)):
            return "Цель и предсказания должны быть строго положительными."
        return None

    def evaluate(self, actual, predicted, selection=None, custom_expression=None, params=None):
        y, p = np.asarray(actual, dtype=float).reshape(-1), np.asarray(predicted, dtype=float).reshape(-1)
        if y.shape != p.shape:
            raise ValueError("Цель и предсказания должны иметь одинаковую длину.")
        if isinstance(selection, str) and selection != "all":
            selection = [selection]
        selected = [s.id for s in self._specs] if selection is None or selection == "all" or "all" in selection else list(selection)
        if custom_expression and "custom" not in selected:
            selected.append("custom")
        values, details = {}, {}
        params = params or {}
        for name in dict.fromkeys(selected):
            spec = self._index.get(name)
            if name == "custom":
                reason = None if custom_expression else "Не указана формула собственной метрики."
                value = None
                if not reason:
                    try:
                        value = evaluate_expression(custom_expression, y, p)
                    except MetricExpressionError as exc:
                        reason = str(exc)
            elif spec:
                domain = spec.domain
                parameter_reason = None
                if name == "tweedie_deviance":
                    try:
                        power = float(params.get("power", 1.5))
                        if not np.isfinite(power):
                            raise ValueError
                        domain = "finite" if power <= 0 else ("positive" if power >= 2 else "poisson")
                    except (ValueError, TypeError):
                        parameter_reason = "Параметр power должен быть конечным числом."
                reason = parameter_reason or self._reason(domain, y, p)
                value = None
                if not reason:
                    try:
                        with np.errstate(all="raise"):
                            if name == "pinball":
                                quantile = float(params.get("quantile", 0.5))
                                value = float(skmetrics.mean_pinball_loss(y, p, alpha=quantile))
                            elif name == "tweedie_deviance":
                                power = float(params.get("power", 1.5))
                                if 0 < power < 1:
                                    raise ValueError("power должен быть ≤0 или ≥1.")
                                if power == 0:
                                    value = float(skmetrics.mean_tweedie_deviance(y, p, power=power))
                                elif power < 0 and np.any(p <= 0):
                                    raise ValueError("При power<0 нужны строго положительные предсказания.")
                                elif power >= 2 and np.any(y <= 0):
                                    raise ValueError("При power≥2 нужны строго положительные значения цели.")
                                else:
                                    value = float(skmetrics.mean_tweedie_deviance(y, p, power=power))
                            else:
                                value = float(spec.function(y, p))
                        if not np.isfinite(value):
                            raise ValueError("Результат не является конечным числом.")
                    except (ValueError, ArithmeticError) as exc:
                        value, reason = None, str(exc)
            else:
                value, reason = None, f"Неизвестная метрика: {name}."
            values[name] = value
            details[name] = {"value": value, "reason": reason, "direction": spec.direction if spec else "min"}
        return values, details


def metrics_catalogue():
    return MetricRegistry().catalogue()


def evaluate_metrics(actual, predicted, selection=None, custom_expression=None, params=None):
    return MetricRegistry().evaluate(actual, predicted, selection, custom_expression, params)


metric_catalogue = metrics_catalogue
custom_metric = evaluate_expression
