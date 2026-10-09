"""Validated catalogue and library-backed linear regression estimators."""

from __future__ import annotations

from copy import deepcopy
from itertools import combinations
from numbers import Integral, Real
from typing import Any, Callable

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import (
    ARDRegression, BayesianRidge, ElasticNet, GammaRegressor, HuberRegressor,
    Lars, Lasso, LassoLars, LinearRegression, OrthogonalMatchingPursuit,
    PoissonRegressor, QuantileRegressor, RANSACRegressor, Ridge, SGDRegressor,
    TheilSenRegressor, TweedieRegressor,
)
from sklearn.metrics import mean_pinball_loss, mean_tweedie_deviance
from sklearn.svm import LinearSVR
from sklearn.utils.validation import check_is_fitted, validate_data


class _LibraryRegressor(RegressorMixin, BaseEstimator):
    """Keep the public sklearn coefficient and prediction contract for adapters."""

    def _finish_fit(self, model: Any, X: np.ndarray, y: np.ndarray):
        self.model_ = model.fit(X, y)
        self.coef_ = np.asarray(self.model_.coef_, dtype=float).reshape(-1)
        self.intercept_ = float(np.asarray(self.model_.intercept_).reshape(-1)[0])
        if hasattr(self.model_, "n_iter_"):
            self.n_iter_ = self.model_.n_iter_
        return self

    def predict(self, X):
        check_is_fitted(self, "coef_")
        X = validate_data(self, X, reset=False, accept_sparse=False)
        return X @ self.coef_ + self.intercept_


class WeightedLassoRegressor(_LibraryRegressor):
    """Feature-order weights, solved by skglm's weighted Lasso implementation."""

    def __init__(self, alpha=0.1, profile="increasing", weight_ratio=3.0,
                 fit_intercept=True, max_iter=100, tol=1e-4):
        self.alpha = alpha
        self.profile = profile
        self.weight_ratio = weight_ratio
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        from skglm import WeightedLasso

        X, y = validate_data(self, X, y, y_numeric=True, accept_sparse=False)
        self.weights_ = np.linspace(1, self.weight_ratio, X.shape[1])
        if self.profile == "decreasing":
            self.weights_ = self.weights_[::-1].copy()
        elif self.profile == "uniform":
            self.weights_ = np.ones(X.shape[1])
        return self._finish_fit(WeightedLasso(
            alpha=self.alpha, weights=self.weights_, fit_intercept=self.fit_intercept,
            max_iter=self.max_iter, tol=self.tol,
        ), X, y)


class AdaptiveLassoRegressor(_LibraryRegressor):
    """Train-only Ridge pilot followed by one adaptive weighted Lasso fit."""

    def __init__(self, alpha=0.1, gamma=1.0, pilot_alpha=1.0, epsilon=0.001,
                 fit_intercept=True, max_iter=100, tol=1e-4):
        self.alpha = alpha
        self.gamma = gamma
        self.pilot_alpha = pilot_alpha
        self.epsilon = epsilon
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        from skglm import WeightedLasso

        X, y = validate_data(self, X, y, y_numeric=True, accept_sparse=False)
        pilot = Ridge(alpha=self.pilot_alpha, fit_intercept=self.fit_intercept).fit(X, y)
        self.pilot_coef_ = pilot.coef_.copy()
        self.weights_ = (np.abs(self.pilot_coef_) + self.epsilon) ** (-self.gamma)
        if not np.isfinite(self.weights_).all():
            raise ValueError("Веса Adaptive Lasso переполнились: уменьшите gamma или увеличьте epsilon.")
        return self._finish_fit(WeightedLasso(
            alpha=self.alpha, weights=self.weights_, fit_intercept=self.fit_intercept,
            max_iter=self.max_iter, tol=self.tol,
        ), X, y)


class GroupLassoRegressor(_LibraryRegressor):
    """Contiguous feature groups; the last group may be smaller."""

    def __init__(self, alpha=0.1, group_size=2, fit_intercept=True,
                 max_iter=100, tol=1e-4):
        self.alpha = alpha
        self.group_size = group_size
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol

    def _groups(self, n_features):
        self.groups_ = [list(range(i, min(i + self.group_size, n_features)))
                        for i in range(0, n_features, self.group_size)]
        self.group_weights_ = np.sqrt([len(group) for group in self.groups_])
        return self.groups_

    def fit(self, X, y):
        from skglm import GroupLasso

        X, y = validate_data(self, X, y, y_numeric=True, accept_sparse=False)
        groups = self._groups(X.shape[1])
        return self._finish_fit(GroupLasso(
            groups=groups, alpha=self.alpha, weights=self.group_weights_,
            fit_intercept=self.fit_intercept, max_iter=self.max_iter, tol=self.tol,
        ), X, y)


class SparseGroupLassoRegressor(GroupLassoRegressor):
    def __init__(self, alpha=0.1, l1_ratio=0.5, group_size=2,
                 fit_intercept=True, max_iter=100, tol=1e-4):
        super().__init__(alpha, group_size, fit_intercept, max_iter, tol)
        self.l1_ratio = l1_ratio

    def fit(self, X, y):
        from skglm import GeneralizedLinearEstimator
        from skglm.datafits import QuadraticGroup
        from skglm.penalties import WeightedL1GroupL2
        from skglm.solvers import GroupBCD
        from skglm.utils.data import grp_converter

        X, y = validate_data(self, X, y, y_numeric=True, accept_sparse=False)
        groups = self._groups(X.shape[1])
        indices, pointers = grp_converter(groups, X.shape[1])
        datafit = QuadraticGroup(pointers, indices)
        penalty = WeightedL1GroupL2(
            self.alpha, (1 - self.l1_ratio) * self.group_weights_,
            np.full(X.shape[1], self.l1_ratio), pointers, indices,
        )
        # WeightedL1GroupL2 weights cover p features. An explicit extra intercept
        # would change that vector length; centering eliminates the unpenalized b.
        self.X_offset_ = X.mean(axis=0) if self.fit_intercept else np.zeros(X.shape[1])
        self.y_offset_ = float(y.mean()) if self.fit_intercept else 0.0
        solver = GroupBCD(max_iter=self.max_iter, tol=self.tol,
                          fit_intercept=False, ws_strategy="fixpoint")
        self._finish_fit(GeneralizedLinearEstimator(
            datafit=datafit, penalty=penalty, solver=solver,
        ), X - self.X_offset_, y - self.y_offset_)
        self.intercept_ = float(self.y_offset_ - self.X_offset_ @ self.coef_)
        return self


class SCADRegressor(_LibraryRegressor):
    def __init__(self, alpha=0.1, gamma=3.7, fit_intercept=True,
                 max_iter=100, tol=1e-4):
        self.alpha = alpha
        self.gamma = gamma
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        from skglm import GeneralizedLinearEstimator
        from skglm.datafits import Quadratic
        from skglm.penalties import SCAD
        from skglm.solvers import AndersonCD

        X, y = validate_data(self, X, y, y_numeric=True, accept_sparse=False)
        return self._finish_fit(GeneralizedLinearEstimator(
            datafit=Quadratic(), penalty=SCAD(self.alpha, self.gamma),
            solver=AndersonCD(max_iter=self.max_iter, tol=self.tol,
                              fit_intercept=self.fit_intercept),
        ), X, y)


class _ConvexRegressor(_LibraryRegressor):
    def _solve(self, X, y, penalty: Callable):
        import cvxpy as cp

        weights = cp.Variable(X.shape[1])
        intercept = cp.Variable() if self.fit_intercept else 0.0
        objective = cp.sum_squares(y - X @ weights - intercept) / (2 * len(y))
        problem = cp.Problem(cp.Minimize(objective + penalty(weights)))
        problem.solve(solver=cp.CLARABEL, max_iter=self.max_iter,
                      tol_gap_abs=self.tol, tol_gap_rel=self.tol,
                      tol_feas=self.tol)
        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise ValueError(f"Решатель CVXPY не получил решение: {problem.status}.")
        self.coef_ = np.asarray(weights.value, dtype=float).reshape(-1)
        self.intercept_ = float(intercept.value) if self.fit_intercept else 0.0
        self.solver_status_ = problem.status
        self.n_iter_ = problem.solver_stats.num_iters
        self.objective_ = float(problem.value)
        return self


class FusedLassoRegressor(_ConvexRegressor):
    def __init__(self, alpha=0.1, fusion_strength=0.1, fit_intercept=True,
                 max_iter=200, tol=1e-6):
        self.alpha = alpha
        self.fusion_strength = fusion_strength
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        import cvxpy as cp

        X, y = validate_data(self, X, y, y_numeric=True, accept_sparse=False)
        self.difference_ = np.diff(np.eye(X.shape[1]), axis=0)

        def penalty(weights):
            fused = cp.norm1(self.difference_ @ weights) if len(self.difference_) else 0
            return self.alpha * cp.norm1(weights) + self.fusion_strength * fused

        return self._solve(X, y, penalty)


class TikhonovRegressor(_ConvexRegressor):
    def __init__(self, alpha=1.0, operator="first_difference", fit_intercept=True,
                 max_iter=200, tol=1e-6):
        self.alpha = alpha
        self.operator = operator
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        import cvxpy as cp

        X, y = validate_data(self, X, y, y_numeric=True, accept_sparse=False)
        orders = {"identity": 0, "first_difference": 1, "second_difference": 2}
        self.operator_ = np.diff(np.eye(X.shape[1]), n=orders[self.operator], axis=0)

        def penalty(weights):
            if not len(self.operator_):
                return 0
            return self.alpha * cp.sum_squares(self.operator_ @ weights) / 2

        return self._solve(X, y, penalty)


class BestSubsetRegressor(_LibraryRegressor):
    """Enumerate all supports up to a user cap; exact only in this small domain."""

    def __init__(self, alpha=0.05, max_features=3, fit_intercept=True):
        self.alpha = alpha
        self.max_features = max_features
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        X, y = validate_data(self, X, y, y_numeric=True, accept_sparse=False)
        n, p = X.shape
        if p > 12:
            raise ValueError("Точный L0 допускает максимум 12 признаков после преобразований; используйте OMP для больших задач.")
        x_mean = X.mean(axis=0) if self.fit_intercept else np.zeros(p)
        y_mean = float(y.mean()) if self.fit_intercept else 0.0
        centered_X, centered_y = X - x_mean, y - y_mean
        self.coef_ = np.zeros(p)
        best_objective = float(centered_y @ centered_y / (2 * n))
        self.subsets_evaluated_ = 1
        for size in range(1, min(p, self.max_features) + 1):
            for support in combinations(range(p), size):
                matrix = centered_X[:, support]
                beta = np.linalg.lstsq(matrix, centered_y, rcond=None)[0]
                residual = centered_y - matrix @ beta
                objective = float(residual @ residual / (2 * n) + self.alpha * size)
                self.subsets_evaluated_ += 1
                if objective < best_objective:
                    best_objective = objective
                    self.coef_.fill(0)
                    self.coef_[list(support)] = beta
        self.intercept_ = float(y_mean - x_mean @ self.coef_)
        self.objective_ = best_objective
        self.support_ = np.flatnonzero(self.coef_)
        return self


def _param(key, label, kind, default, minimum=None, maximum=None, step=None,
           options=None, help_text=""):
    item = {"key": key, "label": label, "type": kind, "default": default,
            "help": help_text}
    if minimum is not None:
        item["min"] = minimum
    if maximum is not None:
        item["max"] = maximum
    if step is not None:
        item["step"] = step
    if options is not None:
        item["options"] = options
        labels = {
            "auto": "Автоматически", "none": "Без штрафа", "l1": "L1 · сумма модулей",
            "l2": "L2 · сумма квадратов", "elasticnet": "Смесь L1 и L2",
            "squared_error": "Квадратичная ошибка", "huber": "Ошибка Хьюбера",
            "epsilon_insensitive": "Линейная ошибка вне полосы ε",
            "squared_epsilon_insensitive": "Квадратичная ошибка вне полосы ε",
            "constant": "Постоянный шаг", "invscaling": "Плавное уменьшение шага",
            "optimal": "Расчетный убывающий шаг", "adaptive": "Уменьшение при остановке улучшения",
            "uniform": "Одинаковые веса", "increasing": "Возрастающие веса",
            "decreasing": "Убывающие веса", "identity": "Единичный оператор · Ridge",
            "first_difference": "Первая разность коэффициентов",
            "second_difference": "Вторая разность коэффициентов",
            "svd": "SVD · сингулярное разложение", "cholesky": "Разложение Холецкого",
            "lsqr": "LSQR · итеративный МНК", "sag": "SAG · усреднение градиентов",
            "saga": "SAGA · градиент с поправкой",
        }
        item["option_labels"] = {value: labels.get(value, value) for value in options}
    return item


def _alpha(default=0.1):
    return _param("alpha", "Сила штрафа α", "float", default, 0, 1e6, 0.01,
                  help_text="Чем больше значение, тем дороже большие коэффициенты. Точная нормировка указана в формуле модели.")


INTERCEPT = _param("fit_intercept", "Свободный член", "bool", True,
                   help_text="Разрешить модели прибавлять постоянное число b к предсказанию.")
ITERATIONS = _param("max_iter", "Предел итераций", "int", 1000, 1, 100000, 1,
                    help_text="Предел работы решателя. У разных моделей итерация означает разное действие.")
TOLERANCE = _param("tol", "Точность остановки", "float", 0.0001, 1e-12, 1, 0.0001,
                  help_text="Меньше значение означает более точное и обычно более долгое решение.")
RATIO = _param("l1_ratio", "Доля L1", "float", 0.5, 0, 1, 0.01,
               help_text="0: только второй штраф; 1: только L1; промежуточные значения смешивают штрафы.")
GROUP_SIZE = _param("group_size", "Размер группы признаков", "int", 2, 1, 1000, 1,
                    help_text="Группы идут по порядку преобразованных столбцов. Последняя группа может быть меньше.")


class ModelRegistry:
    """One authoritative parameter schema for UI validation and model construction."""

    def __init__(self):
        self._specs: dict[str, dict] = {}
        self._factories: dict[str, Callable] = {}
        self._register_models()

    def _add(self, model_id, name, family, description, formula, explanation,
             factory, params=(), trace="final", source=None):
        self._specs[model_id] = {
            "id": model_id, "name": name, "family": family,
            "description": description, "formula": formula,
            "explanation": explanation, "trace": trace,
            "params": deepcopy(list(params)),
            "source": source or "https://scikit-learn.org/stable/modules/linear_model.html",
        }
        self._factories[model_id] = factory

    def _register_models(self):
        ordinary = "Штрафы и отбор"
        robust = "Устойчивые к выбросам"
        exotic = "Структурные и экзотические штрафы"
        native = lambda cls: (lambda p, seed: cls(**p))
        seeded = lambda cls: (lambda p, seed: cls(random_state=seed, **p))
        add = self._add
        add("ols", "Обычный МНК", ordinary, "Минимизирует сумму квадратов ошибок.",
            r"\min_{w,b}\;\|y-Xw-b\|_2^2",
            "У каждой строки есть ошибка y−ŷ. МНК возводит ошибки в квадрат и складывает. Свободный член b не штрафуется.",
            native(LinearRegression), [INTERCEPT],
            source="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html")
        add("ridge", "Ridge · L2", ordinary, "Сжимает коэффициенты квадратичным штрафом.",
            r"\min_{w,b}\;\|y-Xw-b\|_2^2+\alpha\|w\|_2^2",
            "Коэффициенты обычно уменьшаются плавно. Нормировка sklearn Ridge: сумма квадратов без деления на n. При общей цели RSS/(2n)+λ‖w‖²/2 нужно alpha=n·λ.",
            seeded(Ridge), [_alpha(1), INTERCEPT, ITERATIONS, TOLERANCE,
                           _param("solver", "Решатель", "select", "auto", options=["auto", "svd", "cholesky", "lsqr", "sag", "saga"],
                                  help_text="auto выбирает подходящий библиотечный решатель; история его внутренних шагов не доступна.")],
            source="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html")
        add("lasso", "Lasso · L1", ordinary, "Может сделать коэффициенты точно нулевыми.",
            r"\min_{w,b}\;\frac{\|y-Xw-b\|_2^2}{2n}+\alpha\|w\|_1",
            "Штраф складывает модули коэффициентов. Нулевой коэффициент исключает признак. Путь по alpha состоит из решений разных задач. При alpha=0 используйте МНК.",
            seeded(Lasso), [_alpha(), INTERCEPT, ITERATIONS, TOLERANCE,
                           _param("positive", "Только неотрицательные коэффициенты", "bool", False,
                                  help_text="Вводит дополнительное ограничение w≥0.")],
            source="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html")
        add("elasticnet", "ElasticNet · L1 + L2", ordinary, "Совмещает разреженность и сжатие.",
            r"\min_{w,b}\;\frac{\|y-Xw-b\|_2^2}{2n}+\alpha r\|w\|_1+\frac{\alpha(1-r)}2\|w\|_2^2",
            "r — доля L1. Смешанный штраф полезен при похожих признаках. r близко к нулю может затруднять координатный решатель; чистый L2 удобнее запускать через Ridge с учетом нормировки.",
            seeded(ElasticNet), [_alpha(), RATIO, INTERCEPT, ITERATIONS, TOLERANCE],
            source="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html")
        add("lars", "LARS · наименьшие углы", ordinary, "Поэтапно включает признаки по связи с остатками.",
            r"r=y-Xw-b;\quad\text{движение по равным углам к активным признакам}",
            "Это алгоритм построения пути, а не название штрафа. coef_path_ содержит настоящие узлы алгоритма. Обычный LARS и LassoLars могут давать разные пути.",
            seeded(Lars), [INTERCEPT,
                          _param("n_nonzero_coefs", "Предел активных признаков", "int", 5, 1, 10000, 1,
                                 help_text="Предел включения признаков в активное множество.")], trace="active_set",
            source="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lars.html")
        add("lassolars", "LassoLars", ordinary, "Решает задачу Lasso алгоритмом LARS.",
            r"\min_{w,b}\;\frac{\|y-Xw-b\|_2^2}{2n}+\alpha\|w\|_1",
            "Штраф тот же, что у Lasso, но вычисляется кусочно-линейный путь с изменением активного множества. Узлы пути не являются эпохами SGD.",
            seeded(LassoLars), [_alpha(), INTERCEPT, ITERATIONS], trace="active_set",
            source="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLars.html")
        add("omp", "OMP · ортогональный отбор", ordinary, "Жадно выбирает ограниченное число признаков.",
            r"\min_w\;\|y-Xw-b\|_2^2\quad\text{при}\quad\|w\|_0\leq k\quad\text{(жадное приближение)}",
            "Добавляет признак, связанный с остатками, и заново решает МНК на выбранных столбцах. Это приближение выбора подмножества; глобальный оптимум L0 не гарантирован. return_path у orthogonal_mp дает настоящий путь отбора.",
            native(OrthogonalMatchingPursuit), [INTERCEPT,
              _param("n_nonzero_coefs", "Число выбранных признаков", "int", 1, 1, 10000, 1,
                     help_text="Должно быть не больше числа преобразованных признаков.")], trace="active_set",
            source="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.orthogonal_mp.html")
        def sgd_factory(params, seed):
            params = dict(params)
            if params["penalty"] == "none":
                params["penalty"] = None
            if params["learning_rate"] == "optimal" and params["alpha"] == 0:
                raise ValueError("Для learning_rate='optimal' нужен alpha>0.")
            return SGDRegressor(random_state=seed, **params)
        add("sgd", "SGDRegressor · обучение по строкам", "Итеративное обучение", "Позволяет видеть реальные эпохи обучения.",
            r"\min_{w,b}\;\frac1n\sum_i\ell(y_i,x_i^Tw+b)+\alpha R(w)",
            "partial_fit выполняет одну эпоху и сохраняет счетчик обновлений. Скорость eta0 определяет размер шагов. Для квадратичной ошибки ℓ=e²/2; L2 штраф равен ‖w‖²/2. Данные желательно масштабировать.",
            sgd_factory, [_alpha(0.0001), INTERCEPT, ITERATIONS, TOLERANCE, RATIO,
              _param("penalty", "Штраф", "select", "l2", options=["none", "l2", "l1", "elasticnet"], help_text="SGD — способ поиска решения; здесь отдельно выбирается штраф."),
              _param("loss", "Ошибка для обучения", "select", "squared_error", options=["squared_error", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"], help_text="Определяет реакцию на ошибки и выбросы."),
              _param("eta0", "Начальный шаг обучения", "float", 0.01, 1e-8, 10, 0.001, help_text="Слишком большой шаг способен вызвать расходимость."),
              _param("learning_rate", "Изменение шага", "select", "invscaling", options=["constant", "invscaling", "optimal", "adaptive"], help_text="Правило изменения шага. optimal требует alpha>0."),
              _param("epsilon", "Порог ошибки ε", "float", 0.1, 0, 1e6, 0.01, help_text="Для Huber и ошибок с зоной нечувствительности.")], trace="epochs",
            source="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html")
        def ransac_factory(params, seed):
            params = dict(params)
            for key in ("residual_threshold", "min_samples"):
                if params[key] == 0:
                    params[key] = None
            return RANSACRegressor(random_state=seed, **params)
        add("ransac", "RANSAC · согласованное подмножество", robust, "Ищет группу строк, согласующихся с одной моделью.",
            r"\max |I|,\quad I=\{i:|y_i-\hat y_i|\leq t\};\quad\text{МНК на выбранном }I",
            "Обучает кандидатов на случайных подмножествах, выбирает согласованный набор и переобучает модель на нем. Нет единой убывающей MSE-траектории. Показываем итоговую маску строк и число испытаний.",
            ransac_factory, [
              _param("max_trials", "Предел случайных испытаний", "int", 200, 1, 10000, 1, help_text="Больше испытаний повышает шанс найти подходящее подмножество."),
              _param("residual_threshold", "Порог принятия строки", "float", 0, 0, 1e6, 0.1, help_text="0: автоматически MAD целевой переменной; иначе максимальная ошибка строки."),
              _param("min_samples", "Размер случайного подмножества", "int", 0, 0, 100000, 1, help_text="0: автоматически число признаков + 1.")],
            source="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RANSACRegressor.html")
        add("theilsen", "Theil–Sen · медиана оценок", robust, "Объединяет МНК по подмножествам пространственной медианой.",
            r"\hat w=\operatorname{spatial\ median}\{\hat w_{\mathrm{OLS},S}\}",
            "Строит много оценок на подмножествах строк. Пространственная медиана дает общий итог. max_iter относится к расчету медианы, а не к проходам по датасету. Стоимость растет с числом признаков.",
            seeded(TheilSenRegressor), [INTERCEPT, ITERATIONS, TOLERANCE,
              _param("max_subpopulation", "Предел подмножеств", "int", 1000, 1, 100000, 100, help_text="Ограничивает число МНК-кандидатов; большие значения требуют больше времени и памяти.")],
            source="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TheilSenRegressor.html")
        add("huber", "HuberRegressor", robust, "Мягко ограничивает влияние больших остатков.",
            r"\min_{w,b,\sigma>0}\;n\sigma+\sum_i\sigma h_\epsilon(e_i/\sigma)+\alpha\|w\|_2^2",
            "h(u)=u² при |u|≤ε, иначе 2ε|u|−ε². Модель одновременно оценивает масштаб σ. Это отличается от SGD с Huber, где порог задан прямо в единицах ошибки. epsilon≥1.",
            native(HuberRegressor), [_alpha(0.0001), INTERCEPT, ITERATIONS, TOLERANCE,
              _param("epsilon", "Порог выброса ε", "float", 1.35, 1, 100, 0.05, help_text="Порог задается после деления остатка на оцененный масштаб σ.")],
            source="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.HuberRegressor.html")
        add("bayesian_ridge", "BayesianRidge", "Байесовские", "Оценивает коэффициенты и их неопределенность.",
            r"y\mid w\sim\mathcal N(Xw+b,\alpha^{-1}I),\quad w\sim\mathcal N(0,\lambda^{-1}I)",
            "Шум и сила сжатия оцениваются по данным. compute_score сохраняет логарифм маргинального правдоподобия, который максимизируется. alpha здесь означает точность шума, а не ручную силу штрафа Ridge.",
            lambda p, seed: BayesianRidge(compute_score=True, **p), [INTERCEPT, ITERATIONS, TOLERANCE],
            source="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html")
        add("ard", "ARDRegression", "Байесовские", "Каждый признак получает собственную силу сжатия.",
            r"w_j\sim\mathcal N(0,\lambda_j^{-1});\quad\max\log p(y\mid X)",
            "Высокая точность λ_j означает маленькую дисперсию коэффициента. Сильно подавленные признаки удаляются по threshold_lambda. scores_ — целевая функция для максимизации, не MSE.",
            lambda p, seed: ARDRegression(compute_score=True, **p), [INTERCEPT, ITERATIONS, TOLERANCE,
              _param("threshold_lambda", "Порог удаления признака", "float", 10000, 1, 1e10, 100, help_text="Удалять признаки с оцененной точностью λ_j выше порога.")],
            source="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ARDRegression.html")
        add("quantile", "QuantileRegressor", robust, "Предсказывает выбранный условный квантиль.",
            r"\min_{w,b}\;\frac1n\sum_i\max(qe_i,(q-1)e_i)+\alpha\|w\|_1",
            "q=0.5 предсказывает медиану. q=0.9 описывает верхнюю границу для большинства наблюдений при подходящей модели. Обучается через линейное программирование HiGHS; промежуточные коэффициенты не доступны.",
            native(QuantileRegressor), [_alpha(0.01), INTERCEPT,
              _param("quantile", "Квантиль q", "float", 0.5, 0.01, 0.99, 0.01, help_text="Строго между 0 и 1. Квантиль выбирает асимметрию штрафа ошибки.")],
            source="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.QuantileRegressor.html")
        glm_params = [_alpha(1), INTERCEPT, ITERATIONS, TOLERANCE]
        add("poisson", "PoissonRegressor", "Обобщенные линейные", "Модель неотрицательных величин со связью exp.",
            r"\hat y=\exp(Xw+b),\quad\min\frac{D_{\mathrm{Poisson}}(y,\hat y)}{2n}+\frac\alpha2\|w\|_2^2",
            "Все y должны быть ≥0; целые значения не обязательны. Предсказания положительные. По коэффициентам модель линейна до применения экспоненты, поэтому график предсказаний обычно не является плоскостью.",
            native(PoissonRegressor), glm_params,
            source="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PoissonRegressor.html")
        add("gamma", "GammaRegressor", "Обобщенные линейные", "Модель строго положительных величин со связью exp.",
            r"\hat y=\exp(Xw+b),\quad\min\frac{D_{\mathrm{Gamma}}(y,\hat y)}{2n}+\frac\alpha2\|w\|_2^2",
            "Все y должны быть >0. Нули и отрицательные значения недопустимы. Дисперсия Gamma растет с квадратом среднего; связь по умолчанию логарифмическая.",
            native(GammaRegressor), glm_params,
            source="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.GammaRegressor.html")
        def tweedie_factory(params, seed):
            if 0 < params["power"] < 1:
                raise ValueError("Tweedie не определен при 0<power<1; выберите 0 либо power≥1.")
            return TweedieRegressor(**params)
        add("tweedie", "TweedieRegressor", "Обобщенные линейные", "Выбирает семейство ошибки через степень дисперсии.",
            r"\operatorname{Var}(y)\propto\mu^p,\quad\min\frac{D_p(y,\hat y)}{2n}+\frac\alpha2\|w\|_2^2",
            "p=0: нормальная ошибка; p=1: Пуассон, y≥0; 1<p<2: y≥0; p≥2: y>0. Значения 0<p<1 недопустимы. link=auto использует тождественную связь при p≤0 и log при p>0.",
            tweedie_factory, glm_params + [_param("power", "Степень дисперсии p", "float", 1.5, 0, 3, 0.1, help_text="Разрешены 0 и значения ≥1; ограничения y зависят от p.")],
            source="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TweedieRegressor.html")
        add("linear_svr", "LinearSVR", "Другие линейные", "Игнорирует ошибки внутри полосы ε.",
            r"\min_{w,b}\;\frac{\|w\|_2^2+b^2}{2}+C\sum_i\max(|e_i|-\epsilon,0)",
            "LIBLINEAR штрафует также искусственный коэффициент свободного члена (intercept_scaling=1). C задает цену ошибок: увеличение C ослабляет относительное влияние регуляризации. Для squared_epsilon_insensitive ошибка в формуле возводится в квадрат.",
            seeded(LinearSVR), [INTERCEPT, ITERATIONS, TOLERANCE,
              _param("C", "Цена ошибок C", "float", 1, 1e-8, 1e6, 0.1, help_text="C>0. Больше C — дороже ошибки относительно величины коэффициентов."),
              _param("epsilon", "Ширина зоны без штрафа ε", "float", 0.1, 0, 1e6, 0.01, help_text="Ошибки до ε не штрафуются."),
              _param("loss", "Ошибка вне полосы", "select", "epsilon_insensitive", options=["epsilon_insensitive", "squared_epsilon_insensitive"], help_text="Линейный либо квадратичный рост штрафа за превышение ε.")],
            source="https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html")
        add("nonnegative", "МНК · коэффициенты ≥ 0", "Другие линейные", "Запрещает отрицательные коэффициенты.",
            r"\min_{w\geq0,b}\;\|y-Xw-b\|_2^2",
            "Ограничивается знак коэффициентов. Значения y и свободный член могут быть отрицательными. Реализация scipy NNLS через sklearn LinearRegression(positive=True).",
            lambda p, seed: LinearRegression(positive=True, **p), [INTERCEPT],
            source="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html")
        extra_params = [_alpha(), INTERCEPT, ITERATIONS, TOLERANCE]
        add("weighted_lasso", "Weighted Lasso", exotic, "Штрафует признаки с разной силой.",
            r"\min_{w,b}\;\frac{\|e\|_2^2}{2n}+\alpha\sum_j a_j|w_j|",
            "Профиль задает веса в порядке преобразованных признаков: от 1 до weight_ratio либо в обратном порядке. Это явная настройка весов; Adaptive Lasso получает веса из обучающих данных.",
            native(WeightedLassoRegressor), extra_params + [
              _param("profile", "Профиль весов", "select", "increasing", options=["uniform", "increasing", "decreasing"], help_text="uniform: все 1; increasing/decreasing: линейное изменение по порядку признаков."),
              _param("weight_ratio", "Крайний вес", "float", 3, 1, 1000, 0.1, help_text="Отношение самого большого веса штрафа к самому маленькому.")],
            source="https://contrib.scikit-learn.org/skglm/generated/skglm.WeightedLasso.html")
        add("adaptive_lasso", "Adaptive Lasso", exotic, "Определяет разные веса штрафа по первичной оценке.",
            r"a_j=(|\hat w_{\mathrm{Ridge},j}|+\varepsilon)^{-\gamma},\quad\min\frac{\|e\|_2^2}{2n}+\alpha\sum_j a_j|w_j|",
            "Сначала Ridge обучается на той же обучающей части. Затем маленькие первичные коэффициенты получают более сильный штраф. epsilon делает веса конечными: это явно указанная численная стабилизация исходного Adaptive Lasso.",
            native(AdaptiveLassoRegressor), extra_params + [
              _param("gamma", "Степень адаптивных весов", "float", 1, 0.1, 5, 0.1, help_text="Усиливает разницу штрафов для маленьких и больших первичных коэффициентов."),
              _param("pilot_alpha", "Штраф первичного Ridge", "float", 1, 1e-8, 1e6, 0.1, help_text="Ridge использует собственную RSS-нормировку sklearn."),
              _param("epsilon", "Стабилизация весов ε", "float", 0.001, 1e-8, 1, 0.001, help_text="Прибавляется к модулю первичного коэффициента перед расчетом веса.")],
            source="https://pages.cs.wisc.edu/~shao/stat992/zou2006.pdf")
        add("group_lasso", "Group Lasso", exotic, "Отбирает признаки целыми группами.",
            r"\min\frac{\|e\|_2^2}{2n}+\alpha\sum_g\sqrt{|g|}\,\|w_g\|_2",
            "Соседние преобразованные столбцы образуют группы заданного размера. Вес √|g| учитывает размер группы. Отдельные коэффициенты внутри оставшейся группы не обязаны быть нулевыми.",
            native(GroupLassoRegressor), extra_params + [GROUP_SIZE],
            source="https://contrib.scikit-learn.org/skglm/generated/skglm.GroupLasso.html")
        add("sparse_group_lasso", "Sparse Group Lasso", exotic, "Отбирает и группы, и признаки внутри групп.",
            r"\min\frac{\|e\|_2^2}{2n}+\alpha\left[r\|w\|_1+(1-r)\sum_g\sqrt{|g|}\,\|w_g\|_2\right]",
            "L1 обнуляет отдельные признаки; групповой штраф способен обнулить сразу группу. r=1 дает Lasso; r=0 дает Group Lasso. Группы определяются порядком преобразованных столбцов.",
            native(SparseGroupLassoRegressor), extra_params + [GROUP_SIZE, RATIO],
            source="https://contrib.scikit-learn.org/skglm/auto_examples/plot_sparse_group_lasso.html")
        add("fused_lasso", "Fused Lasso", exotic, "Сжимает коэффициенты и сближает соседние.",
            r"\min\frac{\|e\|_2^2}{2n}+\alpha\|w\|_1+\lambda_f\sum_{j=1}^{p-1}|w_{j+1}-w_j|",
            "Порядок ПРИЗНАКОВ имеет смысл: штрафует различия соседних коэффициентов. Не сортирует наблюдения по времени. При одном признаке штраф разностей равен нулю. Задачу решает CVXPY/Clarabel.",
            native(FusedLassoRegressor), extra_params + [
              _param("fusion_strength", "Штраф разностей λf", "float", 0.1, 0, 1e6, 0.01, help_text="Большое значение делает соседние коэффициенты похожими или одинаковыми.")],
            source="https://academic.oup.com/jrsssb/article/67/1/91/7110658")
        add("tikhonov", "Тихонов · общий оператор", exotic, "Штрафует выбранное линейное преобразование коэффициентов.",
            r"\min\frac{\|e\|_2^2}{2n}+\frac\alpha2\|Dw\|_2^2",
            "D=I дает Ridge в нормировке RSS/(2n). Первая разность сближает соседние коэффициенты; вторая поощряет плавное изменение. При p меньше необходимого порядка нет разностей и соответствующий штраф нулевой.",
            native(TikhonovRegressor), [_alpha(1), INTERCEPT, ITERATIONS, TOLERANCE,
              _param("operator", "Оператор D", "select", "first_difference", options=["identity", "first_difference", "second_difference"], help_text="I, первая разность либо вторая разность по порядку коэффициентов.")],
            source="https://www.cvxpy.org/examples/basic/least_squares.html")
        add("l0", "L0 · точный перебор до 12 признаков", exotic, "Перебирает все допустимые подмножества признаков.",
            r"\min\frac{\|e\|_2^2}{2n}+\alpha\|w\|_0\quad\text{при}\quad\|w\|_0\leq k,\;p\leq12",
            "‖w‖₀ — число ненулевых коэффициентов. Для каждого подмножества библиотечный SVD решает МНК. Глобальное решение в этом ограниченном пространстве получается перебором, с обычной погрешностью чисел. До 4096 подмножеств; полиномиальные признаки тоже входят в предел 12.",
            native(BestSubsetRegressor), [_alpha(0.05), INTERCEPT,
              _param("max_features", "Максимум выбранных признаков k", "int", 3, 0, 12, 1, help_text="0 оставляет только свободный член. Самих преобразованных столбцов должно быть ≤12.")],
            source="https://arxiv.org/abs/2202.04820")
        add("scad", "SCAD", exotic, "Невыпуклый штраф с ослаблением сжатия больших коэффициентов.",
            r"\min\frac{\|e\|_2^2}{2n}+\sum_j p_{\alpha,\gamma}(|w_j|),\quad p'(t)=\begin{cases}\alpha&t\leq\alpha\\(\gamma\alpha-t)/(\gamma-1)&\alpha<t\leq\gamma\alpha\\0&t>\gamma\alpha\end{cases}",
            "У нуля штраф похож на L1, затем его производная уменьшается до нуля. gamma>2. Задача невыпуклая: результат решателя может быть локальным минимумом и зависеть от инициализации. Используется skglm SCAD.",
            native(SCADRegressor), extra_params + [
              _param("gamma", "Параметр формы γ", "float", 3.7, 2.01, 100, 0.1, help_text="Должен быть больше 2; определяет диапазон ослабления штрафа.")],
            source="https://contrib.scikit-learn.org/skglm/generated/skglm.penalties.SCAD.html")
        def mcp_factory(params, seed):
            from skglm import MCPRegression
            return MCPRegression(**params)
        add("mcp", "MCP", exotic, "Невыпуклый штраф, становящийся постоянным вдали от нуля.",
            r"p(t)=\begin{cases}\alpha t-t^2/(2\gamma)&t\leq\alpha\gamma\\\gamma\alpha^2/2&t>\alpha\gamma\end{cases},\quad\min\frac{\|e\|_2^2}{2n}+\sum_jp(|w_j|)",
            "У больших коэффициентов исчезает дополнительное сжатие. gamma задает форму; здесь используем gamma>1. Глобальный минимум не гарантируется, поскольку штраф невыпуклый. Реализация skglm MCPRegression.",
            mcp_factory, extra_params + [
              _param("gamma", "Параметр формы γ", "float", 3, 1.01, 100, 0.1, help_text="Больше γ делает поведение ближе к Lasso.")],
            source="https://contrib.scikit-learn.org/skglm/generated/skglm.MCPRegression.html")
        def sqrt_factory(params, seed):
            from skglm.experimental import SqrtLasso
            return SqrtLasso(**params)
        add("sqrt_lasso", "Square-root Lasso", exotic, "Использует длину вектора остатков вместо его квадрата.",
            r"\min_{w,b}\;\|y-Xw-b\|_2+\alpha\|w\|_1",
            "Точная нормировка skglm SqrtLasso: ‖e‖₂ без деления на √n. Для записи ‖e‖₂/√n+λ‖w‖₁ нужно alpha=√n·λ. API библиотеки экспериментальный; промежуточные коэффициенты fit не доступны.",
            sqrt_factory, extra_params,
            source="https://contrib.scikit-learn.org/skglm/generated/skglm.experimental.SqrtLasso.html")

    def catalogue(self) -> list[dict]:
        return deepcopy(list(self._specs.values()))

    def spec(self, model_id: str) -> dict:
        if model_id not in self._specs:
            raise ValueError(f"Неизвестная модель: {model_id}.")
        return deepcopy(self._specs[model_id])

    def create(self, model_id: str, params: dict, seed: int = 42) -> BaseEstimator:
        spec = self.spec(model_id)
        if not isinstance(params, dict):
            raise ValueError("Параметры модели должны быть объектом.")
        schema = {item["key"]: item for item in spec["params"]}
        unknown = set(params) - set(schema)
        if unknown:
            raise ValueError("Неизвестные параметры модели: " + ", ".join(sorted(unknown)))
        validated = {key: item["default"] for key, item in schema.items()}
        validated.update(params)
        for key, value in validated.items():
            item = schema[key]
            kind = item["type"]
            if kind == "bool" and not isinstance(value, bool):
                raise ValueError(f"{item['label']}: требуется true или false.")
            if kind == "select" and value not in item["options"]:
                raise ValueError(f"{item['label']}: недопустимый вариант {value!r}.")
            if kind in ("int", "float"):
                numeric = isinstance(value, Integral if kind == "int" else Real)
                if isinstance(value, bool) or not numeric or not np.isfinite(value):
                    raise ValueError(f"{item['label']}: требуется конечное {'целое ' if kind == 'int' else ''}число.")
                if value < item.get("min", -np.inf) or value > item.get("max", np.inf):
                    raise ValueError(f"{item['label']}: допустимый диапазон [{item.get('min')}, {item.get('max')}].")
                validated[key] = int(value) if kind == "int" else float(value)
        return self._factories[model_id](validated, seed)

    def objective(self, model_id: str, estimator: BaseEstimator, X, y) -> float | None:
        """The documented unweighted objective; unavailable means no invented loss."""
        if model_id in {"lars", "omp", "ransac", "theilsen", "bayesian_ridge", "ard"}:
            return None
        y = np.asarray(y, dtype=float)
        error = y - estimator.predict(X)
        w = np.asarray(estimator.coef_, dtype=float).reshape(-1)
        squared_sum = float(error @ error)
        l1, l2 = float(np.abs(w).sum()), float(w @ w)
        mean_half_square = squared_sum / (2 * len(y))
        alpha = getattr(estimator, "alpha", 0)
        if model_id in {"ols", "nonnegative"}:
            return squared_sum
        if model_id == "ridge":
            return squared_sum + alpha * l2
        if model_id in {"lasso", "lassolars"}:
            return mean_half_square + alpha * l1
        if model_id == "elasticnet":
            r = estimator.l1_ratio
            return mean_half_square + alpha * (r * l1 + (1 - r) * l2 / 2)
        if model_id == "sgd":
            absolute = np.abs(error)
            eps = estimator.epsilon
            loss = {
                "squared_error": lambda: error ** 2 / 2,
                "huber": lambda: np.where(absolute <= eps, error ** 2 / 2, eps * absolute - eps ** 2 / 2),
                "epsilon_insensitive": lambda: np.maximum(absolute - eps, 0),
                "squared_epsilon_insensitive": lambda: np.maximum(absolute - eps, 0) ** 2,
            }[estimator.loss]()
            penalty = 0
            if estimator.penalty == "l1":
                penalty = alpha * l1
            elif estimator.penalty == "l2":
                penalty = alpha * l2 / 2
            elif estimator.penalty == "elasticnet":
                r = estimator.l1_ratio
                penalty = alpha * (r * l1 + (1 - r) * l2 / 2)
            return float(loss.mean() + penalty)
        if model_id == "huber":
            scale = estimator.scale_
            u = np.abs(error) / scale
            huber = np.where(u <= estimator.epsilon, u ** 2,
                             2 * estimator.epsilon * u - estimator.epsilon ** 2)
            return float(len(y) * scale + scale * huber.sum() + alpha * l2)
        if model_id == "quantile":
            return float(mean_pinball_loss(y, estimator.predict(X), alpha=estimator.quantile) + alpha * l1)
        if model_id in {"poisson", "gamma", "tweedie"}:
            power = {"poisson": 1, "gamma": 2}.get(model_id, getattr(estimator, "power", 0))
            return float(mean_tweedie_deviance(y, estimator.predict(X), power=power) / 2 + alpha * l2 / 2)
        if model_id == "linear_svr":
            excess = np.maximum(np.abs(error) - estimator.epsilon, 0)
            if estimator.loss == "squared_epsilon_insensitive":
                excess = excess ** 2
            intercept_penalty = float(estimator.intercept_[0]) ** 2 if estimator.fit_intercept else 0
            return float((l2 + intercept_penalty) / 2 + estimator.C * excess.sum())
        if model_id in {"weighted_lasso", "adaptive_lasso"}:
            return float(mean_half_square + alpha * np.dot(estimator.weights_, np.abs(w)))
        if model_id in {"group_lasso", "sparse_group_lasso"}:
            group_penalty = sum(a * np.linalg.norm(w[g]) for a, g in zip(estimator.group_weights_, estimator.groups_))
            r = getattr(estimator, "l1_ratio", 0)
            return float(mean_half_square + alpha * (r * l1 + (1 - r) * group_penalty))
        if model_id == "fused_lasso":
            return float(mean_half_square + alpha * l1 + estimator.fusion_strength * np.abs(np.diff(w)).sum())
        if model_id == "tikhonov":
            transformed = estimator.operator_ @ w
            return float(mean_half_square + alpha * (transformed @ transformed) / 2)
        if model_id == "l0":
            return float(mean_half_square + alpha * np.count_nonzero(w))
        if model_id in {"scad", "mcp"}:
            from skglm.penalties import MCPenalty, SCAD
            penalty = SCAD(alpha, estimator.gamma) if model_id == "scad" else MCPenalty(alpha, estimator.gamma)
            return float(mean_half_square + penalty.value(w))
        if model_id == "sqrt_lasso":
            return float(np.sqrt(squared_sum) + alpha * l1)
        return None
