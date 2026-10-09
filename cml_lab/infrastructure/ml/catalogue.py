"""Safe algorithm metadata and factories for the classic ML adapters.

An algorithm definition is immutable metadata. This module builds estimators;
it neither trains them nor persists user recipes or fitted artifacts.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from importlib import import_module, util
from numbers import Integral, Real
from typing import Any

import numpy as np

from linear_lab.models import ModelRegistry


TASKS = ("regression", "classification", "clustering", "ranking", "forecasting",
         "panel", "anomaly", "reduction")
REGRESSION_TASKS = ("regression", "forecasting", "panel")
FAMILY_LABELS = {
    "linear": "Линейные модели", "regularization": "Регуляризация",
    "robust": "Устойчивые модели", "tree": "Деревья", "forest": "Леса",
    "boosting": "Бустинг", "ensemble": "Ансамбли", "neighbors": "Соседи",
    "svm": "Метод опорных векторов", "kernel": "Ядерные модели",
    "naive_bayes": "Наивный Байес", "discriminant": "Дискриминантный анализ",
    "gaussian_process": "Гауссовские процессы", "baseline": "Контрольный ответ",
    "centroid": "Центры групп", "density": "Плотность",
    "hierarchy": "Иерархия групп", "graph": "Графовые методы",
    "mixture": "Смеси распределений", "decomposition": "Разложение матрицы",
    "manifold": "Многообразия", "projection": "Случайные проекции",
    "anomaly": "Поиск аномалий", "neural_network": "Нейронные сети sklearn",
}
FAMILY_LESSONS = {
    "linear": "classification-basics", "regularization": "06-ridge",
    "robust": "12-huber", "tree": "tree-depth", "forest": "forest",
    "boosting": "boosting", "ensemble": "model-ensemble", "neighbors": "neighbors",
    "svm": "svm", "kernel": "svm", "naive_bayes": "naive-bayes",
    "discriminant": "discriminant-analysis", "gaussian_process": "gaussian-process",
    "baseline": "classification-basics", "centroid": "clustering-basics",
    "density": "clustering-density", "hierarchy": "clustering-hierarchy",
    "graph": "clustering-basics", "mixture": "clustering-mixtures",
    "decomposition": "dimensionality-reduction", "manifold": "dimensionality-reduction",
    "projection": "dimensionality-reduction", "anomaly": "anomaly-detection",
    "neural_network": "neural-network",
}


def _field(key: str, label: str, kind: str, default: Any, help_text: str,
           minimum: float | int | None = None, maximum: float | int | None = None,
           options: list | None = None, step: float | int | None = None) -> dict:
    result = {"key": key, "label": label, "type": kind, "default": default,
              "help": help_text}
    if minimum is not None:
        result["min"] = minimum
    if maximum is not None:
        result["max"] = maximum
    if options is not None:
        result["options"] = options
    if step is not None:
        result["step"] = step
    return result


def _integer(key, label, default, minimum, maximum, help_text):
    return _field(key, label, "int", default, help_text, minimum, maximum, step=1)


def _number(key, label, default, minimum, maximum, help_text):
    return _field(key, label, "float", default, help_text, minimum, maximum)


def _choice(key, label, default, options, help_text):
    return _field(key, label, "select", default, help_text, options=options)


def _boolean(key, label, default, help_text):
    return _field(key, label, "bool", default, help_text)


def _trees():
    return [
        _integer("max_depth", "Глубина дерева", 0, 0, 64,
                 "Число уровней разделения. 0 снимает предел; глубокое дерево легче запоминает обучение."),
        _integer("min_samples_leaf", "Минимум строк в листе", 1, 1, 10000,
                 "Лист хранит итоговый ответ. Больше строк в листе дает более грубые и устойчивые ответы."),
        _integer("min_samples_split", "Минимум строк для разделения", 2, 2, 10000,
                 "Узел с меньшим количеством строк не разделяется на дочерние узлы."),
    ]


def _forest(classifier: bool):
    return [
        _integer("n_estimators", "Количество деревьев", 100, 1, 1000,
                 "Деревья обучаются на разных выборках и объединяют ответы. Больше деревьев требует больше времени."),
        *_trees(),
        _choice("max_features", "Признаки для одного разделения", "sqrt" if classifier else "all",
                ["sqrt", "log2", "all"],
                "sqrt/log2 ограничивают число случайно выбранных признаков; all рассматривает все признаки."),
        _boolean("bootstrap", "Выборки строк с повторениями", True,
                 "Каждое дерево получает случайную выборку строк с возможными повторениями."),
    ]


def _boosting(histogram: bool = False):
    iterations = "max_iter" if histogram else "n_estimators"
    return [
        _integer(iterations, "Количество шагов бустинга", 100, 1, 1000,
                 "Каждый шаг добавляет дерево, которое исправляет ошибки предыдущих шагов."),
        _number("learning_rate", "Доля исправления за шаг", 0.1, 0.001, 1.0,
                "Меньшее значение ослабляет одно дерево. Для того же качества может понадобиться больше шагов."),
        _integer("max_depth", "Глубина одного дерева", 3 if not histogram else 0, 0, 32,
                 "Глубина задает сложность отдельных исправлений. 0 снимает предел глубины."),
        _integer("min_samples_leaf", "Минимум строк в листе", 20 if histogram else 1, 1, 10000,
                 "Не позволяет строить ответ по слишком малому числу обучающих строк."),
    ]


def _neighbors(radius: bool = False):
    size = (_number("radius", "Радиус поиска соседей", 1.0, 0.000001, 100000,
                    "Используются все строки внутри этого расстояния. При слишком малом радиусе соседей может не быть.")
            if radius else _integer("n_neighbors", "Количество соседей", 5, 1, 1000,
                                    "Модель использует ответы ближайших обучающих строк. Число не может превышать размер обучения."))
    return [size, _choice("weights", "Вклад соседей", "uniform", ["uniform", "distance"],
                          "uniform дает одинаковый вклад; distance усиливает близкие строки."),
            _choice("p", "Расстояние", 2, [1, 2],
                    "1 складывает модули разностей; 2 использует евклидово расстояние.")]


def _svm(classifier: bool, nu: bool = False):
    strength = (_number("nu", "Доля опорных векторов ν", 0.5, 0.001, 0.999,
                        "Ограничивает ошибки и число опорных строк. Для некоторых долей классов значение недопустимо.")
                if nu else _number("C", "Допустимость ошибок C", 1.0, 0.000001, 100000,
                                   "Большое C сильнее штрафует ошибки обучения и ослабляет относительное ограничение сложности."))
    result = [strength,
              _choice("kernel", "Форма границы", "rbf", ["linear", "rbf", "poly", "sigmoid"],
                      "linear строит прямую границу; другие ядра позволяют искривлять границу в пространстве признаков."),
              _choice("gamma", "Масштаб ядра γ", "scale", ["scale", "auto"],
                      "Определяет область влияния строки в нелинейном ядре. scale учитывает разброс обучающих признаков."),
              _integer("degree", "Степень полиномиального ядра", 3, 1, 6,
                       "Используется только с ядром poly. Большая степень усложняет границу.")]
    if not classifier and not nu:
        result.append(_number("epsilon", "Ширина допуска ε", 0.1, 0, 100000,
                              "Ошибка прогноза внутри этого допуска не штрафуется. Измеряется в единицах цели."))
    return result


@dataclass(frozen=True)
class _Algorithm:
    metadata: dict
    class_path: str
    constants: dict
    factory: str = "native"


class AlgorithmCatalogue:
    """One whitelist for form metadata and estimator construction."""

    def __init__(self):
        self._legacy = ModelRegistry()
        self._entries: dict[str, _Algorithm] = {}
        self._import_failures: dict[str, str] = {}
        self._optional_checked: set[str] = set()
        self._register_sklearn()
        self._register_boosting()

    def _add(self, algorithm_id, name, family, tasks, class_path, description,
             params=(), capabilities=None, constants=None, factory="native", lesson_id=None,
             max_rows=None):
        if algorithm_id in self._entries:
            raise ValueError(f"Повторный идентификатор алгоритма: {algorithm_id}.")
        module, class_name = class_path.rsplit(".", 1)
        library = module.split(".", 1)[0]
        lesson = lesson_id or FAMILY_LESSONS[family]
        schema = deepcopy(list(params))
        for item in schema:
            item["help_key"] = f"model.{algorithm_id}.{item['key']}"
            item["lesson_id"] = lesson
        caps = {
            "predict": True, "predict_proba": False, "decision_function": False,
            "fit_predict": False, "partial_fit": False, "staged_predict": False,
            "coefficients": False, "feature_importance": False, "transductive": False,
            "input_domain": "real", "requires_dense": False, "allows_missing": False,
            "native_categorical": False, "transform": False,
        }
        caps.update(capabilities or {})
        if library == "sklearn" and factory != "ensemble":
            from sklearn.utils import get_tags
            cls = getattr(import_module(module), class_name)
            input_tags = get_tags(cls()).input_tags
            caps["requires_dense"] = not input_tags.sparse
            caps["allows_missing"] = input_tags.allow_nan
        if max_rows is not None:
            caps["max_rows"] = max_rows
        source = (f"https://scikit-learn.org/1.8/modules/generated/{class_path}.html"
                  if library == "sklearn" else {
                      "xgboost": "https://xgboost.readthedocs.io/en/stable/python/python_api.html",
                      "lightgbm": "https://lightgbm.readthedocs.io/en/stable/Python-API.html",
                      "catboost": "https://catboost.ai/docs/en/concepts/python-reference_" + class_name.lower(),
                  }[library])
        self._entries[algorithm_id] = _Algorithm({
            "id": algorithm_id, "name": name, "family": family,
            "family_label": FAMILY_LABELS[family], "tasks": list(tasks),
            "library": library, "description": description, "params": schema,
            "capabilities": caps, "lesson_id": lesson, "source": source,
        }, class_path, deepcopy(constants or {}), factory)

    def _register_sklearn(self):
        add = self._add
        classifier_caps = {"predict_proba": True}
        regression = REGRESSION_TASKS
        classification = ("classification",)
        tree_desc = "Разделяет строки по условиям на признаки. В каждом листе хранит итоговый ответ."
        for kind, tasks in (("Regressor", regression), ("Classifier", classification)):
            suffix = "regressor" if kind == "Regressor" else "classifier"
            russian = "регрессия" if kind == "Regressor" else "классификация"
            caps = {"feature_importance": True, "allows_missing": True}
            if kind == "Classifier":
                caps.update(classifier_caps)
            for prefix, title in (("DecisionTree", "Дерево решений"), ("ExtraTree", "Случайное дерево")):
                add(f"{'decision_tree' if prefix == 'DecisionTree' else 'extra_tree'}_{suffix}",
                    f"{title} · {russian}", "tree", tasks, f"sklearn.tree.{prefix}{kind}", tree_desc,
                    _trees(), caps)
            for prefix, title, identifier in (("RandomForest", "Случайный лес", "random_forest"),
                                               ("ExtraTrees", "Предельно случайные деревья", "extra_trees")):
                add(f"{identifier}_{suffix}", f"{title} · {russian}", "forest", tasks,
                    f"sklearn.ensemble.{prefix}{kind}", "Объединяет ответы нескольких случайно обученных деревьев.",
                    _forest(kind == "Classifier"), caps)
            for prefix, title, identifier, hist in (
                ("GradientBoosting", "Градиентный бустинг", "gradient_boosting", False),
                ("HistGradientBoosting", "Гистограммный бустинг", "hist_gradient_boosting", True),
            ):
                params = _boosting(hist)
                constant = {"early_stopping": False} if hist else {}
                stage_caps = {"staged_predict": True, "feature_importance": not hist,
                              "allows_missing": hist, **(classifier_caps if kind == "Classifier" else {})}
                add(f"{identifier}_{suffix}", f"{title} · {russian}", "boosting", tasks,
                    f"sklearn.ensemble.{prefix}{kind}", "Последовательно добавляет деревья, исправляющие ошибки модели.",
                    params, stage_caps, constant)
            add(f"adaboost_{suffix}", f"AdaBoost · {russian}", "boosting", tasks,
                f"sklearn.ensemble.AdaBoost{kind}", "Повышает внимание к строкам, на которых предыдущие модели ошибались.",
                [_integer("n_estimators", "Количество шагов", 50, 1, 1000,
                          "Предел числа последовательно добавляемых моделей."),
                 _number("learning_rate", "Вклад шага", 1.0, 0.001, 2.0,
                         "Множитель вклада очередной модели в итоговый ответ.")],
                {"staged_predict": True, "feature_importance": True,
                 **(classifier_caps if kind == "Classifier" else {})})
            add(f"bagging_{suffix}", f"Бэггинг · {russian}", "ensemble", tasks,
                f"sklearn.ensemble.Bagging{kind}", "Обучает одинаковые базовые модели на разных случайных выборках строк.",
                [_integer("n_estimators", "Количество базовых моделей", 20, 1, 500,
                          "Количество независимо обученных моделей, ответы которых объединяются."),
                 _number("max_samples", "Доля строк для одной модели", 1.0, 0.01, 1.0,
                         "Какая доля обучающих строк попадает в одну случайную выборку."),
                 _boolean("bootstrap", "Выборка с повторениями", True,
                          "Одна строка может несколько раз попасть в выборку одной модели.")],
                classifier_caps if kind == "Classifier" else {})
            for radius in (False, True):
                prefix = "RadiusNeighbors" if radius else "KNeighbors"
                ident = "radius_neighbors" if radius else "knn"
                constants = {"outlier_label": "most_frequent"} if radius and kind == "Classifier" else {}
                add(f"{ident}_{suffix}", f"{'Соседи в радиусе' if radius else 'Ближайшие соседи'} · {russian}",
                    "neighbors", tasks, f"sklearn.neighbors.{prefix}{kind}",
                    "Использует ответы похожих обучающих строк, найденных по расстоянию.",
                    _neighbors(radius), classifier_caps if kind == "Classifier" else {}, constants)
            add(f"dummy_{suffix}", f"Контрольный ответ · {russian}", "baseline", tasks,
                f"sklearn.dummy.Dummy{kind}", "Дает простой ответ без поиска связи с признаками; нужен для сравнения.",
                [_choice("strategy", "Правило ответа", "prior" if kind == "Classifier" else "mean",
                         ["most_frequent", "prior", "stratified", "uniform"] if kind == "Classifier" else ["mean", "median", "quantile"],
                         "Выбирает самый частый класс, среднее или другой простой ответ из обучения.")],
                classifier_caps if kind == "Classifier" else {},
                {"quantile": 0.5} if kind == "Regressor" else {})
            self._add_ensembles(kind, tasks)

        add("logistic_regression", "Логистическая регрессия", "linear", classification,
            "sklearn.linear_model.LogisticRegression", "Оценивает вероятности классов по взвешенной сумме признаков.",
            [_number("C", "Допустимость ошибок C", 1.0, 0.000001, 100000,
                     "Большое C ослабляет регуляризацию; малое сильнее сжимает коэффициенты."),
             _choice("class_weight", "Веса классов", "none", ["none", "balanced"],
                     "balanced повышает вклад редких классов; это не меняет количество строк."),
             _integer("max_iter", "Предел шагов решателя", 500, 10, 10000,
                      "Если решатель не успел сойтись, увеличьте предел или исправьте масштаб признаков.")],
            {"predict_proba": True, "decision_function": True, "coefficients": True},
            {"solver": "lbfgs"})
        add("ridge_classifier", "Ridge · классификация", "linear", classification,
            "sklearn.linear_model.RidgeClassifier", "Линейная граница с квадратичным ограничением коэффициентов.",
            [_number("alpha", "Сила квадратичного ограничения", 1.0, 0, 100000,
                     "Большее значение сильнее уменьшает коэффициенты."),
             _choice("class_weight", "Веса классов", "none", ["none", "balanced"],
                     "balanced повышает вклад редких классов.")],
            {"decision_function": True, "coefficients": True})
        add("sgd_classifier", "SGD · классификация", "linear", classification,
            "sklearn.linear_model.SGDClassifier", "Обновляет линейную границу небольшими шагами по обучающим строкам.",
            [_choice("loss", "Что считать ошибкой", "log_loss", ["log_loss", "hinge", "modified_huber"],
                     "log_loss дает вероятности; hinge соответствует линейному SVM; modified_huber ослабляет часть больших ошибок."),
             _number("alpha", "Сила регуляризации", 0.0001, 0.00000001, 100,
                     "Усиливает ограничение коэффициентов."),
             _choice("penalty", "Вид ограничения", "l2", ["l2", "l1", "elasticnet"],
                     "L2 сжимает коэффициенты; L1 может обнулить их; elasticnet смешивает оба ограничения."),
             _integer("max_iter", "Предел проходов по данным", 1000, 1, 10000,
                      "Максимальное количество эпох; остановка по точности может произойти раньше.")],
            {"predict_proba": True, "decision_function": True, "coefficients": True, "partial_fit": True})
        add("perceptron", "Перцептрон", "linear", classification,
            "sklearn.linear_model.Perceptron", "Меняет линейную границу, когда строка отнесена к неправильному классу.",
            [_number("eta0", "Размер шага", 1.0, 0.000001, 100,
                     "Множитель одного обновления коэффициентов."),
             _integer("max_iter", "Предел проходов", 1000, 1, 10000,
                      "Максимальное количество проходов по обучающим строкам.")],
            {"decision_function": True, "coefficients": True, "partial_fit": True})
        for class_name, identifier, nu in (("SVC", "svc", False), ("NuSVC", "nu_svc", True)):
            add(identifier, class_name + " · классификация", "svm", classification,
                "sklearn.svm." + class_name, "Строит границу классов с помощью опорных обучающих строк.",
                _svm(True, nu), {"predict_proba": True, "decision_function": True},
                {"probability": True, "cache_size": 128}, max_rows=20000)
        add("linear_svc", "Линейный SVM · классификация", "svm", classification,
            "sklearn.svm.LinearSVC", "Строит линейную границу с зазором между классами.",
            [_number("C", "Допустимость ошибок C", 1.0, 0.000001, 100000,
                     "Большое C сильнее штрафует ошибки обучения."),
             _integer("max_iter", "Предел шагов", 3000, 10, 30000,
                      "Предел работы библиотечного решателя.")],
            {"decision_function": True, "coefficients": True})
        for class_name, identifier, nu in (("SVR", "svr", False), ("NuSVR", "nu_svr", True)):
            add(identifier, class_name + " · регрессия", "svm", regression,
                "sklearn.svm." + class_name, "Использует опорные строки для прогноза числовой цели.",
                _svm(False, nu), constants={"cache_size": 128}, max_rows=20000)
        add("nearest_centroid", "Ближайший центр класса", "neighbors", classification,
            "sklearn.neighbors.NearestCentroid", "Сравнивает строку с типичным центром каждого класса.",
            [_choice("metric", "Расстояние до центра", "euclidean", ["euclidean", "manhattan"],
                     "euclidean использует евклидово расстояние; manhattan складывает модули разностей.")],
            {"predict_proba": True, "decision_function": True})
        add("gaussian_nb", "Наивный Байес · нормальные признаки", "naive_bayes", classification,
            "sklearn.naive_bayes.GaussianNB", "Оценивает распределение каждого числового признака внутри класса.",
            [_number("var_smoothing", "Добавка к дисперсии", 0.000000001, 0.000000000001, 1.0,
                     "Не позволяет признаку с почти нулевым разбросом создавать чрезмерно уверенный ответ.")],
            {"predict_proba": True, "partial_fit": True, "requires_dense": True})
        for class_name, identifier in (("MultinomialNB", "multinomial_nb"), ("ComplementNB", "complement_nb"),
                                       ("BernoulliNB", "bernoulli_nb")):
            params = [_number("alpha", "Добавка к частотам", 1.0, 0.0000000001, 100,
                              "Сглаживает редкие значения, чтобы одна нулевая частота не обнулила оценку класса.")]
            if class_name == "BernoulliNB":
                params.append(_number("binarize", "Порог присутствия признака", 0.0, -100000, 100000,
                                      "Значение выше порога превращается в 1, остальные в 0."))
            add(identifier, class_name, "naive_bayes", classification, "sklearn.naive_bayes." + class_name,
                "Считает признаки условно независимыми внутри класса и объединяет их оценки.", params,
                {"predict_proba": True, "partial_fit": True,
                 "input_domain": "real" if class_name == "BernoulliNB" else "nonnegative"})
        add("lda", "Линейный дискриминантный анализ", "discriminant", classification,
            "sklearn.discriminant_analysis.LinearDiscriminantAnalysis",
            "Предполагает нормальные распределения классов с одинаковой ковариацией.",
            [_choice("solver", "Решатель", "svd", ["svd", "lsqr", "eigen"],
                     "Разные библиотечные способы оценить общую ковариацию и границу классов.")],
            {"predict_proba": True, "decision_function": True, "coefficients": True, "requires_dense": True})
        add("qda", "Квадратичный дискриминантный анализ", "discriminant", classification,
            "sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis",
            "Разрешает классам иметь разные ковариации и искривляет границу.",
            [_number("reg_param", "Стабилизация ковариации", 0.1, 0, 1,
                     "При большом значении ковариация ближе к единичной матрице; помогает при нехватке строк.")],
            {"predict_proba": True, "decision_function": True, "requires_dense": True})
        add("kernel_ridge", "Ядерная Ridge-регрессия", "kernel", regression,
            "sklearn.kernel_ridge.KernelRidge", "Применяет квадратичную регуляризацию к модели в пространстве ядра.",
            [_number("alpha", "Сила ограничения", 1.0, 0.00000001, 100000,
                     "Большее значение сильнее ограничивает модель."),
             _choice("kernel", "Ядро", "rbf", ["linear", "rbf", "polynomial", "laplacian"],
                     "Задает способ сравнивать пары строк и форму нелинейной модели.")], max_rows=5000)
        add("pls_regression", "PLS-регрессия", "linear", regression, "sklearn.cross_decomposition.PLSRegression",
            "Создает небольшое число компонент, связанных одновременно с признаками и целью.",
            [_integer("n_components", "Количество компонент", 2, 1, 100,
                      "Число внутренних направлений. Не должно превышать число преобразованных признаков.")],
            {"coefficients": True, "requires_dense": True})
        for classifier in (False, True):
            class_name = "GaussianProcessClassifier" if classifier else "GaussianProcessRegressor"
            identifier = "gaussian_process_classifier" if classifier else "gaussian_process_regressor"
            params = [_choice("kernel", "Форма функции сходства", "rbf", ["rbf", "matern", "dot"],
                              "rbf строит гладкую зависимость; matern допускает менее гладкую; dot соответствует линейному сходству.")]
            if not classifier:
                params.append(_number("alpha", "Дисперсия наблюдательного шума", 0.000001, 0.0000000001, 100000,
                                      "Добавляется к диагонали ковариации; описывает шум наблюдений и стабилизирует вычисления."))
            add(identifier, "Гауссовский процесс · " + ("классификация" if classifier else "регрессия"),
                "gaussian_process", classification if classifier else regression,
                "sklearn.gaussian_process." + class_name, "Описывает распределение возможных зависимостей между признаками и ответом.",
                params, {"predict_proba": classifier, "requires_dense": True}, factory="gaussian_process", max_rows=2000)
        for kind, tasks in (("Regressor", regression), ("Classifier", classification)):
            suffix = "regressor" if kind == "Regressor" else "classifier"
            add(f"mlp_{suffix}", "Многослойный перцептрон · " + ("регрессия" if kind == "Regressor" else "классификация"),
                "neural_network", tasks, "sklearn.neural_network.MLP" + kind,
                "Обучает небольшую полносвязную сеть средствами sklearn.",
                [_integer("hidden_units", "Нейронов в скрытом слое", 50, 1, 512,
                          "Один скрытый слой; больше нейронов повышает гибкость и стоимость обучения."),
                 _choice("activation", "Функция скрытого слоя", "relu", ["relu", "tanh", "logistic"],
                         "Нелинейная функция, применяемая после взвешенной суммы."),
                 _number("alpha", "Сила ограничения весов", 0.0001, 0, 100,
                         "Квадратичное ограничение весов, нормированное библиотекой по размеру обучения."),
                 _integer("max_iter", "Предел эпох", 200, 1, 2000,
                          "Максимум проходов по данным для библиотечного оптимизатора.")],
                {"predict_proba": kind == "Classifier", "partial_fit": True}, factory="mlp")
        self._add_clustering()
        self._add_anomaly()
        self._add_reduction()

    def _add_ensembles(self, kind, tasks):
        suffix = "classifier" if kind == "Classifier" else "regressor"
        classifiers = kind == "Classifier"
        component_ids = [f"decision_tree_{suffix}", f"random_forest_{suffix}",
                         f"extra_trees_{suffix}", f"hist_gradient_boosting_{suffix}",
                         f"knn_{suffix}", "logistic_regression" if classifiers else "ridge",
                         "gaussian_nb" if classifiers else "svr",
                         f"voting_{suffix}", f"stacking_{suffix}"]
        fields = [_choice("base_estimators", "Готовый набор компонентов", "tree_forest_linear",
                          ["tree_forest_linear", "forest_boosting", "neighbors_linear"],
                          "Выберите набор базовых моделей или задайте свои компоненты ниже."),
                  {"key": "components", "label": "Собственные компоненты", "type": "models", "default": [],
                   "options": component_ids, "max_items": 8, "max_depth": 3,
                   "help": "Добавьте модели и настройте их параметры. Пустой список использует готовый набор."}]
        voting_fields = deepcopy(fields)
        if classifiers:
            voting_fields.append(_choice("voting", "Правило голосования", "soft", ["hard", "soft"],
                                         "hard объединяет классы; soft усредняет вероятности. Для soft все компоненты должны давать вероятности."))
        self._add(f"voting_{suffix}", "Голосование моделей · " + ("классификация" if classifiers else "регрессия"),
                  "ensemble", tasks, "sklearn.ensemble.Voting" + kind,
                  "Объединяет ответы разных моделей без обучения отдельного итогового решателя.", voting_fields,
                  {"predict_proba": classifiers}, factory="ensemble")
        self._add(f"stacking_{suffix}", "Стекинг моделей · " + ("классификация" if classifiers else "регрессия"),
                  "ensemble", ("classification",) if classifiers else ("regression",), "sklearn.ensemble.Stacking" + kind,
                  "Обучает итоговую модель на OOF-прогнозах базовых моделей для независимых строк. Группы и временные ряды не поддерживаются; подготовка данных общая внутри внешнего обучения.",
                  [*fields, _integer("cv", "Внутренние части для метамодели", 3, 2, 10,
                                     "Базовые ответы для метамодели получаются внутри этого кросс-разбиения независимых строк. Временные и групповые данные отклоняются, включая вложенный стекинг.")],
                  {"predict_proba": classifiers}, factory="ensemble")

    def _add_clustering(self):
        add = self._add
        task = ("clustering",)
        count = _integer("n_clusters", "Количество групп", 3, 2, 100,
                         "Алгоритм должен построить столько групп; число не выбирается автоматически.")
        component_count = _integer("n_components", "Количество распределений", 3, 1, 100,
                                   "Каждое распределение описывает одну часть смеси.")
        for name, identifier, title in (("KMeans", "kmeans", "K-средних"),
                                        ("MiniBatchKMeans", "minibatch_kmeans", "K-средних на мини-пакетах"),
                                        ("BisectingKMeans", "bisecting_kmeans", "Делящие K-средних")):
            add(identifier, title, "centroid", task, "sklearn.cluster." + name,
                "Находит центры и относит строки к ближайшему центру.",
                [count, _integer("max_iter", "Предел шагов", 300, 1, 3000,
                                 "Максимальное количество обновлений центров.")],
                {"fit_predict": True, "transform": True}, {"n_init": 10} if name != "BisectingKMeans" else {})
        add("agglomerative", "Агломеративная кластеризация", "hierarchy", task,
            "sklearn.cluster.AgglomerativeClustering", "Последовательно объединяет ближайшие группы.",
            [count, _choice("linkage", "Способ сравнивать группы", "ward", ["ward", "complete", "average", "single"],
                            "ward уменьшает суммарный разброс; другие варианты сравнивают расстояния между строками двух групп.")],
            {"predict": False, "fit_predict": True, "transductive": True}, max_rows=10000)
        add("dbscan", "DBSCAN", "density", task, "sklearn.cluster.DBSCAN",
            "Соединяет области с достаточным числом соседей; редкие строки оставляет шумом.",
            [_number("eps", "Радиус соседства ε", 0.5, 0.000001, 100000,
                     "Строки внутри этого расстояния считаются соседями."),
             _integer("min_samples", "Соседей для плотной области", 5, 1, 1000,
                      "Минимальное число строк, включая саму строку, внутри радиуса.")],
            {"predict": False, "fit_predict": True, "transductive": True}, max_rows=20000)
        add("hdbscan", "HDBSCAN", "density", task, "sklearn.cluster.HDBSCAN",
            "Ищет устойчивые плотные группы при разных радиусах соседства.",
            [_integer("min_cluster_size", "Минимальный размер группы", 5, 2, 10000,
                      "Меньшие группы не сохраняются как отдельные кластеры."),
             _integer("min_samples", "Соседей для плотности", 5, 1, 1000,
                      "Большее значение делает определение плотной области строже.")],
            {"predict": False, "fit_predict": True, "transductive": True}, max_rows=20000)
        add("optics", "OPTICS", "density", task, "sklearn.cluster.OPTICS",
            "Упорядочивает строки по достижимости плотных областей и выделяет группы.",
            [_integer("min_samples", "Соседей для плотности", 5, 2, 1000,
                      "Минимальная плотность, необходимая для ядра группы."),
             _number("xi", "Порог изменения плотности", 0.05, 0.001, 0.99,
                     "Определяет, какое изменение достижимости считать границей группы.")],
            {"predict": False, "fit_predict": True, "transductive": True}, max_rows=10000)
        add("mean_shift", "Средний сдвиг", "density", task, "sklearn.cluster.MeanShift",
            "Перемещает центры к областям высокой плотности.",
            [_number("bandwidth", "Радиус усреднения", 1.0, 0.000001, 100000,
                     "Строки внутри радиуса участвуют в перемещении центра.")],
            {"fit_predict": True}, {"bin_seeding": False}, max_rows=5000)
        add("affinity_propagation", "Распространение сходства", "graph", task,
            "sklearn.cluster.AffinityPropagation", "Выбирает характерные обучающие строки как представителей групп.",
            [_number("damping", "Затухание обновления", 0.5, 0.5, 0.99,
                     "Доля предыдущего сообщения, сохраняемая при обновлении.")],
            {"fit_predict": True}, max_rows=3000)
        add("spectral_clustering", "Спектральная кластеризация", "graph", task,
            "sklearn.cluster.SpectralClustering", "Строит граф сходства строк и разделяет его в новом представлении.",
            [count, _number("gamma", "Масштаб сходства γ", 1.0, 0.000001, 1000,
                            "Управляет локальностью сходства для RBF-графа.")],
            {"predict": False, "fit_predict": True, "transductive": True}, max_rows=5000)
        add("birch", "BIRCH", "hierarchy", task, "sklearn.cluster.Birch",
            "Сжимает похожие строки в дерево компактных групп, затем объединяет их.",
            [count, _number("threshold", "Предел радиуса компактной группы", 0.5, 0.000001, 100000,
                            "Меньшее значение создает больше промежуточных групп.")],
            {"fit_predict": True, "partial_fit": True, "transform": True})
        for class_name, identifier in (("GaussianMixture", "gaussian_mixture"),
                                       ("BayesianGaussianMixture", "bayesian_gaussian_mixture")):
            add(identifier, class_name, "mixture", task, "sklearn.mixture." + class_name,
                "Описывает данные смесью нормальных распределений и оценивает принадлежность к ним.",
                [component_count, _choice("covariance_type", "Форма распределений", "full",
                                           ["full", "tied", "diag", "spherical"],
                                           "full допускает зависимости признаков; diag оставляет отдельную дисперсию каждого признака."),
                 _integer("max_iter", "Предел шагов", 100, 1, 3000,
                          "Максимум уточнений распределений смеси.")],
                {"predict_proba": True, "fit_predict": True, "requires_dense": True})

    def _add_anomaly(self):
        task = ("anomaly",)
        contamination = _number("contamination", "Ожидаемая доля аномалий", 0.05, 0.0001, 0.5,
                                "Используется для выбора порога. Это предположение, а не измеренная доля истинных аномалий.")
        self._add("isolation_forest", "Изолирующий лес", "anomaly", task,
                  "sklearn.ensemble.IsolationForest", "Редкую строку удается изолировать меньшим числом случайных разделений.",
                  [contamination, _integer("n_estimators", "Количество деревьев", 100, 1, 1000,
                                          "Количество случайных деревьев для оценки изолируемости строки.")],
                  {"decision_function": True, "fit_predict": True})
        self._add("local_outlier_factor", "Локальная аномальность · LOF", "anomaly", task,
                  "sklearn.neighbors.LocalOutlierFactor", "Сравнивает плотность около строки с плотностью около ее соседей.",
                  [contamination, _integer("n_neighbors", "Количество соседей", 20, 2, 1000,
                                           "Число соседей, по которым оценивается локальная плотность.")],
                  {"predict": False, "fit_predict": True, "transductive": True})
        self._add("one_class_svm", "Одноклассовый SVM", "anomaly", task,
                  "sklearn.svm.OneClassSVM", "Определяет область типичных обучающих строк.",
                  [_number("nu", "Допуск необычных строк ν", 0.1, 0.001, 0.999,
                           "Ограничение доли обучающих выбросов и доли опорных строк."),
                   _choice("kernel", "Ядро", "rbf", ["rbf", "linear", "poly"],
                           "Определяет форму области типичных строк.")],
                  {"decision_function": True, "fit_predict": True}, max_rows=20000)
        self._add("sgd_one_class_svm", "Одноклассовый SVM · SGD", "anomaly", task,
                  "sklearn.linear_model.SGDOneClassSVM", "Обучает линейную область типичных строк небольшими обновлениями.",
                  [_number("nu", "Допуск необычных строк ν", 0.1, 0.001, 0.999,
                           "Задает компромисс между шириной области и долей выбросов.")],
                  {"decision_function": True, "fit_predict": True, "partial_fit": True, "coefficients": True})
        self._add("elliptic_envelope", "Эллипсоид типичных строк", "anomaly", task,
                  "sklearn.covariance.EllipticEnvelope", "Оценивает устойчивый центр и ковариацию почти нормальных числовых данных.",
                  [contamination], {"decision_function": True, "fit_predict": True, "requires_dense": True}, max_rows=10000)

    def _add_reduction(self):
        task = ("reduction",)
        dimensions = _integer("n_components", "Количество новых координат", 2, 1, 256,
                              "Размер полученного представления. Допустимый максимум также зависит от формы данных.")
        caps = {"predict": False, "transform": True, "requires_dense": True}
        for class_name, identifier, description in (
            ("PCA", "pca", "Сохраняет направления с наибольшим разбросом числовых признаков."),
            ("IncrementalPCA", "incremental_pca", "Оценивает главные компоненты по последовательным пакетам строк."),
            ("FactorAnalysis", "factor_analysis", "Описывает признаки меньшим числом скрытых факторов и отдельным шумом."),
        ):
            self._add(identifier, class_name, "decomposition", task, "sklearn.decomposition." + class_name,
                      description, [dimensions], {**caps, "partial_fit": class_name == "IncrementalPCA"})
        self._add("truncated_svd", "Усеченное SVD", "decomposition", task,
                  "sklearn.decomposition.TruncatedSVD", "Разлагает матрицу без центрирования, поэтому подходит для разреженных счетчиков.",
                  [dimensions], {**caps, "requires_dense": False})
        self._add("fast_ica", "Независимые компоненты · ICA", "decomposition", task,
                  "sklearn.decomposition.FastICA", "Ищет компоненты, которые приближенно независимы друг от друга.",
                  [dimensions, _integer("max_iter", "Предел шагов", 500, 10, 5000,
                                         "Максимум уточнений независимых компонент.")], caps)
        self._add("nmf", "Неотрицательное разложение · NMF", "decomposition", task,
                  "sklearn.decomposition.NMF", "Представляет неотрицательные признаки как суммы неотрицательных компонент.",
                  [dimensions, _integer("max_iter", "Предел шагов", 400, 10, 5000,
                                         "Максимум уточнений неотрицательного разложения.")],
                  {**caps, "input_domain": "nonnegative", "requires_dense": False}, {"init": "nndsvda"})
        self._add("kernel_pca", "Ядерный PCA", "decomposition", task,
                  "sklearn.decomposition.KernelPCA", "Выделяет компоненты в нелинейном пространстве ядра.",
                  [dimensions, _choice("kernel", "Ядро", "rbf", ["linear", "poly", "rbf", "sigmoid", "cosine"],
                                       "Способ оценивать сходство двух строк перед разложением.")],
                  {**caps, "requires_dense": False}, max_rows=5000)
        for class_name, identifier, title in (("Isomap", "isomap", "Isomap"),
                                              ("LocallyLinearEmbedding", "locally_linear_embedding", "Локально линейное вложение")):
            self._add(identifier, title, "manifold", task, "sklearn.manifold." + class_name,
                      "Сохраняет отношения ближайших соседей в представлении меньшей размерности.",
                      [dimensions, _integer("n_neighbors", "Количество соседей", 5, 2, 100,
                                             "Размер локального окружения для построения вложения.")], caps, max_rows=5000)
        self._add("spectral_embedding", "Спектральное вложение", "manifold", task,
                  "sklearn.manifold.SpectralEmbedding", "Разлагает граф соседства в новые координаты.",
                  [dimensions], {**caps, "transform": False, "transductive": True}, max_rows=5000)
        self._add("tsne", "t-SNE · исследование данных", "manifold", task, "sklearn.manifold.TSNE",
                  "Строит карту локальных соседств. Расстояния между далекими группами не следует читать как исходные расстояния.",
                  [_integer("n_components", "Количество координат карты", 2, 2, 3,
                            "Карта имеет 2 или 3 координаты."),
                   _number("perplexity", "Размер учитываемого окружения", 30.0, 1.0, 200.0,
                           "Задает эффективный размер окружения и должно быть меньше числа строк."),
                   _integer("max_iter", "Предел шагов", 1000, 250, 5000,
                            "Включает начальную фазу оптимизации. Библиотека требует минимум 250 шагов.")],
                  {**caps, "transform": False, "transductive": True}, max_rows=5000)
        for class_name, identifier in (("GaussianRandomProjection", "gaussian_random_projection"),
                                       ("SparseRandomProjection", "sparse_random_projection")):
            self._add(identifier, class_name, "projection", task, "sklearn.random_projection." + class_name,
                      "Умножает признаки на случайную матрицу для быстрого уменьшения числа координат.",
                      [dimensions], {**caps, "requires_dense": False})

    def _register_boosting(self):
        params = [
            _integer("n_estimators", "Количество деревьев", 100, 1, 2000,
                     "Количество последовательно добавляемых деревьев."),
            _number("learning_rate", "Доля исправления за дерево", 0.1, 0.001, 1,
                    "Множитель вклада очередного дерева."),
            _integer("max_depth", "Глубина дерева", 6, 1, 16,
                     "Предел сложности одного дерева."),
        ]
        for library, prefix, title in (("xgboost", "XGB", "XGBoost"),
                                       ("lightgbm", "LGBM", "LightGBM"),
                                       ("catboost", "CatBoost", "CatBoost")):
            for kind, tasks in (("Regressor", REGRESSION_TASKS), ("Classifier", ("classification",)),
                                 ("Ranker", ("ranking",))):
                suffix = kind.lower()
                fields = deepcopy(params)
                if library == "catboost":
                    for field in fields:
                        if field["key"] == "n_estimators":
                            field["key"] = "iterations"
                        elif field["key"] == "max_depth":
                            field["key"] = "depth"
                    fields.append(_number("l2_leaf_reg", "Ограничение ответов листьев", 3.0, 0, 100000,
                                          "Большее значение сильнее ограничивает числовые ответы листьев дерева."))
                elif library == "lightgbm":
                    fields.append(_integer("min_child_samples", "Минимум строк в листе", 20, 1, 10000,
                                           "Лист требует не меньше этого числа строк. На маленьком обучении слишком большой минимум запрещает все разделения."))
                else:
                    fields.append(_number("reg_lambda", "Ограничение ответов листьев", 1.0, 0, 100000,
                                          "Квадратичное ограничение числовых ответов листьев дерева."))
                constants = {"tree_method": "hist", "device": "cpu"} if library == "xgboost" else (
                    {"verbosity": -1} if library == "lightgbm" else {"verbose": False, "allow_writing_files": False})
                if kind == "Ranker":
                    constants["objective" if library != "catboost" else "loss_function"] = {
                        "xgboost": "rank:ndcg", "lightgbm": "lambdarank", "catboost": "YetiRank"}[library]
                self._add(f"{library}_{suffix}", f"{title} · " + {
                    "Regressor": "регрессия", "Classifier": "классификация", "Ranker": "ранжирование"}[kind],
                    "boosting", tasks, f"{library}.{prefix}{kind}",
                    "Обучает ансамбль деревьев средствами отдельной библиотеки бустинга.", fields,
                    {"predict_proba": kind == "Classifier", "feature_importance": True,
                     "allows_missing": library != "catboost", "ranking": kind == "Ranker"}, constants,
                    lesson_id="ranking-groups" if kind == "Ranker" else "boosting")

    def _legacy_descriptor(self, algorithm_id):
        item = self._legacy.spec(algorithm_id)
        family = ("robust" if algorithm_id in {"huber", "ransac", "theilsen"} else
                  "linear" if algorithm_id in {"ols", "nonnegative", "sgd", "lars", "omp", "linear_svr"} else "regularization")
        item.update({"tasks": list(REGRESSION_TASKS), "family": family,
                     "family_label": FAMILY_LABELS[family], "library": "sklearn",
                     "available": True, "reason": None,
                     "capabilities": {"predict": True, "predict_proba": False,
                                      "decision_function": False, "fit_predict": False,
                                      "partial_fit": algorithm_id == "sgd", "staged_predict": False,
                                      "coefficients": True, "feature_importance": False,
                                      "transductive": False, "input_domain": "real",
                                      "requires_dense": True, "allows_missing": False,
                                      "native_categorical": False, "transform": False,
                                      "target_domain": "positive" if algorithm_id == "gamma" else
                                      "nonnegative" if algorithm_id == "poisson" else "real"}})
        item["lesson_id"] = {
            "ols": "03-least-squares", "nonnegative": "03-least-squares",
            "ridge": "06-ridge", "lasso": "07-lasso", "lassolars": "11-lars-omp",
            "elasticnet": "08-elasticnet", "lars": "11-lars-omp", "omp": "11-lars-omp",
            "sgd": "10-sgd", "huber": "12-huber", "ransac": "13-ransac-theilsen",
            "theilsen": "13-ransac-theilsen", "bayesian_ridge": "14-bayesian", "ard": "14-bayesian",
            "quantile": "15-quantile", "poisson": "16-glm", "gamma": "16-glm", "tweedie": "16-glm",
            "adaptive_lasso": "17-structured-prior", "tikhonov": "17-structured-prior",
            "group_lasso": "18-group-fused", "sparse_group_lasso": "18-group-fused", "fused_lasso": "18-group-fused",
            "l0": "19-nonconvex", "scad": "19-nonconvex", "mcp": "19-nonconvex", "sqrt_lasso": "19-nonconvex",
            "linear_svr": "31-linear-svr", "weighted_lasso": "32-weighted-lasso",
        }.get(algorithm_id, "03-least-squares")
        for field in item["params"]:
            field["help_key"] = f"model.{algorithm_id}.{field['key']}"
            field["lesson_id"] = item["lesson_id"]
        return item

    def descriptor(self, algorithm_id: str) -> dict:
        if not isinstance(algorithm_id, str) or not algorithm_id or len(algorithm_id) > 100:
            raise ValueError("Идентификатор алгоритма должен быть непустой строкой до 100 символов.")
        if algorithm_id not in self._entries:
            return self._legacy_descriptor(algorithm_id)
        item = deepcopy(self._entries[algorithm_id].metadata)
        library = item["library"]
        installed = util.find_spec(library) is not None
        if installed and library != "sklearn" and library not in self._optional_checked:
            try:
                import_module(library)
            except (ImportError, OSError) as exc:
                self._import_failures[library] = f"Библиотека {library} не загрузилась: {exc}. Проверьте нативные зависимости."
            self._optional_checked.add(library)
        failure = self._import_failures.get(library)
        item["available"] = installed and failure is None
        item["reason"] = failure or (None if installed else f"Установите дополнение {library} для этого алгоритма.")
        return item

    def spec(self, algorithm_id: str) -> dict:
        return self.descriptor(algorithm_id)

    def list(self, task: str | None = None) -> list[dict]:
        if task is not None and task not in TASKS:
            raise ValueError(f"Неизвестная задача: {task}.")
        result = [self._legacy_descriptor(item["id"]) for item in self._legacy.catalogue()]
        result.extend(self.descriptor(algorithm_id) for algorithm_id in self._entries)
        return [item for item in result if task is None or task in item["tasks"]]

    def catalogue(self) -> list[dict]:
        return self.list()

    def list_task(self, task: str) -> list[dict]:
        return self.list(task)

    def build(self, task: str, algorithm_id: str, params: dict | None = None,
              seed: int = 42, n_jobs: int = 1):
        return self._build(task, algorithm_id, params, seed, n_jobs, (), 0)

    def _build(self, task, algorithm_id, params, seed, n_jobs, ancestors, depth):
        if task not in TASKS:
            raise ValueError(f"Неизвестная задача: {task}.")
        if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed <= 2**32 - 1:
            raise ValueError("Seed должен быть целым числом от 0 до 4294967295.")
        if isinstance(n_jobs, bool) or not isinstance(n_jobs, Integral) or not 1 <= n_jobs <= 8:
            raise ValueError("Количество потоков должно быть целым числом от 1 до 8.")
        if depth > 3:
            raise ValueError("Ансамбль превышает три уровня вложенности.")
        spec = self.descriptor(algorithm_id)
        if task not in spec["tasks"]:
            raise ValueError(f"Алгоритм {spec['name']} не поддерживает задачу {task}.")
        if not spec["available"]:
            raise ValueError(spec["reason"])
        validated = self._validate(spec, params)
        if algorithm_id not in self._entries:
            estimator = self._legacy.create(algorithm_id, validated, seed=int(seed))
            if "n_jobs" in estimator.get_params(deep=False):
                estimator.set_params(n_jobs=int(n_jobs))
            return estimator
        entry = self._entries[algorithm_id]
        module_name, class_name = entry.class_path.rsplit(".", 1)
        try:
            cls = getattr(import_module(module_name), class_name)
        except (ImportError, OSError) as exc:
            reason = f"Библиотека {spec['library']} не загрузилась: {exc}. Проверьте установку нативных зависимостей."
            self._import_failures[spec["library"]] = reason
            raise ValueError(reason) from exc
        constructor = self._normalize(entry, validated)
        if entry.factory == "ensemble":
            constructor = self._ensemble_params(task, algorithm_id, validated, seed, ancestors, depth)
        elif entry.factory == "gaussian_process":
            from sklearn.gaussian_process.kernels import DotProduct, Matern, RBF
            constructor["kernel"] = {"rbf": RBF, "matern": Matern, "dot": DotProduct}[validated["kernel"]]()
        elif entry.factory == "mlp":
            constructor["hidden_layer_sizes"] = (constructor.pop("hidden_units"),)
        if spec["library"] == "catboost":
            constructor.update(random_seed=int(seed), thread_count=int(n_jobs))
        else:
            supported = cls(**constructor).get_params(deep=False)
            if "random_state" in supported:
                constructor["random_state"] = int(seed)
            if "n_jobs" in supported:
                constructor["n_jobs"] = int(n_jobs)
        return cls(**constructor)

    @staticmethod
    def _normalize(entry: _Algorithm, validated: dict) -> dict:
        result = deepcopy(validated)
        if result.get("max_depth") == 0:
            result["max_depth"] = None
        if result.get("max_features") == "all":
            result["max_features"] = 1.0
        if result.get("class_weight") == "none":
            result["class_weight"] = None
        result.update(deepcopy(entry.constants))
        return result

    @staticmethod
    def _validate(spec: dict, params: dict | None) -> dict:
        if params is None:
            params = {}
        if not isinstance(params, dict):
            raise ValueError("Параметры модели должны быть объектом.")
        fields = {field["key"]: field for field in spec["params"]}
        unknown = set(params) - fields.keys()
        if unknown:
            raise ValueError("Неизвестные параметры модели: " + ", ".join(sorted(map(str, unknown))))
        values = {key: deepcopy(field["default"]) for key, field in fields.items()}
        values.update(deepcopy(params))
        for key, value in values.items():
            field = fields[key]
            kind = field["type"]
            label = field["label"]
            if kind == "bool" and not isinstance(value, bool):
                raise ValueError(f"{label}: требуется true или false.")
            if kind == "select":
                matches = any(type(value) is type(option) and value == option for option in field["options"])
                if not matches:
                    raise ValueError(f"{label}: выберите допустимый вариант.")
            if kind in {"int", "float"}:
                numeric = isinstance(value, Integral if kind == "int" else Real)
                if isinstance(value, bool) or not numeric or not np.isfinite(value):
                    raise ValueError(f"{label}: требуется конечное {'целое ' if kind == 'int' else ''}число.")
                if not field.get("min", -np.inf) <= value <= field.get("max", np.inf):
                    raise ValueError(f"{label}: значение вне допустимого диапазона.")
                values[key] = int(value) if kind == "int" else float(value)
            if kind == "models":
                if not isinstance(value, list) or len(value) > field["max_items"]:
                    raise ValueError(f"{label}: требуется список не более {field['max_items']} моделей.")
                for component in value:
                    if not isinstance(component, dict) or set(component) - {"algorithm_id", "params"}:
                        raise ValueError("Компонент ансамбля должен содержать algorithm_id и params.")
                    if component.get("algorithm_id") not in field["options"]:
                        raise ValueError("Этот алгоритм не разрешен в компонентах ансамбля.")
                    if "params" in component and not isinstance(component["params"], dict):
                        raise ValueError("Параметры компонента должны быть объектом.")
        return values

    def _ensemble_params(self, task, algorithm_id, validated, seed, ancestors, depth):
        classifier = task == "classification"
        suffix = "classifier" if classifier else "regressor"
        linear = "logistic_regression" if classifier else "ridge"
        defaults = {
            "tree_forest_linear": [f"decision_tree_{suffix}", f"random_forest_{suffix}", linear],
            "forest_boosting": [f"random_forest_{suffix}", f"hist_gradient_boosting_{suffix}"],
            "neighbors_linear": [f"knn_{suffix}", linear],
        }
        components = validated["components"] or [
            {"algorithm_id": item, "params": {}} for item in defaults[validated["base_estimators"]]
        ]
        if len(components) < 2:
            raise ValueError("Для ансамбля нужны хотя бы две базовые модели.")
        estimators = []
        for index, component in enumerate(components):
            estimator = self._build(task, component["algorithm_id"], component.get("params", {}),
                                    (int(seed) + index) % 2**32, 1, (*ancestors, algorithm_id), depth + 1)
            if validated.get("voting") == "soft" and not hasattr(estimator, "predict_proba"):
                raise ValueError("Мягкое голосование требует вероятностей от каждого компонента.")
            estimators.append((f"model_{index + 1}", estimator))
        result = {"estimators": estimators}
        if algorithm_id.startswith("voting_"):
            if classifier:
                result["voting"] = validated["voting"]
        else:
            result["cv"] = validated["cv"]
            result["final_estimator"] = self._build(task, linear, {}, seed, 1, (), depth + 1)
        return result


def build(task: str, algorithm_id: str, params: dict | None = None, seed: int = 42, n_jobs: int = 1):
    """Convenience factory for callers which do not retain a catalogue instance."""
    return AlgorithmCatalogue().build(task, algorithm_id, params, seed, n_jobs)
