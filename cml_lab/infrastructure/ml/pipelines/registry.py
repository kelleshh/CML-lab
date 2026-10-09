"""Explicit constructors and references supported by the declarative editor.

The names here are library names, not lab-specific aliases. Imports in user
definitions are resolved against this table and are never executed.
"""

from __future__ import annotations

import importlib
import inspect
import math
from functools import lru_cache

import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from ...hyperparameter_learning import PIPELINE_CLASS_KNOWLEDGE, parameter_knowledge


MODULES = {
    "sklearn.pipeline": ("Pipeline", "FeatureUnion", "make_pipeline", "make_union"),
    "sklearn.compose": ("ColumnTransformer", "make_column_transformer", "make_column_selector"),
    "sklearn.preprocessing": (
        "StandardScaler", "MinMaxScaler", "MaxAbsScaler", "RobustScaler", "Normalizer",
        "Binarizer", "OneHotEncoder", "OrdinalEncoder", "TargetEncoder", "PolynomialFeatures",
        "SplineTransformer", "PowerTransformer", "QuantileTransformer", "FunctionTransformer",
    ),
    "sklearn.impute": ("SimpleImputer", "KNNImputer", "IterativeImputer", "MissingIndicator"),
    "sklearn.decomposition": (
        "PCA", "IncrementalPCA", "KernelPCA", "SparsePCA", "MiniBatchSparsePCA", "TruncatedSVD",
        "FastICA", "NMF", "MiniBatchNMF", "FactorAnalysis", "DictionaryLearning",
        "MiniBatchDictionaryLearning", "SparseCoder", "LatentDirichletAllocation",
    ),
    "sklearn.feature_selection": (
        "VarianceThreshold", "SelectKBest", "SelectPercentile", "SelectFpr", "SelectFdr",
        "SelectFwe", "GenericUnivariateSelect", "SelectFromModel", "RFE", "RFECV",
        "SequentialFeatureSelector", "chi2", "f_classif", "f_regression", "mutual_info_classif",
        "mutual_info_regression", "r_regression",
    ),
    "sklearn.feature_extraction": ("DictVectorizer", "FeatureHasher"),
    "sklearn.feature_extraction.text": ("CountVectorizer", "TfidfVectorizer", "HashingVectorizer", "TfidfTransformer"),
    "sklearn.kernel_approximation": ("RBFSampler", "Nystroem", "AdditiveChi2Sampler", "SkewedChi2Sampler", "PolynomialCountSketch"),
    "sklearn.random_projection": ("GaussianRandomProjection", "SparseRandomProjection"),
    "sklearn.manifold": ("Isomap", "LocallyLinearEmbedding"),
    "sklearn.cross_decomposition": ("PLSRegression", "PLSCanonical", "CCA"),
    # These estimators can be nested in SelectFromModel/RFE/IterativeImputer.
    "sklearn.linear_model": ("LinearRegression", "Ridge", "Lasso", "ElasticNet", "BayesianRidge", "LogisticRegression", "SGDClassifier", "SGDRegressor"),
    "sklearn.ensemble": ("RandomForestClassifier", "RandomForestRegressor", "ExtraTreesClassifier", "ExtraTreesRegressor", "GradientBoostingClassifier", "GradientBoostingRegressor"),
    "sklearn.tree": ("DecisionTreeClassifier", "DecisionTreeRegressor"),
    "sklearn.svm": ("LinearSVC", "LinearSVR"),
}

REFERENCE_MODULES = {
    "builtins": {"object": object, "str": str, "int": int, "float": float, "bool": bool},
    "numpy": {
        "log1p": np.log1p, "log": np.log, "log2": np.log2, "log10": np.log10,
        "exp": np.exp, "expm1": np.expm1, "sqrt": np.sqrt, "square": np.square,
        "abs": np.abs, "absolute": np.absolute, "sign": np.sign, "sin": np.sin,
        "cos": np.cos, "tan": np.tan, "nan": np.nan, "inf": np.inf,
        "float32": np.float32, "float64": np.float64, "int32": np.int32,
        "int64": np.int64, "number": np.number, "integer": np.integer,
        "floating": np.floating, "bool_": np.bool_, "array": np.array,
    }
}

HELP = {
    "steps": "Последовательность именованных преобразований. Выход предыдущего шага становится входом следующего. Все шаги здесь должны уметь transform; обучение модели выполняется отдельно.",
    "transformers": "Ветки (name, transformer, columns). Каждая ветка получает указанные исходные столбцы; результаты объединяются по столбцам. Одну фичу можно направить в несколько веток.",
    "transformer_list": "Параллельные ветки (name, transformer). Все получают один и тот же вход; их выходы соединяются по столбцам.",
    "remainder": "drop удаляет невыбранные столбцы, passthrough передает их без изменения; преобразователь обучается на оставшихся столбцах. Новые столбцы после fit не добавляются.",
    "memory": "Кеширование промежуточных шагов sklearn. В лаборатории допустимо только None: пути для файлового кеша не входят в декларативный формат.",
    "transform_input": "Имена дополнительных аргументов, преобразуемых через metadata routing. В лаборатории дополнительные метаданные не передаются; оставьте None.",
    "n_jobs": "Число параллельных задач sklearn: None или 1 последовательно, положительное число ограничивает задачи, -1 использует доступные ядра. Учитывайте параллельность внешнего поиска.",
    "verbose": "Показывает ход работы преобразователя в журнале процесса; на саму математику не влияет.",
    "verbose_feature_names_out": "Добавляет имя ветки к имени выходной фичи. Помогает различать одинаковые столбцы из разных веток.",
    "transformer_weights": "Числовые множители для выходов именованных веток. Масштаб ветки влияет на расстояния и регуляризацию модели.",
    "sparse_threshold": "Если доля ненулевых элементов объединенного результата ниже порога, ColumnTransformer возвращает sparse matrix. 0 всегда дает плотный результат, который может занимать больше памяти.",
    "copy": "True создает копию входа, когда это возможно. False позволяет изменять вход на месте; библиотека все равно может создать копию.",
    "with_mean": "Вычитает среднее train из каждого столбца. Для sparse matrix должно быть False: центрирование разрушает разреженность.",
    "with_std": "Делит столбец на стандартное отклонение train. Столбец с нулевой дисперсией остается с масштабом 1.",
    "feature_range": "Пара (минимум, максимум) выходного диапазона MinMaxScaler для значений train. Новые значения могут выйти за диапазон без clip.",
    "clip": "Ограничивает новые значения диапазоном feature_range. Это срезает величину выхода за обученный диапазон и теряет часть информации.",
    "with_centering": "Вычитает медиану train. Для sparse matrix центрирование недопустимо.",
    "with_scaling": "Делит значения на выбранный межквантильный размах train.",
    "quantile_range": "Нижний и верхний процентили для RobustScaler, например (25, 75). Ширина диапазона определяет масштаб.",
    "unit_variance": "Корректирует масштаб RobustScaler так, чтобы нормальное распределение имело дисперсию около 1.",
    "norm": "Норма каждой строки: l1 — сумма модулей, l2 — квадратный корень суммы квадратов, max — максимальный модуль. Это нормировка строк, не столбцов.",
    "threshold": "Порог зависит от класса: Binarizer сравнивает значение с порогом; VarianceThreshold удаляет низкую дисперсию; SelectFromModel сравнивает важность. Смотрите раздел конкретного преобразователя.",
    "missing_values": "Значение, которое считается пропуском. Обычно numpy.nan; None и строки имеют другой смысл и должны соответствовать данным.",
    "strategy": "Алгоритм преобразования. SimpleImputer: mean, median, most_frequent, constant; KBinsDiscretizer: uniform, quantile, kmeans. Статистики вычисляются только при fit на train.",
    "fill_value": "Константа для пропуска при strategy='constant'. Не используется при mean или median.",
    "add_indicator": "Добавляет индикаторы исходных пропусков. Модель получает заполненное значение и отдельный факт отсутствия.",
    "keep_empty_features": "Сохраняет столбцы, полностью пустые на train. Без этого они могут исчезнуть из выходной матрицы.",
    "n_neighbors": "Число соседних обучающих строк, используемых для заполнения пропуска или построения локальной геометрии.",
    "weights": "Правило веса соседей: uniform дает одинаковый вес, distance увеличивает вес близких строк.",
    "metric": "Мера расстояния. У KNNImputer nan_euclidean учитывает совместно наблюдаемые координаты.",
    "categories": "auto изучает категории train; список задает допустимые категории каждого столбца. Порядок в OrdinalEncoder задает числовые коды.",
    "drop": "Убирает категорию из OneHotEncoder (first/if_binary или список). Меняет число фич и смысл базовой категории.",
    "handle_unknown": "Правило для категории, отсутствовавшей на train: ошибка, игнорирование или отдельный код — доступные варианты зависят от encoder.",
    "unknown_value": "Числовой код новой категории у OrdinalEncoder при handle_unknown='use_encoded_value'. Должен отличаться от кодов известных категорий.",
    "encoded_missing_value": "Числовой код пропуска у OrdinalEncoder. Он не обязан совпадать с кодом неизвестной категории.",
    "sparse_output": "True хранит выход разреженно; False хранит все ячейки. Для one-hot плотный результат может быть очень большим.",
    "min_frequency": "Категории реже этого количества или доли train объединяются в группу infrequent.",
    "max_categories": "Максимум выходных категорий на исходный столбец, включая infrequent. Ограничивает ширину кодирования.",
    "dtype": "Тип чисел выходной матрицы, например numpy.float32 или numpy.float64. Меняет точность и объем памяти.",
    "degree": "Степень полинома либо пара (минимальная, максимальная). Для x,z степень 2 добавляет x², xz, z²; число фич быстро растет.",
    "interaction_only": "Создает произведения разных фич, без степеней одной фичи: xz остается, x² и z² исчезают.",
    "include_bias": "Добавляет постоянный столбец 1. Часто он не нужен, когда модель уже использует fit_intercept.",
    "order": "Порядок хранения плотного массива C или F. Меняет размещение в памяти, а не значения фич.",
    "n_knots": "Число узлов SplineTransformer. Больше узлов создает больше локальных базисных фич и повышает гибкость.",
    "knots": "uniform размещает узлы между min/max train равномерно, quantile — по квантилям train; массив задает координаты явно.",
    "extrapolation": "Поведение сплайна вне train диапазона: error, constant, linear, continue или periodic. Выбирайте по смыслу фичи.",
    "method": "Конкретное преобразование: PowerTransformer использует yeo-johnson или box-cox. Box-Cox требует строго положительные значения.",
    "standardize": "После степенного преобразования центрирует и масштабирует по train статистикам.",
    "n_quantiles": "Количество квантилей эмпирического распределения train. Не больше числа обучающих строк; больше значений дает более подробную шкалу.",
    "output_distribution": "uniform переводит квантиль в [0,1], normal — в координату нормального распределения. Это нелинейно меняет расстояния.",
    "subsample": "Число train строк для оценки распределения; None использует все. Ускоряет fit больших данных и может немного менять результат.",
    "random_state": "Фиксирует случайные решения при одинаковых данных и версии библиотеки. Это seed, не гарантия одинаковых результатов между версиями.",
    "func": "Разрешенная функция numpy для FunctionTransformer, например numpy.log1p. Функция применяется без обучения; log требует положительных значений.",
    "inverse_func": "Обратная функция numpy, например numpy.expm1 для log1p. Нужна для inverse_transform, не для обучения модели.",
    "kw_args": "Словарь дополнительных аргументов func. Допустимы только литералы декларативного формата.",
    "inv_kw_args": "Словарь дополнительных аргументов inverse_func.",
    "validate": "Проверяет и преобразует вход FunctionTransformer. False сохраняет тип входа; True требует числовую 2D матрицу.",
    "accept_sparse": "Позволяет sparse matrix при проверке входа FunctionTransformer.",
    "check_inverse": "Проверяет на части train, что func и inverse_func обращают друг друга. Включенная проверка не делает неизвестную функцию разрешенной.",
    "feature_names_out": "Правило имен выходных фич. Для FunctionTransformer 'one-to-one' сохраняет имена; None не предоставляет имена.",
    "n_components": "Количество выходных компонентов или библиотечное правило выбора. Компоненты изучаются на train; ограничения зависят от метода и размеров матрицы.",
    "whiten": "Приводит компоненты к единичной дисперсии при PCA/ICA. Это убирает различия масштабов компонентов и может усиливать шум.",
    "svd_solver": "Алгоритм разложения PCA. full точнее работает со всей матрицей; randomized оценивает ограниченное число компонентов; auto выбирает по размеру.",
    "algorithm": "Алгоритм вычисления конкретного преобразователя. Доступные строки перечислены в типе параметра из sklearn.",
    "tol": "Порог остановки итерационного метода. Меньше значение обычно требует больше итераций.",
    "max_iter": "Максимум итераций; при достижении предела сходимость не гарантирована. Больше итераций увеличивает время.",
    "n_iter": "Число итераций аппроксимации или оптимизации. Большое число повышает вычислительную стоимость.",
    "batch_size": "Число строк в одном пакете итерационного обучения. Меняет память и частоту обновления состояния.",
    "score_func": "Статистика связи каждой фичи с y, например f_regression, f_classif или chi2. Она вычисляется только на train конкретного fold.",
    "k": "Число лучших фич SelectKBest или 'all'. Не должно превышать число входных фич train.",
    "percentile": "Процент сохраняемых фич от 0 до 100. Оценки отдельных фич изучаются на train.",
    "alpha": "Смысл зависит от класса: уровень значимости статистического отбора либо сила L1/L2 регуляризации вложенной модели. Смотрите тип и урок конкретного класса.",
    "estimator": "Вложенная модель, используемая для отбора фич или заполнения пропусков. Она обучается внутри fit преобразователя, не на test.",
    "prefit": "Требует уже обученный estimator. Декларативная лаборатория принимает только False, чтобы отбор обучался внутри train fold.",
    "importance_getter": "auto использует coef_ или feature_importances_; строка может указать путь к атрибуту вложенной модели. Пользовательские callable не исполняются.",
    "max_features": "Максимум фич для отбора или верхняя граница словаря текста; у вложенного леса это число проверяемых фич в узле. Смысл зависит от класса.",
    "n_features_to_select": "Количество сохраняемых фич при последовательном отборе; int задает число, float — долю.",
    "step": "Количество или доля фич, удаляемых за шаг RFE. Большое значение ускоряет поиск и делает его грубее.",
    "cv": "Число внутренних folds. Обычно shuffle/folds по умолчанию не сохраняют время и группы; для таких данных эти преобразователи ограничены лабораторией.",
    "smooth": "Сглаживание TargetEncoder между средним категории и общим средним train; auto оценивает величину из данных.",
    "target_type": "Тип y для TargetEncoder: continuous, binary, multiclass либо auto. Кодирование на fit_transform использует внутреннюю перекрестную проверку.",
    "shuffle": "Перемешивает строки внутренних folds. Нельзя применять обычное случайное смешивание для временного прогноза.",
    "vocabulary": "Явный словарь токен -> индекс. None изучает словарь только по train документам.",
    "analyzer": "word строит токены слов, char — символов, char_wb — символов внутри границ слова. Изменяет смысл текстовых фич.",
    "ngram_range": "Минимум и максимум длины последовательности токенов. (1,2) включает отдельные слова и пары слов; расширение резко увеличивает словарь.",
    "lowercase": "Переводит текст в нижний регистр перед токенизацией. Может стереть полезные различия аббревиатур.",
    "stop_words": "Слова, исключаемые из словаря. 'english' — встроенный английский список; для русского нужен явный список.",
    "token_pattern": "Регулярное выражение выделения токенов в word analyzer. Оно влияет, например, на односимвольные слова.",
    "min_df": "Минимум документов train с токеном: int — количество, float — доля. Удаляет редкие токены.",
    "max_df": "Максимум документов train с токеном: int — количество, float — доля. Удаляет слишком частые токены.",
    "binary": "Заменяет счетчик наличием 0/1. Частота повторения токена перестает влиять на фичу.",
    "use_idf": "Уменьшает влияние токенов, частых во многих train документах, через inverse document frequency.",
    "smooth_idf": "Добавляет единицу к частотам IDF, чтобы избежать деления на ноль.",
    "sublinear_tf": "Меняет положительную частоту токена на 1+log(tf). Уменьшает влияние многократных повторов.",
    "n_features": "Число выходных hash фич. Коллизии соединяют разные исходные значения; увеличение снижает коллизии и расходует память.",
    "alternate_sign": "Использует положительные и отрицательные hash вклады. Отрицательные значения несовместимы с некоторыми моделями.",
    "input_type": "Тип записи FeatureHasher: dict, pair или string. Должен совпадать с тем, что подает предыдущий шаг.",
    "gamma": "Масштаб kernel расстояния: большое значение дает более локальные сходства. Для RBF это exp(-gamma*distance²).",
    "kernel": "Функция сходства для kernel метода. Доступные имена и условия входа указаны в документации класса.",
    "degree": "Максимальная степень PolynomialFeatures либо степень polynomial kernel. В обоих случаях увеличение повышает гибкость и вычислительную стоимость.",
}

CLASS_HELP = {
    "Pipeline": "Последовательная композиция преобразований sklearn. Каждый шаг получает выход предыдущего; можно вложить Pipeline внутрь Pipeline.",
    "ColumnTransformer": "Маршрутизирует исходные столбцы в разные преобразователи и соединяет их результаты. Подходит для отдельных веток чисел, категорий и текста.",
    "FeatureUnion": "Передает весь вход всем веткам одновременно, затем соединяет результаты. В отличие от ColumnTransformer, сам не выбирает столбцы.",
    "StandardScaler": "Вычитает среднее train и делит на стандартное отклонение train. Обычно нужен для расстояний и регуляризации.",
    "MinMaxScaler": "Линейно переводит обученный min/max каждого столбца в feature_range. Новые значения могут выйти за диапазон.",
    "RobustScaler": "Центрирует по медиане и масштабирует по квантилям train; меньше зависит от крайних значений, чем StandardScaler.",
    "OneHotEncoder": "Создает отдельные числовые индикаторы категорий. Категории учатся на train; новые категории обрабатываются через handle_unknown.",
    "OrdinalEncoder": "Заменяет каждую категорию кодом. Числа могут создать ложное ощущение порядка и расстояния.",
    "TargetEncoder": "Кодирует категории сглаженной статистикой y. fit_transform использует cross-fitting; для временных и групповых данных обычные внутренние folds непригодны.",
    "SimpleImputer": "Заполняет пропуски средним, медианой, частым значением train либо константой.",
    "KNNImputer": "Заполняет пропуски по соседним обучающим строкам, учитывая совместно наблюдаемые координаты.",
    "IterativeImputer": "По очереди предсказывает отсутствующий столбец по другим. Экспериментальный sklearn API; вложенная модель и все статистики обучаются на train.",
    "PolynomialFeatures": "Создает степени и взаимодействия входных числовых фич. Расширение может многократно увеличить матрицу.",
    "SplineTransformer": "Создает кусочно-полиномиальные базисные фичи вокруг обученных узлов. Полезен для плавной нелинейности.",
    "FunctionTransformer": "Применяет выбранную функцию numpy к входу. В лаборатории разрешены известные функции; произвольный Python код не исполняется.",
    "PCA": "Заменяет числовые фичи компонентами максимальной дисперсии train. Компоненты ортогональны, но не обязаны быть полезными для y.",
    "TruncatedSVD": "Сжимает матрицу без центрирования; подходит для sparse текстовых фич, в отличие от обычного центрированного PCA.",
    "SelectKBest": "Оставляет k лучших отдельных фич по score_func на train. Отдельная оценка может пропустить взаимодействия фич.",
    "SelectFromModel": "Оставляет фичи с высокой важностью вложенной модели, обученной внутри train fold.",
    "TfidfVectorizer": "Создает словарь train токенов и их TF-IDF веса. Тестовые документы не меняют словарь или IDF.",
    "CountVectorizer": "Создает словарь train токенов и их числовые частоты. Требует одномерный поток документов.",
    "HashingVectorizer": "Кодирует текст через фиксированный hash без обучения словаря. Возможны коллизии; IDF требует отдельного TfidfTransformer.",
}


@lru_cache(maxsize=1)
def constructors():
    result = {}
    for module, names in MODULES.items():
        imported = importlib.import_module(module)
        for name in names:
            value = getattr(imported, name, None)
            if value is not None:
                result[f"{module}.{name}"] = value
    return result


def references():
    result = dict(constructors())
    for module, values in REFERENCE_MODULES.items():
        result.update({f"{module}.{name}": value for name, value in values.items()})
    return result


def qualified_name(value):
    for name, known in references().items():
        if value is known:
            return name
    return None


def _json_default(value):
    if value is inspect.Parameter.empty:
        return None
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {"$ref": "numpy.nan" if math.isnan(value) else "numpy.inf"}
    if isinstance(value, (tuple, list)):
        return [_json_default(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_default(item) for key, item in value.items()}
    reference = qualified_name(value)
    return {"$ref": reference} if reference else repr(value)


def parameter_docs(value):
    """Keep the installed version's exact type and parameter explanation."""
    lines = inspect.getdoc(value).splitlines() if inspect.getdoc(value) else []
    result = {}
    active = None
    for index, line in enumerate(lines):
        if " : " in line and not line.startswith(" "):
            names, type_text = line.split(" : ", 1)
            if all(name.strip().replace("_", "").isalnum() for name in names.split(",")):
                active = (names.split(","), type_text, [])
                for name in active[0]:
                    result[name.strip()] = {"type": type_text, "lines": active[2]}
                continue
        if active and line and not line.startswith(" ") and index + 1 < len(lines) and set(lines[index + 1]) <= {"-"}:
            active = None
        if active:
            active[2].append(line.strip())
    return {name: {"type": item["type"], "doc": "\n".join(item["lines"]).strip()} for name, item in result.items()}


@lru_cache(maxsize=1)
def catalogue_classes():
    result = []
    for qualified, value in constructors().items():
        if not inspect.isclass(value):
            continue
        module, name = qualified.rsplit(".", 1)
        docs = parameter_docs(value)
        parameters = []
        for key, parameter in inspect.signature(value).parameters.items():
            if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
                continue
            info = docs.get(key, {})
            knowledge = parameter_knowledge(f"pipeline-{name.lower()}", "preprocessing", key, "preprocessing", class_name=name, library="sklearn")
            parameters.append({
                "key": key, "name": key, "type": info.get("type", "literal"),
                "default": _json_default(parameter.default),
                "required": parameter.default is inspect.Parameter.empty,
                "description": knowledge.get("summary") or HELP.get(key, f"Параметр {key} класса {name}. Тип, точное значение по умолчанию и ограничения берутся из установленного sklearn; исходное описание приведено ниже в учебнике."),
                "doc": info.get("doc", ""), "lesson_id": f"parameter-pipeline-{name.lower()}-{key.replace('_', '-').lower()}",
                "anchor": f"param-{key}",
            })
        kind = "composition" if name in ("Pipeline", "ColumnTransformer", "FeatureUnion") else ("transformer" if hasattr(value, "transform") else "estimator")
        result.append({
            "name": name, "module": module, "qualified_name": qualified, "kind": kind,
            "description": CLASS_HELP.get(name) or PIPELINE_CLASS_KNOWLEDGE.get(name, {}).get("summary") or f"{name}: преобразователь или вложенная модель sklearn. Параметры и исходники соответствуют установленной версии библиотеки.",
            "doc": inspect.getdoc(value) or "", "params": parameters,
            "lesson_id": f"pipeline-{name.lower()}",
            "source_url": f"https://scikit-learn.org/1.8/modules/generated/{qualified}.html",
        })
    return result
