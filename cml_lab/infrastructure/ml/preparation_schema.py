"""Public GUI metadata for the allowlisted preparation adapters."""

from __future__ import annotations

from copy import deepcopy


ALL_TASKS = ("regression", "classification", "clustering", "ranking", "forecasting", "panel", "anomaly", "reduction")
SUPERVISED_TASKS = ("regression", "classification", "ranking", "forecasting", "panel")
SOURCES = {
    "numeric": "https://scikit-learn.org/1.8/modules/preprocessing.html",
    "impute": "https://scikit-learn.org/1.8/modules/impute.html",
    "selection": "https://scikit-learn.org/1.8/modules/generated/sklearn.feature_selection.SelectKBest.html",
    "reduction": "https://scikit-learn.org/1.8/modules/decomposition.html",
    "text": "https://scikit-learn.org/1.8/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html",
    "sampling": "https://imbalanced-learn.org/stable/over_sampling.html",
}
LESSON_IDS = {
    "prep-columns": "dataset-sources", "prep-numeric": "05-scaling-correlation",
    "prep-missing": "22-missing-values", "prep-interactions": "24-feature-engineering",
    "prep-outliers": "24-feature-engineering", "prep-categories": "categorical-encoding",
    "prep-target-encoding": "categorical-encoding", "prep-text": "text-features",
    "prep-datetime": "date-features", "prep-selection": "feature-selection",
    "prep-reduction": "dimensionality-reduction", "prep-resampling": "classification-resampling",
    "prep-regression-sampling": "28-rare-target-smoter",
}


def _field(key, label, default, *, type="select", options=None, min=None, max=None, help="", lesson="prep-numeric"):
    explanations = {
        "imputation": "Среднее/медиана/частое значение учатся на train; KNN использует соседние обучающие строки. none оставляет пропуски и подходит только совместимым моделям.",
        "scaler": "Меняет масштаб числовых столбцов, чтобы единицы измерения не определяли влияние на расстояния и штрафы. Статистики сохраняются из train.",
        "numeric_transform": "Меняет форму распределения. Signed log1p/sqrt сохраняют знак; Yeo-Johnson обучается на train; квантильное преобразование использует порядок значений train.",
        "degree": "Максимальная степень создаваемых произведений. Степень 2 для x,z создаёт x,z,x²,xz,z². Число столбцов быстро растёт.",
        "interaction_only": "Исключает степени одного признака. Для x,z и степени 2 остаются x,z,xz, без x² и z².",
        "missing_indicator": "Добавляет столбец 0/1, показывающий исходный пропуск. Заполненное значение и факт пропуска становятся разными признаками.",
        "strategy": "Выбирает способ обучения преобразования. Конкретные варианты и пример результата описаны в уроке этого этапа.",
        "fill_value": "Значение вместо числового пропуска при strategy=constant. Оно не оценивается по данным и должно иметь предметный смысл.",
        "n_neighbors": "Сколько обучающих соседей участвуют в заполнении числового пропуска. Большое значение усредняет сильнее и увеличивает работу поиска.",
        "add_indicator": "Добавляет 0/1 для исходного пропуска вместе с заполненным числом. Индикаторы зависят от пропусков, увиденных в train.",
        "method": "Выбирает вычислительный метод этого этапа. Он исполняется в заданном порядке рецепта, на выбранных столбцах.",
        "n_quantiles": "Число опорных квантилей для отображения распределения train в равномерное или нормальное. На малом train фактическое число не больше строк.",
        "lower": "Значения ниже этого квантиля train заменяются граничным значением. Например 0.01 ограничивает нижний 1% train.",
        "upper": "Значения выше этого квантиля train заменяются граничным значением. Например 0.99 ограничивает верхний 1% train.",
        "n_bins": "Число интервалов на каждый числовой столбец. Редкие/совпадающие значения могут привести к меньшему фактическому числу интервалов.",
        "encode": "ordinal возвращает номер интервала; onehot-dense создаёт индикаторы интервалов. Номер интервала не является физическим расстоянием.",
        "handle_unknown": "Поведение категории, отсутствовавшей в train: error останавливает прогноз; ignore даёт нули; infrequent_if_exist использует обученную редкую группу, если она есть.",
        "min_frequency": "Категории с меньшим числом обучающих наблюдений объединяются в редкую группу. Validation и test не меняют частоты.",
        "max_categories": "Верхний предел выходных one-hot столбцов на один исходный столбец, включая редкую группу.",
        "sparse_output": "Хранит ненулевые индикаторы вместо полной матрицы. Сохраняет память; следующая модель должна принимать разреженный вход.",
        "normalize": "Доля категории равна её количеству в train, делённому на число строк train. Без нормировки возвращается количество.",
        "n_features": "Число хеш-корзин. Размер фиксирован; разные исходные значения могут попасть в одну корзину.",
        "alternate_sign": "Разрешает отрицательные значения хеширования для сохранения скалярных произведений. Для MultinomialNB/ComplementNB отключите.",
        "smooth": "Смешивает среднюю цель категории с общей средней train. Большее значение сильнее сглаживает редкие категории.",
        "cv": "Части внутреннего cross-fitting target encoder. Каждая обучающая строка кодируется статистикой, вычисленной без её части.",
        "max_features": "Ограничивает размер выходного словаря или число отобранных признаков, в зависимости от этапа. В словаре остаются наиболее частые обучающие единицы.",
        "analyzer": "word считает слова; char считает последовательности символов; char_wb считает символы внутри границ слова. Английские стоп-слова автоматически не применяются.",
        "ngram_max": "Максимальная длина цепочки слов/символов. Значение 2 добавляет пары наряду с одиночными единицами.",
        "lowercase": "Считает Кошка и кошка одним текстовым значением. Отдельные оригинальные текстовые столбцы не изменяются.",
        "parts": "Календарные числа, извлекаемые из каждой выбранной даты. Ошибки разбора становятся пропусками; год/месяц/час не являются одной общей координатой.",
        "cyclic": "Добавляет синус и косинус месяца/дня недели/часа/минуты/дня года: конец периода располагается рядом с его началом.",
        "timezone": "Часовой пояс календарных признаков. Например Europe/Moscow. Дата с явным смещением сначала переводится в UTC, затем в выбранный пояс.",
        "threshold": "Порог удаления или отбора признаков. Дисперсия и важность вычисляются на train; scale меняет величину дисперсии.",
        "k": "Точное число лучших отдельных признаков по обучающей статистике. Оно не может превышать число подготовленных столбцов train.",
        "n_components": "Число новых координат разложения. Не больше числа строк и признаков train; исходные столбцы заменяются компонентами.",
        "neighbors": "Соседи, участвующие в генерации/очистке обучающих примеров. Для SMOTE число должно быть меньше размера редкого класса в каждом fold.",
        "max_iter": "Максимум проходов IterativeImputer по пропущенным столбцам. Каждый столбец предсказывается по другим; большая величина увеличивает время. Это одно заполнение, не множество возможных датасетов.",
        "n_nearest_features": "Не более этого числа других столбцов участвуют в предсказании пропуска. Они выбираются по связям только внутри train. Ограничивает стоимость итеративного заполнения.",
        "n_knots": "Количество узлов, около которых меняются полиномиальные части сплайна. Больше узлов позволяет гибче описать зависимость, но создает больше признаков.",
        "knots": "quantile размещает узлы по квантилям train; uniform равномерно между минимумом и максимумом train. Проверочные данные не меняют узлы.",
        "extrapolation": "Поведение за пределами обученного диапазона: constant продолжает крайним значением, linear продолжает линейно, error запрещает выход, continue продолжает крайний полином.",
        "discrete_features": "Для MI: continuous означает измеряемые числа; discrete — счетчики/коды. auto считает dense непрерывным, sparse дискретным. Смешанная sparse матрица с числовыми измерениями требует плотного представления или SVD.",
    }
    help = help or explanations.get(key, "Настройка выбранного этапа; результат и ограничения приведены в связанном уроке.")
    result = {"key": key, "label": label, "type": type, "default": default, "help": help,
              "help_key": f"preparation.{key}", "lesson_id": LESSON_IDS.get(lesson, lesson)}
    for name, value in (("options", options), ("min", min), ("max", max)):
        if value is not None:
            result[name] = value
    return result


def _stage(id, name, description, types, params, *, tasks=ALL_TASKS, placement="columns", lesson="prep-numeric", source="numeric"):
    params = deepcopy(params)
    for field in params:
        field["help_key"] = f"preparation.{id}.{field['key']}"
    return {"id": id, "adapter_id": id, "name": name, "description": description,
            "accepted_column_types": list(types), "allowed_tasks": list(tasks), "placement": placement,
            "params": params, "help_key": f"preparation.{id}", "lesson_id": LESSON_IDS.get(lesson, lesson),
            "available": True, "reason": None, "source_urls": [SOURCES[source]], "preserves_rows": True}


_STAGES = [
    _stage("columns.drop", "Убрать столбцы", "Выбранные столбцы исключаются из входов модели. Сохранённый исходный набор не меняется.", ("all",), [], placement="input", lesson="prep-columns"),
    _stage("numeric", "Числовая подготовка: совместимый рецепт", "Совместимый составной этап: заполнение, преобразование, полином и масштабирование числовых признаков.", ("numeric",), [
        _field("imputation", "Заполнение пропусков", "median", options=["median", "mean", "most_frequent", "constant", "knn", "none"], lesson="prep-missing"),
        _field("scaler", "Масштаб", "standard", options=["standard", "robust", "minmax", "maxabs", "none"]),
        _field("numeric_transform", "Распределение", "none", options=["none", "log1p", "sqrt", "yeo_johnson", "quantile_normal", "quantile_uniform"]),
        _field("degree", "Степень полинома", 1, type="int", min=1, max=5, lesson="prep-interactions"),
        _field("interaction_only", "Только взаимодействия", False, type="bool", lesson="prep-interactions"),
        _field("missing_indicator", "Индикаторы пропусков", False, type="bool", lesson="prep-missing"),
    ]),
    _stage("numeric.impute", "Заполнить числовые пропуски", "Статистика заполнения вычисляется на обучающей части. Новые строки используют сохранённые значения.", ("numeric",), [
        _field("strategy", "Способ", "median", options=["median", "mean", "most_frequent", "constant", "knn", "iterative"], lesson="prep-missing", help="Медиана/среднее/частое значение учатся на train; constant использует заданное число; KNN ищет обучающих соседей; iterative предсказывает пропуск по другим столбцам с BayesianRidge. IterativeImputer — экспериментальный API sklearn, до 100 признаков и 500 000 ячеек."),
        _field("fill_value", "Значение константы", 0.0, type="float", lesson="prep-missing"),
        _field("n_neighbors", "Соседи KNN", 5, type="int", min=1, max=100, lesson="prep-missing"),
        _field("add_indicator", "Добавить признаки пропусков", False, type="bool", lesson="prep-missing"),
        _field("max_iter", "Проходы итеративного заполнения", 10, type="int", min=1, max=30, lesson="iterative-imputation"),
        _field("n_nearest_features", "Другие признаки для заполнения", 10, type="int", min=1, max=30, lesson="iterative-imputation"),
    ], lesson="prep-missing", source="impute"),
    _stage("numeric.scale", "Изменить масштаб", "Standard/Robust/MinMax/MaxAbs преобразуют столбцы. Normalizer нормирует каждую строку отдельно.", ("numeric",), [
        _field("method", "Способ", "standard", options=["standard", "robust", "minmax", "maxabs", "normalize", "none"]),
        _field("clip", "Ограничить MinMax диапазоном train", False, type="bool", help="Без clip новые значения ниже минимума train могут стать отрицательными."),
    ]),
    _stage("numeric.power", "Изменить распределение", "Yeo-Johnson допускает отрицательные значения, Box-Cox требует положительные. Signed log/sqrt сохраняют знак.", ("numeric",), [
        _field("method", "Способ", "yeo_johnson", options=["log1p", "sqrt", "yeo_johnson", "box_cox", "quantile_normal", "quantile_uniform"]),
        _field("n_quantiles", "Число квантилей", 100, type="int", min=2, max=1000),
    ]),
    _stage("numeric.clip", "Обрезать крайние значения", "Значения ограничиваются квантилями train. Строки проверки и теста сохраняются.", ("numeric",), [
        _field("lower", "Нижний квантиль", 0.01, type="float", min=0, max=1, lesson="prep-outliers"),
        _field("upper", "Верхний квантиль", 0.99, type="float", min=0, max=1, lesson="prep-outliers"),
    ], lesson="prep-outliers"),
    _stage("numeric.polynomial", "Степени и взаимодействия", "Создаёт степени и произведения выбранных числовых признаков. Размер проверяется до выделения матрицы.", ("numeric",), [
        _field("degree", "Степень", 2, type="int", min=1, max=5, lesson="prep-interactions"),
        _field("interaction_only", "Только произведения разных признаков", False, type="bool", lesson="prep-interactions"),
    ], lesson="prep-interactions"),
    _stage("numeric.bin", "Разбить значения на интервалы", "Границы интервалов обучаются на train. Код интервала не означает одинаковое расстояние между объектами.", ("numeric",), [
        _field("n_bins", "Интервалы", 5, type="int", min=2, max=50),
        _field("strategy", "Границы", "quantile", options=["uniform", "quantile", "kmeans"]),
        _field("encode", "Представление", "onehot-dense", options=["ordinal", "onehot-dense"]),
    ]),
    _stage("numeric.spline", "Сплайны: локальные полиномиальные признаки", "Каждый числовой столбец заменяется B-сплайн базисом. Узлы учатся на train; степень и число узлов ограничивают гибкость. Цель не используется.", ("numeric",), [
        _field("n_knots", "Число узлов", 5, type="int", min=2, max=30, lesson="spline-features"),
        _field("degree", "Степень полиномиальных частей", 3, type="int", min=1, max=5, help="Степень 1 дает линейные части, степень 3 — кубические. Это степень каждого локального полинома, а не общий полином на всем диапазоне.", lesson="spline-features"),
        _field("knots", "Расположение узлов", "quantile", options=["quantile", "uniform"], lesson="spline-features"),
        _field("extrapolation", "За пределами диапазона train", "constant", options=["constant", "linear", "error", "continue"], lesson="spline-features"),
    ], lesson="spline-features"),
    _stage("categorical.onehot", "Категории: отдельные столбцы", "Каждая категория получает индикатор. Редкие категории можно объединить; неизвестные обрабатываются явно.", ("categorical",), [
        _field("handle_unknown", "Новая категория", "ignore", options=["ignore", "error", "infrequent_if_exist"], lesson="prep-categories"),
        _field("min_frequency", "Минимум наблюдений категории", 1, type="int", min=1, max=10000, lesson="prep-categories"),
        _field("max_categories", "Максимум категорий столбца", 100, type="int", min=2, max=2000, lesson="prep-categories"),
        _field("sparse_output", "Разреженная матрица", False, type="bool", lesson="prep-categories"),
    ], lesson="prep-categories"),
    _stage("categorical.ordinal", "Категории: порядковые коды", "Категории кодируются числами. Автоматический порядок не имеет предметного смысла; задайте свой порядок при необходимости.", ("categorical",), [
        _field("categories", "Порядок по столбцам", None, type="json", lesson="prep-categories", help="Список списков категорий в порядке выбранных столбцов. null выбирает лексикографический порядок."),
    ], lesson="prep-categories"),
    _stage("categorical.frequency", "Категории: частота", "Заменяет категорию её частотой в train. Новая категория получает ноль; цель не используется.", ("categorical",), [
        _field("normalize", "Доля вместо количества", True, type="bool", lesson="prep-categories"),
    ], lesson="prep-categories"),
    _stage("categorical.hash", "Категории: хеширование", "Категории попадают в фиксированное число корзин. Разные категории могут столкнуться; восстановить исходную категорию нельзя.", ("categorical",), [
        _field("n_features", "Корзины", 128, type="int", min=2, max=2000, lesson="prep-text"),
        _field("alternate_sign", "Чередовать знак", False, type="bool", lesson="prep-text"),
    ], lesson="prep-text"),
    _stage("categorical.target", "Категории: средняя цель", "Кодирует категории по цели с cross-fitting внутри train. Отключён для временных и групповых схем до отдельного crossfit адаптера.", ("categorical",), [
        _field("smooth", "Сглаживание", 10.0, type="float", min=0, max=10000, lesson="prep-target-encoding"),
        _field("cv", "Внутренние части", 3, type="int", min=2, max=10, lesson="prep-target-encoding"),
    ], tasks=("regression", "classification"), lesson="prep-target-encoding"),
    *[_stage(f"text.{method}", title, "Преобразует выбранные текстовые столбцы в разреженную матрицу. Словарь и IDF обучаются только на train; hashing не хранит словарь.", ("text", "categorical"), [
        _field("max_features" if method != "hash" else "n_features", "Размер словаря / корзины", 1000, type="int", min=2, max=20000, lesson="prep-text"),
        _field("analyzer", "Единицы текста", "word", options=["word", "char", "char_wb"], lesson="prep-text"),
        _field("ngram_max", "Максимум слов / символов подряд", 1, type="int", min=1, max=5, lesson="prep-text"),
        _field("lowercase", "Перевести в нижний регистр", True, type="bool", lesson="prep-text"),
        *([_field("alternate_sign", "Чередовать знак", False, type="bool", lesson="prep-text")] if method == "hash" else []),
    ], lesson="prep-text", source="text") for method, title in (("tfidf", "Текст: TF-IDF"), ("count", "Текст: количество слов"), ("hash", "Текст: хеширование"))],
    _stage("datetime.extract", "Извлечь признаки даты", "Разбирает дату в выбранном часовом поясе и создаёт календарные числа. Ошибки даты становятся пропусками, которые заполняются на train.", ("datetime", "categorical"), [
        _field("parts", "Части даты", ["year", "month", "day", "weekday"], type="multiselect", options=["year", "month", "day", "weekday", "hour", "minute", "dayofyear", "weekend"], lesson="prep-datetime"),
        _field("cyclic", "Синус и косинус циклов", False, type="bool", lesson="prep-datetime"),
        _field("timezone", "Часовой пояс", "UTC", type="text", lesson="prep-datetime"),
    ], lesson="prep-datetime"),
    _stage("selection.variance", "Убрать почти постоянные признаки", "Порог сравнивается с дисперсией train. Цель не используется.", ("all",), [
        _field("threshold", "Порог дисперсии", 0.0, type="float", min=0, max=1000000, lesson="prep-selection"),
    ], placement="global", lesson="prep-selection", source="selection"),
    _stage("selection.univariate", "Отобрать признаки по цели", "Оценивает отдельные признаки только на train. F/MI выбирается по задаче; chi2 требует неотрицательных признаков классификации.", ("all",), [
        _field("method", "Статистика", "auto", options=["auto", "f", "mutual_info", "chi2"], lesson="prep-selection"),
        _field("k", "Оставить признаков", 10, type="int", min=1, max=2000, lesson="prep-selection"),
        _field("discrete_features", "Тип признаков для MI", "auto", options=["auto", "continuous", "discrete"], lesson="prep-selection"),
    ], tasks=SUPERVISED_TASKS, placement="global", lesson="prep-selection", source="selection"),
    _stage("selection.model", "Отобрать признаки вспомогательной моделью", "SelectFromModel с небольшим ExtraTrees обучается внутри каждой обучающей части. Отбор видит цель.", ("all",), [
        _field("threshold", "Порог важности", "median", options=["mean", "median"], lesson="prep-selection"),
        _field("max_features", "Оставить не более", 20, type="int", min=1, max=2000, lesson="prep-selection"),
    ], tasks=SUPERVISED_TASKS, placement="global", lesson="prep-selection", source="selection"),
    *[_stage(f"reduction.{method}", title, "Обучает компоненты на train. После преобразования модель использует компоненты, а не исходные столбцы.", ("all",), [
        _field("n_components", "Компоненты", 2, type="int", min=1, max=2000, lesson="prep-reduction"),
    ], placement="global", lesson="prep-reduction", source="reduction") for method, title in (("pca", "PCA: главные компоненты"), ("svd", "SVD: компоненты разреженной матрицы"), ("ica", "ICA: независимые компоненты"), ("nmf", "NMF: неотрицательное разложение"))],
]


class PreparationCatalogue:
    """Return copies so UI callers cannot mutate the executable registry."""

    def stages(self):
        return deepcopy(_STAGES)

    def get(self, adapter_id):
        for item in _STAGES:
            if item["id"] == adapter_id:
                return deepcopy(item)
        raise ValueError(f"Неизвестный этап подготовки: {adapter_id}.")

    def samplers(self):
        classification = ("random_over", "random_under", "smote", "adasyn", "borderline_smote", "svm_smote", "kmeans_smote", "tomek", "enn", "near_miss", "smoteenn", "smotetomek", "smotenc", "smoten")
        result = [{"id": "none", "name": "Не менять обучающие строки", "allowed_tasks": list(ALL_TASKS), "available": True, "reason": None, "params": [], "help_key": "sampling.none", "lesson_id": "classification-resampling"}]
        for method in classification:
            unsupported = method in {"smotenc", "smoten"}
            params = []
            if method not in {"random_over", "random_under", "tomek"}:
                params.append(_field("neighbors", "Соседи", 5, type="int", min=1, max=100, lesson="prep-resampling"))
            if method == "random_under":
                params.append(_field("replacement", "Выбирать строку несколько раз", False, type="bool", help="При случайном уменьшении позволяет выбор с возвращением. Одна исходная строка может повториться в итоговом train.", lesson="prep-resampling"))
            result.append({"id": method, "name": method, "allowed_tasks": ["classification"], "available": not unsupported,
                           "reason": "Нужен отдельный план, сохраняющий категории до пересэмплирования." if unsupported else None,
                           "params": params,
                           "help_key": f"sampling.{method}", "lesson_id": "classification-resampling", "source_urls": [SOURCES["sampling"]]})
        for method in ("regression_over", "regression_under", "smoter", "smogn"):
            params = [
                _field("focus", "Редкие значения цели", "both", options=["both", "high", "low"], help="Выбирает, важны ли верхний, нижний или оба края числовой цели. Это правило относится к train.", lesson="prep-regression-sampling"),
                _field("sampling", "Интенсивность", "balance", options=["balance", "extreme"], help="balance сближает размеры обычных и редких областей цели; extreme сильнее меняет их соотношение.", lesson="prep-regression-sampling"),
                _field("relevance_threshold", "Порог важности цели", 0.5, type="float", min=0.001, max=0.999, help="Строка считается редкой, когда обученная функция важности цели превышает этот порог 0–1. Это не порог значения самой цели.", lesson="prep-regression-sampling"),
            ]
            if method in {"smoter", "smogn"}:
                params.append(_field("neighbors", "Соседи интерполяции", 5, type="int", min=1, max=100, lesson="prep-regression-sampling"))
            if method == "smogn":
                params.append(_field("perturbation", "Добавляемый разброс", 0.02, type="float", min=0.001, max=1.0, help="Масштаб случайного возмущения относительно обучающего разброса. Большое значение может создавать неправдоподобные числовые объекты.", lesson="prep-regression-sampling"))
            result.append({"id": method, "name": method, "allowed_tasks": ["regression"], "available": True, "reason": None,
                           "params": params, "help_key": f"sampling.{method}", "lesson_id": "28-rare-target-smoter",
                           "source_urls": ["https://proceedings.mlr.press/v74/branco17a.html"]})
        labels = {
            "random_over": "Копировать редкие классы", "random_under": "Уменьшить частые классы",
            "smote": "SMOTE: интерполяция редкого класса", "adasyn": "ADASYN: больше примеров у сложных границ",
            "borderline_smote": "Borderline SMOTE: примеры у границы", "svm_smote": "SVM SMOTE: опорные точки границы",
            "kmeans_smote": "KMeans SMOTE: генерация внутри групп", "tomek": "Tomek: очистить близкие противоположные классы",
            "enn": "ENN: очистить несогласованные с соседями", "near_miss": "NearMiss: выбрать часть частого класса",
            "smoteenn": "SMOTE + ENN: генерация и очистка", "smotetomek": "SMOTE + Tomek: генерация и очистка",
            "smotenc": "SMOTENC: числовые и категориальные признаки", "smoten": "SMOTEN: только категории",
            "regression_over": "Копировать редкие области числовой цели", "regression_under": "Уменьшить обычные области числовой цели",
            "smoter": "SMOTER: интерполяция для регрессии", "smogn": "SMOGN: интерполяция и возмущение для регрессии",
        }
        for item in result:
            item["name"] = labels.get(item["id"], item["name"])
            for field in item["params"]:
                field["help_key"] = f"sampling.{item['id']}.{field['key']}"
        return result
