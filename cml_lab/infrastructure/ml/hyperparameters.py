"""Constructor-derived, JSON-safe model parameter schemas.

Only data crosses the recipe boundary. Estimators, kernels and approved callback
objects are constructed from declarations; recipe strings are never evaluated.
"""
from __future__ import annotations

from copy import deepcopy
import inspect
import math
import re
from numbers import Integral, Real
from typing import Any

from sklearn.utils._param_validation import Hidden, Interval, StrOptions


ESTIMATOR_FIELDS = {"estimator", "final_estimator", "covariance_estimator"}

# kwargs accepted by the public sklearn wrappers in addition to signature fields.
# Data loading / output-file settings are intentionally not training parameters.
BOOSTER_PARAMETERS = {
    "xgboost": {
        "max_cached_hist_node": None, "extmem_single_page": None,
        "lambdarank_pair_method": None, "lambdarank_num_pair_per_sample": None,
        "lambdarank_unbiased": None, "lambdarank_bias_norm": None,
        "lambdarank_score_normalization": None, "lambdarank_normalization": None,
        "ndcg_exp_gain": None, "aft_loss_distribution": None,
        "aft_loss_distribution_scale": None, "tweedie_variance_power": None,
        "huber_slope": None, "quantile_alpha": None,
        "rate_drop": None, "skip_drop": None, "sample_type": None,
        "normalize_type": None, "one_drop": None,
        "updater": None, "refresh_leaf": None, "process_type": None,
        "feature_selector": None, "top_k": None,
    },
    "lightgbm": {
        "max_bin": None, "min_data_in_bin": None, "bin_construct_sample_cnt": None,
        "data_random_seed": None, "is_enable_sparse": None, "enable_bundle": None,
        "use_missing": None, "zero_as_missing": None, "feature_pre_filter": None,
        "min_gain_to_split": None, "max_delta_step": None,
        "lambda_l1": None, "lambda_l2": None, "linear_lambda": None,
        "path_smooth": None, "linear_tree": None,
        "max_cat_threshold": None, "cat_l2": None, "cat_smooth": None,
        "max_cat_to_onehot": None, "min_data_per_group": None,
        "monotone_constraints": None, "monotone_constraints_method": None,
        "monotone_penalty": None, "feature_contri": None, "forcedsplits_filename": None,
        "feature_fraction": None, "feature_fraction_bynode": None,
        "feature_fraction_seed": None, "extra_trees": None, "extra_seed": None,
        "bagging_fraction": None, "bagging_freq": None, "bagging_seed": None,
        "pos_bagging_fraction": None, "neg_bagging_fraction": None,
        "bagging_by_query": None, "first_metric_only": None,
        "early_stopping_min_delta": None, "max_drop": None,
        "drop_rate": None, "skip_drop": None, "xgboost_dart_mode": None,
        "uniform_drop": None, "drop_seed": None,
        "top_rate": None, "other_rate": None,
        "deterministic": None, "force_col_wise": None, "force_row_wise": None,
        "histogram_pool_size": None, "num_threads": None, "verbosity": None,
        "device_type": None, "gpu_platform_id": None, "gpu_device_id": None,
        "gpu_use_dp": None, "num_gpu": None,
        "is_unbalance": None, "scale_pos_weight": None, "sigmoid": None,
        "boost_from_average": None, "reg_sqrt": None, "alpha": None,
        "fair_c": None, "poisson_max_delta_step": None, "tweedie_variance_power": None,
        "lambdarank_truncation_level": None, "lambdarank_norm": None,
        "lambdarank_position_bias_regularization": None, "label_gain": None,
        "eval_at": None, "metric": None, "metric_freq": None,
        "objective_seed": None, "num_class": None, "cegb_tradeoff": None,
        "cegb_penalty_split": None, "cegb_penalty_feature_lazy": None,
        "cegb_penalty_feature_coupled": None, "interaction_constraints": None,
        "precise_float_parser": None, "pre_partition": None,
    },
}

PARAMETER_HELP = {
    "random_state": "Начальное состояние генератора случайных чисел. Одинаковое целое число воспроизводит случайные решения; null использует seed проекта.",
    "random_seed": "Начальное состояние генератора CatBoost. Одинаковое целое число воспроизводит случайные решения; null использует seed проекта.",
    "n_jobs": "Число параллельных потоков модели. null использует лимит проекта; лаборатория принимает от 1 до 8 потоков.",
    "thread_count": "Число потоков CatBoost. null использует лимит проекта; лаборатория принимает от 1 до 8 потоков.",
    "num_threads": "Число потоков LightGBM. null использует лимит проекта; лаборатория принимает от 1 до 8 потоков.",
    "hidden_layer_sizes": "Размеры скрытых слоев MLP по порядку. [64, 32] создает два слоя: 64 нейрона, затем 32. Размер выходного слоя определяется задачей.",
    "estimators": "Именованные базовые модели ансамбля. Укажите список объектов с name, algorithm_id и params. Каждая модель обучается только внутри обучающей части.",
    "estimator": "Базовая модель. Объект algorithm_id и params задает другой алгоритм из каталога; null оставляет библиотечный выбор.",
    "final_estimator": "Итоговая модель стекинга. Обучается на OOF-прогнозах базовых моделей, а не на их ответах по собственным обучающим строкам.",
    "covariance_estimator": "Оцениватель ковариационной матрицы. Декларация class и params выбирает разрешенный sklearn.covariance estimator; null оставляет стандартный расчет.",
    "kernel": "Функция сходства строк. В SVM укажите имя ядра или разрешенную функцию; в GaussianProcess используйте декларацию RBF, Matern, DotProduct, WhiteKernel или их sum/product.",
    "max_depth": "Максимальная глубина дерева. Глубокое дерево описывает более сложные зависимости и легче запоминает обучение. null означает библиотечный выбор; точные ограничения зависят от модели.",
    "max_features": "Число или доля признаков, доступных одному разбиению либо одной базовой модели. Смысл зависит от estimator: число, доля, sqrt/log2 или null.",
    "min_samples_leaf": "Минимум строк в конечном листе дерева. Целое значение задает число строк; дробь задает долю обучающей части, если модель поддерживает такой тип.",
    "min_samples_split": "Минимум строк для разделения узла дерева. Целое значение задает число строк; дробь задает долю обучающей части.",
    "class_weight": "Веса ошибок разных классов. balanced усиливает вклад редких классов; словарь задает вес каждому классу. Это не добавляет строки в датасет.",
    "max_iter": "Максимум итераций библиотечного решателя. Это ограничение вычислений; итерации разных алгоритмов выполняют разные действия.",
    "tol": "Допуск остановки решателя. Меньшее значение требует более точного решения и обычно увеличивает время обучения.",
    "fit_intercept": "Нужно ли оценивать свободный член, общий сдвиг предсказания. false предполагает, что сдвиг не нужен или уже учтен преобразованиями.",
    "copy_X": "Копировать ли входную матрицу перед внутренними преобразованиями. true защищает исходную матрицу от изменений и требует дополнительной памяти.",
    "alpha": "Коэффициент регуляризации или сглаживания. Больший alpha обычно сильнее ограничивает модель; у GaussianProcessRegressor это дисперсия наблюдательного шума.",
    "C": "Цена ошибки относительно ограничения коэффициентов в SVM и LogisticRegression. Большое C сильнее штрафует ошибки и ослабляет относительную регуляризацию.",
    "learning_rate": "Скорость обучения. В бустинге это вклад одного дерева; в SGD и MLP строка выбирает правило изменения шага.",
    "n_estimators": "Количество базовых моделей или раундов бустинга. Больше моделей требует больше времени; остановка обучения может дать меньше моделей.",
    "iterations": "Максимальное число шагов CatBoost. Один шаг обычно добавляет дерево; остановка по validation может завершить обучение раньше.",
    "depth": "Максимальная глубина дерева CatBoost. Глубина определяет сложность дерева и число комбинаций условий.",
    "warm_start": "Повторный fit того же объекта может продолжить существующее решение. Новый запуск лаборатории создает отдельный объект и не продолжает старый артефакт автоматически.",
    "callbacks": "Список разрешенных деклараций callback XGBoost. Они выполняются в worker; custom Python callbacks в JSON-рецепте не исполняются.",
    "callback": "Callback объект CatBoost. JSON-рецепт не исполняет произвольный Python; null оставляет callbacks отключенными.",
    "missing": "Значение, которое XGBoost считает пропуском. null в форме соответствует NaN, стандартному маркеру пропущенного числового значения.",
    "memory": "Путь к кэшу sklearn либо null. Кэш сохраняет вычисленные преобразования; путь относится к машине, где запущена лаборатория.",
}


def json_value(value: Any) -> Any:
    """Encode constructor defaults without repr strings or non-finite JSON."""
    if value is inspect.Parameter.empty:
        return None
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, Real):
        return float(value) if math.isfinite(float(value)) else ({'special': 'inf' if value > 0 else '-inf'} if not math.isnan(float(value)) else None)
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if hasattr(value, "get_params"):
        return {"class": type(value).__name__, "params": json_value(value.get_params(deep=False))}
    if hasattr(value, "tolist"):
        return json_value(value.tolist())
    return None


def parameter_docs(cls) -> dict[str, tuple[str, str]]:
    """Read NumPy-style parameter sections from the installed public classes."""
    sources = [inspect.getdoc(cls) or "", inspect.getdoc(cls.__init__) or ""]
    for base in cls.__mro__[1:]:
        if base.__module__.split('.')[0] in {"xgboost", "lightgbm"}:
            sources.append(inspect.getdoc(base) or "")
            sources.append(inspect.getdoc(base.__init__) or "")
    if cls.__module__.startswith("catboost"):
        from catboost import CatBoostClassifier
        sources.append(inspect.getdoc(CatBoostClassifier) or "")
    result = {}
    header = re.compile(r"^\s*([A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)*)\s*:\s*(.*)$")
    for source in sources:
        lines = source.splitlines()
        for index, line in enumerate(lines):
            match = header.match(line)
            if not match:
                continue
            indent = len(line) - len(line.lstrip())
            content = []
            for following in lines[index + 1:]:
                if header.match(following) and len(following) - len(following.lstrip()) <= indent:
                    break
                if following and len(following) - len(following.lstrip()) <= indent and not following.startswith(' '):
                    break
                content.append(following.strip())
            description = re.sub(r"\s+", " ", " ".join(content)).strip()
            for name in match.group(1).split(','):
                result.setdefault(name.strip(), (match.group(2), description))
    return result


def constructor_defaults(cls) -> dict[str, Any]:
    signature = inspect.signature(cls)
    values = {name: parameter.default for name, parameter in signature.parameters.items()
              if parameter.kind not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)}
    kwargs = {name: [] for name, value in values.items()
              if value is inspect.Parameter.empty and name == 'estimators'}
    try:
        values.update(cls(**kwargs).get_params(deep=False))
    except (TypeError, ValueError):
        pass
    library = cls.__module__.split('.', 1)[0]
    values.update({key: value for key, value in BOOSTER_PARAMETERS.get(library, {}).items() if key not in values})
    return values


def _constraint_metadata(cls, key) -> tuple[list, list, bool]:
    constraints = [item.constraint if isinstance(item, Hidden) else item
                   for item in getattr(cls, '_parameter_constraints', {}).get(key, [])
                   if not isinstance(item, Hidden) or isinstance(item.constraint, StrOptions)]
    options, numeric = [], []
    nullable = False
    for constraint in constraints:
        if isinstance(constraint, StrOptions):
            options.extend(sorted(constraint.options))
        elif isinstance(constraint, Interval):
            numeric.append(constraint)
        elif constraint is None:
            nullable = True
    if nullable:
        options.append(None)
    return constraints, options, numeric


def parameter_schema(cls, algorithm_id: str, old_fields: list[dict], defaults: dict,
                     source: str) -> list[dict]:
    from cml_lab.infrastructure.hyperparameter_learning import parameter_knowledge
    docs = parameter_docs(cls)
    library_defaults = constructor_defaults(cls)
    old = {item['key']: item for item in old_fields}
    # Preserve useful task defaults while naming parameters exactly as constructors.
    if 'hidden_units' in old:
        old['hidden_layer_sizes'] = {**old['hidden_units'], 'default': [old['hidden_units']['default']]}
    result = []
    order = [key for key in old if key in library_defaults]
    order.extend(key for key in library_defaults if key not in order)
    for key in order:
        native = library_defaults[key]
        default = defaults.get(key, native)
        if key in {'random_state', 'random_seed', 'n_jobs', 'thread_count', 'num_threads'}:
            default = None
        old_item = old.get(key, {})
        documented_type, original = docs.get(key, ('', ''))
        constraints, options, intervals = _constraint_metadata(cls, key)
        kind = 'json'
        nullable = any(item is None for item in constraints) or default is None
        if key in ESTIMATOR_FIELDS:
            kind = 'estimator'
        elif key == 'estimators':
            kind = 'estimators'
        elif key == 'kernel' and cls.__module__.startswith('sklearn.gaussian_process'):
            kind = 'kernel'
        elif key in {'random_state', 'random_seed', 'n_jobs', 'thread_count', 'num_threads'}:
            kind = 'int'
        elif options and all(isinstance(item, (str, type(None))) for item in options) and not intervals and all(isinstance(item, StrOptions) or item is None for item in constraints):
            kind = 'select'
        elif isinstance(default, bool) and not any(item is callable for item in constraints):
            kind = 'bool'
        elif intervals and len({item.type for item in intervals}) == 1 and all(option is None for option in options) and all(isinstance(item, Interval) or item is None for item in constraints):
            kind = 'int' if intervals[0].type is Integral else 'float'
        elif not constraints and default is not None:
            kind = ('bool' if isinstance(default, bool) else 'int' if isinstance(default, Integral)
                    else 'float' if isinstance(default, Real) else 'string' if isinstance(default, str) else 'json')
        elif isinstance(default, str) and not constraints and not documented_type.lower().startswith(('list', 'dict', 'tuple')):
            kind = 'string'
        elif constraints and all(item == 'boolean' for item in constraints):
            kind = 'bool'
        elif constraints and all(item in ('boolean', 'verbose') for item in constraints) and isinstance(default, bool):
            kind = 'bool'
        elif constraints and all(isinstance(item, StrOptions) for item in constraints):
            kind = 'select'
        if not constraints and default is None:
            atomic = documented_type.lower().split(',', 1)[0].strip()
            atomic = re.sub(r'^(?:typing\.)?optional\[(int|float|str|bool)\]$', r'\1', atomic)
            if atomic in {'int', 'integer', 'float', 'str', 'string'}:
                kind = {'integer': 'int', 'str': 'string'}.get(atomic, atomic)
        if isinstance(default, Real) and not isinstance(default, (bool, Integral)) and not math.isfinite(float(default)):
            kind = 'json'
        if not constraints and cls.__module__.split('.', 1)[0] in {'linear_lab', 'skglm'} and old_item.get('type') in {'int', 'float', 'bool', 'select'}:
            kind = old_item['type']
            options = deepcopy(old_item.get('options', []))
        if key == 'hidden_layer_sizes':
            kind = 'json'
        field = {
            'key': key, 'native_name': key, 'label': key, 'type': kind,
            'default': json_value(default), 'native_default': json_value(native),
            'help': parameter_knowledge(algorithm_id, '', key, class_name=cls.__name__, library=cls.__module__.split('.', 1)[0]).get('summary') or PARAMETER_HELP.get(key) or old_item.get('help') or f'{key}: подробное объяснение в учебнике.',
            'help_key': f'model.{algorithm_id}.{key}',
            'lesson_id': f"parameter-{algorithm_id}-{key.lower().replace('_', '-')}",
            'source': source, 'constructor_doc': original,
            'documented_type': documented_type, 'advanced': key not in old,
            'nullable': nullable,
            'numeric_only': bool(intervals) and not any(item == 'boolean' or item is bool for item in constraints),
        }
        if options:
            field['options'] = options
            field['choices'] = options
        if kind in {'int', 'float'} and len(intervals) == 1:
            interval = intervals[0]
            if interval.left is not None and math.isfinite(float(interval.left)):
                field['min'] = interval.left
                field['min_exclusive'] = interval.closed not in ('left', 'both')
            if interval.right is not None and math.isfinite(float(interval.right)):
                field['max'] = interval.right
                field['max_exclusive'] = interval.closed not in ('right', 'both')
            if kind == 'int':
                field['step'] = 1
        if not constraints and cls.__module__.split('.', 1)[0] in {'linear_lab', 'skglm'} and kind in {'int', 'float'}:
            if 'min' in old_item:
                field['min'] = old_item['min']
            if cls.__name__ == 'BestSubsetRegressor' and key == 'max_features':
                field['max'] = 12
        if key in {'random_state', 'random_seed'}:
            field.update(min=0, max=2**32 - 1)
        if key in {'n_jobs', 'thread_count', 'num_threads'}:
            field.update(min=1, max=8, execution_limit=True)
        if key in {'callback', 'callbacks', 'is_data_valid', 'is_model_valid'}:
            field['declarative_only'] = True
        result.append(field)
    return result


def validate_json(value, *, key='parameter', depth=0):
    if depth > 15:
        raise ValueError(f'{key}: слишком глубокая декларация параметра.')
    if value is None or isinstance(value, (bool, str, Integral)):
        if isinstance(value, str) and len(value) > 100000:
            raise ValueError(f'{key}: строка параметра слишком длинная.')
        return
    if isinstance(value, Real):
        if not math.isfinite(float(value)):
            raise ValueError(f'{key}: требуется конечное число.')
        return
    if isinstance(value, (list, tuple)):
        if len(value) > 10000:
            raise ValueError(f'{key}: список параметра слишком длинный.')
        for item in value:
            validate_json(item, key=key, depth=depth + 1)
        return
    if isinstance(value, dict) and all(isinstance(name, (str, int)) for name in value):
        if len(value) > 10000:
            raise ValueError(f'{key}: объект параметра слишком большой.')
        for item in value.values():
            validate_json(item, key=key, depth=depth + 1)
        return
    raise ValueError(f'{key}: требуется JSON-значение, а не исполняемый Python объект.')
