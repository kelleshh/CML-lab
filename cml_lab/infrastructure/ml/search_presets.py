"""Editable, validated search grids with an explicit breadth and fit budget."""
from math import prod

TEMPLATES = {
    'alpha': [.0001, .001, .01, .1, 1., 10., 100.],
    'C': [.001, .01, .1, 1., 10., 100., 1000.],
    'learning_rate': [.005, .01, .03, .05, .1, .2, .3],
    'learning_rate_init': [.0001, .0003, .001, .003, .01, .03],
    'n_estimators': [25, 50, 100, 150, 200, 300, 500],
    'iterations': [25, 50, 100, 150, 200, 300, 500],
    'max_depth': [2, 3, 4, 6, 8, 12, None],
    'min_samples_leaf': [1, 2, 3, 5, 8, 12, 20],
    'min_samples_split': [2, 3, 5, 8, 12, 20],
    'n_neighbors': [3, 5, 7, 11, 15, 21, 31],
    'num_leaves': [7, 15, 31, 63, 127],
    'l2_leaf_reg': [.1, .5, 1., 3., 5., 10., 30.],
    'var_smoothing': [1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6],
    'gamma': ['scale', 'auto', .001, .01, .1, 1., 10.],
    'weights': ['uniform', 'distance'], 'max_features': ['sqrt', 'log2', None],
    'loss': ['squared_error', 'absolute_error', 'huber'],
}
LEVELS = ('Tiny', 'Small', 'Medium', 'Large', 'Wide')


def search_preset(catalogue, payload):
    level = payload.get('breadth', 2)
    if isinstance(level, bool) or not isinstance(level, int) or not 1 <= level <= 5:
        raise ValueError('breadth: целое число от 1 до 5.')
    descriptor = catalogue.descriptor(payload['algorithm_id'])
    task = payload.get('task', descriptor['tasks'][0])
    if task not in {'regression', 'classification', 'ranking', 'forecasting', 'panel'}:
        raise ValueError('Автоматическая сетка требует задачу с учителем и явную метрику.')
    base = payload.get('params') or {}
    catalogue.build(task, descriptor['id'], base)
    schema = {field['key']: field for field in descriptor['params']}
    budget = (4, 12, 30, 60, 100)[level - 1]
    size = (2, 3, 4, 5, 7)[level - 1]
    maximum_parameters = (1, 2, 2, 3, 3)[level - 1]
    keys = [key for key in TEMPLATES if key in schema]
    keys += [key for key, field in schema.items() if key not in keys and field['type'] in {'select', 'choice', 'bool', 'boolean'} and key not in {'verbose', 'warm_start', 'copy_X', 'fit_intercept'}]
    space = {}
    for key in keys:
        field = schema[key]
        candidates = TEMPLATES.get(key) or field.get('choices') or field.get('options') or ([True, False] if field['type'] in {'bool', 'boolean'} else [])
        if not candidates or any(isinstance(value, dict) for value in candidates): continue
        # Distributed choices keep both extremes visible at every breadth.
        positions = sorted({round(index * (len(candidates) - 1) / max(1, min(size, len(candidates)) - 1)) for index in range(min(size, len(candidates)))})
        valid = []
        for index in positions:
            value = candidates[index]
            try: catalogue.build(task, descriptor['id'], {**base, key: value})
            except (ValueError, TypeError): continue
            if value not in valid: valid.append(value)
        if len(valid) > 1: space[key] = valid
        if len(space) >= maximum_parameters: break
    if not space:
        raise ValueError('Для этого алгоритма нет готовой содержательной сетки. Добавьте параметры и диапазоны вручную.')
    while prod(map(len, space.values())) > budget:
        key = max(space, key=lambda candidate: len(space[candidate]))
        values = space[key]
        values.pop(len(values) // 2)
    combinations = prod(map(len, space.values()))
    return {'breadth': level, 'label': LEVELS[level - 1], 'param_space': space,
            'combinations': combinations, 'trials': combinations,
            'description': 'Стартовые варианты проверены по конструктору. Проверьте их смысл и размер данных; совместимость сочетаний проверяется при обучении.',
            'warning': 'Это стартовая сетка, а не рекомендация лучших параметров. Test не участвует в подборе.'}
