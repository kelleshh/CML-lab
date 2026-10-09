"""Task-specific metric adapters. No task is evaluated by an unrelated score."""
from __future__ import annotations

import numpy as np
from sklearn import metrics as sm
from linear_lab.metrics import MetricRegistry, evaluate_expression

REGRESSION_TASKS = {"regression", "forecasting", "panel"}
CLASS_METRICS = {
    "accuracy": ("Доля верных ответов", "max"),
    "balanced_accuracy": ("Средняя полнота по классам", "max"),
    "f1_macro": ("F1: каждый класс одинаково важен", "max"),
    "f1_weighted": ("F1: вес по числу объектов класса", "max"),
    "precision_macro": ("Точность положительных ответов по классам", "max"),
    "recall_macro": ("Полнота по классам", "max"),
    "mcc": ("Корреляция Мэтьюса", "max"),
    "kappa": ("Каппа Коэна", "max"),
    "log_loss": ("Ошибка вероятностей", "min"),
    "roc_auc": ("Площадь под ROC-кривой", "max"),
    "average_precision": ("Средняя точность бинарного ранжирования", "max"),
    "brier": ("Квадратичная ошибка вероятности", "min"),
}
CLASS_METRICS.update({
    'fbeta': ('F-beta', 'max'), 'fbeta_macro': ('F-beta (macro)', 'max'),
    'fbeta_weighted': ('F-beta (weighted)', 'max'), 'fbeta_micro': ('F-beta (micro)', 'max'),
    'fbeta_binary': ('F-beta (binary)', 'max'),
    'f1_micro': ('F1 (micro)', 'max'), 'f1_binary': ('F1 (binary)', 'max'),
    'precision_weighted': ('Precision (weighted)', 'max'), 'precision_micro': ('Precision (micro)', 'max'),
    'precision_binary': ('Precision (binary)', 'max'), 'recall_weighted': ('Recall (weighted)', 'max'),
    'recall_micro': ('Recall (micro)', 'max'), 'recall_binary': ('Recall (binary)', 'max'),
    'jaccard_macro': ('Jaccard (macro)', 'max'), 'hamming_loss': ('Hamming loss', 'min'),
    'zero_one_loss': ('Zero-one loss', 'min'), 'roc_auc_weighted': ('ROC AUC (OvR weighted)', 'max'),
    'roc_auc_ovo': ('ROC AUC (OvO macro)', 'max'),
})
CLUSTER_METRICS = {
    "silhouette": ("Силуэт: разделение групп", "max"),
    "calinski_harabasz": ("Разделение групп относительно разброса", "max"),
    "davies_bouldin": ("Похожесть соседних кластеров", "min"),
    "adjusted_rand": ("Совпадение с известными группами", "max"),
    "adjusted_mutual_info": ("Общая информация с известными группами", "max"),
}
CLUSTER_METRICS.update({
    'normalized_mutual_info': ('Normalized mutual information', 'max'),
    'homogeneity': ('Homogeneity', 'max'), 'completeness': ('Completeness', 'max'),
    'v_measure': ('V-measure', 'max'), 'fowlkes_mallows': ('Fowlkes-Mallows', 'max'),
})
RANK_METRICS = {"ndcg": ("NDCG: качество порядка внутри запросов", "max"),
                "map": ("Средняя точность списков", "max"),
                "mrr": ("Обратная позиция первого полезного объекта", "max")}

METRIC_NAMES = {
    'accuracy': 'Accuracy', 'balanced_accuracy': 'Balanced accuracy',
    'f1_macro': 'F1 (macro)', 'f1_weighted': 'F1 (weighted)',
    'precision_macro': 'Precision (macro)', 'recall_macro': 'Recall (macro)',
    'mcc': 'MCC', 'kappa': "Cohen's kappa", 'log_loss': 'Log loss', 'roc_auc': 'ROC AUC',
    'average_precision': 'Average precision', 'brier': 'Brier score',
    'silhouette': 'Silhouette score', 'calinski_harabasz': 'Calinski-Harabasz score',
    'davies_bouldin': 'Davies-Bouldin score', 'adjusted_rand': 'Adjusted Rand index',
    'adjusted_mutual_info': 'Adjusted mutual information', 'ndcg': 'NDCG@k', 'map': 'MAP@k', 'mrr': 'MRR@k',
    'mse': 'MSE', 'rmse': 'RMSE', 'mae': 'MAE', 'medae': 'MedAE', 'r2': 'R²',
    'mape': 'MAPE', 'smape': 'sMAPE', 'msle': 'MSLE', 'rmsle': 'RMSLE',
    'explained_variance': 'Explained variance', 'max_error': 'Max error',
    'poisson_deviance': 'Mean Poisson deviance', 'gamma_deviance': 'Mean Gamma deviance',
    'tweedie_deviance': 'Mean Tweedie deviance', 'pinball': 'Mean pinball loss',
    'custom': 'Custom metric', 'anomaly_fraction': 'Anomaly fraction', 'reconstruction_mse': 'Reconstruction MSE',
}

def _help(task, item):
    key = item['id']
    if key.startswith('fbeta'):
        return 'F-beta объединяет Precision и Recall: (1+beta²)PR/(beta²P+R). beta=2 сильнее учитывает пропущенные ответы, beta=0.5 — ложные срабатывания. macro одинаково взвешивает классы; weighted учитывает их размер; micro суммирует ошибки.'
    if key.startswith('precision'): return 'Доля правильных ответов среди объектов, которые модель отнесла к классу. Низкая Precision означает много ложных срабатываний. macro усредняет классы поровну, weighted — по их размеру, micro — по общим счетчикам.'
    if key.startswith('recall'): return 'Доля найденных объектов среди всех настоящих объектов класса. Низкая Recall означает много пропусков. binary оценивает положительный класс; другие режимы усредняют несколько классов.'
    if key.startswith('f1'): return 'Гармоническое среднее Precision и Recall при равной важности. Высокая F1 требует обеих величин. Для неизвестных или отсутствующих классов проверьте способ усреднения.'
    descriptions = {
        'jaccard_macro': 'Для каждого класса размер пересечения настоящих и предсказанных объектов делится на размер их объединения; результаты классов усредняются поровну.',
        'hamming_loss': 'Доля неправильных меток; для одной метки на строку совпадает с долей неверных классификаций. Меньше лучше.',
        'zero_one_loss': 'Доля строк с неверным классом, равная 1 − Accuracy. Меньше лучше.',
        'roc_auc_weighted': 'ROC AUC для каждого класса против остальных; среднее взвешено числом настоящих объектов класса. Требует вероятностей и представленных классов.',
        'roc_auc_ovo': 'ROC AUC усредняется по парам классов. Требует вероятностей; отсутствующий класс делает оценку неопределенной.',
        'normalized_mutual_info': 'Общая информация настоящих групп и найденных кластеров нормирована на их неопределенность. Нужны независимые известные метки; они не обучают кластеризацию.',
        'homogeneity': 'Показывает, насколько каждый найденный кластер содержит объекты одного настоящего класса. Разбиение на множество маленьких кластеров может повысить показатель.',
        'completeness': 'Показывает, собраны ли объекты каждого настоящего класса в одном кластере. Объединение всех объектов в один кластер может повысить показатель.',
        'v_measure': 'Гармоническое среднее Homogeneity и Completeness. Нужны независимые известные метки для внешней оценки кластеризации.',
        'fowlkes_mallows': 'Сравнивает пары объектов: вместе ли они и в настоящих группах, и в найденных кластерах. Корень из произведения точности и полноты пар.',
        'anomaly_fraction': 'Доля строк, отмеченных аномалиями. Это характеристика результата, а не качество поиска; больше или меньше не означает лучше.',
        'reconstruction_mse': 'MSE между исходными подготовленными признаками и их восстановлением из новых координат. Требуется inverse_transform.',
    }
    return descriptions.get(key, item.get('description') or item['name'])


class TaskMetrics:
    def catalogue(self, task=None):
        tasks = [task] if task else ["regression", "classification", "clustering", "ranking", "forecasting", "panel", "anomaly", "reduction"]
        items = []
        for current in tasks:
            if current in REGRESSION_TASKS:
                definitions = MetricRegistry().catalogue()
            else:
                definitions = [{"id": key, "name": label, "direction": direction} for key, (label, direction) in self.definitions(current).items()]
            for item in definitions:
                items.append({**item, 'name': METRIC_NAMES.get(item['id'], item['name'].split(' · ')[0]), 'help': _help(current, item), "optimizable": item.get("direction") in {"min", "max"}, "diagnostic": current == "anomaly", "task": current, "help_key":f"metric.{current}.{item['id']}", "lesson_id": 'fbeta-metrics' if item['id'].startswith('fbeta') else {"classification":"classification-metrics", "clustering":"clustering-metrics", "ranking":"ranking-metrics", "reduction":"dimensionality-reduction", "anomaly":"anomaly-detection", "forecasting":"forecasting-validation", "panel":"panel-groups"}.get(current, "20-metrics-experiment")})
        return items

    @staticmethod
    def definitions(task):
        if task == "classification": return CLASS_METRICS
        if task == "clustering": return CLUSTER_METRICS
        if task == "ranking": return RANK_METRICS
        if task == "anomaly": return {"anomaly_fraction": ("Доля объектов, отмеченных необычными", None)}
        if task == "reduction": return {"explained_variance": ("Сохраненная доля дисперсии", "max"), "reconstruction_mse": ("Ошибка восстановления признаков", "min")}
        return {}

    def direction(self, task, name):
        return next((x["direction"] for x in self.catalogue(task) if x["id"] == name), None)

    def evaluate(self, task, actual, predicted, *, selection=None, probabilities=None,
                 n_classes=None, X=None, groups=None, reference=None, model=None,
                 expression=None, metric_params=None):
        if task in REGRESSION_TASKS:
            return MetricRegistry().evaluate(actual, predicted, selection, expression, metric_params)
        definitions = self.definitions(task)
        selected = list(definitions) if selection is None or selection == "all" or "all" in selection else [selection] if isinstance(selection, str) else list(selection)
        values, details = {}, {}
        for name in dict.fromkeys(selected):
            value, reason = None, None
            try:
                if name not in definitions: raise ValueError("Метрика не относится к выбранной задаче.")
                value = self._value(task, name, actual, predicted, probabilities, n_classes, X, groups, reference, model, metric_params or {})
                if not np.isfinite(value): raise ValueError("Метрика не определена на этой выборке.")
                value = float(value)
            except (ValueError, TypeError, ArithmeticError) as error:
                reason = str(error)
            values[name] = value
            details[name] = {"value": value, "reason": reason, "direction": definitions.get(name, ("", None))[1]}
        return values, details

    @staticmethod
    def _value(task, name, y, p, proba, n_classes, X, groups, reference, model, params):
        if task == "classification":
            functions = {"accuracy":sm.accuracy_score, "balanced_accuracy":sm.balanced_accuracy_score,
                         "mcc":sm.matthews_corrcoef, "kappa":sm.cohen_kappa_score}
            if name in functions: return functions[name](y, p)
            if name.startswith(('f1_', 'fbeta', 'precision_', 'recall_', 'jaccard_')):
                average = params.get('average', 'macro') if name == 'fbeta' else name.rsplit('_', 1)[1]
                if average not in {'binary', 'macro', 'weighted', 'micro'}:
                    raise ValueError('average: binary, macro, weighted или micro.')
                if average == 'binary' and n_classes != 2:
                    raise ValueError('binary average требует ровно два класса; используйте macro/weighted/micro.')
                function = sm.fbeta_score if name.startswith('fbeta') else sm.f1_score if name.startswith('f1') else sm.precision_score if name.startswith('precision') else sm.recall_score if name.startswith('recall') else sm.jaccard_score
                options = {'average': average, 'zero_division': 0, 'labels': np.arange(n_classes)}
                if name.startswith('fbeta'):
                    beta = params.get('beta', 1.0)
                    if isinstance(beta, bool) or not isinstance(beta, (int, float)) or not np.isfinite(beta) or not 0 < beta <= 100:
                        raise ValueError('beta: конечное число больше 0 и не больше 100.')
                    options['beta'] = float(beta)
                return function(y, p, **options)
            if name == 'hamming_loss': return sm.hamming_loss(y, p)
            if name == 'zero_one_loss': return sm.zero_one_loss(y, p)
            if proba is None: raise ValueError("Модель не возвращает вероятности; выбери вероятностный классификатор.")
            if name == "log_loss": return sm.log_loss(y, proba, labels=np.arange(n_classes))
            if name in {"roc_auc", 'roc_auc_weighted', 'roc_auc_ovo'}:
                if n_classes == 2: return sm.roc_auc_score(y, proba[:, 1])
                return sm.roc_auc_score(y, proba, multi_class='ovo' if name == 'roc_auc_ovo' else 'ovr', average='weighted' if name == 'roc_auc_weighted' else 'macro', labels=np.arange(n_classes))
            if n_classes != 2: raise ValueError("Эта метрика требует ровно два класса.")
            if name == "average_precision": return sm.average_precision_score(y, proba[:, 1])
            if name == "brier": return sm.brier_score_loss(y, proba[:, 1])
        if task == "clustering":
            labels = np.asarray(p)
            keep = labels != -1
            external = {'adjusted_rand': sm.adjusted_rand_score, 'adjusted_mutual_info': sm.adjusted_mutual_info_score,
                        'normalized_mutual_info': sm.normalized_mutual_info_score, 'homogeneity': sm.homogeneity_score,
                        'completeness': sm.completeness_score, 'v_measure': sm.v_measure_score, 'fowlkes_mallows': sm.fowlkes_mallows_score}
            if name in external:
                if reference is None: raise ValueError("Выбери колонку известных групп для внешней оценки.")
                return external[name](reference, labels)
            usable = X[keep]
            labels = labels[keep]
            if not 1 < len(np.unique(labels)) < len(labels): raise ValueError("Нужны минимум два кластера и больше объектов, чем кластеров; шум исключен.")
            if name == "silhouette": return sm.silhouette_score(usable, labels, sample_size=min(2000, len(labels)), random_state=42)
            from .matrices import dense
            usable=dense(usable)
            if name == "calinski_harabasz": return sm.calinski_harabasz_score(usable, labels)
            if name == "davies_bouldin": return sm.davies_bouldin_score(usable, labels)
        if task == "ranking":
            if groups is None: raise ValueError("Для ранжирования нужен идентификатор запроса.")
            y, p, groups = np.asarray(y), np.asarray(p), np.asarray(groups)
            scores = []
            k = int(params.get("k", 10))
            if not 1 <= k <= 1000: raise ValueError("Глубина списка k: целое число от 1 до 1000.")
            for group in np.unique(groups):
                mask = groups == group
                relevance, prediction = y[mask], p[mask]
                if len(relevance) < 2: continue
                gains = np.exp2(relevance) - 1
                if name == "ndcg":
                    scores.append(sm.ndcg_score(gains[None, :], prediction[None, :], k=k))
                else:
                    ranked = relevance[np.argsort(-prediction, kind="stable")][:k] > 0
                    positions = np.flatnonzero(ranked) + 1
                    if name == "mrr": scores.append(1 / positions[0] if len(positions) else 0)
                    else:
                        denominator = min(k, np.sum(relevance > 0))
                        scores.append(float(np.sum(np.arange(1, len(positions) + 1) / positions) / denominator) if denominator else 0)
            if not scores: raise ValueError("Нет запросов с двумя и более объектами.")
            return float(np.mean(scores))
        if task == "anomaly": return float(np.mean(np.asarray(p) == -1))
        if task == "reduction":
            if name == "explained_variance":
                if not hasattr(model, "explained_variance_ratio_"): raise ValueError("Решатель не имеет доли объясненной дисперсии.")
                return float(np.sum(model.explained_variance_ratio_))
            if not hasattr(model, "inverse_transform"): raise ValueError("Метод не умеет восстанавливать исходные признаки.")
            from .matrices import dense
            # sklearn MSE accepts dense arrays; apply the same allocation bound
            # as reducer coordinates instead of silently losing sparse metrics.
            original = dense(X)
            return float(sm.mean_squared_error(original, dense(model.inverse_transform(p))))
        raise ValueError("Эта метрика не определена для выбранной задачи.")
