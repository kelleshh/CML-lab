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
CLUSTER_METRICS = {
    "silhouette": ("Силуэт: разделение групп", "max"),
    "calinski_harabasz": ("Разделение групп относительно разброса", "max"),
    "davies_bouldin": ("Похожесть соседних кластеров", "min"),
    "adjusted_rand": ("Совпадение с известными группами", "max"),
    "adjusted_mutual_info": ("Общая информация с известными группами", "max"),
}
RANK_METRICS = {"ndcg": ("NDCG: качество порядка внутри запросов", "max"),
                "map": ("Средняя точность списков", "max"),
                "mrr": ("Обратная позиция первого полезного объекта", "max")}


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
                items.append({**item, "optimizable": item.get("direction") in {"min", "max"}, "diagnostic": current == "anomaly", "task": current, "help_key":f"metric.{current}.{item['id']}", "lesson_id": {"classification":"classification-metrics", "clustering":"clustering-metrics", "ranking":"ranking-metrics", "reduction":"dimensionality-reduction", "anomaly":"anomaly-detection", "forecasting":"forecasting-validation", "panel":"panel-groups"}.get(current, "20-metrics-experiment")})
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
            if name in {"f1_macro", "f1_weighted", "precision_macro", "recall_macro"}:
                function = sm.f1_score if name.startswith("f1") else sm.precision_score if name.startswith("precision") else sm.recall_score
                return function(y, p, average="weighted" if name.endswith("weighted") else "macro", zero_division=0, labels=np.arange(n_classes))
            if proba is None: raise ValueError("Модель не возвращает вероятности; выбери вероятностный классификатор.")
            if name == "log_loss": return sm.log_loss(y, proba, labels=np.arange(n_classes))
            if name == "roc_auc":
                if n_classes == 2: return sm.roc_auc_score(y, proba[:, 1])
                return sm.roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=np.arange(n_classes))
            if n_classes != 2: raise ValueError("Эта метрика требует ровно два класса.")
            if name == "average_precision": return sm.average_precision_score(y, proba[:, 1])
            if name == "brier": return sm.brier_score_loss(y, proba[:, 1])
        if task == "clustering":
            labels = np.asarray(p)
            keep = labels != -1
            if name in {"adjusted_rand", "adjusted_mutual_info"}:
                if reference is None: raise ValueError("Выбери колонку известных групп для внешней оценки.")
                return (sm.adjusted_rand_score if name == "adjusted_rand" else sm.adjusted_mutual_info_score)(reference, labels)
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
