"""Compose public catalogue metadata without importing numerical libraries."""
TASKS = (
    ("regression", "Регрессия", "Предсказать числовой ответ", "01-prediction"),
    ("classification", "Классификация", "Предсказать категорию", "classification-basics"),
    ("clustering", "Кластеризация", "Найти группы без известных ответов", "clustering-basics"),
    ("ranking", "Ранжирование", "Упорядочить объекты внутри запросов", "ranking-groups"),
    ("forecasting", "Временные ряды", "Предсказать будущие наблюдения", "forecasting-lags"),
    ("panel", "Панельные ряды", "Прогноз для нескольких объектов во времени", "panel-groups"),
    ("anomaly", "Поиск аномалий", "Найти необычные наблюдения", "anomaly-detection"),
    ("reduction", "Снижение размерности", "Получить компактное представление признаков", "dimensionality-reduction"),
)


class CatalogueApplication:
    def __init__(self, algorithms, data, preparation, metrics, learning):
        self.algorithms, self.data, self.preparation = algorithms, data, preparation
        self.metrics, self.learning = metrics, learning

    def get(self):
        return {"tasks": [{"id": key, "name": name, "label": name, "description": description, "lesson_id": lesson} for key, name, description, lesson in TASKS],
                "algorithms": self.algorithms.catalogue(), "datasets": self.data.catalogue(),
                "preprocessors": self.preparation.stages(), "samplers": self.preparation.samplers(),
                "metrics": self.metrics.catalogue(), "lessons": self.learning.catalogue()}
