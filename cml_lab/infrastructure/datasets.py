"""Dataset port adapter and bounded, reproducible CML example generators."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs, make_classification

from cml_lab.shared.domain import ValidationError, integer
from linear_lab.datasets import DataService, MAX_CELLS, MAX_ROWS, TASK_LABELS


_TASK_ALIASES = {"time_series": "forecasting"}
_LESSONS = {
    "classification": "classification-basics", "clustering": "clustering-basics",
    "ranking": "ranking-groups", "forecasting": "forecasting-lags",
    "panel": "panel-groups", "anomaly": "anomaly-detection",
    "reduction": "dimensionality-reduction",
}
_GENERATORS = {
    "classification": ("Классы", "classification", "Два или несколько классов с числовыми признаками."),
    "clustering": ("Группы наблюдений", "clustering", "Облака точек; известная группа сохранена только для оценки результата."),
    "ranking": ("Объекты внутри запросов", "ranking", "Запросы, числовые признаки и целая оценка полезности объекта от 0 до 4."),
    "forecasting": ("Временной ряд", "forecasting", "Последовательность дат, наблюдаемый сезонный признак и числовая цель."),
    "panel": ("Панельные ряды", "panel", "Несколько объектов с собственными временными рядами и числовой целью."),
    "anomaly": ("Редкие аномалии", "anomaly", "Обычные точки и редкие удаленные точки; метка не участвует в обучении."),
    "reduction": ("Связанные признаки", "reduction", "Много наблюдаемых признаков получены из двух скрытых координат."),
}


def _metadata(value: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(value)
    result["task"] = _TASK_ALIASES.get(result.get("task"), result.get("task"))
    result["tasks"] = list(dict.fromkeys(_TASK_ALIASES.get(task, task) for task in result.get("tasks", [result.get("task", "other")])))
    result["task_label"] = TASK_LABELS.get(result["task"], TASK_LABELS["other"])
    if result["task"] == "classification":
        obsolete = ("Выберите числовое измерение как цель.", "Это набор классификации или другого назначения.")
        result["warnings"] = [warning for warning in result.get("warnings", []) if not str(warning).startswith(obsolete)]
    return result


def _number(params: dict, key: str, default: float, low: float, high: float) -> float:
    value = params.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not low <= value <= high:
        raise ValidationError(f"{key}: нужно конечное число от {low} до {high}.")
    return float(value)


def _count(params: dict, key: str, default: int, low: int, high: int) -> int:
    return integer(params.get(key, default), key, low, high)


def _validate_custom_metadata(metadata: dict, rows) -> None:
    """Reject malformed portable descriptions before the legacy adapter writes."""
    for key, maximum in (("source", 100), ("generator", 200), ("description", 5000)):
        if key in metadata and (not isinstance(metadata[key], str) or len(metadata[key]) > maximum):
            raise ValidationError(f"metadata.{key}: нужна строка не длиннее {maximum} символов.")
    if "params" in metadata and not isinstance(metadata["params"], dict):
        raise ValidationError("metadata.params: нужен объект настроек источника.")
    # The adapter validates row counts, cell types and normalized header names.
    # Here references are checked only when rows already have the expected shape.
    columns = {str(name).strip() for row in rows for name in row} if isinstance(rows, list) and rows and all(isinstance(row, dict) for row in rows) else None
    for key in ("excluded_features", "excluded_targets"):
        if key not in metadata:
            continue
        values = metadata[key]
        if not isinstance(values, list) or len(values) > 1000 or any(not isinstance(value, str) for value in values) or len(values) != len(set(values)):
            raise ValidationError(f"metadata.{key}: нужен список неповторяющихся имен столбцов.")
        if columns is not None and any(value not in columns for value in values):
            raise ValidationError(f"metadata.{key}: указан отсутствующий столбец.")
    if "warnings" in metadata:
        warnings = metadata["warnings"]
        if not isinstance(warnings, list) or len(warnings) > 100 or any(not isinstance(value, str) or len(value) > 5000 for value in warnings):
            raise ValidationError("metadata.warnings: нужен список не более 100 текстовых предупреждений.")
    if "roles" in metadata:
        roles = metadata["roles"]
        if not isinstance(roles, dict) or len(roles) > 32 or any(not isinstance(key, str) or not 0 < len(key) <= 100 or not isinstance(column, str) for key, column in roles.items()):
            raise ValidationError("metadata.roles: нужен объект с именами ролей и столбцов.")
        if columns is not None and any(column not in columns for column in roles.values()):
            raise ValidationError("metadata.roles: роль ссылается на отсутствующий столбец.")
    target = metadata.get("task_target")
    if target is not None and (not isinstance(target, str) or columns is not None and target not in columns):
        raise ValidationError("metadata.task_target: выберите существующий целевой столбец либо null.")


class LegacyDataGateway:
    """Expose the data port using only public methods of the existing adapter.

    Dataset IDs remain immutable snapshots. Metadata/row updates return a new ID;
    references held by existing experiments continue to point at their original
    table. Algorithm-specific target types are decided by the execution adapter.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.legacy = DataService(self.root, allow_categorical_only=True)

    def frame(self, dataset_id: str) -> pd.DataFrame:
        return self.legacy.read_frame(dataset_id)

    def describe(self, dataset_id: str) -> dict:
        return _metadata(self.legacy.describe(dataset_id))

    def catalogue(self) -> list[dict]:
        entries = [_metadata(entry) for entry in self.legacy.catalogue()]
        for entry in entries:
            if entry["supported"] and entry["task"] == "classification":
                entry["description"] = "Табличные признаки и метка класса. Выберите классификацию, чтобы предсказывать категорию; для регрессии можно выбрать отдельное числовое измерение."
        common = [
            {"key": "n_samples", "label": "Число строк", "type": "int", "default": 180, "min": 12, "max": 10000, "step": 1, "help": "Количество наблюдений в создаваемой таблице."},
            {"key": "n_features", "label": "Число признаков", "type": "int", "default": 2, "min": 2, "max": 100, "step": 1, "help": "Число входных числовых столбцов. Даты и идентификаторы объектов задаются отдельно."},
            {"key": "noise", "label": "Случайный разброс", "type": "float", "default": 0.3, "min": 0, "max": 10, "step": 0.1, "help": "Для временных рядов это стандартное отклонение добавляемой случайной ошибки. Для классификации — доля случайно замененных меток, ограниченная 0.4."},
            {"key": "seed", "label": "Случайное зерно", "type": "int", "default": 42, "min": 0, "max": 1000000, "step": 1, "help": "Повторное создание с тем же зерном и настройками дает те же данные."},
        ]
        for identifier, (label, task, description) in _GENERATORS.items():
            params = deepcopy(common)
            if task in {"forecasting", "panel"}:
                params = [param for param in params if param["key"] != "n_features"]
            if task == "anomaly":
                params = [param for param in params if param["key"] != "noise"]
                params.append({"key": "contamination", "label": "Доля аномалий", "type": "float", "default": 0.08, "min": 0.01, "max": 0.4, "step": 0.01, "help": "Такую долю точек генератор помещает далеко от обычного облака. Это известный ответ для проверки, а не признак модели."})
            if task == "classification":
                noise_param = next(param for param in params if param["key"] == "noise")
                noise_param.update(label="Доля случайных меток", default=0.03, max=0.4, step=0.01, help="Генератор с этой вероятностью заменяет исходную метку случайным классом. Чем выше доля, тем сложнее найти зависимость.")
                params.append({"key": "n_classes", "label": "Число классов", "type": "int", "default": 2, "min": 2, "max": 10, "step": 1, "help": "Число разных категорий цели. Для большого числа классов увеличьте число признаков."})
            if task == "clustering":
                params.append({"key": "n_clusters", "label": "Число исходных групп", "type": "int", "default": 3, "min": 2, "max": 20, "step": 1, "help": "Генератор создаст столько облаков точек. Алгоритм кластеризации этих ответов не получает."})
            if task == "ranking":
                params.append({"key": "n_queries", "label": "Число запросов", "type": "int", "default": 12, "min": 3, "max": 1000, "step": 1, "help": "Объекты ранжируются внутри query_id. Все строки одного запроса попадают в одну часть разбиения."})
            if task == "panel":
                params.append({"key": "n_entities", "label": "Число объектов", "type": "int", "default": 5, "min": 2, "max": 100, "step": 1, "help": "Для каждого entity создается отдельная последовательность дат. Лаги вычисляются внутри объекта."})
            for param in params:
                param["help_key"] = f"dataset.{param['key']}"
                param["lesson_id"] = _LESSONS[task]
            entries.append({"id": identifier, "name": identifier, "label": label, "kind": "synthetic", "task": task, "tasks": [task], "task_label": TASK_LABELS[task], "modality": "tabular", "supported": True, "requires_network": False, "description": description, "params": params, "lesson_id": _LESSONS[task]})
        return entries

    def load(self, spec: dict) -> dict:
        if not isinstance(spec, dict):
            raise ValidationError("Настройки набора должны быть объектом.")
        request = deepcopy(spec)
        name = request.get("name", "linear")
        kind = request.get("kind", "synthetic")
        if not isinstance(name, str) or not isinstance(kind, str):
            raise ValidationError("Название и тип источника должны быть строками из каталога.")
        if kind == "sklearn":
            request["kind"] = "synthetic" if isinstance(name, str) and name.startswith("make_") else "fetch" if isinstance(name, str) and name.startswith("fetch_") else "builtin"
            kind = request["kind"]
        if kind == "synthetic" and name in _GENERATORS:
            params = request.get("params", {})
            return self._generate(name, {} if params is None else params)
        if kind == "custom":
            task = request.pop("task", None)
            task_target = request.pop("task_target", None)
            metadata = request.setdefault("metadata", {})
            if not isinstance(metadata, dict):
                raise ValidationError("Описание набора должно быть объектом.")
            if task is not None:
                if not isinstance(task, str) or task not in TASK_LABELS:
                    raise ValidationError("Выберите задачу из каталога.")
                metadata.update(task=_TASK_ALIASES.get(task, task), tasks=[_TASK_ALIASES.get(task, task)])
            if task_target is not None:
                metadata["task_target"] = task_target
            if metadata.get("task") == "classification" and request.get("target") is not None:
                metadata["task_target"] = request.pop("target")
                metadata.setdefault("excluded_features", [metadata["task_target"]])
                metadata.setdefault("excluded_targets", [metadata["task_target"]])
            _validate_custom_metadata(metadata, request.get("rows"))
        return _metadata(self.legacy.load(request))

    def import_bytes(self, data: bytes, filename: str) -> dict:
        return _metadata(self.legacy.import_bytes(data, filename))

    def list_library(self, query: str = "", task: str | None = None, tags: list[str] | None = None, source: str | None = None, offset: int = 0, limit: int = 200) -> dict:
        integer(offset, "Начальный набор", 0, 100000)
        integer(limit, "Наборов на странице", 1, 500)
        if task is not None and (not isinstance(task, str) or task not in TASK_LABELS):
            raise ValidationError("Выберите задачу из каталога.")
        result = self.legacy.list_library(query=query, task=task, tags=tags, source=source, offset=offset, limit=limit)
        result["items"] = [_metadata(item) for item in result["items"]]
        result["tasks"] = [{"id": key, "label": value} for key, value in TASK_LABELS.items() if key != "time_series"]
        return result

    def update_metadata(self, dataset_id: str, patch: dict) -> dict:
        request = deepcopy(patch)
        if isinstance(request, dict) and isinstance(request.get("task"), str):
            request["task"] = _TASK_ALIASES.get(request["task"], request["task"])
        return _metadata(self.legacy.update_metadata(dataset_id, request))

    def delete(self, dataset_id: str) -> dict:
        return self.legacy.delete(dataset_id)

    def export_csv(self, dataset_id: str) -> bytes:
        return self.legacy.export_csv(dataset_id)

    def read_column(self, dataset_id: str, column: str) -> pd.Series:
        return self.legacy.read_column(dataset_id, column)

    def rows(self, dataset_id: str, offset: int = 0, limit: int = 500) -> dict:
        return self.legacy.rows(dataset_id, offset=offset, limit=limit)

    def update_rows(self, dataset_id: str, changes: list[dict], additions: list[dict] | None = None, deletes: list[int] | None = None) -> dict:
        return _metadata(self.legacy.update_rows(dataset_id, changes, additions, deletes))

    def explore(self, dataset_id: str, **options: Any) -> dict:
        result = self.legacy.explore(dataset_id, **options)
        result["metadata"] = _metadata(result["metadata"])
        return result

    def _generate(self, name: str, params: dict) -> dict:
        if not isinstance(params, dict):
            raise ValidationError("Параметры генератора должны быть объектом.")
        n = _count(params, "n_samples", 180, 12, min(10000, MAX_ROWS))
        temporal = name in {"forecasting", "panel"}
        p = _count(params, "n_features", 1 if temporal else 2, 1 if temporal else 2, 100)
        seed = _count(params, "seed", 42, 0, 2**32 - 1)
        noise = _number(params, "noise", 0.03 if name == "classification" else 0.3, 0, 0.4 if name == "classification" else 100)
        effective_params = dict(params, n_samples=n, n_features=p, seed=seed, noise=noise)
        if n * (p + 3) > MAX_CELLS:
            raise ValidationError("Генератор превышает лимит числа ячеек.")
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(n, p))
        frame = pd.DataFrame(X, columns=[f"x{index + 1}" for index in range(p)])
        roles: dict = {}
        target: str | None = None
        reference_target: str | None = None
        if name == "classification":
            classes = _count(params, "n_classes", _count(params, "classes", 2, 2, 10), 2, 10)
            informative = max(2, int(np.ceil(np.log2(classes))))
            if informative > p:
                raise ValidationError(f"Для {classes} классов нужно минимум {informative} признака.")
            effective_params["n_classes"] = classes
            X, y = make_classification(n_samples=n, n_features=p, n_informative=informative, n_redundant=0, n_classes=classes, n_clusters_per_class=1, flip_y=noise, random_state=seed)
            frame = pd.DataFrame(X, columns=frame.columns)
            frame["target"] = [f"класс {value}" for value in y]
            reference_target = "target"
        elif name == "clustering":
            clusters = _count(params, "n_clusters", _count(params, "clusters", 3, 2, 20), 2, min(20, n))
            effective_params["n_clusters"] = clusters
            X, y = make_blobs(n_samples=n, n_features=p, centers=clusters, cluster_std=noise, random_state=seed)
            frame = pd.DataFrame(X, columns=frame.columns)
            frame["class_label"] = [f"группа {value}" for value in y]
            reference_target = "class_label"
            roles["reference_target"] = reference_target
        elif name == "ranking":
            queries = _count(params, "n_queries", min(12, n // 3), 3, min(1000, n // 3))
            effective_params["n_queries"] = queries
            query_codes = np.arange(n) % queries
            scores = 2 * X[:, 0] - X[:, 1] + rng.normal(scale=noise, size=n)
            relevance = np.zeros(n, dtype=int)
            for query in range(queries):
                positions = np.flatnonzero(query_codes == query)
                order = np.argsort(np.argsort(scores[positions], kind="stable"), kind="stable")
                relevance[positions] = np.minimum(4, (5 * order // len(positions)).astype(int))
            frame["query_id"] = [f"запрос {value:04}" for value in query_codes]
            frame["relevance"] = relevance
            target = "relevance"
            roles["query_column"] = "query_id"
        elif name in {"forecasting", "panel"}:
            entities = 1 if name == "forecasting" else _count(params, "n_entities", 5, 2, min(100, n // 6))
            effective_params["n_entities"] = entities
            effective_params["n_features"] = 1
            frame = self._temporal_frame(n, entities, noise, rng)
            target = "y"
            roles["time_column"] = "time"
            if name == "panel":
                roles["entity_column"] = "entity"
        elif name == "anomaly":
            fraction = _number(params, "contamination", 0.08, 0.01, 0.4)
            effective_params["contamination"] = fraction
            effective_params.pop("noise", None)
            count = max(1, int(round(n * fraction)))
            selected = rng.choice(n, count, replace=False)
            frame.iloc[selected] = rng.uniform(5, 9, size=(count, p)) * rng.choice([-1, 1], size=(count, p))
            selected_indices = set(selected.tolist())
            frame["is_anomaly"] = ["аномалия" if index in selected_indices else "обычная точка" for index in range(n)]
            reference_target = "is_anomaly"
            roles["reference_target"] = reference_target
        elif name == "reduction":
            latent = rng.normal(size=(n, 2))
            projection = rng.normal(size=(2, p))
            frame = pd.DataFrame(latent @ projection + rng.normal(scale=noise, size=(n, p)), columns=frame.columns)
        label, task, description = _GENERATORS[name]
        excluded = [reference_target] if reference_target else list(roles.values())
        metadata = {"source": "synthetic", "generator": name, "task": task, "tasks": [task], "task_target": target or reference_target, "excluded_features": excluded, "excluded_targets": [reference_target] if reference_target else [], "params": effective_params, "roles": roles, "description": description}
        records = frame.where(pd.notna(frame), None).to_dict(orient="records")
        return _metadata(self.legacy.load({"kind": "custom", "name": label, "rows": records, "target": target, "metadata": metadata}))

    @staticmethod
    def _temporal_frame(n: int, entities: int, noise: float, rng: np.random.Generator) -> pd.DataFrame:
        rows = []
        for entity in range(entities):
            length = n // entities + int(entity < n % entities)
            values = rng.normal(scale=0.1, size=length)
            for time in range(length):
                feature = float(np.sin(2 * np.pi * time / 12))
                previous = values[time - 1] if time else 0.0
                values[time] = 0.65 * previous + 2.5 * feature + 0.02 * time + entity + rng.normal(scale=noise)
                row = {"time": (pd.Timestamp("2024-01-01") + pd.Timedelta(days=time)).isoformat(), "x1": feature, "y": float(values[time])}
                if entities > 1:
                    row["entity"] = f"объект {entity + 1}"
                rows.append(row)
        return pd.DataFrame(rows)
