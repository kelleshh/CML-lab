"""Task roles, evaluation plans and reproducible experiment snapshots."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any

from cml_lab.contexts.data.domain import Dataset
from cml_lab.shared.domain import ConflictError, TaskKind, ValidationError, identifier, integer, json_object, new_id, text, utc_now

SUPERVISED_TASKS = frozenset({TaskKind.REGRESSION, TaskKind.CLASSIFICATION, TaskKind.RANKING, TaskKind.FORECASTING, TaskKind.PANEL})
TEMPORAL_TASKS = frozenset({TaskKind.FORECASTING, TaskKind.PANEL})
UNSUPERVISED_TASKS = frozenset(set(TaskKind) - SUPERVISED_TASKS)
ROLE_KEYS = ("time_column", "entity_column", "query_column", "weight_column", "reference_target")


@dataclass(frozen=True)
class ColumnRoles:
    time_column: str | None = None
    entity_column: str | None = None
    query_column: str | None = None
    weight_column: str | None = None
    reference_target: str | None = None

    @classmethod
    def from_dict(cls, value: dict | None) -> "ColumnRoles":
        value = json_object(value or {}, "Роли колонок")
        if set(value) - set(ROLE_KEYS):
            raise ValidationError("Неизвестная роль колонки.")
        for name, column in value.items():
            if column is not None:
                text(column, "Колонка роли", maximum=500, required=True)
        return cls(**value)

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(getattr(self, key) for key in ROLE_KEYS if getattr(self, key) is not None)

    def to_dict(self) -> dict:
        return {key: getattr(self, key) for key in ROLE_KEYS}


@dataclass(frozen=True)
class DatasetSelection:
    dataset_id: str
    task: TaskKind
    features: tuple[str, ...]
    target: str | None
    roles: ColumnRoles

    @classmethod
    def resolve(cls, dataset: Dataset, task: TaskKind, payload: dict) -> "DatasetSelection":
        roles = ColumnRoles.from_dict(payload.get("roles"))
        default_target=(dataset.task_target or dataset.default_target) if task==TaskKind.CLASSIFICATION else dataset.default_target
        target = payload.get("target", default_target if task in SUPERVISED_TASKS else None)
        if target is not None:
            text(target, "Целевая колонка", maximum=500, required=True)
        if task in UNSUPERVISED_TASKS:
            reference = roles.reference_target or target or dataset.task_target
            roles = ColumnRoles.from_dict(dict(roles.to_dict(), reference_target=reference))
            target = None
        names = set(dataset.column_names)
        if target is not None and target not in names or any(column not in names for column in roles.columns):
            raise ValidationError("Цель или служебная колонка отсутствует в выбранном наборе.")
        if task in SUPERVISED_TASKS and target is None:
            raise ValidationError("Выберите колонку, которую модель должна предсказывать.")
        if target is not None and target in roles.columns:
            raise ValidationError("Целевая колонка не может одновременно быть служебной колонкой.")
        required = {TaskKind.RANKING: ("query_column",), TaskKind.FORECASTING: ("time_column",),
                    TaskKind.PANEL: ("time_column", "entity_column")}.get(task, ())
        for role in required:
            if getattr(roles, role) is None:
                labels = {"query_column": "идентификатор запроса", "time_column": "время наблюдения", "entity_column": "идентификатор объекта"}
                raise ValidationError(f"Для задачи «{task.value}» выберите {labels[role]}.")
        if task == TaskKind.PANEL and roles.time_column == roles.entity_column:
            raise ValidationError("Время и идентификатор объекта панели должны быть разными колонками.")
        if task in {TaskKind.REGRESSION, TaskKind.RANKING, TaskKind.FORECASTING, TaskKind.PANEL}:
            if not next(column.numeric for column in dataset.columns if column.name == target):
                raise ValidationError("Для этой задачи нужна числовая целевая колонка. Для категорий выберите классификацию.")
        group_column = json_object(payload.get("validation") or {}, "Кросс-валидация").get("group_column")
        if group_column is not None:
            if group_column not in names or group_column == target:
                raise ValidationError("Колонка групп кросс-валидации отсутствует или совпадает с целью.")
        excluded = set(roles.columns) | {target, group_column}
        requested = payload.get("features")
        if requested is None:
            requested = [column.name for column in dataset.columns if column.name not in excluded]
        if not isinstance(requested, list) or len(requested) > 1000 or not all(isinstance(column, str) and column for column in requested):
            raise ValidationError("Выберите список из не более 1000 признаков.")
        if len(requested) != len(set(requested)) or any(column not in names for column in requested):
            raise ValidationError("Выберите уникальные признаки из колонок набора.")
        features = tuple(column for column in requested if column not in excluded)
        if not features and task not in TEMPORAL_TASKS:
            raise ValidationError("После исключения цели и служебных колонок не осталось признаков.")
        return cls(identifier(dataset.id), task, features, target, roles)

    def to_dict(self) -> dict:
        return {"dataset_id": self.dataset_id, "task": self.task.value, "features": list(self.features),
                "target": self.target, "roles": self.roles.to_dict()}


@dataclass(frozen=True)
class SplitPlan:
    train: float = 0.6
    validation: float = 0.2
    test: float = 0.2
    shuffle: bool = True

    @classmethod
    def from_dict(cls, value: dict | None, task: TaskKind) -> "SplitPlan":
        value = json_object(value or {}, "Разбиение данных")
        shares = {name: value.get(name, default) for name, default in (("train", 0.6), ("validation", 0.2), ("test", 0.2))}
        if any(isinstance(share, bool) or not isinstance(share, (int, float)) or not math.isfinite(share) or not 0 < share < 1 for share in shares.values()):
            raise ValidationError("Каждой части данных задайте долю больше 0 и меньше 1.")
        if not math.isclose(sum(shares.values()), 1.0, rel_tol=0, abs_tol=1e-8):
            raise ValidationError("Доли обучения, проверки и теста должны в сумме составлять 1.")
        shuffle = value.get("shuffle", True)
        if not isinstance(shuffle, bool):
            raise ValidationError("Перемешивание строк должно быть true или false.")
        return cls(**shares, shuffle=False if task in TEMPORAL_TASKS else shuffle)

    def to_dict(self) -> dict:
        return {"train": self.train, "validation": self.validation, "test": self.test, "shuffle": self.shuffle}


def validate_validation(value: dict | None, task: TaskKind) -> dict:
    value = json_object(value or {"strategy": "none"}, "Кросс-валидация")
    strategy = text(value.get("strategy", "none"), "Способ кросс-валидации", required=True)
    if task in TEMPORAL_TASKS and strategy not in {"none", "timeseries", "time_series", "rolling_origin", "expanding_window"}:
        raise ValidationError("Для временных и панельных рядов выберите хронологическую кросс-валидацию.")
    if "folds" in value:
        integer(value["folds"], "Число частей кросс-валидации", 2, 100)
    if "gap" in value:
        integer(value["gap"], "Пропуск между частями", 0, 10000)
    value["strategy"] = strategy
    return value


def validate_temporal(value: dict | None) -> dict:
    value = json_object(value or {}, "Временные признаки")
    result = {"lags": value.get("lags", [1, 2, 3]), "rolling_windows": value.get("rolling_windows", [3]), "horizon": value.get("horizon", 1)}
    for name in ("lags", "rolling_windows"):
        sequence = result[name]
        if not isinstance(sequence, list) or len(sequence) > 100:
            raise ValidationError("Лаги и окна задаются списками из не более 100 значений.")
        for size in sequence:
            integer(size, "Размер лага или окна", 1, 10000)
        if len(sequence) != len(set(sequence)):
            raise ValidationError("Лаги и размеры окон не должны повторяться.")
    integer(result["horizon"], "Горизонт прогноза", 1, 1000)
    return result


@dataclass(frozen=True, init=False)
class Experiment:
    """A retained computation owns its resolved spec, not live recipe references."""

    id: str
    name: str
    description: str
    run_id: str
    revision: int
    _spec_json: str
    _result_json: str
    created_at: str
    updated_at: str

    def __init__(self, id: str, name: str, spec: dict, result: dict, run_id: str,
                 description: str = "", revision: int = 1, created_at: str | None = None, updated_at: str | None = None):
        now = utc_now()
        values = {"id": identifier(id), "name": text(name, "Название эксперимента", required=True),
                  "description": text(description, "Описание", maximum=5000), "run_id": identifier(run_id),
                  "revision": integer(revision, "Ревизия", 1, 2**31 - 1),
                  "_spec_json": json.dumps(json_object(spec, "Снимок настроек"), ensure_ascii=False, allow_nan=False),
                  "_result_json": json.dumps(json_object(result, "Результат", maximum_bytes=100_000_000), ensure_ascii=False, allow_nan=False),
                  "created_at": created_at or now, "updated_at": updated_at or now}
        for key, value in values.items():
            object.__setattr__(self, key, value)

    @property
    def spec(self) -> dict:
        return json.loads(self._spec_json)

    @property
    def result(self) -> dict:
        return json.loads(self._result_json)

    @classmethod
    def create(cls, name: str, spec: dict, result: dict, run_id: str) -> "Experiment":
        return cls(new_id(), name, spec, result, run_id)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Experiment":
        return cls(**{key: value[key] for key in ("id", "name", "spec", "result", "run_id", "description", "revision", "created_at", "updated_at") if key in value})

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "description": self.description, "run_id": self.run_id,
                "revision": self.revision, "spec": self.spec, "result": self.result,
                "created_at": self.created_at, "updated_at": self.updated_at}

    def revise_metadata(self, patch: dict) -> "Experiment":
        patch = json_object(patch, "Описание эксперимента")
        if set(patch) - {"name", "description", "expected_revision"}:
            raise ValidationError("Расчет эксперимента неизменен. Менять можно название и описание.")
        expected = patch.get("expected_revision", self.revision)
        integer(expected, "Ожидаемая ревизия", 1, 2**31 - 1)
        if expected != self.revision:
            raise ConflictError("Эксперимент изменился. Обновите форму перед сохранением.")
        return Experiment(self.id, patch.get("name", self.name), self.spec, self.result, self.run_id,
                          patch.get("description", self.description), self.revision + 1, self.created_at, utc_now())
