"""Editable recipe heads and immutable revisions of portable configurations."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from cml_lab.shared.domain import ConflictError, ValidationError, identifier, integer, json_object, new_id, text, utc_now

RECIPE_KINDS = ("model", "preprocessor", "project")
MAX_PREPARATION_STAGES = 32


@dataclass(frozen=True)
class PreparationStage:
    adapter_id: str
    columns: tuple[str, ...] = ()
    enabled: bool = True

    @classmethod
    def from_dict(cls, value: dict) -> "PreparationStage":
        value = json_object(value, "Шаг подготовки")
        adapter = text(value.get("adapter_id", ""), "Преобразование", required=True)
        columns = value.get("columns", [])
        if not isinstance(columns, list) or not all(isinstance(column, str) and column for column in columns) or len(set(columns)) != len(columns):
            raise ValidationError("Для шага подготовки выберите уникальные названия колонок.")
        if not isinstance(value.get("enabled", True), bool):
            raise ValidationError("Переключатель шага подготовки должен быть true или false.")
        json_object(value.get("params", {}), "Параметры преобразования")
        return cls(adapter, tuple(columns), value.get("enabled", True))


def validate_config(kind: str, config: dict) -> dict:
    if kind not in RECIPE_KINDS:
        raise ValidationError("Доступны рецепты моделей, подготовки и проектов.")
    config = json_object(config)
    if kind == "model":
        text(config.get("algorithm_id", ""), "Алгоритм", maximum=200, required=True)
        json_object(config.get("params", {}), "Параметры модели")
    elif kind == "preprocessor":
        steps = config.get("steps", [])
        if not isinstance(steps, list) or len(steps) > MAX_PREPARATION_STAGES:
            raise ValidationError(f"Рецепт подготовки содержит список из не более {MAX_PREPARATION_STAGES} шагов.")
        for stage in steps:
            PreparationStage.from_dict(stage)
        json_object(config.get("resampling", {}), "Пересэмплирование")
    return config


@dataclass(frozen=True, init=False)
class Recipe:
    id: str
    kind: str
    name: str
    description: str
    revision: int
    _config_json: str
    created_at: str
    updated_at: str
    deleted: bool

    def __init__(self, id: str, kind: str, name: str, description: str, revision: int,
                 config: dict, created_at: str, updated_at: str, deleted: bool = False):
        values = {"id": identifier(id), "kind": kind, "name": text(name, "Название", required=True),
                  "description": text(description, "Описание", maximum=5000),
                  "revision": integer(revision, "Ревизия", 1, 2**31 - 1),
                  "_config_json": json.dumps(validate_config(kind, config), ensure_ascii=False, allow_nan=False),
                  "created_at": created_at, "updated_at": updated_at, "deleted": bool(deleted)}
        for key, value in values.items():
            object.__setattr__(self, key, value)

    @property
    def config(self) -> dict:
        return json.loads(self._config_json)

    @classmethod
    def create(cls, kind: str, payload: dict) -> "Recipe":
        payload = json_object(payload, "Рецепт")
        now = utc_now()
        return cls(new_id(), kind, payload.get("name", ""), payload.get("description", ""), 1,
                   payload.get("config", {}), now, now)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Recipe":
        return cls(**{key: value[key] for key in ("id", "kind", "name", "description", "revision", "config", "created_at", "updated_at")},
                   deleted=value.get("deleted", False))

    def to_dict(self) -> dict:
        return {"id": self.id, "kind": self.kind, "name": self.name, "description": self.description,
                "revision": self.revision, "config": self.config, "created_at": self.created_at,
                "updated_at": self.updated_at, "deleted": self.deleted}

    def revise(self, patch: dict) -> "Recipe":
        patch = json_object(patch, "Изменения рецепта")
        if self.deleted:
            raise ConflictError("Рецепт удален. Создайте новый рецепт из сохраненной копии.")
        expected = patch.get("expected_revision", self.revision)
        integer(expected, "Ожидаемая ревизия", 1, 2**31 - 1)
        if expected != self.revision:
            raise ConflictError("Рецепт изменился. Обновите форму перед сохранением.")
        return Recipe(self.id, self.kind, patch.get("name", self.name), patch.get("description", self.description),
                      self.revision + 1, patch.get("config", self.config), self.created_at, utc_now())
