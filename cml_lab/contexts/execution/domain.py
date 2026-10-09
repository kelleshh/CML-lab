"""Run state transitions and one projection for hidden-test visibility."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any

from cml_lab.shared.domain import ConflictError, ValidationError, identifier, integer, json_object, new_id, text, utc_now

TERMINAL_STATUSES = frozenset({"completed", "error", "cancelled", "interrupted"})
_TRANSITIONS = {
    "queued": {"queued", "running", "cancelled", "error", "interrupted"},
    "running": {"running", "completed", "error", "cancelled", "interrupted"},
    **{status: {status} for status in TERMINAL_STATUSES},
}


def public_result(result: dict | None, test_revealed: bool = False) -> dict | None:
    if result is None:
        return None
    result = json_object(result, "Результат", maximum_bytes=100_000_000)
    if not test_revealed:
        evaluations = result.get("evaluations")
        if isinstance(evaluations, dict) and "test" in evaluations:
            evaluations["test"] = {"hidden": True}
        result["test_hidden"] = True
    else:
        result["test_hidden"] = False
    return result


@dataclass(frozen=True, init=False)
class Run:
    id: str
    _spec_json: str
    status: str
    progress: float
    message: str
    error: str | None
    _result_json: str | None
    test_revealed: bool
    name: str
    description: str
    created_at: str
    updated_at: str
    revision: int

    def __init__(self, id: str, spec: dict, status: str = "queued", progress: float = 0,
                 message: str = "", error: str | None = None, result: dict | None = None,
                 test_revealed: bool = False, name: str = "", description: str = "",
                 created_at: str | None = None, updated_at: str | None = None, revision: int = 1):
        if status not in _TRANSITIONS:
            raise ValidationError("Неизвестное состояние запуска.")
        if isinstance(progress, bool) or not isinstance(progress, (int, float)) or not math.isfinite(progress) or not 0 <= progress <= 1:
            raise ValidationError("Прогресс запуска должен быть числом между 0 и 1.")
        if not isinstance(test_revealed, bool):
            raise ValidationError("Признак открытия теста должен быть true или false.")
        if test_revealed and status != "completed":
            raise ConflictError("Тест можно открыть после успешного обучения.")
        now = utc_now()
        values = {"id": identifier(id), "_spec_json": json.dumps(json_object(spec, "Запрос расчета"), ensure_ascii=False, allow_nan=False),
                  "status": status, "progress": float(progress), "message": text(message, "Состояние", maximum=5000),
                  "error": None if error is None else text(error, "Ошибка", maximum=20000),
                  "_result_json": None if result is None else json.dumps(json_object(result, "Результат", maximum_bytes=100_000_000), ensure_ascii=False, allow_nan=False),
                  "test_revealed": test_revealed, "name": text(name, "Название", maximum=200),
                  "description": text(description, "Описание", maximum=5000), "created_at": created_at or now,
                  "updated_at": updated_at or now, "revision": integer(revision, "Ревизия запуска", 1, 2**31 - 1)}
        for key, value in values.items():
            object.__setattr__(self, key, value)

    @property
    def spec(self) -> dict:
        return json.loads(self._spec_json)

    @property
    def result(self) -> dict | None:
        return None if self._result_json is None else json.loads(self._result_json)

    @classmethod
    def create(cls, spec: dict) -> "Run":
        return cls(new_id(), spec, message="Расчет ожидает запуска.")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Run":
        fields = ("id", "spec", "status", "progress", "message", "error", "result", "test_revealed", "name", "description", "created_at", "updated_at", "revision")
        return cls(**{key: value[key] for key in fields if key in value})

    def to_dict(self) -> dict:
        return {"id": self.id, "spec": self.spec, "status": self.status, "progress": self.progress,
                "message": self.message, "error": self.error, "result": self.result, "test_revealed": self.test_revealed,
                "name": self.name, "description": self.description, "created_at": self.created_at,
                "updated_at": self.updated_at, "revision": self.revision}

    def public_dict(self) -> dict:
        result = self.to_dict()
        result["result"] = public_result(result["result"], self.test_revealed)
        return result

    def transition(self, status: str, **updates) -> "Run":
        if status not in _TRANSITIONS[self.status]:
            raise ConflictError(f"Запуск в состоянии «{self.status}» нельзя перевести в «{status}».")
        allowed = {"progress", "message", "error", "result", "test_revealed", "name", "description"}
        if set(updates) - allowed:
            raise ValidationError("Изменение запуска содержит неизвестные поля.")
        if self.status in TERMINAL_STATUSES:
            unchanged = {"progress": self.progress, "error": self.error, "result": self.result}
            if any(key in updates and updates[key] != value for key, value in unchanged.items()):
                raise ConflictError("Результат завершенного запуска неизменен. Создайте новый запуск.")
        if self.test_revealed and updates.get("test_revealed") is False:
            raise ConflictError("Открытый итоговый тест нельзя скрыть повторно.")
        if updates.get("progress", self.progress) < self.progress and status == "running":
            raise ConflictError("Прогресс выполняющегося запуска не должен уменьшаться.")
        values = self.to_dict()
        values.update(updates, status=status, revision=self.revision + 1, updated_at=utc_now())
        if status == "completed":
            values["progress"] = 1.0
        return Run.from_dict(values)
