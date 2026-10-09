"""Small domain vocabulary shared by the five bounded contexts."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
import json
import re
from typing import Any
from uuid import uuid4


class ValidationError(ValueError):
    pass


class ConflictError(ValueError):
    pass


class NotFoundError(FileNotFoundError):
    pass


class TaskKind(str, Enum):
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    CLUSTERING = "clustering"
    RANKING = "ranking"
    FORECASTING = "forecasting"
    PANEL = "panel"
    ANOMALY = "anomaly"
    REDUCTION = "reduction"

    @classmethod
    def parse(cls, value: Any) -> "TaskKind":
        try:
            return cls(value)
        except (ValueError, TypeError) as exc:
            raise ValidationError("Выберите задачу из каталога CML-lab.") from exc


def new_id() -> str:
    return uuid4().hex


def identifier(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{32}", value):
        raise ValidationError("Неверный идентификатор: выберите сохраненный объект из библиотеки.")
    return value


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def text(value: Any, label: str, *, maximum: int = 200, required: bool = False) -> str:
    if not isinstance(value, str) or len(value) > maximum:
        raise ValidationError(f"{label}: нужен текст длиной до {maximum} символов.")
    result = value.strip()
    if required and not result:
        raise ValidationError(f"{label}: введите непустое название.")
    return result


def json_object(value: Any, label: str = "Настройки", *, maximum_bytes: int = 1_000_000) -> dict:
    """Accept portable JSON and sever mutable references at application boundaries."""
    if not isinstance(value, dict):
        raise ValidationError(f"{label}: нужен объект настроек, например {{}}.")
    try:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValidationError(f"{label}: разрешены конечные числа, строки и обычные JSON-объекты.") from exc
    if len(encoded.encode("utf-8")) > maximum_bytes:
        raise ValidationError(f"{label}: объем настроек превышает {maximum_bytes} байт.")
    return json.loads(encoded)


def integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValidationError(f"{label}: нужно целое число от {minimum} до {maximum}.")
    return value
