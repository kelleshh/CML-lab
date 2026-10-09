"""Учебные материалы: неизменяемые записи без вычислительных зависимостей."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any


_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_HELP_KEY = re.compile(r"^[a-z0-9][a-zA-Z0-9._-]*$")


@dataclass(frozen=True, slots=True)
class SourceReference:
    title: str
    url: str
    kind: str = "primary"
    language: str = "en"

    def __post_init__(self) -> None:
        if not self.title.strip() or not self.url.startswith("https://"):
            raise ValueError("Источник требует название и HTTPS-ссылку.")


@dataclass(frozen=True, slots=True)
class LessonSection:
    title: str
    text: str
    formula: str | None = None
    anchor: str | None = None


@dataclass(frozen=True, slots=True)
class Lesson:
    id: str
    title: str
    summary: str
    sections: tuple[LessonSection, ...]
    sources: tuple[SourceReference, ...]
    tasks: tuple[str, ...] = ()
    families: tuple[str, ...] = ()
    level: str = "beginner"
    example: str = ""
    exercise: str = ""
    common_mistakes: tuple[str, ...] = ()
    preset_json: str | None = None

    def __post_init__(self) -> None:
        if not _IDENTIFIER.fullmatch(self.id):
            raise ValueError("Некорректный идентификатор урока.")
        if not self.title.strip() or not self.sections or not self.sources:
            raise ValueError("Урок требует название, объяснение и источник.")
        if self.preset_json is not None:
            if not isinstance(json.loads(self.preset_json), dict):
                raise ValueError("Пример запуска должен быть объектом.")

    @property
    def preset(self) -> dict[str, Any] | None:
        return json.loads(self.preset_json) if self.preset_json is not None else None


@dataclass(frozen=True, slots=True)
class HelpEntry:
    key: str
    label: str
    summary: str
    details: str
    example: str
    lesson_id: str
    sources: tuple[SourceReference, ...]
    effects: tuple[str, ...] = ()
    cautions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not _HELP_KEY.fullmatch(self.key) or not _IDENTIFIER.fullmatch(self.lesson_id):
            raise ValueError("Некорректная ссылка подсказки.")
        if not all((self.label.strip(), self.summary.strip(), self.example.strip(), self.sources)):
            raise ValueError("Подсказка требует объяснение, пример и источник.")
