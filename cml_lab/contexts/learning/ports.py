"""Порт чтения учебника. Каталог алгоритмов не является зависимостью учебника."""

from typing import Protocol

from .domain import HelpEntry, Lesson


class LearningRepository(Protocol):
    def lessons(self) -> tuple[Lesson, ...]: ...

    def lesson(self, identifier: str) -> Lesson | None: ...

    def help_entry(self, key: str) -> HelpEntry | None: ...

    def help_entries(self) -> tuple[HelpEntry, ...]: ...
