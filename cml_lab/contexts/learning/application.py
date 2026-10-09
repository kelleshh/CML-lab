"""Поиск и чтение материалов. Возвращаемые значения принадлежат вызывающему коду."""

from dataclasses import asdict
from typing import Any

from .domain import HelpEntry, Lesson
from .ports import LearningRepository


def _summary(lesson: Lesson) -> dict[str, Any]:
    return {
        "id": lesson.id, "title": lesson.title, "summary": lesson.summary,
        "tasks": list(lesson.tasks), "families": list(lesson.families), "level": lesson.level,
        "source_count": len(lesson.sources), "preset_available": lesson.preset_json is not None,
    }


def _lesson_payload(lesson: Lesson) -> dict[str, Any]:
    payload = _summary(lesson)
    payload.update({
        "sections": [asdict(section) for section in lesson.sections],
        "sources": [asdict(source) for source in lesson.sources],
        "example": lesson.example, "exercise": lesson.exercise,
        "common_mistakes": list(lesson.common_mistakes), "preset": lesson.preset,
    })
    return payload


def _help_payload(entry: HelpEntry) -> dict[str, Any]:
    payload = asdict(entry)
    payload["sources"] = [asdict(source) for source in entry.sources]
    payload["source_urls"] = [source.url for source in entry.sources]
    payload["effects"] = list(entry.effects)
    payload["cautions"] = list(entry.cautions)
    return payload


class LearningService:
    def __init__(self, repository: LearningRepository) -> None:
        self._repository = repository

    def list_lessons(self, *, query: str = "", task: str | None = None,
                     family: str | None = None, level: str | None = None) -> dict[str, Any]:
        term = query.strip().casefold()
        lessons = (
            lesson for lesson in self._repository.lessons()
            if (not task or task in lesson.tasks)
            and (not family or family in lesson.families)
            and (not level or lesson.level == level)
            and (not term or term in " ".join((lesson.title, lesson.summary, lesson.example)).casefold())
        )
        items = [_summary(lesson) for lesson in lessons]
        return {"items": items, "total": len(items)}

    def get_lesson(self, identifier: str) -> dict[str, Any]:
        lesson = self._repository.lesson(identifier)
        if lesson is None:
            raise KeyError(f"Урок {identifier!r} отсутствует.")
        return _lesson_payload(lesson)

    def get_help(self, key: str) -> dict[str, Any]:
        entry = self._repository.help_entry(key)
        if entry is None:
            raise KeyError(f"Подсказка {key!r} отсутствует.")
        return _help_payload(entry)

    def list_help(self) -> dict[str, Any]:
        items = [_help_payload(entry) for entry in self._repository.help_entries()]
        return {"items": items, "total": len(items)}

    def catalogue(self) -> list[dict[str, Any]]:
        """Полные записи для клиентов, применяющих учебные рецепты из каталога."""
        return [(_summary(lesson) | {"preset": None}) if lesson.level == "reference"
                else _lesson_payload(lesson) for lesson in self._repository.lessons()]
