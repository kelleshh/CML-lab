"""Каталог учебника. Внешние схемы передает composition root обычными словарями."""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any, Iterable, Mapping

from cml_lab.contexts.learning.domain import HelpEntry, Lesson, LessonSection, SourceReference
from . import learning_content as content


TASK_LESSONS = {
    "regression": "01-prediction", "classification": "classification-basics",
    "clustering": "clustering-basics", "ranking": "ranking-groups",
    "forecasting": "forecasting-lags", "panel": "panel-groups",
    "anomaly": "anomaly-detection", "reduction": "dimensionality-reduction",
}

METRIC_LESSONS = {
    "regression": "20-metrics-experiment", "classification": "classification-metrics",
    "clustering": "clustering-metrics", "ranking": "ranking-metrics",
    "forecasting": "forecasting-validation", "panel": "panel-groups",
    "anomaly": "anomaly-detection", "reduction": "dimensionality-reduction",
}


def _source(record: Mapping[str, Any]) -> SourceReference:
    title, url = str(record["title"]), str(record["url"])
    language = "ru" if any(site in url for site in ("yandex.ru", "habr.com/ru", "ru.wikipedia")) else "en"
    kind = "supplementary" if any(site in url for site in ("wikipedia.org", "habr.com", "mlwiki")) else "primary"
    return SourceReference(title, url, str(record.get("kind", kind)), str(record.get("language", language)))


def _lesson(record: Mapping[str, Any]) -> Lesson:
    return Lesson(
        id=record["id"], title=record["title"], summary=record["summary"],
        sections=tuple(LessonSection(part["title"], part["text"], part.get("formula")) for part in record["sections"]),
        sources=tuple(_source(part) for part in record["sources"]),
        tasks=tuple(record.get("tasks", ())), families=tuple(record.get("families", ())),
        level=record.get("level", "beginner"), example=record.get("example", ""),
        exercise=record.get("exercise", ""), common_mistakes=tuple(record.get("common_mistakes", ())),
        preset_json=json.dumps(record["preset"], ensure_ascii=False) if record.get("preset") is not None else None,
    )


def _legacy_lessons() -> list[dict[str, Any]]:
    from linear_lab.teaching import LESSONS

    records = deepcopy(LESSONS)
    for record in records:
        record["tasks"] = ["regression"]
        record["families"] = ["linear"]
        record["example"] = record["sections"][0]["text"]
        record["common_mistakes"] = [section["text"] for section in record["sections"]
                                     if "ошибк" in section["title"].casefold()]
        previous = record.get("preset")
        if previous and "experiment" not in previous:
            experiment = {key: value for key, value in previous.items() if key not in {"dataset", "model"}}
            experiment.update({"task": "regression", "algorithm_id": previous["model"]})
            record["preset"] = {"dataset": previous["dataset"], "experiment": experiment}
    return records


class InMemoryLearningRepository:
    def __init__(self, *, include_legacy: bool = True,
                 algorithms: Iterable[Mapping[str, Any]] = (),
                 stages: Iterable[Mapping[str, Any]] = (),
                 metrics: Iterable[Mapping[str, Any]] = ()) -> None:
        records = (_legacy_lessons() if include_legacy else []) + deepcopy(content.LESSONS)
        self._lessons: dict[str, Lesson] = {}
        for record in records:
            lesson = _lesson(record)
            if lesson.id in self._lessons:
                raise ValueError(f"Повторный идентификатор урока: {lesson.id}")
            self._lessons[lesson.id] = lesson
        self._help: dict[str, HelpEntry] = {}
        for record in content.HELP:
            self._add_help(record)
        self.register_schemas(algorithms=algorithms, stages=stages, metrics=metrics)

    def lessons(self) -> tuple[Lesson, ...]:
        return tuple(self._lessons.values())

    def lesson(self, identifier: str) -> Lesson | None:
        return self._lessons.get(identifier)

    def help_entry(self, key: str) -> HelpEntry | None:
        return self._help.get(key)

    def help_entries(self) -> tuple[HelpEntry, ...]:
        return tuple(self._help.values())

    def _add_help(self, record: Mapping[str, Any]) -> None:
        if record["lesson_id"] not in self._lessons:
            if record["lesson_id"].startswith(tuple(f"{n:02d}-" for n in range(1, 33))):
                return  # include_legacy=False is useful for embedding only the new curriculum.
            raise ValueError(f"Подсказка ссылается на отсутствующий урок: {record['lesson_id']}")
        self._help[record["key"]] = HelpEntry(
            key=record["key"], label=record["label"], summary=record["summary"],
            details=record.get("details", record["summary"]), example=record["example"],
            lesson_id=record["lesson_id"], sources=tuple(_source(item) for item in record["sources"]),
            effects=tuple(record.get("effects", ())), cautions=tuple(record.get("cautions", ())),
        )

    def register_schemas(self, *, algorithms: Iterable[Mapping[str, Any]] = (),
                         stages: Iterable[Mapping[str, Any]] = (),
                         metrics: Iterable[Mapping[str, Any]] = ()) -> None:
        """Связывает авторские объяснения с фактическими полями открытого каталога."""
        for definition in algorithms:
            self._register_definition(definition, "model")
        for definition in stages:
            self._register_definition(definition, "preprocessing")
        for metric in metrics:
            metric_id = metric.get("id", metric.get("key"))
            if not metric_id:
                continue
            tasks = metric.get("tasks", [metric.get("task", "regression")])
            lesson_id = metric.get("lesson_id", METRIC_LESSONS.get(next(iter(tasks), "regression")))
            if lesson_id not in self._lessons:
                raise ValueError(f"Метрика {metric_id} ссылается на отсутствующий урок: {lesson_id}")
            lesson = self._lessons[lesson_id]
            note = str(metric.get("help", metric.get("description", lesson.summary)))
            diagnostic = metric.get("diagnostic", False) or metric.get("optimizable") is False
            direction = metric.get("direction") or ("maximize" if metric.get("higher_is_better", True) else "minimize")
            direction_note = " Диагностический показатель; направление улучшения не задано." if diagnostic else (" Больше — лучше." if direction in {"maximize", "max"} else " Меньше — лучше.")
            self._add_help({
                "key": metric.get("help_key", f"metric.{metric.get('task', next(iter(tasks), 'regression'))}.{metric_id}"), "label": metric.get("label", metric.get("name", metric_id)),
                "summary": note, "details": note + direction_note,
                "example": lesson.example, "lesson_id": lesson.id,
                "sources": [vars_source(source) for source in lesson.sources],
                "cautions": ["Метрика считается только для доступного ответа модели и допустимых данных."],
            })

    def _register_definition(self, definition: Mapping[str, Any], namespace: str) -> None:
        identifier = str(definition["id"])
        lesson_id = definition.get("lesson_id")
        if not lesson_id or lesson_id not in self._lessons:
            raise ValueError(f"{identifier} ссылается на отсутствующий урок: {lesson_id}")
        lesson = self._lessons[lesson_id]
        sources = [vars_source(source) for source in lesson.sources]
        source_url = definition.get("source", definition.get("source_url"))
        if isinstance(source_url, str) and source_url.startswith("https://"):
            sources.insert(0, content.source(f"Параметры {definition.get('name', identifier)}", source_url))
        self._add_help({
            "key": definition.get("help_key", f"{namespace}.{identifier}"),
            "label": definition.get("name", definition.get("label", identifier)),
            "summary": definition.get("description", lesson.summary) or lesson.summary,
            "details": lesson.summary, "example": lesson.example, "lesson_id": lesson_id, "sources": sources,
            "cautions": [str(definition.get("reason"))] if definition.get("reason") else [],
        })
        for parameter in definition.get("params", []):
            key = parameter["key"]
            knowledge = content.parameter_help(identifier, str(definition.get("family", "")), key, namespace)
            help_id = parameter.get("help_key", f"{namespace}.{identifier}.{key}")
            linked_lesson = parameter.get("lesson_id", lesson_id)
            inline = parameter.get("help", knowledge.get("summary", ""))
            if not inline:
                raise ValueError(f"Поле {help_id} не имеет объяснения.")
            limits: list[str] = []
            if "default" in parameter:
                limits.append(f"Начальное значение: {parameter['default']!r}.")
            if "min" in parameter and "max" in parameter:
                limits.append(f"Допустимые границы формы: от {parameter['min']} до {parameter['max']}.")
            if parameter.get("options"):
                options = parameter["options"]
                limits.append("Варианты: " + ", ".join(str(option.get("label", option.get("value"))) if isinstance(option, dict) else str(option) for option in options) + ".")
            self._add_help({
                "key": help_id, "label": parameter.get("label", key), "summary": inline,
                "details": " ".join([knowledge.get("details", inline), *limits]),
                "example": knowledge.get("example", lesson.example), "lesson_id": linked_lesson,
                "sources": sources, "effects": knowledge.get("effects", []),
                "cautions": knowledge.get("cautions", []),
            })


def vars_source(source: SourceReference) -> dict[str, str]:
    return {"title": source.title, "url": source.url, "kind": source.kind, "language": source.language}
