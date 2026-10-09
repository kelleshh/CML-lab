"""Каталог учебника. Внешние схемы передает composition root обычными словарями."""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any, Iterable, Mapping

from cml_lab.contexts.learning.domain import HelpEntry, Lesson, LessonSection, SourceReference
from . import learning_content as content
from .hyperparameter_learning import (model_article, parameter_article, parameter_knowledge,
                                      PIPELINE_CLASS_KNOWLEDGE)


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
        sections=tuple(LessonSection(part["title"], part["text"], part.get("formula"), part.get("anchor")) for part in record["sections"]),
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
                 metrics: Iterable[Mapping[str, Any]] = (),
                 pipeline_classes: Iterable[Mapping[str, Any]] = ()) -> None:
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
        self.register_pipeline_classes(pipeline_classes)

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
        if namespace == "model":
            tasks = set(definition.get("tasks", []))
            overview = next((item for item in content.LESSONS
                if str(definition.get("family", "")) in item.get("families", [])
                and tasks.intersection(item.get("tasks", []))), None)
            if lesson_id not in self._lessons:
                article = model_article(definition, overview)
                self._lessons[article["id"]] = _lesson(article)
            for field in definition.get("params", []):
                linked = field.get("lesson_id")
                if linked and linked not in self._lessons:
                    article = parameter_article(definition, field)
                    self._lessons[article["id"]] = _lesson(article)
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
            knowledge = parameter_knowledge(identifier, str(definition.get("family", "")), key, namespace,
                                            class_name=str(definition.get("class_path", definition.get("name", identifier))).rsplit(".", 1)[-1],
                                            library=str(definition.get("library", "")))
            help_id = parameter.get("help_key", f"{namespace}.{identifier}.{key}")
            linked_lesson = parameter.get("lesson_id", lesson_id)
            inline = knowledge.get("summary", parameter.get("help", ""))
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


    def register_pipeline_classes(self, definitions: Iterable[Mapping[str, Any]]) -> None:
        """Статьи классов, якоря параметров и самостоятельные ссылки Mini-IDE."""
        all_tasks = tuple(TASK_LESSONS)
        for raw in definitions:
            name = str(raw["name"])
            identifier = "pipeline-" + name.lower()
            source_url = f"https://scikit-learn.org/1.8/modules/generated/{raw['qualified_name']}.html"
            knowledge = PIPELINE_CLASS_KNOWLEDGE.get(name, {})
            description = knowledge.get("summary", raw.get("description", ""))
            # Вложенный estimator читается как модель, а не обещанный transformer.
            if "преобразователь или вложенная модель" in description:
                matched = next((lesson for lesson in self._lessons.values()
                                if lesson.id.startswith("model-") and lesson.title == name), None)
                description = matched.summary if matched else "Вложенный estimator используется для отбора признаков или допустимой композиции. Его fit обучается по текущему train; прямой transform возможен только при наличии метода transform."
            definition = {"id": identifier, "name": name, "class_path": raw["qualified_name"],
                          "library": "sklearn", "family": "preprocessing", "tasks": all_tasks,
                          "description": description, "params": raw.get("params", []),
                          "lesson_id": raw.get("lesson_id", identifier), "source": source_url}
            overview = {"summary": "fit учит состояние преобразователя по train; transform применяет сохраненное правило к новым строкам. Графическая ветвь и Python-декларация описывают тот же объект sklearn.",
                        "example": knowledge.get("example", "Строки train проходят fit_transform, контрольные строки проходят только transform. Число выходных координат проверяется в preview.")}
            article = model_article(definition, overview)
            for field in definition["params"]:
                parameter = dict(field)
                linked_article = parameter_article(definition, parameter, namespace="preprocessing")
                self._lessons[linked_article["id"]] = _lesson(linked_article)
                semantic = parameter_knowledge(identifier, "preprocessing", field["key"], "preprocessing", class_name=name, library="sklearn")
                article["sections"].append({"title": str(field["key"]),
                    "text": semantic["summary"] + " " + semantic["example"] + " " + " ".join(semantic.get("cautions", [])),
                    "anchor": "param-" + field["key"]})
                self._add_help({"key": field.get("help_key", f"pipeline.{name}.{field['key']}"),
                    "label": field["key"], "summary": semantic["summary"], "details": semantic.get("details", semantic["summary"]),
                    "example": semantic["example"], "lesson_id": linked_article["id"],
                    "sources": linked_article["sources"], "cautions": semantic.get("cautions", [])})
            self._lessons[article["id"]] = _lesson(article)
            self._add_help({"key": raw.get("help_key", f"pipeline.{name}"), "label": name,
                "summary": description, "details": description, "example": article["example"],
                "lesson_id": article["id"], "sources": article["sources"]})


def vars_source(source: SourceReference) -> dict[str, str]:
    return {"title": source.title, "url": source.url, "kind": source.kind, "language": source.language}
