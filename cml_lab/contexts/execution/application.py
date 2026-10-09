"""Start, cancel, inspect and retain runs through explicit execution ports."""

from contextlib import nullcontext
from typing import Callable

from .domain import Run, TERMINAL_STATUSES
from .ports import RunRepository, RunReferences, SchedulerPort
from cml_lab.shared.domain import ConflictError, identifier, integer, json_object, text
from cml_lab.shared.ports import MutationGuard


class ExecutionService:
    def __init__(self, repository: RunRepository, scheduler: SchedulerPort,
                 references: RunReferences | None = None,
                 cleanup: Callable[[str], None] | None = None,
                 mutation_guard: MutationGuard | None = None,
                 validate_input: Callable[[dict], None] | None = None):
        self.repository = repository
        self.scheduler = scheduler
        self.references = references
        self.cleanup = cleanup
        self.mutation_guard = mutation_guard
        self.validate_input = validate_input

    def start(self, spec: dict) -> Run:
        with self.mutation_guard.hold() if self.mutation_guard is not None else nullcontext():
            run = Run.create(spec)
            if self.validate_input is not None:
                self.validate_input(run.spec)
            run = self.repository.create(run)
        try:
            self.scheduler.start(run)
        except Exception as exc:
            # Failed scheduling is still a visible, reproducible failed run.
            failed = run.transition("error", message="Не удалось запустить расчет.", error=str(exc))
            self.repository.update(failed, expected_status="queued")
            raise
        return run

    def get(self, run_id: str) -> dict:
        run_id=identifier(run_id)
        state=self.repository.get(run_id).public_dict()
        events=getattr(self.scheduler,"events",None)
        state["events"]=events(run_id) if events is not None else []
        return state

    def list(self) -> dict:
        items = [run.public_dict() for run in self.repository.list()]
        return {"items": items, "total": len(items)}

    def cancel(self, run_id: str) -> dict:
        current = self.repository.get(identifier(run_id))
        if current.status in TERMINAL_STATUSES:
            return current.public_dict()
        cancelled = current.transition("cancelled", message="Расчет остановлен пользователем.")
        self.repository.update(cancelled, expected_status=current.status)
        self.scheduler.cancel(current.id)
        return cancelled.public_dict()

    def reveal_test(self, run_id: str) -> dict:
        current = self.repository.get(identifier(run_id))
        if current.status != "completed":
            raise ConflictError("Сначала завершите обучение, затем откройте итоговый тест.")
        if current.test_revealed:
            return current.public_dict()
        updated = current.transition(current.status, test_revealed=True)
        return self.repository.update(updated, expected_status=current.status).public_dict()

    def update_metadata(self, run_id: str, payload: dict) -> dict:
        current = self.repository.get(identifier(run_id))
        patch = json_object(payload, "Описание запуска")
        expected = patch.get("expected_revision", current.revision)
        integer(expected, "Ожидаемая ревизия", 1, 2**31 - 1)
        if expected != current.revision:
            raise ConflictError("Запуск изменился. Обновите форму перед сохранением.")
        updates = {key: text(patch[key], "Название" if key == "name" else "Описание", maximum=200 if key == "name" else 5000)
                   for key in ("name", "description") if key in patch}
        return self.repository.update(current.transition(current.status, **updates), expected_status=current.status).public_dict()

    def delete(self, run_id: str) -> dict:
        current = self.repository.get(identifier(run_id))
        if current.status not in TERMINAL_STATUSES:
            raise ConflictError("Остановите расчет перед удалением.")
        if self.references is not None and self.references.run_in_use(current.id):
            raise ConflictError("Модель используется сохраненным экспериментом. Сначала удалите его запись.")
        self.repository.delete(current.id)
        if self.cleanup is not None:
            self.cleanup(current.id)
        return {"id": current.id, "deleted": True}

    def close(self) -> None:
        self.scheduler.close()
