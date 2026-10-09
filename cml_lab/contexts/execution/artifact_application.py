"""Expose completed artifacts without bypassing run visibility rules."""

from .ports import RunRepository
from .ports_artifacts import ArtifactGateway, ExportPayload
from cml_lab.shared.domain import ConflictError, ValidationError, identifier, json_object


class ArtifactApplication:
    def __init__(self, repository: RunRepository, gateway: ArtifactGateway):
        self.repository = repository
        self.gateway = gateway

    def _completed(self, run_id: str):
        run = self.repository.get(identifier(run_id))
        if run.status != "completed":
            raise ConflictError("Экспорт и прогноз доступны после успешного завершения обучения.")
        return run

    def capabilities(self, run_id: str) -> dict:
        run = self._completed(run_id)
        return {"formats": self.gateway.capabilities(run.id, run.public_dict())}

    def export(self, run_id: str, format: str = "joblib") -> ExportPayload:
        run = self._completed(run_id)
        if not isinstance(format, str) or len(format) > 30:
            raise ValidationError("Выберите формат экспорта из списка.")
        return self.gateway.export(run.id, format, run.public_dict())

    @staticmethod
    def _rows(value, label: str, maximum: int) -> list[dict]:
        if not isinstance(value, list) or not 1 <= len(value) <= maximum:
            raise ValidationError(f"{label}: нужен список от 1 до {maximum} строк.")
        json_object({"rows": value}, label, maximum_bytes=5_000_000)
        return [json_object(row, label, maximum_bytes=100_000) for row in value]

    def predict(self, run_id: str, rows: list[dict]) -> dict:
        run = self._completed(run_id)
        return self.gateway.predict(run.id, self._rows(rows, "Строки прогноза", 2000))

    def forecast(self, run_id: str, history: list[dict], future: list[dict]) -> dict:
        run = self._completed(run_id)
        if run.spec.get("task") not in {"forecasting", "panel"}:
            raise ValidationError("Прогноз по истории доступен временным и панельным моделям.")
        return self.gateway.forecast(run.id, self._rows(history, "История ряда", 10000),
                                     self._rows(future, "Будущие строки", 2000))
