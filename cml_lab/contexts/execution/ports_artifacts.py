"""Portable exports and inference cross an explicit artifact gateway."""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ExportPayload:
    data: bytes
    mimetype: str
    filename: str

    @property
    def media_type(self) -> str:
        return self.mimetype


class ArtifactGateway(Protocol):
    def capabilities(self, run_id: str, public_run: dict) -> list[dict]: ...
    def export(self, run_id: str, format: str, public_run: dict) -> ExportPayload: ...
    def predict(self, run_id: str, rows: list[dict]) -> dict: ...
    def forecast(self, run_id: str, history: list[dict], future: list[dict]) -> dict: ...
