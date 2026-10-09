"""Export and run only server-owned, completed fitted artifacts."""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
import csv
from importlib import metadata
from io import BytesIO, StringIO
import json
from pathlib import Path
import pickle
import platform
import threading
from zipfile import ZIP_DEFLATED, ZipFile

import joblib
import numpy as np
import pandas as pd

from cml_lab.contexts.execution.ports_artifacts import ExportPayload
from cml_lab.shared.domain import NotFoundError, ValidationError, identifier


_FORMATS = (
    ("joblib", "Joblib · полный Python-конвейер"),
    ("pickle", "Pickle · полный Python-конвейер"),
    ("skops", "Skops · проверка типов перед загрузкой"),
    ("onnx", "ONNX · совместимая исходная схема и ответы"),
    ("bundle", "ZIP · модель, паспорт, зависимости и исходники"),
    ("passport", "JSON · паспорт модели"),
    ("json", "JSON · паспорт и открытые результаты"),
    ("csv", "CSV · открытые прогнозы запуска"),
)
_PACKAGES = ("numpy", "scipy", "pandas", "scikit-learn", "joblib", "skglm", "cvxpy",
             "xgboost-cpu", "xgboost", "lightgbm", "catboost", "imbalanced-learn",
             "linearmodels", "skops", "skl2onnx", "onnx", "onnxruntime", "threadpoolctl",
             "ImbalancedLearningRegression", "smogn", "tqdm")


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()
                if key not in {"artifact_path", "internal_path"}}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _json(value):
    return json.dumps(_plain(value), ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")


def _versions():
    result = {"python": platform.python_version()}
    for package in _PACKAGES:
        try:
            result[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            pass
    return result


class LocalArtifactGateway:
    """Numerical persistence stays outside domain and application services."""

    def __init__(self, root, catalogue=None, data_gateway=None):
        self.root = Path(root).resolve()
        self.directory = self.root / "artifacts"
        self.catalogue = catalogue
        self.data = data_gateway
        self._cache: OrderedDict[str, dict] = OrderedDict()
        self._cache_bytes = 0
        self._lock = threading.RLock()

    def _path(self, run_id):
        run_id = identifier(run_id)
        path = self.directory / f"{run_id}.joblib"
        if path.is_symlink() or self.directory.is_symlink() or path.resolve().parent != self.directory.resolve():
            raise ValidationError("Артефакт должен находиться в собственном хранилище завершенных запусков.")
        if not path.is_file():
            raise NotFoundError("Обученный артефакт не найден; восстановите его из сохраненного экспорта или обучите рецепт заново.")
        if path.stat().st_size > 256 * 1024 * 1024:
            raise ValidationError("Экспорт в интерактивной лаборатории ограничен артефактом до 256 МБ.")
        return path

    def _load(self, run_id):
        return joblib.load(self._path(run_id))

    def _sample(self, artifact, public_run):
        if self.data is None:
            raise ValueError("Для проверки ONNX нужен доступ к исходному сохраненному датасету.")
        dataset_id = public_run.get("spec", {}).get("dataset_id")
        frame = self.data.frame(dataset_id)
        if not len(frame):
            raise ValueError("Исходный датасет не содержит строк для проверки ONNX.")
        selected = np.unique(np.linspace(0, len(frame) - 1, min(16, len(frame)), dtype=int))
        return frame.iloc[selected].loc[:, artifact.features].copy()

    def _prepare(self, run_id, public_run):
        path = self._path(run_id)
        stamp = (path.stat().st_mtime_ns, path.stat().st_size)
        with self._lock:
            cached = self._cache.get(run_id)
            if cached is not None and cached["stamp"] == stamp:
                self._cache.move_to_end(run_id)
                return cached
            artifact = self._load(run_id)
            result = {"stamp": stamp, "skops": None, "onnx": None,
                      "skops_reason": None, "onnx_reason": None, "trusted_types": [], "onnx_schema": None}
            try:
                import skops.io as sio
                result["skops"] = sio.dumps(artifact)
                result["trusted_types"] = sio.get_untrusted_types(data=result["skops"])
            except Exception as exc:
                result["skops_reason"] = "Skops не сохранил полный артефакт: " + str(exc)
            try:
                from .onnx_artifacts import convert_artifact
                if getattr(artifact, "task", None) in {"forecasting", "panel", "reduction", "anomaly"}:
                    raise ValueError("Этот тип артефакта требует истории или особого ответа, для которого нет проверенного ONNX-конвертера. Используйте Joblib/ZIP.")
                if not hasattr(artifact, "estimator") or not type(artifact.estimator).__module__.startswith("sklearn."):
                    raise ValueError("Внешняя/структурная модель не имеет проверенного конвертера полного конвейера. Используйте Joblib/ZIP.")
                result["onnx"], result["onnx_schema"] = convert_artifact(artifact, self._sample(artifact, public_run))
            except Exception as exc:
                result["onnx_reason"] = "ONNX недоступен для полного конвейера: " + str(exc)
            size = sum(len(result[key] or b"") for key in ("skops", "onnx"))
            result["size"] = size
            # Capability probes must not retain unlimited serialized forests.
            if size <= 50 * 1024 * 1024:
                old = self._cache.pop(run_id, None)
                if old is not None:
                    self._cache_bytes -= old["size"]
                while self._cache and (len(self._cache) >= 4 or self._cache_bytes + size > 50 * 1024 * 1024):
                    _, removed = self._cache.popitem(last=False)
                    self._cache_bytes -= removed["size"]
                self._cache[run_id] = result
                self._cache_bytes += size
            return result

    def capabilities(self, run_id: str, public_run: dict) -> list[dict]:
        prepared = self._prepare(run_id, public_run)
        return [{"format": kind, "label": label,
                 "available": prepared[kind] is not None if kind in {"skops", "onnx"} else True,
                 "reason": prepared[f"{kind}_reason"] if kind in {"skops", "onnx"} else None}
                for kind, label in _FORMATS]

    def _passport(self, run_id, public_run, prepared=None):
        artifact = self._load(run_id)
        prepared = prepared or self._prepare(run_id, public_run)
        public_run = _plain(deepcopy(public_run))
        return {"schema": "cml-lab.artifact.v1", "run_id": run_id,
                "task": public_run["spec"].get("task"), "run": public_run,
                "raw_input": {"features": list(getattr(artifact, "features", [])),
                              "temporal": _plain(getattr(artifact, "temporal", {})),
                              "classes": _plain(getattr(artifact, "classes_", None))},
                "versions": _versions(), "skops_untrusted_types": list(prepared["trusted_types"]),
                "skops_loading": "Проверьте перечисленные типы и исходники перед явным выбором trusted; код прогноза не доверяет им автоматически.",
                "onnx_input_schema": prepared["onnx_schema"],
                "format_limits": {kind: prepared[f"{kind}_reason"] for kind in ("skops", "onnx")
                                  if prepared[f"{kind}_reason"]},
                "test_visibility": "revealed" if public_run.get("test_revealed") else "hidden",
                "source": "https://scikit-learn.org/1.8/model_persistence.html"}

    def export(self, run_id: str, format: str, public_run: dict) -> ExportPayload:
        if format not in {kind for kind, _ in _FORMATS}:
            raise ValidationError("Неизвестный формат экспорта.")
        path = self._path(run_id)
        filename = f"cml-model-{run_id[:8]}"
        if format == "joblib":
            return ExportPayload(path.read_bytes(), "application/octet-stream", filename + ".joblib")
        if format == "pickle":
            return ExportPayload(pickle.dumps(self._load(run_id), protocol=5), "application/octet-stream", filename + ".pkl")
        if format == "csv":
            return ExportPayload(self._csv(public_run), "text/csv; charset=utf-8", filename + "-predictions.csv")
        prepared = self._prepare(run_id, public_run)
        if format in {"skops", "onnx"}:
            if prepared[format] is None:
                raise ValidationError(prepared[f"{format}_reason"])
            return ExportPayload(prepared[format], "application/octet-stream", filename + "." + format)
        passport = self._passport(run_id, public_run, prepared)
        if format in {"passport", "json"}:
            return ExportPayload(_json(passport), "application/json; charset=utf-8", filename + "-passport.json")
        return ExportPayload(self._bundle(path, passport, prepared), "application/zip", filename + ".zip")

    @staticmethod
    def _csv(public_run):
        rows = []
        result = public_run.get("result") or {}
        for split, evaluation in (result.get("evaluations") or {}).items():
            if not isinstance(evaluation, dict) or evaluation.get("hidden"):
                continue
            for row in evaluation.get("rows", []):
                rows.append({"split": split, **_plain(row)})
        keys = ["split", "index", "actual", "predicted"]
        keys.extend(sorted({key for row in rows for key in row} - set(keys)))
        output = StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})
        return b"\xef\xbb\xbf" + output.getvalue().encode("utf-8")

    @staticmethod
    def _bundle(path, passport, prepared):
        output = BytesIO()
        source_root = Path(__file__).resolve().parents[2]
        with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
            archive.writestr("model.joblib", path.read_bytes())
            archive.writestr("passport.json", _json(passport))
            archive.writestr("inference.py", _INFERENCE)
            archive.writestr("README.txt", _README)
            packages = dict(passport["versions"])
            packages.pop("python", None)
            if "xgboost-cpu" in packages:
                packages.pop("xgboost", None)
            archive.writestr("requirements.txt", "\n".join(f"{package}=={version}" for package, version in packages.items()) + "\n")
            for kind in ("skops", "onnx"):
                if prepared[kind] is not None:
                    archive.writestr("model." + kind, prepared[kind])
            for package in ("cml_lab", "linear_lab"):
                for source in sorted((source_root / package).rglob("*")):
                    relative = source.relative_to(source_root)
                    if not source.is_file() or source.is_symlink() or any(part.startswith(".") or part == "__pycache__" for part in relative.parts):
                        continue
                    if source.suffix == ".py" or source.name.endswith("LICENSE") or source.name in {"PROVENANCE.md", "upstream-sha256.json"}:
                        archive.write(source, relative.as_posix())
            for name in ("THIRD_PARTY_NOTICES.md", "LICENSE"):
                if (source_root / name).is_file():
                    archive.write(source_root / name, name)
        return output.getvalue()

    def predict(self, run_id: str, rows: list[dict]) -> dict:
        artifact = self._load(run_id)
        if getattr(artifact, "task", None) in {"forecasting", "panel"}:
            raise ValidationError("Для этой модели передайте историю и будущие строки через прогноз временного ряда.")
        if getattr(artifact, "task", None) == "reduction":
            return {"task": "reduction", "coordinates": _plain(artifact.reduce(rows))}
        predicted = artifact.predict(rows)
        result = {"task": artifact.task, "predictions": _plain(predicted)}
        if artifact.task == "classification" and hasattr(artifact.estimator, "predict_proba"):
            result.update(probabilities=_plain(artifact.predict_proba(rows)), classes=_plain(artifact.classes_))
        return result

    def forecast(self, run_id: str, history: list[dict], future: list[dict]) -> dict:
        from .ml.temporal import TemporalArtifact
        artifact = self._load(run_id)
        if isinstance(artifact, TemporalArtifact):
            temporal = artifact
        elif getattr(artifact, "task", None) in {"forecasting", "panel"} and getattr(artifact, "temporal", None):
            temporal = TemporalArtifact(artifact, artifact.temporal)
        else:
            raise ValidationError("В артефакте отсутствует сохраненный протокол временного прогноза.")
        forecast = temporal.forecast(pd.DataFrame(history), pd.DataFrame(future))
        return {"task": getattr(artifact, "task", "forecasting"), "rows": _plain(forecast.to_dict(orient="records"))}


_README = """CML-lab: экспорт обученной модели

Архив содержит полный обученный препроцессор, модель и код для исходных строк.
Установите зависимости в отдельном Python-окружении: python -m pip install -r requirements.txt
Прогноз для CSV: python inference.py --input rows.csv --output predictions.csv
Ряд/панель: python inference.py --history history.csv --future future.csv --output forecast.csv
Используйте тот же Python, который указан в passport.json. Windows/macOS/Linux поддерживаются библиотечными колесами; LightGBM на macOS может потребовать OpenMP.

Joblib/Pickle загружают исполняемые Python-объекты: используйте собственный проверенный экспорт.
Для Skops сначала прочитайте passport.json/skops_untrusted_types и проверьте исходники типов.
inference.py никогда автоматически не доверяет неизвестным Skops-типам.
ONNX включается только при совместимости полного конвейера и проверке исходных строк.
Его схема входа указана в passport.json; строковые категории ONNX не представляют null.
Закрытые тестовые метрики и строки не попадают в паспорт до явного открытия теста.
Исходные файлы датасета в архив не включаются.
"""

_INFERENCE = '''"""Inference for a trusted CML-lab bundle; no training is performed."""
import argparse
from pathlib import Path
import joblib
import pandas as pd
from cml_lab.infrastructure.ml.temporal import TemporalArtifact

def main():
    parser = argparse.ArgumentParser(description="Прогноз собственной экспортированной модели CML-lab")
    parser.add_argument("--input", help="Исходные строки CSV для обычного прогноза")
    parser.add_argument("--history", help="Известная история CSV для ряда/панели")
    parser.add_argument("--future", help="Будущие строки CSV для ряда/панели")
    parser.add_argument("--output", default="predictions.csv")
    args = parser.parse_args()
    artifact = joblib.load(Path(__file__).resolve().parent / "model.joblib")
    if args.history or args.future:
        if not args.history or not args.future:
            parser.error("Для прогноза ряда одновременно задайте --history и --future.")
        temporal = artifact if isinstance(artifact, TemporalArtifact) else TemporalArtifact(artifact, artifact.temporal)
        output = temporal.forecast(pd.read_csv(args.history), pd.read_csv(args.future))
    elif args.input:
        rows = pd.read_csv(args.input)
        if artifact.task in {"forecasting", "panel"}:
            parser.error("Этой модели нужны --history и --future.")
        if artifact.task == "reduction":
            output = pd.DataFrame(artifact.reduce(rows))
        else:
            output = rows.copy()
            output["prediction"] = artifact.predict(rows)
    else:
        parser.error("Задайте --input либо пару --history и --future.")
    output.to_csv(args.output, index=False)

if __name__ == "__main__":
    main()
'''
