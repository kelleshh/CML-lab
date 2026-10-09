"""Export an executable project with its resolved intent and immutable data."""

from __future__ import annotations

import base64
from copy import deepcopy
import gzip
from importlib import metadata
import json
import platform
from pprint import pformat
import re

import numpy as np
import pandas as pd

from cml_lab import __version__
from cml_lab.contexts.execution.ports_artifacts import ExportPayload
from cml_lab.shared.domain import ValidationError, json_object


MAX_SNAPSHOT_BYTES = 50 * 1024 * 1024
_EXPORT_FORMATS = {"py", "ipynb"}
_PACKAGES = ("numpy", "scipy", "pandas", "scikit-learn", "joblib", "optuna", "skglm", "cvxpy",
             "xgboost", "xgboost-cpu", "lightgbm", "catboost", "imbalanced-learn",
             "ImbalancedLearningRegression", "smogn")
_LOCAL_FIELDS = {"artifact_path", "artifact_dir", "artifact_root", "recipe_snapshots",
                 "project_id", "project_recipe_id", "model_recipe_id", "preprocessor_recipe_id"}


def _json(value, *, indent=None):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, indent=indent)


def _plain_cell(value):
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(value)
    if not isinstance(value, (str, bool, int, float)):
        raise ValidationError("Экспорт проекта поддерживает табличные числа, строки, bool и пропуски.")
    return value


def _snapshot(frame):
    result = {
        "columns": list(frame.columns),
        "dtypes": {name: str(frame[name].dtype) for name in frame},
        "data": [[_plain_cell(value) for value in row]
                 for row in frame.itertuples(index=False, name=None)],
    }
    data = _json(result).encode("utf-8")
    if len(data) > MAX_SNAPSHOT_BYTES:
        raise ValidationError("Встроенный снимок проекта ограничен 50 МБ JSON. Уменьшите набор для экспорта .py/.ipynb.")
    return base64.b64encode(gzip.compress(data, mtime=0)).decode("ascii")


def _versions():
    values = {"cml-lab": __version__, "python": platform.python_version()}
    for package in _PACKAGES:
        try:
            values[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            pass
    return values


_IMPORTS = '''from copy import deepcopy
import argparse
import base64
import gzip
import json
from pathlib import Path
import tempfile

import pandas as pd

from cml_lab.contexts.execution.domain import public_result
from cml_lab.infrastructure.ml.catalogue import AlgorithmCatalogue
from cml_lab.infrastructure.ml.engine import ExperimentEngine
'''

_DATA_FUNCTIONS = '''def load_dataset(csv_path=None):
    """Load the original snapshot, or an explicitly selected replacement CSV."""
    if csv_path is not None:
        return pd.read_csv(csv_path)
    snapshot = json.loads(gzip.decompress(base64.b64decode(DATA_SNAPSHOT)))
    frame = pd.DataFrame(snapshot["data"], columns=snapshot["columns"])
    for name, dtype in snapshot["dtypes"].items():
        if dtype == "object":
            frame[name] = frame[name].astype(object)
        elif dtype.startswith("datetime64"):
            frame[name] = pd.to_datetime(frame[name])
        else:
            frame[name] = frame[name].astype(dtype)
    return frame


class SnapshotData:
    def __init__(self, frame):
        self._frame = frame.copy(deep=True)

    def frame(self, dataset_id):
        if dataset_id != SPEC["dataset_id"]:
            raise ValueError("This project contains one immutable dataset snapshot.")
        return self._frame.copy(deep=True)
'''

_TRAINING_FUNCTION = '''def run_project(output_dir=None, csv_path=None, reveal_test=False):
    """Pipeline -> training-only tuning -> training -> serialization.

    CML-lab rebuilds preprocessing inside every training fold. Validation and
    test rows never fit preprocessing, resampling, or parameter search.
    Unsupervised tasks instead fit their selected working table, as in the GUI.
    """
    output = Path(output_dir) if output_dir is not None else Path(
        tempfile.mkdtemp(prefix="cml-project-", dir="."))
    output.mkdir(parents=True, exist_ok=True)
    for name in ("artifact.joblib", "result.json", "project.json"):
        if (output / name).exists():
            raise FileExistsError(f"{output / name} already exists; choose another output directory.")
    spec = deepcopy(SPEC)
    spec["artifact_path"] = str(output / "artifact.joblib")
    engine = ExperimentEngine(SnapshotData(load_dataset(csv_path)), AlgorithmCatalogue())

    def progress(event):
        if event.get("message"):
            print(event["message"])

    result = engine.run(spec, progress=progress)
    visible = public_result(result, test_revealed=reveal_test)
    (output / "result.json").write_text(
        json.dumps(visible, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
    (output / "project.json").write_text(
        json.dumps({"metadata": PROJECT_METADATA, "spec": SPEC},
                   ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
    print(f"Saved project: {output.resolve()}")
    return {"output_dir": str(output.resolve()), "result": visible}
'''

_CLI = '''def main():
    parser = argparse.ArgumentParser(description="Execute the exported CML-lab project")
    parser.add_argument("--output", type=Path, help="New output directory; saved results are never overwritten")
    parser.add_argument("--dataset", type=Path, help="Replacement CSV; defaults to the embedded original snapshot")
    parser.add_argument("--reveal-test", action="store_true", help="Explicitly include held-out test metrics")
    options = parser.parse_args()
    run_project(options.output, options.dataset, options.reveal_test)


if __name__ == "__main__":
    main()
'''


class LocalProjectExporter:
    """Generate source files; exporting never fits a model or executes user code."""

    def __init__(self, data_gateway, catalogue=None):
        self.data = data_gateway
        self.catalogue = catalogue

    @staticmethod
    def formats():
        return [{"format": name, "available": True, "requires": "CML-lab Python environment"}
                for name in ("py", "ipynb")]

    def export(self, spec, format="py", *, name="CML-lab project", origin=None):
        if format not in _EXPORT_FORMATS:
            raise ValidationError("Проект экспортируется в py или ipynb.")
        spec = deepcopy(json_object(spec, "Настройки экспортируемого проекта"))
        for field in _LOCAL_FIELDS:
            spec.pop(field, None)
        if not spec.get("dataset_id") or not spec.get("algorithm_id"):
            raise ValidationError("Для экспорта проекта выберите датасет и модель.")
        snapshot = _snapshot(self.data.frame(spec["dataset_id"]))
        metadata_value = {"format": "cml.project", "version": 1, "name": str(name)[:200],
                          "versions": _versions(), "origin": deepcopy(origin or {}),
                          "execution": "CML-lab ExperimentEngine; training-only preprocessing and search",
                          "snapshot": "Embedded JSON data compressed with gzip; no pickle data"}
        configuration = (
            "# Exact resolved configuration; names and parameters match the saved project.\n"
            f"PROJECT_METADATA = {pformat(metadata_value, width=100, sort_dicts=False)}\n"
            f"SPEC = {pformat(spec, width=100, sort_dicts=False)}\n"
            "# Original table values and dtypes. This is data, never executable Python.\n"
            f"DATA_SNAPSHOT = {snapshot!r}\n"
        )
        inspection = (
            "# The declarative pipeline is stored in SPEC and compiled by CML-lab at training time.\n"
            "# Printing it does not fit transformers or execute its source.\n"
            "pipeline_config = SPEC.get('preprocessing', {}).get('declarative_pipeline')\n"
            "if pipeline_config:\n"
            "    print(pipeline_config['source'])\n"
            "else:\n"
            "    print(SPEC.get('preprocessing', {}))\n"
        )
        title = "# CML-lab executable project\n# Run with the Python environment of the CML-lab source checkout.\n"
        if format == "py":
            source = "\n\n".join((title, _IMPORTS, configuration, _DATA_FUNCTIONS,
                                      inspection, _TRAINING_FUNCTION, _CLI))
            data = source.encode("utf-8")
            mimetype = "text/x-python; charset=utf-8"
        else:
            cells = []

            def cell(kind, source):
                item = {"cell_type": kind, "id": f"cml-{len(cells) + 1}", "metadata": {}, "source": source}
                if kind == "code":
                    item.update(execution_count=None, outputs=[])
                cells.append(item)

            cell("markdown", "# CML-lab: исполнимый проект\n\n"
                 "Сохраните Notebook рядом с `run.py` и используйте Python-окружение исходников CML-lab. Запускайте ячейки сверху вниз. "
                 "Исходный датасет встроен в файл. `SPEC` хранит Pipeline, Tuning и настройки Training. "
                 "`run_project()` сохраняет обученный конвейер в `artifact.joblib`, описание в `project.json` "
                 "и доступные метрики в `result.json`.\n\n"
                 "В задачах с учителем подготовка и поиск обучаются только на train; "
                 "test скрыт, пока явно не указан `reveal_test=True`. Обучение без учителя использует "
                 "всю выбранную рабочую таблицу, без оценки новых объектов. Временные и панельные "
                 "задачи используют сохраненные лаги, окна и хронологические части. "
                 "Экспорт сохраняет границы задачи и не заменяет их обычным случайным разбиением.")
            cell("code", _IMPORTS)
            cell("markdown", "## Dataset и настройки проекта\n\n"
                 "Снимок таблицы сохраняет названия, значения, пропуски и типы столбцов. "
                 "Изменения таблицы в лаборатории после экспорта не меняют этот снимок.")
            cell("code", configuration + "\n" + _DATA_FUNCTIONS + "\nframe = load_dataset()\nframe.head()\n")
            cell("markdown", "## Pipeline\n\n"
                 "Декларативный Python описывает преобразования признаков. Каждый fit происходит "
                 "внутри своей обучающей части, включая части поиска параметров.")
            cell("code", inspection + "\n"
                 "if pipeline_config:\n"
                 "    from cml_lab.infrastructure.ml.pipelines import compile_pipeline\n"
                 "    preprocessing_pipeline = compile_pipeline(pipeline_config)\n"
                 "else:\n"
                 "    preprocessing_pipeline = SPEC.get('preprocessing', {})\n"
                 "preprocessing_pipeline\n")
            cell("markdown", "## Tuning → Training → Serialization\n\n"
                 "`SPEC['search']` задает поиск. `None` запускает обучение без поиска. "
                 "Сохраненное зерно, CV, роли столбцов и ограничения алгоритма передаются реальному "
                 "движку CML-lab. Каждый вызов создает новую папку результатов.")
            cell("code", _TRAINING_FUNCTION)
            cell("code", "outputs = run_project()\noutputs['result']\n")
            notebook = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
                        "metadata": {"kernelspec": {"display_name": "Python 3 (CML-lab environment)",
                                                     "language": "python", "name": "python3"},
                                     "language_info": {"name": "python", "version": platform.python_version()},
                                     "cml_lab": metadata_value}}
            data = _json(notebook, indent=1).encode("utf-8")
            mimetype = "application/x-ipynb+json"
        filename = re.sub(r"[^a-zA-Z0-9_-]+", "-", str(name)).strip("-")[:70] or "cml-project"
        return ExportPayload(data, mimetype, filename + "." + format)
