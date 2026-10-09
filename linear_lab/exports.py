"""Export a trusted, fitted raw-input pipeline without accepting model uploads.

ONNX conversion is conservative: unsupported preprocessing disables that format
instead of silently exporting only the regression coefficient vector. Python
formats retain the complete fitted preprocessing and prediction pipeline.
"""

from __future__ import annotations

from copy import deepcopy
from importlib import metadata
from io import BytesIO
import json
import pickle
from pathlib import Path
import platform
import re
import threading
from zipfile import ZIP_DEFLATED, ZipFile

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


_FORMATS = (
    ("joblib", "Joblib · полный Python-пайплайн"),
    ("pickle", "Pickle · полный Python-пайплайн"),
    ("skops", "Skops · проверка типов перед загрузкой"),
    ("onnx", "ONNX · полный пайплайн для ONNX Runtime"),
    ("bundle", "ZIP · модель, паспорт, зависимости и код"),
    ("passport", "JSON · паспорт модели, без исполнения"),
)
_PACKAGES = (
    "numpy", "scipy", "pandas", "scikit-learn", "joblib", "skglm",
    "cvxpy", "skops", "skl2onnx", "onnx", "onnxruntime", "threadpoolctl",
    "smogn", "ImbalancedLearningRegression",
)
_SOURCE_URLS = {
    "persistence": "https://scikit-learn.org/stable/model_persistence.html",
    "skops": "https://skops.readthedocs.io/en/stable/persistence.html",
    "onnx": "https://onnx.ai/sklearn-onnx/auto_tutorial/plot_gbegin_dataframe.html",
    "custom_conversion": "https://onnx.ai/sklearn-onnx/auto_tutorial/plot_jfunction_transformer.html",
}


def _versions():
    result = {"python": platform.python_version()}
    for package in _PACKAGES:
        try:
            result[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            pass
    return result


def _json_bytes(value):
    def convert(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(f"Неизвестный тип в паспорте модели: {type(item).__name__}.")

    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=convert).encode("utf-8")


class _OnnxNumericFunction(TransformerMixin, BaseEstimator):
    """An export-only adapter for the project's four explicit numeric functions."""

    def __init__(self, mode="finite", bounds=None):
        self.mode = mode
        self.bounds = bounds

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        values = np.asarray(X, dtype=float)
        if self.mode == "finite":
            return np.where(np.isfinite(values), values, np.nan)
        if self.mode == "log1p":
            return np.sign(values) * np.log1p(np.abs(values))
        if self.mode == "sqrt":
            return np.sign(values) * np.sqrt(np.abs(values))
        if self.mode == "clip":
            return np.clip(values, self.bounds[0], self.bounds[1])
        if self.mode == "impute":
            return np.where(np.isnan(values), self.bounds, values)
        return values


_converter_lock = threading.RLock()


def _register_numeric_converter():
    """Use ordinary ONNX arithmetic; no Python callback is left in the graph."""
    from onnx import TensorProto
    from skl2onnx import update_registered_converter
    from skl2onnx.common.data_types import DoubleTensorType, FloatTensorType

    def shape(operator):
        operator.outputs[0].type = deepcopy(operator.inputs[0].type)

    def converter(scope, operator, container):
        source, output = operator.inputs[0].full_name, operator.outputs[0].full_name
        dtype = TensorProto.DOUBLE if isinstance(operator.inputs[0].type, DoubleTensorType) else TensorProto.FLOAT
        value_type = np.float64 if dtype == TensorProto.DOUBLE else np.float32

        def constant(name, values):
            identifier = scope.get_unique_variable_name(name)
            values = np.asarray(values, dtype=value_type)
            container.add_initializer(identifier, dtype, list(values.shape), values.ravel().tolist())
            return identifier

        mode = operator.raw_operator.mode
        if mode == "finite":
            infinite = scope.get_unique_variable_name("is_infinite")
            missing = scope.get_unique_variable_name("is_missing")
            nonfinite = scope.get_unique_variable_name("is_nonfinite")
            container.add_node("IsInf", source, infinite)
            container.add_node("IsNaN", source, missing)
            container.add_node("Or", [infinite, missing], nonfinite)
            container.add_node("Where", [nonfinite, constant("nan", np.nan), source], output)
        elif mode in {"log1p", "sqrt"}:
            absolute = scope.get_unique_variable_name("absolute")
            sign = scope.get_unique_variable_name("sign")
            transformed = scope.get_unique_variable_name("transformed")
            container.add_node("Abs", source, absolute)
            container.add_node("Sign", source, sign)
            if mode == "log1p":
                shifted = scope.get_unique_variable_name("shifted")
                container.add_node("Add", [absolute, constant("one", 1.0)], shifted)
                container.add_node("Log", shifted, transformed)
            else:
                container.add_node("Sqrt", absolute, transformed)
            container.add_node("Mul", [sign, transformed], output)
        elif mode == "clip":
            clipped = scope.get_unique_variable_name("clipped_lower")
            container.add_node("Max", [source, constant("lower", operator.raw_operator.bounds[0])], clipped)
            container.add_node("Min", [clipped, constant("upper", operator.raw_operator.bounds[1])], output)
        elif mode == "impute":
            missing = scope.get_unique_variable_name("is_missing")
            container.add_node("IsNaN", source, missing)
            container.add_node("Where", [missing, constant("training_fill_values", operator.raw_operator.bounds), source], output)
        else:
            container.add_node("Identity", source, output)

    with _converter_lock:
        update_registered_converter(_OnnxNumericFunction, "LinearLabNumericFunction", shape, converter)


def _onnx_preprocessor(preprocessor):
    """Separate custom ColumnTransformer selection while preserving fitted state."""
    from .preprocessing import (
        FeatureEngineeringTransformer, PreservingKNNImputer, QuantileClipper,
        _finite_numeric, _require_finite, _signed_log1p, _signed_sqrt, _string_categories,
    )

    if not isinstance(preprocessor, ColumnTransformer):
        raise ValueError("ONNX: подготовка данных должна быть ColumnTransformer.")
    standard = ColumnTransformer([])
    standard.__dict__.update(deepcopy(preprocessor.__dict__))
    numeric, categorical = [], []
    adapted = []
    for name, transformer, columns in standard.transformers_:
        if isinstance(transformer, str):
            if transformer == "passthrough":
                raise ValueError("ONNX: неописанные исходные столбцы passthrough не поддерживаются.")
            adapted.append((name, transformer, columns))
            continue
        if not isinstance(transformer, Pipeline):
            raise ValueError(f"ONNX: неизвестная подготовка ветки {name}.")
        branch = deepcopy(transformer)
        steps = []
        is_categorical = any(isinstance(step, FunctionTransformer) and step.func is _string_categories for _, step in branch.steps)
        (categorical if is_categorical else numeric).extend(columns)
        for step_name, step in branch.steps:
            if isinstance(step, PreservingKNNImputer):
                raise ValueError("ONNX не поддерживает выбранное KNN-заполнение. Экспортируй Joblib, Skops или ZIP.")
            if isinstance(step, QuantileClipper):
                step = _OnnxNumericFunction("clip", np.asarray(step.bounds_))
            if isinstance(step, FunctionTransformer):
                functions = {_finite_numeric: "finite", _require_finite: "identity", _signed_log1p: "log1p", _signed_sqrt: "sqrt"}
                if step.func in functions:
                    step = _OnnxNumericFunction(functions[step.func])
                elif step.func is _string_categories:
                    # ONNX string tensors are already strings and have no null.
                    # The accepted raw schema is explicit in the model metadata.
                    continue
                elif step.func is not None:
                    raise ValueError("ONNX: пользовательская функция не имеет проверенного конвертера.")
            if is_categorical and step_name == "impute":
                # A string input has no missing value representation. Nulls must
                # be rejected by the caller, never encoded as an empty category.
                continue
            if step_name == "impute" and getattr(step, "add_indicator", False):
                raise ValueError("ONNX: индикаторы пропусков SimpleImputer пока не поддержаны. Выбери Python-формат или отключи индикаторы.")
            if isinstance(step, SimpleImputer):
                if not np.isfinite(step.statistics_).all():
                    raise ValueError("ONNX: заполнение удаляет полностью пустые признаки. Выбери Python-формат.")
                # ONNX Runtime's ai.onnx.ml Imputer has no float64 kernel. Use
                # exact learned fill values in regular tensor operations instead.
                step = _OnnxNumericFunction("impute", np.asarray(step.statistics_))
            steps.append((step_name, step))
        branch.steps = steps
        adapted.append((name, branch, columns))
    standard.transformers_ = adapted
    standard.transformers = [(name, transformer, columns) for name, transformer, columns in adapted if name != "remainder"]
    selectors = deepcopy(getattr(preprocessor, "selection_steps_", [])) if isinstance(preprocessor, FeatureEngineeringTransformer) else []
    return standard, selectors, numeric, categorical


class ModelExportService:
    """Export only artifacts created by JobManager inside this service root."""

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self._lock = threading.RLock()
        self._prepared = {}

    def _artifact(self, job_id):
        if not isinstance(job_id, str) or re.fullmatch(r"[0-9a-f]{32}", job_id) is None:
            raise FileNotFoundError("Обученная модель не найдена.")
        jobs = self.root / "jobs"
        directory = jobs / job_id
        artifact = directory / "model.joblib"
        if directory.is_symlink() or artifact.is_symlink() or not artifact.is_file():
            raise FileNotFoundError("Обученная модель не найдена.")
        if jobs.resolve() != jobs or directory.resolve().parent != jobs or artifact.resolve().parent != directory:
            raise FileNotFoundError("Обученная модель не найдена.")
        return artifact

    def _load(self, job_id):
        artifact = self._artifact(job_id)
        pipeline = joblib.load(artifact)
        if not isinstance(pipeline, Pipeline) or "preprocessing" not in pipeline.named_steps or "model" not in pipeline.named_steps:
            raise ValueError("Артефакт не является полным обученным пайплайном Linear Lab.")
        return pipeline

    def _record(self, job_id, name):
        path = self._artifact(job_id).parent / name
        if path.is_symlink() or not path.is_file():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def _schema(self, pipeline):
        preprocessor = pipeline.named_steps["preprocessing"]
        names = list(getattr(preprocessor, "feature_names_in_", []))
        numeric, categorical = set(), set()
        vocabularies = {}
        from .preprocessing import _string_categories
        for _, transformer, columns in preprocessor.transformers_:
            if isinstance(transformer, Pipeline):
                categories = any(isinstance(step, FunctionTransformer) and step.func is _string_categories for _, step in transformer.steps)
                (categorical if categories else numeric).update(columns)
                if categories and "onehot" in transformer.named_steps:
                    for column, values in zip(columns, transformer.named_steps["onehot"].categories_):
                        vocabularies[column] = [str(value) for value in values if not pd.isna(value)]
        schema = [{"name": str(name), "type": "string" if name in categorical else "number", "role": "feature"} for name in names]
        for item in schema:
            if item["name"] in vocabularies:
                item["known_categories"] = vocabularies[item["name"]]
                item["unknown_categories"] = "Новая категория кодируется нулевым one-hot вектором."
        return schema

    def _passport(self, job_id, pipeline, result, request, skops_types=None):
        clean_request = {key: value for key, value in request.items() if key != "artifact_path"}
        return {
            "schema_version": 1, "job_id": job_id,
            "model": result.get("model", request.get("model")),
            "model_name": result.get("model_name"),
            "target": result.get("target_name", request.get("target")),
            "inputs": self._schema(pipeline),
            "transformed_features": result.get("feature_names", pipeline.named_steps["preprocessing"].get_feature_names_out().tolist()),
            "preprocessing": result.get("preprocessing", request.get("preprocessing", {})),
            "resampling": result.get("resampling", request.get("resampling", {})),
            "metrics": result.get("metrics", {}),
            "configuration": clean_request, "versions": _versions(),
            "skops_required_trusted_types": skops_types or [],
            "prediction_contract": "Подай pandas.DataFrame с исходными признаками из inputs. Подготовка данных уже обучена и повторно не fit-ится. Пересэмплирование выполнялось только при обучении.",
            "python_loading": "Python-форматы требуют совместимых версий зависимостей и пакета linear_lab. Загружай только собственные доверенные артефакты.",
            "sources": _SOURCE_URLS,
        }

    def _skops(self, pipeline):
        try:
            import skops.io as sio
        except ImportError as error:
            raise ValueError("Skops не установлен. Установи зависимости экспорта и перезапусти сервис.") from error
        try:
            data = sio.dumps(pipeline, compression=ZIP_DEFLATED)
            required = sorted(sio.get_untrusted_types(data=data))
        except Exception as error:
            raise ValueError(f"Skops не поддерживает этот пайплайн: {type(error).__name__}: {error}") from error
        return data, required

    def _probe(self, job_id, request, numeric, categorical, pipeline):
        """Use raw feature rows for parity, including numeric NaN/±inf probes."""
        from .datasets import DataService
        rows = None
        if request.get("dataset_id"):
            try:
                bundle = DataService(self.root / "datasets").resolve(request["dataset_id"], request.get("target"), request.get("features"))
                rows = bundle.X.head(32).copy()
            except (FileNotFoundError, ValueError):
                pass
        if rows is None:
            # Fitted category vocabulary and imputer/scaler statistics suffice
            # for testing conversion without storing raw training observations.
            columns = {}
            preprocessor = pipeline.named_steps["preprocessing"]
            for name in numeric:
                columns[name] = np.asarray([0.0, 1.0, -1.0, 3.5])
            for _, branch, selected in preprocessor.transformers_:
                if isinstance(branch, Pipeline) and "onehot" in branch.named_steps:
                    for name, values in zip(selected, branch.named_steps["onehot"].categories_):
                        finite = [str(value) for value in values if not pd.isna(value)]
                        columns[name] = [finite[index % len(finite)] if finite else "__unknown__" for index in range(4)]
            rows = pd.DataFrame(columns)
        for name in categorical:
            rows[name] = rows[name].map(lambda value: str(value) if not pd.isna(value) else None)
        if categorical:
            rows = rows.dropna(subset=categorical)
            if rows.empty:
                raise ValueError("ONNX: в исходных строковых признаках нет строк без пропусков для проверки экспорта.")
        names = self._schema(pipeline)
        rows = rows[[item["name"] for item in names]].reset_index(drop=True)
        extras = []
        numeric_branch = next((branch for _, branch, selected in pipeline.named_steps["preprocessing"].transformers_ if isinstance(branch, Pipeline) and set(selected).intersection(numeric)), None)
        if numeric and numeric_branch is not None and "impute" in numeric_branch.named_steps:
            for value in (np.nan, np.inf, -np.inf):
                example = rows.iloc[[0]].copy()
                example[numeric[0]] = value
                extras.append(example)
        return pd.concat([rows, *extras], ignore_index=True) if extras else rows

    def _onnx(self, job_id, pipeline, request):
        try:
            import onnxruntime as ort
            ort.disable_telemetry_events()
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import DoubleTensorType, StringTensorType
        except ImportError as error:
            raise ValueError("ONNX требует skl2onnx, onnx и onnxruntime. Установи зависимости экспорта.") from error
        _register_numeric_converter()
        preprocessor, selectors, numeric, categorical = _onnx_preprocessor(pipeline.named_steps["preprocessing"])
        standard = Pipeline([("preprocessing", preprocessor)] + [(f"selection{index}", selector) for index, selector in enumerate(selectors)] + [("model", deepcopy(pipeline.named_steps["model"]))])
        schema = [(item["name"], StringTensorType([None, 1]) if item["type"] == "string" else DoubleTensorType([None, 1])) for item in self._schema(pipeline)]
        imputed = set()
        for _, branch, selected in pipeline.named_steps["preprocessing"].transformers_:
            if isinstance(branch, Pipeline) and "impute" in branch.named_steps:
                imputed.update(selected)
        try:
            converted = convert_sklearn(standard, "LinearLabRawInputPipeline", initial_types=schema, target_opset={"": 18, "ai.onnx.ml": 3})
            descriptor = {"input_schema": [{"name": name, "type": "string" if name in categorical else "float64", "shape": [None, 1], "nullable": name not in categorical and name in imputed} for name, _ in schema], "categorical_contract": "Строковые признаки: только строки, без null/NaN. Нормализация строк включена типом ONNX StringTensor. Для входов с пропусками категорий используй Python-форматы." if categorical else None, "numeric_contract": "NaN и ±inf в числовых признаках заменяются обученным заполнителем, если заполнение включено. При отключённом заполнении подавай только конечные числа.", "raw_input_pipeline": True}
            prop = converted.metadata_props.add()
            prop.key, prop.value = "linear_lab_schema", _json_bytes(descriptor).decode("utf-8")
            data = converted.SerializeToString()
            options = ort.SessionOptions()
            options.intra_op_num_threads = options.inter_op_num_threads = 1
            session = ort.InferenceSession(data, sess_options=options, providers=["CPUExecutionProvider"])
            rows = self._probe(job_id, request, numeric, categorical, pipeline)
            feed = {name: rows[[name]].to_numpy(dtype=object if name in categorical else np.float64) for name, _ in schema}
            actual = np.asarray(session.run(None, feed)[0]).reshape(-1)
            expected = np.asarray(pipeline.predict(rows)).reshape(-1)
            if not np.isfinite(actual).all() or not np.allclose(actual, expected, rtol=2e-5, atol=2e-5):
                raise ValueError("Предсказания ONNX отличаются от исходного полного пайплайна. Такой экспорт отключён.")
            descriptor["parity_check"] = {"rows": len(rows), "rtol": 2e-5, "atol": 2e-5, "max_absolute_error": float(np.max(np.abs(actual - expected)))}
            return data, descriptor
        except ValueError:
            raise
        except Exception as error:
            raise ValueError(f"ONNX не поддерживает выбранную модель или подготовку данных ({type(error).__name__}). Выбери Joblib, Skops или ZIP. Деталь: {str(error)[:280]}") from error

    def _prepare(self, job_id):
        artifact = self._artifact(job_id)
        signature = (artifact.stat().st_mtime_ns, artifact.stat().st_size)
        with self._lock:
            cached = self._prepared.get(job_id)
            if cached and cached["signature"] == signature:
                return cached
            pipeline = self._load(job_id)
            request = self._record(job_id, "request.json")
            prepared = {"signature": signature, "skops": None, "onnx": None, "reasons": {}}
            for name, method in (("skops", lambda: self._skops(pipeline)), ("onnx", lambda: self._onnx(job_id, pipeline, request))):
                try:
                    prepared[name] = method()
                except ValueError as error:
                    prepared["reasons"][name] = str(error)
            # Caches retain at most eight modest fitted-model export blobs.
            if len(self._prepared) >= 8:
                self._prepared.pop(next(iter(self._prepared)))
            self._prepared[job_id] = prepared
            return prepared

    def capabilities(self, job_id):
        prepared = self._prepare(job_id)
        result = []
        for name, label in _FORMATS:
            available = name not in prepared["reasons"]
            reason = prepared["reasons"].get(name)
            if name == "skops" and available and prepared["skops"][1]:
                reason = "Перед загрузкой проверь типы из паспорта. Для custom linear_lab типов нужен код из ZIP-пакета."
            if name == "onnx" and available:
                reason = prepared["onnx"][1].get("categorical_contract") or "Полный пайплайн проверен через ONNX Runtime на исходных признаках; тип числовых входов float64."
            result.append({"format": name, "label": label, "available": available, "reason": reason})
        return result

    def export(self, job_id, format, result, request):
        if format not in dict(_FORMATS):
            raise ValueError("Неизвестный формат экспорта модели.")
        artifact = self._artifact(job_id)
        pipeline = self._load(job_id)
        filename = f"linear-lab-{job_id[:8]}"
        if format == "joblib":
            return artifact.read_bytes(), filename + ".joblib", "application/octet-stream"
        if format == "pickle":
            return pickle.dumps(pipeline, protocol=pickle.HIGHEST_PROTOCOL), filename + ".pkl", "application/octet-stream"
        prepared = self._prepare(job_id)
        if format in prepared["reasons"]:
            raise ValueError(prepared["reasons"][format])
        passport = self._passport(job_id, pipeline, result, request, prepared["skops"][1] if prepared["skops"] else [])
        if prepared["onnx"]:
            passport["onnx"] = prepared["onnx"][1]
        if format == "skops":
            return prepared["skops"][0], filename + ".skops", "application/octet-stream"
        if format == "onnx":
            return prepared["onnx"][0], filename + ".onnx", "application/octet-stream"
        if format == "passport":
            return _json_bytes(passport), filename + "-passport.json", "application/json"
        output = BytesIO()
        with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
            archive.writestr("model.joblib", artifact.read_bytes())
            archive.writestr("passport.json", _json_bytes(passport))
            archive.writestr("README.md", _BUNDLE_README)
            archive.writestr("requirements.txt", "\n".join(f"{package}=={version}" for package, version in passport["versions"].items() if package != "python") + "\n")
            archive.writestr("predict.py", _BUNDLE_PREDICT)
            if prepared["skops"]:
                archive.writestr("model.skops", prepared["skops"][0])
            if prepared["onnx"]:
                archive.writestr("model.onnx", prepared["onnx"][0])
            # Keep fully qualified custom class/function references resolvable.
            # No data files, models from other jobs or service secrets are copied.
            package = Path(__file__).parent
            for source in sorted(package.rglob("*")):
                relative = source.relative_to(package)
                if not source.is_file() or source.is_symlink() or any(part.startswith(".") or part == "__pycache__" for part in relative.parts):
                    continue
                # Nested adapters retain their package path and licence. An
                # allowlist prevents a future cache/data directory from being
                # accidentally included in a reusable prediction package.
                name = source.name.upper()
                licensing = name.startswith(("LICENSE", "LICENCE", "NOTICE")) or name.endswith(("-LICENSE", "-LICENCE"))
                provenance = relative.parent == Path("_vendor") and source.name in {"PROVENANCE.md", "upstream-sha256.json"}
                if source.suffix == ".py" or licensing or provenance:
                    archive.writestr("linear_lab/" + relative.as_posix(), source.read_bytes())
            notices = package.parent / "THIRD_PARTY_NOTICES.md"
            if notices.is_file() and not notices.is_symlink():
                archive.writestr("THIRD_PARTY_NOTICES.md", notices.read_bytes())
        return output.getvalue(), filename + ".zip", "application/zip"


_BUNDLE_README = """# Обученная модель Linear Lab

В `model.joblib` находятся уже обученные преобразования и модель. Подай исходные
признаки, перечисленные в `passport.json`, без таргета. Повторно вызывать `fit` не
нужно. Пересэмплирование из обучения при предсказании не выполняется.

Распакуй ZIP, открой терминал в этой папке и выполни:

```sh
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python predict.py your-data.csv predictions.csv
```

Версия Python записана в паспорте. Для совместимости используй ту же основную
версию Python и закреплённые версии пакетов. Каталоги `linear_lab` содержат код
пользовательских преобразований и моделей; оставь их рядом с `predict.py`.

Joblib и Pickle загружают исполняемые Python-объекты. Загружай только свой пакет
из доверенного источника. Сервис не принимает загрузку сторонних моделей.

Skops (`model.skops`, если поддержан) позволяет сначала посмотреть структуру и
список типов. Типы из `skops_required_trusted_types` в паспорте требуют проверки
по коду. Не передавай результат `get_untrusted_types()` в `trusted` автоматически.
После ручной проверки передай только проверенные имена типов в `skops.io.load`.

ONNX (`model.onnx`, если поддержан) содержит всю поддерживаемую цепочку от
исходных признаков до предсказания, а не только коэффициенты. Схема входов есть
в паспорте и metadata `linear_lab_schema` графа. Каждый вход: имя столбца и tensor
[число строк, 1]. Числовые входы — float64, категориальные — строки без null/NaN.
Для категориальных пропусков используй Python-пайплайн. Экспорт сверяется с
исходным пайплайном через ONNX Runtime; результаты проверки есть в паспорте.

`passport.json` описывает модель и конфигурацию; он сам не исполняет предсказания.
Сырые обучающие строки в пакет не включены.
"""

_BUNDLE_PREDICT = '''"""Предсказания полным обученным пайплайном из доверенного ZIP-пакета."""
from pathlib import Path
import argparse
import joblib
import pandas as pd

parser = argparse.ArgumentParser(description="Linear Lab: предсказания для исходных признаков")
parser.add_argument("input", help="CSV с исходными признаками из passport.json")
parser.add_argument("output", help="CSV с исходными строками и prediction")
args = parser.parse_args()
model = joblib.load(Path(__file__).parent / "model.joblib")
rows = pd.read_csv(args.input)
rows["prediction"] = model.predict(rows)
rows.to_csv(args.output, index=False)
print(f"Сохранено {len(rows)} предсказаний: {args.output}")
'''
