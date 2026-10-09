"""Conservative whole-pipeline ONNX conversion, checked by ONNX Runtime."""

from __future__ import annotations

from copy import deepcopy
import json
import threading

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


_LOCK = threading.RLock()


class _NumericFunction(TransformerMixin, BaseEstimator):
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
        return values


def _register_converters():
    from onnx import TensorProto
    from skl2onnx import update_registered_converter
    from skl2onnx.operator_converters.one_hot_encoder import convert_sklearn_one_hot_encoder
    from skl2onnx.operator_converters.polynomial_features import convert_sklearn_polynomial_features
    from skl2onnx.shape_calculators.one_hot_encoder import calculate_sklearn_one_hot_encoder_output_shapes
    from skl2onnx.shape_calculators.polynomial_features import calculate_sklearn_polynomial_features
    from cml_lab.infrastructure.ml.preparation import BoundedOneHotEncoder, BoundedPolynomial

    def shape(operator):
        operator.outputs[0].type = deepcopy(operator.inputs[0].type)

    def convert(scope, operator, container):
        source = operator.inputs[0].full_name
        output = operator.outputs[0].full_name

        def constant(name, value):
            identifier = scope.get_unique_variable_name(name)
            value = np.asarray(value, dtype=np.float32)
            container.add_initializer(identifier, TensorProto.FLOAT, list(value.shape), value.ravel().tolist())
            return identifier

        mode = operator.raw_operator.mode
        if mode == "finite":
            infinite = scope.get_unique_variable_name("infinite")
            missing = scope.get_unique_variable_name("missing")
            nonfinite = scope.get_unique_variable_name("nonfinite")
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
                container.add_node("Add", [absolute, constant("one", 1)], shifted)
                container.add_node("Log", shifted, transformed)
            else:
                container.add_node("Sqrt", absolute, transformed)
            container.add_node("Mul", [sign, transformed], output)
        elif mode == "clip":
            lower = scope.get_unique_variable_name("lower_clipped")
            container.add_node("Max", [source, constant("lower", operator.raw_operator.bounds[0])], lower)
            container.add_node("Min", [lower, constant("upper", operator.raw_operator.bounds[1])], output)
        else:
            container.add_node("Identity", source, output)

    update_registered_converter(_NumericFunction, "CMLNumericFunction", shape, convert)
    update_registered_converter(BoundedOneHotEncoder, "CMLBoundedOneHot",
                                calculate_sklearn_one_hot_encoder_output_shapes, convert_sklearn_one_hot_encoder)
    update_registered_converter(BoundedPolynomial, "CMLBoundedPolynomial",
                                calculate_sklearn_polynomial_features, convert_sklearn_polynomial_features)


def _adapt(stage, input_kinds):
    from cml_lab.infrastructure.ml.preparation import (
        BoundedReducer, FiniteNumeric, StringCategories, SignedPower,
    )
    from linear_lab.preprocessing import QuantileClipper

    if type(stage) is ColumnTransformer:
        converted = deepcopy(stage)
        branches = []
        for name, transformer, columns in stage.transformers_:
            if isinstance(transformer, str):
                branches.append((name, transformer, columns))
                continue
            if not isinstance(columns, (list, tuple, np.ndarray)) or any(not isinstance(column, str) for column in columns):
                raise ValueError("ONNX требует сохраненные имена исходных столбцов для каждой ветки.")
            categorical = isinstance(transformer, Pipeline) and any(isinstance(item, StringCategories) for _, item in transformer.steps)
            for column in columns:
                input_kinds[column] = "string" if categorical else "float"
            branches.append((name, _adapt(transformer, input_kinds), columns))
        converted.transformers_ = branches
        return converted
    if isinstance(stage, Pipeline):
        steps = []
        categorical = any(isinstance(item, StringCategories) for _, item in stage.steps)
        for name, transformer in stage.steps:
            if isinstance(transformer, StringCategories):
                continue
            if categorical and isinstance(transformer, SimpleImputer):
                # A string tensor cannot represent Python null; this restricted
                # input contract is exported explicitly and checked at inference.
                continue
            if isinstance(transformer, SimpleImputer) and transformer.add_indicator:
                raise ValueError("ONNX не поддерживает индикаторы пропусков этого рецепта; выберите Python-формат.")
            steps.append((name, _adapt(transformer, input_kinds)))
        if not steps:
            raise ValueError("ONNX: ветка подготовки не содержит преобразования.")
        return Pipeline(steps)
    if type(stage) is FiniteNumeric:
        return _NumericFunction("finite")
    if isinstance(stage, SignedPower):
        return _NumericFunction(stage.method)
    if isinstance(stage, QuantileClipper):
        return _NumericFunction("clip", np.asarray(stage.bounds_))
    if isinstance(stage, BoundedReducer):
        return deepcopy(stage.model_)
    if isinstance(stage, str):
        return stage
    module = type(stage).__module__
    if module.startswith("cml_lab") and type(stage).__name__ not in {"BoundedOneHotEncoder", "BoundedPolynomial"}:
        raise ValueError(f"ONNX: этап {type(stage).__name__} не имеет проверенного конвертера. Выберите Joblib или ZIP.")
    if module.startswith("linear_lab"):
        raise ValueError("ONNX: совместимый старый препроцессор не имеет нового проверенного конвертера. Сохраните Joblib/ZIP или пересоздайте рецепт этапами.")
    return deepcopy(stage)


def convert_artifact(artifact, sample: pd.DataFrame) -> tuple[bytes, dict]:
    """Return a model only after raw-input and decoded-output parity succeeds."""
    from onnx import TensorProto, helper
    import onnxruntime as ort
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType, StringTensorType

    if artifact.task in {"forecasting", "panel", "reduction", "anomaly"}:
        raise ValueError("ONNX этого типа артефакта пока не поддержан: необходима его задача, история или особый ответ. Используйте Joblib/ZIP.")
    if not hasattr(artifact, "features") or not hasattr(artifact.estimator, "predict"):
        raise ValueError("ONNX требует модель, которая прогнозирует новые строки.")
    if not len(sample):
        raise ValueError("Для проверки ONNX необходимы исходные строки сохраненного датасета.")
    if not type(artifact.estimator).__module__.startswith("sklearn."):
        raise ValueError("ONNX: для этой внешней/структурной модели не установлен проверенный конвертер полного конвейера.")
    input_kinds = {feature: "float" for feature in artifact.features}
    with _LOCK:
        _register_converters()
        preprocessing = _adapt(artifact.preprocessing, input_kinds)
        pipeline = Pipeline([("preprocessing", preprocessing), ("estimator", deepcopy(artifact.estimator))])
        inputs = [(name, StringTensorType([None, 1]) if kind == "string" else FloatTensorType([None, 1]))
                  for name, kind in input_kinds.items()]
        options = {id(pipeline.steps[-1][1]): {"zipmap": False}} if artifact.task == "classification" and hasattr(artifact.estimator, "predict_proba") else None
        model = convert_sklearn(pipeline, initial_types=inputs, options=options,
                                target_opset={"": 17, "ai.onnx.ml": 3})
    if artifact.task == "classification" and artifact.label_encoder is not None:
        label = model.graph.output[0]
        if label.type.tensor_type.elem_type != TensorProto.INT64:
            raise ValueError("ONNX: тип классов не соответствует сохраненному целочисленному кодированию.")
        classes = [str(value) for value in artifact.label_encoder.classes_]
        for imported in model.opset_import:
            if imported.domain == "ai.onnx.ml":
                imported.version = max(imported.version, 3)
        model.graph.node.append(helper.make_node("LabelEncoder", [label.name], ["cml_decoded_label"],
                                                domain="ai.onnx.ml", keys_int64s=list(range(len(classes))),
                                                values_strings=classes))
        label.name = "cml_decoded_label"
        label.type.tensor_type.elem_type = TensorProto.STRING
    schema = {"inputs": [{"name": name, "dtype": kind, "shape": [None, 1],
                           "nullable": kind == "float"} for name, kind in input_kinds.items()],
              "output": "decoded_class" if artifact.task == "classification" else "prediction",
              "categorical_contract": "Строковые категории без null; неизвестная категория обрабатывается сохраненным кодировщиком.",
              "numeric_contract": "float32; null передается как NaN; бесконечности заменяются на пропуск до заполнения."}
    helper.set_model_props(model, {"cml_input_schema": json.dumps(schema, ensure_ascii=False),
                                   "cml_task": artifact.task})
    data = model.SerializeToString()
    ort.disable_telemetry_events()
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1
    session = ort.InferenceSession(data, sess_options=session_options, providers=["CPUExecutionProvider"])
    feed = {}
    sample = sample.loc[:, artifact.features].copy()
    for name, kind in input_kinds.items():
        if kind == "string":
            if sample[name].isna().any():
                raise ValueError("ONNX не представляет null в строковой категории. Используйте Joblib/ZIP для строк с пропусками.")
            feed[name] = sample[name].astype(str).to_numpy(dtype=object).reshape(-1, 1)
        else:
            feed[name] = pd.to_numeric(sample[name], errors="raise").to_numpy(dtype=np.float32).reshape(-1, 1)
    expected = np.asarray(artifact.predict(sample)).reshape(-1)
    outputs = session.run(None, {item.name: feed[item.name] for item in session.get_inputs()})
    actual = np.asarray(outputs[0]).reshape(-1)
    if artifact.task == "classification":
        if not np.array_equal(actual.astype(str), expected.astype(str)):
            raise ValueError("ONNX не сохранил классы на проверочных исходных строках. Используйте Python-формат.")
        if hasattr(artifact.estimator, "predict_proba"):
            expected_proba = artifact.predict_proba(sample)
            if len(outputs) < 2 or not np.allclose(outputs[1], expected_proba, atol=1e-5, rtol=1e-4):
                raise ValueError("ONNX не сохранил вероятности классов с достаточной точностью. Используйте Python-формат.")
    elif not np.allclose(actual, expected, atol=1e-4, rtol=1e-4, equal_nan=False):
        raise ValueError("ONNX не сохранил прогнозы с достаточной точностью. Используйте Python-формат.")
    schema["parity_checked_rows"] = len(sample)
    return data, schema
