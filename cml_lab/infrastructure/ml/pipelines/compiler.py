"""Parse declarative Python without eval/exec and construct unfitted sklearn.

Only names in the explicit registry can be imported or called. A definition is
imports, literal assignments and constructors, ending in ``pipeline = ...``.
"""

from __future__ import annotations

import ast
import inspect
import json
import math
from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FeatureUnion, Pipeline

from .registry import MODULES, REFERENCE_MODULES, constructors, qualified_name, references


FORMAT = "cml.pipeline"
VERSION = 1
MAX_SOURCE_BYTES = 100_000
MAX_AST_NODES = 12_000
MAX_DEPTH = 40
MAX_CONSTRUCTORS = 200
MAX_LITERAL_ITEMS = 10_000
MAX_ARRAY_BYTES = 10_000_000


def _check_literal_array(data, dtype, node):
    """Bound dtype padding and scalar count before numpy allocates anything."""
    count, max_text_width, has_text = 0, 0, False
    pending = [data]
    while pending:
        value = pending.pop()
        if isinstance(value, (list, tuple)):
            pending.extend(value)
            if len(pending) + count > MAX_LITERAL_ITEMS:
                raise PipelineDefinitionError("numpy.array содержит больше 10 000 элементов.", node)
            continue
        if isinstance(value, np.ndarray):
            # A nested np.array has already passed this guard. Its values still
            # contribute to the outer allocation's inferred width and count.
            if value.size + count > MAX_LITERAL_ITEMS:
                raise PipelineDefinitionError("numpy.array содержит больше 10 000 элементов.", node)
            pending.extend(value.ravel().tolist())
            continue
        if value is None or not isinstance(value, (str, bool, int, float)):
            raise PipelineDefinitionError("numpy.array принимает только числовые, bool и строковые литералы.", node)
        count += 1
        if count > MAX_LITERAL_ITEMS:
            raise PipelineDefinitionError("numpy.array содержит больше 10 000 элементов.", node)
        has_text |= isinstance(value, str)
        max_text_width = max(max_text_width, len(value) if isinstance(value, str) else len(str(value)))
    if dtype is not None:
        try:
            resolved = np.dtype(dtype)
        except (TypeError, ValueError) as error:
            raise PipelineDefinitionError(f"numpy.array.dtype: {error}", node) from None
        if resolved.kind not in "biufUS":
            raise PipelineDefinitionError("numpy.array.dtype допускает только bool, целые, float и строки; object/void запрещены.", node)
        # U0/S0 ask numpy to infer width. Treat all strings conservatively as
        # Unicode, including conversion from numbers into a string dtype.
        itemsize = resolved.itemsize or (4 * max_text_width if resolved.kind in "US" else 8)
    else:
        itemsize = 4 * max_text_width if has_text else 8
    if itemsize > MAX_ARRAY_BYTES or count * itemsize > MAX_ARRAY_BYTES:
        raise PipelineDefinitionError("numpy.array превысит 10 млн байт. Уменьшите dtype строки или число литералов.", node)


class PipelineDefinitionError(ValueError):
    def __init__(self, message, node=None, *, line=None, column=None):
        self.message = message
        self.line = line if line is not None else getattr(node, "lineno", None)
        self.column = column if column is not None else (getattr(node, "col_offset", -1) + 1 if node is not None else None)
        position = f"Строка {self.line}, столбец {self.column}: " if self.line else ""
        super().__init__(position + message)

    def to_dict(self):
        return {"message": self.message, "line": self.line, "column": self.column}


@dataclass(frozen=True)
class _Module:
    name: str


class _Parser:
    def __init__(self, source):
        self.source = source
        self.names = dict(REFERENCE_MODULES["builtins"])
        self.count = 0

    def parse(self):
        if not isinstance(self.source, str) or not self.source.strip():
            raise PipelineDefinitionError("Определение Pipeline должно быть непустым текстом Python.")
        if len(self.source.encode("utf-8")) > MAX_SOURCE_BYTES:
            raise PipelineDefinitionError("Определение Pipeline превышает 100 000 байт.")
        try:
            module = ast.parse(self.source, mode="exec")
        except (RecursionError, MemoryError) as error:
            raise PipelineDefinitionError("Определение Python слишком глубокое или большое.") from error
        except SyntaxError as error:
            raise PipelineDefinitionError(error.msg, line=error.lineno, column=error.offset) from None
        if sum(1 for _ in ast.walk(module)) > MAX_AST_NODES:
            raise PipelineDefinitionError("Определение Pipeline содержит слишком много выражений.")
        for statement in module.body:
            if isinstance(statement, ast.ImportFrom):
                self._import_from(statement)
            elif isinstance(statement, ast.Import):
                self._import(statement)
            elif isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
                name = statement.targets[0].id
                if name.startswith("_") or name in self.names:
                    raise PipelineDefinitionError("Имена должны быть уникальными и не начинаться с '_'.", statement)
                self.names[name] = self.value(statement.value)
            elif isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant) and isinstance(statement.value.value, str):
                # A module docstring does not execute anything.
                continue
            else:
                raise PipelineDefinitionError("Допустимы только разрешенные imports и присваивания конструкторам или литералам. Циклы, функции, обучение и произвольный Python не исполняются.", statement)
        if "pipeline" not in self.names:
            raise PipelineDefinitionError("Последний объект должен быть присвоен имени pipeline: pipeline = Pipeline(...).")
        pipeline = self.names["pipeline"]
        if not isinstance(pipeline, BaseEstimator) or not hasattr(pipeline, "transform"):
            raise PipelineDefinitionError("pipeline должен быть преобразователем sklearn с fit и transform. Модель задается отдельно от preprocessing.")
        self._validate_estimator(pipeline)
        return pipeline

    def _bind(self, name, value, node):
        if name in self.names or name.startswith("_"):
            raise PipelineDefinitionError(f"Имя {name} уже занято или недопустимо.", node)
        self.names[name] = value

    def _import_from(self, node):
        module = node.module
        if node.level or module not in MODULES and module not in REFERENCE_MODULES:
            # The experimental import only signals intent; registry already enables it.
            if not node.level and module == "sklearn.experimental" and all(item.name == "enable_iterative_imputer" for item in node.names):
                return
            raise PipelineDefinitionError(f"Import {module} отсутствует в каталоге разрешенных преобразований.", node)
        allowed = references()
        for item in node.names:
            key = f"{module}.{item.name}"
            if key not in allowed:
                raise PipelineDefinitionError(f"Имя {key} не разрешено. Используйте каталог конструктора.", node)
            self._bind(item.asname or item.name, allowed[key], node)

    def _import(self, node):
        for item in node.names:
            if item.name not in MODULES and item.name not in REFERENCE_MODULES and item.name != "sklearn":
                raise PipelineDefinitionError(f"Import {item.name} не разрешен.", node)
            name = item.asname or item.name.split(".")[0]
            module_name = item.name if item.asname else item.name.split(".")[0]
            if name in self.names and self.names[name] == _Module(module_name):
                continue
            self._bind(name, _Module(module_name), node)

    def value(self, node, depth=0):
        if depth > MAX_DEPTH:
            raise PipelineDefinitionError("Вложенность Pipeline превышает 40 уровней.", node)
        if isinstance(node, ast.Constant):
            value = node.value
            if value is None or isinstance(value, (str, bool, int, float)):
                if isinstance(value, str) and len(value) > MAX_SOURCE_BYTES:
                    raise PipelineDefinitionError("Строковый параметр слишком длинный.", node)
                if isinstance(value, int) and value.bit_length() > 64:
                    raise PipelineDefinitionError("Числовой литерал слишком большой.", node)
                if isinstance(value, float) and not math.isfinite(value):
                    raise PipelineDefinitionError("Используйте явный numpy.inf или numpy.nan вместо бесконечного литерала.", node)
                return value
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            if isinstance(node, ast.Set):
                raise PipelineDefinitionError("Используйте список вместо множества: порядок фич должен быть явным.", node)
            if len(node.elts) > MAX_LITERAL_ITEMS:
                raise PipelineDefinitionError("Список параметра слишком большой.", node)
            values = [self.value(item, depth + 1) for item in node.elts]
            return tuple(values) if isinstance(node, ast.Tuple) else values
        if isinstance(node, ast.Dict):
            if len(node.keys) > MAX_LITERAL_ITEMS or any(key is None for key in node.keys):
                raise PipelineDefinitionError("Словарь слишком большой или содержит распаковку **.", node)
            result = {}
            for key_node, value_node in zip(node.keys, node.values):
                key = self.value(key_node, depth + 1)
                if not isinstance(key, (str, int, float, bool)):
                    raise PipelineDefinitionError("Ключ словаря должен быть строкой или числом.", key_node)
                if key in result:
                    raise PipelineDefinitionError(f"Повторяется ключ {key!r}.", key_node)
                result[key] = self.value(value_node, depth + 1)
            return result
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            value = self.value(node.operand, depth + 1)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise PipelineDefinitionError("Знак + или - допустим только для числового литерала.", node)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.Name):
            if node.id in self.names:
                return self.names[node.id]
            if node.id == "slice":
                return slice
            raise PipelineDefinitionError(f"Неизвестное имя {node.id}. Добавьте разрешенный import или присваивание выше.", node)
        if isinstance(node, ast.Attribute):
            owner = self.value(node.value, depth + 1)
            if not isinstance(owner, _Module) or node.attr.startswith("_"):
                raise PipelineDefinitionError("Доступ к атрибутам объектов не разрешен. Разрешены только имена из sklearn и numpy каталога.", node)
            name = f"{owner.name}.{node.attr}"
            if name in references():
                return references()[name]
            if name in MODULES or any(module.startswith(name + ".") for module in MODULES):
                return _Module(name)
            raise PipelineDefinitionError(f"Имя {name} не разрешено.", node)
        if isinstance(node, ast.Call):
            # Configuration methods are interpreted directly on allowlisted
            # constructors; fit/transform and I/O never occur in declarations.
            if isinstance(node.func, ast.Attribute) and node.func.attr == "set_params":
                owner = self.value(node.func.value, depth + 1)
                if not isinstance(owner, BaseEstimator) or node.args or any(keyword.arg is None for keyword in node.keywords):
                    raise PipelineDefinitionError("set_params принимает только явно именованные параметры sklearn.", node)
                params = {}
                for keyword in node.keywords:
                    if keyword.arg in params:
                        raise PipelineDefinitionError(f"Параметр {keyword.arg} указан дважды.", node)
                    params[keyword.arg] = self.value(keyword.value, depth + 1)
                try:
                    owner.set_params(**params)
                    validate_pipeline_object(owner)
                    return owner
                except (ValueError, TypeError, AttributeError) as error:
                    if isinstance(error, PipelineDefinitionError):
                        raise PipelineDefinitionError(error.message, node) from None
                    raise PipelineDefinitionError(str(error), node) from None
            if isinstance(node.func, ast.Attribute) and node.func.attr == "set_output":
                owner = self.value(node.func.value, depth + 1)
                if not isinstance(owner, BaseEstimator) or node.args or len(node.keywords) != 1 or node.keywords[0].arg != "transform":
                    raise PipelineDefinitionError("Допустим только transformer.set_output(transform='default'|'pandas').", node)
                output = self.value(node.keywords[0].value, depth + 1)
                if output not in ("default", "pandas"):
                    raise PipelineDefinitionError("set_output поддерживает default или pandas.", node)
                try:
                    return owner.set_output(transform=output)
                except (ValueError, TypeError, AttributeError) as error:
                    raise PipelineDefinitionError(str(error), node) from None
            target = self.value(node.func, depth + 1)
            key = qualified_name(target)
            if target is not slice and (key not in constructors() and key != "numpy.array"):
                raise PipelineDefinitionError("Разрешены только конструкторы каталога, slice и numpy.array. Функцию numpy укажите как func, не вызывайте при определении.", node)
            if any(keyword.arg is None for keyword in node.keywords) or any(isinstance(argument, ast.Starred) for argument in node.args):
                raise PipelineDefinitionError("Распаковка *args и **kwargs не поддерживается; укажите параметры явно.", node)
            args = [self.value(argument, depth + 1) for argument in node.args]
            kwargs = {}
            for keyword in node.keywords:
                if keyword.arg in kwargs:
                    raise PipelineDefinitionError(f"Параметр {keyword.arg} указан дважды.", keyword.value)
                kwargs[keyword.arg] = self.value(keyword.value, depth + 1)
            self.count += 1
            if self.count > MAX_CONSTRUCTORS:
                raise PipelineDefinitionError("Определение содержит больше 200 конструкторов.", node)
            if key == "numpy.array":
                if not args or len(args) > 1 or any(name not in ("dtype",) for name in kwargs):
                    raise PipelineDefinitionError("numpy.array принимает список литералов и необязательный dtype.", node)
                _check_literal_array(args[0], kwargs.get("dtype"), node)
                try:
                    value = np.array(args[0], **kwargs)
                except (ValueError, TypeError) as error:
                    raise PipelineDefinitionError(f"numpy.array: {error}", node) from None
                if value.size > MAX_LITERAL_ITEMS or value.dtype.kind not in "biufUS":
                    raise PipelineDefinitionError("numpy.array поддерживает только ограниченные числовые или строковые литералы.", node)
                return value
            if target is slice:
                if kwargs or not 1 <= len(args) <= 3 or any(item is not None and not isinstance(item, (int, str)) for item in args):
                    raise PipelineDefinitionError("slice принимает до трех имен или индексов столбцов.", node)
                if len(args) == 3 and (args[2] == 0 or args[2] is not None and not isinstance(args[2], int)):
                    raise PipelineDefinitionError("Шаг slice должен быть ненулевым целым числом.", node)
                return slice(*args)
            try:
                inspect.signature(target).bind(*args, **kwargs)
                value = target(*args, **kwargs)
                if isinstance(value, BaseEstimator):
                    self._validate_estimator(value, node)
                return value
            except PipelineDefinitionError:
                raise
            except (ValueError, TypeError, AttributeError) as error:
                raise PipelineDefinitionError(f"{key}: {error}", node) from None
        raise PipelineDefinitionError(f"Выражение {type(node).__name__} не входит в декларативный Python. Используйте литералы, разрешенные ссылки и конструкторы.", node)

    def _validate_estimator(self, estimator, node=None):
        params = estimator.get_params(deep=False)
        if params.get("memory") is not None:
            raise PipelineDefinitionError("Pipeline.memory должен быть None: файловый кеш не входит в декларативный формат.", node)
        if params.get("transform_input") is not None:
            raise PipelineDefinitionError("Pipeline.transform_input должен быть None: дополнительные метаданные не передаются этим сервисом.", node)
        if params.get("prefit") is True:
            raise PipelineDefinitionError("prefit=True не разрешен: селектор обучается внутри каждой train части.", node)
        # Filename/URL input and custom analyzers can perform I/O during fit.
        if params.get("input") in ("filename", "file"):
            raise PipelineDefinitionError("Текстовый input должен быть 'content'; чтение файлов внутри преобразователя не разрешено.", node)
        for key, maximum in (("n_features", 20_000), ("max_categories", 20_000), ("n_knots", 2000), ("n_estimators", 10_000), ("max_iter", 100_000), ("n_components", 2000)):
            value = params.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value > maximum:
                raise PipelineDefinitionError(f"{key}={value} превышает ограничение лаборатории {maximum}.", node)
        if type(estimator).__name__ == "PolynomialFeatures":
            degrees = params["degree"] if isinstance(params["degree"], tuple) else (params["degree"],)
            if any(isinstance(degree, int) and degree > 20 for degree in degrees):
                raise PipelineDefinitionError("PolynomialFeatures.degree ограничен 20; число выходных фич отдельно проверяется до выделения матрицы.", node)
        if isinstance(estimator, Pipeline):
            estimator._validate_steps()
            if not estimator.steps or not hasattr(estimator, "transform"):
                raise PipelineDefinitionError("Все шаги preprocessing Pipeline должны поддерживать transform; модель задается отдельно.", node)
        if isinstance(estimator, ColumnTransformer):
            estimator._validate_transformers()
        if isinstance(estimator, FeatureUnion):
            estimator._validate_transformers()
        if hasattr(estimator, "_validate_params") and hasattr(estimator, "_parameter_constraints"):
            estimator._validate_params()


def _literal(value):
    if isinstance(value, BaseEstimator):
        return transformer_tree(value)
    if isinstance(value, slice):
        return {"$slice": [value.start, value.stop, value.step]}
    if isinstance(value, np.ndarray):
        return {"$array": value.tolist(), "dtype": str(value.dtype)}
    if isinstance(value, tuple):
        return {"$tuple": [_literal(item) for item in value]}
    if isinstance(value, list):
        return [_literal(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _literal(item) for key, item in value.items()}
    if isinstance(value, float) and math.isnan(value):
        return {"$ref": "numpy.nan"}
    if isinstance(value, float) and math.isinf(value):
        return {"$ref": "numpy.inf", "negative": value < 0}
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    reference = qualified_name(value)
    if reference:
        return {"$ref": reference}
    if type(value).__name__ == "make_column_selector":
        return {"$call": "sklearn.compose.make_column_selector", "params": {key: _literal(getattr(value, key)) for key in ("pattern", "dtype_include", "dtype_exclude")}}
    raise PipelineDefinitionError(f"Параметр типа {type(value).__name__} не представлен в формате Pipeline.")


def transformer_tree(transformer):
    qualified = qualified_name(type(transformer))
    if qualified is None:
        raise PipelineDefinitionError(f"Класс {type(transformer).__name__} отсутствует в каталоге Pipeline.")
    module, name = qualified.rsplit(".", 1)
    params = transformer.get_params(deep=False)
    result = {"type": name, "module": module, "params": {}}
    if isinstance(transformer, Pipeline):
        result["steps"] = [{"name": step_name, "transformer": _literal(step)} for step_name, step in params.pop("steps")]
    elif isinstance(transformer, ColumnTransformer):
        result["transformers"] = [{"name": step_name, "transformer": _literal(step), "columns": _literal(columns)} for step_name, step, columns in params.pop("transformers")]
    elif isinstance(transformer, FeatureUnion):
        result["transformer_list"] = [{"name": step_name, "transformer": _literal(step)} for step_name, step in params.pop("transformer_list")]
    result["params"] = {key: _literal(value) for key, value in params.items()}
    output = getattr(transformer, "_sklearn_output_config", {}).get("transform")
    if output:
        result["output"] = output
    return result


class _Emitter:
    def __init__(self):
        self.imports = set()
        self.nodes = 0

    def reference(self, name):
        if name not in references():
            raise PipelineDefinitionError(f"Ссылка {name} отсутствует в каталоге Pipeline.")
        if name.startswith("numpy."):
            self.imports.add("import numpy as np")
            return "np." + name.rsplit(".", 1)[1]
        if name.startswith("builtins."):
            return name.rsplit(".", 1)[1]
        module, symbol = name.rsplit(".", 1)
        self.imports.add(f"from {module} import {symbol}")
        if symbol == "IterativeImputer":
            self.imports.add("from sklearn.experimental import enable_iterative_imputer")
        return symbol

    def value(self, value, depth=0):
        self.nodes += 1
        if depth > MAX_DEPTH or self.nodes > MAX_AST_NODES:
            raise PipelineDefinitionError("Дерево Pipeline слишком большое или глубокое.")
        if value is None or isinstance(value, (str, bool, int, float)):
            return repr(value)
        if isinstance(value, list):
            return "[" + ", ".join(self.value(item, depth + 1) for item in value) + "]"
        if not isinstance(value, dict):
            raise PipelineDefinitionError("Параметры дерева должны быть JSON литералами.")
        if "$ref" in value:
            result = self.reference(value["$ref"])
            return "-" + result if value.get("negative") else result
        if "$slice" in value:
            args = value["$slice"]
            if not isinstance(args, list) or len(args) != 3:
                raise PipelineDefinitionError("$slice должен содержать [start, stop, step].")
            return "slice(" + ", ".join(self.value(item, depth + 1) for item in args) + ")"
        if "$tuple" in value:
            if not isinstance(value["$tuple"], list):
                raise PipelineDefinitionError("$tuple должен содержать список литералов.")
            items = [self.value(item, depth + 1) for item in value["$tuple"]]
            return "(" + ", ".join(items) + ("," if len(items) == 1 else "") + ")"
        if "$array" in value:
            self.imports.add("import numpy as np")
            return f"np.array({self.value(value['$array'], depth + 1)}, dtype={self.value(value.get('dtype', 'float64'), depth + 1)})"
        if "$call" in value:
            if value["$call"] != "sklearn.compose.make_column_selector":
                raise PipelineDefinitionError("$call поддерживает только make_column_selector.")
            target = self.reference(value["$call"])
            return self._call(target, value.get("params", {}), depth)
        if "type" in value:
            name = value["type"]
            module = value.get("module")
            qualified = f"{module}.{name}" if module else next((key for key in constructors() if key.rsplit(".", 1)[1] == name), None)
            if not qualified or qualified not in constructors():
                raise PipelineDefinitionError(f"Конструктор {name} отсутствует в каталоге Pipeline.")
            params = dict(value.get("params", {}))
            for container, width in (("steps", 2), ("transformers", 3), ("transformer_list", 2)):
                if container in value:
                    entries = value[container]
                    if not isinstance(entries, list):
                        raise PipelineDefinitionError(f"{container} должен быть списком веток.")
                    expressions = []
                    for entry in entries:
                        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str) or "transformer" not in entry:
                            raise PipelineDefinitionError("Ветка должна содержать name и transformer.")
                        fields = [self.value(entry["name"], depth + 1), self.value(entry["transformer"], depth + 1)]
                        if width == 3:
                            fields.append(self.value(entry.get("columns", []), depth + 1))
                        expressions.append("(" + ", ".join(fields) + ")")
                    params.pop(container, None)
                    params[container] = _Code("[" + ", ".join(expressions) + "]")
            result = self._call(self.reference(qualified), params, depth)
            if "output" in value:
                result += f".set_output(transform={self.value(value['output'], depth + 1)})"
            return result
        return "{" + ", ".join(f"{key!r}: {self.value(item, depth + 1)}" for key, item in value.items()) + "}"

    def _call(self, name, params, depth):
        if not isinstance(params, dict) or any(not isinstance(key, str) or not key.isidentifier() for key in params):
            raise PipelineDefinitionError("Параметры конструктора должны быть словарем Python имен.")
        args = [f"{key}={value.text if isinstance(value, _Code) else self.value(value, depth + 1)}" for key, value in params.items()]
        return name + "(" + ", ".join(args) + ")"


@dataclass(frozen=True)
class _Code:
    text: str


def emit_pipeline_source(tree):
    """Serialize a tree as an ordinary standalone .py declaration."""
    emitter = _Emitter()
    expression = emitter.value(tree)
    if len(expression.encode("utf-8")) > MAX_SOURCE_BYTES:
        raise PipelineDefinitionError("Дерево Pipeline создает Python определение больше 100 000 байт.")
    try:
        parsed = ast.parse(expression, mode="eval").body
    except (SyntaxError, RecursionError) as error:
        raise PipelineDefinitionError("Дерево Pipeline не удалось преобразовать в Python.") from error
    expression = _pretty_expression(parsed)
    return "\n".join(sorted(emitter.imports, key=lambda line: ("experimental" not in line, line))) + "\n\npipeline = " + expression + "\n"


def _pretty_expression(node, indent=0):
    """Readable nested constructors without changing Python's AST semantics."""
    compact = ast.unparse(node)
    if len(compact) + indent <= 88:
        return compact
    padding, child_padding = " " * indent, " " * (indent + 4)
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Call):
            name = _pretty_expression(node.func.value, indent) + "." + node.func.attr
        else:
            name = ast.unparse(node.func)
        args = [_pretty_expression(argument, indent + 4) for argument in node.args]
        args.extend(keyword.arg + "=" + _pretty_expression(keyword.value, indent + 4) for keyword in node.keywords)
        return name + "(\n" + "\n".join(child_padding + argument + "," for argument in args) + "\n" + padding + ")"
    if isinstance(node, (ast.List, ast.Tuple)):
        opening, closing = ("[", "]") if isinstance(node, ast.List) else ("(", ")")
        return opening + "\n" + "\n".join(child_padding + _pretty_expression(item, indent + 4) + "," for item in node.elts) + "\n" + padding + closing
    if isinstance(node, ast.Dict):
        entries = [ast.unparse(key) + ": " + _pretty_expression(value, indent + 4) for key, value in zip(node.keys, node.values)]
        return "{\n" + "\n".join(child_padding + value + "," for value in entries) + "\n" + padding + "}"
    return compact


def _source(spec):
    if isinstance(spec, str):
        return spec
    if not isinstance(spec, dict):
        raise PipelineDefinitionError("Pipeline должен быть текстом Python или объектом cml.pipeline.")
    if spec.get("format", FORMAT) != FORMAT or spec.get("version", VERSION) != VERSION:
        raise PipelineDefinitionError("Поддерживается формат cml.pipeline версии 1.")
    if "source" in spec:
        return spec["source"]
    if "tree" in spec:
        return emit_pipeline_source(spec["tree"])
    raise PipelineDefinitionError("Pipeline должен содержать source или tree.")


def compile_pipeline(spec):
    """Return a new, unfitted sklearn transformer without reading any dataset."""
    return _Parser(_source(spec)).parse()


def validate_pipeline_object(transformer):
    """Recheck literal overrides before fit, including nested estimators."""
    active, complete = set(), set()
    nodes = 0

    def check(value, depth=0):
        nonlocal nodes
        nodes += 1
        if depth > MAX_DEPTH or nodes > MAX_AST_NODES:
            raise PipelineDefinitionError("Граф Pipeline слишком глубокий или большой.")
        if not isinstance(value, (BaseEstimator, list, tuple, dict)):
            return
        identifier = id(value)
        if identifier in active:
            raise PipelineDefinitionError("Pipeline содержит цикл ссылок; шаг не может содержать самого себя.")
        if identifier in complete:
            return
        active.add(identifier)
        children = value.get_params(deep=False).values() if isinstance(value, BaseEstimator) else value.values() if isinstance(value, dict) else value
        for child in children:
            check(child, depth + 1)
        active.remove(identifier)
        complete.add(identifier)

    check(transformer)
    validator = _Parser("")
    validator._validate_estimator(transformer)
    seen = {id(transformer)}
    for value in transformer.get_params(deep=True).values():
        if isinstance(value, BaseEstimator) and id(value) not in seen:
            validator._validate_estimator(value)
            seen.add(id(value))
    return transformer


def normalize_pipeline(spec):
    source = _source(spec)
    _Parser(source).parse()
    return {"format": FORMAT, "version": VERSION, "source": source}


def pipeline_warnings(transformer):
    warnings = []
    params = transformer.get_params(deep=True)
    for key, value in params.items():
        if isinstance(value, BaseEstimator):
            name = type(value).__name__
            if name in ("TargetEncoder", "RFECV", "SequentialFeatureSelector"):
                warnings.append(f"{key}: {name} имеет собственную CV, которая не получает time/group split лаборатории; для ordered/grouped задач он запрещен.")
            if name in ("KernelPCA", "Isomap", "LocallyLinearEmbedding"):
                warnings.append(f"{key}: {name} может строить попарную матрицу train; ограничьте число строк.")
            if name in ("PLSRegression", "PLSCanonical", "CCA"):
                warnings.append(f"{key}: при обучении лаборатория передает дальше только X scores; sklearn.fit_transform вне лаборатории возвращает также y scores.")
    if type(transformer).__name__ in ("TargetEncoder", "RFECV", "SequentialFeatureSelector"):
        warnings.append("Корневой преобразователь имеет собственную CV; для ordered/grouped задач он запрещен.")
    if type(transformer).__name__ in ("PLSRegression", "PLSCanonical", "CCA"):
        warnings.append("При preprocessing лаборатория передает только X scores; sklearn.fit_transform вне лаборатории возвращает также y scores.")
    return warnings
