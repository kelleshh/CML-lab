"""Public editor metadata, directed feature graphs and installed source view."""

from __future__ import annotations

import inspect
from copy import deepcopy
from pathlib import Path

import sklearn

from .compiler import FORMAT, VERSION, PipelineDefinitionError, compile_pipeline, normalize_pipeline, pipeline_warnings, transformer_tree
from .registry import catalogue_classes, constructors


TEMPLATES = [
    {
        "id": "mixed-columns", "name": "ColumnTransformer: numeric + categorical",
        "description": "Раздельные ветки чисел и категорий; пропуски и неизвестные категории обрабатываются по train.",
        "source": '''from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

numeric = Pipeline([
    ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
    ("scaler", StandardScaler()),
])
categorical = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])
pipeline = ColumnTransformer([
    ("numeric", numeric, make_column_selector(dtype_include="number")),
    ("categorical", categorical, make_column_selector(dtype_exclude="number")),
], remainder="drop", verbose_feature_names_out=True)
''',
    },
    {
        "id": "numeric-union", "name": "Pipeline + FeatureUnion + PCA",
        "description": "Одна входная числовая матрица идет в две ветки: исходный масштаб и взаимодействия; затем их компоненты объединяются.",
        "source": '''from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
    ("features", FeatureUnion([
        ("scaled", StandardScaler()),
        ("interactions", Pipeline([
            ("polynomial", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
            ("scale", StandardScaler()),
            ("reduce", PCA(n_components=0.95)),
        ])),
    ])),
])
''',
    },
    {
        "id": "text-and-numeric", "name": "ColumnTransformer: text + numeric",
        "description": "Скалярное имя text передает одномерные документы TfidfVectorizer. Список numeric сохраняет двумерную числовую ветку; замените имена своими фичами.",
        "source": '''from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

pipeline = ColumnTransformer([
    ("text", TfidfVectorizer(max_features=1000, ngram_range=(1, 2)), "text"),
    ("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),
    ]), ["numeric"]),
], remainder="drop")
''',
    },
    {
        "id": "function-selection", "name": "FunctionTransformer + SelectKBest",
        "description": "log1p для неотрицательных чисел и отбор двух фич по f_regression; y используется только в train fit.",
        "source": '''import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("log", FunctionTransformer(func=np.log1p, inverse_func=np.expm1, feature_names_out="one-to-one")),
    ("scale", StandardScaler()),
    ("select", SelectKBest(score_func=f_regression, k=2)),
])
''',
    },
]


def pipeline_graph(tree):
    """Show actual routing: serial edges, selected branches and merge edges."""
    nodes, edges = [], []

    def node(label, *, kind="transformer", path="", data=None, parent=None):
        identifier = f"n{len(nodes)}"
        value = {"id": identifier, "label": label, "kind": kind, "path": path}
        if data:
            value.update({"class": data.get("type"), "module": data.get("module"), "params": deepcopy(data.get("params", {}))})
        if parent:
            value["parent_id"] = parent
        nodes.append(value)
        return identifier

    def edge(source, target, label="", columns=None):
        value = {"source": source, "target": target, "label": label}
        if columns is not None:
            value["columns"] = deepcopy(columns)
        edges.append(value)

    def route(item, incoming, label="", path="pipeline", parent=None, columns=None):
        if isinstance(item, str) or item is None:
            current = node("passthrough" if item is None else item, kind="passthrough" if item in (None, "passthrough") else "drop", path=path, parent=parent)
            edge(incoming, current, label, columns)
            return current
        name = item["type"]
        current = node(name, kind="composition" if name in ("Pipeline", "ColumnTransformer", "FeatureUnion") else "transformer", path=path, data=item, parent=parent)
        edge(incoming, current, label, columns)
        if name == "Pipeline":
            previous = current
            for step in item.get("steps", []):
                previous = route(step["transformer"], previous, step["name"], path + ".steps." + step["name"], current)
            return previous
        if name in ("ColumnTransformer", "FeatureUnion"):
            merger = node("concatenate", kind="merge", path=path + ".output", parent=current)
            field = "transformers" if name == "ColumnTransformer" else "transformer_list"
            for branch in item.get(field, []):
                selected = branch.get("columns") if name == "ColumnTransformer" else "all input features"
                selected_label = branch["name"] + (": " + _selector_label(selected) if name == "ColumnTransformer" else "")
                last = route(branch["transformer"], current, selected_label, path + "." + field + "." + branch["name"], current, selected)
                if branch["transformer"] != "drop":
                    edge(last, merger, "output features")
            if name == "ColumnTransformer":
                remainder = item.get("params", {}).get("remainder", "drop")
                if remainder != "drop":
                    last = route(remainder, current, "remaining columns", path + ".remainder", current, "remaining columns")
                    edge(last, merger, "output features")
            return merger
        return current

    incoming = node("X: selected dataset features", kind="input", path="input")
    last = route(tree, incoming)
    output = node("X transformed → tuning / training", kind="output", path="output")
    edge(last, output, "output features")
    return {"nodes": nodes, "edges": edges}


def _selector_label(selector):
    if isinstance(selector, dict) and "$call" in selector:
        return "make_column_selector(" + ", ".join(f"{key}={value!r}" for key, value in selector.get("params", {}).items() if value is not None) + ")"
    if isinstance(selector, dict) and "$slice" in selector:
        return "slice" + repr(selector["$slice"])
    return repr(selector)


def describe_pipeline(spec):
    normalized = normalize_pipeline(spec)
    transformer = compile_pipeline(normalized)
    tree = transformer_tree(transformer)
    return {**normalized, "spec": normalized, "tree": tree, "graph": pipeline_graph(tree), "warnings": pipeline_warnings(transformer)}


def pipeline_catalogue():
    return {
        "format": FORMAT, "version": VERSION, "sklearn_version": sklearn.__version__,
        "classes": deepcopy(catalogue_classes()), "templates": deepcopy(TEMPLATES),
        "syntax": {"assignment": "pipeline", "constructors": sorted(constructors()), "methods": ["set_output", "set_params"],
                   "allowed": ["imports from catalogue", "literal assignments", "nested constructors", "numpy function references", "slice", "numpy.array"],
                   "unsupported": ["arbitrary Python", "lambda", "loops", "functions/classes", "fit/transform in declaration", "file/URL I/O", "custom estimators", "metadata routing", "fitted estimators"]},
        "limits": {"source_bytes": 100_000, "constructors": 200, "depth": 40, "literal_items": 10_000},
    }


def inspect_pipeline_source(name):
    """Return installed sklearn source for a catalogue name, never a user path."""
    candidates = constructors()
    qualified = name if name in candidates else next((key for key in candidates if key.rsplit(".", 1)[1] == name), None)
    if qualified is None:
        raise PipelineDefinitionError("Исходники доступны только для имени из каталога sklearn.")
    value = candidates[qualified]
    try:
        filename = inspect.getsourcefile(value)
        package = Path(sklearn.__file__).resolve().parent
        path = Path(filename).resolve() if filename else None
        if path is None or not path.is_relative_to(package) or path.suffix != ".py":
            raise PipelineDefinitionError("Исходник не является Python файлом установленного sklearn.")
        lines, first_line = inspect.getsourcelines(value)
    except (OSError, TypeError) as error:
        raise PipelineDefinitionError("Исходник этого конструктора недоступен в установленном sklearn.") from error
    source = "".join(lines)
    truncated = len(source.encode("utf-8")) > 500_000
    if truncated:
        source = source[:125_000]
    return {"name": qualified.rsplit(".", 1)[1], "qualified_name": qualified, "path": str(path),
            "relative_path": str(path.relative_to(package)), "line": first_line, "source": source,
            "truncated": truncated, "sklearn_version": sklearn.__version__, "license": "BSD-3-Clause"}
