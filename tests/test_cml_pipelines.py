"""User contracts for nested sklearn declarations, train isolation and exports."""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from cml_lab.infrastructure.ml.pipelines import (
    PipelineDefinitionError, compile_pipeline, describe_pipeline, emit_pipeline_source,
    inspect_pipeline_source, normalize_pipeline, pipeline_catalogue, prepare_pipeline_for_fit,
)


MIXED = '''from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures

numeric = Pipeline([
    ("fill", SimpleImputer(strategy="median", keep_empty_features=True)),
    ("scale", StandardScaler()),
])
pipeline = Pipeline([("columns", ColumnTransformer([
    ("number", numeric, ["x"]),
    ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["kind"]),
], remainder="drop"))])
'''


def test_numeric_and_category_branches_use_train_only_and_drop_id():
    train = pd.DataFrame({"x": [0., 2., np.nan], "kind": ["a", "b", "a"], "id": [1, 2, 3]})
    holdout = pd.DataFrame({"x": [1000., np.nan], "kind": ["new", "a"], "id": [-99, -999]})
    declared = compile_pipeline(MIXED)
    assert not hasattr(declared["columns"], "transformers_")
    pipeline = prepare_pipeline_for_fit(declared)
    observed = pipeline.fit_transform(train)
    # Median 1; mean 1; standard deviation sqrt(2/3). The unseen category
    # produces zeros and held-out values cannot move the learned mean.
    np.testing.assert_allclose(observed[:, 0], [-np.sqrt(1.5), np.sqrt(1.5), 0])
    result = pipeline.transform(holdout)
    np.testing.assert_allclose(result[0, 0], 999 * np.sqrt(1.5))
    np.testing.assert_allclose(result[0, 1:], 0)
    np.testing.assert_allclose(result[1, 0], 0)
    assert result.shape == (2, 3)
    assert not any("id" in name for name in pipeline.get_feature_names_out())
    assert not hasattr(declared["columns"], "transformers_")


def test_nested_union_returns_both_real_feature_branches_and_survives_pickle():
    source = '''from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
pipeline = FeatureUnion([
    ("identity", "passthrough"),
    ("squared", Pipeline([("powers", PolynomialFeatures(degree=2, include_bias=False))])),
])
'''
    X = pd.DataFrame({"x": [1., 2., 3.]})
    model = prepare_pipeline_for_fit(compile_pipeline(source))
    np.testing.assert_allclose(model.fit_transform(X), [[1, 1, 1], [2, 2, 4], [3, 3, 9]])
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_allclose(restored.transform(X), model.transform(X))
    np.testing.assert_allclose(clone(model).fit_transform(X), model.transform(X))


def test_fold_clones_have_independent_statistics_and_no_fitted_template():
    source = "from sklearn.pipeline import Pipeline\nfrom sklearn.preprocessing import StandardScaler\npipeline = Pipeline([('scale', StandardScaler())])\n"
    template = compile_pipeline(source)
    first = prepare_pipeline_for_fit(template).fit(pd.DataFrame({"x": [0., 2.]}))
    second = prepare_pipeline_for_fit(template).fit(pd.DataFrame({"x": [10., 20.]}))
    np.testing.assert_allclose(first["scale"].mean_, [1])
    np.testing.assert_allclose(second["scale"].mean_, [15])
    assert not hasattr(template["scale"], "mean_")


@pytest.mark.parametrize("template", pipeline_catalogue()["templates"], ids=lambda item: item["id"])
def test_editor_tree_source_roundtrip_preserves_nested_construction(template):
    described = describe_pipeline({"format": "cml.pipeline", "version": 1, "source": template["source"]})
    serialized = json.loads(json.dumps(described["tree"]))
    source = emit_pipeline_source(serialized)
    rewritten = describe_pipeline({"source": source})
    assert rewritten["tree"] == described["tree"]
    assert described["spec"]["source"] == template["source"]
    assert any(node["kind"] == "input" for node in described["graph"]["nodes"])
    assert any(node["kind"] == "output" for node in described["graph"]["nodes"])


def test_graph_labels_actual_column_directions_and_has_parallel_merge():
    graph = describe_pipeline(MIXED)["graph"]
    selected = [edge["columns"] for edge in graph["edges"] if isinstance(edge.get("columns"), list)]
    assert ["x"] in selected
    assert ["kind"] in selected
    merger = next(node["id"] for node in graph["nodes"] if node["kind"] == "merge")
    assert sum(edge["target"] == merger for edge in graph["edges"]) == 2


def test_function_reference_and_numpy_array_are_roundtrippable():
    source = '''import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, SplineTransformer
pipeline = Pipeline([
    ("log", FunctionTransformer(func=np.log1p, inverse_func=np.expm1, feature_names_out="one-to-one")),
    ("spline", SplineTransformer(knots=np.array([[0.], [1.], [2.]]), degree=2)),
])
'''
    result = describe_pipeline(source)
    assert result["tree"]["steps"][0]["transformer"]["params"]["func"] == {"$ref": "numpy.log1p"}
    assert result["tree"]["steps"][1]["transformer"]["params"]["knots"]["$array"] == [[0.], [1.], [2.]]
    replica = compile_pipeline({"tree": result["tree"]})
    X = pd.DataFrame({"x": [0., 1., 2., 3.]})
    np.testing.assert_allclose(replica.fit_transform(X), compile_pipeline(source).fit_transform(X))


def test_module_aliases_selector_and_set_output_are_real_sklearn():
    source = '''import sklearn.preprocessing as prep
import sklearn.compose as compose
import numpy as np
pipeline = compose.ColumnTransformer([
    ("numbers", prep.StandardScaler(), compose.make_column_selector(dtype_include=np.number)),
]).set_output(transform="pandas")
'''
    pipeline = compile_pipeline(source)
    X = pd.DataFrame({"number": [1., 3.], "text": ["a", "b"]})
    output = pipeline.fit_transform(X)
    assert isinstance(output, pd.DataFrame)
    assert output.columns.tolist() == ["numbers__number"]
    np.testing.assert_allclose(output.iloc[:, 0], [-1, 1])
    assert describe_pipeline({"tree": describe_pipeline(source)["tree"]})["tree"]["output"] == "pandas"


def test_select_from_model_fits_nested_estimator_and_exports_tree():
    source = '''from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
pipeline = Pipeline([("select", SelectFromModel(estimator=Lasso(alpha=0.01), threshold="mean"))])
'''
    X = pd.DataFrame({"signal": [-2., -1., 0., 1., 2.], "constant": [1., 1., 1., 1., 1.]})
    result = prepare_pipeline_for_fit(compile_pipeline(source)).fit_transform(X, X.signal * 3)
    np.testing.assert_allclose(result.ravel(), X.signal)
    tree = describe_pipeline(source)["tree"]
    assert tree["steps"][0]["transformer"]["params"]["estimator"]["type"] == "Lasso"
    assert describe_pipeline({"tree": tree})["tree"] == tree


@pytest.mark.parametrize("source", [
    "import os\npipeline = os.system('echo bad')",
    "from sklearn.preprocessing import StandardScaler\npipeline = StandardScaler().fit([[1]])",
    "from sklearn.preprocessing import FunctionTransformer\npipeline = FunctionTransformer(lambda x: x)",
    "from sklearn.pipeline import Pipeline\npipeline = Pipeline([]); print('bad')",
    "from sklearn.pipeline import Pipeline\npipeline = Pipeline([], memory='/tmp/cache')",
    "from sklearn.feature_extraction.text import TfidfVectorizer\npipeline = TfidfVectorizer(input='filename')",
    "from sklearn.feature_selection import SelectFromModel\nfrom sklearn.linear_model import Lasso\npipeline = SelectFromModel(Lasso(), prefit=True)",
    "from sklearn.preprocessing import StandardScaler\npipeline = StandardScaler().__class__",
    "from sklearn.preprocessing import StandardScaler\npipeline = [StandardScaler() for i in range(10)]",
    "from sklearn.preprocessing import StandardScaler\npipeline = StandardScaler(**{'copy':False})",
])
def test_arbitrary_code_and_file_actions_are_rejected_with_position(source):
    with pytest.raises(PipelineDefinitionError) as captured:
        compile_pipeline(source)
    assert captured.value.line >= 1
    assert captured.value.column >= 1


def test_parse_error_reports_line_column_and_original_source_is_preserved():
    with pytest.raises(PipelineDefinitionError) as captured:
        compile_pipeline("from sklearn.preprocessing import StandardScaler\npipeline = StandardScaler(typo=1)\n")
    assert captured.value.line == 2
    assert captured.value.column == 12
    assert "typo" in captured.value.message
    commented = "# train statistics only\n" + MIXED
    assert normalize_pipeline(commented)["source"] == commented


def test_catalogue_uses_actual_constructor_names_keys_and_installed_defaults():
    classes = {item["name"]: item for item in pipeline_catalogue()["classes"]}
    standard = classes["StandardScaler"]
    assert standard["module"] == "sklearn.preprocessing"
    assert {item["key"] for item in standard["params"]} == {"copy", "with_mean", "with_std"}
    assert all(item["description"] and item["doc"] for item in standard["params"])
    assert {item["key"]: item["default"] for item in standard["params"]}["with_mean"] is True
    assert classes["RandomForestRegressor"]["kind"] == "estimator"
    assert standard["lesson_id"] == "pipeline-standardscaler"


def test_source_view_reads_real_sklearn_class_only():
    source = inspect_pipeline_source("StandardScaler")
    assert "class StandardScaler" in source["source"]
    assert source["relative_path"] == "preprocessing/_data.py"
    assert source["line"] > 1
    with pytest.raises(PipelineDefinitionError):
        inspect_pipeline_source("../../run.py")
    with pytest.raises(PipelineDefinitionError):
        inspect_pipeline_source("os.system")


@pytest.mark.parametrize("task,split", [("forecasting", "timeseries"), ("panel", "group_kfold"), ("classification", "group_shuffle_split")])
def test_target_encoder_internal_cv_is_rejected_in_nested_ordered_pipeline(task, split):
    source = '''from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import TargetEncoder
pipeline = ColumnTransformer([("target", TargetEncoder(), ["category"])])
'''
    with pytest.raises(PipelineDefinitionError, match="TargetEncoder"):
        prepare_pipeline_for_fit(compile_pipeline(source), task=task, split_kind=split)


def test_polynomial_budget_rejects_before_expanded_matrix_is_allocated():
    source = "from sklearn.preprocessing import PolynomialFeatures\npipeline = PolynomialFeatures(degree=5)\n"
    X = pd.DataFrame(np.ones((2, 30)))
    guarded = prepare_pipeline_for_fit(compile_pipeline(source))
    with pytest.raises(ValueError, match="2000"):
        guarded.fit_transform(X)
    assert not hasattr(guarded, "powers_")


def test_one_hot_budget_counts_real_output_including_drop_and_infrequent():
    source = "from sklearn.preprocessing import OneHotEncoder\npipeline = OneHotEncoder(sparse_output=False, max_categories=2, drop='first', handle_unknown='infrequent_if_exist')\n"
    X = pd.DataFrame({"category": ["a", "b", "c", "a"]})
    guarded = prepare_pipeline_for_fit(compile_pipeline(source))
    assert guarded.fit_transform(X).shape == (4, 1)
    wide = prepare_pipeline_for_fit(compile_pipeline("from sklearn.preprocessing import OneHotEncoder\npipeline = OneHotEncoder(sparse_output=False)"))
    with pytest.raises(ValueError, match="2000"):
        wide.fit_transform(pd.DataFrame({"category": [f"c{i}" for i in range(2001)]}))


def test_kernel_and_iterative_budgets_cover_fit_transform_path():
    kernel = prepare_pipeline_for_fit(compile_pipeline("from sklearn.decomposition import KernelPCA\npipeline = KernelPCA(n_components=2)"))
    with pytest.raises(ValueError, match="попарных"):
        kernel.fit_transform(pd.DataFrame(np.zeros((4000, 2))))
    imputer = prepare_pipeline_for_fit(compile_pipeline("from sklearn.impute import IterativeImputer\npipeline = IterativeImputer()"))
    with pytest.raises(ValueError, match="100 фич"):
        imputer.fit_transform(pd.DataFrame(np.zeros((5, 101))))


def test_declarative_limits_reject_excessive_literal_and_constructor_input():
    with pytest.raises(PipelineDefinitionError, match="100 000"):
        compile_pipeline("#" * 100_001)
    with pytest.raises(PipelineDefinitionError, match="20"):
        compile_pipeline("from sklearn.preprocessing import PolynomialFeatures\npipeline = PolynomialFeatures(degree=1000)")


@pytest.mark.parametrize("name", ["PLSRegression", "PLSCanonical", "CCA"])
def test_supervised_decomposition_passes_only_x_scores_and_frozen_y_state(name):
    source = f"from sklearn.cross_decomposition import {name}\nfrom sklearn.pipeline import Pipeline\npipeline = Pipeline([('reduce', {name}(n_components=1))])"
    X = pd.DataFrame(np.random.default_rng(4).normal(size=(30, 3)), columns=["a", "b", "c"])
    y = X.a.to_numpy() * 3 + X.b.to_numpy()
    fitted = prepare_pipeline_for_fit(compile_pipeline(source))
    observed = fitted.fit_transform(X, y)
    assert observed.shape == (30, 1)
    np.testing.assert_allclose(observed, fitted.transform(X))
    holdout = pd.DataFrame({"a": [100.], "b": [20.], "c": [1.]})
    assert fitted.transform(holdout).shape == (1, 1)
    assert any("X scores" in item for item in describe_pipeline(source)["warnings"])


@pytest.mark.parametrize("key,value", [("memory", "/tmp/user-cache"), ("transform_input", ["y"])])
def test_post_parse_tuning_overrides_cannot_enable_file_cache_or_metadata(key, value):
    declared = compile_pipeline("from sklearn.pipeline import Pipeline\nfrom sklearn.preprocessing import StandardScaler\npipeline = Pipeline([('scale', StandardScaler())])")
    declared.set_params(**{key: value})
    with pytest.raises(PipelineDefinitionError):
        prepare_pipeline_for_fit(declared)


def test_slice_and_boolean_column_masks_are_faithful():
    source = '''from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
pipeline = ColumnTransformer([
    ("first", MinMaxScaler(), slice(0, 2)),
    ("last", "passthrough", [False, False, True]),
])
'''
    tree = describe_pipeline(source)["tree"]
    assert tree["transformers"][0]["columns"] == {"$slice": [0, 2, None]}
    pipeline = compile_pipeline({"tree": tree})
    np.testing.assert_allclose(pipeline.fit_transform(pd.DataFrame([[1., 10., 7.], [3., 20., 9.]], columns=["a", "b", "c"])), [[0, 0, 7], [1, 1, 9]])


def test_nested_set_params_is_configuration_only_and_roundtrips_actual_values():
    source = '''from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
pipeline = Pipeline([("scale", StandardScaler())]).set_params(scale__with_mean=False)
'''
    value = compile_pipeline(source)
    assert value["scale"].with_mean is False
    np.testing.assert_allclose(value.fit_transform(pd.DataFrame({"x": [1., 3.]})), [[1], [3]])
    assert describe_pipeline({"tree": describe_pipeline(source)["tree"]})["tree"]["steps"][0]["transformer"]["params"]["with_mean"] is False
    with pytest.raises(PipelineDefinitionError, match="memory"):
        compile_pipeline(source.replace("scale__with_mean=False", "memory='/tmp/cache'"))


def test_builtin_dtype_references_match_common_column_selector_examples():
    source = '''from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder
pipeline = ColumnTransformer([("categories", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=float), make_column_selector(dtype_include=object))])
'''
    tree = describe_pipeline(source)["tree"]
    model = compile_pipeline({"tree": tree})
    X = pd.DataFrame({"kind": ["a", "b"], "x": [1., 2.]})
    np.testing.assert_allclose(model.fit_transform(X), np.eye(2))
    with pytest.raises(PipelineDefinitionError):
        compile_pipeline("pipeline = object()")
