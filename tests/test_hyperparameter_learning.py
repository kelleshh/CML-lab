"""Model and Mini-IDE help navigate to contextual Russian constructor articles."""

import json

import numpy as np
import pytest

from cml_lab.contexts.learning.application import LearningService
from cml_lab.infrastructure.hyperparameter_learning import parameter_knowledge
from cml_lab.infrastructure.learning import InMemoryLearningRepository
from cml_lab.infrastructure.ml.catalogue import AlgorithmCatalogue
from cml_lab.infrastructure.ml.pipelines import pipeline_catalogue
from cml_lab.infrastructure.ml.metrics import TaskMetrics


@pytest.fixture(scope="module")
def knowledge():
    models = AlgorithmCatalogue().catalogue()
    pipelines = pipeline_catalogue()["classes"]
    repository = InMemoryLearningRepository(algorithms=models, pipeline_classes=pipelines,
                                            metrics=TaskMetrics().catalogue())
    return models, pipelines, LearningService(repository)


def test_every_model_parameter_opens_its_own_russian_article(knowledge):
    models, _, service = knowledge
    # Independent expectation: every field is publicly navigable by its own URL.
    for model in models:
        assert service.get_lesson(model["lesson_id"])["title"] == model["name"]
        for parameter in model["params"]:
            article = service.get_lesson(parameter["lesson_id"])
            assert article["title"] == f"{model['name']}.{parameter['key']}"
            assert service.get_help(parameter["help_key"])["lesson_id"] == article["id"]
            assert article["example"] and article["exercise"] and article["common_mistakes"]
            assert any("а" <= letter <= "я" for letter in article["summary"].lower())
            assert not article["summary"].startswith("Параметр constructor")
            assert article["sources"][0]["kind"] == "primary"


def test_pipeline_class_anchors_and_individual_parameter_articles_resolve(knowledge):
    _, definitions, service = knowledge
    for definition in definitions:
        article = service.get_lesson(definition["lesson_id"])
        assert article["title"] == definition["name"]
        assert "преобразователь или вложенная модель" not in article["summary"]
        anchors = {section["anchor"] for section in article["sections"] if section.get("anchor")}
        for field in definition["params"]:
            assert f"param-{field['key']}" in anchors
            parameter = service.get_lesson(field["lesson_id"])
            assert parameter["title"] == f"{definition['name']}.{field['key']}"
            assert "train" in " ".join(section["text"] for section in article["sections"])


def test_reference_bodies_are_fetched_on_demand_instead_of_startup_payload(knowledge):
    _, _, service = knowledge
    catalogue = service.catalogue()
    reference = [record for record in catalogue if record["level"] == "reference"]
    assert len(reference) > 2000
    assert all("sections" not in record and record["preset"] is None for record in reference)
    assert service.get_lesson(reference[0]["id"])["sections"]
    assert next(record for record in catalogue if record["id"] == "classification-basics")["preset"]
    assert len(json.dumps(catalogue, ensure_ascii=False)) < 3_000_000


@pytest.mark.parametrize("identifier,class_name,library,key,required,excluded", [
    ("xgboost_classifier", "XGBClassifier", "xgboost", "gamma", "разделения", "локальность влияния"),
    ("lightgbm_regressor", "LGBMRegressor", "lightgbm", "alpha", "quantile", "Сила штрафа"),
    ("adaptive_lasso", "AdaptiveLassoRegressor", "linear_lab", "gamma", "степень", "Локальность"),
    ("sgd_regressor", "SGDRegressor", "sklearn", "epsilon", "epsilon-insensitive", "Добавка Adam"),
    ("mlp_classifier", "MLPClassifier", "sklearn", "epsilon", "Adam", "Ширина полосы"),
    ("tsne", "TSNE", "sklearn", "learning_rate", "t-SNE", "вклад дерева"),
    ("best_subset", "BestSubsetRegressor", "linear_lab", "max_features", "поднабора", "одного разделения"),
    ("ransac", "RANSACRegressor", "sklearn", "min_samples", "подвыборки", "плотности соседства"),
])
def test_same_key_keeps_the_actual_estimator_meaning(identifier, class_name, library, key, required, excluded):
    record = parameter_knowledge(identifier, "", key, class_name=class_name, library=library)
    assert required.lower() in " ".join((record["summary"], record["details"], record["example"])).lower()
    assert excluded.lower() not in record["summary"].lower()


@pytest.mark.parametrize("name,key,required", [
    ("PowerTransformer", "method", "Box-Cox"),
    ("SelectFdr", "alpha", "p-values"),
    ("PolynomialFeatures", "degree", "полиномиальных"),
    ("Binarizer", "threshold", "индикатора"),
    ("SimpleImputer", "strategy", "заполнения"),
    ("TfidfVectorizer", "norm", "строки"),
])
def test_pipeline_fields_use_transformation_semantics(name, key, required):
    record = parameter_knowledge("pipeline-" + name.lower(), "preprocessing", key,
                                 "preprocessing", class_name=name, library="sklearn")
    assert required.lower() in (record["summary"] + record["details"]).lower()


def test_fbeta_lesson_numbers_match_independent_counts_and_executable_metric(knowledge):
    _, _, service = knowledge
    article = service.get_lesson("fbeta-metrics")
    tp, fp, fn = 8, 2, 8
    y = np.array([1]*tp + [0]*fp + [1]*fn)
    prediction = np.array([1]*(tp+fp) + [0]*fn)
    for beta in (0.5, 1, 2):
        expected = (1+beta**2)*tp / ((1+beta**2)*tp + beta**2*fn + fp)
        values, _ = TaskMetrics().evaluate("classification", y, prediction, n_classes=2,
            selection=["fbeta_binary"], metric_params={"beta": beta})
        assert values["fbeta_binary"] == pytest.approx(expected)
        assert f"{expected:.3f}" in article["example"]
    assert "Среднее F-beta" in " ".join(article["common_mistakes"])


def test_unknown_parameter_requires_authoring_instead_of_vague_placeholder():
    assert parameter_knowledge("ridge", "linear", "invented_parameter") == {}


def test_mcp_penalty_weights_and_random_projection_eps_are_not_neighbor_controls(knowledge):
    _, _, service = knowledge
    weights = service.get_help("model.mcp.weights")
    assert "штрафа" in weights["summary"]
    assert "[1,2,1]" in weights["example"]
    assert "Pipeline" in " ".join(weights["cautions"])
    assert "sample_weight" in " ".join(weights["cautions"])
    projection = service.get_help("model.gaussian_random_projection.eps")
    assert "искажения расстояний" in projection["summary"]
    assert "Johnson-Lindenstrauss" in projection["details"]
    assert "радиус" not in projection["summary"]


def test_quantiles_onehot_and_internal_cv_do_not_inherit_boosting_or_ordinal_meaning(knowledge):
    _, _, service = knowledge
    quantile = service.get_lesson("parameter-pipeline-quantiletransformer-subsample")
    assert "число train-строк" in quantile["summary"]
    assert "10000" in quantile["example"]
    onehot = service.get_lesson("parameter-pipeline-onehotencoder-categories")
    assert "one-hot" in onehot["summary"]
    assert "не числовые коды" in onehot["example"]
    stacking = service.get_help("model.stacking_regressor.cv")
    assert "OOF" in stacking["details"]
    assert "TargetEncoder" not in stacking["summary"]


def test_transductive_articles_explain_the_new_rows_boundary(knowledge):
    models, _, service = knowledge
    for definition in models:
        article = service.get_lesson(definition["lesson_id"])
        application = next(section["text"] for section in article["sections"] if section["title"] == "Применение сохраненной модели")
        if definition["capabilities"].get("transductive"):
            assert "новых исходных строк" in application
            assert "не поддержан" in application
            assert "отдельным fit" in application
        else:
            assert "При поддержке" in application
    ols = service.get_lesson("model-ols")
    assert "Матрица ошибок" not in " ".join(section["text"] for section in ols["sections"])


@pytest.mark.parametrize("name,key,required,forbidden", [
    ("NMF", "shuffle", "координат", "наблюдений"),
    ("TargetEncoder", "shuffle", "cross-fitting", "SGD"),
    ("BaggingRegressor", "max_features", "одного базового", "разделения"),
    ("BestSubsetRegressor", "alpha", "L0", "величины параметров"),
    ("PCA", "n_components", "правило выбора", "кластеров"),
])
def test_additional_constructor_overloads_have_exact_operation(name, key, required, forbidden):
    record = parameter_knowledge("pipeline-"+name.lower(), "", key, class_name=name, library="sklearn")
    assert required in record["summary"]
    assert forbidden not in record["summary"]


def test_extended_classification_lesson_averages_follow_independent_class_counts(knowledge):
    _, _, service = knowledge
    article = service.get_lesson("classification-metrics")
    text = " ".join(part["text"] for part in article["sections"])
    actual = np.array([0,0,0,1])
    predicted = np.array([0,0,1,1])
    values, _ = TaskMetrics().evaluate("classification", actual, predicted, n_classes=2,
        selection=["precision_macro", "recall_macro", "f1_macro", "f1_weighted", "jaccard_macro", "hamming_loss", "zero_one_loss"])
    assert values["precision_macro"] == pytest.approx((1+.5)/2)
    assert values["recall_macro"] == pytest.approx((2/3+1)/2)
    assert values["f1_macro"] == pytest.approx((.8+2/3)/2)
    assert values["f1_weighted"] == pytest.approx((3*.8+2/3)/4)
    assert values["jaccard_macro"] == pytest.approx((2/3+.5)/2)
    assert values["hamming_loss"] == values["zero_one_loss"] == .25
    for key in ("recall_macro", "f1_macro", "f1_weighted", "jaccard_macro"):
        assert f"{values[key]:.3f}" in text
    assert "multilabel" in text and "OvO" in text and "OvR" in text


def test_extended_cluster_lesson_distinguishes_singletons_from_one_cluster(knowledge):
    _, _, service = knowledge
    article = service.get_lesson("clustering-metrics")
    text = " ".join(part["text"] for part in article["sections"])
    truth = np.array([0,0,1,1])
    metrics = ["homogeneity", "completeness", "v_measure", "normalized_mutual_info", "fowlkes_mallows"]
    singletons, _ = TaskMetrics().evaluate("clustering", None, np.arange(4), reference=truth, selection=metrics)
    assert singletons["homogeneity"] == pytest.approx(1)
    assert singletons["completeness"] == pytest.approx(.5)
    assert singletons["v_measure"] == pytest.approx(2/3)
    assert singletons["normalized_mutual_info"] == pytest.approx(2/3)
    assert singletons["fowlkes_mallows"] == 0
    all_together, _ = TaskMetrics().evaluate("clustering", None, np.zeros(4), reference=truth, selection=metrics)
    assert all_together["completeness"] == pytest.approx(1)
    assert all_together["homogeneity"] == 0
    assert all_together["fowlkes_mallows"] == pytest.approx(np.sqrt(1/3))
    assert f"{np.sqrt(1/3):.3f}" in text
    assert "арифметическую" in text and "не поправлена" in text
