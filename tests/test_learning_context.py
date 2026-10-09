"""Учебные ссылки, воспроизводимые формы и смысл перегруженных параметров."""

import json
import re
from urllib.parse import urlparse

import pytest

from cml_lab.contexts.learning.application import LearningService
from cml_lab.infrastructure.learning import InMemoryLearningRepository
from cml_lab.infrastructure.ml.catalogue import AlgorithmCatalogue
from cml_lab.infrastructure.ml.metrics import TaskMetrics
from cml_lab.infrastructure.ml.preparation_schema import PreparationCatalogue
from linear_lab.teaching import LESSONS as LEGACY_LESSONS


@pytest.fixture(scope="module")
def algorithms():
    return AlgorithmCatalogue()


@pytest.fixture(scope="module")
def service(algorithms):
    preparation = PreparationCatalogue()
    return LearningService(InMemoryLearningRepository(
        algorithms=algorithms.catalogue(),
        stages=[*preparation.stages(), *preparation.samplers()],
        metrics=TaskMetrics().catalogue(),
    ))


def test_legacy_links_and_explanations_are_preserved():
    service = LearningService(InMemoryLearningRepository())
    for previous in LEGACY_LESSONS:
        current = service.get_lesson(previous["id"])
        assert current["title"] == previous["title"]
        assert [(item["title"], item["text"], item["formula"]) for item in current["sections"]] == [
            (item["title"], item["text"], item.get("formula")) for item in previous["sections"]
        ]
        assert current["preset"]["dataset"] == previous["preset"]["dataset"]
        assert current["preset"]["experiment"]["algorithm_id"] == previous["preset"]["model"]


def test_curriculum_has_concrete_material_for_every_supported_task(service):
    for task in ("regression", "classification", "clustering", "ranking", "forecasting", "panel", "anomaly", "reduction"):
        listing = service.list_lessons(task=task)
        assert listing["items"], task
        for summary in listing["items"]:
            lesson = service.get_lesson(summary["id"])
            assert lesson["example"] and lesson["exercise"] and lesson["common_mistakes"]
            assert all(section["text"].strip() for section in lesson["sections"])
            for source in lesson["sources"]:
                parsed = urlparse(source["url"])
                assert parsed.scheme == "https" and parsed.hostname


def test_every_open_catalogue_form_links_to_real_material(service, algorithms):
    preparation = PreparationCatalogue()
    for definition in [*algorithms.catalogue(), *preparation.stages(), *preparation.samplers()]:
        assert service.get_lesson(definition["lesson_id"])["id"] == definition["lesson_id"]
        for field in definition.get("params", []):
            help_item = service.get_help(field["help_key"])
            assert help_item["lesson_id"] == field["lesson_id"]
            assert help_item["summary"].strip() and help_item["example"].strip()
            assert help_item["sources"] and help_item["details"]
    for metric in TaskMetrics().catalogue():
        key = metric.get("help_key", f"metric.{metric['task']}.{metric['id']}")
        assert service.get_help(key)["lesson_id"] == metric["lesson_id"]


def test_parameter_help_does_not_mix_svr_huber_or_scad_with_rbf(service):
    huber = service.get_help("model.huber.epsilon")
    svr = service.get_help("model.linear_svr.epsilon")
    scad = service.get_help("model.scad.gamma")
    rbf = service.get_help("model.svc.gamma")
    assert "остаток/масштаб" in huber["details"]
    assert "1.35" in huber["example"]
    assert "1000" in svr["example"]
    assert "невыпукл" in scad["details"] + " ".join(scad["cautions"])
    assert "локаль" in rbf["details"]
    assert scad["lesson_id"] != rbf["lesson_id"]
    assert re.search(r"(?:меньше|малое).*сильн", service.get_help("model.logistic_regression.C")["summary"].casefold())


def test_numerical_examples_use_the_declared_metric_conventions(service):
    # Counts are chosen independently of the prose: TP=6, FN=4, FP=4, TN=86.
    tp, fn, fp, tn = 6, 4, 4, 86
    classification = service.get_lesson("classification-metrics")["example"]
    assert f"{(tp + tn) / (tp + tn + fp + fn):.2f}" in classification
    assert f"{tp / (tp + fp):.1f}" in classification
    # Public CML contract uses gains 2**relevance-1, independently calculated here.
    import math
    observed = 7.5 / (7 + 1 / math.log2(3))
    assert f"{observed:.3f}" in service.get_lesson("ranking-metrics")["example"]


def test_safety_explanations_cover_time_groups_target_encoding_and_test(service):
    for identifier, tokens in {
        "forecasting-lags": ("прошл", "будущ"),
        "panel-groups": ("A", "B", "лаг"),
        "ranking-groups": ("запрос", "разрез"),
        "categorical-encoding": ("fit_transform", "врем"),
        "classification-resampling": ("обучен", "провер"),
        "experiment-versions": ("100", "300", "снимок"),
    }.items():
        article = service.get_lesson(identifier)
        text = " ".join(section["text"] for section in article["sections"])
        assert all(token in text for token in tokens), identifier


def test_unknown_links_fail_and_public_values_are_independent(service):
    with pytest.raises(KeyError):
        service.get_lesson("not-a-lesson")
    with pytest.raises(KeyError):
        service.get_help("model.unknown.epsilon")
    lesson = service.get_lesson("tree-depth")
    lesson["preset"]["experiment"]["params"]["max_depth"] = 99
    lesson["sections"][0]["text"] = "изменено"
    assert service.get_lesson("tree-depth")["preset"]["experiment"]["params"]["max_depth"] == 2
    assert service.get_lesson("tree-depth")["sections"][0]["text"] != "изменено"
    help_item = service.get_help("target")
    help_item["sources"].clear()
    assert service.get_help("target")["sources"]
    json.dumps(service.catalogue(), allow_nan=False)


def test_search_and_filters_keep_task_semantics(service):
    found = service.list_lessons(query="запрос", task="ranking")
    assert {item["id"] for item in found["items"]} >= {"ranking-groups"}
    assert all("ranking" in item["tasks"] for item in found["items"])
    assert not service.list_lessons(query="совсемнесуществующееслово")["items"]
    assert service.list_lessons(family="tree")["items"]


@pytest.mark.parametrize("lesson", InMemoryLearningRepository().lessons(), ids=lambda lesson: lesson.id)
def test_every_lesson_preset_uses_real_algorithm_and_accepted_settings(algorithms, lesson):
    preset = lesson.preset
    assert preset is not None
    spec = preset["experiment"]
    definition = algorithms.descriptor(spec["algorithm_id"])
    assert spec["task"] in definition["tasks"]
    assert set(spec.get("params", {})) <= {field["key"] for field in definition["params"]}
    assert preset["dataset"]["kind"] in {"synthetic", "sklearn"}
    if not definition["available"]:
        assert definition["reason"]
        return
    model = algorithms.build(spec["task"], spec["algorithm_id"], spec.get("params", {}), seed=42, n_jobs=1)
    assert callable(model.fit)


def test_hand_calculated_ranking_example_matches_executable_metric(service):
    import math
    import numpy as np

    values, details = TaskMetrics().evaluate(
        "ranking", np.array([3, 0, 1]), np.array([0.9, 0.5, 0.1]),
        groups=np.array(["query", "query", "query"]), selection=["ndcg"], metric_params={"k": 3},
    )
    expected = (7 + 1 / 2) / (7 + 1 / math.log2(3))
    assert details["ndcg"]["reason"] is None
    assert values["ndcg"] == pytest.approx(expected)
    assert f"{expected:.3f}" in service.get_lesson("ranking-metrics")["example"]


def test_probability_examples_match_small_independent_dataset(service):
    import math
    import numpy as np

    actual = np.array([1, 0])
    proba = np.array([[0.2, 0.8], [0.7, 0.3]])
    values, _ = TaskMetrics().evaluate("classification", actual, actual,
                                     probabilities=proba, n_classes=2, selection=["brier", "log_loss"])
    expected_brier = ((0.8 - 1)**2 + 0.3**2) / 2
    expected_log_loss = -(math.log(0.8) + math.log(0.7)) / 2
    assert values["brier"] == pytest.approx(expected_brier)
    assert values["log_loss"] == pytest.approx(expected_log_loss)
    text = " ".join(section["text"] for section in service.get_lesson("classification-metrics")["sections"])
    assert f"{expected_brier:.3f}" in text
    assert f"{expected_log_loss:.3f}" in text


def test_new_preparation_lessons_build_real_pipeline_with_declared_shapes(service):
    import numpy as np
    import pandas as pd
    from cml_lab.infrastructure.ml.preparation import build_preprocessor

    lesson = service.get_lesson("spline-features")
    x = pd.DataFrame({"age": np.linspace(1, 10, 20)})
    pipeline = build_preprocessor(x, lesson["preset"]["experiment"]["preprocessing"])
    output = pipeline.fit_transform(x)
    assert output.shape == (20, 6)
    assert np.isfinite(output).all()
    imputation = service.get_lesson("iterative-imputation")
    inputs = pd.DataFrame({"area": [30., 60., 90., np.nan], "rooms": [1., 2., 3., 4.]})
    prepared = build_preprocessor(inputs, imputation["preset"]["experiment"]["preprocessing"])
    result = prepared.fit_transform(inputs)
    assert result.shape == inputs.shape
    assert np.isfinite(result).all()
    # This known relationship tests filling from the other column, rather than the global median.
    assert result[-1, 0] == pytest.approx(120, abs=1)


def test_application_startup_registers_every_new_preparation_link(tmp_path):
    from fastapi.testclient import TestClient
    from cml_lab.presentation.http.api import create_app

    with TestClient(create_app(tmp_path)) as client:
        for identifier in ("iterative-imputation", "spline-features"):
            response = client.get(f"/api/learning/lessons/{identifier}")
            assert response.status_code == 200
            assert response.json()["id"] == identifier


def test_diagnostic_anomaly_fraction_does_not_claim_quality_direction(service):
    help_item = service.get_help("metric.anomaly.anomaly_fraction")
    assert "Диагностический" in help_item["details"]
    assert "Больше — лучше" not in help_item["details"]
    assert "Меньше — лучше" not in help_item["details"]
