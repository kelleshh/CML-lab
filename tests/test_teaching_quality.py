"""Public learning catalogue, reproducible presets and contextual instruction links.

These checks catch broken lesson navigation and examples that cannot be loaded.
They do not claim to assess writing quality or verify remote page availability.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from urllib.parse import urlparse

import numpy as np
import pytest

from linear_lab.datasets import DataService
from linear_lab.metrics import MetricRegistry
from linear_lab.models import ModelRegistry
from linear_lab.teaching import GLOSSARY, LESSONS, METRIC_HELP


PROJECT = Path(__file__).resolve().parents[1]


def test_public_learning_catalogue_has_complete_articles_and_https_sources():
    """Every article opened from the GUI has an exercise, example and reading."""
    ids = {lesson["id"] for lesson in LESSONS}
    assert len(LESSONS) == len(ids) == 32
    assert {int(identifier.split("-", 1)[0]) for identifier in ids} == set(range(1, 33))
    json.dumps({"lessons": LESSONS, "glossary": GLOSSARY, "metric_help": METRIC_HELP}, allow_nan=False)
    for lesson in LESSONS:
        assert lesson["title"].strip() and lesson["summary"].strip()
        assert lesson["exercise"].strip() and lesson["preset"]
        assert lesson["sections"], lesson["id"]
        assert all(part["title"].strip() and part["text"].strip() for part in lesson["sections"])
        assert lesson["sources"], lesson["id"]
        for source in lesson["sources"]:
            parsed = urlparse(source["url"])
            assert parsed.scheme == "https" and parsed.hostname, (lesson["id"], source)
            assert source["title"].strip()
    assert all(term["lesson_id"] in ids for term in GLOSSARY)
    assert {metric["id"] for metric in MetricRegistry().catalogue()} <= METRIC_HELP.keys()


@pytest.mark.parametrize("lesson", LESSONS, ids=lambda item: item["id"])
def test_every_learning_preset_loads_valid_data_and_supported_model_settings(tmp_path, lesson):
    """The run-example button sends real load/create requests, not prose labels."""
    preset = lesson["preset"]
    # These are the inputs the lesson-run button consumes. Unexpected options
    # would otherwise be silently ignored and make an exercise irreproducible.
    assert set(preset) <= {"dataset", "model", "params", "epochs", "preprocessing", "resampling", "cv_config", "search"}
    registry = ModelRegistry()
    definitions = {model["id"]: model for model in registry.catalogue()}
    definition = definitions[preset["model"]]
    assert set(preset["params"]) <= {field["key"] for field in definition["params"]}
    estimator = registry.create(preset["model"], preset["params"], seed=42)
    assert callable(estimator.fit) and callable(estimator.predict)
    assert 1 <= preset.get("epochs", 100) <= 3000

    data = DataService(tmp_path / "data")
    metadata = data.load(preset["dataset"])
    bundle = data.resolve(metadata["id"], target=metadata["default_target"])
    assert len(bundle.y) == preset["dataset"]["params"]["n_samples"]
    assert len(bundle.feature_names) == preset["dataset"]["params"]["n_features"]
    assert np.isfinite(bundle.y).all()
    assert np.isfinite(bundle.X.to_numpy(dtype=float)).all()
    assert bundle.target_name not in bundle.feature_names
    if preset["model"] in {"poisson", "gamma"}:
        assert np.all(bundle.y >= 0 if preset["model"] == "poisson" else bundle.y > 0)


def test_contextual_help_points_to_existing_and_model_specific_lessons():
    """A question mark next to SVR must open its own epsilon/C explanation."""
    source = (PROJECT / "web" / "help.js").read_text()
    ids = {lesson["id"] for lesson in LESSONS}
    referenced = set(re.findall(r"'(\d{2}-[a-z-]+)'", source))
    assert referenced <= ids
    block = source.split("export const MODEL_LESSONS = Object.freeze({", 1)[1].split("});", 1)[0]
    mappings = dict(re.findall(r"(\w+)\s*:\s*'(\d{2}-[a-z-]+)'", block))
    assert {model["id"] for model in ModelRegistry().catalogue()} <= mappings.keys()
    assert mappings["linear_svr"] == "31-linear-svr"
    assert mappings["weighted_lasso"] == "32-weighted-lasso"
    assert mappings["sqrt_lasso"] == "19-nonconvex"

    settings = {
        "cv": "26-cross-validation",
        "cv-method": "26-cross-validation",
        "search-method": "27-hyperparameter-search",
        "search-trials": "27-hyperparameter-search",
        "resampling-method": "28-rare-target-smoter",
        "method": "28-rare-target-smoter",
        "relevance_threshold": "28-rare-target-smoter",
        "focus": "28-rare-target-smoter",
        "sampling": "28-rare-target-smoter",
        "neighbors": "28-rare-target-smoter",
        "perturbation": "28-rare-target-smoter",
    }
    for key, identifier in settings.items():
        line = next(line for line in source.splitlines() if re.match(rf"\s*(?:'{re.escape(key)}'|{re.escape(key)}): entry\(", line))
        assert re.search(r"'(\d{2}-[a-z-]+)'", line).group(1) == identifier

