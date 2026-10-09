"""Search winners, fold-local preparation, budgets and cancellation on real ML."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.metrics import mean_squared_error, log_loss
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from sklearn.preprocessing import LabelEncoder

from cml_lab.infrastructure.ml.catalogue import AlgorithmCatalogue
from cml_lab.infrastructure.ml.fitting import fit_artifact
from cml_lab.infrastructure.ml.metrics import TaskMetrics
from cml_lab.infrastructure.ml import tuning


@pytest.fixture(scope="module")
def catalogue():
    return AlgorithmCatalogue()


@pytest.fixture
def regression():
    random = np.random.default_rng(812)
    X = pd.DataFrame(random.normal(size=(126, 3)), columns=["a", "b", "c"])
    y = 2 * X.a.to_numpy() - X.b.to_numpy() + random.normal(0, .8, len(X))
    return X, y, list(KFold(3, shuffle=True, random_state=41).split(X))


def spec(method="grid", space=None, **overrides):
    return {"task": "regression", "algorithm_id": "ridge", "params": {"fit_intercept": True},
            "preprocessing": {"steps": [], "resampling": {"method": "none"}}, "seed": 51, "n_jobs": 1,
            "search": {"method": method, "trials": 4, "metric": "rmse", "direction": "auto",
                       "param_space": space or {"alpha": [.01, 1, 100]}}, **overrides}


@pytest.mark.parametrize("method,backend", [("grid", "ParameterGrid"), ("random", "ParameterSampler"),
    ("optuna_tpe", "TPESampler"), ("optuna_random", "RandomSampler"),
    ("halving_grid", "HalvingSearchCV"), ("halving_random", "HalvingSearchCV")])
def test_all_six_methods_publish_real_scores_and_correct_winner(method, backend, catalogue, regression):
    X, y, plans = regression
    request = spec(method)
    original = deepcopy(request)
    events = []
    best, summary = tuning.tune(request, X, y, catalogue, TaskMetrics(), plans, progress=events.append)
    assert backend in summary["backend"]
    assert request == original and best == summary["best_params"]
    assert summary["direction"] == "min" and summary["refit"] is False
    assert 0 < summary["n_fits"] <= 500 and events
    records = summary["trials"]
    eligible = records
    if method.startswith("halving"):
        last = max(record["iteration"] for record in records)
        eligible = [record for record in records if record["iteration"] == last]
        resources = sorted({record["resources"] for record in records})
        assert len(resources) >= 2 and resources[-1] > resources[0]
    scores = [record["score"] for record in eligible if record["score"] is not None]
    assert summary["best_score"] == min(scores)
    assert all(record["score"] == pytest.approx(np.mean(record["fold_scores"])) for record in eligible if record["score"] is not None)
    assert summary["n_fits"] == sum(record["fits"] for record in records)


def test_grid_scores_equal_independent_complete_fold_fits(catalogue, regression):
    X, y, plans = regression
    request = spec()
    _, summary = tuning.tune(request, X, y, catalogue, TaskMetrics(), plans)
    for record in summary["trials"]:
        expected = []
        for train, valid in plans:
            fitted, _, _ = fit_artifact(dict(request, params=record["params"], search=None), X.iloc[train], y[train], catalogue)
            expected.append(np.sqrt(mean_squared_error(y[valid], fitted.predict_encoded(X.iloc[valid]))))
        assert record["fold_scores"] == pytest.approx(expected, abs=1e-12)


def test_explicit_max_direction_selects_the_requested_maximum(catalogue, regression):
    X, y, plans = regression
    request = spec()
    request["search"]["direction"] = "max"
    best, summary = tuning.tune(request, X, y, catalogue, TaskMetrics(), plans)
    winner = max(summary["trials"], key=lambda record: record["score"])
    assert summary["best_score"] == winner["score"] and best == winner["params"]


def test_classification_probability_search_uses_log_loss(catalogue):
    array, y = make_classification(n_samples=150, n_features=5, n_informative=4, n_redundant=0, random_state=83)
    X = pd.DataFrame(array, columns=[f"x{index}" for index in range(array.shape[1])])
    encoder = LabelEncoder().fit(["no", "yes"])
    plans = list(StratifiedKFold(3, shuffle=True, random_state=83).split(X, y))
    request = spec("grid", {"C": [.01, 1, 10]}, task="classification", algorithm_id="logistic_regression", params={})
    request["search"]["metric"] = "log_loss"
    _, summary = tuning.tune(request, X, y, catalogue, TaskMetrics(), plans, encoder)
    assert summary["direction"] == "min"
    first = summary["trials"][0]
    independent = []
    for train, valid in plans:
        artifact, _, _ = fit_artifact(dict(request, params=first["params"], search=None), X.iloc[train], y[train], catalogue, label_encoder=encoder)
        independent.append(log_loss(y[valid], artifact.predict_proba(X.iloc[valid]), labels=[0, 1]))
    assert first["fold_scores"] == pytest.approx(independent, abs=1e-12)


def test_fold_scalers_and_samplers_fit_only_supplied_training_rows(monkeypatch, catalogue):
    X = pd.DataFrame({"x": np.r_[np.arange(60.), np.arange(60.) + 10000]})
    y = np.r_[np.arange(60.), np.arange(60.) + 30000]
    plans = [(np.arange(60), np.arange(60, 120)), (np.arange(60, 120), np.arange(60))]
    observed = []
    original = tuning.fit_artifact
    def observe(request, fold_X, fold_y, *args, **kwargs):
        artifact, sampling, trace = original(request, fold_X, fold_y, *args, **kwargs)
        numeric = artifact.preprocessing.named_steps["columns"].named_transformers_["numeric0"]
        observed.append((tuple(fold_X.index), numeric.named_steps["scale"].mean_[0], sampling["before"], fold_y.copy()))
        return artifact, sampling, trace
    monkeypatch.setattr(tuning, "fit_artifact", observe)
    tuning.tune(spec(space={"alpha": [1]}), X, y, catalogue, TaskMetrics(), plans)
    assert len(observed) == 2
    assert observed[0][1] == np.mean(X.iloc[:60].x)
    assert observed[1][1] == np.mean(X.iloc[60:].x)
    assert [item[2] for item in observed] == [60, 60]
    np.testing.assert_array_equal(observed[0][3], y[:60])
    np.testing.assert_array_equal(observed[1][3], y[60:])


def test_weights_are_fold_local_and_estimators_do_not_nest_parallelism(monkeypatch, catalogue, regression):
    X, y, plans = regression
    weights = np.linspace(.1, 3, len(y))
    seen = []
    original = tuning.fit_artifact
    def observe(request, fold_X, fold_y, *args, **kwargs):
        np.testing.assert_array_equal(kwargs["weights"], weights[fold_X.index])
        seen.append(request["n_jobs"])
        return original(request, fold_X, fold_y, *args, **kwargs)
    monkeypatch.setattr(tuning, "fit_artifact", observe)
    best1, summary1 = tuning.tune(spec(n_jobs=1), X, y, catalogue, TaskMetrics(), plans, weights=weights)
    best2, summary2 = tuning.tune(spec(n_jobs=2), X, y, catalogue, TaskMetrics(), plans, weights=weights)
    assert seen and set(seen) == {1}
    assert best1 == best2
    assert [trial["score"] for trial in summary1["trials"]] == pytest.approx([trial["score"] for trial in summary2["trials"]], abs=1e-12)


def test_real_classification_oversampling_is_fresh_in_each_fold(monkeypatch, catalogue):
    random = np.random.default_rng(10)
    X = pd.DataFrame(random.normal(size=(126, 2)), columns=["x", "z"])
    y = (np.arange(126) % 7 == 0).astype(int)
    plans = list(StratifiedKFold(3, shuffle=True, random_state=15).split(X, y))
    encoder = LabelEncoder().fit(["ordinary", "rare"])
    request = spec(space={"max_depth": [2, 4]}, task="classification", algorithm_id="decision_tree_classifier", params={})
    request["preprocessing"]["resampling"] = {"method": "random_over"}
    request["search"]["metric"] = "f1_macro"
    seen = []
    original = tuning.fit_artifact
    def observe(request, fold_X, fold_y, *args, **kwargs):
        artifact, summary, trace = original(request, fold_X, fold_y, *args, **kwargs)
        seen.append((len(fold_X), summary))
        return artifact, summary, trace
    monkeypatch.setattr(tuning, "fit_artifact", observe)
    tuning.tune(request, X, y, catalogue, TaskMetrics(), plans, encoder)
    assert len(seen) == 6
    for before, summary in seen:
        assert summary["before"] == before == 84
        assert summary["after"] > before and summary["fitted_on"] == "train"


@pytest.mark.parametrize("method", ["random", "optuna_tpe", "optuna_random"])
def test_seeded_range_search_is_reproducible(method, catalogue, regression):
    X, y, plans = regression
    request = spec(method, {"alpha": {"type": "float", "low": .001, "high": 100, "log": True}})
    first = tuning.tune(request, X, y, catalogue, TaskMetrics(), plans)[1]
    second = tuning.tune(request, X, y, catalogue, TaskMetrics(), plans)[1]
    assert [trial["params"] for trial in first["trials"]] == [trial["params"] for trial in second["trials"]]
    assert [trial["score"] for trial in first["trials"]] == pytest.approx([trial["score"] for trial in second["trials"]], abs=1e-12)


def test_large_grid_is_rejected_before_parameter_grid_allocation(monkeypatch, regression):
    X, y, plans = regression
    class WideCatalogue:
        def descriptor(self, identifier):
            return {"params": [{"key": f"p{i}"} for i in range(12)]}
        def build(self, *args, **kwargs):
            return None
    monkeypatch.setattr(tuning, "ParameterGrid", lambda *args: pytest.fail("huge grid allocated"))
    request = spec(space={f"p{i}": list(range(100)) for i in range(12)})
    with pytest.raises(ValueError, match="1000000000000000000000000"):
        tuning.tune(request, X, y, WideCatalogue(), TaskMetrics(), plans)


def test_fit_budget_fails_before_any_training(monkeypatch, catalogue, regression):
    X, y, plans = regression
    request = spec(space={"alpha": list(range(1, 101))})
    request["search"]["trials"] = 100
    monkeypatch.setattr(tuning, "fit_artifact", lambda *args, **kwargs: pytest.fail("budget passed"))
    with pytest.raises(ValueError, match="600.*500"):
        tuning.tune(request, X, y, catalogue, TaskMetrics(), plans + plans)


@pytest.mark.parametrize("change", [{"search": {"method": "grid", "trials": 101, "metric": "rmse", "param_space": {"alpha": [1]}}},
    {"search": {"method": "random", "trials": 2, "param_space": {"alpha": {"type": "float", "low": 0, "high": 1, "log": True}}}},
    {"search": {"method": "grid", "trials": 3, "param_space": {"unknown": [1]}}},
    {"search": {"method": "grid", "trials": 3, "param_space": {"alpha": {"type": "float", "low": 1, "high": 2}}}}])
def test_invalid_search_spaces_fail_before_fit(change, monkeypatch, catalogue, regression):
    X, y, plans = regression
    monkeypatch.setattr(tuning, "fit_artifact", lambda *args, **kwargs: pytest.fail("invalid space fitted"))
    with pytest.raises(ValueError):
        tuning.tune(spec(**change), X, y, catalogue, TaskMetrics(), plans)


def test_group_plans_keep_whole_queries_and_halving_explicitly_rejects_groups(catalogue, regression):
    X, y, _ = regression
    groups = np.repeat(np.arange(21), 6)
    plans = list(GroupKFold(3).split(X, y, groups))
    best, summary = tuning.tune(spec(), X, y, catalogue, TaskMetrics(), plans, groups=groups)
    assert best and summary["n_splits"] == 3
    with pytest.raises(ValueError, match="группы"):
        tuning.tune(spec("halving_grid"), X, y, catalogue, TaskMetrics(), plans, groups=groups)
    bad = [(np.arange(30), np.arange(30, 60))]
    mixed_groups = np.zeros(len(X))
    with pytest.raises(ValueError, match="Группы"):
        tuning.tune(spec(), X, y, catalogue, TaskMetrics(), bad, groups=mixed_groups)


def test_temporal_tuning_accepts_supplied_past_future_plans_but_no_row_halving(catalogue, regression):
    X, y, _ = regression
    plans = [(np.arange(50), np.arange(50, 75)), (np.arange(75), np.arange(75, 100))]
    request = spec(task="forecasting")
    best, summary = tuning.tune(request, X, y, catalogue, TaskMetrics(), plans)
    assert best and summary["n_splits"] == 2
    with pytest.raises(ValueError, match="хронологию"):
        tuning.tune(spec("halving_grid", task="forecasting"), X, y, catalogue, TaskMetrics(), plans)


def test_undefined_probability_metric_fails_before_fit(catalogue, regression, monkeypatch):
    X, y, plans = regression
    request = spec(task="classification", algorithm_id="ridge_classifier", params={})
    request["search"]["metric"] = "log_loss"
    monkeypatch.setattr(tuning, "fit_artifact", lambda *args, **kwargs: pytest.fail("probability capability ignored"))
    with pytest.raises(ValueError, match="вероятности"):
        tuning.tune(request, X, y, catalogue, TaskMetrics(), plans)


def test_all_failed_candidates_provide_reason_instead_of_fake_winner(catalogue, regression):
    X, y, plans = regression
    request = spec(space={"n_neighbors": [110, 120]}, algorithm_id="knn_regressor", params={})
    with pytest.raises(ValueError, match="Ни одна проба.*n_neighbors"):
        tuning.tune(request, X, y, catalogue, TaskMetrics(), plans)


def test_optuna_cancellation_stops_after_first_real_trial(catalogue, regression):
    X, y, plans = regression
    flag = [False]
    observed = []
    def progress(event):
        if event.get("trial"):
            observed.append(event["trial"])
            flag[0] = True
    with pytest.raises(InterruptedError):
        tuning.tune(spec("optuna_tpe"), X, y, catalogue, TaskMetrics(), plans, progress=progress, cancelled=lambda: flag[0])
    assert len(observed) == 1 and observed[0]["status"] == "complete"


def test_immediate_cancel_does_not_touch_catalogue():
    class NoCatalogue:
        def descriptor(self, *args):
            pytest.fail("cancelled search inspected catalogue")
    with pytest.raises(InterruptedError):
        tuning.tune(spec(), pd.DataFrame({"x": [1]}), [1], NoCatalogue(), TaskMetrics(), [], cancelled=lambda: True)


def test_halving_small_grid_uses_actual_candidates_not_larger_budget(catalogue, regression):
    X, y, plans = regression
    request = spec("halving_grid", {"alpha": [1, 10]})
    request["search"]["trials"] = 100
    _, summary = tuning.tune(request, X, y, catalogue, TaskMetrics(), plans)
    assert summary["n_fits"] < 100


def test_optuna_summary_matches_actual_study_trials_and_sampler(monkeypatch, catalogue, regression):
    import optuna
    X, y, plans = regression
    studies = []
    create = optuna.create_study
    def observed_create(*args, **kwargs):
        study = create(*args, **kwargs)
        studies.append(study)
        return study
    monkeypatch.setattr(optuna, "create_study", observed_create)
    request = spec("optuna_tpe", {"alpha": {"type": "float", "low": .001, "high": 100, "log": True}})
    _, summary = tuning.tune(request, X, y, catalogue, TaskMetrics(), plans)
    assert len(studies) == 1
    study = studies[0]
    assert isinstance(study.sampler, optuna.samplers.TPESampler)
    assert len(study.trials) == len(summary["trials"]) == 4
    assert [trial.params["alpha"] for trial in study.trials] == [trial["params"]["alpha"] for trial in summary["trials"]]
    assert [trial.value for trial in study.trials] == [trial["score"] for trial in summary["trials"]]
    assert study.best_value == summary["best_score"]


def test_custom_metric_and_categorical_bool_options_are_preserved(catalogue, regression):
    X, y, plans = regression
    request = spec("grid", {"fit_intercept": [False, True], "solver": ["auto", "svd"]},
                   custom_metric="mean(abs(error))")
    request["search"]["metric"] = "custom"
    request["search"]["direction"] = "min"
    _, summary = tuning.tune(request, X, y, catalogue, TaskMetrics(), plans)
    assert len(summary["trials"]) == 4
    assert {trial["params"]["fit_intercept"] for trial in summary["trials"]} == {True, False}
    assert all(type(trial["params"]["fit_intercept"]) is bool for trial in summary["trials"])
    first = summary["trials"][0]
    train, valid = plans[0]
    artifact, _, _ = fit_artifact(dict(request, search=None, params=first["params"]), X.iloc[train], y[train], catalogue)
    assert first["fold_scores"][0] == pytest.approx(np.mean(np.abs(y[valid] - artifact.predict_encoded(X.iloc[valid]))))
