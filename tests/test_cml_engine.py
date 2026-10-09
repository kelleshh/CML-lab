"""Task results, held-out isolation and complete-pipeline inference contracts."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.metrics import accuracy_score, mean_squared_error

from cml_lab.contexts.experiments.application import ExperimentResolver
from cml_lab.infrastructure.datasets import LegacyDataGateway
from cml_lab.infrastructure.storage import SqliteRecipeRepository
from cml_lab.infrastructure.ml.catalogue import AlgorithmCatalogue
from cml_lab.infrastructure.ml.engine import ExperimentEngine
from cml_lab.infrastructure.ml.fitting import fit_artifact
from cml_lab.infrastructure.ml.splitting import holdout
from cml_lab.infrastructure.ml.metrics import TaskMetrics


@pytest.fixture
def laboratory(tmp_path):
    data=LegacyDataGateway(tmp_path/"datasets")
    catalogue=AlgorithmCatalogue()
    return data,catalogue,ExperimentResolver(data,SqliteRecipeRepository(tmp_path),catalogue)


@pytest.mark.parametrize("task,algorithm",[
    ("regression","ridge"),("classification","logistic_regression"),
    ("clustering","kmeans"),("ranking","xgboost_ranker"),
    ("forecasting","ridge"),("panel","ridge"),
    ("anomaly","isolation_forest"),("reduction","pca"),
])
def test_eight_task_results_and_complete_saved_pipeline(laboratory,tmp_path,task,algorithm):
    data,catalogue,resolver=laboratory
    dataset=data.load({"kind":"synthetic","name":"linear" if task=="regression" else task,"params":{"n_samples":180,"seed":42}})
    spec=resolver.resolve({"task":task,"dataset_id":dataset["id"],"algorithm_id":algorithm,
                           "roles":dataset.get("roles",{}),"validation":{"strategy":"none"}})
    path=tmp_path/f"{task}.joblib"
    result=ExperimentEngine(data,catalogue).run(dict(spec,artifact_path=str(path)))
    assert result["task"]==task and result["algorithm_id"]==algorithm
    assert "artifact_path" not in result["effective_spec"]
    assert path.is_file()
    artifact=joblib.load(path)
    X,y,context=ExperimentEngine(data,catalogue).prepare_data(spec)
    if task in {"clustering","anomaly","reduction"}:
        assert set(result["evaluations"])=={"train"}
        assert len(result["diagnostics"]["embedding"]["coordinates"])==len(X)
        assert result["evaluations"]["train"]["rows"]
        if task=="reduction":
            assert artifact.reduce(X.iloc[:3]).shape[0]==3
            assert spec["features"]==["x1","x2"] # No numeric feature silently becomes a reference label.
        else:assert len(artifact.predict(X.iloc[:3]))==3
    else:
        assert set(result["evaluations"])=={"train","validation","test"}
        assert all(part["n_rows"]>=2 for part in result["evaluations"].values())
        assert len(artifact.predict(X.iloc[:3]))==3
        if task=="classification":
            validation=result["evaluations"]["validation"]
            assert set(artifact.predict(X.iloc[:10])).issubset({"класс 0","класс 1"})
            assert validation["metrics"]["accuracy"]>=.7
            assert sum(map(sum,validation["classification"]["confusion"]))==validation["n_rows"]
        if task=="ranking":
            assert result["evaluations"]["validation"]["metrics"]["ndcg"]>.75
            querysets=[{row["query"] for row in part["rows"]} for part in result["evaluations"].values()]
            assert not querysets[0]&querysets[1] and not querysets[0]&querysets[2] and not querysets[1]&querysets[2]
        if task in {"forecasting","panel"}:
            train=result["evaluations"]["train"]["rows"]
            validation=result["evaluations"]["validation"]["rows"]
            assert max(row["time"] for row in train)<min(row["origin"] for row in validation)


def test_regression_metrics_equal_independent_holdout_predictions(laboratory,tmp_path):
    data,catalogue,resolver=laboratory
    ds=data.load({"kind":"synthetic","name":"linear","params":{"n_samples":120,"noise":.2}})
    spec=resolver.resolve({"dataset_id":ds["id"],"algorithm_id":"ridge","metrics":["rmse","mae"],"validation":{"strategy":"kfold","folds":3}})
    path=tmp_path/"model.joblib"
    result=ExperimentEngine(data,catalogue).run(dict(spec,artifact_path=str(path)))
    artifact=joblib.load(path)
    X,y,_=ExperimentEngine(data,catalogue).prepare_data(spec)
    validation=holdout(spec,len(X),y)["validation"]
    expected=np.sqrt(mean_squared_error(y[validation],artifact.predict(X.iloc[validation])))
    assert result["evaluations"]["validation"]["metrics"]["rmse"]==pytest.approx(expected)
    assert result["diagnostics"]["cv"]["n_splits"]==3


def test_heldout_extremes_do_not_change_fitted_statistics_or_coefficients(laboratory,tmp_path):
    data,catalogue,resolver=laboratory
    ds=data.load({"kind":"synthetic","name":"linear","params":{"n_samples":100}})
    first=resolver.resolve({"dataset_id":ds["id"],"algorithm_id":"ridge","validation":{"strategy":"none"}})
    engine=ExperimentEngine(data,catalogue)
    X,y,_=engine.prepare_data(first)
    parts=holdout(first,len(X),y)
    frame=data.frame(ds["id"])
    modified=frame.copy()
    modified.iloc[np.r_[parts["validation"],parts["test"]],modified.columns.get_indexer(first["features"])]=1e12
    new=data.load({"kind":"custom","name":"Changed heldout","rows":modified.to_dict("records"),"target":first["target"]})
    second=resolver.resolve(dict(first,dataset_id=new["id"]))
    paths=[tmp_path/"a.joblib",tmp_path/"b.joblib"]
    for spec,path in zip((first,second),paths):engine.run(dict(spec,artifact_path=str(path)))
    a,b=map(joblib.load,paths)
    np.testing.assert_allclose(a.estimator.coef_,b.estimator.coef_,rtol=0,atol=0)
    np.testing.assert_allclose(a.predict(X),b.predict(X),rtol=0,atol=0)


def test_text_pipeline_retains_sparse_matrix_and_decoded_classes(laboratory):
    _,catalogue,_=laboratory
    text=["red apple sweet fruit" if i%2 else "blue ocean water coast" for i in range(80)]
    X=pd.DataFrame({"review":text});y=np.arange(80)%2
    from sklearn.preprocessing import LabelEncoder
    encoder=LabelEncoder().fit(["вода","еда"])
    spec={"task":"classification","algorithm_id":"logistic_regression","seed":42,"preprocessing":{"steps":[{"adapter_id":"text.tfidf","columns":["review"],"params":{}}]}}
    artifact,_,_=fit_artifact(spec,X,y,catalogue,label_encoder=encoder)
    assert sparse.issparse(artifact.transform(X.iloc[:2]))
    assert accuracy_score(y,artifact.predict_encoded(X))==1
    assert artifact.predict(pd.DataFrame({"review":["apple sweet","water ocean"]})).tolist()==["еда","вода"]


def test_group_target_encoding_is_rejected_before_fit(laboratory):
    _,catalogue,_=laboratory
    X=pd.DataFrame({"category":["a","b"]*30})
    spec={"task":"regression","algorithm_id":"ridge","preprocessing":{"steps":[{"adapter_id":"categorical.target","columns":["category"],"params":{}}]}}
    with pytest.raises(ValueError,match="групп|cross|врем"):
        fit_artifact(spec,X,np.arange(60),catalogue,groups=np.repeat(np.arange(6),10))


def test_real_sgd_epochs_have_independently_recomputed_loss(laboratory):
    _,catalogue,_=laboratory
    X=pd.DataFrame({"x":np.linspace(-1,1,80)})
    y=2*X["x"].to_numpy()+1
    events=[]
    artifact,_,trace=fit_artifact({"task":"regression","algorithm_id":"sgd","epochs":12,"params":{"learning_rate":"constant","eta0":.02}},X,y,catalogue,progress=events.append)
    assert len(trace)==12 and len(events)==12
    assert trace[-1]["train_loss"]==pytest.approx(mean_squared_error(y,artifact.predict(X)))
    assert trace[-1]["train_loss"]<trace[0]["train_loss"]


def test_probability_metrics_missing_for_margin_only_estimator(laboratory):
    _,catalogue,_=laboratory
    from sklearn.preprocessing import LabelEncoder
    X=pd.DataFrame({"x":np.r_[np.linspace(-3,-1,20),np.linspace(1,3,20)]});y=np.repeat([0,1],20)
    spec={"task":"classification","algorithm_id":"linear_svc","metrics":"all"}
    artifact,_,_=fit_artifact(spec,X,y,catalogue,label_encoder=LabelEncoder().fit(["no","yes"]))
    values,details=TaskMetrics().evaluate("classification",y,artifact.predict_encoded(X),n_classes=2)
    assert values["accuracy"]==1 and values["roc_auc"] is None
    assert "вероятност" in details["roc_auc"]["reason"]


def test_negative_bayes_inputs_get_actionable_error(laboratory):
    _,catalogue,_=laboratory
    with pytest.raises(ValueError,match="неотрицатель"):
        fit_artifact({"task":"classification","algorithm_id":"multinomial_nb"},pd.DataFrame({"x":[-1,1]*20}),np.repeat([0,1],20),catalogue)


def test_group_fold_probabilities_keep_outer_class_columns(laboratory):
    """A fold missing the middle class must not shift class-2 probabilities."""
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import log_loss
    _, catalogue, _ = laboratory
    X = pd.DataFrame({"x": [-3., -2., -1., 0., 1., 2., 3., 4., 5.]})
    y = np.repeat([0, 1, 2], 3)
    train = np.array([0, 1, 2, 6, 7, 8])
    encoder = LabelEncoder().fit(["a", "b", "c"])
    spec = {"task": "classification", "algorithm_id": "gaussian_nb", "metrics": ["log_loss"]}
    artifact, _, _ = fit_artifact(spec, X.iloc[train], y[train], catalogue,
                                  label_encoder=encoder, groups=np.repeat(["a", "c"], 3))
    probabilities = artifact.predict_proba(X)
    assert probabilities.shape == (9, 3)
    np.testing.assert_allclose(probabilities[:, 1], 0.)
    np.testing.assert_allclose(probabilities[:, [0, 2]], artifact.estimator.predict_proba(artifact.transform(X)))
    values, details = TaskMetrics().evaluate("classification", y, artifact.predict_encoded(X),
                                             probabilities=probabilities, n_classes=3, selection=["log_loss"])
    assert values["log_loss"] == pytest.approx(log_loss(y, probabilities, labels=[0, 1, 2]))
    assert details["log_loss"]["reason"] is None


@pytest.mark.parametrize("algorithm,params", [
    ("stacking_regressor", {}),
    ("voting_regressor", {"components": [{"algorithm_id": "ridge"}, {"algorithm_id": "stacking_regressor"}]}),
])
@pytest.mark.parametrize("task,grouped,ordered", [("regression", True, False), ("forecasting", False, False), ("panel", False, False), ("regression", False, True)])
def test_stacking_cannot_silently_use_iid_internal_cv_for_dependent_rows(laboratory, algorithm, params, task, grouped, ordered):
    _, catalogue, _ = laboratory
    X = pd.DataFrame({"x": np.arange(40.)})
    spec = {"task": task, "algorithm_id": algorithm, "params": params, "split": {"shuffle": not ordered}}
    with pytest.raises(ValueError, match="не поддерживает|Стекинг.*независимые"):
        fit_artifact(spec, X, 2 * X.x.to_numpy(), catalogue,
                     groups=np.repeat(np.arange(8), 5) if grouped else None)


def test_custom_dense_ensemble_accepts_sparse_text_and_retains_inference_contract(laboratory, tmp_path):
    from sklearn.preprocessing import LabelEncoder
    _, catalogue, _ = laboratory
    X = pd.DataFrame({"text": ["red apple sweet" if i % 2 else "blue ocean coast" for i in range(40)]})
    y = np.arange(40) % 2
    spec = {"task": "classification", "algorithm_id": "voting_classifier",
            "params": {"components": [{"algorithm_id": "gaussian_nb"}, {"algorithm_id": "logistic_regression"}]},
            "preprocessing": {"steps": [{"adapter_id": "text.tfidf", "columns": ["text"], "params": {}}]}}
    artifact, _, _ = fit_artifact(spec, X, y, catalogue, label_encoder=LabelEncoder().fit(["water", "fruit"]))
    assert artifact.input_capabilities["requires_dense"] is True
    assert accuracy_score(y, artifact.predict_encoded(X)) == 1
    path = tmp_path / "text-ensemble.joblib"
    joblib.dump(artifact, path)
    np.testing.assert_array_equal(joblib.load(path).predict(X), artifact.predict(X))


def test_sparse_reduction_artifact_returns_numeric_coordinates(laboratory):
    from cml_lab.infrastructure.ml.artifacts import FittedArtifact
    from cml_lab.infrastructure.ml.preparation import build_preprocessor
    _, catalogue, _ = laboratory
    X = pd.DataFrame({"text": ["red apple sweet" if i % 2 else "blue ocean coast" for i in range(40)]})
    preprocessing = build_preprocessor(X, {"steps": [{"adapter_id": "text.tfidf", "columns": ["text"], "params": {}}]}, task="reduction")
    Xt = preprocessing.fit_transform(X)
    model = catalogue.build("reduction", "sparse_random_projection").fit(Xt)
    artifact = FittedArtifact("reduction", preprocessing, model, ["text"])
    actual = artifact.reduce(X.iloc[:3])
    assert actual.shape == (3, 2)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, model.transform(Xt[:3]).toarray())


def test_anomaly_fraction_is_diagnostic_without_a_quality_direction():
    metrics = TaskMetrics()
    descriptor = next(item for item in metrics.catalogue("anomaly") if item["id"] == "anomaly_fraction")
    assert descriptor["direction"] is None
    assert descriptor["optimizable"] is False and descriptor["diagnostic"] is True
    values, details = metrics.evaluate("anomaly", None, [-1, 1, 1, 1])
    assert values["anomaly_fraction"] == .25
    assert details["anomaly_fraction"]["direction"] is None


def test_sparse_reducer_reconstruction_metric_matches_manual_error(laboratory):
    _, catalogue, _ = laboratory
    X = sparse.csr_matrix(np.random.default_rng(17).uniform(.1, 3, (40, 5)))
    model = catalogue.build("reduction", "truncated_svd").fit(X)
    coordinates = model.transform(X)
    expected = np.mean((X.toarray() - model.inverse_transform(coordinates)) ** 2)
    values, details = TaskMetrics().evaluate("reduction", None, coordinates, X=X, model=model)
    assert values["reconstruction_mse"] == pytest.approx(expected)
    assert details["reconstruction_mse"]["reason"] is None
