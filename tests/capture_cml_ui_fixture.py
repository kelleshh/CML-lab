"""Capture generated, non-user data from the real CML numerical adapters."""
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cml_lab.contexts.experiments.catalogue_application import CatalogueApplication
from cml_lab.contexts.execution.domain import public_result
from cml_lab.contexts.learning.application import LearningService
from cml_lab.infrastructure.datasets import LegacyDataGateway
from cml_lab.infrastructure.model_artifacts import LocalArtifactGateway
from cml_lab.infrastructure.learning import InMemoryLearningRepository
from cml_lab.infrastructure.ml.catalogue import AlgorithmCatalogue
from cml_lab.infrastructure.ml.engine import ExperimentEngine
from cml_lab.infrastructure.ml.metrics import TaskMetrics
from cml_lab.infrastructure.ml.preparation_schema import PreparationCatalogue
from cml_lab.infrastructure.preparation_preview import PreparationPreviewGateway


def capture():
    with TemporaryDirectory(prefix="cml-ui-fixture-") as directory:
        data = LegacyDataGateway(Path(directory))
        algorithms = AlgorithmCatalogue()
        preparation = PreparationCatalogue()
        metrics = TaskMetrics()
        learning = LearningService(InMemoryLearningRepository(algorithms=algorithms.catalogue(), stages=preparation.stages() + preparation.samplers(), metrics=metrics.catalogue()))
        catalogue = CatalogueApplication(algorithms, data, preparation, metrics, learning).get()
        fixture = {"catalogue": catalogue, "lessons": learning.list_lessons(), "datasets": {}, "results": {}, "previews": {}, "explorations": {}, "rows": {}, "capabilities": {}}
        choices = {"regression": "ridge", "classification": "decision_tree_classifier", "clustering": "kmeans", "ranking": "xgboost_ranker", "forecasting": "ridge", "panel": "ridge", "anomaly": "isolation_forest", "reduction": "pca"}
        engine = ExperimentEngine(data, algorithms)
        artifacts = LocalArtifactGateway(Path(directory), algorithms, data)
        preview = PreparationPreviewGateway(data, algorithms)
        for task, algorithm in choices.items():
            metadata = data.load({"kind": "synthetic", "name": "linear" if task == "regression" else task, "params": {"n_samples": 80 if task != "panel" else 150, "n_features": 3, "seed": 42}})
            fixture["datasets"][task] = metadata
            fixture["rows"][metadata["id"]] = data.rows(metadata["id"], limit=100)
            excluded = [metadata.get("task_target"), *metadata.get("excluded_features", []), *metadata.get("roles", {}).values()]
            features = [column["name"] for column in metadata["columns"] if column["name"] not in excluded]
            spec = {"task": task, "dataset_id": metadata["id"], "algorithm_id": algorithm, "params": {"n_estimators": 10} if task in {"ranking", "anomaly"} else {}, "target": metadata.get("task_target") if task not in {"clustering", "anomaly", "reduction"} else None, "features": features,
                    "roles": metadata.get("roles", {}), "temporal": {"lags": [1, 2], "rolling_windows": [3], "horizon": 1}, "preprocessing": {"steps": [], "resampling": {"method": "none"}}, "split": {"train": .6, "validation": .2, "test": .2, "shuffle": task not in {"forecasting", "panel"}}, "validation": {"strategy": "none"}, "metrics": None, "seed": 42, "n_jobs": 1}
            run_id = uuid4().hex
            result = engine.run({**spec, "artifact_path": str(Path(directory) / "artifacts" / f"{run_id}.joblib")})
            fixture["capabilities"][task] = {"formats": artifacts.capabilities(run_id, {"spec": spec, "result": public_result(result), "test_revealed": False})}
            fixture["results"][task] = {"hidden": public_result(result), "revealed": public_result(result, True)}
            fixture["previews"][task] = preview.preview(spec)
            numeric = [column["name"] for column in metadata["columns"] if column["numeric"]]
            fixture["explorations"][metadata["id"]] = data.explore(metadata["id"], x=numeric[0], y=numeric[1], color=metadata.get("task_target"), sample_size=80)
        sgd = {**fixture["results"]["regression"]["revealed"]["effective_spec"], "algorithm_id": "sgd", "epochs": 6}
        fixture["sgd"] = public_result(engine.run(sgd))
        searched = {**fixture["results"]["classification"]["revealed"]["effective_spec"], "validation": {"strategy": "stratified_kfold", "folds": 3}, "search": {"method": "grid", "trials": 2, "metric": "accuracy", "direction": "max", "param_space": {"max_depth": [2, 4]}}}
        fixture["searched"] = public_result(engine.run(searched))
        diagnostic_spec = {**fixture["results"]["regression"]["revealed"]["effective_spec"], "regularization_path": True, "learning_curve": True, "permutation_importance": True, "metrics": ["mse"]}
        fixture["diagnostic_regression"] = public_result(engine.run(diagnostic_spec))
        return fixture


if __name__ == "__main__":
    destination = Path(__file__).parent / "fixtures" / "cml-ui.json"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(capture(), ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(destination)
