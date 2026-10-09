"""Resolve recipe snapshots and validate task-specific experiment intent."""

from __future__ import annotations

from copy import deepcopy

from cml_lab.contexts.data.domain import Dataset
from cml_lab.contexts.data.ports import DataGateway
from cml_lab.contexts.recipes.domain import validate_config
from cml_lab.contexts.recipes.ports import RecipeRepository
from cml_lab.shared.domain import TaskKind, ValidationError, identifier, integer, json_object, text
from .domain import DatasetSelection, SplitPlan, TEMPORAL_TASKS, validate_temporal, validate_validation
from .ports import AlgorithmCataloguePort, ExperimentRepository


class ExperimentResolver:
    def __init__(self, data_gateway: DataGateway, recipe_repository: RecipeRepository,
                 algorithm_catalogue: AlgorithmCataloguePort):
        self.data_gateway = data_gateway
        self.recipe_repository = recipe_repository
        self.algorithm_catalogue = algorithm_catalogue

    def resolve(self, payload: dict) -> dict:
        request = json_object(payload, "Запрос расчета")
        if any(key in request for key in ("artifact_path", "artifact_dir", "artifact_root")):
            raise ValidationError("Путь сохранения модели выбирает сервер.")
        snapshots = {}
        project_id = request.get("project_recipe_id", request.get("project_id"))
        if project_id:
            project = self.recipe_repository.get(identifier(project_id), "project")
            snapshots["project"] = project.to_dict()
            request = dict(project.config, **request)
        if request.get("model_recipe_id"):
            model = self.recipe_repository.get(identifier(request["model_recipe_id"]), "model")
            snapshots["model"] = model.to_dict()
            request["algorithm_id"] = request.get("algorithm_id", model.config["algorithm_id"])
            request["params"] = dict(model.config.get("params", {}), **json_object(request.get("params", {}), "Параметры модели"))
        if request.get("preprocessor_recipe_id"):
            preparation = self.recipe_repository.get(identifier(request["preprocessor_recipe_id"]), "preprocessor")
            snapshots["preprocessor"] = preparation.to_dict()
            request["preprocessing"] = request.get("preprocessing", preparation.config)
        task = TaskKind.parse(request.get("task", "regression"))
        request["validation"] = validate_validation(request.get("validation", request.get("cv_config")), task)
        dataset = Dataset.from_metadata(self.data_gateway.describe(identifier(request.get("dataset_id"))))
        selection = DatasetSelection.resolve(dataset, task, request)
        algorithm_id = text(request.get("algorithm_id", request.get("model", "ridge")), "Алгоритм", required=True)
        algorithm = self.algorithm_catalogue.spec(algorithm_id)
        if task.value not in algorithm.get("tasks", []):
            raise ValidationError("Этот алгоритм не поддерживает выбранную задачу. Выберите алгоритм из ее каталога.")
        if not algorithm.get("available", True):
            raise ValidationError(algorithm.get("reason") or "Для этого алгоритма установите дополнительную библиотеку.")
        params = json_object(request.get("params", {}), "Параметры модели")
        allowed_params = {param["key"] for param in algorithm.get("params", [])}
        if set(params) - allowed_params:
            raise ValidationError("Параметры модели содержат неизвестные поля: " + ", ".join(sorted(set(params) - allowed_params)))
        preprocessing = validate_config("preprocessor", request.get("preprocessing", {"steps": []}))
        if task in TEMPORAL_TASKS and preprocessing.get("resampling", {}).get("method", "none") != "none":
            raise ValidationError("Пересэмплирование выключено для временных и панельных рядов: оно меняет порядок наблюдений.")
        seed = integer(request.get("seed", 42), "Случайное зерно", 0, 2**32 - 1)
        n_jobs = integer(request.get("n_jobs", 1), "Параллельных вычислений", 1, 4)
        search = request.get("search")
        if search is not None:
            search = json_object(search, "Подбор параметров")
            integer(search.get("trials", 20), "Число вариантов поиска", 1, 100)
        metrics = request.get("metrics", "all")
        if not (isinstance(metrics, str) and metrics or isinstance(metrics, list) and metrics and all(isinstance(name, str) and name for name in metrics)):
            raise ValidationError("Метрики задаются именем, «all» или непустым списком имен.")
        spec = {**selection.to_dict(), "algorithm_id": algorithm_id, "params": params, "preprocessing": preprocessing,
                "split": SplitPlan.from_dict(request.get("split"), task).to_dict(), "validation": request["validation"],
                "search": search, "metrics": deepcopy(metrics), "seed": seed, "n_jobs": n_jobs,
                "temporal": validate_temporal(request.get("temporal")) if task in TEMPORAL_TASKS else {},
                "recipe_snapshots": snapshots}
        if "plot_features" in request:
            plot_features = request["plot_features"]
            if not isinstance(plot_features, list) or not 1 <= len(plot_features) <= 2 or any(
                    not isinstance(name, str) or name not in spec["features"] for name in plot_features
            ) or len(set(plot_features)) != len(plot_features):
                raise ValidationError("Для среза модели выберите один или два уникальных признака из эксперимента.")
            spec["plot_features"] = list(plot_features)
        for field in ("custom_metric", "metric_params", "epochs", "regularization_path", "learning_curve", "permutation_importance"):
            if field in request:
                if field in {"regularization_path", "learning_curve", "permutation_importance"} and not isinstance(request[field], bool):
                    raise ValidationError("Переключатели дополнительных исследований должны быть true или false.")
                spec[field] = deepcopy(request[field])
        return spec


class ExperimentService:
    def __init__(self, repository: ExperimentRepository):
        self.repository = repository

    def list(self) -> dict:
        items = self.repository.list()
        return {"items": items, "total": len(items)}

    def get(self, experiment_id: str) -> dict:
        return self.repository.get(identifier(experiment_id))

    def save(self, name: str, spec: dict, result: dict, run_id: str) -> dict:
        return self.repository.save(text(name, "Название эксперимента", required=True), json_object(spec),
                                    json_object(result, "Результат", maximum_bytes=100_000_000), identifier(run_id))

    def update(self, experiment_id: str, patch: dict) -> dict:
        patch = json_object(patch, "Описание эксперимента")
        if set(patch) - {"name", "description", "expected_revision"}:
            raise ValidationError("Сохраненный эксперимент хранит неизменный расчет; менять можно название и описание.")
        return self.repository.update(identifier(experiment_id), patch)

    def delete(self, experiment_id: str) -> dict:
        self.repository.delete(identifier(experiment_id))
        return {"id": experiment_id, "deleted": True}
