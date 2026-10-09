"""The sole composition root: infrastructure satisfies application ports."""
from dataclasses import dataclass
import os
from pathlib import Path

from .contexts.data.application import DataApplication
from .contexts.data.preparation_application import PreparationApplication
from .contexts.recipes.application import RecipeService
from .contexts.experiments.application import ExperimentResolver, ExperimentService
from .contexts.experiments.catalogue_application import CatalogueApplication
from .contexts.execution.application import ExecutionService
from .contexts.execution.model_application import ModelLibrary, ExperimentArchive
from .contexts.learning.application import LearningService
from .infrastructure.storage import SqliteRecipeRepository, SqliteRunRepository, SqliteExperimentRepository
from .infrastructure.coordination import SqliteMutationGuard
from .infrastructure.datasets import LegacyDataGateway
from .infrastructure.execution import ProcessScheduler
from .infrastructure.references import DatasetReferences, RunReferences
from .infrastructure.configuration import ConfigurationPolicy
from .infrastructure.ml.catalogue import AlgorithmCatalogue
from .infrastructure.ml.preparation_schema import PreparationCatalogue
from .infrastructure.ml.metrics import TaskMetrics
from .infrastructure.learning import InMemoryLearningRepository


@dataclass
class Services:
    data: object
    recipes: object
    resolver: object
    execution: object
    experiments: object
    archive: object
    models: object
    artifacts: object
    preparation: object
    learning: object
    catalogue: object
    policy: object

    def resolve(self, payload):
        return self.policy.spec(self.resolver.resolve(payload))

    def close(self):
        self.execution.close()


def build_services(root=None):
    from .infrastructure.preparation_preview import PreparationPreviewGateway
    from .infrastructure.model_artifacts import LocalArtifactGateway
    from .contexts.execution.artifact_application import ArtifactApplication

    root=Path(root or os.environ.get("CML_LAB_DATA") or os.environ.get("LINEAR_LAB_DATA") or Path.home()/".cml-lab").expanduser().resolve()
    root.mkdir(parents=True,exist_ok=True)
    recipes_repository=SqliteRecipeRepository(root)
    runs_repository=SqliteRunRepository(root)
    experiments_repository=SqliteExperimentRepository(root)
    mutation_guard=SqliteMutationGuard(root/"coordination.sqlite3")
    gateway=LegacyDataGateway(root/"datasets")
    data=DataApplication(gateway,DatasetReferences(runs_repository,experiments_repository),mutation_guard)
    algorithms=AlgorithmCatalogue()
    preparation_catalogue=PreparationCatalogue()
    metrics=TaskMetrics()
    from .infrastructure.ml.pipelines import pipeline_catalogue
    preparation=PreparationApplication(PreparationPreviewGateway(gateway,algorithms))
    learning=LearningService(InMemoryLearningRepository(algorithms=algorithms.catalogue(),
        stages=preparation_catalogue.stages()+preparation_catalogue.samplers(),metrics=metrics.catalogue(),
        pipeline_classes=pipeline_catalogue()['classes']))
    resolver=ExperimentResolver(gateway,recipes_repository,algorithms)
    policy=ConfigurationPolicy(algorithms,preparation,metrics,resolver)
    scheduler=ProcessScheduler(root,runs_repository)
    execution=ExecutionService(runs_repository,scheduler,RunReferences(experiments_repository),scheduler.remove_artifact,
        mutation_guard=mutation_guard,validate_input=lambda spec: gateway.describe(spec["dataset_id"]))
    experiments=ExperimentService(experiments_repository)
    return Services(data,RecipeService(recipes_repository,policy.recipe),resolver,execution,experiments,
        ExperimentArchive(execution,experiments),ModelLibrary(execution),
        ArtifactApplication(runs_repository,LocalArtifactGateway(root,algorithms,gateway)),preparation,learning,
        CatalogueApplication(algorithms,data,preparation_catalogue,metrics,learning),policy)
