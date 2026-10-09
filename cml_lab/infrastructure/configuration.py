"""Validate executable adapter configuration before scheduling a worker."""
from .ml.metrics import REGRESSION_TASKS


class ConfigurationPolicy:
    def __init__(self, catalogue, preparation, metrics, resolver):
        self.catalogue, self.preparation, self.metrics, self.resolver = catalogue, preparation, metrics, resolver

    def recipe(self, kind, config):
        if kind == "model":
            descriptor=self.catalogue.descriptor(config["algorithm_id"])
            self.catalogue.build(descriptor["tasks"][0],config["algorithm_id"],config.get("params"))
        elif kind == "preprocessor":
            self.preparation.validate_config(config)
        elif config.get("dataset_id"):
            self.spec(self.resolver.resolve(config))

    def spec(self, spec):
        self.catalogue.build(spec["task"],spec["algorithm_id"],spec["params"],spec["seed"],spec["n_jobs"])
        self.preparation.validate_config(spec["preprocessing"],spec["task"])
        selection=spec.get("metrics","all")
        selected=[selection] if isinstance(selection,str) else selection
        allowed={metric["id"] for metric in self.metrics.catalogue(spec["task"])}
        if "all" not in selected and set(selected)-allowed-({"custom"} if spec["task"] in REGRESSION_TASKS else set()):
            raise ValueError("Выберите метрики, относящиеся к этой задаче.")
        if spec.get("search") and spec["task"] in {"clustering","anomaly","reduction"}:
            raise ValueError("Подбор без учителя требует отдельного внешнего критерия; отключите supervised поиск.")
        return spec
