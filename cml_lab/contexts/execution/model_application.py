"""Model library and experiment snapshots share run identity, not mutable state."""
from cml_lab.shared.domain import ConflictError, ValidationError, identifier


class ModelLibrary:
    def __init__(self, execution):
        self.execution = execution

    @staticmethod
    def _model(run):
        return {**run, "run_id": run["id"], "task": run["spec"]["task"],
                "algorithm_id": run["spec"]["algorithm_id"], "model_name": (run.get("result") or {}).get("model_name", "")}

    def list(self):
        items = [self._model(run) for run in self.execution.list()["items"] if run["status"] == "completed"]
        return {"items": items, "total": len(items)}

    def get(self, run_id):
        run = self.execution.get(run_id)
        if run["status"] != "completed":
            raise ValidationError("Обученная модель появляется после завершенного расчета.")
        return self._model(run)

    def update(self, run_id, patch):
        self.get(run_id)
        return self._model(self.execution.update_metadata(run_id, patch))

    def delete(self, run_id):
        run_id = identifier(run_id)
        return self.execution.delete(run_id)


class ExperimentArchive:
    def __init__(self, execution, experiments):
        self.execution = execution
        self.experiments = experiments

    def save(self, run_id, payload):
        run = self.execution.get(run_id)
        if run["status"] != "completed":
            raise ConflictError("Сохраните эксперимент после завершения расчета.")
        return self.experiments.save(payload.get("name") or run["name"] or run["result"]["model_name"],
                                     run["spec"], run["result"], run["id"])
