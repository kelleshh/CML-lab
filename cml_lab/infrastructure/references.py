"""Reference checks across context snapshots; no database paths cross inward."""
class DatasetReferences:
    def __init__(self, runs, experiments):
        self.runs = runs
        self.experiments = experiments

    def dataset_in_use(self, dataset_id):
        if any(run.spec.get("dataset_id") == dataset_id for run in self.runs.list() if run.status in {"queued", "running", "completed"}):
            return True
        return any(self.experiments.get(item["id"]).get("spec", {}).get("dataset_id") == dataset_id for item in self.experiments.list())


class RunReferences:
    def __init__(self, experiments):
        self.experiments = experiments

    def run_in_use(self, run_id):
        return any(item.get("run_id") == run_id for item in self.experiments.list())
