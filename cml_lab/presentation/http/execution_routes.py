"""Execution, model and experiment HTTP operations delegate to applications."""
from fastapi import APIRouter
from fastapi.responses import Response


def execution_routes(services):
    router=APIRouter(prefix="/api/cml",tags=["Experiments"])

    @router.get("/runs")
    def runs():return services.execution.list()

    @router.post("/runs")
    def start(payload:dict):return services.execution.start(services.resolve(payload)).public_dict()

    @router.get("/runs/{run_id}")
    def run(run_id:str):
        return services.execution.get(run_id)

    @router.delete("/runs/{run_id}")
    def delete_run(run_id:str):
        current=services.execution.get(run_id)
        return services.execution.cancel(run_id) if current["status"] in {"queued","running"} else services.models.delete(run_id)

    @router.post("/runs/{run_id}/reveal-test")
    def reveal(run_id:str):return services.execution.reveal_test(run_id)

    @router.post("/runs/{run_id}/save")
    def save(run_id:str,payload:dict):return services.archive.save(run_id,payload)

    @router.get("/runs/{run_id}/export-capabilities")
    def capabilities(run_id:str):return services.artifacts.capabilities(run_id)

    @router.get("/runs/{run_id}/export")
    def export(run_id:str,format:str="bundle"):
        artifact=services.artifacts.export(run_id,format)
        return Response(artifact.data,media_type=artifact.mimetype,headers={"Content-Disposition":f'attachment; filename="{artifact.filename}"'})

    @router.post("/predict")
    def predict(payload:dict):return services.artifacts.predict(payload.get("run_id"),payload.get("rows"))

    @router.post("/forecast")
    def forecast(payload:dict):return services.artifacts.forecast(payload.get("run_id"),payload.get("history"),payload.get("future"))

    @router.get("/models")
    def models():return services.models.list()

    @router.post("/models")
    def retain_model(payload:dict):return services.models.update(payload.get("run_id"),{key:payload[key] for key in ("name","description","expected_revision") if key in payload})

    @router.get("/models/{run_id}")
    def model(run_id:str):return services.models.get(run_id)

    @router.patch("/models/{run_id}")
    def update_model(run_id:str,payload:dict):return services.models.update(run_id,payload)

    @router.delete("/models/{run_id}")
    def delete_model(run_id:str):return services.models.delete(run_id)

    @router.get("/experiments")
    def experiments():return services.experiments.list()

    @router.post("/experiments")
    def create_experiment(payload:dict):return services.archive.save(payload.get("run_id"),payload)

    @router.get("/experiments/{experiment_id}")
    def experiment(experiment_id:str):return services.experiments.get(experiment_id)

    @router.patch("/experiments/{experiment_id}")
    def update_experiment(experiment_id:str,payload:dict):return services.experiments.update(experiment_id,payload)

    @router.delete("/experiments/{experiment_id}")
    def delete_experiment(experiment_id:str):return services.experiments.delete(experiment_id)

    return router
