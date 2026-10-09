"""Source export routes preserve project and run snapshots."""

from fastapi import APIRouter
from fastapi.responses import Response

from cml_lab.shared.domain import json_object


def project_exports(services, exporter):
    router = APIRouter(prefix="/api/cml", tags=["Project exports"])

    def download(spec, format, name, origin):
        artifact = exporter.export(spec, format, name=name, origin=origin)
        return Response(artifact.data, media_type=artifact.mimetype,
                        headers={"Content-Disposition": f'attachment; filename="{artifact.filename}"'})

    @router.post("/project-exports")
    def export_draft(payload: dict, format: str = "py"):
        request = json_object(payload, "Экспорт проекта")
        spec = services.resolve(request.get("spec", request))
        return download(spec, format, request.get("name", "cml-project"), {"type": "draft"})

    @router.get("/projects/{project_id}/export")
    def export_project(project_id: str, format: str = "py", revision: int | None = None):
        project = services.recipes.get("project", project_id, revision)
        spec = services.resolve(project["config"])
        return download(spec, format, project["name"],
                        {"type": "project", "id": project_id, "revision": project["revision"]})

    @router.get("/runs/{run_id}/project-export")
    def export_run(run_id: str, format: str = "py"):
        run = services.execution.get(run_id)
        return download(run["spec"], format, run.get("name") or "cml-project",
                        {"type": "run", "id": run_id, "revision": run["revision"],
                         "test_revealed": run.get("test_revealed", False)})

    @router.get("/experiments/{experiment_id}/project-export")
    def export_experiment(experiment_id: str, format: str = "ipynb"):
        experiment = services.experiments.get(experiment_id)
        return download(experiment["spec"], format, experiment["name"],
                        {"type": "experiment", "id": experiment_id, "revision": experiment["revision"]})

    return router
