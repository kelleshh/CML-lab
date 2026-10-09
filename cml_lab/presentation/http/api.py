"""Local web application. Controllers contain no dataframe or estimator logic."""
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from cml_lab import __version__
from cml_lab.bootstrap import build_services
from cml_lab.shared.domain import ConflictError
from .data_routes import data_routes
from .recipe_routes import recipe_routes
from .execution_routes import execution_routes
from .pipeline_routes import pipeline_routes
from .project_exports import project_exports
from cml_lab.infrastructure.project_export import LocalProjectExporter
from cml_lab.infrastructure.analysis import DatasetAnalysis
from .analysis_routes import analysis_routes


def create_app(root=None,*,services=None):
    services=services or build_services(root)
    web=Path(__file__).resolve().parents[3]/"web"

    @asynccontextmanager
    async def lifespan(app):
        yield
        services.close()

    app=FastAPI(title="CML-lab",version=__version__,lifespan=lifespan)
    app.state.services=services

    @app.exception_handler(ValueError)
    async def invalid(request:Request,error:ValueError):
        return JSONResponse({"detail":str(error)},status_code=409 if isinstance(error,ConflictError) else 422)

    @app.exception_handler(FileNotFoundError)
    async def missing(request:Request,error:FileNotFoundError):return JSONResponse({"detail":str(error)},status_code=404)

    @app.middleware("http")
    async def local_request_guard(request:Request,call_next):
        origin=request.headers.get("origin")
        if request.method not in {"GET","HEAD","OPTIONS"} and origin and urlparse(origin).netloc!=request.headers.get("host"):
            return JSONResponse({"detail":"Откройте лабораторию и отправьте запрос из её интерфейса."},status_code=403)
        response=await call_next(request)
        response.headers["X-Content-Type-Options"]="nosniff"
        return response

    @app.get("/api/health")
    def health():return {"status":"ok","version":__version__,"name":"CML-lab"}

    @app.get("/api/cml/catalogue")
    def catalogue():return services.catalogue.get()

    @app.get("/api/cml/preprocessing/stages")
    def stages(task:str|None=None):return services.preparation.stages(task)

    @app.post("/api/cml/preprocessing/preview")
    def preview(payload:dict):return services.preparation.preview(services.resolve(payload))

    @app.get("/api/learning/lessons")
    def lessons(query:str="",task:str|None=None,family:str|None=None,level:str|None=None):
        return services.learning.list_lessons(query=query,task=task,family=family,level=level)

    @app.get("/api/learning/lessons/{lesson_id}")
    def lesson(lesson_id:str):
        try:return services.learning.get_lesson(lesson_id)
        except KeyError as error:raise FileNotFoundError(str(error)) from error

    @app.get("/api/learning/help")
    def help_catalogue():return services.learning.list_help()

    @app.get("/api/learning/help/{key:path}")
    def help_entry(key:str):
        try:return services.learning.get_help(key)
        except KeyError as error:raise FileNotFoundError(str(error)) from error

    app.include_router(data_routes(services.data))
    for kind,prefix in (("model","model-recipes"),("preprocessor","preprocessor-recipes"),("project","projects")):
        app.include_router(recipe_routes(services.recipes,kind,prefix))
    app.include_router(execution_routes(services))
    app.include_router(pipeline_routes(services))
    app.include_router(project_exports(services, LocalProjectExporter(services.data.gateway)))
    app.include_router(analysis_routes(DatasetAnalysis(services.data.gateway)))

    @app.get("/")
    def index():return FileResponse(web/"index.html")

    @app.get("/lesson")
    def standalone_lesson():return FileResponse(web/"lesson.html")

    app.mount("/assets",StaticFiles(directory=web),name="assets")
    app.mount("/",StaticFiles(directory=web),name="web")
    return app
