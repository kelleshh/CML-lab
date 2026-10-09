"""Read-only exploratory analysis endpoints."""

from fastapi import APIRouter


def analysis_routes(gateway):
    router = APIRouter(prefix="/api/cml", tags=["Dataset analysis"])

    @router.get("/analysis/kinds")
    def kinds():
        return gateway.catalogue()

    @router.post("/analysis")
    def analyze(payload: dict):
        return gateway.analyze(payload)

    return router
