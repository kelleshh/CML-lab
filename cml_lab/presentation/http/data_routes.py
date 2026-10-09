"""HTTP translation for dataset use cases."""
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import Response


def data_routes(data):
    router=APIRouter(prefix="/api/datasets",tags=["Datasets"])

    @router.get("/library")
    def library(query:str="",task:str|None=None,source:str|None=None,offset:int=0,limit:int=200):
        return data.list_library(query=query,task=task,source=source,offset=offset,limit=limit)

    @router.post("/load")
    def load(payload:dict):return data.load(payload)

    @router.post("/upload")
    async def upload(file:UploadFile=File(...)):
        return data.import_bytes(await file.read(25*1024*1024+1),file.filename or "data.csv")

    @router.get("/{dataset_id}")
    def describe(dataset_id:str):return data.describe(dataset_id)

    @router.get("/{dataset_id}/rows")
    def rows(dataset_id:str,offset:int=0,limit:int=100):return data.rows(dataset_id,offset,limit)

    @router.patch("/{dataset_id}/rows")
    def update_rows(dataset_id:str,payload:dict):return data.update_rows(dataset_id,payload)

    @router.patch("/{dataset_id}/metadata")
    def metadata(dataset_id:str,payload:dict):return data.update_metadata(dataset_id,payload)

    @router.delete("/{dataset_id}")
    def delete(dataset_id:str):return data.delete(dataset_id)

    @router.get("/{dataset_id}/explore")
    def explore(dataset_id:str,x:str|None=None,y:str|None=None,z:str|None=None,color:str|None=None,target:str|None=None,sample_size:int=2000,seed:int=42):
        return data.explore(dataset_id,x=x,y=y,z=z,color=color,target=target,sample_size=sample_size,seed=seed)

    @router.get("/{dataset_id}/export")
    def export(dataset_id:str):
        return Response(data.export_csv(dataset_id),media_type="text/csv",headers={"Content-Disposition":'attachment; filename="dataset.csv"'})

    return router
