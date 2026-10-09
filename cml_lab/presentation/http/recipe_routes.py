"""One CRUD contract for three kinds of reusable configuration."""
from fastapi import APIRouter


def recipe_routes(recipes,kind,prefix):
    router=APIRouter(prefix=f"/api/cml/{prefix}",tags=["Recipes"])

    @router.get("")
    def list_recipes(query:str=""):return recipes.list(kind,query)

    @router.post("")
    def create(payload:dict):return recipes.create(kind,payload)

    @router.get("/{recipe_id}")
    def get(recipe_id:str,revision:int|None=None):return recipes.get(kind,recipe_id,revision)

    @router.patch("/{recipe_id}")
    def update(recipe_id:str,payload:dict):return recipes.update(kind,recipe_id,payload)

    @router.delete("/{recipe_id}")
    def delete(recipe_id:str):return recipes.delete(kind,recipe_id)

    return router
