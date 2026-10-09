"""Recipe CRUD resolves revisions without depending on a database or ML library."""

from .domain import Recipe, RECIPE_KINDS
from .ports import RecipeRepository, RecipeValidator
from cml_lab.shared.domain import ValidationError, identifier, integer, text


class RecipeService:
    def __init__(self, repository: RecipeRepository, validator: RecipeValidator | None = None):
        self.repository = repository
        self.validator = validator

    @staticmethod
    def _kind(kind: str) -> str:
        if kind not in RECIPE_KINDS:
            raise ValidationError("Неизвестный вид рецепта.")
        return kind

    def list(self, kind: str, query: str = "") -> dict:
        items = self.repository.list(self._kind(kind), text(query, "Поиск", maximum=500))
        return {"items": [item.to_dict() for item in items], "total": len(items)}

    def get(self, kind: str, recipe_id: str, revision: int | None = None) -> dict:
        if revision is not None:
            integer(revision, "Ревизия", 1, 2**31 - 1)
        return self.repository.get(identifier(recipe_id), self._kind(kind), revision).to_dict()

    def create(self, kind: str, payload: dict) -> dict:
        recipe = Recipe.create(self._kind(kind), payload)
        if self.validator is not None:
            self.validator(recipe.kind, recipe.config)
        return self.repository.create(recipe).to_dict()

    def update(self, kind: str, recipe_id: str, payload: dict) -> dict:
        current = self.repository.get(identifier(recipe_id), self._kind(kind))
        updated = current.revise(payload)
        if self.validator is not None:
            self.validator(updated.kind, updated.config)
        return self.repository.update(updated, current.revision).to_dict()

    def delete(self, kind: str, recipe_id: str) -> dict:
        self.repository.delete(identifier(recipe_id), self._kind(kind))
        return {"id": recipe_id, "deleted": True}
