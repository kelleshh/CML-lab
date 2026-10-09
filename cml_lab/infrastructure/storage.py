"""Transactional SQLite repositories for recipes and execution snapshots.

Connections live for one operation. WAL permits simultaneous readers; IMMEDIATE
write transactions plus revision comparisons reject lost updates across threads
and worker processes. Tables and columns are infrastructure details, never domain
entities or paths supplied by a client.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
from typing import Iterator

from cml_lab.contexts.execution.domain import Run, TERMINAL_STATUSES, public_result
from cml_lab.contexts.experiments.domain import Experiment
from cml_lab.contexts.recipes.domain import Recipe, RECIPE_KINDS
from cml_lab.shared.domain import ConflictError, NotFoundError, ValidationError, identifier, integer, json_object, text, utc_now


_SCHEMA = """
CREATE TABLE IF NOT EXISTS recipe_heads (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    revision INTEGER NOT NULL CHECK (revision >= 1),
    deleted INTEGER NOT NULL DEFAULT 0 CHECK (deleted IN (0, 1)),
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS recipe_revisions (
    id TEXT NOT NULL,
    revision INTEGER NOT NULL CHECK (revision >= 1),
    kind TEXT NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY (id, revision),
    FOREIGN KEY (id) REFERENCES recipe_heads(id)
);
CREATE INDEX IF NOT EXISTS recipe_heads_kind ON recipe_heads(kind, deleted, updated_at);
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    revision INTEGER NOT NULL CHECK (revision >= 1),
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS runs_created ON runs(created_at);
CREATE TABLE IF NOT EXISTS experiment_snapshots (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    spec TEXT NOT NULL,
    result TEXT NOT NULL,
    test_revealed INTEGER NOT NULL CHECK (test_revealed IN (0, 1)),
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS experiment_heads (
    id TEXT PRIMARY KEY,
    revision INTEGER NOT NULL CHECK (revision >= 1),
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    deleted INTEGER NOT NULL DEFAULT 0 CHECK (deleted IN (0, 1)),
    FOREIGN KEY (id) REFERENCES experiment_snapshots(id)
);
CREATE TABLE IF NOT EXISTS experiment_revisions (
    id TEXT NOT NULL,
    revision INTEGER NOT NULL CHECK (revision >= 1),
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    deleted INTEGER NOT NULL CHECK (deleted IN (0, 1)),
    PRIMARY KEY(id, revision),
    FOREIGN KEY (id) REFERENCES experiment_heads(id)
);
"""


def _kind(value: str | None) -> str | None:
    if value is not None and (not isinstance(value, str) or value not in RECIPE_KINDS):
        raise ValidationError("Выберите модель, препроцессор или проект.")
    return value


def _encode(value: dict) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


class _SqliteRepository:
    def __init__(self, rootpath: str | Path):
        root = Path(rootpath)
        self.path = root if root.suffix in {".sqlite", ".sqlite3", ".db"} else root / "cml.sqlite3"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(_SCHEMA)

    @contextmanager
    def _connection(self, *, write: bool = False) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        try:
            if write:
                connection.execute("BEGIN IMMEDIATE")
            yield connection
            if write:
                connection.commit()
        except BaseException:
            if write:
                connection.rollback()
            raise
        finally:
            connection.close()


class SqliteRecipeRepository(_SqliteRepository):
    """Store editable heads and every historical revision, including deletion."""

    def list(self, kind: str, query: str = "") -> list[Recipe]:
        _kind(kind)
        needle = text(query, "Поиск", maximum=500).casefold()
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT r.payload FROM recipe_heads h JOIN recipe_revisions r "
                "ON r.id=h.id AND r.revision=h.revision "
                "WHERE h.kind=? AND h.deleted=0 ORDER BY h.updated_at DESC,h.id DESC", (kind,)
            ).fetchall()
        recipes = [Recipe.from_dict(json.loads(row["payload"])) for row in rows]
        # Python casefold handles Russian text consistently; SQLite LIKE/NOCASE
        # only fold ASCII in the standard cross-platform build.
        return [recipe for recipe in recipes if not needle or needle in f"{recipe.name} {recipe.description}".casefold()]

    def get(self, identifier: str, kind: str | None = None, revision: int | None = None) -> Recipe:
        recipe_id = _identifier(identifier)
        _kind(kind)
        if revision is not None:
            integer(revision, "Ревизия", 1, 2**31 - 1)
        with self._connection() as connection:
            if revision is None:
                row = connection.execute(
                    "SELECT r.payload FROM recipe_heads h JOIN recipe_revisions r "
                    "ON r.id=h.id AND r.revision=h.revision WHERE h.id=? AND h.deleted=0", (recipe_id,)
                ).fetchone()
            else:
                row = connection.execute("SELECT payload FROM recipe_revisions WHERE id=? AND revision=?", (recipe_id, revision)).fetchone()
        if row is None:
            raise NotFoundError("Рецепт или его ревизия не найдены.")
        recipe = Recipe.from_dict(json.loads(row["payload"]))
        if kind is not None and recipe.kind != kind:
            raise NotFoundError("Рецепт выбранного типа не найден.")
        return recipe

    def create(self, recipe: Recipe) -> Recipe:
        if not isinstance(recipe, Recipe) or recipe.revision != 1 or recipe.deleted:
            raise ValidationError("Новый рецепт должен начинаться с первой активной ревизии.")
        try:
            with self._connection(write=True) as connection:
                connection.execute(
                    "INSERT INTO recipe_heads(id,kind,revision,deleted,name,description,updated_at) VALUES(?,?,?,0,?,?,?)",
                    (recipe.id, recipe.kind, recipe.revision, recipe.name, recipe.description, recipe.updated_at)
                )
                connection.execute("INSERT INTO recipe_revisions(id,revision,kind,payload) VALUES(?,?,?,?)", (recipe.id, recipe.revision, recipe.kind, _encode(recipe.to_dict())))
        except sqlite3.IntegrityError as exc:
            raise ConflictError("Рецепт с таким идентификатором уже существует.") from exc
        return recipe

    def update(self, recipe: Recipe, expected_revision: int) -> Recipe:
        integer(expected_revision, "Ожидаемая ревизия", 1, 2**31 - 1)
        if not isinstance(recipe, Recipe) or recipe.deleted or recipe.revision != expected_revision + 1:
            raise ValidationError("Сохранение рецепта должно создать следующую активную ревизию.")
        with self._connection(write=True) as connection:
            row = connection.execute(
                "SELECT h.kind,h.revision,h.deleted,r.payload FROM recipe_heads h JOIN recipe_revisions r "
                "ON r.id=h.id AND r.revision=h.revision WHERE h.id=?", (recipe.id,)
            ).fetchone()
            if row is None:
                raise NotFoundError("Рецепт не найден.")
            if row["deleted"] or row["revision"] != expected_revision:
                raise ConflictError("Рецепт изменился или удален. Обновите форму перед сохранением.")
            current = json.loads(row["payload"])
            if row["kind"] != recipe.kind or current["created_at"] != recipe.created_at:
                raise ValidationError("Тип и время создания рецепта изменить нельзя.")
            connection.execute("INSERT INTO recipe_revisions(id,revision,kind,payload) VALUES(?,?,?,?)", (recipe.id, recipe.revision, recipe.kind, _encode(recipe.to_dict())))
            connection.execute(
                "UPDATE recipe_heads SET revision=?,name=?,description=?,updated_at=? WHERE id=? AND revision=?",
                (recipe.revision, recipe.name, recipe.description, recipe.updated_at, recipe.id, expected_revision)
            )
        return recipe

    def delete(self, identifier: str, kind: str | None = None) -> None:
        recipe_id = _identifier(identifier)
        _kind(kind)
        with self._connection(write=True) as connection:
            row = connection.execute(
                "SELECT h.kind,h.deleted,r.payload FROM recipe_heads h JOIN recipe_revisions r "
                "ON r.id=h.id AND r.revision=h.revision WHERE h.id=?", (recipe_id,)
            ).fetchone()
            if row is None or row["deleted"] or (kind is not None and row["kind"] != kind):
                raise NotFoundError("Рецепт не найден.")
            payload = json.loads(row["payload"])
            payload.update(revision=payload["revision"] + 1, updated_at=utc_now(), deleted=True)
            tombstone = Recipe.from_dict(payload)
            connection.execute("INSERT INTO recipe_revisions(id,revision,kind,payload) VALUES(?,?,?,?)", (recipe_id, tombstone.revision, tombstone.kind, _encode(tombstone.to_dict())))
            connection.execute("UPDATE recipe_heads SET revision=?,deleted=1,updated_at=? WHERE id=?", (tombstone.revision, tombstone.updated_at, recipe_id))


class SqliteRunRepository(_SqliteRepository):
    """Persist full runs; domain projection owns hidden-test visibility."""

    def create(self, run: Run) -> Run:
        if not isinstance(run, Run) or run.revision != 1:
            raise ValidationError("Новый запуск должен начинаться с первой ревизии.")
        try:
            with self._connection(write=True) as connection:
                connection.execute(
                    "INSERT INTO runs(id,revision,status,created_at,updated_at,payload) VALUES(?,?,?,?,?,?)",
                    (run.id, run.revision, run.status, run.created_at, run.updated_at, _encode(run.to_dict()))
                )
        except sqlite3.IntegrityError as exc:
            raise ConflictError("Запуск с таким идентификатором уже существует.") from exc
        return run

    def get(self, identifier: str) -> Run:
        run_id = _identifier(identifier)
        with self._connection() as connection:
            row = connection.execute("SELECT payload FROM runs WHERE id=?", (run_id,)).fetchone()
        if row is None:
            raise NotFoundError("Запуск не найден.")
        return Run.from_dict(json.loads(row["payload"]))

    def list(self) -> list[Run]:
        with self._connection() as connection:
            rows = connection.execute("SELECT payload FROM runs ORDER BY created_at DESC,id DESC").fetchall()
        return [Run.from_dict(json.loads(row["payload"])) for row in rows]

    def update(self, run: Run, expected_status: str | None = None) -> Run:
        if not isinstance(run, Run) or run.revision < 2:
            raise ValidationError("Обновление запуска должно создать следующую ревизию.")
        with self._connection(write=True) as connection:
            row = connection.execute("SELECT revision,status,payload FROM runs WHERE id=?", (run.id,)).fetchone()
            if row is None:
                raise NotFoundError("Запуск не найден.")
            if row["revision"] != run.revision - 1 or (expected_status is not None and row["status"] != expected_status):
                raise ConflictError("Состояние запуска изменилось. Прочитайте актуальный запуск.")
            current = Run.from_dict(json.loads(row["payload"]))
            if current.spec != run.spec or current.created_at != run.created_at:
                raise ValidationError("Исходный запрос и время создания запуска изменить нельзя.")
            # Validate the transition even if a caller constructs a Run directly.
            current.transition(run.status, progress=run.progress, message=run.message, error=run.error,
                               result=run.result, test_revealed=run.test_revealed, name=run.name, description=run.description)
            connection.execute(
                "UPDATE runs SET revision=?,status=?,updated_at=?,payload=? WHERE id=? AND revision=?",
                (run.revision, run.status, run.updated_at, _encode(run.to_dict()), run.id, current.revision)
            )
        return run

    def delete(self, identifier: str) -> None:
        run_id = _identifier(identifier)
        with self._connection(write=True) as connection:
            row = connection.execute("SELECT status FROM runs WHERE id=?", (run_id,)).fetchone()
            if row is None:
                raise NotFoundError("Запуск не найден.")
            if row["status"] not in TERMINAL_STATUSES:
                raise ConflictError("Сначала остановите выполняющийся запуск.")
            reference = connection.execute(
                "SELECT 1 FROM experiment_snapshots s JOIN experiment_heads h ON h.id=s.id "
                "WHERE s.run_id=? AND h.deleted=0 LIMIT 1", (run_id,)
            ).fetchone()
            if reference is not None:
                raise ConflictError("Модель используется сохраненным экспериментом. Сначала удалите его запись.")
            connection.execute("DELETE FROM runs WHERE id=?", (run_id,))


class SqliteExperimentRepository(_SqliteRepository):
    """Immutable calculation snapshots with separately versioned descriptions.

    The result's explicit ``test_hidden=False`` records that the user opened the
    test before saving. Missing flags keep test hidden. Editing a name never
    rewrites spec/result or changes their visibility policy.
    """

    @staticmethod
    def _dto(row: sqlite3.Row) -> dict:
        revealed = bool(row["test_revealed"])
        return {"id": row["id"], "run_id": row["run_id"], "name": row["name"],
                "description": row["description"], "revision": row["revision"],
                "created_at": row["created_at"], "updated_at": row["updated_at"],
                "spec": json.loads(row["spec"]), "result": public_result(json.loads(row["result"]), revealed),
                "test_revealed": revealed}

    _SELECT = "SELECT h.id,h.revision,h.name,h.description,h.updated_at,s.run_id,s.spec,s.result,s.test_revealed,s.created_at FROM experiment_heads h JOIN experiment_snapshots s ON s.id=h.id "

    def save(self, name: str, spec: dict, result: dict, run_id: str) -> dict:
        experiment = Experiment.create(name, spec, result, run_id)
        experiment_id = experiment.id
        now = experiment.created_at
        revealed = experiment.result.get("test_hidden") is False
        with self._connection(write=True) as connection:
            run = connection.execute("SELECT status FROM runs WHERE id=?", (experiment.run_id,)).fetchone()
            if run is None:
                raise NotFoundError("Исходный запуск удален. Сохранить эксперимент невозможно.")
            if run["status"] != "completed":
                raise ConflictError("Сохраните эксперимент после успешного завершения расчета.")
            connection.execute("INSERT INTO experiment_snapshots(id,run_id,spec,result,test_revealed,created_at) VALUES(?,?,?,?,?,?)", (experiment_id, experiment.run_id, _encode(experiment.spec), _encode(experiment.result), int(revealed), now))
            connection.execute("INSERT INTO experiment_heads(id,revision,name,description,updated_at,deleted) VALUES(?,1,?,'',?,0)", (experiment_id, experiment.name, now))
            connection.execute("INSERT INTO experiment_revisions(id,revision,name,description,updated_at,deleted) VALUES(?,1,?,'',?,0)", (experiment_id, experiment.name, now))
        return self.get(experiment_id)

    def get(self, identifier: str) -> dict:
        experiment_id = _identifier(identifier)
        with self._connection() as connection:
            row = connection.execute(self._SELECT + "WHERE h.id=? AND h.deleted=0", (experiment_id,)).fetchone()
        if row is None:
            raise NotFoundError("Сохраненный эксперимент не найден.")
        return self._dto(row)

    def list(self) -> list[dict]:
        with self._connection() as connection:
            rows = connection.execute(self._SELECT + "WHERE h.deleted=0 ORDER BY s.created_at DESC,h.id DESC").fetchall()
        return [self._dto(row) for row in rows]

    def update(self, identifier: str, patch: dict) -> dict:
        experiment_id = _identifier(identifier)
        patch = json_object(patch, "Описание эксперимента")
        if not patch or set(patch) - {"name", "description", "expected_revision"}:
            raise ValidationError("Менять можно название и описание сохраненного эксперимента.")
        expected = integer(patch.get("expected_revision"), "Ожидаемая ревизия", 1, 2**31 - 1)
        with self._connection(write=True) as connection:
            current = connection.execute("SELECT * FROM experiment_heads WHERE id=? AND deleted=0", (experiment_id,)).fetchone()
            if current is None:
                raise NotFoundError("Сохраненный эксперимент не найден.")
            if current["revision"] != expected:
                raise ConflictError("Описание эксперимента изменилось. Обновите форму перед сохранением.")
            snapshot = connection.execute(self._SELECT + "WHERE h.id=?", (experiment_id,)).fetchone()
            revised = Experiment.from_dict(self._dto(snapshot)).revise_metadata(patch)
            connection.execute("INSERT INTO experiment_revisions(id,revision,name,description,updated_at,deleted) VALUES(?,?,?,?,?,0)", (experiment_id, revised.revision, revised.name, revised.description, revised.updated_at))
            connection.execute("UPDATE experiment_heads SET revision=?,name=?,description=?,updated_at=? WHERE id=?", (revised.revision, revised.name, revised.description, revised.updated_at, experiment_id))
            updated = connection.execute(self._SELECT + "WHERE h.id=?", (experiment_id,)).fetchone()
            result = self._dto(updated)
        return result

    def delete(self, identifier: str) -> None:
        experiment_id = _identifier(identifier)
        with self._connection(write=True) as connection:
            current = connection.execute("SELECT * FROM experiment_heads WHERE id=? AND deleted=0", (experiment_id,)).fetchone()
            if current is None:
                raise NotFoundError("Сохраненный эксперимент не найден.")
            now = utc_now()
            revision = current["revision"] + 1
            connection.execute("INSERT INTO experiment_revisions(id,revision,name,description,updated_at,deleted) VALUES(?,?,?,?,?,1)", (experiment_id, revision, current["name"], current["description"], now))
            connection.execute("UPDATE experiment_heads SET revision=?,updated_at=?,deleted=1 WHERE id=?", (revision, now, experiment_id))


# Method argument names follow the port (`identifier`), while the shared validator
# stays explicit instead of being accidentally shadowed by those arguments.
_identifier = identifier
