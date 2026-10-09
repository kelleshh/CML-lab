"""Serialize short dataset deletion/start sections across threads and processes.

This database is separate from repository storage: holding its write lock must
not block the run repository's own transaction. SQLite releases the lock when a
process exits. The guard protects application operations, not external file edits.
"""

from contextlib import contextmanager
from pathlib import Path
import sqlite3
from typing import Iterator


class SqliteMutationGuard:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def hold(self) -> Iterator[None]:
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()
