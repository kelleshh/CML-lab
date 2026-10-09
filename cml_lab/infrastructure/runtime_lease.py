"""One active scheduler owns a data directory for its entire lifecycle."""

from pathlib import Path
import sqlite3
import threading

from cml_lab.shared.domain import ConflictError


class RuntimeLease:
    def __init__(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._connection = None
        connection = sqlite3.connect(path, timeout=0, isolation_level=None, check_same_thread=False)
        try:
            connection.execute("BEGIN EXCLUSIVE")
        except sqlite3.OperationalError as error:
            connection.close()
            if error.sqlite_errorcode in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}:
                raise ConflictError(
                    "Папка данных уже используется другим сервисом CML-lab. "
                    "Остановите его или выберите другую папку через CML_LAB_DATA."
                ) from error
            raise
        except BaseException:
            connection.close()
            raise
        self._connection = connection

    def close(self) -> None:
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None
