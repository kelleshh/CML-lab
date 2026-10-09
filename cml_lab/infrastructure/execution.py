"""Bounded spawn workers; domain transitions remain in the run repository."""
from __future__ import annotations

from dataclasses import dataclass, field
import multiprocessing as mp
from pathlib import Path
import queue
import threading
import time

from cml_lab.contexts.execution.domain import TERMINAL_STATUSES
from cml_lab.shared.domain import ConflictError, NotFoundError
from .runtime_lease import RuntimeLease


def _calculate(root: str, spec: dict, run_id: str, messages, cancelled):
    from threadpoolctl import threadpool_limits
    from .datasets import LegacyDataGateway
    from .ml.catalogue import AlgorithmCatalogue
    from .ml.engine import ExperimentEngine

    def progress(event):
        try:
            messages.put_nowait({"kind": "event", "event": event})
        except queue.Full:
            pass

    try:
        engine = ExperimentEngine(LegacyDataGateway(Path(root) / "datasets"), AlgorithmCatalogue())
        owned_spec = dict(spec, artifact_path=str(Path(root) / "artifacts" / f"{run_id}.joblib"))
        with threadpool_limits(limits=1):
            result = engine.run(owned_spec, progress, cancelled.is_set)
        messages.put({"kind": "completed", "result": result})
    except InterruptedError:
        messages.put({"kind": "cancelled"})
    except Exception as error:
        messages.put({"kind": "error", "error": f"{type(error).__name__}: {error}"})


@dataclass
class _Worker:
    process: object
    messages: object
    cancellation: object
    events: list[dict] = field(default_factory=list)


class ProcessScheduler:
    def __init__(self, root: Path, repository, *, maximum_workers=2):
        self.root = Path(root)
        self.repository = repository
        self.maximum_workers = maximum_workers
        self.context = mp.get_context("spawn")
        self._workers: dict[str, _Worker] = {}
        self._events: dict[str, list[dict]] = {}
        self._lock = threading.RLock()
        self._closed = False
        self._runtime_lease = RuntimeLease(self.root / "runtime.sqlite3")
        try:
            (self.root / "artifacts").mkdir(parents=True, exist_ok=True)
            for run in self.repository.list():
                if run.status not in TERMINAL_STATUSES:
                    self.repository.update(run.transition("interrupted", message="Сервис перезапущен; запустите опыт заново."), expected_status=run.status)
        except BaseException:
            self._runtime_lease.close()
            raise

    def start(self, run):
        with self._lock:
            if self._closed:
                raise ConflictError("Сервис завершает работу.")
            if sum(worker.process.is_alive() for worker in self._workers.values()) >= self.maximum_workers:
                raise ConflictError("Уже выполняются два опыта. Дождитесь завершения или остановите один.")
            messages = self.context.Queue(maxsize=512)
            cancellation = self.context.Event()
            process = self.context.Process(target=_calculate, args=(str(self.root), run.spec, run.id, messages, cancellation), daemon=True)
            worker = _Worker(process, messages, cancellation)
            self._workers[run.id] = worker
            try:
                process.start()
            except Exception:
                self._workers.pop(run.id, None)
                messages.close()
                raise
            threading.Thread(target=self._collect, args=(run.id, worker), daemon=True, name=f"cml-run-{run.id[:8]}").start()

    def _change(self, run_id, status=None, **values):
        for _ in range(5):
            try:
                current = self.repository.get(run_id)
                if current.status in TERMINAL_STATUSES:
                    return current
                if "progress" in values:
                    values["progress"] = max(current.progress, min(.99, float(values["progress"])))
                changed = current.transition(status or "running", **values)
                return self.repository.update(changed, expected_status=current.status)
            except ConflictError:
                continue
            except NotFoundError:
                return None
        return None

    def _collect(self, run_id, worker):
        self._change(run_id, "running", message="Запущен отдельный процесс обучения")
        finished = False
        dead_since = None
        try:
            while not finished:
                try:
                    message = worker.messages.get(timeout=.15)
                except queue.Empty:
                    if not worker.process.is_alive():
                        dead_since = dead_since or time.monotonic()
                        if time.monotonic() - dead_since > .5:
                            self._change(run_id, "error", error=f"Процесс завершился без результата (код {worker.process.exitcode}).", message="Обучение прервано")
                            break
                    continue
                kind = message["kind"]
                if kind == "event":
                    event = message["event"]
                    with self._lock:
                        worker.events.append(event)
                        worker.events[:] = worker.events[-1000:]
                    self._change(run_id, progress=event.get("progress", 0), message=event.get("message", "Обучение"))
                elif kind == "completed":
                    self._change(run_id, "completed", result=message["result"], message="Обучение завершено")
                    finished = True
                elif kind == "cancelled":
                    self._change(run_id, "cancelled", message="Обучение остановлено")
                    finished = True
                elif kind == "error":
                    self._change(run_id, "error", error=message["error"], message="Проверьте данные и настройки")
                    finished = True
        finally:
            worker.process.join(timeout=2)
            if worker.process.is_alive():
                worker.process.terminate()
                worker.process.join(timeout=2)
            with self._lock:
                self._events[run_id] = list(worker.events)
                self._workers.pop(run_id, None)
                # Retain recent traces, not an unbounded process-local history.
                while len(self._events) > 100:
                    self._events.pop(next(iter(self._events)))
            worker.messages.close()
            try:
                if self.repository.get(run_id).status != "completed":
                    self.remove_artifact(run_id)
            except NotFoundError:
                self.remove_artifact(run_id)

    def events(self, run_id):
        with self._lock:
            worker = self._workers.get(run_id)
            return list(worker.events if worker else self._events.get(run_id, []))

    def cancel(self, run_id):
        with self._lock:
            worker = self._workers.get(run_id)
            if worker:
                worker.cancellation.set()
                if worker.process.is_alive():
                    worker.process.terminate()
                    worker.process.join(timeout=2)

    def remove_artifact(self, run_id):
        for suffix in (".joblib", ".tmp"):
            (self.root / "artifacts" / f"{run_id}{suffix}").unlink(missing_ok=True)

    def close(self):
        with self._lock:
            self._closed = True
            identifiers = list(self._workers)
        for run_id in identifiers:
            self._change(run_id, "interrupted", message="Сервис закрыт во время обучения")
            self.cancel(run_id)
        self._runtime_lease.close()
