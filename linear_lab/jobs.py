"""Изолированные расчеты, отмена процесса и сохранение экспериментов."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import queue
import threading
import time
import uuid
from pathlib import Path
from typing import Any


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def _train_worker(root: str, request: dict, channel: Any) -> None:
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    from .datasets import DataService
    from .models import ModelRegistry
    from .training import TrainingService
    try:
        from threadpoolctl import threadpool_limits
        data, registry = DataService(Path(root) / 'datasets'), ModelRegistry()
        if request.get('search'):
            from .search import SearchService
            service = SearchService(data, registry)
        else:
            service = TrainingService(data, registry)
        with threadpool_limits(limits=1):
            result = service.run(request, channel.put, lambda: False)
        channel.put({'type': 'result', 'result': result})
    except Exception as exc:
        channel.put({'type': 'error', 'error': str(exc) or type(exc).__name__})


class JobManager:
    """Один расчет занимает отдельный процесс, который можно остановить."""

    def __init__(self, root: Path, max_active: int = 2):
        self.root = root
        self.directory = root / 'jobs'
        self.directory.mkdir(parents=True, exist_ok=True)
        self.max_active = max_active
        self._jobs: dict[str, dict] = {}
        self._processes: dict[str, mp.Process] = {}
        self._lock = threading.RLock()
        self._context = mp.get_context('spawn')

    def start(self, request: dict) -> dict:
        with self._lock:
            running = sum(p.is_alive() for p in self._processes.values())
            if running >= self.max_active:
                raise ValueError('Уже идут два расчета. Дождитесь завершения или отмените один.')
            identifier = uuid.uuid4().hex
            job_directory = self.directory / identifier
            job_directory.mkdir()
            request = dict(request, artifact_path=str(job_directory / 'model.joblib'))
            write_json(job_directory / 'request.json', request)
            state = {'id': identifier, 'status': 'running', 'progress': 0, 'message': 'Подготовка данных', 'events': [], 'created': time.time()}
            write_json(job_directory / 'state.json', state)
            self._jobs[identifier] = state
            channel = self._context.Queue()
            process = self._context.Process(target=_train_worker, args=(str(self.root), request, channel), daemon=True)
            process.start()
            self._processes[identifier] = process
            threading.Thread(target=self._collect, args=(identifier, channel, process), daemon=True).start()
            return {'id': identifier}

    def _collect(self, identifier: str, channel: Any, process: mp.Process) -> None:
        directory = self.directory / identifier
        while True:
            try:
                event = channel.get(timeout=.25)
            except queue.Empty:
                if process.is_alive():
                    continue
                with self._lock:
                    state = self._jobs[identifier]
                    if state['status'] == 'running':
                        state.update(status='error', error='Расчет завершился без результата. Возможно, закончилась память.')
                break
            with self._lock:
                state = self._jobs[identifier]
                if state['status'] == 'cancelled':
                    break
                if event.get('type') == 'result':
                    write_json(directory / 'result.json', event['result'])
                    state.update(status='completed', progress=1, message='Расчет завершен')
                    break
                if event.get('type') == 'error':
                    state.update(status='error', error=event['error'], message='Ошибка расчета')
                    break
                state['events'].append(event)
                state['events'] = state['events'][-500:]
                state.update(progress=float(event.get('progress', state['progress'])), message=event.get('message', state['message']))
        process.join(timeout=2)
        with self._lock:
            write_json(directory / 'state.json', self._jobs[identifier])
        channel.close()

    def get(self, identifier: str) -> dict:
        if len(identifier) != 32 or not all(c in '0123456789abcdef' for c in identifier):
            raise FileNotFoundError('Расчет не найден.')
        with self._lock:
            state = self._jobs.get(identifier)
            directory = self.directory / identifier
            if state is None:
                path = directory / 'state.json'
                if not path.exists():
                    raise FileNotFoundError('Расчет не найден.')
                state = json.loads(path.read_text())
                if state['status'] == 'running':
                    state.update(status='interrupted', message='Расчет прерван перезапуском приложения.')
            response = {**state, 'events': list(state.get('events', []))}
            if state['status'] == 'completed':
                response['result'] = json.loads((directory / 'result.json').read_text())
            return response

    def cancel(self, identifier: str) -> dict:
        with self._lock:
            state = self.get(identifier)
            process = self._processes.get(identifier)
            if state['status'] == 'running' and process:
                self._jobs[identifier].update(status='cancelled', message='Расчет отменен')
                process.terminate()
                process.join(timeout=2)
                if process.is_alive():
                    process.kill()
                write_json(self.directory / identifier / 'state.json', self._jobs[identifier])
            return self.get(identifier)

    def reveal_test(self, identifier: str) -> dict:
        with self._lock:
            state = self.get(identifier)
            if state['status'] != 'completed':
                raise ValueError('Сначала завершите обучение.')
            state.pop('result', None)
            state['test_revealed'] = True
            self._jobs[identifier] = state
            write_json(self.directory / identifier / 'state.json', state)
            return self.get(identifier)

    def close(self) -> None:
        for identifier in list(self._processes):
            try:
                self.cancel(identifier)
            except FileNotFoundError:
                pass


class ExperimentStore:
    def __init__(self, root: Path):
        self.root = root / 'experiments'
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, name: str, request: dict, result: dict, job_id: str) -> dict:
        identifier = uuid.uuid4().hex
        experiment = {'id': identifier, 'name': name[:150] or result.get('model_name', 'Эксперимент'), 'created': time.time(), 'request': request, 'result': result, 'job_id': job_id}
        write_json(self.root / f'{identifier}.json', experiment)
        return experiment

    def get(self, identifier: str) -> dict:
        if len(identifier) != 32 or not all(c in '0123456789abcdef' for c in identifier):
            raise FileNotFoundError('Эксперимент не найден.')
        path = self.root / f'{identifier}.json'
        if not path.exists():
            raise FileNotFoundError('Эксперимент не найден.')
        return json.loads(path.read_text())

    def list(self) -> list[dict]:
        values = []
        for path in self.root.glob('*.json'):
            record = json.loads(path.read_text())
            values.append({key: record[key] for key in ('id', 'name', 'created', 'job_id')} | {'model': record['result'].get('model_name'), 'metrics': record['result'].get('metrics', {}).get('validation', {})})
        return sorted(values, key=lambda v: v['created'], reverse=True)

    def delete(self, identifier: str) -> None:
        self.get(identifier)
        (self.root / f'{identifier}.json').unlink()
