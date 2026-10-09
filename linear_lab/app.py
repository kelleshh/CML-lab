"""HTTP API и локальный интерфейс лаборатории."""

from __future__ import annotations

import csv
import copy
import io
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .datasets import DataService
from .jobs import ExperimentStore, JobManager
from .models import ModelRegistry
from .teaching import LESSONS, GLOSSARY, METRIC_HELP


def public_job(state: dict) -> dict:
    if state.get('test_revealed'):
        return state
    if 'result' not in state:
        state = copy.deepcopy(state)
        for event in state.get('events', []):
            event.get('frame', {}).pop('predicted', None)
        return state
    state = copy.deepcopy(state)
    result = state['result']
    for section in ('metrics', 'metric_details'):
        if section in result:
            result[section]['test'] = {}
    retained = []
    original_count = 0
    for section in ('predictions', 'plot_data'):
        data = result.get(section, {})
        splits = data.get('split', [])
        indices = [i for i, split in enumerate(splits) if split != 'test']
        if section == 'predictions':
            retained = indices
            original_count = len(splits)
        for key, values in list(data.items()):
            if isinstance(values, list) and len(values) == len(splits) and key != 'feature_names':
                data[key] = [values[i] for i in indices]
    for frame in result.get('trace', []):
        if isinstance(frame.get('predicted'), list) and len(frame['predicted']) == original_count:
            frame['predicted'] = [frame['predicted'][i] for i in retained]
    for event in state.get('events', []):
        frame = event.get('frame', {})
        if isinstance(frame.get('predicted'), list) and len(frame['predicted']) == original_count:
            frame['predicted'] = [frame['predicted'][i] for i in retained]
    result['test_hidden'] = True
    return state


def create_app(root: Path | None = None) -> FastAPI:
    root = Path(root or os.environ.get('LINEAR_LAB_DATA', Path.home() / '.linear-lab')).resolve()
    root.mkdir(parents=True, exist_ok=True)
    datasets = DataService(root / 'datasets')
    registry = ModelRegistry()
    jobs = JobManager(root)
    experiments = ExperimentStore(root)

    @asynccontextmanager
    async def lifespan(app):
        yield
        jobs.close()

    app = FastAPI(title='Линейная лаборатория', version='1.0.0', lifespan=lifespan)
    app.state.jobs = jobs
    app.state.datasets = datasets

    @app.exception_handler(ValueError)
    async def invalid(request: Request, exc: ValueError):
        return JSONResponse({'detail': str(exc)}, status_code=422)

    @app.exception_handler(FileNotFoundError)
    async def missing(request: Request, exc: FileNotFoundError):
        return JSONResponse({'detail': str(exc)}, status_code=404)

    @app.middleware('http')
    async def local_request_guard(request: Request, call_next):
        # Локальный API не принимает запросы изменения данных с чужих сайтов.
        origin = request.headers.get('origin')
        if request.method not in {'GET', 'HEAD', 'OPTIONS'} and origin:
            from urllib.parse import urlparse
            if urlparse(origin).netloc != request.headers.get('host'):
                return JSONResponse({'detail': 'Запрос должен идти из интерфейса лаборатории.'}, status_code=403)
        response = await call_next(request)
        response.headers['X-Content-Type-Options'] = 'nosniff'
        return response

    @app.get('/api/health')
    def health():
        return {'status': 'ok'}

    @app.get('/api/catalogue')
    def catalogue():
        from .metrics import metric_catalogue
        metrics = metric_catalogue()
        for metric in metrics:
            metric['help'] = METRIC_HELP.get(metric['id'], metric['description'])
        return {'models': registry.catalogue(), 'datasets': datasets.catalogue(), 'metrics': metrics, 'lessons': LESSONS, 'glossary': GLOSSARY}

    @app.post('/api/datasets/load')
    def load(spec: dict):
        return datasets.load(spec)

    @app.post('/api/datasets/upload')
    async def upload(file: UploadFile = File(...)):
        data = await file.read(25 * 1024 * 1024 + 1)
        if len(data) > 25 * 1024 * 1024:
            raise ValueError('Файл превышает 25 МБ.')
        return datasets.import_bytes(data, file.filename or 'data.csv')

    @app.get('/api/datasets/{identifier}')
    def describe(identifier: str):
        return datasets.describe(identifier)

    @app.get('/api/datasets/{identifier}/rows')
    def dataset_rows(identifier: str, offset: int = 0, limit: int = 100):
        return datasets.rows(identifier, offset, limit)

    @app.patch('/api/datasets/{identifier}/rows')
    def update_dataset_rows(identifier: str, payload: dict):
        return datasets.update_rows(identifier, payload.get('changes', []), payload.get('additions', []))

    @app.post('/api/jobs')
    def start(request: dict):
        datasets.describe(request.get('dataset_id', ''))
        registry.spec(request.get('model', 'ridge'))
        return jobs.start(request)

    @app.get('/api/jobs/{identifier}')
    def job(identifier: str):
        return public_job(jobs.get(identifier))

    @app.post('/api/jobs/{identifier}/reveal-test')
    def reveal(identifier: str):
        return jobs.reveal_test(identifier)

    @app.delete('/api/jobs/{identifier}')
    def cancel(identifier: str):
        return jobs.cancel(identifier)

    @app.get('/api/jobs/{identifier}/export')
    def export(identifier: str, format: str = 'json'):
        state = public_job(jobs.get(identifier))
        if state['status'] != 'completed':
            raise ValueError('Экспорт доступен после завершения расчета.')
        directory = root / 'jobs' / identifier
        if format == 'model':
            return FileResponse(directory / 'model.joblib', filename='linear-model.joblib', media_type='application/octet-stream')
        if format == 'json':
            return Response(json.dumps(state['result'], ensure_ascii=False), media_type='application/json', headers={'Content-Disposition': 'attachment; filename="experiment.json"'})
        if format != 'csv':
            raise ValueError('Формат экспорта: json, csv или model.')
        predictions = state['result'].get('predictions', {})
        stream = io.StringIO()
        writer = csv.writer(stream)
        writer.writerow(['actual', 'predicted', 'residual', 'split'])
        writer.writerows(zip(predictions.get('actual', []), predictions.get('predicted', []), predictions.get('residual', []), predictions.get('split', [])))
        return Response('\ufeff' + stream.getvalue(), media_type='text/csv', headers={'Content-Disposition': 'attachment; filename="predictions.csv"'})

    @app.get('/api/experiments')
    def list_experiments():
        return experiments.list()

    @app.post('/api/experiments')
    def save_experiment(payload: dict):
        job_id = payload.get('job_id', '')
        state = public_job(jobs.get(job_id))
        if state['status'] != 'completed':
            raise ValueError('Сначала завершите обучение.')
        request = json.loads((root / 'jobs' / job_id / 'request.json').read_text())
        return experiments.save(str(payload.get('name', '')), request, state['result'], job_id)

    @app.get('/api/experiments/{identifier}')
    def get_experiment(identifier: str):
        return experiments.get(identifier)

    @app.delete('/api/experiments/{identifier}')
    def delete_experiment(identifier: str):
        experiments.delete(identifier)
        return {'deleted': True}

    @app.post('/api/metrics/preview')
    def metric_preview(payload: dict):
        from .metrics import custom_metric
        return {'value': custom_metric(payload.get('expression', ''), payload.get('actual', []), payload.get('predicted', []))}

    @app.post('/api/predict')
    def predict(payload: dict):
        import joblib
        import pandas as pd
        identifier = payload.get('job_id', '')
        state = jobs.get(identifier)
        if state['status'] != 'completed':
            raise ValueError('Сначала завершите обучение.')
        rows = payload.get('rows', [])
        if not isinstance(rows, list) or not 0 < len(rows) <= 1000:
            raise ValueError('Передайте от 1 до 1000 строк.')
        artifact = joblib.load(root / 'jobs' / identifier / 'model.joblib')
        pipeline = artifact.get('pipeline', artifact.get('model')) if isinstance(artifact, dict) else artifact
        prediction = pipeline.predict(pd.DataFrame(rows))
        return {'predictions': prediction.tolist()}

    @app.get('/api/jobs/{identifier}/grid')
    def prediction_grid(identifier: str, x_feature: str, y_feature: str | None = None):
        import joblib
        from .training import TrainingService
        state = jobs.get(identifier)
        if state['status'] != 'completed':
            raise ValueError('Сначала завершите обучение.')
        directory = root / 'jobs' / identifier
        pipeline = joblib.load(directory / 'model.joblib')
        request = json.loads((directory / 'request.json').read_text())
        return TrainingService(datasets, registry).prediction_grid(request, pipeline, x_feature, y_feature)

    web = Path(__file__).resolve().parent.parent / 'web'
    app.mount('/', StaticFiles(directory=web, html=True), name='interface')
    return app
