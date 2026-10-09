"""HTTP API и локальный интерфейс лаборатории."""

from __future__ import annotations

import csv
import copy
import io
import json
import math
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


def _training_request(payload: dict) -> dict:
    """Validate public configuration shapes before calling numerical services.

    Numerical algorithms still own their parameter domains. This boundary only
    rejects malformed JSON structures that would otherwise become AttributeError
    or TypeError in a synchronous preview, or an opaque worker failure.
    """
    request = dict(payload)
    for name in ('params', 'split', 'preprocessing', 'resampling', 'cv_config', 'metric_params'):
        if name in request and not isinstance(request[name], dict):
            raise ValueError(f'Поле «{name}» должно быть объектом настроек, например {{}}.')
    if 'search' in request and request['search'] is not None and not isinstance(request['search'], dict):
        raise ValueError('Поле «search» должно быть объектом настроек поиска или null.')
    seed = request.get('seed', 42)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
        raise ValueError('Случайное зерно seed должно быть целым числом от 0 до 4294967295.')
    if 'model' in request and (not isinstance(request['model'], str) or not request['model']):
        raise ValueError('Поле «model» должно содержать имя модели из каталога.')
    if 'target' in request and request['target'] is not None and not isinstance(request['target'], str):
        raise ValueError('Поле «target» должно содержать название целевой колонки.')
    features = request.get('features')
    if features is not None and (not isinstance(features, list) or not features or not all(isinstance(name, str) and name for name in features)):
        raise ValueError('Поле «features» должно быть непустым списком названий признаков.')
    metrics = request.get('metrics')
    if metrics is not None and not (isinstance(metrics, str) and metrics or isinstance(metrics, list) and metrics and all(isinstance(name, str) and name for name in metrics)):
        raise ValueError('Поле «metrics» должно быть именем метрики, «all» или непустым списком имен.')
    for name in ('train', 'validation', 'test'):
        if name in request.get('split', {}):
            value = request['split'][name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f'Доля «{name}» должна быть конечным числом между 0 и 1.')
    return request


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
    from .exports import ModelExportService
    model_exports = ModelExportService(root)

    @asynccontextmanager
    async def lifespan(app):
        yield
        jobs.close()

    app = FastAPI(title='Линейная лаборатория', version='2.0.0', lifespan=lifespan)
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
        return {'status': 'ok', 'version': '2.0.0'}

    @app.get('/api/catalogue')
    def catalogue():
        from .metrics import metric_catalogue
        metrics = metric_catalogue()
        for metric in metrics:
            metric['help'] = METRIC_HELP.get(metric['id'], metric['description'])
        from .validation import CV_STRATEGIES
        return {'models': registry.catalogue(), 'datasets': datasets.catalogue(), 'metrics': metrics, 'lessons': LESSONS, 'glossary': GLOSSARY, 'cv_strategies': CV_STRATEGIES}

    @app.get('/api/datasets/library')
    def dataset_library(query: str = '', task: str | None = None, source: str | None = None, offset: int = 0, limit: int = 200):
        return datasets.list_library(query=query, task=task, source=source, offset=offset, limit=limit)

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
        return datasets.update_rows(identifier, payload.get('changes', []), payload.get('additions', []), payload.get('deletes', []))

    @app.patch('/api/datasets/{identifier}/metadata')
    def update_dataset_metadata(identifier: str, payload: dict):
        return datasets.update_metadata(identifier, payload)

    @app.delete('/api/datasets/{identifier}')
    def delete_dataset(identifier: str):
        # A saved experiment must still be reproducible after reopening it.
        for summary in experiments.list():
            experiment = experiments.get(summary['id'])
            if experiment.get('request', {}).get('dataset_id') == identifier:
                raise ValueError('Набор используется сохраненным экспериментом. Сначала удалите этот эксперимент.')
        return datasets.delete(identifier)

    @app.get('/api/datasets/{identifier}/explore')
    def explore_dataset(identifier: str, x: str | None = None, y: str | None = None, z: str | None = None,
                        color: str | None = None, target: str | None = None, sample_size: int = 2000, seed: int = 42):
        return datasets.explore(identifier, x=x, y=y, z=z, color=color, target=target, sample_size=sample_size, seed=seed)

    @app.get('/api/datasets/{identifier}/export')
    def export_dataset(identifier: str):
        return Response(datasets.export_csv(identifier), media_type='text/csv', headers={'Content-Disposition': 'attachment; filename="dataset.csv"'})

    @app.post('/api/datasets/{identifier}/preprocessing-preview')
    def preview_preprocessing(identifier: str, payload: dict):
        import numpy as np
        from .training import TrainingService, json_safe
        request = _training_request(dict(payload, dataset_id=identifier))
        service = TrainingService(datasets, registry)
        bundle, X, numeric, y, groups = service._dataset_context(request)
        parts = service._partition(X, y, groups, request)
        if request.get('resampling', {}).get('method', 'none') != 'none' and (
            request.get('cv_config', {}).get('strategy') == 'timeseries' or request.get('split', {}).get('shuffle') is False
        ):
            raise ValueError('Для временных рядов пересэмплирование выключено: оно меняет хронологию наблюдений.')
        preprocessor, _, _ = service._preprocessor(X, request.get('preprocessing', {}))
        train = parts['train']
        transformed = preprocessor.fit_transform(X.iloc[train], y[train])
        from .preprocessing import ResamplingService
        sampled, sampled_y, sampling = ResamplingService().fit_resample(
            transformed, y[train], request.get('resampling', {}), seed=int(request.get('seed', 42)),
            categorical=bool(len(X.select_dtypes(exclude='number').columns)),
        )
        columns = preprocessor.get_feature_names_out().tolist()
        sample = np.asarray(transformed[:100], dtype=float)
        return json_safe({'fitted_on': 'train', 'train_rows': len(train), 'original_features': X.columns.tolist(),
                          'features': columns, 'rows': sample.tolist(), 'indices': train[:100],
                          'original_rows': X.iloc[train[:100]].to_dict(orient='records'),
                          'target': {'name': bundle.target_name, 'values': y[train[:100]]},
                          'means': np.mean(transformed, axis=0), 'std': np.std(transformed, axis=0),
                          'sampling': dict(sampling, target_before=y[train][:2000], target_after=sampled_y[:2000], rows=sampled[:100]),
                          'note': 'Преобразования обучены только на train. Показаны первые 100 обучающих строк; модель не обучалась.'})

    @app.post('/api/jobs')
    def start(request: dict):
        request = _training_request(request)
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
        if format in {'model', 'joblib', 'pickle', 'skops', 'onnx', 'bundle', 'passport'}:
            request = json.loads((directory / 'request.json').read_text())
            if state['result'].get('search', {}).get('best_params'):
                request['params'] = state['result']['search']['best_params']
            data, filename, media_type = model_exports.export(identifier, 'joblib' if format == 'model' else format, state['result'], request)
            return Response(data, media_type=media_type, headers={'Content-Disposition': f'attachment; filename="{filename}"'})
        if format == 'json':
            return Response(json.dumps(state['result'], ensure_ascii=False), media_type='application/json', headers={'Content-Disposition': 'attachment; filename="experiment.json"'})
        if format != 'csv':
            raise ValueError('Формат экспорта: json, csv, model, pickle, skops, onnx или bundle.')
        predictions = state['result'].get('predictions', {})
        stream = io.StringIO()
        writer = csv.writer(stream)
        writer.writerow(['actual', 'predicted', 'residual', 'split'])
        writer.writerows(zip(predictions.get('actual', []), predictions.get('predicted', []), predictions.get('residual', []), predictions.get('split', [])))
        return Response('\ufeff' + stream.getvalue(), media_type='text/csv', headers={'Content-Disposition': 'attachment; filename="predictions.csv"'})

    @app.get('/api/jobs/{identifier}/export-capabilities')
    def export_capabilities(identifier: str):
        state = jobs.get(identifier)
        if state['status'] != 'completed':
            raise ValueError('Сначала завершите обучение.')
        return {'formats': model_exports.capabilities(identifier)}

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
        request.pop('artifact_path', None)
        if state['result'].get('search', {}).get('best_params'):
            request['params'] = state['result']['search']['best_params']
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
