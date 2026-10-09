"""Public API failure boundaries and real worker lifecycle regression tests."""

from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from linear_lab.app import create_app
from linear_lab.jobs import JobManager


@pytest.fixture
def client_and_data(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        response = client.post('/api/datasets/load', json={
            'kind': 'custom', 'name': 'Данные для API', 'target': 'y',
            'rows': [{'x': i / 10, 'z': (i % 7) - 3, 'y': 2 + 3 * i / 10} for i in range(80)],
        })
        assert response.status_code == 200, response.text
        yield client, response.json(), tmp_path


def _request(dataset, **overrides):
    return {
        'dataset_id': dataset['id'], 'target': 'y', 'features': ['x', 'z'],
        'model': 'ols', 'seed': 19, 'regularization_path': False,
        **overrides,
    }


def _terminal(client, identifier, timeout=45):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f'/api/jobs/{identifier}')
        assert response.status_code == 200, response.text
        state = response.json()
        if state['status'] != 'running':
            return state
        time.sleep(.03)
    pytest.fail('Настоящий расчет не завершился за 45 секунд.')


def test_json_endpoints_reject_nonobjects_and_broken_json(client_and_data):
    client, dataset, _ = client_and_data
    urls = ['/api/datasets/load', '/api/jobs', '/api/metrics/preview',
            '/api/experiments', f"/api/datasets/{dataset['id']}/preprocessing-preview"]
    for url in urls:
        for body in ('null', '[]', '42', '"settings"', '{"broken":'):
            response = client.post(url, content=body, headers={'Content-Type': 'application/json'})
            assert response.status_code == 422, (url, body, response.text)
            assert response.json()['detail']
    for actual, predicted in (({}, [1]), ([{}], [1]), ([1], {}), ([1], [{}]), (['text'], [1])):
        response = client.post('/api/metrics/preview', json={
            'expression': 'mean(abs(error))', 'actual': actual, 'predicted': predicted,
        })
        assert response.status_code == 422, response.text
        assert 'числовые' in response.json()['detail']


def test_training_shapes_and_seed_fail_before_starting_worker(client_and_data):
    client, dataset, root = client_and_data
    malformed = [(key, value) for key in ('params', 'split', 'preprocessing', 'resampling', 'cv_config', 'metric_params')
                 for value in (None, [], 'advanced')]
    malformed += [('seed', None), ('seed', True), ('seed', -1), ('seed', 2**32), ('seed', 1.5),
                  ('features', []), ('features', 'x'), ('features', [3]), ('metrics', {}),
                  ('metrics', True), ('model', []), ('target', [])]
    for key, value in malformed:
        payload = _request(dataset, **{key: value})
        for endpoint in ('/api/jobs', f"/api/datasets/{dataset['id']}/preprocessing-preview"):
            response = client.post(endpoint, json=payload)
            assert response.status_code == 422, (key, value, endpoint, response.text)
            assert response.json()['detail']
    for fragment in ('"seed": NaN', '"seed": Infinity', '"split": {"train": NaN}'):
        body = '{"dataset_id": ' + json.dumps(dataset['id']) + ', ' + fragment + '}'
        response = client.post('/api/jobs', content=body, headers={'Content-Type': 'application/json'})
        assert response.status_code == 422, response.text
    assert not list((root / 'jobs').iterdir())


def test_preprocessing_preview_rejects_invalid_values_and_bounds(client_and_data):
    client, dataset, root = client_and_data
    endpoint = f"/api/datasets/{dataset['id']}/preprocessing-preview"
    invalid = [
        {'preprocessing': {'degree': None}}, {'preprocessing': {'degree': True}},
        {'preprocessing': {'degree': 6}}, {'preprocessing': {'scaler': 'magic'}},
        {'preprocessing': {'numeric_transform': 'does-not-exist'}},
        {'preprocessing': {'clip_quantiles': [.9, .1]}},
        {'preprocessing': {'variance_threshold': -1}},
        {'split': {'train': None}}, {'split': {'train': True}},
        {'split': {'train': .7, 'validation': .2, 'test': .2}},
        {'resampling': {'method': 'classification_smote'}},
        {'resampling': {'method': 'random_over', 'relevance_threshold': None}},
        {'resampling': {'method': 'random_over', 'relevance_threshold': []}},
        {'resampling': {'method': 'random_over', 'relevance_threshold': 'invalid'}},
        {'resampling': {'method': 'random_over', 'relevance': 'manual', 'control_points': {}}},
    ]
    for overrides in invalid:
        response = client.post(endpoint, json=_request(dataset, **overrides))
        assert response.status_code == 422, (overrides, response.text)
        assert response.json()['detail']
    large = client.post('/api/datasets/load', json={
        'kind': 'synthetic', 'name': 'linear', 'params': {'n_samples': 60, 'n_features': 40},
    })
    assert large.status_code == 200, large.text
    response = client.post(f"/api/datasets/{large.json()['id']}/preprocessing-preview", json={
        'preprocessing': {'degree': 5},
    })
    assert response.status_code == 422, response.text
    assert '2000' in response.json()['detail'] or 'признак' in response.json()['detail']
    assert not list((root / 'jobs').iterdir())


def test_unknown_sources_and_invalid_generator_sizes_are_rejected_without_files(client_and_data):
    client, _, root = client_and_data
    before = set((root / 'datasets').glob('*.json'))
    for spec in (
        {'kind': 'url', 'name': 'http://127.0.0.1/private.csv'},
        {'kind': 'synthetic', 'name': 'linear', 'params': ['not', 'settings']},
        {'kind': 'synthetic', 'name': 'linear', 'params': {'n_samples': 100001}},
        {'kind': 'synthetic', 'name': 'linear', 'params': {'n_features': 1001}},
        {'kind': 'custom', 'rows': [{'x': {}, 'y': 1}], 'target': 'y'},
    ):
        response = client.post('/api/datasets/load', json=spec)
        assert response.status_code == 422, (spec, response.text)
        assert response.json()['detail']
    assert set((root / 'datasets').glob('*.json')) == before


def test_same_origin_mutations_work_and_external_sites_cannot_modify(client_and_data):
    client, dataset, root = client_and_data
    before = set((root / 'datasets').glob('*.json'))
    payload = {'changes': [{'index': 0, 'values': {'y': 99}}]}
    for origin in ('https://malicious.example', 'null', 'http://testserver.malicious.example'):
        response = client.patch(f"/api/datasets/{dataset['id']}/rows", json=payload, headers={'Origin': origin})
        assert response.status_code == 403, (origin, response.text)
    assert set((root / 'datasets').glob('*.json')) == before
    response = client.patch(f"/api/datasets/{dataset['id']}/rows", json=payload, headers={'Origin': 'http://testserver'})
    assert response.status_code == 200, response.text
    assert response.headers['x-content-type-options'] == 'nosniff'
    assert client.get(f"/api/datasets/{dataset['id']}").json()['preview'][0]['y'] == 2


def test_untrusted_identifiers_cannot_resolve_files_or_model_artifacts(client_and_data):
    client, dataset, root = client_and_data
    for identifier in ('not-an-id', '..', 'G' * 32, '0' * 31):
        for resource in ('jobs', 'experiments'):
            response = client.get(f'/api/{resource}/{identifier}')
            assert response.status_code == 404, (resource, identifier, response.text)
        response = client.post('/api/predict', json={'job_id': identifier, 'rows': [{'x': 1}]})
        assert response.status_code == 404, response.text
        response = client.post('/api/jobs', json=_request(dataset, dataset_id=identifier))
        assert response.status_code == 422, response.text
    assert client.get('/api/jobs/' + 'f' * 32).status_code == 404
    assert client.get('/api/datasets/' + 'f' * 32).status_code == 404
    assert not list((root / 'jobs').iterdir())


def test_upload_filename_does_not_become_a_storage_path(client_and_data):
    client, _, root = client_and_data
    response = client.post('/api/datasets/upload', files={
        'file': ('../../outside/<script>.csv', b'x,y\n1,2\n3,4\n5,6\n', 'text/csv'),
    })
    assert response.status_code == 200, response.text
    dataset = response.json()
    assert dataset['name'] == '<script>.csv'
    assert len(dataset['id']) == 32
    assert (root / 'datasets' / f"{dataset['id']}.json").exists()
    assert not (root / 'outside').exists()
    assert response.headers['content-type'].startswith('application/json')
    assert response.headers['x-content-type-options'] == 'nosniff'
    bad = client.post('/api/datasets/upload', files={'file': ('script.py', b'print(1)', 'text/plain')})
    assert bad.status_code == 422, bad.text


def test_metadata_invalid_fields_and_large_text_leave_original_unchanged(client_and_data):
    client, dataset, root = client_and_data
    before = set((root / 'datasets').glob('*.json'))
    for patch in ({'path': '../escape'}, {'name': ''}, {'name': 'x' * 201},
                  {'description': 'x' * 5001}, {'tags': ['tag'] * 31},
                  {'task': 'mind-reading'}, {'default_target': 'missing'}):
        response = client.patch(f"/api/datasets/{dataset['id']}/metadata", json=patch)
        assert response.status_code == 422, (patch, response.text)
    assert set((root / 'datasets').glob('*.json')) == before
    original = client.get(f"/api/datasets/{dataset['id']}").json()
    assert original['name'] == dataset['name']
    assert original['default_target'] == 'y'


def test_invalid_row_changes_are_atomic_and_preserve_source(client_and_data):
    client, dataset, root = client_and_data
    before = set((root / 'datasets').glob('*.json'))
    patches = [
        {'changes': [{'index': True, 'values': {'y': 7}}]},
        {'changes': [{'index': 80, 'values': {'y': 7}}]},
        {'changes': [{'index': 1, 'values': {'absent': 7}}]},
        {'changes': [{'index': 1, 'values': {'y': 7}}, {'index': 1, 'values': {'y': 8}}]},
        {'changes': [{'index': 1, 'values': {'y': []}}]},
        {'changes': [{'index': 1, 'values': {'y': 7}}], 'deletes': [1]},
        {'deletes': [1, 1]}, {'additions': [{'x': 1, 'y': 2}]},
    ]
    for patch in patches:
        response = client.patch(f"/api/datasets/{dataset['id']}/rows", json=patch)
        assert response.status_code == 422, (patch, response.text)
    assert set((root / 'datasets').glob('*.json')) == before
    assert client.get(f"/api/datasets/{dataset['id']}/rows").json()['rows'][1]['y'] == pytest.approx(2.3)


def test_genuine_preprocessing_preview_is_finite_train_only_and_has_no_model(client_and_data):
    client, dataset, root = client_and_data
    response = client.post(f"/api/datasets/{dataset['id']}/preprocessing-preview", json=_request(
        dataset, preprocessing={'scaler': 'standard', 'degree': 2, 'imputation': 'median'}, metrics='all',
    ))
    assert response.status_code == 200, response.text
    preview = response.json()
    json.dumps(preview, allow_nan=False)
    assert preview['fitted_on'] == 'train'
    assert preview['train_rows'] == 48
    assert len(preview['rows']) == 48
    assert len(preview['features']) == 5
    assert len(set(preview['indices'])) == 48
    np.testing.assert_allclose(np.mean(preview['rows'], axis=0), 0, atol=1e-12)
    assert not list((root / 'jobs').iterdir())
    assert not list(root.rglob('*.joblib'))


def test_genuine_job_uses_internal_artifact_and_finite_api_outputs(client_and_data):
    client, dataset, root = client_and_data
    unauthorized = root / 'outside-model.joblib'
    response = client.post('/api/jobs', json=_request(
        dataset, metrics='all', artifact_path=str(unauthorized), preprocessing={'scale': True},
    ))
    assert response.status_code == 200, response.text
    identifier = response.json()['id']
    state = _terminal(client, identifier)
    assert state['status'] == 'completed', state
    json.dumps(state, allow_nan=False)
    assert state['result']['metrics']['validation']['mse'] < 1e-20
    assert len(state['result']['metrics']['validation']) >= 15
    assert state['result']['metrics']['test'] == {}
    assert not unauthorized.exists()
    assert (root / 'jobs' / identifier / 'model.joblib').exists()
    saved_request = json.loads((root / 'jobs' / identifier / 'request.json').read_text())
    assert Path(saved_request['artifact_path']) == root / 'jobs' / identifier / 'model.joblib'
    assert client.get(f'/api/jobs/{identifier}/grid?x_feature=').status_code == 422
    assert client.get(f'/api/jobs/{identifier}/grid?x_feature=x&y_feature=x').status_code == 422
    assert client.get(f'/api/jobs/{identifier}/grid?x_feature=not-there').status_code == 422
    grid = client.get(f'/api/jobs/{identifier}/grid?x_feature=z&y_feature=x')
    assert grid.status_code == 200, grid.text
    json.dumps(grid.json(), allow_nan=False)
    prediction = client.post('/api/predict', json={'job_id': identifier, 'rows': [{'x': 3, 'z': 1}]})
    assert prediction.status_code == 200, prediction.text
    assert prediction.json()['predictions'][0] == pytest.approx(11)
    assert client.post('/api/predict', json={'job_id': identifier, 'rows': []}).status_code == 422
    assert client.get(f'/api/jobs/{identifier}/export?format=invalid').status_code == 422
    export = client.get(f'/api/jobs/{identifier}/export?format=joblib')
    assert export.status_code == 200, export.text
    experiment = client.post('/api/experiments', json={'job_id': identifier, 'name': 'Проверка границ'})
    assert experiment.status_code == 200, experiment.text
    assert 'artifact_path' not in experiment.json()['request']


def test_shutdown_cancels_live_jobs_and_restart_marks_crash_state_interrupted(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        dataset = client.post('/api/datasets/load', json={
            'kind': 'synthetic', 'name': 'linear', 'params': {'n_samples': 1000, 'n_features': 3},
        }).json()
        started = client.post('/api/jobs', json={
            'dataset_id': dataset['id'], 'model': 'sgd', 'epochs': 1000, 'regularization_path': False,
        })
        assert started.status_code == 200, started.text
        identifier = started.json()['id']
        assert client.get(f'/api/jobs/{identifier}').json()['status'] == 'running'
        # A genuinely started job is durable before its first progress event.
        # A fresh manager represents recovery without the original process table.
        durable = json.loads((tmp_path / 'jobs' / identifier / 'state.json').read_text())
        assert durable['status'] == 'running'
        recovered = JobManager(tmp_path)
        assert recovered.get(identifier)['status'] == 'interrupted'
        assert client.get(f'/api/jobs/{identifier}').json()['status'] == 'running'
        recovered.close()
    with TestClient(create_app(tmp_path)) as restarted:
        state = restarted.get(f'/api/jobs/{identifier}')
        assert state.status_code == 200, state.text
        assert state.json()['status'] == 'cancelled'
        assert restarted.get(f'/api/jobs/{identifier}/export?format=joblib').status_code == 422
