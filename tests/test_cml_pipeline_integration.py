"""Independent pipeline journeys across fold fitting, search, HTTP and exports."""
from __future__ import annotations

from copy import deepcopy
from io import BytesIO
import json
import time

from fastapi.testclient import TestClient
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from threadpoolctl import threadpool_limits

from cml_lab.infrastructure.ml.catalogue import AlgorithmCatalogue
from cml_lab.infrastructure.ml.fitting import fit_artifact
from cml_lab.infrastructure.ml.metrics import TaskMetrics
from cml_lab.infrastructure.ml.pipelines import compile_pipeline, PipelineDefinitionError
from cml_lab.infrastructure.ml import tuning
from cml_lab.presentation.http.api import create_app


NESTED = '''from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
numeric = Pipeline([
    ('fill', SimpleImputer(strategy='median', keep_empty_features=True)),
    ('features', FeatureUnion([
        ('scaled', StandardScaler()),
        ('powers', Pipeline([('poly', PolynomialFeatures(degree=2, include_bias=False)), ('scale', StandardScaler())])),
    ])),
])
pipeline = Pipeline([('columns', ColumnTransformer([
    ('numeric', numeric, ['a', 'b']),
    ('category', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['kind']),
], remainder='drop'))])
'''

NUMERIC = '''from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
pipeline = Pipeline([('poly', PolynomialFeatures(degree=1, include_bias=False)), ('scale', StandardScaler())])
'''


def request(source=NESTED, **changes):
    return {'task': 'regression', 'algorithm_id': 'ridge', 'params': {'alpha': .3},
            'preprocessing': {'declarative_pipeline': {'format': 'cml.pipeline', 'version': 1, 'source': source},
                              'resampling': {'method': 'none'}},
            'seed': 27, 'n_jobs': 1, 'metrics': ['rmse'], **changes}


@pytest.fixture(scope='module')
def catalogue():
    return AlgorithmCatalogue()


@pytest.fixture
def mixed():
    generator = np.random.default_rng(83)
    X = pd.DataFrame({'a': generator.normal(size=90), 'b': generator.normal(size=90),
                      'kind': np.where(np.arange(90) % 3, 'ordinary', 'special'),
                      'id': np.arange(90) + 1000})
    y = 2 * X.a.to_numpy() + X.b.to_numpy() ** 2 + .7 * (X.kind == 'special').to_numpy()
    X.loc[[1, 23, 57], 'a'] = np.nan
    return X, y


def independent_pipeline():
    numeric = Pipeline([
        ('fill', SimpleImputer(strategy='median', keep_empty_features=True)),
        ('features', FeatureUnion([
            ('scaled', StandardScaler()),
            ('powers', Pipeline([('poly', PolynomialFeatures(degree=2, include_bias=False)), ('scale', StandardScaler())])),
        ])),
    ])
    return Pipeline([('columns', ColumnTransformer([
        ('numeric', numeric, ['a', 'b']),
        ('category', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['kind']),
    ], remainder='drop')), ('model', Ridge(alpha=.3, random_state=27))])


def test_nested_fit_predict_serialization_and_actual_feature_names_match_native_sklearn(catalogue, mixed):
    X, y = mixed
    train = np.arange(60)
    holdout = X.iloc[60:].copy()
    holdout.loc[holdout.index[0], 'kind'] = 'never_in_train'
    with threadpool_limits(1):
        artifact, _, _ = fit_artifact(request(), X.iloc[train], y[train], catalogue)
        expected = independent_pipeline().fit(X.iloc[train], y[train])
    np.testing.assert_allclose(artifact.predict(holdout), expected.predict(holdout), atol=1e-12)
    names = artifact.feature_names
    assert len(names) == 9
    assert 'numeric__scaled__a' in names
    assert 'numeric__powers__a^2' in names
    assert 'category__kind_special' in names
    assert not any('id' in name for name in names)
    frozen_mean = artifact.preprocessing['columns'].named_transformers_['numeric']['features'].transformer_list[0][1].mean_.copy()
    extreme = holdout.copy()
    extreme[['a', 'b']] = 1e9
    artifact.predict(extreme)
    np.testing.assert_array_equal(artifact.preprocessing['columns'].named_transformers_['numeric']['features'].transformer_list[0][1].mean_, frozen_mean)
    serialized = BytesIO()
    joblib.dump(artifact, serialized)
    serialized.seek(0)
    replica = joblib.load(serialized)
    np.testing.assert_allclose(replica.predict(holdout), expected.predict(holdout), atol=1e-12)


def test_cross_validation_refits_every_nested_train_branch_without_holdout_leakage(catalogue, mixed):
    X, y = mixed
    plans = list(KFold(3, shuffle=False).split(X))
    first, _ = plans[0]
    altered = X.copy()
    altered.iloc[plans[0][1], altered.columns.get_indexer(['a', 'b'])] = 1e8
    spec = request()
    fit_a, _, _ = fit_artifact(spec, X.iloc[first], y[first], catalogue)
    fit_b, _, _ = fit_artifact(spec, altered.iloc[first], y[first], catalogue)
    np.testing.assert_array_equal(fit_a.estimator.coef_, fit_b.estimator.coef_)
    settings = dict(spec, search={'method': 'grid', 'metric': 'rmse', 'trials': 1, 'param_space': {'alpha': [.3]}})
    _, summary = tuning.tune(settings, X, y, catalogue, TaskMetrics(), plans)
    expected = []
    for training, validation in plans:
        native = independent_pipeline().fit(X.iloc[training], y[training])
        expected.append(np.sqrt(mean_squared_error(y[validation], native.predict(X.iloc[validation]))))
    assert summary['trials'][0]['fold_scores'] == pytest.approx(expected, abs=1e-12)
    assert settings['preprocessing']['declarative_pipeline']['source'] == NESTED


@pytest.mark.parametrize('method', ['grid', 'random', 'optuna_tpe', 'optuna_random', 'halving_grid', 'halving_random'])
def test_numeric_pipeline_search_parameters_change_real_fold_construction(catalogue, method):
    X = pd.DataFrame({'x': np.linspace(-2, 2, 90)})
    y = X.x.to_numpy() ** 2
    plans = list(KFold(3, shuffle=True, random_state=13).split(X))
    spec = request(NUMERIC, params={'alpha': 1e-8}, search={
        'method': method, 'metric': 'rmse', 'trials': 2,
        'param_space': {'pipeline__poly__degree': [1, 2]}, 'min_resources': 45,
    })
    original = deepcopy(spec)
    with threadpool_limits(1):
        winner, report = tuning.tune(spec, X, y, catalogue, TaskMetrics(), plans)
    assert spec == original
    completed = [trial for trial in report['trials'] if trial['score'] is not None]
    assert completed, report
    # Two-candidate grid/random and seeded Optuna runs all visit the quadratic.
    assert winner['pipeline__poly__degree'] == 2, report
    fitted, _, _ = fit_artifact(dict(spec, params=winner, search=None), X, y, catalogue)
    assert fitted.preprocessing['poly'].degree == 2
    np.testing.assert_allclose(fitted.predict(X), y, atol=1e-7)


def test_function_transformer_without_names_keeps_honest_fallback_and_complex_model_values(catalogue):
    source = '''import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
pipeline = Pipeline([('log', FunctionTransformer(func=np.log1p)), ('scale', StandardScaler())])
'''
    X = pd.DataFrame({'a': np.linspace(0, 3, 40), 'b': np.linspace(3, 8, 40)})
    y = np.log1p(X.a.to_numpy())
    fitted, _, _ = fit_artifact(request(source, algorithm_id='gaussian_process_regressor', params={
        'kernel': {'op': 'sum', 'left': {'class': 'RBF', 'params': {'length_scale': 1.0}},
                   'right': {'class': 'WhiteKernel', 'params': {'noise_level': .001}}},
        'optimizer': None,
    }), X, y, catalogue)
    assert fitted.feature_names == ['output_0', 'output_1']
    assert np.isfinite(fitted.predict(X)).all()


@pytest.mark.parametrize('literal', [
    "np.array([1], dtype='<U1000000000')",
    "np.array(['" + 'a' * 40000 + "'," + ','.join("''" for _ in range(400)) + '])',
])
def test_compilation_checks_array_allocation_before_calling_numpy(monkeypatch, literal):
    from cml_lab.infrastructure.ml.pipelines import registry
    calls = []
    def allocating_array(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError('The declaration attempted an oversized allocation')
    monkeypatch.setattr(np, 'array', allocating_array)
    monkeypatch.setitem(registry.REFERENCE_MODULES['numpy'], 'array', allocating_array)
    source = 'import numpy as np\nfrom sklearn.preprocessing import SplineTransformer\nknots = ' + literal + '\npipeline = SplineTransformer(knots=knots)\n'
    with pytest.raises(PipelineDefinitionError):
        compile_pipeline(source)
    assert calls == []


def wait_completed(client, identifier):
    deadline = time.monotonic() + 40
    while time.monotonic() < deadline:
        result = client.get(f'/api/cml/runs/{identifier}').json()
        if result['status'] in {'completed', 'error', 'cancelled', 'interrupted'}:
            assert result['status'] == 'completed', result
            return result
        time.sleep(.03)
    pytest.fail('Declarative pipeline worker did not finish in 40 seconds')


def test_http_nested_pipeline_validate_sources_parameters_export_and_training(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        payload = {'spec': {'format': 'cml.pipeline', 'version': 1, 'source': NUMERIC}}
        result = client.post('/api/cml/pipelines/validate', json=payload)
        assert result.status_code == 200, result.text
        diagram = result.json()['graph']
        assert any(node.get('class') == 'PolynomialFeatures' for node in diagram['nodes'])
        assert len(diagram['edges']) >= 4
        rows = client.post('/api/cml/pipelines/parameters', json=payload).json()['params']
        degree = next(item for item in rows if item['key'] == 'pipeline__poly__degree')
        assert degree['default'] == 1 and degree['lesson_id']
        source = client.get('/api/cml/pipelines/source', params={'class': 'StandardScaler'})
        assert source.status_code == 200
        assert 'class StandardScaler' in source.json()['source']
        assert source.json()['license'] == 'BSD-3-Clause'
        assert client.get('/api/cml/pipelines/source', params={'class': '../../etc/passwd'}).status_code == 422
        for format in ('py', 'json'):
            exported = client.post('/api/cml/pipelines/export', params={'format': format}, json=payload)
            assert exported.status_code == 200, exported.text
            replica = compile_pipeline(exported.text if format == 'py' else exported.json())
            assert replica['poly'].degree == 1
        invalid = client.post('/api/cml/pipelines/validate', json={'source': 'import os\npipeline = os.getcwd()'})
        assert invalid.status_code == 422 and invalid.json()['error']['line'] == 1
        dataset = client.post('/api/datasets/load', json={'name': 'linear', 'kind': 'synthetic', 'params': {'n_samples': 90}}).json()
        source = NUMERIC
        draft = {'dataset_id': dataset['id'], 'algorithm_id': 'ridge', 'params': {'alpha': .1},
                 'preprocessing': {'declarative_pipeline': {'source': source}},
                 'validation': {'strategy': 'kfold', 'folds': 3}}
        preset = client.post('/api/cml/search/preset', json={**draft, 'breadth': 2})
        assert preset.status_code == 200 and preset.json()['combinations'] > 1
        run_response = client.post('/api/cml/runs', json=draft)
        assert run_response.status_code == 200, run_response.text
        run = wait_completed(client, run_response.json()['id'])
        assert run['result']['diagnostics']['cv']['n_splits'] == 3
        artifact_response = client.get(f"/api/cml/runs/{run['id']}/export", params={'format': 'joblib'})
        assert artifact_response.status_code == 200
        artifact = joblib.load(BytesIO(artifact_response.content))
        rows = client.get(f"/api/datasets/{dataset['id']}/rows", params={'limit': 3}).json()['rows']
        raw = [{key: row[key] for key in run['spec']['features']} for row in rows]
        response = client.post('/api/cml/predict', json={'run_id': run['id'], 'rows': raw})
        assert response.status_code == 200, response.text
        np.testing.assert_allclose(response.json()['predictions'], artifact.predict(raw))
        for format in ('py', 'ipynb'):
            exported = client.get(f"/api/cml/runs/{run['id']}/project-export", params={'format': format})
            assert exported.status_code == 200, exported.text
            if format == 'py':
                compile(exported.text, '<exported-project>', 'exec')
                assert 'declarative_pipeline' in exported.text and 'PolynomialFeatures' in exported.text
            else:
                notebook = exported.json()
                assert notebook['nbformat'] == 4
                assert any('declarative_pipeline' in cell['source'] for cell in notebook['cells'] if cell['cell_type'] == 'code')


def test_saved_pipeline_exports_preserve_effective_parameter_overrides(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        created = client.post('/api/cml/preprocessor-recipes', json={
            'name': 'Cubic expansion', 'config': {'declarative_pipeline': {'source': NUMERIC},
                                               'pipeline_params': {'poly__degree': 3}},
        })
        assert created.status_code == 200, created.text
        for format in ('py', 'json'):
            response = client.get(f"/api/cml/preprocessor-recipes/{created.json()['id']}/export", params={'format': format})
            assert response.status_code == 200, response.text
            restored = compile_pipeline(response.text if format == 'py' else response.json())
            assert restored['poly'].degree == 3


def test_self_referencing_nested_estimator_is_reported_as_definition_error():
    source = '''from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
selector = SelectFromModel(Lasso())
pipeline = selector.set_params(estimator=selector)
'''
    with pytest.raises(PipelineDefinitionError) as error:
        compile_pipeline(source)
    assert error.value.line == 4
    assert error.value.column >= 1
