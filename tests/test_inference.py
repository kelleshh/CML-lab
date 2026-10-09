"""Смена отображаемых осей использует уже обученную модель."""
import joblib
import numpy as np
import pandas as pd

from linear_lab.datasets import DataService
from linear_lab.models import ModelRegistry
from linear_lab.training import TrainingService
from linear_lab.app import public_job


def test_late_feature_grid_matches_loaded_pipeline_without_fitting(tmp_path):
    data = DataService(tmp_path / 'datasets')
    dataset = data.load({'kind': 'synthetic', 'name': 'linear', 'params': {'n_samples': 100, 'n_features': 4, 'noise': 0, 'seed': 21}})
    request = {'dataset_id': dataset['id'], 'model': 'ols', 'params': {}, 'seed': 42, 'regularization_path': False, 'artifact_path': str(tmp_path / 'model.joblib')}
    service = TrainingService(data, ModelRegistry())
    result = service.run(request)
    pipeline = joblib.load(request['artifact_path'])
    before = pipeline.named_steps['model'].coef_.copy()
    names = data.resolve(dataset['id']).feature_names
    grid = service.prediction_grid(request, pipeline, names[3], names[2])
    training = data.resolve(dataset['id']).X.iloc[result['split']['indices']['train']]
    x, y = np.meshgrid(grid['x'], grid['y'])
    rows = pd.DataFrame({name: np.full(x.size, training[name].median()) for name in names})
    rows[names[3]], rows[names[2]] = x.ravel(), y.ravel()
    expected = pipeline.predict(rows).reshape(x.shape)
    np.testing.assert_allclose(grid['z'], expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(pipeline.named_steps['model'].coef_, before)
    assert grid['features'] == [names[3], names[2]]


def test_pending_progress_hides_unpartitioned_predictions_without_mutation():
    state = {'status': 'running', 'events': [{'frame': {'step': 1, 'coef': [2], 'train_loss': 4, 'predicted': [1, 2, 3]}}]}
    response = public_job(state)
    assert 'predicted' not in response['events'][0]['frame']
    assert response['events'][0]['frame']['train_loss'] == 4
    assert state['events'][0]['frame']['predicted'] == [1, 2, 3]
