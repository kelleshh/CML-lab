import numpy as np
import pytest
from sklearn import metrics as sm
from cml_lab.infrastructure.ml.metrics import TaskMetrics


@pytest.mark.parametrize('beta', [.5, 1., 2.])
@pytest.mark.parametrize('average', ['binary', 'macro', 'weighted', 'micro'])
def test_fbeta_matches_independent_library_evaluation(beta, average):
    y = np.array([0, 0, 0, 1, 1, 1, 1])
    p = np.array([0, 1, 0, 1, 0, 0, 1])
    values, details = TaskMetrics().evaluate('classification', y, p, selection=['fbeta'],
        n_classes=2, metric_params={'beta': beta, 'average': average})
    assert values['fbeta'] == pytest.approx(sm.fbeta_score(y, p, beta=beta, average=average, zero_division=0))
    assert details['fbeta']['direction'] == 'max'


def test_fbeta_numeric_example_changes_weight_toward_recall():
    # TP=2, FP=1, FN=2: precision=2/3, recall=1/2.
    y, p = [0, 0, 0, 1, 1, 1, 1], [0, 1, 0, 1, 0, 0, 1]
    scores = [TaskMetrics().evaluate('classification', y, p, selection=['fbeta_binary'],
        n_classes=2, metric_params={'beta': beta})[0]['fbeta_binary'] for beta in [.5, 1, 2]]
    assert scores == pytest.approx([.625, 4/7, 10/19])
    assert scores[0] > scores[1] > scores[2]


@pytest.mark.parametrize('beta', [0, -1, True, '2', float('nan'), float('inf')])
def test_invalid_beta_returns_a_reason(beta):
    values, details = TaskMetrics().evaluate('classification', [0, 1], [0, 1],
        selection=['fbeta'], n_classes=2, metric_params={'beta': beta})
    assert values['fbeta'] is None and 'beta' in details['fbeta']['reason']


def test_binary_fbeta_is_not_reported_for_multiclass():
    values, details = TaskMetrics().evaluate('classification', [0, 1, 2], [0, 1, 2],
        selection=['fbeta_binary'], n_classes=3)
    assert values['fbeta_binary'] is None and 'ровно два класса' in details['fbeta_binary']['reason']


@pytest.mark.parametrize('name,function', [
    ('homogeneity', sm.homogeneity_score), ('completeness', sm.completeness_score),
    ('v_measure', sm.v_measure_score), ('fowlkes_mallows', sm.fowlkes_mallows_score),
    ('normalized_mutual_info', sm.normalized_mutual_info_score),
])
def test_external_cluster_scores_require_reference_and_match_library(name, function):
    actual, found = [0, 0, 1, 1, 2, 2], [1, 1, 0, 2, 2, 2]
    values, _ = TaskMetrics().evaluate('clustering', None, found, selection=[name], reference=actual)
    assert values[name] == pytest.approx(function(actual, found))
    missing, details = TaskMetrics().evaluate('clustering', None, found, selection=[name])
    assert missing[name] is None and 'известных групп' in details[name]['reason']


def test_metric_catalogue_uses_standard_names_and_meaningful_help():
    definitions = {item['id']: item for item in TaskMetrics().catalogue('classification')}
    assert definitions['accuracy']['name'] == 'Accuracy'
    assert definitions['fbeta']['name'] == 'F-beta'
    assert 'beta=2' in definitions['fbeta']['help']
    assert definitions['zero_one_loss']['direction'] == 'min'
