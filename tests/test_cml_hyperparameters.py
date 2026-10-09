"""Complete native constructor forms and executable advanced parameter recipes."""
from __future__ import annotations

from copy import deepcopy
import inspect
import io
import json

import joblib
import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from threadpoolctl import threadpool_limits

from cml_lab.infrastructure.ml.catalogue import AlgorithmCatalogue


@pytest.fixture(scope='module')
def catalogue():
    return AlgorithmCatalogue()


def test_every_native_constructor_parameter_has_exact_name_and_documentation_route(catalogue):
    for descriptor in catalogue.list():
        if not descriptor['available']:
            continue
        estimator = catalogue.build(descriptor['tasks'][0], descriptor['id'])
        names = {name for name, parameter in inspect.signature(type(estimator)).parameters.items()
                 if parameter.kind not in (parameter.VAR_KEYWORD, parameter.VAR_POSITIONAL)}
        names.update(estimator.get_params(deep=False))
        fields = {field['key']: field for field in descriptor['params']}
        assert names <= fields.keys(), descriptor['id']
        assert descriptor['name'] == type(estimator).__name__
        assert descriptor['lesson_id'] == f"model-{descriptor['id']}"
        assert not {'hidden_units', 'components', 'base_estimators'} & fields.keys()
        for name, field in fields.items():
            assert field['label'] == name
            assert field['native_name'] == name
            assert field['help_key'] == f"model.{descriptor['id']}.{name}"
            assert field['lesson_id'] == f"parameter-{descriptor['id']}-{name.lower().replace('_', '-')}"
            assert 'native_default' in field
            assert field['help']
    # The HTTP payload is real JSON, including non-finite library defaults.
    json.dumps(catalogue.list(), allow_nan=False)


def test_all_sklearn_choices_are_available_instead_of_small_handwritten_subset(catalogue):
    solver = next(field for field in catalogue.spec('ridge')['params'] if field['key'] == 'solver')
    assert 'lbfgs' in solver['options']
    assert 'sparse_cg' in solver['options']
    criterion = next(field for field in catalogue.spec('decision_tree_classifier')['params'] if field['key'] == 'criterion')
    assert set(criterion['options']) == {'gini', 'entropy', 'log_loss'}


@pytest.mark.parametrize('task,algorithm_id,params,expected', [
    ('classification', 'decision_tree_classifier', {'criterion': 'entropy', 'ccp_alpha': .025, 'min_samples_leaf': .1}, {'criterion': 'entropy', 'ccp_alpha': .025, 'min_samples_leaf': .1}),
    ('regression', 'ridge', {'copy_X': False, 'positive': True, 'solver': 'lbfgs'}, {'copy_X': False, 'positive': True, 'solver': 'lbfgs'}),
    ('classification', 'mlp_classifier', {'hidden_layer_sizes': [8, 4], 'batch_size': 8, 'learning_rate_init': .02, 'random_state': 17, 'max_iter': 10}, {'hidden_layer_sizes': (8, 4), 'batch_size': 8, 'learning_rate_init': .02, 'random_state': 17}),
    ('classification', 'logistic_regression', {'solver': 'saga', 'l1_ratio': 1.0, 'class_weight': {'0': 1, '1': 2}}, {'solver': 'saga', 'l1_ratio': 1.0}),
    ('regression', 'bayesian_ridge', {'alpha_1': .01, 'lambda_init': .5, 'compute_score': False}, {'alpha_1': .01, 'lambda_init': .5, 'compute_score': False}),
    ('regression', 'mcp', {'max_epochs': 100, 'p0': 2, 'ws_strategy': 'fixpoint'}, {'max_epochs': 100, 'p0': 2, 'ws_strategy': 'fixpoint'}),
])
def test_advanced_constructor_values_are_applied_not_silently_replaced(catalogue, task, algorithm_id, params, expected):
    original = deepcopy(params)
    model = catalogue.build(task, algorithm_id, params, seed=99)
    actual = model.get_params(deep=False)
    for key, value in expected.items():
        assert actual[key] == value
    assert params == original


def test_multi_layer_recipe_fits_serializes_and_keeps_explicit_seed(catalogue):
    X, y = make_classification(n_samples=50, n_features=4, n_redundant=0, random_state=11)
    model = catalogue.build('classification', 'mlp_classifier', {
        'hidden_layer_sizes': [8, 4], 'solver': 'lbfgs', 'max_iter': 40,
        'random_state': 17, 'tol': .01,
    }, seed=99)
    with threadpool_limits(1):
        model.fit(X, y)
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    buffer.seek(0)
    copy = joblib.load(buffer)
    assert copy.hidden_layer_sizes == (8, 4)
    assert copy.random_state == 17
    np.testing.assert_array_equal(copy.predict(X), model.predict(X))


def test_named_ensemble_models_weights_and_final_estimator_are_executable(catalogue):
    components = [
        {'name': 'tree', 'algorithm_id': 'decision_tree_classifier', 'params': {'max_depth': 2}},
        {'name': 'linear', 'algorithm_id': 'logistic_regression', 'params': {'C': .5}},
    ]
    voting = catalogue.build('classification', 'voting_classifier', {
        'estimators': components, 'weights': [3, 1], 'flatten_transform': False,
    })
    assert [name for name, _ in voting.estimators] == ['tree', 'linear']
    assert voting.weights == [3, 1]
    assert not voting.flatten_transform
    stacking = catalogue.build('classification', 'stacking_classifier', {
        'estimators': components, 'final_estimator': {
            'algorithm_id': 'decision_tree_classifier', 'params': {'max_depth': 1}},
        'passthrough': True, 'stack_method': 'predict_proba', 'cv': 2,
    })
    X, y = make_classification(n_samples=50, n_features=4, n_redundant=0, random_state=11)
    with threadpool_limits(1):
        stacking.fit(X, y)
    assert stacking.final_estimator_.max_depth == 1
    assert stacking.final_estimator_.n_features_in_ == 6
    assert stacking.predict_proba(X).shape == (50, 2)


def test_gaussian_process_kernel_composition_and_covariance_estimator_fit(catalogue):
    X, y = make_regression(n_samples=20, n_features=2, random_state=11)
    kernel = {'op': 'sum', 'left': {'class': 'RBF', 'params': {'length_scale': 2.0}},
              'right': {'class': 'WhiteKernel', 'params': {'noise_level': .1}}}
    model = catalogue.build('regression', 'gaussian_process_regressor', {'kernel': kernel, 'optimizer': None})
    model.fit(X, y)
    assert model.kernel.k1.length_scale == 2.0
    assert np.isfinite(model.predict(X)).all()
    classifier = catalogue.build('classification', 'lda', {
        'solver': 'lsqr', 'covariance_estimator': {'class': 'LedoitWolf', 'params': {'assume_centered': False}},
    })
    classifier.fit(X, y > np.median(y))
    assert type(classifier.covariance_estimator).__name__ == 'LedoitWolf'
    assert np.isfinite(classifier.predict_proba(X)).all()


@pytest.mark.parametrize('algorithm_id,params,expected', [
    ('catboost_regressor', {'n_estimators': 7, 'eta': .2}, {'iterations': 7, 'learning_rate': .2}),
    ('lightgbm_regressor', {'max_bin': 31, 'lambda_l1': .2}, {'max_bin': 31, 'lambda_l1': .2}),
    ('xgboost_regressor', {'grow_policy': 'lossguide', 'max_leaves': 8, 'reg_alpha': .2}, {'grow_policy': 'lossguide', 'max_leaves': 8, 'reg_alpha': .2}),
])
def test_boosting_advanced_native_parameters_reach_real_libraries(catalogue, algorithm_id, params, expected):
    spec = catalogue.spec(algorithm_id)
    if not spec['available']:
        pytest.skip(spec['reason'])
    model = catalogue.build('regression', algorithm_id, params)
    X, y = make_regression(n_samples=30, n_features=3, random_state=11)
    with threadpool_limits(1):
        model.fit(X, y)
    actual = model.get_all_params() if algorithm_id.startswith('catboost') else model.get_params()
    for key, value in expected.items():
        assert actual[key] == pytest.approx(value) if isinstance(value, float) else actual[key] == value
    assert np.isfinite(model.predict(X)).all()


def test_old_saved_aliases_work_but_are_absent_from_constructor_forms(catalogue):
    model = catalogue.build('classification', 'mlp_classifier', {'hidden_units': 7})
    assert model.hidden_layer_sizes == (7,)
    model = catalogue.build('classification', 'voting_classifier', {
        'components': [{'algorithm_id': 'gaussian_nb'}, {'algorithm_id': 'logistic_regression'}],
    })
    assert len(model.estimators) == 2
    assert catalogue.build('regression', 'random_forest_regressor', {'max_features': 'all', 'max_depth': 0}).max_depth is None


@pytest.mark.parametrize('algorithm_id,params', [
    ('random_forest_regressor', {'n_jobs': -1}),
    ('random_forest_regressor', {'n_jobs': True}),
    ('mlp_regressor', {'hidden_layer_sizes': [4, True]}),
    ('gaussian_process_regressor', {'kernel': {'class': 'os.system', 'params': {'command': 'echo nope'}}}),
    ('voting_classifier', {'estimators': [{'name': 'same', 'algorithm_id': 'gaussian_nb'}, {'name': 'same', 'algorithm_id': 'logistic_regression'}]}),
    ('ols', {'unknown_parameter': 1}),
    ('ridge', {'alpha': {'function': 'os.system'}}),
    ('catboost_regressor', {'iterations': 3, 'n_estimators': 4}),
])
def test_invalid_native_and_executable_declarations_are_rejected(catalogue, algorithm_id, params):
    task = catalogue.spec(algorithm_id)['tasks'][0]
    with pytest.raises(ValueError):
        catalogue.build(task, algorithm_id, params)


def test_capabilities_follow_configured_estimator_methods_and_keep_domain_limits(catalogue):
    novelty = catalogue.build('anomaly', 'local_outlier_factor', {'novelty': True})
    caps = catalogue.effective_capabilities(novelty, catalogue.spec('local_outlier_factor'))
    assert caps['predict'] and not caps['fit_predict'] and not caps['transductive']
    ordinary = catalogue.build('anomaly', 'local_outlier_factor')
    caps = catalogue.effective_capabilities(ordinary, catalogue.spec('local_outlier_factor'))
    assert not caps['predict'] and caps['fit_predict'] and caps['transductive']
    for algorithm, params in [('svc', {'probability': False}), ('sgd_classifier', {'loss': 'hinge'})]:
        model = catalogue.build('classification', algorithm, params)
        assert not catalogue.effective_capabilities(model, catalogue.spec(algorithm))['predict_proba']
    model = catalogue.build('classification', 'multinomial_nb')
    assert catalogue.effective_capabilities(model, catalogue.spec('multinomial_nb'))['input_domain'] == 'nonnegative'


def test_alias_edits_work_when_form_sends_all_default_values(catalogue):
    params = {field['key']: deepcopy(field['default']) for field in catalogue.spec('catboost_regressor')['params']}
    params['n_estimators'] = 7
    model = catalogue.build('regression', 'catboost_regressor', params)
    assert model.get_params()['n_estimators'] == 7
    assert 'iterations' not in model.get_params()
    params = {field['key']: deepcopy(field['default']) for field in catalogue.spec('lightgbm_regressor')['params']}
    params['lambda_l1'] = .2
    model = catalogue.build('regression', 'lightgbm_regressor', params)
    assert model.get_params()['lambda_l1'] == .2
    assert model.get_params()['reg_alpha'] == 0  # wrapper default, native alias takes priority


def test_every_parameter_has_individual_russian_knowledge_and_custom_domains_are_checked(catalogue):
    from cml_lab.infrastructure.hyperparameter_learning import parameter_knowledge
    for definition in catalogue.list():
        for field in definition['params']:
            knowledge = parameter_knowledge(definition['id'], definition['family'], field['key'],
                                            class_name=definition['name'], library=definition['library'])
            assert knowledge.get('summary'), (definition['id'], field['key'])
            assert knowledge.get('example'), (definition['id'], field['key'])
    for algorithm, params in [('group_lasso', {'group_size': 0}), ('fused_lasso', {'fusion_strength': -1}),
                              ('weighted_lasso', {'profile': 'invented-profile'}), ('l0', {'max_features': 13})]:
        with pytest.raises(ValueError):
            catalogue.build('regression', algorithm, params)


def test_every_full_gui_default_form_builds_real_estimator(catalogue):
    """The browser sends all controls, including untouched nullable aliases."""
    from sklearn.base import clone
    for descriptor in catalogue.list():
        if not descriptor['available']:
            continue
        values = {field['key']: deepcopy(field['default']) for field in descriptor['params']}
        original = deepcopy(values)
        try:
            model = catalogue.build(descriptor['tasks'][0], descriptor['id'], values, seed=83, n_jobs=2)
        except Exception as error:
            pytest.fail(f"Full GUI defaults for {descriptor['id']} failed: {error}")
        assert type(model).__name__ == descriptor['name']
        assert values == original
        assert type(clone(model)) is type(model)
