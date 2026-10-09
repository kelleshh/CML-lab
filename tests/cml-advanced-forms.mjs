/** Advanced forms exercise live constructor/metric/pipeline/search HTTP DTOs.
 * jsdom substitutes browser layout; the numerical catalogue and FastAPI are real.
 */
import assert from 'node:assert/strict';
import { test, after } from 'node:test';
import { createRequire } from 'node:module';
import { spawn } from 'node:child_process';
import { createInterface } from 'node:readline';
import { fileURLToPath } from 'node:url';
import { resolve } from 'node:path';
const require = createRequire(import.meta.url); const { JSDOM } = require('jsdom');
const root = fileURLToPath(new URL('..', import.meta.url));
const bridge = spawn(process.env.CML_LAB_PYTHON || resolve(root, '.venv/bin/python'), ['-u', '-c', `
import json,sys,tempfile
from fastapi.testclient import TestClient
from cml_lab.presentation.http.api import create_app
from cml_lab.infrastructure.ml.hyperparameters import json_value
with tempfile.TemporaryDirectory(prefix='cml-advanced-forms-') as directory:
    with TestClient(create_app(directory)) as client:
        for line in sys.stdin:
            try:
                item=json.loads(line)
                if item.get('operation')=='build':
                    model=client.app.state.services.policy.catalogue.build(item['task'],item['algorithm_id'],item['params'])
                    result={'class':type(model).__name__,'params':json_value(model.get_params(deep=False))}
                else:
                    options={'json':item['body']} if 'body' in item else {}
                    response=client.request(item.get('method','GET'),'/api'+item['path'],**options)
                    if response.status_code>=400: raise ValueError(str(response.json().get('detail',response.text)))
                    result=response.json()
                print(json.dumps({'result':result},ensure_ascii=False),flush=True)
            except Exception as exc:
                print(json.dumps({'error':str(exc)},ensure_ascii=False),flush=True)
`], { cwd: root, env: { ...process.env, ORT_DISABLE_TELEMETRY: '1' } });
const pending = [], requests = []; let stderr = '';
bridge.stderr.on('data', value => { stderr += value; });
createInterface({ input: bridge.stdout }).on('line', line => {
  const next = pending.shift(); if (!next) return;
  try { const response = JSON.parse(line); response.error ? next.reject(new Error(response.error)) : next.resolve(response.result); }
  catch (error) { next.reject(error); }
});
bridge.on('exit', code => { while (pending.length) pending.shift().reject(new Error(`API bridge exited ${code}: ${stderr}`)); });
function exchange(item) { return new Promise((resolve, reject) => { pending.push({ resolve, reject }); bridge.stdin.write(`${JSON.stringify(item)}\n`); }); }
const api = { async request(path, options = {}) { requests.push({ path, ...structuredClone(options) }); const result = await exchange({ path, ...options }); requests.at(-1).result = structuredClone(result); return result; } };
const dom = new JSDOM('<!doctype html><body></body>', { url: 'http://localhost/', pretendToBeVisual: true });
const { window } = dom;
Object.assign(globalThis, { window, document: window.document, Node: window.Node });
const { SchemaForm } = await import('../web/cml/schema-form.js');
const { ValidationBuilder } = await import('../web/cml/validation.js');
const { parseChoices, searchDefinition } = await import('../web/cml/search-space.js');
const catalogue = await api.request('/cml/catalogue');
const algorithm = name => { const item = catalogue.algorithms.find(item => item.id === name); assert.ok(item, name); return item; };
const parameter = (name, key) => { const item = algorithm(name).params.find(item => item.key === key); assert.ok(item, `${name}.${key}`); return item; };
const container = () => { const element = document.createElement('div'); document.body.append(element); return element; };
const change = (control, value, event = 'change') => { assert.ok(control); if (control.type === 'checkbox') control.checked = value; else control.value = value; control.dispatchEvent(new window.Event(event, { bubbles: true })); };
const errors = [];
function validation(model = 'ridge', task = 'regression') {
  const subject = new ValidationBuilder({ catalogue, api, getModelConfig: () => ({ params: {} }), onChange: () => {}, onError: error => errors.push(error) }).mount(container());
  subject.setTask(task); subject.setModel(algorithm(model)); return subject;
}
const pipeline = { format: 'cml.pipeline', version: 1, source: 'from sklearn.pipeline import Pipeline\nfrom sklearn.preprocessing import StandardScaler,PolynomialFeatures\npipeline=Pipeline([("scale",StandardScaler()),("poly",PolynomialFeatures(degree=2,include_bias=False))])\n' };
after(async () => { bridge.stdin.end(); await new Promise(resolve => { if (bridge.exitCode !== null) return resolve(); const timer = setTimeout(() => { bridge.kill(); resolve(); }, 5000); bridge.once('exit', () => { clearTimeout(timer); resolve(); }); }); dom.window.close(); });

test('native English parameter keys and exact per-parameter Russian lesson routes render from the actual catalogue', () => {
  const descriptor = algorithm('decision_tree_classifier'); const subject = new SchemaForm().mount(container(), descriptor.params);
  assert.equal(descriptor.name, 'DecisionTreeClassifier');
  assert.deepEqual([...subject.controls.keys()], descriptor.params.map(item => item.key));
  for (const [key, { control, field }] of subject.controls) {
    assert.equal(control.dataset.parameter, key);
    const wrapper = control.closest('.cml-field');
    assert.equal(wrapper.querySelector('label').textContent, key);
    const help = wrapper.querySelector('.cml-help-button');
    assert.equal(new URL(help.href).searchParams.get('id'), field.lesson_id);
    assert.equal(help.target, '_blank');
    assert.ok(wrapper.querySelector('[role=tooltip]').textContent.includes(field.help));
  }
});

test('structured estimator and kernel JSON forms preserve executable constructor recipes', async () => {
  const estimator = { algorithm_id: 'decision_tree_classifier', params: { max_depth: 3 } };
  const bagging = new SchemaForm().mount(container(), [parameter('bagging_classifier', 'estimator')], { estimator });
  assert.equal(bagging.controls.get('estimator').control.tagName, 'TEXTAREA');
  assert.deepEqual(bagging.values(), { estimator });
  const built = await exchange({ operation: 'build', task: 'classification', algorithm_id: 'bagging_classifier', params: bagging.values() });
  assert.equal(built.class, 'BaggingClassifier'); assert.match(JSON.stringify(built.params.estimator), /DecisionTreeClassifier|max_depth/);
  const kernel = { op: 'sum', left: { class: 'RBF', params: { length_scale: 2 } }, right: { class: 'WhiteKernel', params: { noise_level: .1 } } };
  const form = new SchemaForm().mount(container(), [parameter('gaussian_process_regressor', 'kernel')], { kernel });
  assert.deepEqual(form.values(), { kernel });
  const gaussian = await exchange({ operation: 'build', task: 'regression', algorithm_id: 'gaussian_process_regressor', params: form.values() });
  assert.equal(gaussian.class, 'GaussianProcessRegressor'); assert.match(JSON.stringify(gaussian.params.kernel), /RBF/);
});

test('invalid JSON remains editable, invalidates submission, then valid null and updated structures clear errors', () => {
  const subject = new SchemaForm().mount(container(), [parameter('bagging_classifier', 'estimator')]);
  const control = subject.controls.get('estimator').control;
  change(control, '{"algorithm_id":', 'input');
  assert.equal(subject.valid(), false); assert.throws(() => subject.values(), /JSON/); assert.equal(control.value, '{"algorithm_id":');
  change(control, 'null', 'input'); assert.equal(subject.valid(), true); assert.deepEqual(subject.values(), { estimator: null });
  const value = { algorithm_id: 'decision_tree_classifier', params: { max_depth: 2 } };
  subject.setValues({ estimator: value }); assert.equal(subject.valid(), true); assert.deepEqual(subject.values(), { estimator: value });
});

test('nullable numerical parameters preserve None instead of a zero sentinel or NaN', async () => {
  const subject = new SchemaForm().mount(container(), [parameter('decision_tree_classifier', 'max_depth')]);
  const control = subject.controls.get('max_depth').control;
  assert.equal(control.value, ''); assert.equal(control.required, false); assert.deepEqual(subject.values(), { max_depth: null });
  const defaultTree = await exchange({ operation: 'build', task: 'classification', algorithm_id: 'decision_tree_classifier', params: subject.values() });
  assert.equal(defaultTree.params.max_depth, null);
  change(control, '4'); assert.deepEqual(subject.values(), { max_depth: 4 });
  subject.setValues({ max_depth: null }); assert.equal(control.value, ''); assert.deepEqual(subject.values(), { max_depth: null });
});

test('Search breadth fetches and applies actual validated Tiny and Wide grids, preserving editable values', async () => {
  const subject = validation();
  change(subject.breadth, 1, 'input'); assert.equal(subject.breadthOutput.textContent, 'Tiny');
  await subject.applyPreset();
  const tinyRequest = requests.findLast(item => item.path === '/cml/search/preset');
  assert.equal(tinyRequest.body.breadth, 1); assert.equal(tinyRequest.body.algorithm_id, 'ridge');
  assert.deepEqual(subject.values().search.param_space, tinyRequest.result.param_space); assert.equal(subject.searchEnabled.checked, true);
  const tiny = tinyRequest.result.combinations;
  change(subject.breadth, 5, 'input'); assert.equal(subject.breadthOutput.textContent, 'Wide'); await subject.applyPreset();
  const wideRequest = requests.findLast(item => item.path === '/cml/search/preset');
  assert.deepEqual(subject.values().search.param_space, wideRequest.result.param_space); assert.ok(wideRequest.result.combinations >= tiny);
  assert.equal(Number(subject.trials.value), wideRequest.result.trials);
  const row = subject.searchRows.find(item => item.key.value === 'alpha'); assert.ok(row);
  change(row.values, '[0.25, 3]'); assert.deepEqual(subject.values().search.param_space.alpha, [.25, 3]);
  assert.match(subject.searchBudget.textContent, /CV fits/); assert.equal(errors.length, 0);
});

test('continuous logarithmic ranges remain typed ranges through restoration and model rerender', () => {
  const subject = validation();
  const range = { type: 'float', low: .0001, high: 100, log: true };
  subject.setConfig({ search: { method: 'optuna_tpe', metric: 'rmse', direction: 'min', trials: 7, param_space: { alpha: range } }, validation: { strategy: 'kfold', folds: 4 } });
  assert.deepEqual(subject.values().search.param_space.alpha, range);
  subject.setModel(algorithm('ridge')); assert.deepEqual(subject.values().search.param_space.alpha, range);
  const row = subject.searchRows[0];
  change(row.low, '0'); assert.throws(() => subject.values(), /low > 0/);
  change(row.low, '.001'); assert.deepEqual(subject.values().search.param_space.alpha, { ...range, low: .001 });
  change(row.mode, 'int'); change(row.low, '1.5'); assert.throws(() => subject.values(), /целыми/);
});

test('mixed native JSON parameter choices parse actual numerical values and retain strings/null/bool', () => {
  assert.deepEqual(parseChoices('0.1, 1, 10', parameter('ridge', 'alpha')), [.1, 1, 10]);
  assert.deepEqual(parseChoices('scale, 0.1, null, True, false', { type: 'json', key: 'gamma' }), ['scale', .1, null, true, false]);
  assert.deepEqual(parseChoices('[{"0": 1, "1": 3}, null]', parameter('decision_tree_classifier', 'class_weight')), [{ 0: 1, 1: 3 }, null]);
});

test('F-beta metrics use standard names, numeric beta and averaging in the project configuration', async () => {
  const subject = validation('decision_tree_classifier', 'classification');
  const metric = catalogue.metrics.find(item => item.id === 'fbeta'); assert.ok(metric); assert.equal(metric.name, 'F-beta');
  const checkbox = [...subject.metricList.querySelectorAll('input')].find(item => item.value === 'fbeta'); assert.ok(checkbox);
  change(checkbox, true); change(subject.beta, '2'); change(subject.average, 'weighted');
  const values = subject.values(); assert.ok(values.metrics.includes('fbeta')); assert.equal(values.metric_params.beta, 2); assert.equal(values.metric_params.average, 'weighted');
  const help = [...subject.container.querySelectorAll('.cml-help-button')].find(item => new URL(item.href).searchParams.get('id') === 'fbeta-metrics'); assert.ok(help); assert.equal(help.target, '_blank');
  const lesson = await api.request('/learning/lessons/fbeta-metrics'); assert.ok(JSON.stringify(lesson).includes('beta'));
});

test('nested Pipeline parameters and exact lesson metadata become real search choices with explicit fit budgets', async () => {
  const subject = validation(); await subject.setPipeline(pipeline);
  const definition = subject.parameters().find(item => item.key === 'pipeline__poly__degree'); assert.ok(definition);
  assert.equal(definition.lesson_id, 'parameter-pipeline-polynomialfeatures-degree'); assert.ok(definition.help);
  const request = requests.findLast(item => item.path === '/cml/pipelines/parameters'); assert.deepEqual(request.body.spec, pipeline);
  subject.setConfig({ search: { method: 'optuna_tpe', trials: 100, metric: 'rmse', param_space: { pipeline__poly__degree: [1, 2, 3] } }, validation: { strategy: 'repeated_kfold', folds: 10, repeats: 5 } });
  assert.deepEqual(subject.values().search.param_space.pipeline__poly__degree, [1, 2, 3]);
  assert.match(subject.searchBudget.textContent, /CV fits: 150/);
  const row = subject.searchRows[0]; change(row.values, '1, 2');
  assert.deepEqual(subject.values().search.param_space.pipeline__poly__degree, [1, 2]);
  const tooltip = row.key.closest('.cml-field').querySelector('[role=tooltip]'); assert.ok(tooltip.textContent.includes(definition.help));
  change(row.mode, 'int'); change(row.low, '1'); change(row.high, '20');
  assert.match(subject.searchBudget.textContent, /CV fits: 5000/);
  assert.match(subject.searchBudget.textContent, /максимум 100 проб и 500 обучений/);
});
