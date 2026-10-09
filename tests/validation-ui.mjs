/** Test public GUI requests and actual server report rendering, not model maths. */
import assert from 'node:assert/strict';
import {test} from 'node:test';
import {readFileSync} from 'node:fs';
import {createRequire} from 'node:module';
import {dirname, join, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {ValidationWorkspace} from '../web/validation-ui.js';
import {installHelp} from '../web/help.js';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const require = createRequire(import.meta.url);
let JSDOM;
try { ({JSDOM} = require('jsdom')); }
catch { ({JSDOM} = require(join(process.env.LINEAR_LAB_NODE_MODULES || join(root, '..', 'build-tools', 'node_modules'), 'jsdom'))); }
const fixture = JSON.parse(readFileSync(join(root, 'tests/fixtures/qa-fixture.json'), 'utf8'));

function setup() {
  const dom = new JSDOM(`<!doctype html><html><body>
    <select id="model"><option value="ridge">Ridge</option><option value="sgd">SGD</option><option value="ols">OLS</option></select>
    <select id="target"><option value="y">y</option></select>
    <div id="features"><input type="checkbox" checked value="x1"><input type="checkbox" checked value="x2"></div>
    <input id="train-share" value="60"><input id="shuffle" type="checkbox" checked>
    <select id="ranking-metric"><option value="rmse">RMSE</option><option value="r2">R²</option><option value="custom">Своя</option></select>
    <select id="ranking-direction"><option value="auto">auto</option><option value="min">min</option><option value="max">max</option></select>
    <input id="custom-metric" value=""><div id="mount"></div><div id="cv-results"></div><div id="search-results"></div>
    </body></html>`, {url: 'http://localhost:8765/', pretendToBeVisual: true});
  const {window} = dom, {document} = window;
  const plots = [], errors = [], changes = [], searches = [];
  window.Plotly = {
    async react(target, traces, layout, config) { plots.push({id: target.id, traces: structuredClone(traces), layout, config}); return target; },
    purge() {},
  };
  const workspace = new ValidationWorkspace({getRequest: () => {throw new Error('getCVConfig must not recurse through requestFor');}, onChange: () => changes.push(true), onSearch: () => searches.push(true), onError: error => errors.push(error)});
  workspace.mount(document.getElementById('mount'));
  workspace.setCatalogue({...fixture.catalogue, lessons: [{id: '26-cross-validation'}, {id: '27-hyperparameter-search'}, {id: '20-metrics-experiment'}]});
  workspace.setDataset({...fixture.dataset, rows: 90, columns: [...fixture.dataset.columns, {name: 'patient_group', numeric: false, unique: 9, missing: 0}]});
  const $ = id => document.getElementById(id);
  const change = (element, value, type = 'change') => {
    if (value !== undefined) {if (element.type === 'checkbox') element.checked = value; else element.value = String(value);}
    element.dispatchEvent(new window.Event(type, {bubbles: true}));
  };
  const selectModel = id => {change($('model'), id); return fixture.catalogue.models.find(item => item.id === id);};
  const cleanup = () => { workspace.destroy(); window.close(); };
  return {dom, window, document, workspace, $, change, selectModel, plots, errors, changes, searches, cleanup};
}

test('default CV is off; all nine strategies and real model parameter schemas are available', () => {
  const s = setup();
  try {
    assert.deepEqual(s.workspace.getCVConfig(), {strategy: 'none'});
    assert.equal(s.$('vw-cv-method').options.length, 10);
    assert.equal(s.workspace.getParallelism(), 1);
    assert.equal(s.document.querySelectorAll('[data-vw-param]').length, fixture.catalogue.models.find(model => model.id === 'ridge').params.length);
    assert.deepEqual(s.workspace.getSearchConfig().param_space.alpha, [.001, .01, .1, 1, 10, 100]);
    s.$('vw-search-start').click();
    assert.equal(s.searches.length, 1);
    assert.equal(s.errors.length, 0);
  } finally { s.cleanup(); }
});

test('time validation sends exact gap/window/horizon and preserves chronology', () => {
  const s = setup();
  try {
    s.change(s.$('vw-cv-method'), 'timeseries');
    s.change(s.$('vw-cv-folds'), 3);
    s.change(s.$('vw-cv-gap'), 2);
    s.change(s.$('vw-cv-window'), 12);
    s.change(s.$('vw-cv-horizon'), 6);
    assert.deepEqual(s.workspace.getCVConfig(), {strategy: 'timeseries', folds: 3, shuffle: false, gap: 2, max_train_size: 12, test_size: 6});
    assert.equal(s.$('shuffle').checked, false);
    assert.equal(s.document.querySelector('[data-vw-field="time"]').hidden, false);
    assert.equal(s.document.querySelector('[data-vw-field="group"]').hidden, true);
    assert.ok(s.$('vw-cv-description').textContent.includes('по времени'));
  } finally { s.cleanup(); }
});

test('group and repeat CV require valid split-only groups and bounded fit counts', () => {
  const s = setup();
  try {
    s.change(s.$('vw-cv-method'), 'group_kfold');
    const config = s.workspace.getCVConfig();
    assert.equal(config.group_column, 'patient_group');
    assert.equal(config.shuffle, true);
    assert.equal([...s.$('vw-cv-group').options].some(item => item.value === 'y'), false);
    s.change(s.$('vw-cv-group'), '');
    assert.throws(() => s.workspace.getCVConfig(), /идентификатором группы/);
    s.change(s.$('vw-cv-method'), 'repeated_kfold');
    s.change(s.$('vw-cv-folds'), 20);
    s.change(s.$('vw-cv-repeats'), 10);
    assert.throws(() => s.workspace.getCVConfig(), /100 обучений/);
    assert.ok(s.$('vw-cv-budget').classList.contains('vw-invalid'));
  } finally { s.cleanup(); }
});

test('range editor builds a real logarithmic grid and preserves typed input across getSearchConfig', () => {
  const s = setup();
  try {
    const row = s.document.querySelector('[data-vw-param="alpha"]');
    s.change(row.querySelector('[data-vw-values]'), '0.002; 0.4; 7');
    assert.deepEqual(s.workspace.getSearchConfig().param_space.alpha, [.002, .4, 7]);
    assert.deepEqual(s.workspace.getSearchConfig().param_space.alpha, [.002, .4, 7]);
    s.change(row.querySelector('[data-vw-space-kind]'), 'range');
    s.change(row.querySelector('[data-vw-low]'), .001);
    s.change(row.querySelector('[data-vw-high]'), 100);
    s.change(row.querySelector('[data-vw-log]'), true);
    s.change(s.$('vw-search-method'), 'grid');
    assert.deepEqual(s.workspace.getSearchConfig().param_space.alpha, [.001, Number(Math.sqrt(.1).toPrecision(12)), 100]);
    s.change(s.$('vw-search-method'), 'optuna_tpe');
    assert.deepEqual(s.workspace.getSearchConfig().param_space.alpha, {type: 'float', low: .001, high: 100, log: true});
    assert.equal(s.errors.length, 0);
  } finally { s.cleanup(); }
});

test('search rejects invalid numeric text, oversized grids, fit budgets and incompatible halving', () => {
  const s = setup();
  try {
    const row = s.document.querySelector('[data-vw-param="alpha"]');
    s.change(row.querySelector('[data-vw-values]'), '0.1; rubbish; 1');
    assert.throws(() => s.workspace.getSearchConfig(), /значение должно быть числом/);
    s.change(row.querySelector('[data-vw-values]'), '0.1; 1; 10');
    s.change(s.$('vw-search-method'), 'grid');
    s.change(s.$('vw-search-trials'), 2);
    assert.throws(() => s.workspace.getSearchConfig(), /3 комбинаций/);
    s.change(s.$('vw-search-method'), 'random');
    s.change(s.$('vw-search-trials'), 100);
    s.change(s.$('vw-cv-method'), 'kfold');
    s.change(s.$('vw-cv-folds'), 10);
    assert.throws(() => s.workspace.getSearchConfig(), /1000 обучений/);
    s.change(s.$('vw-search-method'), 'halving_random');
    s.change(s.$('vw-cv-method'), 'timeseries');
    assert.throws(() => s.workspace.getSearchConfig(), /для KFold/);
  } finally { s.cleanup(); }
});

test('metric choices synchronize root ranking and custom search respects explicit direction', () => {
  const s = setup();
  try {
    s.change(s.$('vw-search-metric'), 'r2');
    assert.equal(s.$('ranking-metric').value, 'r2');
    assert.equal(s.workspace.getSearchConfig().direction, 'max');
    s.change(s.$('vw-search-metric'), 'custom');
    assert.throws(() => s.workspace.getSearchConfig(), /задайте формулу/);
    s.$('custom-metric').value = '-mean(abs(error))';
    s.change(s.$('vw-search-direction'), 'max');
    const request = s.workspace.getSearchConfig();
    assert.equal(request.metric, 'custom');
    assert.equal(request.direction, 'max');
    assert.equal(s.$('ranking-direction').value, 'max');
    assert.equal(s.$('vw-custom-note').hidden, false);
  } finally { s.cleanup(); }
});

test('graphical preset changes marked parameters; bool/category/range/history round-trip without JSON', () => {
  const s = setup();
  try {
    s.document.querySelector('[data-vw-space-preset="extended"]').click();
    assert.equal(s.document.querySelector('[data-vw-param="fit_intercept"] [data-vw-enabled]').checked, true);
    s.workspace.setConfig({model: 'ridge', cv_config: {strategy: 'repeated_kfold', folds: 4, repeats: 3}, n_jobs: 2,
      search: {method: 'optuna_tpe', trials: 7, metric: 'r2', direction: 'max', param_space: {alpha: {type: 'float', low: .02, high: 2, log: true}, fit_intercept: [false], solver: ['svd', 'cholesky']}}});
    assert.deepEqual(s.workspace.getCVConfig(), {strategy: 'repeated_kfold', folds: 4, repeats: 3});
    const restored = s.workspace.getSearchConfig();
    assert.equal(restored.n_jobs, 2);
    assert.equal(restored.trials, 7);
    assert.deepEqual(restored.param_space, {alpha: {type: 'float', low: .02, high: 2, log: true}, fit_intercept: [false], solver: ['svd', 'cholesky']});
    s.workspace.setBusy(true);
    assert.equal(s.$('vw-search-start').disabled, true);
    s.workspace.setBusy(false);
    assert.equal(s.$('vw-search-start').disabled, false);
    s.selectModel('sgd');
    assert.equal(s.document.querySelector('[data-vw-param="max_iter"] [data-vw-enabled]').disabled, true);
    assert.equal(s.document.querySelector('[data-vw-param="tol"] [data-vw-enabled]').disabled, true);
    assert.deepEqual(Object.keys(s.workspace.getSearchConfig().param_space), ['alpha']);
    s.workspace.reset();
    assert.deepEqual(s.workspace.getCVConfig(), {strategy: 'none'});
    assert.equal(s.workspace.getParallelism(), 1);
  } finally { s.cleanup(); }
});

test('CV results show actual mean/std/fold scores and only returned training indices', () => {
  const s = setup();
  try {
    const cv = {kind: 'kfold', note: 'Настоящие разрезы обучения', summary: {rmse: {mean: 2, std: 1, valid_folds: 2}, r2: {mean: null, std: null, valid_folds: 0}},
      folds: [{fold: 1, train_size: 2, validation_size: 2, metrics: {rmse: 1, r2: null}, metric_details: {r2: {reason: 'Ответы постоянны'}}, seconds: .01, train_indices: [0, 1], validation_indices: [2, 3]}, {fold: 2, train_size: 2, validation_size: 2, metrics: {rmse: 3, r2: null}, metric_details: {}, seconds: .02, train_indices: [2, 3], validation_indices: [0, 1]}]};
    s.workspace.renderResults({cv});
    assert.ok(s.$('cv-results').textContent.includes('Разброс (σ)'));
    assert.ok(s.$('cv-results').textContent.includes('0 / 2'));
    assert.ok(!s.$('cv-results').querySelector('pre'));
    assert.deepEqual(s.plots.find(plot => plot.id === 'vw-cv-score-plot').traces[0].y, [1, 3]);
    s.change(s.$('vw-cv-result-metric'), 'r2');
    assert.ok(s.$('vw-cv-fold-table').textContent.includes('Ответы постоянны'));
    const details = s.document.querySelector('.vw-split-details');
    details.open = true;
    details.dispatchEvent(new s.window.Event('toggle'));
    const map = s.plots.find(plot => plot.id === 'vw-cv-split-plot');
    assert.deepEqual(map.traces[0].z, [[0, 0, 1, 1], [1, 1, 0, 0]]);
    assert.deepEqual(map.traces[0].x, [0, 1, 2, 3]);
  } finally { s.cleanup(); }
});

test('search plots use real trial events, deduplicate polling and break halving winner lines by stage', () => {
  const s = setup();
  try {
    const trials = [{trial: 1, params: {alpha: .1}, score: 8, std: .2, best_score: 8, status: 'complete', iteration: 0, resources: 10, seconds: .01}, {trial: 2, params: {alpha: 1}, score: 5, std: .3, best_score: 5, status: 'complete', iteration: 0, resources: 10, seconds: .02}, {trial: 3, params: {alpha: 1}, score: 6, std: .1, best_score: 6, status: 'complete', iteration: 1, resources: 30, seconds: .03}];
    const events = trials.map((trial, index) => ({message: `Готово ${index + 1}`, search_trial: trial, search_curve: trials.slice(0, index + 1)}));
    s.workspace.setBusy(true);
    s.workspace.renderProgress(events);
    s.workspace.renderProgress(events);
    assert.equal(s.workspace.progressRecords.length, 3);
    assert.equal(s.$('vw-live').textContent, 'Готово 3');
    const curve = s.plots.at(-1);
    assert.deepEqual(curve.traces[0].y, [8, 5, 6]);
    assert.deepEqual(curve.traces[1].y, [8, 5, null, 6]);
    s.workspace.renderResults({search: {method: 'halving_grid', metric: 'rmse', direction: 'min', trials, best_score: 6, best_trial: 3, best_params: {alpha: 1}, folds: 3}});
    assert.equal(s.$('vw-live').hidden, true);
    assert.equal(s.document.querySelectorAll('#vw-search-trial-table tbody tr').length, 3);
    assert.ok(s.document.querySelector('#vw-search-trial-table .vw-best-row').textContent.includes('3'));
    assert.ok(s.$('vw-search-curve-note').textContent.includes('последнем'));
  } finally { s.cleanup(); }
});

test('help provides keyboard access and correct instruction links for CV and search', () => {
  const s = setup();
  const opened = [];
  const help = installHelp(s.document, {openLesson: id => opened.push(id)});
  try {
    const trigger = s.document.querySelector('[data-help-for="vw-cv-method"]');
    assert.ok(trigger);
    trigger.focus();
    const panel = s.document.querySelector('.help-popover:not([hidden])');
    assert.ok(panel.textContent.includes('перекрестная проверка'));
    panel.querySelector('a').click();
    assert.deepEqual(opened, ['26-cross-validation']);
    s.document.querySelector('[data-help-for="vw-search-method"]').click();
    const searchPanel = s.document.querySelector('.help-popover:not([hidden])');
    assert.ok(searchPanel.querySelector('a').href.endsWith('27-hyperparameter-search'));
    s.document.dispatchEvent(new s.window.KeyboardEvent('keydown', {key: 'Escape'}));
    assert.equal(s.document.querySelector('.help-popover:not([hidden])'), null);
  } finally { help.destroy(); s.cleanup(); }
});
