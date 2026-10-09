/**
 * DOM integration of the actual app.js/charts.js, without a browser or canvas.
 * The base response is captured from TrainingService. HTTP and Plotly's renderer
 * are substituted at their public boundaries; model mathematics is tested by
 * pytest, not by synthetic scores used here to exercise GUI sorting.
 * Run: node --test tests/frontend.mjs (jsdom must be installed).
 */

import assert from 'node:assert/strict';
import {after, before, test} from 'node:test';
import {readFileSync, existsSync} from 'node:fs';
import {createRequire} from 'node:module';
import {dirname, join, resolve} from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const require = createRequire(import.meta.url);
let JSDOM;
try {
  ({JSDOM} = require('jsdom'));
} catch {
  const modules = process.env.LINEAR_LAB_NODE_MODULES || join(root, '..', 'build-tools', 'node_modules');
  ({JSDOM} = require(join(modules, 'jsdom')));
}
const fixturePath = process.env.LINEAR_LAB_UI_FIXTURE || [
  join(root, 'tests', 'fixtures', 'qa-fixture.json'),
  join(root, 'web', 'qa-fixture.json'),
].find(existsSync);
assert.ok(fixturePath, 'A captured API fixture is required in tests/fixtures/qa-fixture.json');
const fixture = JSON.parse(readFileSync(fixturePath, 'utf8'));
const dom = new JSDOM(readFileSync(join(root, 'web', 'index.html'), 'utf8'), {
  url: 'http://localhost:8765/',
  pretendToBeVisual: true,
});
const {window} = dom;
const {document} = window;
Object.assign(globalThis, {window, document, localStorage: window.localStorage, FormData: window.FormData});
window.HTMLElement.prototype.scrollIntoView = function () {};
window.HTMLDialogElement.prototype.showModal = function () { this.setAttribute('open', ''); };
window.HTMLDialogElement.prototype.close = function () { this.removeAttribute('open'); };

const requests = [];
const plots = [];
const errors = [];
const jobs = new Map();
const saved = new Map();
const datasets = new Map([[fixture.dataset.id, structuredClone(fixture.dataset)]]);
let nextDataset = null;
let currentDataset = fixture.dataset;
let lastOpenedPage = null;
let sequence = 0;
let editSequence = 1000;
const clone = value => structuredClone(value);
const $ = id => document.getElementById(id);
const event = (element, type = 'change') => element.dispatchEvent(new window.Event(type, {bubbles: true}));
const posts = path => requests.filter(item => item.path === path && item.method === 'POST');

const originalError = console.error;
console.error = (...args) => { errors.push(args.map(String).join(' ')); };

window.Plotly = {
  async react(target, data, layout, config) {
    const element = typeof target === 'string' ? $(target) : target;
    assert.ok(element, 'Plotly must receive an existing chart container');
    element.data = data;
    element.layout = layout;
    if (!element.plotListeners) {
      element.plotListeners = new Map();
      element.on = (name, listener) => element.plotListeners.set(name, listener);
      element.removeListener = (name, listener) => {
        if (element.plotListeners.get(name) === listener) element.plotListeners.delete(name);
      };
      element.emitPlot = (name, detail) => element.plotListeners.get(name)?.(detail);
    }
    plots.push({id: element.id, data, layout, config});
    return element;
  },
  purge(element) { element.data = null; },
  async relayout(element, update) {
    element.layout = {...element.layout, ...clone(update)};
    element.emitPlot?.('plotly_relayout', clone(update));
    return element;
  },
  Plots: {resize() {}},
};

function resultFor(request) {
  const result = clone(fixture.job.result);
  result.model = request.model;
  result.model_name = fixture.catalogue.models.find(model => model.id === request.model)?.name || request.model;
  if (request.custom_metric) {
    // Explicit HTTP scenario values, not an asserted numerical model result.
    const score = request.model === 'ridge' ? 7 : request.model === 'ols' ? 2 : 3;
    for (const part of ['train', 'validation']) {
      result.metrics[part].custom = score;
      result.metric_details[part].custom = {value: score, reason: null, direction: 'min'};
    }
  }
  return result;
}

globalThis.fetch = async (address, options = {}) => {
  const url = new URL(address, window.location.href);
  const path = url.pathname;
  const method = options.method || 'GET';
  const body = typeof options.body === 'string' ? JSON.parse(options.body) : options.body;
  requests.push({path, search: url.search, method, body: clone(body)});
  let value;
  if (path === '/api/catalogue' && method === 'GET') value = fixture.catalogue;
  else if (path === '/api/datasets/load' && method === 'POST') {
    currentDataset = nextDataset ? clone(nextDataset) : clone(fixture.dataset);
    nextDataset = null;
    if (body.kind === 'synthetic') {
      currentDataset.params = {...currentDataset.params, ...body.params};
      currentDataset.generator = body.name;
    }
    datasets.set(currentDataset.id, clone(currentDataset));
    value = currentDataset;
  } else if (/^\/api\/datasets\/[^/]+\/rows$/.test(path)) {
    if (method === 'GET') {
      const offset = Number(url.searchParams.get('offset') || 0);
      lastOpenedPage = {
        ...clone(fixture.rows), id: currentDataset.id, offset,
        rows: clone(fixture.rows.rows.slice(0, Math.min(100, fixture.rows.total - offset))),
      };
      value = lastOpenedPage;
    } else if (method === 'PATCH') {
      currentDataset = {...currentDataset, id: (++editSequence).toString(16).padStart(32, '0')};
      datasets.set(currentDataset.id, clone(currentDataset));
      value = currentDataset;
    }
  } else if (/^\/api\/datasets\/[^/]+$/.test(path) && method === 'GET') {
    currentDataset = clone(datasets.get(path.split('/')[3]));
    value = currentDataset;
  }
  else if (path === '/api/jobs' && method === 'POST') {
    const id = (++sequence).toString(16).padStart(32, '0');
    jobs.set(id, {id, status: 'completed', result: resultFor(body), request: clone(body)});
    value = {id};
  } else if (/^\/api\/jobs\/[^/]+\/reveal-test$/.test(path)) {
    const job = clone(jobs.get(path.split('/')[3]));
    job.result.test_hidden = false;
    job.result.metrics.test = clone(job.result.metrics.validation);
    value = job;
  } else if (/^\/api\/jobs\/[^/]+\/grid$/.test(path)) value = fixture.job.result.prediction_grid;
  else if (/^\/api\/jobs\/[^/]+$/.test(path) && method === 'GET') value = jobs.get(path.split('/')[3]);
  else if (path === '/api/experiments' && method === 'POST') {
    const job = jobs.get(body.job_id);
    assert.ok(job, 'Saving must refer to a real earlier job identifier');
    const id = `experiment-${saved.size + 1}`;
    value = {id, name: body.name, created: 1700000000, job_id: body.job_id, request: job.request, result: job.result};
    saved.set(id, clone(value));
  } else if (path === '/api/experiments' && method === 'GET') {
    value = [...saved.values()].map(item => ({
      id: item.id, name: item.name, created: item.created,
      model: item.result.model_name, metrics: item.result.metrics.validation,
    }));
  } else if (/^\/api\/experiments\/[^/]+$/.test(path) && method === 'GET') value = saved.get(path.split('/')[3]);
  else if (path === '/api/predict' && method === 'POST') value = {predictions: [17]};
  else throw new Error(`Unhandled mock API boundary: ${method} ${path}`);
  assert.notEqual(value, undefined, `Fixture must cover ${method} ${path}`);
  return {ok: true, status: 200, async json() { return clone(value); }};
};

async function waitFor(predicate, description, timeout = 3000) {
  const deadline = Date.now() + timeout;
  while (Date.now() < deadline) {
    if (predicate()) return;
    await new Promise(resolve => setTimeout(resolve, 10));
  }
  assert.fail(`Timed out waiting for ${description}; toast=${$('toast')?.textContent}; errors=${errors.join('; ')}`);
}

async function clickAndWaitForJob(element) {
  const count = posts('/api/jobs').length;
  element.click();
  await waitFor(() => posts('/api/jobs').length > count, 'submitted job');
  await waitFor(() => !$('run').disabled && $('plot-fit').data?.length, 'rendered completed result');
  return posts('/api/jobs').at(-1).body;
}

before(async () => {
  await import(pathToFileURL(join(root, 'web', 'app.js')).href);
  await waitFor(() => posts('/api/jobs').length > 0 && !$('run').disabled && $('plot-fit').data?.length, 'initialization');
});

after(() => {
  console.error = originalError;
  dom.window.close();
  assert.deepEqual(errors, [], 'Application and actual ChartManager must report no unexpected errors');
});

test('initialization fills the model, metric, lesson and dataset catalogues and renders captured data', () => {
  assert.equal($('model').options.length, 29);
  assert.equal($('model-catalogue').querySelectorAll('[data-model-use]').length, 29);
  assert.equal($('metric-options').querySelectorAll('input').length, fixture.catalogue.metrics.filter(m => m.id !== 'custom').length);
  assert.equal($('lesson-list').querySelectorAll('[data-lesson]').length, fixture.catalogue.lessons.length);
  assert.equal($('dataset-catalogue').querySelectorAll('[data-load]').length, fixture.catalogue.datasets.length);
  assert.equal($('model').value, 'ridge');
  assert.equal($('degree').value, '1');
  assert.equal(posts('/api/jobs')[0].body.dataset_id, fixture.dataset.id);
  assert.ok(plots.some(plot => plot.id === 'plot-fit' && plot.data.some(trace => trace.type === 'scattergl')));
  const traces = $('plot-fit').data.filter(trace => trace.mode === 'markers' && trace.name !== 'Предсказания');
  assert.equal(traces.reduce((sum, trace) => sum + trace.x.length, 0), fixture.job.result.plot_data.y.length);
});

test('all metrics, custom formula and changed split reach the actual run request', async () => {
  $('all-metrics').click();
  $('custom-metric').value = 'mean(abs(error))';
  $('train-share').value = 70;
  $('val-share').value = 15;
  event($('train-share'));
  event($('val-share'));
  const submitted = await clickAndWaitForJob($('run'));
  assert.deepEqual(new Set(submitted.metrics), new Set(fixture.catalogue.metrics.filter(m => m.id !== 'custom').map(m => m.id)));
  assert.equal(submitted.custom_metric, 'mean(abs(error))');
  assert.equal(submitted.split.train, 0.7);
  assert.equal(submitted.split.validation, 0.15);
  assert.ok(Math.abs(submitted.split.test - 0.15) < 1e-12);
  assert.equal($('test-share').textContent, '15');
});

test('changing target removes it from features while authoritative class exclusions preserve legitimate names', async () => {
  nextDataset = {
    ...clone(fixture.dataset), source: 'custom', name: 'Measured classroom data',
    columns: [...clone(fixture.dataset.columns),
      {name: 'diagnosis', numeric: false, dtype: 'object', missing: 0},
      {name: 'classroom_size', numeric: true, dtype: 'float64', missing: 0}],
    targets: [...fixture.dataset.targets, 'classroom_size'],
    excluded_features: ['diagnosis'],
  };
  const button = $('dataset-catalogue').querySelector('[data-load="load_diabetes"]');
  assert.ok(button);
  button.click();
  await waitFor(() => $('dataset-info').textContent.includes('Measured classroom'), 'custom contract dataset');
  const names = () => Array.from($('features').querySelectorAll('input')).map(input => input.value);
  assert.ok(names().includes('classroom_size'), 'An ordinary numeric feature is not a class label because of its name');
  assert.ok(!names().includes('diagnosis'));
  assert.ok(!names().includes('y'));
  $('target').value = 'x1';
  event($('target'));
  assert.ok(!names().includes('x1'));
  assert.ok(names().includes('y'));
  const submitted = await clickAndWaitForJob($('run'));
  assert.equal(submitted.target, 'x1');
  assert.ok(!submitted.features.includes('x1'));
  assert.ok(!submitted.features.includes('diagnosis'));
});

test('navigation and 3D chart tab select visible panels and send actual surface traces', async () => {
  document.querySelector('[data-view="models"]').click();
  assert.ok($('view-models').classList.contains('active'));
  assert.equal(window.location.hash, '#models');
  document.querySelector('[data-view="lab"]').click();
  document.querySelector('[data-charts="3d"]').click();
  await waitFor(() => plots.some(plot => plot.id === 'plot-3d'), '3D traces');
  assert.equal(document.querySelector('[data-chart="3d"]').hidden, false);
  assert.equal(document.querySelector('[data-chart="fit"]').hidden, true);
  assert.ok($('plot-3d').data.some(trace => trace.type === 'surface'));
  document.querySelector('[data-charts="all"]').click();
  await waitFor(() => plots.some(plot => plot.id === 'plot-correlation'), 'all charts');
  assert.ok(Array.from(document.querySelectorAll('[data-chart]')).every(panel => !panel.hidden));
});

test('lesson preset resets unrelated user settings to a reproducible simple regression', async () => {
  $('degree').value = 3;
  $('scale').checked = false;
  $('impute').checked = false;
  $('shuffle').checked = false;
  $('cv').value = 5;
  $('learning-curve').checked = true;
  $('permutation-importance').checked = true;
  $('custom-metric').value = 'mean(error**2)';
  document.querySelector('[data-view="learn"]').click();
  document.querySelector('[data-lesson="0"]').click();
  const submitted = await clickAndWaitForJob($('lesson-run'));
  assert.equal(submitted.model, fixture.catalogue.lessons[0].preset.model);
  assert.deepEqual(submitted.preprocessing, {scale: true, impute: true, degree: 1});
  assert.deepEqual(submitted.split, {train: 0.6, validation: 0.2, test: 0.2, shuffle: true});
  assert.equal(submitted.cv, 0);
  assert.equal(submitted.custom_metric, '');
  assert.equal(submitted.learning_curve, false);
  assert.equal(submitted.permutation_importance, false);
  assert.ok($('view-lab').classList.contains('active'));
});

test('custom ranking direction orders comparison results and selects the same best model', async () => {
  for (const input of $('model-catalogue').querySelectorAll('[data-model-compare]')) {
    input.checked = ['ols', 'ridge'].includes(input.dataset.modelCompare);
    event(input);
  }
  $('custom-metric').value = 'mean(abs(error))';
  $('ranking-metric').value = 'custom';
  $('ranking-direction').value = 'max';
  event($('ranking-metric'));
  event($('ranking-direction'));
  const count = posts('/api/jobs').length;
  $('compare').click();
  await waitFor(() => posts('/api/jobs').length === count + 2 && !$('run').disabled, 'two completed comparisons');
  const firstName = $('comparison-table').querySelector('tbody tr td').textContent;
  const ridgeName = fixture.catalogue.models.find(model => model.id === 'ridge').name;
  assert.equal(firstName, ridgeName);
  assert.equal($('model').value, 'ridge');
  assert.ok(posts('/api/jobs').slice(-2).every(item => item.body.metrics.includes('custom')));
  $('ranking-direction').value = 'min';
  event($('ranking-direction'));
  const ordinaryName = fixture.catalogue.models.find(model => model.id === 'ols').name;
  assert.equal($('comparison-table').querySelector('tbody tr td').textContent, ordinaryName);
});

test('opening a tuned comparison restores boolean parameters and saves the selected actual job', async () => {
  $('model').value = 'ridge';
  event($('model'));
  const intercept = $('model-params').querySelector('[data-param="fit_intercept"]');
  intercept.checked = false;
  const count = posts('/api/jobs').length;
  $('tune').click();
  await waitFor(() => posts('/api/jobs').length === count + 9 && !$('run').disabled, 'tuned comparisons');
  const open = $('comparison-table').querySelector('[data-open-job]');
  const expectedJob = open.dataset.openJob;
  open.click();
  await waitFor(() => $('model-params').querySelector('[data-param="fit_intercept"]')?.checked === false, 'restored boolean parameter');
  document.dispatchEvent(new window.KeyboardEvent('keydown', {key: 's', ctrlKey: true, bubbles: true, cancelable: true}));
  assert.ok($('save-dialog').open);
  $('experiment-name').value = 'My selected experiment';
  $('confirm-save').click();
  await waitFor(() => posts('/api/experiments').length > 0, 'saved experiment request');
  const savedRequest = posts('/api/experiments').at(-1).body;
  assert.deepEqual(savedRequest, {job_id: expectedJob, name: 'My selected experiment'});
});

test('point editor patches absolute row indices on a nonzero page without rewriting other rows', async () => {
  document.querySelector('[data-charts="overview"]').click();
  await waitFor(() => $('plot-fit').plotListeners?.has('plotly_click'), 'point editor chart listener');
  $('plot-fit').emitPlot('plotly_click', {points: [{x: 1, y: 2, customdata: 125, data: {name: 'Обучение'}}]});
  await waitFor(() => $('data-dialog').open && lastOpenedPage?.offset === 100, 'editor second page');
  const input = $('data-editor').querySelector('[data-row="25"][data-column="x1"]');
  assert.ok(input);
  input.value = '123.5';
  event(input, 'input');
  $('apply-data').click();
  await waitFor(() => requests.some(item => item.method === 'PATCH'), 'data patch');
  const patch = requests.filter(item => item.method === 'PATCH').at(-1).body;
  assert.equal(patch.changes.length, 1);
  assert.equal(patch.changes[0].index, 125);
  assert.equal(patch.changes[0].values.x1, 123.5);
  assert.deepEqual(patch.additions, []);
  await waitFor(() => !$('data-dialog').open, 'editor closes after update');
});

test('saved experiment restores optional diagnostics and metric parameters as well as the model', async () => {
  const stored = [...saved.values()][0];
  assert.ok(stored);
  const expected = stored.request;
  $('path-enabled').checked = !expected.regularization_path;
  $('learning-curve').checked = !expected.learning_curve;
  $('permutation-importance').checked = !expected.permutation_importance;
  $('metric-quantile').value = 0.2;
  $('metric-power').value = 3.5;
  $('degree').value = 3;
  document.querySelector('[data-view="history"]').click();
  await waitFor(() => $('history-list').querySelector('[data-experiment]'), 'saved history card');
  $('history-list').querySelector('[data-experiment]').click();
  await waitFor(() => $('view-lab').classList.contains('active') && $('plot-fit').data?.length, 'reopened experiment');
  assert.equal($('degree').value, String(expected.preprocessing.degree));
  assert.equal($('path-enabled').checked, expected.regularization_path);
  assert.equal($('learning-curve').checked, expected.learning_curve);
  assert.equal($('permutation-importance').checked, expected.permutation_importance);
  assert.equal(Number($('metric-quantile').value), expected.metric_params.quantile);
  assert.equal(Number($('metric-power').value), expected.metric_params.power);
});

test('moving a point uses original row identity and data coordinates then trains the new dataset version', async () => {
  $('edit-points').checked = true;
  event($('edit-points'));
  await waitFor(() => $('plot-fit').plotListeners?.has('plotly_relayout'), 'point move listener');
  const pointTrace = $('plot-fit').data.find(trace => trace.customdata?.includes(125));
  assert.ok(pointTrace, 'The edited point must be an actual displayed observation');
  const pointIndex = pointTrace.customdata.indexOf(125);
  const patchCount = requests.filter(item => item.method === 'PATCH').length;
  const jobCount = posts('/api/jobs').length;
  $('plot-fit').emitPlot('plotly_click', {points: [{
    x: pointTrace.x[pointIndex], y: pointTrace.y[pointIndex], customdata: 125,
    data: {name: pointTrace.name},
  }]});
  await waitFor(() => $('plot-fit').layout.shapes?.length === 1, 'editable point shape');
  assert.equal(requests.filter(item => item.method === 'PATCH').length, patchCount, 'Selection does not change data');
  const shape = $('plot-fit').layout.shapes[0];
  const rx = (shape.x1 - shape.x0) / 2;
  const ry = (shape.y1 - shape.y0) / 2;
  $('plot-fit').emitPlot('plotly_relayout', {
    'shapes[0].x0': 11 - rx, 'shapes[0].x1': 11 + rx,
    'shapes[0].y0': 22 - ry, 'shapes[0].y1': 22 + ry,
  });
  await waitFor(() => requests.filter(item => item.method === 'PATCH').length > patchCount, 'moved point patch');
  await waitFor(() => posts('/api/jobs').length > jobCount && !$('run').disabled, 'retrained moved point');
  const patch = requests.filter(item => item.method === 'PATCH').at(-1).body;
  assert.deepEqual(patch.changes, [{index: 125, values: {x1: 11, y: 22}}]);
  assert.equal(posts('/api/jobs').at(-1).body.dataset_id, currentDataset.id);
  assert.notEqual(currentDataset.id, fixture.dataset.id, 'Editing creates a new dataset version');
});
