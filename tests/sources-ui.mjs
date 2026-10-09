/** Source-selection UX contracts. HTTP and Plotly are external boundaries;
 * this suite does not claim to inspect a real WebGL render. */
import assert from 'node:assert/strict';
import {test} from 'node:test';
import {createRequire} from 'node:module';
import {dirname, join} from 'node:path';
import {fileURLToPath} from 'node:url';
import {DatasetWorkspace} from '../web/datasets-ui.js';

const require = createRequire(import.meta.url);
let JSDOM;
try { ({JSDOM} = require('jsdom')); }
catch { ({JSDOM} = require(join(dirname(fileURLToPath(import.meta.url)), '..', '..', 'build-tools', 'node_modules', 'jsdom'))); }

function setup(api) {
  const dom = new JSDOM('<!doctype html><html lang="ru"><body><main id="workspace"></main></body></html>', {url: 'http://localhost/', pretendToBeVisual: true});
  Object.assign(globalThis, {window: dom.window, document: dom.window.document});
  window.Plotly = {react: async () => {}, purge() {}, Plots: {resize() {}}};
  const errors = [];
  const workspace = new DatasetWorkspace({api, onError: error => errors.push(error)});
  workspace.mount(document.getElementById('workspace'));
  return {dom, workspace, errors, field: key => document.querySelector(`[data-dw="${key}"]`)};
}

test('source and library task filters use actual task IDs and unsupported sources remain visible but blocked', async () => {
  const state = setup(async () => ({items: []}));
  try {
    const validTasks = ['', 'regression', 'classification', 'clustering', 'time_series', 'other'];
    for (const key of ['library-task', 'source-task']) {
      assert.deepEqual([...state.field(key).options].map(item => item.value), validTasks);
    }
    state.workspace.setCatalogue([
      {id: 'make_moons', name: 'make_moons', label: 'Полумесяцы', kind: 'synthetic', task: 'classification', tasks: ['clustering', 'classification'], modality: 'tabular', supported: true},
      {id: 'fetch_lfw_people', name: 'fetch_lfw_people', label: 'Лица', kind: 'fetch', task: 'classification', tasks: ['classification'], modality: 'image', supported: false, reason: 'Требуется обработка изображений'},
    ]);
    state.field('source-task').value = 'clustering';
    state.workspace.renderSources();
    assert.ok(document.querySelector('[data-dw-source="make_moons"]'));
    assert.equal(document.querySelector('[data-dw-source="fetch_lfw_people"]'), null);
    state.field('source-task').value = 'classification';
    state.field('source-support').value = 'all';
    state.workspace.renderSources();
    const unavailable = document.querySelector('[data-dw-source="fetch_lfw_people"]');
    assert.equal(unavailable.disabled, true);
    assert.match(unavailable.closest('article').textContent, /Требуется обработка изображений/);
    assert.equal(document.querySelector('[data-dw-source="make_moons"]').disabled, false);
    await state.workspace.loadSource('fetch_lfw_people');
    assert.equal(state.workspace.dataset, null);
    assert.deepEqual(state.errors, []);
  } finally { state.workspace.destroy(); state.dom.window.close(); }
});

test('clicking the available Friedman 1 source requests five features and enters exploration without training', async () => {
  const calls = [];
  const metadata = {id: '1'.repeat(32), name: 'make_friedman1', rows: 180, task: 'regression', task_target: 'y', default_target: 'y', targets: ['x1', 'x2', 'x3', 'x4', 'x5', 'y'], columns: ['x1', 'x2', 'x3', 'x4', 'x5', 'y'].map(name => ({name, numeric: true, missing: 0, unique: 180})), preview: [], stats: {missing_cells: 0}, source: 'sklearn'};
  const state = setup(async (path, options = {}) => {
    calls.push({path, ...options});
    if (path.startsWith('/datasets/library')) return {items: []};
    if (path === '/datasets/load') {
      // Friedman 1's public contract needs x1 through x5, regardless of UI defaults.
      if (options.body.params.n_features < 5) throw new Error('Friedman 1 requires at least five features');
      return metadata;
    }
    if (path.includes('/explore?')) return {points: {indices: [], x: [], y: [], z: [], color: []}, correlation: {}, profile: []};
    throw new Error(`Unexpected request: ${path}`);
  });
  try {
    state.workspace.setCatalogue([{id: 'make_friedman1', name: 'make_friedman1', kind: 'synthetic', task: 'regression', tasks: ['regression'], supported: true}]);
    await state.workspace.loadSource('make_friedman1');
    await new Promise(resolve => setImmediate(resolve));
    assert.deepEqual(state.errors, []);
    const load = calls.find(item => item.path === '/datasets/load');
    assert.equal(load.body.params.n_features, 5);
    assert.equal(state.workspace.dataset.id, metadata.id);
    assert.equal(document.querySelector('.dw-explorer').hidden, false);
    assert.equal(state.field('explore-target').value, 'y');
    assert.ok(calls.some(item => item.path.includes('/explore?')));
    assert.equal(calls.some(item => item.path.startsWith('/jobs')), false);
  } finally { state.workspace.destroy(); state.dom.window.close(); }
});
