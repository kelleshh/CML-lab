/**
 * DOM integration of the actual app.js/charts.js, without a browser or canvas.
 * The base response is captured from TrainingService. HTTP and Plotly's renderer
 * are substituted at their public boundaries; model mathematics is tested by
 * pytest, not by synthetic scores used here to exercise GUI sorting.
 * Run: node --test tests/frontend.mjs (jsdom must be installed).
 */

import assert from 'node:assert/strict';
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
const datasets = new Map([fixture.dataset, fixture.classification].filter(Boolean).map(dataset => [dataset.id, structuredClone(dataset)]));
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
  const result = clone(request.search ? fixture.searchJobs[request.search.direction || 'min'].result
    : request.cv_config?.strategy === 'repeated_kfold' ? fixture.cvJob.result : fixture.job.result);
  result.model = request.model;
  result.model_name = fixture.catalogue.models.find(model => model.id === request.model)?.name || request.model;
  if (request.custom_metric && !request.search) {
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
  else if (path === '/api/datasets/library' && method === 'GET') {
    const task = url.searchParams.get('task'), source = url.searchParams.get('source');
    const query = (url.searchParams.get('query') || '').toLocaleLowerCase('ru');
    const items = [...datasets.values()].filter(item => (!task || item.tasks?.includes(task) || item.task === task)
      && (!source || item.source === source)
      && (!query || [item.name, item.description, ...(item.tags || [])].join(' ').toLocaleLowerCase('ru').includes(query)));
    value = {...fixture.library, items, total: items.length};
  }
  else if (path === '/api/datasets/load' && method === 'POST') {
    currentDataset = nextDataset ? clone(nextDataset)
      : body.name === 'load_iris' ? clone(fixture.classification) : clone(fixture.dataset);
    nextDataset = null;
    if (body.kind === 'synthetic') {
      currentDataset.params = {...currentDataset.params, ...body.params};
      currentDataset.generator = body.name;
    }
    datasets.set(currentDataset.id, clone(currentDataset));
    value = currentDataset;
  } else if (/^\/api\/datasets\/[^/]+\/explore$/.test(path) && method === 'GET') {
    const id = path.split('/')[3];
    const base = id === fixture.classification.id ? fixture.classification : fixture.dataset;
    const y = url.searchParams.get('y');
    const captured = fixture.explorations[base.id + ':' + y] || fixture.explorations[base.id + ':' + base.default_target]
      || Object.entries(fixture.explorations).find(([key]) => key.startsWith(base.id + ':'))?.[1];
    value = clone(captured);
    if (base === fixture.dataset) {
      const rows = fixture.allRows.rows;
      for (const axis of ['x', 'y', 'z', 'color']) {
        const column = url.searchParams.get(axis);
        if (column && rows.every(row => Object.hasOwn(row, column))) value.points[axis] = rows.map(row => row[column]);
      }
    }
  } else if (/^\/api\/datasets\/[^/]+\/preprocessing-preview$/.test(path) && method === 'POST') {
    value = fixture.preprocessingPreview;
  } else if (/^\/api\/datasets\/[^/]+\/metadata$/.test(path) && method === 'PATCH') {
    const id = path.split('/')[3];
    const dataset = datasets.get(id);
    value = {...dataset, ...body, parent_id: id, id: (++editSequence).toString(16).padStart(32, '0')};
    datasets.set(value.id, clone(value));
    currentDataset = value;
  } else if (/^\/api\/datasets\/[^/]+\/rows$/.test(path)) {
    if (method === 'GET') {
      const offset = Number(url.searchParams.get('offset') || 0);
      lastOpenedPage = {
        ...clone(fixture.rows), id: currentDataset.id, offset,
        rows: clone(fixture.allRows.rows.slice(offset, offset + 100)),
      };
      value = lastOpenedPage;
    } else if (method === 'PATCH') {
      currentDataset = {...currentDataset, id: (++editSequence).toString(16).padStart(32, '0')};
      datasets.set(currentDataset.id, clone(currentDataset));
      value = currentDataset;
    }
  } else if (/^\/api\/datasets\/[^/]+$/.test(path) && method === 'DELETE') {
    const id = path.split('/')[3];
    datasets.delete(id);value = {id, deleted:true};
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
  else if (/^\/api\/jobs\/[^/]+\/export-capabilities$/.test(path)) value = fixture.exportCapabilities;
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

async function waitFor(predicate, description, timeout = 15000) {
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

export async function initializeApp() {
  await import(pathToFileURL(join(root, 'web', 'app.js')).href);
  await waitFor(() => posts('/api/jobs').length > 0 && !$('run').disabled && $('plot-fit').data?.length, 'initialization');
}
export function closeApp() {
  console.error = originalError;
  dom.window.close();
  assert.deepEqual(errors, [], 'Application and actual chart modules must report no unexpected errors');
}
export {fixture, dom, window, document, requests, plots, jobs, saved, datasets, errors, clone, $, event, posts, waitFor, clickAndWaitForJob};
export function setNextDataset(dataset) {nextDataset = dataset;}
export function getCurrentDataset() {return currentDataset;}
export function getLastOpenedPage() {return lastOpenedPage;}
