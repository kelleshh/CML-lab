/** CRUD/card/export journeys against an isolated actual FastAPI application.
 * Only the HTTP transport and DOM layout are replaced; workers/artifact bytes are real.
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
const processBridge = spawn(process.env.CML_LAB_PYTHON || resolve(root, '.venv/bin/python'), ['-u', '-c', `
import json,sys,tempfile
from fastapi.testclient import TestClient
from cml_lab.presentation.http.api import create_app
with tempfile.TemporaryDirectory(prefix='cml-ui-library-') as directory:
    with TestClient(create_app(directory)) as client:
        for line in sys.stdin:
            try:
                item=json.loads(line)
                options={'json':item['body']} if 'body' in item else {}
                response=client.request(item.get('method','GET'),'/api'+item['path'],**options)
                if response.status_code>=400:
                    raise ValueError(str(response.json().get('detail',response.text)))
                if item.get('download'):
                    result={'bytes':len(response.content),'type':response.headers.get('content-type'),'disposition':response.headers.get('content-disposition'),'prefix':response.content[:1500].decode('utf-8','replace')}
                else: result=response.json()
                print(json.dumps({'result':result},ensure_ascii=False),flush=True)
            except Exception as exc:
                print(json.dumps({'error':str(exc)},ensure_ascii=False),flush=True)
`], { cwd: root, env: { ...process.env, ORT_DISABLE_TELEMETRY: '1' } });
const pending = [], requests = [], downloads = [];
let stderr = '';
processBridge.stderr.on('data', value => { stderr += value; });
createInterface({ input: processBridge.stdout }).on('line', line => {
  const next = pending.shift(); if (!next) return;
  try { const response = JSON.parse(line); response.error ? next.reject(new Error(response.error)) : next.resolve(response.result); } catch (error) { next.reject(error); }
});
processBridge.on('exit', code => { while (pending.length) pending.shift().reject(new Error(`API bridge exited ${code}: ${stderr}`)); });
const transport = (path, options = {}, download = false) => new Promise((resolve, reject) => { requests.push({ path, ...structuredClone(options), download }); pending.push({ resolve, reject }); processBridge.stdin.write(`${JSON.stringify({ path, ...options, download })}\n`); });
const api = { request: (path, options) => transport(path, options), download: async (path, name, options) => { const result = await transport(path, options, true); downloads.push({ path, name, result }); return result; } };
const dom = new JSDOM('<!doctype html><body></body>', { url: 'http://localhost/' }); const { window } = dom;
Object.assign(globalThis, { window, document: window.document, Node: window.Node }); window.confirm = () => true;
const { DatasetLibrary } = await import('../web/cml/dataset-library.js'); const { RecipeLibrary } = await import('../web/cml/recipes.js'); const { SavedRuns } = await import('../web/cml/saved-runs.js');
const errors = [], selected = [], edited = []; const notify = () => {};
const container = () => { const node = document.createElement('div'); document.body.append(node); return node; };
const button = (node, label) => { const item = [...node.querySelectorAll('button')].find(item => item.textContent === label || item.querySelector('strong')?.textContent === label); assert.ok(item, label); return item; };
async function waitFor(predicate, label) { const deadline = Date.now() + 45000; while (!predicate()) { if (Date.now() > deadline) assert.fail(`Timeout ${label}. ${errors.at(-1)?.message || ''}; ${stderr}`); await new Promise(resolve => setTimeout(resolve, 10)); } }
const catalogue = await api.request('/cml/catalogue');
let parentDataset = await api.request('/datasets/load', { method: 'POST', body: { kind: 'synthetic', name: 'linear', params: { n_samples: 80, seed: 8 } } });
after(async () => { processBridge.stdin.end(); await new Promise(resolve => { if (processBridge.exitCode !== null) return resolve(); const timer = setTimeout(() => { processBridge.kill(); resolve(); }, 5000); processBridge.once('exit', () => { clearTimeout(timer); resolve(); }); }); dom.window.close(); });

test('dataset cards show actual characteristics, export actual CSV and delegate use/edit', async () => {
  const library = new DatasetLibrary({ api, onSelect: item => selected.push(item), onEdit: item => edited.push(item), notify, onError: error => errors.push(error) }).mount(container());
  await library.open();
  const card = library.container.querySelector(`[data-library-id="${parentDataset.id}"]`); assert.ok(card);
  assert.match(card.textContent, /80 rows/); assert.match(card.textContent, /columns/); assert.match(card.textContent, /missing values/);
  button(card, 'Download .csv').click(); await waitFor(() => downloads.length === 1, 'CSV downloaded');
  assert.ok(downloads[0].result.bytes > 100); assert.match(downloads[0].result.prefix, /x1/);
  button(card, 'Use in project').click(); await waitFor(() => selected.length === 1, 'dataset selected'); assert.equal(selected[0].id, parentDataset.id);
  button(card, parentDataset.name).click(); button(library.container, 'Edit rows').click(); await waitFor(() => edited.length === 1, 'edit callback'); assert.equal(edited[0].id, parentDataset.id);
});

test('dataset metadata edit creates a new immutable snapshot; delete cancellation and deletion preserve parent', async () => {
  const library = new DatasetLibrary({ api, notify, onError: error => errors.push(error) }).mount(container()); await library.open();
  button(library.container, parentDataset.name).click();
  const name = [...library.container.querySelectorAll('input')].find(item => item.required); name.value = 'Renamed dataset';
  button(library.container, 'Сохранить новую версию').click();
  await waitFor(() => !library.busy && library.selected?.name === 'Renamed dataset', 'snapshot created');
  const updated = library.selected; assert.notEqual(updated.id, parentDataset.id); assert.equal(updated.parent_id, parentDataset.id);
  assert.equal((await api.request(`/datasets/${parentDataset.id}`)).name, parentDataset.name);
  window.confirm = () => false; const count = requests.filter(item => item.method === 'DELETE').length;
  button(library.container, 'Удалить dataset').click(); await waitFor(() => !library.busy, 'cancelled delete'); assert.equal(requests.filter(item => item.method === 'DELETE').length, count);
  window.confirm = () => true; button(library.container, 'Удалить dataset').click(); await waitFor(() => !library.busy && library.selected === null, 'dataset deleted');
  assert.equal((await api.request(`/datasets/${parentDataset.id}`)).id, parentDataset.id);
});

test('pipeline cards draw real nested transforms and export .py plus CML format; rename preserves code', async () => {
  const source = 'from sklearn.pipeline import Pipeline\nfrom sklearn.preprocessing import StandardScaler,PolynomialFeatures\npipeline=Pipeline([("scale",StandardScaler()),("poly",PolynomialFeatures(degree=2))])';
  const checked = await api.request('/cml/pipelines/validate', { method: 'POST', body: { source } });
  const item = await api.request('/cml/preprocessor-recipes', { method: 'POST', body: { name: 'Saved pipeline', description: 'Nested preprocessing', config: { steps: [], declarative_pipeline: checked.spec } } });
  const library = new RecipeLibrary({ api, getConfig: () => ({ steps: [] }), applyConfig: () => {}, notify, onError: error => errors.push(error) }).mount(container()); await library.open('pipelines');
  const card = library.container.querySelector(`[data-library-id="${item.id}"]`); await waitFor(() => card.querySelector('.cml-pipeline-svg'), 'real pipeline diagram');
  assert.match(card.textContent, /StandardScaler/); assert.match(card.textContent, /PolynomialFeatures/);
  const before = downloads.length; button(card, 'Download .py').click(); await waitFor(() => downloads.length === before + 1, 'pipeline Python export'); assert.match(downloads.at(-1).result.prefix, /Pipeline/);
  button(card, 'Download .json').click(); await waitFor(() => downloads.length === before + 2, 'pipeline JSON export'); assert.match(downloads.at(-1).result.prefix, /cml.pipeline/);
  button(card, item.name).click(); const name = [...library.container.querySelectorAll('input')].find(item => item.required); name.value = 'Renamed pipeline';
  button(library.container, 'Обновить объект').click(); await waitFor(() => !library.busy && library.selection?.name === 'Renamed pipeline', 'pipeline renamed');
  assert.deepEqual(library.selection.config, item.config);
});

test('project cards export executable Python and ipynb while copies retain the selected project', async () => {
  const spec = { task: 'regression', dataset_id: parentDataset.id, algorithm_id: 'ridge', params: { alpha: 3 }, validation: { strategy: 'none' } };
  const item = await api.request('/cml/projects', { method: 'POST', body: { name: 'Regression project', description: 'Actual dataset', config: spec } });
  const library = new RecipeLibrary({ api, getConfig: () => ({ ...spec, params: { alpha: 999 } }), applyConfig: () => {}, notify, onError: error => errors.push(error) }).mount(container()); await library.open('projects');
  const card = library.container.querySelector(`[data-library-id="${item.id}"]`); const before = downloads.length;
  button(card, 'Download .py').click(); await waitFor(() => downloads.length === before + 1, 'project py'); assert.ok(downloads.at(-1).result.bytes > 100); assert.match(downloads.at(-1).result.prefix, /CML|import|from/);
  button(card, 'Download .ipynb').click(); await waitFor(() => downloads.length === before + 2, 'project notebook'); assert.match(downloads.at(-1).result.prefix, /nbformat|cells/);
  button(card, item.name).click(); button(library.container, 'Сохранить копию').click(); button(library.container, 'Создать из текущих настроек').click();
  await waitFor(() => !library.busy && library.selection?.id !== item.id, 'project copied'); assert.equal(library.selection.config.params.alpha, 3);
});

test('trained model cards show actual validation metrics, export fitted bytes and disable unavailable formats', async () => {
  const started = await api.request('/cml/runs', { method: 'POST', body: { task: 'regression', dataset_id: parentDataset.id, algorithm_id: 'ridge', validation: { strategy: 'none' } } });
  let run; const deadline = Date.now() + 45000;
  do { run = await api.request(`/cml/runs/${started.id}`); if (run.status === 'completed') break; if (['error', 'interrupted', 'cancelled'].includes(run.status)) assert.fail(JSON.stringify(run)); if (Date.now() > deadline) assert.fail('Actual worker timed out'); await new Promise(resolve => setTimeout(resolve, 30)); } while (true);
  const library = new SavedRuns({ api, catalogue, onOpen: () => {}, notify, onError: error => errors.push(error) }).mount(container()); await library.open('models');
  const card = library.container.querySelector(`[data-library-id="${started.id}"]`); assert.ok(card); assert.match(card.textContent, /Validation metrics/); assert.match(card.textContent, /ridge/);
  assert.ok(card.querySelectorAll('.cml-help-button').length > 0);
  const before = downloads.length; button(card, 'Download .joblib').click(); await waitFor(() => downloads.length === before + 1, 'fitted model bytes'); assert.ok(downloads.at(-1).result.bytes > 100);
  card.querySelector('.cml-entity').click(); const formats = library.container.querySelector('.cml-library-export-choices'); formats.open = true; formats.dispatchEvent(new window.Event('toggle'));
  await waitFor(() => formats.querySelector('button'), 'real capabilities');
  const actual = await api.request(`/cml/runs/${started.id}/export-capabilities`);
  for (const format of actual.formats) { const control = [...formats.querySelectorAll('button')].find(item => item.textContent === (format.label || format.format)); assert.ok(control); assert.equal(control.disabled, format.available === false); }
  assert.equal(run.result.test_hidden, true);
});

test('library journeys do not silently report API failures', () => { assert.deepEqual(errors, []); });
