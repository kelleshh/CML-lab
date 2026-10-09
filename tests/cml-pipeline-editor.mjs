/** Browser journeys use the actual Python declarative parser and sklearn source service.
 * DOM/layout are jsdom; HTTP is replaced by a persistent JSON-lines subprocess.
 */
import assert from 'node:assert/strict';
import { test, after } from 'node:test';
import { createRequire } from 'node:module';
import { spawn } from 'node:child_process';
import { createInterface } from 'node:readline';
import { fileURLToPath } from 'node:url';
import { resolve } from 'node:path';
import { readFileSync } from 'node:fs';
const require = createRequire(import.meta.url);
const { JSDOM } = require('jsdom');
const root = fileURLToPath(new URL('..', import.meta.url));
const bridge = spawn(process.env.CML_LAB_PYTHON || resolve(root, '.venv/bin/python'), ['-u', '-c', `
import json,sys
from cml_lab.infrastructure.ml.pipelines import describe_pipeline,pipeline_catalogue,inspect_pipeline_source
for line in sys.stdin:
    try:
        item=json.loads(line)
        if item['path']=='/cml/pipelines/catalogue': result=pipeline_catalogue()
        elif item['path']=='/cml/pipelines/validate': result=describe_pipeline(item['body'])
        elif item['path'].startswith('/cml/pipelines/source?class='):
            from urllib.parse import unquote
            result=inspect_pipeline_source(unquote(item['path'].split('=',1)[1]))
        else: raise ValueError('Unexpected request '+item['path'])
        print(json.dumps({'result':result},ensure_ascii=False),flush=True)
    except Exception as exc:
        print(json.dumps({'error':str(exc)},ensure_ascii=False),flush=True)
`], { cwd: root, env: { ...process.env, ORT_DISABLE_TELEMETRY: '1' } });
const pending = [], requests = [];
let stderr = '';
bridge.stderr.on('data', value => { stderr += value; });
createInterface({ input: bridge.stdout }).on('line', line => {
  const next = pending.shift(); if (!next) return;
  try { const response = JSON.parse(line); response.error ? next.reject(new Error(response.error)) : next.resolve(response.result); }
  catch (error) { next.reject(new Error(`${error.message}: ${line}`)); }
});
bridge.on('exit', code => { while (pending.length) pending.shift().reject(new Error(`Python bridge exited ${code}: ${stderr}`)); });
const api = { request(path, options = {}) { requests.push({ path, ...structuredClone(options) }); return new Promise((resolve, reject) => { pending.push({ resolve, reject }); bridge.stdin.write(`${JSON.stringify({ path, body: options.body })}\n`); }); } };
const dom = new JSDOM('<!doctype html><body></body>', { url: 'http://localhost/', pretendToBeVisual: true });
const { window } = dom;
Object.assign(globalThis, { window, document: window.document, Node: window.Node });
window.HTMLDialogElement.prototype.showModal = function () { this.open = true; };
window.HTMLDialogElement.prototype.close = function () { this.open = false; };
const { renderPipelineEditor, highlightPipelinePython, pipelineCompletions } = await import('../web/cml/pipeline-editor.js');
const { pipelineGraphData, pipelineColumnsLabel } = await import('../web/cml/pipeline-graph.js');
const { PreparationBuilder } = await import('../web/cml/preparation.js');
const editors = [];
after(() => { editors.forEach(editor => editor.destroy()); bridge.stdin.end(); bridge.kill(); dom.window.close(); });
const initial = `from sklearn.pipeline import Pipeline\nfrom sklearn.compose import ColumnTransformer\nfrom sklearn.preprocessing import StandardScaler, OneHotEncoder\nfrom sklearn.impute import SimpleImputer\npipeline = Pipeline([('features', ColumnTransformer([('numeric', Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), ['age','income']),('categorical',OneHotEncoder(handle_unknown='ignore'),['city'])],remainder='drop'))])`;
async function editor(options = {}) { const target = document.createElement('div'); document.body.append(target); const controller = renderPipelineEditor(target, { api, source: initial, columns: ['age', 'income', 'city'], ...options }); editors.push(controller); const result = await controller.ready; assert.ok(result, target.textContent); return controller; }
async function waitFor(predicate, label) { const limit = Date.now() + 10000; while (!predicate()) { if (Date.now() > limit) assert.fail(`Timeout ${label}; ${stderr}`); await new Promise(resolve => setTimeout(resolve, 5)); } }
const button = (node, label) => { const target = [...node.querySelectorAll('button')].find(item => item.textContent === label); assert.ok(target, label); return target; };
const change = (node, value) => { assert.ok(node); node.value = value; node.dispatchEvent(new window.Event('change', { bubbles: true })); };
function select(controller, path) { const graphNode = [...controller.host.querySelectorAll('.cml-pipeline-node')].find(item => item.dataset.nodePath === JSON.stringify(path)); assert.ok(graphNode, JSON.stringify(path)); graphNode.dispatchEvent(new window.MouseEvent('click', { bubbles: true })); }
function key(controller, key, properties = {}) { const event = new window.KeyboardEvent('keydown', { key, bubbles: true, cancelable: true, ...properties }); controller.textarea.dispatchEvent(event); return event; }

test('Python produces real nested sklearn branches and explicit feature routes', async () => {
  const subject = await editor();
  assert.equal(subject.getTree().steps[0].transformer.type, 'ColumnTransformer');
  const data = pipelineGraphData(subject.getTree());
  assert.ok(data.edges.some(edge => edge.label === 'age, income'));
  assert.ok(data.edges.some(edge => edge.label === 'city'));
  assert.ok(data.nodes.some(node => node.label === 'Concatenate features'));
  assert.ok(data.nodes.some(node => node.label === 'remainder: drop'));
  assert.deepEqual([pipelineColumnsLabel({ $tuple: ['age', 'income'] }), pipelineColumnsLabel({ $array: [true, false, true], dtype: 'bool' })], ['age, income', 'true, false, true']);
  assert.ok(subject.isValid());
  assert.equal(subject.getSpec().format, 'cml.pipeline');
  assert.match(subject.getSource(), /ColumnTransformer/);
});

test('GUI parameter edits update actual compiled Python and exact parameter lesson', async () => {
  const changes = []; const subject = await editor({ onChange: value => changes.push(value) });
  select(subject, [0, 0, 1]);
  const control = subject.host.querySelector('[data-parameter=with_mean]');
  assert.equal(control.type, 'checkbox'); control.checked = false; control.dispatchEvent(new window.Event('change', { bubbles: true }));
  await waitFor(() => subject.isValid() && changes.length === 1, 'with_mean applied');
  assert.equal(subject.getTree().steps[0].transformer.transformers[0].transformer.steps[1].transformer.params.with_mean, false);
  assert.match(subject.getSource(), /with_mean=False/);
  const link = subject.host.querySelector('[data-parameter=with_mean]').parentElement.querySelector('.cml-help-button');
  assert.equal(new URL(link.href).searchParams.get('id'), 'parameter-pipeline-standardscaler-with-mean');
  assert.equal(link.target, '_blank');
  assert.equal(changes[0].spec.source, subject.getSource());
  subject.host.querySelector('[aria-label="with_mean: использовать default sklearn"]').click();
  await waitFor(() => subject.isValid() && changes.length === 2, 'default restored');
  assert.equal(subject.getTree().steps[0].transformer.transformers[0].transformer.steps[1].transformer.params.with_mean, true);
  assert.equal(subject.host.querySelector('[aria-label="with_mean: использовать default sklearn"]').disabled, true);
});

test('nested GUI constructors add and remove a branch, preserving existing selectors', async () => {
  const subject = await editor(); select(subject, [0]);
  change(subject.host.querySelector('[aria-label="New transformer class"]'), 'Pipeline');
  button(subject.host, 'Добавить step').click();
  await waitFor(() => subject.isValid() && subject.getTree().steps[0].transformer.transformers.length === 3, 'nested branch');
  const branches = subject.getTree().steps[0].transformer.transformers;
  assert.deepEqual(branches[0].columns, ['age', 'income']); assert.deepEqual(branches[1].columns, ['city']);
  assert.equal(branches[2].transformer.type, 'Pipeline');
  assert.equal(branches[2].transformer.steps[0].transformer.type, 'FunctionTransformer');
  select(subject, [0, 2]);
  button(subject.host, 'Добавить step').click();
  await waitFor(() => subject.isValid() && subject.getTree().steps[0].transformer.transformers[2].transformer.steps.length === 2, 'nested step');
  assert.equal(subject.getTree().steps[0].transformer.transformers[2].transformer.steps[1].transformer.type, 'StandardScaler');
  button(subject.host, 'Удалить step').click();
  await waitFor(() => subject.isValid() && subject.getTree().steps[0].transformer.transformers[2].transformer.steps.length === 1, 'step removed');
});

test('invalid Python keeps text and last valid graph, blocks save and GUI overwrite', async () => {
  const subject = await editor(); const before = subject.getTree(); const text = 'import os\npipeline = os.system("echo impossible")';
  subject.textarea.value = text; subject.textarea.dispatchEvent(new window.Event('input'));
  button(subject.host, 'Применить Python').click();
  await waitFor(() => !subject.host.querySelector('.cml-pipeline-errors').hidden && subject.host.dataset.busy === 'false', 'parser rejection');
  assert.equal(subject.getSource(), text); assert.deepEqual(subject.getTree(), before); assert.equal(subject.isValid(), false);
  assert.equal(button(subject.host, 'Сохранить Pipeline').disabled, true);
  const count = requests.length; select(subject, [0, 0, 1]);
  const control = subject.host.querySelector('[data-parameter=with_mean]'); control.checked = false; control.dispatchEvent(new window.Event('change'));
  assert.equal(requests.length, count); assert.equal(subject.getSource(), text);
  assert.match(subject.host.querySelector('.cml-pipeline-errors').textContent, /непримененные изменения/);
});

test('JSON export preserves an invalid source draft instead of silently exporting the previous Pipeline', async () => {
  const subject = await editor(); const draft = 'pipeline = StillWriting('; subject.textarea.value = draft; subject.textarea.dispatchEvent(new window.Event('input'));
  const originalURL = URL.createObjectURL, originalClick = window.HTMLAnchorElement.prototype.click; let exported;
  URL.createObjectURL = blob => { exported = blob; return originalURL(blob); };
  window.HTMLAnchorElement.prototype.click = function () {};
  try {
    button(subject.host, 'Export .json').click(); const config = JSON.parse(await exported.text());
    assert.equal(config.format, 'cml.pipeline'); assert.equal(config.source, draft); assert.equal(subject.isValid(), false);
  } finally { URL.createObjectURL = originalURL; window.HTMLAnchorElement.prototype.click = originalClick; }
});

test('keyboard order controls and dragging graph nodes change actual sequence', async () => {
  const subject = await editor();
  const list = subject.host.querySelector('[data-step-path="[0,0,1]"]');
  list.querySelector('[aria-label="scale: переместить выше"]').click();
  await waitFor(() => subject.isValid() && subject.getTree().steps[0].transformer.transformers[0].transformer.steps[0].name === 'scale', 'keyboard reorder');
  const from = [...subject.host.querySelectorAll('.cml-pipeline-node')].find(item => item.dataset.nodePath === '[0,0,0]');
  const to = [...subject.host.querySelectorAll('.cml-pipeline-node')].find(item => item.dataset.nodePath === '[0,0,1]');
  from.dispatchEvent(new window.MouseEvent('pointerdown', { button: 0, clientX: 10, clientY: 10, bubbles: true }));
  to.dispatchEvent(new window.MouseEvent('pointermove', { button: 0, clientX: 30, clientY: 80, bubbles: true }));
  to.dispatchEvent(new window.MouseEvent('pointerup', { button: 0, clientX: 30, clientY: 80, bubbles: true }));
  await waitFor(() => subject.isValid() && subject.getTree().steps[0].transformer.transformers[0].transformer.steps[0].name === 'impute', 'graph reorder');
  assert.match(subject.getSource(), /impute[\s\S]*scale/);
});

test('Tab and CtrlSpace complete actual class and parameter names; Escape permits leaving editor', async () => {
  const subject = await editor(); subject.textarea.value = 'pipeline = StandardSc'; subject.textarea.setSelectionRange(subject.textarea.value.length, subject.textarea.value.length);
  key(subject, 'Tab'); assert.equal(subject.getSource(), 'pipeline = StandardScaler()');
  subject.textarea.value = 'pipeline = StandardScaler(with_m'; subject.textarea.setSelectionRange(subject.textarea.value.length, subject.textarea.value.length);
  key(subject, 'Tab'); assert.equal(subject.getSource(), 'pipeline = StandardScaler(with_mean=');
  subject.textarea.value = 'pipeline = StandardScaler('; subject.textarea.setSelectionRange(subject.textarea.value.length, subject.textarea.value.length);
  key(subject, ' ', { code: 'Space', ctrlKey: true });
  assert.equal(subject.host.querySelector('[role=listbox]').hidden, false);
  assert.match(subject.host.querySelector('[role=listbox]').textContent, /with_mean/);
  key(subject, 'ArrowDown'); assert.ok(subject.textarea.getAttribute('aria-activedescendant'));
  key(subject, 'Escape'); assert.equal(subject.host.querySelector('[role=listbox]').hidden, true);
  assert.equal(key(subject, 'Tab').defaultPrevented, false);
});

test('context menu opens installed sklearn source, keyboard closes the modal', async () => {
  const subject = await editor();
  const node = [...subject.host.querySelectorAll('.cml-pipeline-node')].find(item => item.dataset.nodePath === '[0,0,1]');
  node.dispatchEvent(new window.MouseEvent('contextmenu', { bubbles: true, cancelable: true }));
  await waitFor(() => subject.host.querySelector('dialog pre')?.textContent.includes('class StandardScaler'), 'installed source');
  assert.match(subject.host.querySelector('dialog').textContent, /sklearn.*preprocessing/);
  const dialog = subject.host.querySelector('dialog'); dialog.dispatchEvent(new window.Event('cancel', { cancelable: true }));
  assert.equal(subject.host.querySelector('dialog'), null);
});

test('imported Python and saved CML JSON round-trip with nested FeatureUnion and tuples', async () => {
  const subject = await editor();
  const source = 'from sklearn.pipeline import FeatureUnion,Pipeline\nfrom sklearn.preprocessing import MinMaxScaler,PolynomialFeatures\npipeline=FeatureUnion([("plain",MinMaxScaler(feature_range=(0,1))),("poly",Pipeline([("polynomial",PolynomialFeatures(degree=2,include_bias=False))]))])';
  const upload = subject.host.querySelector('input[type=file]');
  Object.defineProperty(upload, 'files', { configurable: true, value: [{ name: 'project.py', text: async () => source }] }); upload.dispatchEvent(new window.Event('change'));
  await waitFor(() => subject.isValid() && subject.getTree().type === 'FeatureUnion', 'Python import');
  assert.deepEqual(subject.getTree().transformer_list[0].transformer.params.feature_range, { $tuple: [0, 1] });
  const saved = subject.getSpec(); const copy = await editor({ source: saved });
  assert.deepEqual(copy.getTree(), subject.getTree());
  Object.defineProperty(upload, 'files', { configurable: true, value: [{ name: 'project.json', text: async () => JSON.stringify(saved) }] }); upload.dispatchEvent(new window.Event('change'));
  await waitFor(() => subject.isValid(), 'JSON import'); assert.equal(subject.getSpec().format, 'cml.pipeline');
});

test('Python highlighting is inert and completion parameters come from real constructor signatures', async () => {
  const catalog = await api.request('/cml/pipelines/catalogue');
  const result = pipelineCompletions('StandardScaler(with_', 'StandardScaler(with_'.length, catalog.classes);
  assert.deepEqual(result.items.filter(item => item.kind === 'parameter').map(item => item.label), ['with_mean', 'with_std']);
  const markup = highlightPipelinePython('# <img src=x onerror=alert(1)>\npipeline = StandardScaler(with_mean=False)');
  assert.doesNotMatch(markup, /<img/); assert.match(markup, /&lt;img/); assert.match(markup, /cml-python-keyword/);
});

test('PreparationBuilder preserves Legacy steps, integrates fullwidth Pipeline and saves the verified spec', async () => {
  const fixture = JSON.parse(readFileSync(new URL('./fixtures/cml-ui.json', import.meta.url), 'utf8'));
  const target = document.createElement('div'); document.body.append(target);
  const saved = [], changes = [];
  const builder = new PreparationBuilder({ api, catalogue: fixture.catalogue, onSavePipeline: value => saved.push(value), onChange: () => changes.push(true), onError: assert.fail }).mount(target);
  builder.setDataset(fixture.datasets.regression);
  assert.equal(builder.isDeclarative(), false); assert.deepEqual(builder.values(), { steps: [], resampling: { method: 'none' } });
  builder.setConfig({ declarative_pipeline: { format: 'cml.pipeline', version: 1, source: initial } }); await builder.pipelineEditor.ready;
  await waitFor(() => builder.valid(), 'declarative preparation ready');
  assert.equal(builder.isDeclarative(), true); assert.deepEqual(builder.values().steps, []); assert.equal(builder.values().declarative_pipeline.format, 'cml.pipeline');
  const workspace = target.querySelector('.cml-preparation-pipeline'); assert.ok(workspace.querySelector('.cml-pipeline-editor'));
  assert.equal(workspace.closest('.cml-workspace'), null, 'Python editor must precede narrow sampler/preview columns');
  button(workspace, 'Сохранить Pipeline').click(); assert.equal(saved.length, 1); assert.deepEqual(saved[0].spec, builder.values().declarative_pipeline);
  assert.ok(changes.length > 0); builder.destroy();
});

test('invalid Pipeline drafts survive task, dataset and mode switches and never become training values', async () => {
  const target = document.createElement('div'); document.body.append(target); const errors = [];
  const builder = new PreparationBuilder({ api, onError: error => errors.push(error) }).mount(target);
  builder.setConfig({ declarative_pipeline: { format: 'cml.pipeline', version: 1, source: initial } }); await builder.pipelineEditor.ready;
  const invalid = 'pipeline = unknown_transformer('; const old = builder.currentSpec();
  builder.pipelineEditor.textarea.value = invalid; builder.pipelineEditor.textarea.dispatchEvent(new window.Event('input'));
  assert.equal(builder.valid(), false); assert.throws(() => builder.values(), /не применен|ошибку/); assert.equal(builder.draftValues().declarative_pipeline.source, invalid);
  assert.equal(builder.currentSpec().source, old.source);
  builder.setDataset({ id: 'new-data', columns: [{ name: 'age', numeric: true }] }); await builder.pipelineEditor.ready;
  assert.equal(builder.pipelineEditor.getSource(), invalid); assert.equal(builder.valid(), false);
  builder.setTask('classification'); await builder.pipelineEditor.ready; assert.equal(builder.pipelineEditor.getSource(), invalid);
  const selector = target.querySelector('[aria-label="Preprocessing editor mode"]'); change(selector, 'legacy'); assert.equal(builder.valid(), true); assert.equal(builder.isDeclarative(), false);
  change(target.querySelector('[aria-label="Preprocessing editor mode"]'), 'declarative'); await builder.pipelineEditor.ready;
  assert.equal(builder.pipelineEditor.getSource(), invalid); assert.throws(() => builder.values(), /Pipeline Python/); assert.ok(errors.length >= 1); builder.destroy();
});

test('a declaration restored from a draft updates pipeline parameters while GUI values remain separate from sampler', async () => {
  const target = document.createElement('div'); document.body.append(target);
  const builder = new PreparationBuilder({ api, onError: assert.fail }).mount(target);
  builder.setConfig({ declarative_pipeline: { format: 'cml.pipeline', version: 1, source: initial }, pipeline_params: { 'features__numeric__scale__with_std': false }, resampling: { method: 'none' } }); await builder.pipelineEditor.ready;
  await waitFor(() => builder.valid(), 'draft restored');
  const value = builder.values(); assert.equal(value.pipeline_params.features__numeric__scale__with_std, false); assert.equal(value.resampling.method, 'none');
  assert.deepEqual(value.steps, []); assert.equal(builder.draftValues().declarative_pipeline.source, value.declarative_pipeline.source); builder.destroy();
});
