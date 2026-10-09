/** Semantic accessibility scenarios. Layout, physical target sizes and visual
 * rendering still require browser QA; JSDOM does not measure those properties. */
import assert from 'node:assert/strict';
import {test} from 'node:test';
import {readFile} from 'node:fs/promises';
import {createRequire} from 'node:module';
import {dirname, join} from 'node:path';
import {fileURLToPath} from 'node:url';
import {installHelp} from '../web/help.js';
import {renderBeginnerControls} from '../web/beginner.js';
import {DatasetWorkspace} from '../web/datasets-ui.js';
import {PreprocessingWorkspace} from '../web/preprocessing-ui.js';
import {ValidationWorkspace} from '../web/validation-ui.js';

const require = createRequire(import.meta.url);
let JSDOM;
try { ({JSDOM} = require('jsdom')); }
catch { ({JSDOM} = require(join(dirname(fileURLToPath(import.meta.url)), '..', '..', 'build-tools', 'node_modules', 'jsdom'))); }

function setup(markup = '<main id="workspace"></main>') {
  const dom = new JSDOM(`<!doctype html><html lang="ru"><body>${markup}</body></html>`, {url:'http://localhost/', pretendToBeVisual:true});
  const {window} = dom;
  Object.assign(globalThis, {window, document:window.document});
  window.HTMLDialogElement.prototype.showModal = function () { this.setAttribute('open', ''); };
  window.HTMLDialogElement.prototype.close = function () { this.removeAttribute('open'); };
  window.Plotly = {react:async()=>{}, purge(){}, Plots:{resize(){}}};
  return {dom, window, doc:window.document, container:window.document.getElementById('workspace')};
}

function name(control) {
  const direct = control.getAttribute('aria-label');
  if (direct?.trim()) return direct.trim();
  const references = control.getAttribute('aria-labelledby')?.split(/\s+/).map(id => control.ownerDocument.getElementById(id)?.textContent ?? '').join(' ');
  if (references?.trim()) return references.trim();
  return [...control.labels || []].map(label => {
    const copy = label.cloneNode(true);
    copy.querySelectorAll('input,select,textarea,.help-trigger,.dw-help-wrap').forEach(node => node.remove());
    return copy.textContent;
  }).join(' ').trim();
}

function assertNamedFields(scope) {
  for (const field of scope.querySelectorAll('input,select,textarea')) {
    assert.ok(name(field), `Unnamed ${field.tagName} ${field.id || field.dataset.dw || ''}`);
  }
}

test('context help has a named nonmodal dialog, instruction link, and a complete keyboard close path', () => {
  const {doc, window, container, dom} = setup('<main id="workspace"><label>Привести к одному масштабу<input type="checkbox" id="scale"></label></main>');
  const opened = [];
  const layer = installHelp(container, {openLesson:id=>opened.push(id)});
  const trigger = doc.querySelector('.help-trigger');
  assert.match(trigger.getAttribute('aria-label'), /^Пояснение:/);
  trigger.focus();
  const panel = doc.getElementById(trigger.getAttribute('aria-controls'));
  assert.equal(panel.getAttribute('role'), 'dialog');
  assert.equal(panel.getAttribute('aria-modal'), 'false');
  assert.ok(name(panel));
  assert.equal(panel.hidden, false);
  assert.equal(trigger.getAttribute('aria-expanded'), 'true');
  trigger.dispatchEvent(new window.KeyboardEvent('keydown', {key:'ArrowDown', bubbles:true, cancelable:true}));
  assert.equal(doc.activeElement.tagName, 'A');
  assert.match(doc.activeElement.textContent, /инструкции/);
  doc.dispatchEvent(new window.KeyboardEvent('keydown', {key:'Escape', bubbles:true, cancelable:true}));
  assert.equal(panel.hidden, true);
  assert.equal(trigger.getAttribute('aria-expanded'), 'false');
  assert.equal(doc.activeElement, trigger);
  trigger.click();
  assert.equal(doc.getElementById('scale').checked, false, 'Question mark must not activate the enclosing checkbox label');
  panel.querySelector('a').click();
  assert.deepEqual(opened, ['05-scaling-correlation']);
  assert.equal(panel.hidden, true);
  layer.destroy();
  assert.equal(doc.querySelector('.help-popover'), null);
  dom.window.close();
});

test('beginner choices announce their group, selected state and exact numeric value', () => {
  const {doc, container, dom} = setup();
  const changes = [];
  const controls = renderBeginnerControls(container, {id:'ridge',params:[{key:'alpha',label:'Сила штрафа',type:'float',min:0,max:100,default:1}]}, {onChange:patch=>changes.push(patch)});
  const group = container.querySelector('[role="group"]');
  assert.equal(group.getAttribute('aria-label'), 'Сила штрафа');
  const buttons = [...group.querySelectorAll('button')];
  assert.equal(buttons.filter(button=>button.getAttribute('aria-pressed')==='true').length, 1);
  buttons.find(button=>button.textContent==='Сильнее').click();
  assert.deepEqual(changes, [{alpha:10}]);
  assert.equal(buttons.filter(button=>button.getAttribute('aria-pressed')==='true').length, 1);
  assert.match(container.querySelector('[aria-live="polite"]').textContent, /10/);
  controls.setParams({alpha:2});
  assert.equal(buttons.filter(button=>button.getAttribute('aria-pressed')==='true').length, 0);
  assert.match(container.textContent, /своё значение/);
  controls.destroy();
  dom.window.close();
});

function datasetSetup() {
  const state = setup();
  const calls = [];
  const workspace = new DatasetWorkspace({api:async(path)=>{calls.push(path);return path.startsWith('/datasets/library') ? [] : {points:{indices:[],x:[],y:[],z:[],color:[]},correlation:{},profile:[]};}});
  workspace.mount(state.container);
  return {...state, workspace, calls};
}

test('dataset tabs support arrow/home/end navigation and connect to named panels', async () => {
  const {doc, window, workspace, dom} = datasetSetup();
  const tabs = [...doc.querySelectorAll('[role="tab"]')];
  assert.equal(tabs.filter(tab=>tab.tabIndex===0).length, 1);
  for (const tab of tabs) {
    const panel = doc.getElementById(tab.getAttribute('aria-controls'));
    assert.equal(panel.getAttribute('role'), 'tabpanel');
    assert.equal(panel.getAttribute('aria-labelledby'), tab.id);
  }
  tabs[0].focus();
  tabs[0].dispatchEvent(new window.KeyboardEvent('keydown', {key:'ArrowRight', bubbles:true, cancelable:true}));
  assert.equal(doc.activeElement, tabs[1]);
  assert.equal(tabs[1].getAttribute('aria-selected'), 'true');
  assert.equal(doc.getElementById(tabs[1].getAttribute('aria-controls')).hidden, false);
  assert.equal(doc.getElementById(tabs[0].getAttribute('aria-controls')).hidden, true);
  tabs[1].dispatchEvent(new window.KeyboardEvent('keydown', {key:'End', bubbles:true, cancelable:true}));
  assert.equal(doc.activeElement, tabs[2]);
  tabs[2].dispatchEvent(new window.KeyboardEvent('keydown', {key:'Home', bubbles:true, cancelable:true}));
  assert.equal(doc.activeElement, tabs[0]);
  assert.equal(tabs.filter(tab=>tab.tabIndex===0).length, 1);
  workspace.destroy();
  dom.window.close();
});

test('dataset help remains keyboard reachable and Escape hides it while the question mark retains focus', () => {
  const {doc, window, workspace, dom} = datasetSetup();
  const trigger = doc.querySelector('.dw-help');
  const panel = doc.getElementById(trigger.getAttribute('aria-controls'));
  assert.equal(panel.hidden, true);
  trigger.focus();
  assert.equal(panel.hidden, false);
  assert.equal(panel.getAttribute('role'), 'dialog', 'Interactive instruction links must not live inside role=tooltip');
  assert.ok(name(panel));
  assert.equal(trigger.getAttribute('aria-expanded'), 'true');
  trigger.dispatchEvent(new window.KeyboardEvent('keydown', {key:'ArrowDown', bubbles:true, cancelable:true}));
  assert.equal(doc.activeElement, panel.querySelector('a'));
  doc.activeElement.dispatchEvent(new window.KeyboardEvent('keydown', {key:'Escape', bubbles:true, cancelable:true}));
  assert.equal(doc.activeElement, trigger);
  assert.equal(panel.hidden, true);
  assert.equal(trigger.getAttribute('aria-expanded'), 'false');
  trigger.click();
  assert.equal(panel.hidden, false);
  panel.querySelector('.dw-help-close').click();
  assert.equal(panel.hidden, true);
  assert.equal(doc.activeElement, trigger);
  workspace.destroy();
  dom.window.close();
});

test('dataset labels remain associated with fields when help buttons precede selects; role colours also have text', async () => {
  const {doc, container, workspace, dom} = datasetSetup();
  assertNamedFields(container);
  const withHelp = workspace.field('noise-preset');
  assert.equal(withHelp.labels.length, 1);
  assert.equal(withHelp.labels[0].htmlFor, withHelp.id);
  assert.equal(name(withHelp), 'Разброс цели');
  workspace.setDataset({id:'demo',name:'Квартиры',rows:2,task:'regression',default_target:'price',targets:['price'],columns:[{name:'area',numeric:true},{name:'floor',numeric:true},{name:'price',numeric:true}],preview:[{area:30,floor:2,price:80},{area:80,floor:6,price:210}]});
  assertNamedFields(container);
  const legend = doc.querySelector('.dw-role-legend');
  assert.match(legend.textContent, /Признак.*сведения/);
  assert.match(legend.textContent, /Цель.*ответ/);
  assert.equal(name(workspace.field('explore-target')), 'Цель');
  workspace.destroy();
  await Promise.resolve();
  dom.window.close();
});

test('switching the raw-data theme updates readable role text without fetching or changing coordinates', async () => {
  const {doc, window, workspace, calls, dom} = datasetSetup();
  const relayouts = [], restyles = [];
  window.Plotly.react = async(node, traces, layout) => { node.data = traces; node.layout = layout; };
  window.Plotly.relayout = async(node, patch) => relayouts.push({name:node.dataset.dwPlot,patch});
  window.Plotly.restyle = async(node, patch, indices) => restyles.push({name:node.dataset.dwPlot,patch,indices});
  workspace.setDataset({id:'theme-data',name:'Измерения',rows:2,task:'regression',default_target:'price',columns:[{name:'area',numeric:true},{name:'floor',numeric:true},{name:'price',numeric:true}],preview:[]});
  await new Promise(resolve=>setTimeout(resolve,0));
  const scatter = doc.querySelector('[data-dw-plot="scatter2d"]');
  const originalData = JSON.stringify(scatter.data);
  const fetches = calls.length;
  doc.body.dataset.theme = 'light';
  await workspace.refreshTheme();
  assert.equal(calls.length,fetches);
  assert.equal(JSON.stringify(scatter.data),originalData);
  const update = relayouts.find(item=>item.name==='scatter2d').patch;
  assert.equal(update['xaxis.title.font.color'],'#076950');
  assert.equal(update['yaxis.title.font.color'],'#a63d2a');
  assert.ok(contrast(update['font.color'],'#ffffff')>=4.5);
  assert.equal(restyles.find(item=>item.name==='scatter2d').patch['marker.colorbar.title.font.color'],'#a63d2a');
  workspace.destroy();
  dom.window.close();
});

test('preprocessing and validation fields have real labels, live status and named preview dialog', () => {
  const {doc, container, dom} = setup('<main id="workspace"><div id="preprocessing"></div><div id="validation"></div></main>');
  const preprocessing = new PreprocessingWorkspace({api:async()=>({}),getRequest:()=>({model:'ridge',split:{train:.6},features:['area']})});
  preprocessing.mount(doc.getElementById('preprocessing'));
  const validation = new ValidationWorkspace({getRequest:()=>({model:'ridge',split:{train:.6},features:['area']})});
  validation.mount(doc.getElementById('validation'));
  assertNamedFields(container);
  const preview = container.querySelector('dialog');
  assert.match(name(preview), /До и после/);
  assert.equal(container.querySelector('.pw-status').getAttribute('aria-live'), 'polite');
  assert.equal(container.querySelector('#vw-live').getAttribute('role'), 'status');
  const questionMarks = [...container.querySelectorAll('.help-trigger')];
  assert.ok(questionMarks.length > 10);
  assert.ok(questionMarks.every(button=>name(button).length > 10));
  preprocessing.destroy();
  validation.destroy();
  dom.window.close();
});

test('an open preprocessing preview changes theme from recorded values without repeating preparation', async () => {
  const {doc, window, container, dom} = setup();
  const renders = [], calls = [];
  window.Plotly.react = async(node,traces,layout)=>renders.push({name:node.dataset.pwPlot,traces,layout});
  const result = {fitted_on:'train',train_rows:2,features:['area'],original_features:['area'],rows:[[-1],[1]],indices:[4,8],original_rows:[{area:30},{area:80}],means:[0],std:[1],target:{name:'price',values:[90,200]}};
  const workspace = new PreprocessingWorkspace({api:async(path)=>{calls.push(path);return result;},getRequest:()=>({dataset_id:'theme-data',target:'price',features:['area'],preprocessing:{}})});
  workspace.mount(container);
  workspace.setDataset({id:'theme-data',columns:[{name:'area',numeric:true},{name:'price',numeric:true}],preview:[]});
  await workspace.showPreview();
  const first = renders.find(item=>item.name==='scatter');
  doc.body.dataset.theme = 'light';
  await workspace.refreshTheme();
  const current = renders.filter(item=>item.name==='scatter').at(-1);
  assert.equal(calls.length,1);
  assert.deepEqual(current.traces,first.traces);
  assert.equal(current.layout.xaxis.title.font.color,'#076950');
  assert.equal(current.layout.yaxis.title.font.color,'#a63d2a');
  assert.ok(contrast(current.layout.font.color,'#ffffff')>=4.5);
  workspace.destroy();
  dom.window.close();
});

test('the static shell names its active controls, dialogs and chart view selector without relying on placeholders', async () => {
  const html = await readFile(new URL('../web/index.html', import.meta.url), 'utf8');
  const dom = new JSDOM(html);
  const doc = dom.window.document;
  assert.equal(doc.documentElement.lang, 'ru');
  assert.ok(doc.querySelector('meta[name="viewport"]'));
  assert.ok(!/user-scalable\s*=\s*no/.test(doc.querySelector('meta[name="viewport"]').content));
  for (const field of doc.querySelectorAll('input,select,textarea')) {
    if (field.closest('[hidden]')) continue; // Archived forms are intentionally not exposed.
    assert.ok(name(field), `Static field must not use its placeholder as a label: ${field.id}`);
  }
  for (const dialog of doc.querySelectorAll('dialog')) assert.ok(name(dialog));
  const chartSelector = doc.querySelector('.chart-tabs');
  assert.equal(chartSelector.getAttribute('role'), 'group');
  assert.ok(chartSelector.getAttribute('aria-label'));
  assert.equal(chartSelector.querySelectorAll('[aria-pressed="true"]').length, 1);
  dom.window.close();
});

function contrast(foreground, background) {
  const luminance = color => {
    const hex = color.slice(1);
    const channels = [0,2,4].map(i=>parseInt(hex.slice(i,i+2),16)/255).map(value=>value <= .04045 ? value/12.92 : ((value+.055)/1.055)**2.4);
    return .2126*channels[0]+.7152*channels[1]+.0722*channels[2];
  };
  const values = [luminance(foreground),luminance(background)].sort((a,b)=>b-a);
  return (values[0]+.05)/(values[1]+.05);
}

test('both themes provide readable contrast for feature and target text on panel surfaces', async () => {
  const base = await readFile(new URL('../web/style.css', import.meta.url), 'utf8');
  const accessibility = await readFile(new URL('../web/accessibility.css', import.meta.url), 'utf8');
  const dark = base.slice(0,base.indexOf('body[data-theme=light]'));
  const color = variable => dark.match(new RegExp(`--${variable}:([#a-f0-9]+)`))[1];
  for (const foreground of [color('accent'),color('coral'),color('muted')]) {
    for (const background of [color('panel'),color('panel2')]) assert.ok(contrast(foreground,background) >= 4.5, `${foreground} on ${background}`);
  }
  const lightFeature = accessibility.match(/\.pw-feature \{ color: (#[a-f0-9]+) !important; \}/)[1];
  const lightTarget = accessibility.match(/\.pw-target \{ color: (#[a-f0-9]+) !important; \}/)[1];
  for (const foreground of [lightFeature,lightTarget]) {
    for (const background of ['#ffffff','#f3f7f9','#f7fafb']) assert.ok(contrast(foreground,background) >= 4.5, `${foreground} on ${background}`);
  }
});
