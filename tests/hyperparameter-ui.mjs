/** Model grouping retains every native field and all values while filtering. */
import assert from 'node:assert/strict';
import {test} from 'node:test';
import {createRequire} from 'node:module';
const require = createRequire(import.meta.url);
let JSDOM;
try { ({JSDOM} = require('jsdom')); }
catch { ({JSDOM} = require(`${process.env.CODEX_PRIMARY_RUNTIME_NODE_MODULES}/jsdom`)); }
const dom = new JSDOM('<!doctype html><body></body>', {url:'http://localhost/'});
const {window} = dom;
Object.assign(globalThis, {window, document:window.document, Node:window.Node});
const {ModelBuilder} = await import('../web/cml/models.js');

function modelBuilder() {
  const container = document.createElement('div'); document.body.append(container);
  const params = [
    {key:'n_estimators',label:'Переименованное поле',type:'int',default:100,min:1,advanced:false,help:'Количество деревьев.',lesson_id:'parameter-forest-n-estimators'},
    {key:'max_depth',type:'int',default:3,min:1,advanced:false,help:'Глубина дерева.',lesson_id:'parameter-forest-max-depth'},
    {key:'bootstrap',type:'bool',default:true,advanced:true,help:'Выборка с повторениями.',lesson_id:'parameter-forest-bootstrap'},
    {key:'criterion',type:'select',options:['squared_error','absolute_error'],default:'squared_error',advanced:true,help:'Правило оценки split.',lesson_id:'parameter-forest-criterion'},
    {key:'monotonic_cst',type:'json',default:null,advanced:true,help:'Направления изменения прогноза.',lesson_id:'parameter-forest-monotonic-cst'},
  ];
  const builder = new ModelBuilder({catalogue:{algorithms:[{id:'forest',name:'RandomForestRegressor',family:'forest',tasks:['regression'],params}]},onChange:()=>{},onError:assert.fail}).mount(container);
  return {builder,container};
}

test('basic and advanced groups expose exact parameter names and own lessons', () => {
  const {builder,container} = modelBuilder();
  assert.equal(container.querySelector('[data-parameter-group="basic"]').querySelectorAll('[data-parameter]').length, 2);
  assert.equal(container.querySelector('[data-parameter-group="advanced"]').querySelectorAll('[data-parameter]').length, 3);
  assert.equal(container.querySelector('details').open, false);
  assert.equal(container.querySelector('[data-parameter="n_estimators"]').labels[0].textContent, 'n_estimators');
  assert.equal(container.querySelectorAll('.cml-parameter-advanced .cml-help-button').length, 3);
  assert.deepEqual(builder.values().params, {n_estimators:100,max_depth:3,bootstrap:true,criterion:'squared_error',monotonic_cst:null});
});

test('search opens matching advanced fields without rebuilding or dropping their edited values', () => {
  const {builder,container} = modelBuilder();
  const criterion = container.querySelector('[data-parameter="criterion"]');
  criterion.value='absolute_error'; criterion.dispatchEvent(new window.Event('change'));
  const constraints=container.querySelector('[data-parameter="monotonic_cst"]'); constraints.value='[1, 0, -1]'; constraints.dispatchEvent(new window.Event('input'));
  const search = container.querySelector('input[type="search"]');
  search.value='criterion'; search.dispatchEvent(new window.Event('input'));
  assert.equal(container.querySelector('details').open, true);
  assert.equal(container.querySelector('[data-parameter="criterion"]'), criterion);
  assert.equal(container.querySelector('[data-parameter="n_estimators"]').closest('.cml-field').hidden, true);
  assert.equal(builder.values().params.criterion,'absolute_error');
  assert.deepEqual(builder.values().params.monotonic_cst,[1,0,-1]);
  assert.equal(builder.values().params.n_estimators,100);
  search.value='not_a_parameter'; search.dispatchEvent(new window.Event('input'));
  assert.match(container.querySelector('[role="status"]').textContent,/0 из 5/);
  assert.equal(builder.values().params.criterion,'absolute_error');
  search.value=''; search.dispatchEvent(new window.Event('input'));
  assert.equal(container.querySelector('[data-parameter="criterion"]'),criterion);
});
