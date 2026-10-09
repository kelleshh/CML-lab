/** Focused CML user journeys: help, configurable workspace, model selection and recipe CRUD. */
import assert from 'node:assert/strict';
import {test} from 'node:test';
import {createRequire} from 'node:module';
import {readFileSync} from 'node:fs';
const require = createRequire(import.meta.url);
let JSDOM;
try { ({JSDOM} = require('jsdom')); }
catch { ({JSDOM} = require(`${process.env.CODEX_PRIMARY_RUNTIME_NODE_MODULES}/jsdom`)); }
const dom = new JSDOM('<!doctype html><body><main id="root"></main></body>', {url:'http://localhost/'});
const {window} = dom;
Object.assign(globalThis, {window, document:window.document, Node:window.Node, localStorage:window.localStorage});
const {helpButton} = await import('../web/cml/help.js');
const {RecipeLibrary} = await import('../web/cml/recipes.js');
const {ModelBuilder} = await import('../web/cml/models.js');
const {DataStep} = await import('../web/cml/data.js');
const {SchemaForm} = await import('../web/cml/schema-form.js');
const container = () => { const node=document.createElement('div'); document.body.append(node); return node; };
const change = (node, value) => {node.value=value; node.dispatchEvent(new window.Event('change'));};
const click = (node, text) => {const button=[...node.querySelectorAll('button')].find(button=>button.textContent===text || button.querySelector('strong')?.textContent===text); assert.ok(button, text); button.click();};
const flush = async () => {for(let i=0;i<5;i++) await new Promise(resolve=>setTimeout(resolve,0));};

test('help works on keyboard focus and Escape; click points to the exact lesson in a new tab', () => {
 const node=helpButton({label:'Глубина дерева',help:'Ограничивает количество разделений.',lesson_id:'tree-depth'}); document.body.append(node);
 const link=node.querySelector('a'); link.focus();
 assert.equal(link.getAttribute('href'),'/lesson?id=tree-depth'); assert.equal(link.target,'_blank'); assert.equal(link.rel,'noopener');
 const panel=document.getElementById(link.getAttribute('aria-describedby')); assert.equal(panel.getAttribute('role'),'tooltip'); assert.match(panel.textContent,/количество разделений/);
 link.dispatchEvent(new window.KeyboardEvent('keydown',{key:'Escape',bubbles:true})); assert.equal(node.dataset.dismissed,'true');
 node.dispatchEvent(new window.Event('mouseenter')); assert.equal(node.dataset.dismissed,undefined);
});

test('family filter selects a compatible actual model and survives form rerendering', () => {
 const catalogue={algorithms:[{id:'ridge',name:'Ridge',family:'linear',tasks:['regression'],params:[]},{id:'forest',name:'Лес',family:'trees',tasks:['regression'],params:[{key:'depth',label:'Глубина',type:'int',default:3,min:1,lesson_id:'tree-depth'}]}]};
 const changes=[]; const builder=new ModelBuilder({catalogue,onChange:model=>changes.push(model?.id),onError:assert.fail}).mount(container());
 change(builder.container.querySelector('select'),'trees');
 assert.equal(builder.values().algorithm_id,'forest'); assert.equal(builder.selected().name,'Лес');
 assert.equal(builder.container.querySelector('select').value,'trees'); assert.deepEqual(changes,['forest']);
 assert.ok(builder.container.querySelector('[data-parameter="depth"]'));
});

async function library(kind='model-recipes') {
 const original={id:'saved-1',revision:1,name:'Исходная модель',description:'Проверенные параметры',config:{algorithm_id:'ridge',params:{alpha:7}}};
 const requests=[]; let items=[structuredClone(original)]; let current={algorithm_id:'forest',params:{depth:5}};
 const api={request:async(path,options={})=>{requests.push({path,...options}); if(!options.method)return structuredClone(items); if(options.method==='DELETE'){items=[];return{};} const saved={...original,...structuredClone(options.body),id:options.method==='POST'?'copy-2':original.id,revision:2}; items=[saved];return saved;}};
 const errors=[];const subject=new RecipeLibrary({api,getConfig:()=>structuredClone(current),applyConfig:async(_,config)=>{current=structuredClone(config);},notify:()=>{},onError:error=>errors.push(error)}).mount(container());
 await subject.open(kind); return {subject,requests,original,errors};
}

test('rename preserves saved parameters; explicit replacement writes current constructor configuration', async () => {
 const {subject,requests,original,errors}=await library();
 click(subject.container,original.name); const name=[...subject.container.querySelectorAll('input')].find(node=>node.placeholder==='Шаблон модели'); name.value='Новое название';
 click(subject.container,'Обновить объект'); await flush();
 assert.deepEqual(requests.find(request=>request.method==='PATCH').body.config,original.config);
 subject.container.querySelector('input[type=checkbox]').checked=true;
 click(subject.container,'Обновить объект'); await flush();
 assert.deepEqual(requests.filter(request=>request.method==='PATCH').at(-1).body.config,{algorithm_id:'forest',params:{depth:5}}); assert.deepEqual(errors,[]);
});

test('copy saves the selected configuration even when current constructor has different settings', async () => {
 const {subject,requests,original,errors}=await library(); click(subject.container,original.name); click(subject.container,'Сохранить копию');
 assert.match(subject.container.querySelector('input[required]').value,/копия/); click(subject.container,'Создать из текущих настроек'); await flush();
 const post=requests.find(request=>request.method==='POST'); assert.deepEqual(post.body.config,original.config); assert.match(post.body.name,/копия/); assert.equal(post.body.expected_revision,undefined); assert.deepEqual(errors,[]);
});

test('search survives selecting an entity and removal requires confirmation', async () => {
 const {subject,requests,original}=await library(); const search=subject.container.querySelector('input[type=search]');search.value='Исходная';search.dispatchEvent(new window.Event('input'));
 click(subject.container,original.name);assert.equal(subject.container.querySelector('input[type=search]').value,'Исходная');
 window.confirm=()=>false;click(subject.container,'Удалить');await flush();assert.equal(requests.filter(request=>request.method==='DELETE').length,0);
 window.confirm=()=>true;click(subject.container,'Удалить');await flush();assert.equal(requests.filter(request=>request.method==='DELETE').length,1);assert.equal(subject.selection,null);
});

test('each parameter form control has an associated label and question mark', () => {
 const target=container();const form=new SchemaForm().mount(target,[{key:'count',label:'Число соседей',type:'int',default:5,min:1,help:'Сколько соседних объектов использовать.',lesson_id:'nearest-neighbors'},{key:'scale',label:'Масштабировать',type:'bool',default:true,lesson_id:'05-scaling-correlation'}]);
 assert.equal(form.values().count,5);
 for(const control of target.querySelectorAll('input'))assert.ok(target.querySelector(`label[for="${control.id}"]`));
 assert.equal(target.querySelectorAll('.cml-help-button').length,2);
});

test('active layout has accessible persistent width control, classic palette and reduced-motion support', () => {
 const html=readFileSync(new URL('../web/index.html',import.meta.url),'utf8'); const page=new JSDOM(html);const slider=page.window.document.getElementById('settings-width');
 assert.equal(slider.min,'360');assert.equal(slider.max,'760');assert.ok(page.window.document.querySelector('label[for="settings-width"]'));assert.ok(page.window.document.getElementById(slider.getAttribute('aria-describedby')));
 const css=readFileSync(new URL('../web/cml.css',import.meta.url),'utf8');assert.match(css,/#803748/);assert.match(css,/var\(--settings-width\)/);assert.match(css,/prefers-reduced-motion/);assert.doesNotMatch(css,/#3947a8/);
 page.window.close();
});


test('eight real generated datasets populate target, task, roles and safe feature defaults before training', async () => {
 const fixture=JSON.parse(readFileSync(new URL('./fixtures/cml-ui.json',import.meta.url),'utf8'));
 const plots=[]; globalThis.getComputedStyle=window.getComputedStyle.bind(window);globalThis.Plotly={react:async(node,traces,layout)=>{plots.push({traces,layout});}};
 const errors=[];const changes=[];
 const step=new DataStep({api:{request:async path=>fixture.explorations[path.split('/')[2]]},catalogue:fixture.catalogue,onChange:()=>{},onTaskChange:task=>changes.push(task),onBrowse:()=>{},onEdit:()=>{},onError:error=>errors.push(error)}).mount(container());
 for(const [task,dataset] of Object.entries(fixture.datasets)) {
  step.setDataset(dataset);await flush(); const spec=step.values();assert.equal(spec.task,task);
  assert.equal(spec.target,['clustering','anomaly','reduction'].includes(task)?null:dataset.task_target);
  for(const [role,column] of Object.entries(dataset.roles||{}))assert.equal(spec.roles[role],column);
  const excluded=[dataset.task_target,...(dataset.excluded_features||[]),...Object.values(dataset.roles||{})];
  assert.ok(spec.features.length>0);assert.ok(spec.features.every(feature=>!excluded.includes(feature)),task);
  assert.ok(step.rawPlot);assert.ok(step.rawPlot.previousElementSibling);
 }
 assert.equal(changes.length,8);assert.ok(plots.length>=8);assert.deepEqual(errors,[]);
 delete globalThis.Plotly;
});


test('nullable choice uses an automatic option; required numeric field rejects a blank value', () => {
 const form=new SchemaForm().mount(container(),[{key:'mode',label:'Режим',type:'select',default:null,options:[{value:null,label:'Автоматически'},{value:'one',label:'Первый'}]},{key:'neighbors',label:'Соседи',type:'int',default:5,min:1}]);
 assert.equal(form.controls.get('mode').control.options[0].value,'');assert.equal(form.values().mode,null);
 form.controls.get('neighbors').control.value='';assert.equal(form.valid(),false);
});

test('all actual catalogue parameter forms have exact existing learning links', () => {
 const fixture=JSON.parse(readFileSync(new URL('./fixtures/cml-ui.json',import.meta.url),'utf8'));
 const lessons=new Set((Array.isArray(fixture.lessons)?fixture.lessons:fixture.lessons.items).map(lesson=>lesson.id));
 for(const descriptor of [...fixture.catalogue.algorithms,...fixture.catalogue.preprocessors,...fixture.catalogue.samplers]) {
  const node=container();const form=new SchemaForm({definitions:fixture.catalogue.algorithms,task:descriptor.tasks?.[0]||descriptor.allowed_tasks?.[0]}).mount(node,descriptor.params||[]);
  for(const link of node.querySelectorAll('.cml-help-button'))assert.ok(lessons.has(new URL(link.href).searchParams.get('id')),`${descriptor.id}: ${link.href}`);
  assert.ok(form.values());node.remove();
 }
});

test('prediction slice chooses only selected numeric features and removes a feature when deselected', async () => {
 const fixture=JSON.parse(readFileSync(new URL('./fixtures/cml-ui.json',import.meta.url),'utf8'));
 globalThis.getComputedStyle=window.getComputedStyle.bind(window);
 const step=new DataStep({api:{request:async()=>fixture.explorations[fixture.datasets.regression.id]},catalogue:fixture.catalogue,onChange:()=>{},onTaskChange:()=>{},onBrowse:()=>{},onEdit:()=>{},onError:assert.fail}).mount(container());
 step.setDataset(fixture.datasets.regression);await flush();
 assert.equal(step.plotY.disabled,true);change(step.plotX,'x2');change(step.plotY,'x3');assert.deepEqual(step.values().plot_features,['x2','x3']);
 const check=step.features.querySelector('input[value="x2"]');check.checked=false;check.dispatchEvent(new window.Event('change',{bubbles:true}));
 assert.equal(step.plotX.value,'');assert.equal(step.plotY.disabled,true);assert.equal(step.values().plot_features,undefined);
});

test('theme rules do not override the user-selected workspace width inherited from the root', () => {
 const css=readFileSync(new URL('../web/cml.css',import.meta.url),'utf8');
 for(const theme of ['light','dark']){
  const block=css.match(new RegExp(`body\\[data-theme=${theme}\\]\\s*\\{([^}]+)\\}`));assert.ok(block);
  assert.doesNotMatch(block[1],/--settings-width\s*:/,`${theme} theme must inherit the adjustable width`);
 }
});
