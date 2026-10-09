/** Full CML UI journeys against captured numerical-adapter responses.
 * New declarative runs, parser metadata and export bytes use a real isolated FastAPI application.
 * HTTP and the Plotly renderer are replaced; reported scores are actual adapter results.
 */
import assert from 'node:assert/strict';
import {test,after} from 'node:test';
import {createRequire} from 'node:module';
import {readFileSync} from 'node:fs';
import {spawn} from 'node:child_process';
import {createInterface} from 'node:readline';
import {fileURLToPath} from 'node:url';
import {resolve} from 'node:path';
const require=createRequire(import.meta.url);const {JSDOM}=require('jsdom');
const fixture=JSON.parse(readFileSync(new URL('./fixtures/cml-ui.json',import.meta.url),'utf8'));
const dom=new JSDOM(readFileSync(new URL('../web/index.html',import.meta.url),'utf8'),{url:'http://localhost/',pretendToBeVisual:true});
const {window}=dom;const {document}=window;
Object.assign(globalThis,{window,document,Node:window.Node,localStorage:window.localStorage,location:window.location,FormData:window.FormData,getComputedStyle:window.getComputedStyle.bind(window)});
window.HTMLDialogElement.prototype.showModal=function(){this.open=true;};window.HTMLDialogElement.prototype.close=function(){this.open=false;};window.HTMLElement.prototype.scrollIntoView=function(){};window.confirm=()=>true;
const plots=[];const renderer={react:async(node,traces,layout)=>{node.data=structuredClone(traces);node.layout=structuredClone(layout);node.classList.add('js-plotly-plot');node.on ||=()=>{};plots.push({node,traces,layout});},relayout:async()=>{},restyle:async()=>{},purge:()=>{},Plots:{resize:()=>{}}};
globalThis.Plotly=window.Plotly=renderer;
const requests=[],runs=new Map(),stores=new Map(['projects','model-recipes','preprocessor-recipes','models','experiments'].map(kind=>[kind,[]]));
const metadata=new Map(Object.values(fixture.datasets).map(dataset=>[dataset.id,structuredClone(dataset)]));
const tasksById=new Map(Object.entries(fixture.datasets).map(([task,dataset])=>[dataset.id,task]));
const clone=value=>structuredClone(value);
const root=fileURLToPath(new URL('..',import.meta.url));
const backend=spawn(process.env.CML_LAB_PYTHON||resolve(root,'.venv/bin/python'),['-u','-c',`
import base64,json,sys,tempfile
from fastapi.testclient import TestClient
from cml_lab.presentation.http.api import create_app
fixture=json.load(open('tests/fixtures/cml-ui.json',encoding='utf-8'))
with tempfile.TemporaryDirectory(prefix='cml-full-app-') as directory:
    app=create_app(directory)
    ids={}
    allowed={'task','tasks','task_target','description','tags','generator','params','roles','excluded_features','excluded_targets','source','warnings'}
    for metadata in fixture['datasets'].values():
        payload={'kind':'custom','name':metadata['name'],'rows':fixture['rows'][metadata['id']]['rows'],'target':metadata.get('default_target'),'metadata':{key:value for key,value in metadata.items() if key in allowed}}
        saved=app.state.services.data.load(payload)
        ids[metadata['id']]=saved['id']
    reverse={value:key for key,value in ids.items()}
    def mapped(value,mapping):
        if isinstance(value,list): return [mapped(item,mapping) for item in value]
        if isinstance(value,dict): return {key:mapping.get(item,item) if key=='dataset_id' and isinstance(item,str) else mapped(item,mapping) for key,item in value.items()}
        return value
    with TestClient(app) as client:
        for line in sys.stdin:
            try:
                item=json.loads(line)
                path=item['path']
                options={'json':mapped(item['body'],ids)} if 'body' in item and item['body'] is not None else {}
                response=client.request(item.get('method','GET'),'/api'+path,**options)
                result={'status':response.status_code,'headers':dict(response.headers)}
                if item.get('binary'): result['base64']=base64.b64encode(response.content).decode('ascii')
                else: result['json']=mapped(response.json(),reverse)
                print(json.dumps(result,ensure_ascii=False),flush=True)
            except Exception as error:
                print(json.dumps({'status':500,'json':{'detail':str(error)}}),flush=True)
`],{cwd:root,env:{...process.env,ORT_DISABLE_TELEMETRY:'1'}});
const backendPending=[];let backendStderr='';
backend.stderr.on('data',value=>{backendStderr+=value;});
createInterface({input:backend.stdout}).on('line',line=>{const next=backendPending.shift();if(!next)return;try{next.resolve(JSON.parse(line));}catch(error){next.reject(error);}});
backend.on('exit',code=>{while(backendPending.length)backendPending.shift().reject(new Error(`Backend exited ${code}: ${backendStderr}`));});
const backendRequest=(path,method,body,binary=false)=>new Promise((resolve,reject)=>{backendPending.push({resolve,reject});backend.stdin.write(`${JSON.stringify({path,method,body,binary})}\n`);});
const realRuns=new Set(),downloads=[];
async function realResponse(path,method,body,binary=false){
 const payload=await backendRequest(path,method,body,binary);
 const bytes=payload.base64?Buffer.from(payload.base64,'base64'):null;
 if(binary&&payload.status<400)downloads.push({path,method,body,bytes,headers:payload.headers});
 return {ok:payload.status>=200&&payload.status<300,status:payload.status,headers:new Headers(payload.headers||{}),json:async()=>clone(payload.json),blob:async()=>new Blob([bytes||JSON.stringify(payload.json)],{type:payload.headers?.['content-type']}),text:async()=>bytes?bytes.toString('utf8'):JSON.stringify(payload.json)};
}
window.HTMLAnchorElement.prototype.click=function(){this.dataset.downloaded='true';};
function exploration(id,query) {
 const value=clone(fixture.explorations[id]);const rows=fixture.rows[id].rows;
 for(const axis of ['x','y','z'])value.points[axis]=query.has(axis)?rows.map(row=>row[query.get(axis)]):null;
 const color=query.get('color');value.points.color=color?rows.map(row=>row[color]):null;
 value.points.color_kind=color?metadata.get(id).columns.find(column=>column.name===color)?.numeric?'numeric':'categorical':null;
 return value;
}
globalThis.fetch=async function(address,options={}){
 assert.equal(this,globalThis,'Browser fetch requires the global receiver');
 const url=new URL(address,window.location.href);const path=url.pathname.replace(/^\/api/,'');const method=options.method||'GET';const body=typeof options.body==='string'?JSON.parse(options.body):options.body;requests.push({path,method,body,query:url.searchParams});
 let value;
 if(path==='/cml/catalogue')value=fixture.catalogue;
 else if(path==='/learning/lessons')value=fixture.lessons;
 else if(path==='/datasets/library'){const matched=[...metadata.values()].filter(item=>(!url.searchParams.get('task')||item.tasks.includes(url.searchParams.get('task')))&&(!url.searchParams.get('query')||`${item.name} ${item.description}`.toLowerCase().includes(url.searchParams.get('query').toLowerCase())));const offset=Number(url.searchParams.get('offset')||0),limit=Number(url.searchParams.get('limit')||200);value={items:matched.slice(offset,offset+limit),total:matched.length,offset,limit};}
 else if(path==='/datasets/load'){const task=body.name==='linear'?'regression':body.name;value=fixture.datasets[task];assert.ok(value,`Real fixture ${task}`);}
 else if(/^\/datasets\/[^/]+\/explore$/.test(path))value=exploration(path.split('/')[2],url.searchParams);
 else if(/^\/datasets\/[^/]+\/rows$/.test(path))value=fixture.rows[path.split('/')[2]];
 else if(/^\/datasets\/[^/]+\/metadata$/.test(path)){const id=path.split('/')[2];value={...metadata.get(id),...body};metadata.set(id,value);}
 else if(/^\/datasets\/[^/]+$/.test(path)){const id=path.split('/')[2];if(method==='DELETE'){metadata.delete(id);value={deleted:true};}else value=metadata.get(id);}
 else if(path==='/cml/preprocessing/preview')value=fixture.previews[tasksById.get(body.dataset_id)];
 else if(path==='/cml/pipelines/catalogue')value=fixture.pipeline_catalogue;
 else if(path==='/cml/analysis'&&body.kind==='overview')value=fixture.analyses[body.dataset_id];
 else if(path==='/cml/analysis'||path==='/cml/pipelines/validate'||path==='/cml/pipelines/parameters'||path==='/cml/pipelines/source'||path==='/cml/search/preset')return realResponse(`${path}${url.search}`,method,body);
 else if(path==='/cml/project-exports')return realResponse(`${path}${url.search}`,method,body,true);
 else if(path==='/cml/runs'&&method==='POST'){
  if(body.preprocessing?.declarative_pipeline){const response=await realResponse(path,method,body);const result=await response.json();if(response.ok)realRuns.add(result.id);return response;}
  const task=body.task;const reference=fixture.results[task];assert.ok(reference,`Actual ${task} numerical result exists`);
  assert.equal(body.algorithm_id,reference.hidden.algorithm_id,'Run must request the algorithm used to produce captured scores');
  const id=`run-${runs.size+1}`;value={id};runs.set(id,{id,status:'completed',progress:1,result:clone(reference.hidden),task,request:clone(body)});
  stores.get('models').push({id:`model-${id}`,run_id:id,name:reference.hidden.model_name,task,algorithm_id:body.algorithm_id,revision:1,result:clone(reference.hidden),status:'completed',spec:clone(body)});
 }
 else if(/^\/cml\/runs\/[^/]+/.test(path)&&realRuns.has(path.split('/')[3]))return realResponse(`${path}${url.search}`,method,body,/\/export$/.test(path));
 else if(/^\/cml\/runs\/[^/]+\/export-capabilities$/.test(path)){const run=runs.get(path.split('/')[3]);value=run?fixture.capabilities[run.task]:{formats:[]};}
 else if(/^\/cml\/runs\/[^/]+\/reveal-test$/.test(path)){const run=runs.get(path.split('/')[3]);run.result=clone(fixture.results[run.task].revealed);value={result:run.result};}
 else if(/^\/cml\/runs\/[^/]+\/save$/.test(path)){const run=runs.get(path.split('/')[3]);value={id:`experiment-${run.id}`,run_id:run.id,...body,revision:1,result:clone(run.result)};stores.get('experiments').push(value);}
 else if(/^\/cml\/runs\/[^/]+$/.test(path))value=runs.get(path.split('/')[3]);
 else if(path==='/cml/runs')value={items:[...runs.values()]};
 else if(path==='/cml/models'&&method==='GET'&&realRuns.size){const response=await realResponse(path,method,body);const actual=await response.json();assert.equal(response.ok,true,JSON.stringify(actual));value={items:[...stores.get('models'),...(actual.items||[])]};}
 else if(/^\/cml\/models\/[^/]+$/.test(path)&&realRuns.has(path.split('/')[3]))return realResponse(path,method,body);
 else if(/^\/cml\/(projects|model-recipes|preprocessor-recipes|models|experiments)(\/[^/]+)?$/.test(path)){
  const [, ,kind,id]=path.split('/');const items=stores.get(kind);
  if(method==='GET')value=id?items.find(item=>item.id===id):{items};
  else if(method==='POST'){value={...clone(body),id:`${kind}-${items.length+1}`,revision:1};items.push(value);}
  else if(method==='PATCH'){const item=items.find(item=>item.id===id);Object.assign(item,clone(body),{revision:item.revision+1});value=item;}
  else if(method==='DELETE'){items.splice(items.findIndex(item=>item.id===id),1);value={deleted:true};}
 }
 else throw new Error(`Unhandled HTTP boundary: ${method} ${path}`);
 assert.notEqual(value,undefined,`Missing real response: ${method} ${path}`);
 return {ok:true,status:200,json:async()=>clone(value)};
};
const $=id=>document.getElementById(id);const change=(node,value)=>{node.value=value;node.dispatchEvent(new window.Event('change',{bubbles:true}));};
const click=(node,text)=>{const button=[...node.querySelectorAll('button')].find(button=>button.textContent===text||button.querySelector('strong')?.textContent===text);assert.ok(button,text);button.click();};
const posts=path=>requests.filter(request=>request.path===path&&request.method==='POST');
async function waitFor(predicate,label){const limit=Date.now()+45000;while(!predicate()){if(Date.now()>limit)assert.fail(`Timeout ${label}; ${$('toast')?.textContent||''}; ${backendStderr}`);await new Promise(resolve=>setTimeout(resolve,5));}}
const {application:app}=await import('../web/cml-app.js');
await waitFor(()=>app.catalogue&&app.recipes&&$('learning-workspace').children.length,'application initialization');
after(async()=>{clearTimeout(app.timer);clearTimeout(app.draftTimer);clearTimeout(app.toastTimer);app.results?.tracePlayer?.destroy();app.analysis?.destroy();app.preparation?.destroy();backend.stdin.end();await new Promise(resolve=>{if(backend.exitCode!==null)return resolve();const timer=setTimeout(()=>{backend.kill();resolve();},5000);backend.once('exit',()=>{clearTimeout(timer);resolve();});});dom.window.close();});

test('initial mount shows a real catalogue, empty data view and accessible keyboard wizard',async()=>{
 assert.equal(app.catalogue.algorithms.length,fixture.catalogue.algorithms.length);assert.equal($('run').disabled,true);
 assert.equal($('view-workflow').hidden,false);assert.equal($('step-data').hidden,false);
 const first=document.querySelector('[data-step=data]');first.focus();first.dispatchEvent(new window.KeyboardEvent('keydown',{key:'ArrowRight',bubbles:true}));
 assert.equal(app.step,'analysis');assert.equal(document.activeElement.dataset.step,'analysis');app.showStep('data');
 change($('settings-width'),640);$('settings-width').dispatchEvent(new window.Event('input'));assert.equal(localStorage.getItem('cml-settings-width'),'640');
});

test('dataset library selection sets task and roles, renders honest raw 2D/3D, previews preparation and trains all eight task families',async()=>{
 for(const [task,dataset] of Object.entries(fixture.datasets)){
  app.showView('datasets');await waitFor(()=>document.querySelector(`[data-dw-select="${dataset.id}"]`),`${task} in library`);
  document.querySelector(`[data-dw-select="${dataset.id}"]`).click();await waitFor(()=>app.dataset?.id===dataset.id,`${task} selected`);
  app.showView('workflow');app.showStep('data');await waitFor(()=>app.data.rawPlot?.data?.length,`${task} raw plot`);
  assert.equal(app.data.taskId,task);const spec=app.data.values();assert.equal(spec.target,['clustering','anomaly','reduction'].includes(task)?null:dataset.task_target);
  for(const [role,column] of Object.entries(dataset.roles||{})){assert.equal(spec.roles[role],column);assert.ok(!spec.features.includes(column));}
  const z=dataset.columns.filter(column=>column.numeric)[1]?.name;change(app.data.zAxis,z);await waitFor(()=>app.data.rawPlot?.data?.[0]?.type==='scatter3d',`${task} 3D`);
  assert.ok(app.data.rawPlot.data.every(trace=>trace.z.length>0));
  app.model.setConfig(fixture.results[task].hidden.effective_spec);app.preparation.setConfig(fixture.results[task].hidden.effective_spec.preprocessing);app.validation.setConfig(fixture.results[task].hidden.effective_spec);
  app.showStep('preparation');click($('step-preparation'),'Посмотреть данные после подготовки');await waitFor(()=>app.preparation.previewContainer.querySelector('table'),`${task} preview`);
  assert.ok(app.preparation.previewContainer.querySelectorAll('tbody tr').length>0);
  assert.equal(app.model.form.valid(),true,JSON.stringify([...app.model.form.controls].map(([k,{control,field}])=>[k,control.value,control.validity?.valid,field])));await app.run();assert.equal(app.busy,false);assert.ok(app.result,`${task} run: ${$('toast').textContent}`);assert.equal(app.result.task,task);assert.equal(app.step,'results');
  assert.deepEqual(app.result.evaluations,fixture.results[task].hidden.evaluations);
  const plotData=[...$('step-results').querySelectorAll('.cml-plot')].filter(node=>node.data?.length);assert.ok(plotData.length>0,`${task} result charts`);
  if(!['clustering','anomaly','reduction'].includes(task)){assert.ok(!app.result.evaluations.test||app.result.evaluations.test.hidden);click($('step-results'),'Открыть итоговую проверку');await waitFor(()=>app.result.test_hidden===false,`${task} reveal test`);assert.deepEqual(app.result.evaluations.test,fixture.results[task].revealed.evaluations.test);}
 }
 assert.equal(posts('/cml/runs').length,8);assert.equal(posts('/cml/preprocessing/preview').length,8);
});

test('real SGD trace replay changes displayed steps without sending a new training request',async()=>{
 const count=posts('/cml/runs').length;await app.results.setResult(fixture.sgd,'trace-fixture');
 assert.ok(app.results.tracePlayer);assert.equal(app.results.tracePlayer.trace.length,fixture.sgd.trace.length);
 const player=app.results.tracePlayer;player.slider.value='2';player.slider.dispatchEvent(new window.Event('input'));await waitFor(()=>player.plot.data[0].y.length===2,'two real frames');
 assert.deepEqual(player.plot.data[0].y,fixture.sgd.trace.slice(0,2).map(frame=>frame.train_loss));assert.equal(posts('/cml/runs').length,count);
 player.destroy();
});

test('model and preprocessor recipes complete create/read/update/copy/delete without replacing saved settings',async()=>{
 app.selectDataset(fixture.datasets.regression);app.model.setConfig({algorithm_id:'ridge',params:{alpha:7}});
 for(const kind of ['model-recipes','preprocessor-recipes']){
  app.showView('library');await app.openLibrary(kind);const node=$('recipe-workspace');node.querySelector('input[required]').value=`Проверка ${kind}`;click(node,'Создать из текущих настроек');await waitFor(()=>stores.get(kind).length===1&&!app.recipes.busy&&node.querySelectorAll('button').length>3,`${kind} created`);
  const saved=clone(stores.get(kind)[0]);assert.ok(saved.config);app.model.setConfig({algorithm_id:'ridge',params:{alpha:9}});
  node.querySelector('input[required]').value='Новое имя';click(node,'Обновить объект');await waitFor(()=>stores.get(kind)[0].revision===2&&!app.recipes.busy,`${kind} updated`);assert.deepEqual(stores.get(kind)[0].config,saved.config);
  click(node,'Сохранить копию');click(node,'Создать из текущих настроек');await waitFor(()=>stores.get(kind).length===2&&!app.recipes.busy,`${kind} copied`);assert.deepEqual(stores.get(kind)[1].config,saved.config);
  click(node,'Удалить');await waitFor(()=>stores.get(kind).length===1&&!app.recipes.busy,`${kind} deleted`);
 }
});

test('project save, experiment save and reopening restore the real snapshot',async()=>{
 app.selectDataset(fixture.datasets.classification);app.model.setConfig(fixture.results.classification.hidden.effective_spec);await app.run();
 app.saveDialog('project');$('save-fields').querySelector('input').value='Проект классификации';$('confirm-save').click();await waitFor(()=>stores.get('projects').length===1&&!$('save-dialog').open,'saved project');assert.equal($('save-dialog').open,false);
 app.saveDialog('experiment');$('save-fields').querySelector('input').value='Проверенный эксперимент';$('confirm-save').click();await waitFor(()=>stores.get('experiments').length===1,'saved experiment');
 app.showView('history');await app.history.open('experiments');click($('history-workspace'),'Проверенный эксперимент');click($('history-workspace'),'Открыть результат');await waitFor(()=>app.view==='workflow'&&app.step==='results','reopen experiment');
 assert.deepEqual(app.result.evaluations,fixture.results.classification.hidden.evaluations);
});


test('diagnostic settings survive saved projects and actual reports render measured values with their direction',async()=>{
 app.selectDataset(fixture.datasets.regression);app.validation.setConfig({regularization_path:true,learning_curve:true,permutation_importance:true});
 const request=app.request();assert.equal(request.regularization_path,true);assert.equal(request.learning_curve,true);assert.equal(request.permutation_importance,true);
 const result=fixture.diagnostic_regression;assert.ok(result.diagnostics.learning_curve.points.length>0);
 await app.results.setResult(result,'diagnostic-fixture');
 const titles=[...$('step-results').querySelectorAll('.cml-plot-panel h3')].map(node=>node.textContent);
 assert.ok(titles.includes('Коэффициенты при разных силах штрафа'));assert.ok(titles.includes('Качество при разных силах штрафа'));assert.ok(titles.includes('Качество при разном количестве обучающих строк'));assert.ok(titles.includes('Изменение качества после перемешивания признака'));
 const curve=[...$('step-results').querySelectorAll('.cml-plot-panel')].find(node=>node.querySelector('h3').textContent==='Качество при разном количестве обучающих строк').querySelector('.cml-plot');
 assert.deepEqual(curve.data[1].y,result.diagnostics.learning_curve.points.map(point=>point.validation_score));
 assert.match(curve.layout.yaxis.title,/меньше лучше/);
 const importance=[...$('step-results').querySelectorAll('.cml-plot-panel')].find(node=>node.querySelector('h3').textContent==='Изменение качества после перемешивания признака').querySelector('.cml-plot');
 assert.deepEqual(importance.data[0].x,result.diagnostics.permutation_importance.mean);assert.deepEqual(importance.data[0].error_x.array,result.diagnostics.permutation_importance.std);
});

test('actual export capabilities disable incompatible formats with a visible reason',async()=>{
 const run=[...runs.values()].find(run=>run.task==='classification');await app.results.setResult(run.result,run.id);
 const exportChoice=$('step-results').querySelector('select[aria-label="Формат экспорта"]');
 for(const format of fixture.capabilities.classification.formats){const option=[...exportChoice.options].find(option=>option.value===format.format);assert.ok(option);assert.equal(option.disabled,format.available===false);if(format.reason)assert.equal(option.title,format.reason);}
});

test('dataset constructor exposes the specialized parameters for seven non-linear task generators',async()=>{
 app.showView('datasets');app.datasets.switchTab('create');
 for(const task of ['classification','clustering','ranking','forecasting','panel','anomaly','reduction']){
  const dataset=fixture.datasets[task];change(app.datasets.field('generator'),task);assert.ok(app.datasets.advancedGenerator,`${task} has a schema-driven form`);
  const form=app.datasets.advancedGenerator;form.setValues(dataset.params);
  for(const field of form.schema){const control=form.controls.get(field.key).control;assert.ok(control);assert.ok(form.container.querySelector(`label[for="${control.id}"]`));assert.ok(form.container.querySelector('.cml-help-button'));}
  const count=posts('/datasets/load').length;click($('dataset-workspace'),'Создать и исследовать');await waitFor(()=>posts('/datasets/load').length>count&&!app.datasets.busy,`${task} dataset created`);
  const request=posts('/datasets/load').at(-1).body;assert.equal(request.name,task);assert.equal(request.params.n_samples,dataset.params.n_samples);assert.equal(app.dataset.task,task);
 }
 const taskFilter=app.datasets.field('library-task');for(const task of Object.keys(fixture.datasets))assert.ok([...taskFilter.options].some(option=>option.value===task),`${task} library filter`);
});

test('project Analysis keeps main plots and creates an actual extra histogram only on request',async()=>{
 app.selectDataset(fixture.datasets.regression);app.showView('workflow');app.showStep('analysis');
 await waitFor(()=>app.analysis.overview?.summary.rows===fixture.datasets.regression.rows,'analysis overview');
 assert.ok($('step-analysis').querySelectorAll('.cml-analysis-main .cml-analysis-plot').length>0);
 assert.equal(app.analysis.generated.children.length,0);
 const count=posts('/cml/analysis').length;
 change(app.analysis.kind,'histogram');change(app.analysis.columnControls[0],'x1');change(app.analysis.optionControls.get('bins'),7);
 assert.equal(posts('/cml/analysis').length,count,'changing controls must not eagerly generate plots');
 app.analysis.generateButton.click();await waitFor(()=>app.analysis.generated.querySelector('.cml-analysis-plot')?.data?.length,'actual histogram');
 const request=posts('/cml/analysis').at(-1).body;assert.equal(request.kind,'histogram');assert.deepEqual(request.columns,['x1']);
 const trace=app.analysis.generated.querySelector('.cml-analysis-plot').data[0];
 assert.equal(trace.y.reduce((sum,value)=>sum+value,0),fixture.datasets.regression.rows);
 assert.equal(trace.y.length,7);assert.equal(app.analysis.generated.children.length,1);
 click($('step-analysis'),'Очистить дополнительные графики');assert.equal(app.analysis.generated.children.length,0);
});

test('My Library Dataset tab exposes source cards and selects data back into the project',async()=>{
 app.showView('library');await app.openLibrary('datasets');
 assert.equal($('dataset-library-workspace').hidden,false);assert.equal($('recipe-workspace').hidden,true);assert.equal($('artifact-workspace').hidden,true);
 const card=$('dataset-library-workspace').querySelector(`[data-library-id="${fixture.datasets.regression.id}"]`);assert.ok(card);
 assert.match(card.textContent,new RegExp(`${fixture.datasets.regression.rows} rows`));assert.ok([...card.querySelectorAll('button')].some(button=>button.textContent==='Download .csv'));
 click(card,'Use in project');await waitFor(()=>app.view==='workflow'&&app.step==='data','dataset returned to project');assert.equal(app.dataset.id,fixture.datasets.regression.id);
});

test('Pipeline, Tuning, real Training and native-fetch Serialization complete a project with nested feature routes',async()=>{
 app.selectDataset(fixture.datasets.regression);app.model.setConfig({algorithm_id:'ridge',params:{alpha:1}});
 app.validation.setConfig({...fixture.results.regression.hidden.effective_spec,regularization_path:false,learning_curve:false,permutation_importance:false,search:null});
 app.showStep('preparation');
 const source=`from sklearn.pipeline import Pipeline\nfrom sklearn.compose import ColumnTransformer\nfrom sklearn.preprocessing import StandardScaler\npipeline=Pipeline([('features',ColumnTransformer([('numeric',StandardScaler(),['x1','x2']),('rest','passthrough',['x3'])],remainder='drop'))])`;
 app.preparation.setConfig({declarative_pipeline:{format:'cml.pipeline',version:1,source},resampling:{method:'none'}});
 await app.preparation.pipelineEditor.ready;await waitFor(()=>app.preparation.valid()&&app.validation.pipelineParameters.length>0,'Pipeline editor and tuning metadata');
 assert.ok($('step-preparation').querySelector('.cml-pipeline-svg'));assert.equal($('run').disabled,false);
 const runCount=posts('/cml/runs').length;
 app.preparation.pipelineEditor.textarea.dispatchEvent(new window.KeyboardEvent('keydown',{key:'Enter',ctrlKey:true,bubbles:true,cancelable:true}));
 await waitFor(()=>app.preparation.valid(),'CtrlEnter applies Python');assert.equal(posts('/cml/runs').length,runCount,'app hotkeys must not turn editor Apply into Training');
 click($('step-preparation'),'Сохранить Pipeline');$('save-fields').querySelector('input').value='Directed feature Pipeline';$('confirm-save').click();
 await waitFor(()=>stores.get('preprocessor-recipes').some(item=>item.name==='Directed feature Pipeline')&&!$('save-dialog').open,'Pipeline recipe saved');
 const saved=stores.get('preprocessor-recipes').find(item=>item.name==='Directed feature Pipeline');assert.match(saved.config.declarative_pipeline.source,/ColumnTransformer/);
 app.showStep('validation');const presetCount=posts('/cml/search/preset').length;change(app.validation.breadth,1);
 await waitFor(()=>posts('/cml/search/preset').length>presetCount&&app.validation.searchRows.length>0&&app.validation.searchEnabled.checked,'preset breadth applied');
 assert.ok(app.validation.searchBudget.textContent.includes('CV fits:'));
 const pipelineKey='pipeline__features__numeric__with_std';assert.ok(app.validation.pipelineParameters.some(param=>param.key===pipelineKey));
 app.validation.setConfig({...fixture.results.regression.hidden.effective_spec,regularization_path:false,learning_curve:false,permutation_importance:false,metrics:['rmse','r2'],validation:{strategy:'kfold',folds:3},search:{method:'grid',trials:4,metric:'rmse',direction:'auto',param_space:{alpha:[.1,1],[pipelineKey]:[true,false]}}});
 const prepared=app.request();assert.deepEqual(prepared.search.param_space[pipelineKey],[true,false]);
 await app.run();await waitFor(()=>!app.busy&&app.result?.effective_spec?.preprocessing?.declarative_pipeline&&realRuns.has(app.runId),'actual Pipeline trained');
 assert.equal(app.result.test_hidden,true);assert.equal(app.result.diagnostics.search.trials.length,4);
 assert.ok(app.result.diagnostics.search.trials.every(trial=>trial.status==='complete'&&Number.isFinite(trial.score)));
 assert.ok(Number.isFinite(app.result.evaluations.validation.metrics.rmse));
 assert.equal(typeof app.result.effective_spec.preprocessing.pipeline_params['features__numeric__with_std'],'boolean');
 app.showStep('serialization');assert.equal($('step-serialization').hidden,false);
 const before=downloads.length;click($('step-serialization'),'Download .py');await waitFor(()=>downloads.length===before+1,'real Python download');
 const python=downloads.at(-1);assert.equal(python.method,'POST');assert.match(python.path,/project-exports\?format=py/);assert.match(python.bytes.toString('utf8'),/declarative_pipeline/);assert.match(python.bytes.toString('utf8'),/ColumnTransformer/);
 click($('step-serialization'),'Download .ipynb');await waitFor(()=>downloads.length===before+2,'real notebook download');
 const notebook=JSON.parse(downloads.at(-1).bytes.toString('utf8'));assert.equal(notebook.nbformat,4);assert.ok(notebook.cells.some(cell=>cell.cell_type==='code'&&(Array.isArray(cell.source)?cell.source.join(''):cell.source).includes('declarative_pipeline')));
 click($('step-serialization'),'Download .joblib');await waitFor(()=>downloads.length===before+3,'real fitted artifact download');assert.ok(downloads.at(-1).bytes.length>100);
 app.showView('library');await app.openLibrary('models');const card=$('artifact-workspace').querySelector(`[data-library-id="${app.runId}"]`);assert.ok(card);assert.match(card.textContent,/Validation metrics/);assert.ok(card.querySelector('.cml-help-button'));
});

test('invalid mini IDE source is autosaved as a draft and cannot send a training request',async()=>{
 app.showView('workflow');app.showStep('preparation');const controller=app.preparation.pipelineEditor;assert.ok(controller);const validSource=controller.getSource();
 const invalid='pipeline = Unfinished(';controller.textarea.value=invalid;controller.textarea.dispatchEvent(new window.Event('input',{bubbles:true}));
 assert.equal($('run').disabled,true);assert.equal(app.preparation.valid(),false);
 await waitFor(()=>{try{return JSON.parse(localStorage.getItem('cml-lab-draft')).preprocessing.declarative_pipeline.source===invalid;}catch{return false;}},'raw Python draft autosaved');
 const count=posts('/cml/runs').length;await app.run();assert.equal(posts('/cml/runs').length,count);assert.match($('toast').textContent,/Pipeline/);
 await controller.setSource(validSource);await waitFor(()=>app.preparation.valid(),'valid Python restored');assert.equal($('run').disabled,false);
});
