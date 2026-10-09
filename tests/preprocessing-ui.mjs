/** GUI scenarios for real preprocessing requests; the Plotly renderer is substituted. */
import assert from 'node:assert/strict';
import {test} from 'node:test';
import {createRequire} from 'node:module';
import {join,dirname} from 'node:path';
import {fileURLToPath} from 'node:url';
import {PreprocessingWorkspace} from '../web/preprocessing-ui.js';

const require=createRequire(import.meta.url);
let JSDOM;
try{({JSDOM}=require('jsdom'));}catch{({JSDOM}=require(join(dirname(fileURLToPath(import.meta.url)),'..','..','build-tools','node_modules','jsdom')));}

function setup(preview={}){
  const dom=new JSDOM('<body><main id="workspace"></main></body>',{url:'http://localhost/',pretendToBeVisual:true});
  const {window}=dom;
  Object.assign(globalThis,{window,document:window.document});
  window.HTMLDialogElement.prototype.showModal=function(){this.setAttribute('open','');};
  window.HTMLDialogElement.prototype.close=function(){this.removeAttribute('open');};
  const plots=[],requests=[],changes=[],errors=[];
  window.Plotly={async react(node,traces,layout){plots.push({name:node.dataset.pwPlot,traces,layout});node.data=traces;},purge(){}};
  const dataset={id:'dataset-one',rows:300,columns:[{name:'area',numeric:true},{name:'age',numeric:true},{name:'price',numeric:true}],preview:[{area:99,age:11,price:1000}]};
  let workspace;
  const request=()=>({dataset_id:dataset.id,target:'price',features:['area','age'],model:'ridge',split:{train:.6,validation:.2,test:.2},preprocessing:{scale:true,impute:true,degree:1,...workspace.getConfig().preprocessing},resampling:workspace.getConfig().resampling});
  workspace=new PreprocessingWorkspace({api:async(path,options)=>{requests.push({path,...options});return preview;},getRequest:request,onChange:config=>changes.push(config),onError:error=>errors.push(error)});
  workspace.mount(window.document.getElementById('workspace'));workspace.setDataset(dataset);
  const change=(key,value)=>{const field=workspace.field(key);if(field.type==='checkbox')field.checked=value;else field.value=value;field.dispatchEvent(new window.Event('change',{bubbles:true}));};
  return {workspace,dom,plots,requests,changes,errors,change};
}

test('a beginner can choose a recipe, inspect its exact settings, and reset legacy-compatible defaults',()=>{
  const {workspace,change,changes}=setup();
  assert.deepEqual(workspace.getConfig(),{preprocessing:{},resampling:{}});
  change('preset','robust');
  const config=workspace.getConfig();
  assert.equal(config.preprocessing.imputation,'median');assert.equal(config.preprocessing.scaler,'robust');assert.deepEqual(config.preprocessing.clip_quantiles,[.01,.99]);
  assert.match(workspace.container.querySelector('.pw-preset-note').textContent,/1% и 99%/);
  assert.ok(workspace.container.querySelector('[aria-label="Пояснение: Как изменить масштабы"]'));
  assert.equal(changes.length,1);
  workspace.reset();assert.deepEqual(workspace.getConfig(),{preprocessing:{},resampling:{}});
  workspace.setConfig({preprocessing:{scale:false,impute:false,degree:3}});
  assert.equal(workspace.field('scaler').value,'none');assert.equal(workspace.field('imputation').value,'none');assert.equal(workspace.field('degree').value,'3');
  assert.deepEqual(workspace.getConfig().preprocessing,{degree:3});
  workspace.destroy();
});

test('missing indicators and selection remain compatible, while regression sampling exposes its genuine parameters',()=>{
  const {workspace,change}=setup();
  change('missing_indicator',true);change('imputation','none');
  assert.equal(workspace.getConfig().preprocessing.missing_indicator,false);
  assert.equal(workspace.field('missing_indicator').disabled,true);
  change('selection','mutual_info');change('max_features','4');
  assert.equal(workspace.getConfig().preprocessing.max_features,4);
  change('selection','none');assert.equal(workspace.getConfig().preprocessing.max_features,null);
  change('method','smogn');change('focus','high');change('neighbors','3');change('perturbation','.03');change('undersample',false);
  assert.deepEqual(workspace.getConfig().resampling,{method:'smogn',focus:'high',neighbors:3,perturbation:.03,undersample:false});
  assert.equal(workspace.container.querySelector('[data-pw-section="smogn"]').hidden,false);
  workspace.setBusy(true);assert.equal(workspace.field('method').disabled,true);assert.equal(workspace.container.querySelector('[data-pw-action="preview"]').disabled,true);
  workspace.destroy();
});

test('preview uses actual paired train rows, original indices, numeric target colours, and 3D axes',async()=>{
  const response={fitted_on:'train',train_rows:180,original_features:['area','age'],features:['area','age','area^2'],rows:[[-1,.5,1],[1,-.5,1]],indices:[127,3],original_rows:[{area:20,age:6},{area:80,age:2}],means:[0,0,1],std:[1,.5,0],target:{name:'price',values:[40,150]},note:'Первые обучающие строки. Модель не обучалась.',sampling:{method:'random_over',before:180,after:220,target_before:[40,150],target_after:[40,150,150],synthetic:false}};
  const {workspace,change,requests,plots,errors}=setup(response);
  change('preset','nonlinear');change('method','random_over');
  await workspace.showPreview();
  assert.equal(requests[0].path,'/datasets/dataset-one/preprocessing-preview');assert.equal(requests[0].body.preprocessing.degree,2);assert.equal(requests[0].body.resampling.method,'random_over');
  const distribution=plots.find(plot=>plot.name==='distribution');
  assert.deepEqual(distribution.traces[0].x,[20,80]);assert.deepEqual(distribution.traces[1].x,[-1,1]);
  const scatter=plots.find(plot=>plot.name==='scatter');
  assert.deepEqual(scatter.traces[0].x,[-1,1]);assert.deepEqual(scatter.traces[0].y,[40,150]);assert.deepEqual(scatter.traces[0].marker.color,[40,150]);assert.deepEqual(scatter.traces[0].customdata,[[127,40],[3,150]]);
  change('preview-mode','3');change('preview-y','age');change('preview-z','__target__');await workspace.renderPreviewPlots();
  const last=plots.filter(plot=>plot.name==='scatter').at(-1);
  assert.equal(last.traces[0].type,'scatter3d');assert.deepEqual(last.traces[0].y,[.5,-.5]);assert.deepEqual(last.traces[0].z,[40,150]);
  const sampling=plots.find(plot=>plot.name==='sampling');assert.deepEqual(sampling.traces[1].x,[40,150,150]);
  assert.match(workspace.container.querySelector('.pw-preview-stats').textContent,/180 → 220/);
  assert.deepEqual(errors,[]);workspace.destroy();
});

test('a changed dataset suppresses a stale preprocessing result',async()=>{
  const {workspace,errors}=setup();let complete;
  workspace.api=()=>new Promise(resolve=>{complete=resolve;});
  const pending=workspace.showPreview();
  workspace.setDataset({id:'new-data',columns:[],preview:[]});
  complete({features:['old'],rows:[[1]],train_rows:1});await pending;
  assert.equal(workspace.preview,null);assert.equal(workspace.container.querySelector('.pw-preview').open,false);assert.deepEqual(errors,[]);workspace.destroy();
});
