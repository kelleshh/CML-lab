import {ChartManager} from './charts.js';

const $ = id => document.getElementById(id);
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const number = value => value == null || !Number.isFinite(Number(value)) ? '—' : new Intl.NumberFormat('ru', {maximumFractionDigits:4}).format(Number(value));
const state = {catalogue:null,dataset:null,result:null,job:null,request:null,busy:false,view:'lab',chartTab:'overview',frame:0,playing:false,selectedModels:new Set(['ols','ridge','lasso','elasticnet','huber','sgd']),comparisons:[],editor:null,lesson:0,revealed:false};
const charts = new ChartManager();
let playbackTimer = null, autoTimer = null, toastTimer = null;

function toast(message) {
  $('toast').textContent=message; $('toast').classList.add('visible');
  clearTimeout(toastTimer); toastTimer=setTimeout(()=>$('toast').classList.remove('visible'),6000);
}
async function api(path, options={}) {
  if(options.body && !(options.body instanceof FormData)) options={...options,body:JSON.stringify(options.body),headers:{'Content-Type':'application/json',...options.headers}};
  const response=await fetch(`/api${path}`,options);
  const data=await response.json().catch(()=>({detail:'Сервер вернул некорректный ответ.'}));
  if(!response.ok) throw new Error(typeof data.detail==='string'?data.detail:JSON.stringify(data.detail));
  return data;
}
function safely(handler) {return async event=>{try{await handler(event);}catch(error){toast(error.message);console.error(error);}};}
function switchView(view) {
  state.view=view;
  document.querySelectorAll('.view').forEach(el=>el.classList.toggle('active',el.id===`view-${view}`));
  document.querySelectorAll('.nav').forEach(el=>el.classList.toggle('active',el.dataset.view===view));
  if(view==='lab'&&state.result) setTimeout(()=>renderCharts(),20);
  if(view==='history') refreshHistory().catch(error=>toast(error.message));
  window.location.hash=view;
}
function selectedModel() {return state.catalogue.models.find(model=>model.id===$('model').value);}
function showModelParameters() {
  const model=selectedModel(); if(!model)return;
  $('model-description').textContent=model.description;
  $('model-params').innerHTML=model.params.map(param=>{
    const description=esc(param.help || '');
    if(param.type==='bool')return `<label class="check" title="${description}"><input type="checkbox" data-param="${esc(param.key)}" ${param.default?'checked':''}>${esc(param.label)}</label>`;
    if(param.type==='select')return `<label title="${description}">${esc(param.label)}<select data-param="${esc(param.key)}">${(param.options||[]).map(option=>{const value=typeof option==='object'?option.value:option;const label=typeof option==='object'?option.label:param.option_labels?.[option]||option;return `<option value="${esc(value)}" ${value===param.default?'selected':''}>${esc(label)}</option>`;}).join('')}</select></label>`;
    const isAlpha=param.key==='alpha';
    return `<label title="${description}">${esc(param.label)}<input type="number" data-param="${esc(param.key)}" min="${param.min??0}" max="${param.max??1e9}" step="${param.step??(param.type==='int'?1:'any')}" value="${param.default??0}"></label>${isAlpha?`<input class="alpha-slider" type="range" min="-5" max="4" step="0.05" value="${Math.log10(param.default||.00001)}" aria-label="Сила регуляризации по логарифмической шкале"><p class="help-text">${description} Крутилка: от 0.00001 до 10000.</p>`:`<p class="help-text">${description}</p>`}`;
  }).join('');
  $('model-params').querySelectorAll('[data-param]').forEach(el=>el.addEventListener('change',()=>scheduleFit()));
  const slider=$('model-params').querySelector('.alpha-slider');
  if(slider)slider.addEventListener('input',()=>{const input=$('model-params').querySelector('[data-param="alpha"]');input.value=Number((10**Number(slider.value)).toPrecision(4));scheduleFit();});
  showExplanation();
}
function modelParameters() {
  const model=selectedModel(), params={};
  $('model-params').querySelectorAll('[data-param]').forEach(el=>{
    const specification=model.params.find(p=>p.key===el.dataset.param);
    params[el.dataset.param]=specification.type==='bool'?el.checked:specification.type==='select'?el.value:Number(el.value);
  });
  return params;
}
function showExplanation() {
  const model=selectedModel(); if(!model)return;
  $('explanation-title').textContent=model.name;
  const explanation=Array.isArray(model.explanation)?model.explanation.join('\n\n'):model.explanation||model.description;
  $('explanation-text').innerHTML=String(explanation).split('\n').filter(Boolean).map(p=>`<p>${esc(p)}</p>`).join('');
  $('objective-formula').textContent=model.formula||'';
  renderFormulas($('objective-formula').parentElement);
}
function renderFormulas(root) {
  if(!window.katex)return;
  root.querySelectorAll('code').forEach(element=>{const expression=element.textContent;if(expression.includes('\\')){try{window.katex.render(expression,element,{throwOnError:false,displayMode:true,strict:false});}catch{element.textContent=expression;}}});
}
function scheduleFit() {if(!$('auto-fit').checked)return;clearTimeout(autoTimer);autoTimer=setTimeout(()=>{if(!state.busy)runModel().catch(error=>toast(error.message));},600);}
function generatedSpec() {return {kind:'synthetic',name:$('dataset-preset').value,params:{n_samples:Number($('n-samples').value),n_features:Number($('n-features').value),noise:Number($('noise').value),correlation:Number($('correlation').value),outliers:Number($('outliers').value),sparsity:Number($('sparsity').value),scale_spread:Number($('scale-spread').value),seed:Number($('seed').value)}};}
async function loadData(spec) {if(state.busy)throw new Error('Дождитесь завершения или остановите текущий расчет.');const dataset=await api('/datasets/load',{method:'POST',body:spec});setDataset(dataset);return dataset;}
function setDataset(dataset) {
  state.dataset=dataset;state.result=null;state.revealed=false;stopPlayback();
  if(dataset.source==='synthetic'&&dataset.params){
    $('dataset-preset').value=dataset.generator;
    const mapping={'n_samples':'n-samples','n_features':'n-features','noise':'noise','correlation':'correlation','outliers':'outliers','scale_spread':'scale-spread','sparsity':'sparsity','seed':'seed'};
    for(const[key,id]of Object.entries(mapping))if(dataset.params[key]!=null){$(id).value=dataset.params[key];if($(id+'-value'))$(id+'-value').value=dataset.params[key];}
  }
  $('dataset-info').textContent=`${dataset.name} · ${dataset.rows.toLocaleString('ru')} строк · ${dataset.columns.length} столбцов`;
  const targets=dataset.targets||dataset.columns.filter(c=>c.numeric).map(c=>c.name);
  $('target').innerHTML=(!dataset.default_target?'<option value="">Выберите числовую цель…</option>':'')+targets.map(name=>`<option value="${esc(name)}" ${name===dataset.default_target?'selected':''}>${esc(name)}</option>`).join('');
  refreshFeatures();
  $('metrics-summary').innerHTML='<div class="metric-card"><div class="metric-name">Данные готовы</div><strong>'+number(dataset.rows)+'</strong><div class="metric-subtitle">Обучите выбранную модель</div></div>';
  $('job-status').textContent='Данные загружены. Настройте модель и запустите обучение.';
  $('warnings').innerHTML='';$('comparison-table').innerHTML='';state.comparisons=[];
  $('prediction-inputs').innerHTML='';
  charts.destroy();
  document.querySelectorAll('.plot').forEach(el=>el.innerHTML='<div class="plot-empty">Обучите модель на выбранных данных.</div>');
  $('timeline').disabled=true;
}
function refreshFeatures() {
  if(!state.dataset)return;
  const target=$('target').value;
  const excluded=state.dataset.excluded_features||[];
  $('features').innerHTML=state.dataset.columns.filter(c=>c.name!==target&&!excluded.includes(c.name)).map(column=>`<label class="check" title="Пропусков: ${column.missing||0}"><input type="checkbox" value="${esc(column.name)}" checked>${esc(column.name)} <span class="help-text">${column.numeric?'число':'категория'}</span></label>`).join('');
  $('features').querySelectorAll('input').forEach(input=>input.addEventListener('change',scheduleFit));
}
function requestFor(model=$('model').value,params=modelParameters()) {
  if(!state.dataset)throw new Error('Сначала загрузите данные.');
  if(!$('target').value)throw new Error('Выберите числовую величину, которую предсказываем.');
  const train=Number($('train-share').value)/100,validation=Number($('val-share').value)/100,test=1-train-validation;
  if(train<=0||validation<=0||test<.049)throw new Error('Оставьте каждой части данные. Итоговая проверка должна занимать хотя бы 5%.');
  const features=Array.from($('features').querySelectorAll('input:checked')).map(input=>input.value);
  if(!features.length)throw new Error('Выберите хотя бы один признак.');
  const metrics=Array.from($('metric-options').querySelectorAll('input:checked')).map(input=>input.value);
  const custom_metric=$('custom-metric').value.trim();
  if(!metrics.length&&!custom_metric)throw new Error('Выберите метрику или задайте свою формулу.');
  return {dataset_id:state.dataset.id,target:$('target').value,features,model,params,seed:Number($('seed').value),split:{train,validation,test,shuffle:$('shuffle').checked},preprocessing:{scale:$('scale').checked,impute:$('impute').checked,degree:Number($('degree').value)},metrics,custom_metric,metric_params:{quantile:Number($('metric-quantile').value),power:Number($('metric-power').value)},epochs:Number($('epochs').value),cv:Number($('cv').value),regularization_path:$('path-enabled').checked,learning_curve:$('learning-curve').checked,permutation_importance:$('permutation-importance').checked};
}
function setBusy(busy) {
  state.busy=busy;$('run').disabled=busy;$('compare').disabled=busy;$('tune').disabled=busy;$('generate').disabled=busy;
  document.querySelectorAll('.control-panel input,.control-panel select,.control-panel button').forEach(element=>element.disabled=busy);
  $('cancel').disabled=false;$('save').disabled=busy;$('upload').disabled=busy;$('openml-load').disabled=busy;$('edit-points').disabled=busy;
  $('cancel').hidden=!busy;$('progress').hidden=!busy;
  if(busy)stopPlayback();
}
function liveTraining(events) {
  const frames=events.filter(event=>event.frame).map(event=>event.frame);
  if(frames.length<2||!window.Plotly)return;
  const isDark=document.body.dataset.theme==='dark';
  const last=frames.at(-1);
  const loss=frames.map(frame=>frame.train_loss??frame.loss);
  if(loss.some(Number.isFinite))window.Plotly.react('plot-loss',[{x:frames.map(f=>f.step),y:loss,type:'scatter',mode:'lines',name:'Обучение',line:{color:isDark?'#66e4bc':'#087959',width:2.5}},{x:frames.map(f=>f.step),y:frames.map(f=>f.validation_loss),type:'scatter',mode:'lines',name:'Выбор',line:{color:'#b7a4ff',width:2}}],{paper_bgcolor:'transparent',plot_bgcolor:'transparent',font:{color:isDark?'#94a8b3':'#536c7b'},margin:{l:55,r:20,t:20,b:45},xaxis:{title:{text:'Шаг'},gridcolor:isDark?'#273740':'#cfdae1'},yaxis:{title:{text:'MSE'},gridcolor:isDark?'#273740':'#cfdae1'},legend:{orientation:'h',y:1.15}},{responsive:true,displayModeBar:false});
  $('frame-label').textContent=`Обучение · шаг ${last.step}`;
}
async function waitJob(identifier) {
  while(true) {
    const job=await api(`/jobs/${identifier}`);
    $('job-status').textContent=job.message||job.status;$('progress').querySelector('span').style.width=`${Math.round((job.progress||0)*100)}%`;
    if(job.status==='completed')return job.result;
    if(['error','failed','cancelled','interrupted'].includes(job.status))throw new Error(job.error||job.message||'Расчет остановлен.');
    liveTraining(job.events||[]);
    await new Promise(resolve=>setTimeout(resolve,400));
  }
}
async function execute(request) {
  const job=await api('/jobs',{method:'POST',body:request});state.job=job.id;state.request=request;state.revealed=false;
  return await waitJob(job.id);
}
async function runModel() {
  const request=requestFor();setBusy(true);
  try{const result=await execute(request);await setResult(result);toast('Модель обучена. Графики и метрики рассчитаны.');}finally{setBusy(false);}
}
function visibleCharts() {
  const groups={overview:['fit','loss','coef','prediction'],'3d':['3d','surface'],regularization:['path','geometry','surface','coef'],diagnostics:['residual','prediction','correlation','loss','learning-curve','importance'],all:['fit','3d','loss','coef','prediction','path','geometry','surface','residual','correlation','learning-curve','importance']};return groups[state.chartTab];
}
async function renderCharts() {
  const visible=visibleCharts();document.querySelectorAll('[data-chart]').forEach(panel=>panel.hidden=!visible.includes(panel.dataset.chart));
  if(!state.result)return;
  await charts.render(state.result,{theme:document.body.dataset.theme,xFeature:Number($('x-feature').value)||0,yFeature:Number($('y-feature').value)||0,visible,revealTest:state.revealed,editPoints:$('edit-points').checked,onPointMove:async point=>{try{if(state.busy)throw new Error('Сначала завершите текущий расчет.');const target=state.request.target;const dataset=await api(`/datasets/${state.dataset.id}/rows`,{method:'PATCH',body:{changes:[{index:point.index,values:{[point.featureName]:point.x,[target]:point.y}}]}});const request={...state.request,dataset_id:dataset.id};setDataset(dataset);restoreRequest(request);await runModel();}catch(error){toast(error.message);}},onPoint:(x,y,point)=>{if(!state.busy&&!$('edit-points').checked&&point?.index!=null)openDataEditor(point.index).catch(error=>toast(error.message));}});
}
async function changeAxes() {
  if(!state.result||!state.job)return;
  const names=state.result.plot_data?.feature_names||[];
  const x=names[Number($('x-feature').value)],y=names[Number($('y-feature').value)];
  if(x)state.result.prediction_grid=await api(`/jobs/${state.job}/grid?x_feature=${encodeURIComponent(x)}${y&&y!==x?'&y_feature='+encodeURIComponent(y):''}`);
  await renderCharts();
}
async function setResult(result) {
  state.result=result;state.frame=(result.trace||[]).length-1;
  const names=result.plot_data?.feature_names||result.raw_feature_names||result.feature_names||[];
  $('x-feature').innerHTML=names.map((name,index)=>`<option value="${index}">${esc(name)}</option>`).join('');
  $('y-feature').innerHTML=names.map((name,index)=>`<option value="${index}" ${index===Math.min(1,names.length-1)?'selected':''}>${esc(name)}</option>`).join('');
  $('timeline').max=Math.max(0,(result.trace||[]).length-1);$('timeline').value=state.frame;$('timeline').disabled=(result.trace||[]).length<2;
  $('play').disabled=$('timeline').disabled;$('step-back').disabled=$('timeline').disabled;$('step-forward').disabled=$('timeline').disabled;
  $('trace-label').textContent=result.trace_label||'Итоговое решение';
  $('frame-label').textContent=result.trace?.length>1?`Шаг ${result.trace.at(-1).step} / ${result.trace.at(-1).step}`:'Итоговое решение';
  $('warnings').innerHTML=(result.warnings||[]).map(warning=>`<div class="warning">${esc(warning)}</div>`).join('');
  renderMetrics();showExplanation();renderPredictionInputs();await renderCharts();
  $('reveal-test').disabled=state.revealed;
}
function metricName(id) {return state.catalogue.metrics.find(metric=>metric.id===id)?.name||id;}
function renderMetrics() {
  const result=state.result;if(!result)return;
  const validation=result.metrics?.validation||{};const selected=Object.keys(validation);const preferred=['r2','rmse','mae','mse'].filter(id=>selected.includes(id));const cards=[...preferred,...selected.filter(id=>!preferred.includes(id))].slice(0,4);
  $('metrics-summary').innerHTML=cards.map(id=>`<div class="metric-card"><div class="metric-name">${esc(metricName(id))}</div><strong>${number(validation[id])}</strong><div class="metric-subtitle">Выбор настроек · обучение: ${number(result.metrics.train?.[id])}</div></div>`).join('');
  $('metrics-table').innerHTML=`<div class="table-wrapper"><table><thead><tr><th>Метрика</th><th>Обучение</th><th>Выбор</th><th>Итоговая проверка</th></tr></thead><tbody>${selected.map(id=>`<tr title="${esc(result.metric_details?.validation?.[id]?.reason||'')}"><td>${esc(metricName(id))}</td><td>${number(result.metrics.train?.[id])}</td><td>${number(validation[id])}</td><td>${state.revealed?number(result.metrics.test?.[id]):'Скрыта'}</td></tr>`).join('')}</tbody></table></div>`;
  const invalid=Object.entries(result.metric_details?.validation||{}).filter(([id,detail])=>detail&&typeof detail==='object'&&(detail.reason||detail.error));
  $('metrics-table').innerHTML+=invalid.map(([id,detail])=>`<p class="help-text">${esc(metricName(id))}: ${esc(detail.reason||detail.error)}</p>`).join('');
  $('cv-results').innerHTML=result.cv&&Object.keys(result.cv).length?`<details><summary>Перекрестная проверка на обучающей части</summary><pre class="help-text">${esc(JSON.stringify(result.cv,null,2))}</pre></details>`:'';
}
function stopPlayback() {state.playing=false;clearTimeout(playbackTimer);$('play').textContent='▶';}
async function showFrame(index) {
  if(!state.result?.trace?.length)return;state.frame=Math.max(0,Math.min(index,state.result.trace.length-1));$('timeline').value=state.frame;
  const frame=state.result.trace[state.frame];$('frame-label').textContent=`Шаг ${frame.step} / ${state.result.trace.at(-1).step}`;
  await charts.frame(frame,state.result);
}
async function tick() {if(!state.playing)return;await showFrame(state.frame+1);if(state.frame>=state.result.trace.length-1){stopPlayback();return;}playbackTimer=setTimeout(tick,Number($('speed').value));}
function play() {if(!state.result?.trace?.length)return;if(state.playing){stopPlayback();return;}state.playing=true;$('play').textContent='Ⅱ';if(state.frame>=state.result.trace.length-1)state.frame=-1;tick();}

function renderCatalogue() {
  const search=$('dataset-search').value.toLowerCase(),filter=$('dataset-filter').value;
  const datasets=state.catalogue.datasets.filter(item=>(filter==='all'||item.kind===filter)&&`${item.name} ${item.label} ${item.description}`.toLowerCase().includes(search));
  $('dataset-catalogue').innerHTML=datasets.map(item=>`<article class="catalogue-card"><span class="badge">${esc(({synthetic:'Генератор',builtin:'Встроенный',fetch:'Загрузка',openml:'OpenML'})[item.kind]||item.kind)}${item.requires_network?' · интернет':''}</span><h3>${esc(item.label||item.name)}</h3><p>${esc(item.description)}</p><p class="help-text">${esc(item.name)}</p><button class="button outlined" data-load="${esc(item.id||item.name)}" ${item.supported===false?'disabled':''}>${item.supported===false?'Нужен отдельный адаптер':'Загрузить данные'}</button></article>`).join('');
  $('dataset-catalogue').querySelectorAll('[data-load]').forEach(button=>button.onclick=safely(async()=>{const item=state.catalogue.datasets.find(d=>(d.id||d.name)===button.dataset.load);button.disabled=true;try{if(item.name==='fetch_openml'){ $('openml-id').focus();toast('Введите идентификатор OpenML в поле выше.');return;}await loadData({kind:item.kind,name:item.name,params:item.kind==='synthetic'?generatedSpec().params:{seed:Number($('seed').value)}});switchView('lab');toast('Данные загружены. Проверьте цель и признаки.');}finally{button.disabled=false;}}));
}
function renderModels() {
  const search=$('model-search').value.toLowerCase();
  $('model-catalogue').innerHTML=state.catalogue.models.filter(item=>`${item.name} ${item.family} ${item.description}`.toLowerCase().includes(search)).map(model=>`<article class="catalogue-card ${state.selectedModels.has(model.id)?'selected-card':''}"><span class="badge">${esc(model.family)}</span><h3>${esc(model.name)}</h3><p>${esc(model.description)}</p><code>${esc(model.formula)}</code><details><summary>Объяснение</summary><p>${esc(Array.isArray(model.explanation)?model.explanation.join('\n'):model.explanation||model.description)}</p>${model.source?`<a href="${esc(model.source)}" target="_blank" rel="noopener">Документация</a>`:''}</details><div class="card-actions"><button class="button outlined" data-model-use="${esc(model.id)}">Выбрать</button><label class="check"><input type="checkbox" data-model-compare="${esc(model.id)}" ${state.selectedModels.has(model.id)?'checked':''}>Сравнить</label></div></article>`).join('');
  $('model-catalogue').querySelectorAll('[data-model-use]').forEach(button=>button.onclick=()=>{if(state.busy){toast('Дождитесь завершения или остановите текущий расчет.');return;}$('model').value=button.dataset.modelUse;showModelParameters();switchView('lab');});
  $('model-catalogue').querySelectorAll('[data-model-compare]').forEach(input=>input.onchange=()=>{if(input.checked)state.selectedModels.add(input.dataset.modelCompare);else state.selectedModels.delete(input.dataset.modelCompare);input.closest('.catalogue-card').classList.toggle('selected-card',input.checked);});
  renderFormulas($('model-catalogue'));
}
function renderLearning() {
  $('lesson-list').innerHTML=state.catalogue.lessons.map((lesson,index)=>`<button class="lesson-button ${index===state.lesson?'active':''}" data-lesson="${index}">${String(index+1).padStart(2,'0')} · ${esc(lesson.title)}</button>`).join('');
  const lesson=state.catalogue.lessons[state.lesson];if(!lesson)return;
  $('lesson-content').innerHTML=`<h2>${esc(lesson.title)}</h2><p>${esc(lesson.summary)}</p>${lesson.sections.map(section=>`<h3>${esc(section.title)}</h3><p>${esc(section.text)}</p>${section.formula?`<code>${esc(section.formula)}</code>`:''}`).join('')}<div class="exercise"><h3>Попробуй сам</h3><p>${esc(typeof lesson.exercise==='object'?lesson.exercise.text||JSON.stringify(lesson.exercise):lesson.exercise)}</p>${lesson.preset?'<button id="lesson-run" class="button primary">Открыть этот эксперимент</button>':''}</div>`;
  $('lesson-list').querySelectorAll('[data-lesson]').forEach(button=>button.onclick=()=>{state.lesson=Number(button.dataset.lesson);renderLearning();});
  if($('lesson-run'))$('lesson-run').onclick=safely(async()=>{const preset=lesson.preset;$('scale').checked=true;$('impute').checked=true;$('degree').value=1;$('train-share').value=60;$('val-share').value=20;$('shuffle').checked=true;$('cv').value=0;$('custom-metric').value='';$('learning-curve').checked=false;$('permutation-importance').checked=false;$('path-enabled').checked=true;$('metric-options').querySelectorAll('input').forEach(input=>input.checked=['mse','rmse','mae','r2'].includes(input.value));$('metric-quantile').value=preset.params?.quantile??.5;updateSplit();await loadData(preset.dataset);$('model').value=preset.model;showModelParameters();for(const [key,value]of Object.entries(preset.params||{})){const input=$('model-params').querySelector(`[data-param="${key}"]`);if(input){if(input.type==='checkbox')input.checked=value;else input.value=value;}}if(preset.epochs)$('epochs').value=preset.epochs;switchView('lab');await runModel();});
  $('glossary').innerHTML='<dl>'+state.catalogue.glossary.map(item=>`<dt>${esc(item.term)}</dt><dd>${esc(item.definition)}</dd>`).join('')+'</dl>';
  renderFormulas($('lesson-content'));
}
function rankingDirection() {return $('ranking-direction').value==='auto'?(state.catalogue.metrics.find(m=>m.id===$('ranking-metric').value)?.direction||'min'):$('ranking-direction').value;}
function defaultsFor(model) {return Object.fromEntries(model.params.map(param=>[param.key,param.default]));}
function comparisonTable() {
  const metric=$('ranking-metric').value;const direction=rankingDirection();
  const list=[...state.comparisons].sort((a,b)=>{const x=a.result?.metrics?.validation?.[metric],y=b.result?.metrics?.validation?.[metric];if(x==null)return 1;if(y==null)return -1;return direction==='max'?y-x:x-y;});
  $('comparison-table').innerHTML=`<div class="table-wrapper"><table><thead><tr><th>Модель / настройка</th><th>${esc(metricName(metric))} · выбор</th><th>Обучение</th><th>Результат</th></tr></thead><tbody>${list.map(item=>`<tr><td>${esc(item.name)}</td><td>${number(item.result?.metrics?.validation?.[metric])}</td><td>${number(item.result?.metrics?.train?.[metric])}</td><td>${item.error?esc(item.error):`<button class="text-button" data-open-job="${item.job}">Открыть</button>`}</td></tr>`).join('')}</tbody></table></div>`;
  $('comparison-table').querySelectorAll('[data-open-job]').forEach(button=>button.onclick=safely(async()=>{const item=state.comparisons.find(item=>item.job===button.dataset.openJob);state.job=item.job;state.request=item.request;state.revealed=false;$('model').value=item.request.model;showModelParameters();for(const[key,value]of Object.entries(item.request.params)){const input=$('model-params').querySelector(`[data-param="${key}"]`);if(input){if(input.type==='checkbox')input.checked=value;else input.value=value;}}await setResult(item.result);}));
}
async function compareModels(tune=false) {
  if(state.busy)return;
  const current=selectedModel();if(tune&&!current.params.some(p=>p.key==='alpha'))throw new Error('У этой модели нет параметра α. Выберите модель с силой регуляризации.');
  const list=tune?Array.from({length:9},(_,i)=>({model:current,params:{...modelParameters(),alpha:10**(i-4)},name:`${current.name} · α=${10**(i-4)}`})):state.catalogue.models.filter(m=>state.selectedModels.has(m.id)).map(model=>({model,params:defaultsFor(model),name:model.name}));
  if(!list.length)throw new Error('Отметьте модели для сравнения в разделе «Модели».');
  const base=requestFor();if($('ranking-metric').value==='custom'&&!base.custom_metric)throw new Error('Для сравнения по своей метрике сначала задайте ее формулу.');base.metrics=[...new Set([...base.metrics,$('ranking-metric').value])];setBusy(true);state.comparisons=[];
  try{for(let i=0;i<list.length;i++){const item=list[i];$('job-status').textContent=`Расчет ${i+1}/${list.length}: ${item.name}`;const request={...base,model:item.model.id,params:item.params,regularization_path:false};try{const result=await execute(request);state.comparisons.push({name:item.name,result,request,job:state.job});}catch(error){if(error.message.includes('отменен'))throw error;state.comparisons.push({name:item.name,error:error.message});}comparisonTable();}
    const metric=$('ranking-metric').value,direction=rankingDirection();const valid=state.comparisons.filter(item=>item.result&&item.result.metrics.validation[metric]!=null).sort((a,b)=>direction==='max'?b.result.metrics.validation[metric]-a.result.metrics.validation[metric]:a.result.metrics.validation[metric]-b.result.metrics.validation[metric]);
    if(valid.length){const best=valid[0];state.job=best.job;state.request=best.request;state.revealed=false;$('model').value=best.request.model;showModelParameters();for(const[key,value]of Object.entries(best.request.params)){const input=$('model-params').querySelector(`[data-param="${key}"]`);if(input){if(input.type==='checkbox')input.checked=value;else input.value=value;}}await setResult(best.result);toast(`По ${metricName(metric)} выбрана ${best.name}. Итоговая проверка еще скрыта.`);}else toast('Нет сравнимых результатов для выбранной метрики. Смотрите причины в таблице.');
  }finally{setBusy(false);}
}
async function openDataEditor(index=null) {
  if(!state.dataset)throw new Error('Загрузите данные.');
  const offset=index!=null?Math.max(0,Math.floor(index/100)*100):0;
  state.editor=await api(`/datasets/${state.dataset.id}/rows?offset=${offset}&limit=100`);state.editor.original=JSON.parse(JSON.stringify(state.editor.rows));state.editor.additions=[];
  renderEditor();$('data-dialog').showModal();
}
function renderEditor() {
  const editor=state.editor;const columns=editor.columns.map(column=>typeof column==='string'?column:column.name);
  $('data-editor').innerHTML=`<p class="help-text">Строки ${editor.offset+1}–${editor.offset+editor.rows.length} из ${editor.total}. Остальные строки сохраняются.</p><table><thead><tr><th>№</th>${columns.map(column=>`<th>${esc(column)}</th>`).join('')}</tr></thead><tbody>${[...editor.rows,...editor.additions].map((row,index)=>`<tr><td>${index<editor.rows.length?editor.offset+index+1:'Новая'}</td>${columns.map(column=>`<td><input data-row="${index}" data-column="${esc(column)}" value="${esc(row[column])}" aria-label="Строка ${index+1}, ${esc(column)}"></td>`).join('')}</tr>`).join('')}</tbody></table>`;
  $('data-editor').querySelectorAll('input').forEach(input=>input.oninput=()=>{const index=Number(input.dataset.row),column=input.dataset.column;const definition=editor.columns.find(c=>c.name===column);const value=input.value===''?null:definition?.numeric?Number(input.value):input.value;const row=index<editor.rows.length?editor.rows[index]:editor.additions[index-editor.rows.length];row[column]=value;});
}
async function applyData() {
  const editor=state.editor;const changes=editor.rows.map((row,index)=>({index:editor.offset+index,values:row})).filter((patch,index)=>JSON.stringify(patch.values)!==JSON.stringify(editor.original[index]));
  const dataset=await api(`/datasets/${state.dataset.id}/rows`,{method:'PATCH',body:{changes,additions:editor.additions}});$('data-dialog').close();setDataset(dataset);toast('Изменения применены. Можно обучить модель на новых точках.');
}
function renderPredictionInputs() {
  if(!state.dataset)return;
  const features=state.request?.features||[];
  $('prediction-inputs').innerHTML=features.slice(0,60).map(name=>{const column=state.dataset.columns.find(c=>c.name===name),value=state.dataset.preview?.[0]?.[name]??'';return `<label>${esc(name)}<input data-predict="${esc(name)}" type="${column?.numeric?'number':'text'}" value="${esc(value)}" step="any"></label>`;}).join('');
}
async function refreshHistory() {
  const experiments=await api('/experiments');
  $('history-list').innerHTML=experiments.length?experiments.map(item=>`<article class="catalogue-card"><span class="badge">${esc(new Date(item.created*1000).toLocaleString('ru'))}</span><h3>${esc(item.name)}</h3><p>${esc(item.model||'')}</p><p class="help-text">${Object.entries(item.metrics||{}).slice(0,3).map(([key,value])=>`${esc(key)}: ${number(value)}`).join(' · ')}</p><div class="card-actions"><button class="button outlined" data-experiment="${item.id}">Открыть</button><button class="text-button" data-delete-experiment="${item.id}">Удалить</button></div></article>`).join(''):'<p class="empty-state">Сохраните первый эксперимент из лаборатории.</p>';
  $('history-list').querySelectorAll('[data-experiment]').forEach(button=>button.onclick=safely(async()=>{const item=await api(`/experiments/${button.dataset.experiment}`);setDataset(await api(`/datasets/${item.request.dataset_id}`));restoreRequest(item.request);state.job=item.job_id;state.request=item.request;state.revealed=!item.result.test_hidden;await setResult(item.result);switchView('lab');}));
  $('history-list').querySelectorAll('[data-delete-experiment]').forEach(button=>button.onclick=safely(async()=>{await api(`/experiments/${button.dataset.deleteExperiment}`,{method:'DELETE'});await refreshHistory();}));
}
function restoreRequest(request) {
  $('target').value=request.target;refreshFeatures();$('features').querySelectorAll('input').forEach(input=>input.checked=request.features.includes(input.value));
  $('model').value=request.model;showModelParameters();for(const[key,value]of Object.entries(request.params)){const input=$('model-params').querySelector(`[data-param="${key}"]`);if(input){if(input.type==='checkbox')input.checked=value;else input.value=value;}}
  $('train-share').value=Math.round(request.split.train*100);$('val-share').value=Math.round(request.split.validation*100);updateSplit();$('shuffle').checked=request.split.shuffle;$('scale').checked=request.preprocessing.scale;$('impute').checked=request.preprocessing.impute;$('degree').value=request.preprocessing.degree;$('epochs').value=request.epochs;$('cv').value=request.cv;$('seed').value=request.seed;$('custom-metric').value=request.custom_metric||'';$('path-enabled').checked=request.regularization_path??true;$('learning-curve').checked=!!request.learning_curve;$('permutation-importance').checked=!!request.permutation_importance;$('metric-quantile').value=request.metric_params?.quantile??.5;$('metric-power').value=request.metric_params?.power??1.5;$('metric-options').querySelectorAll('input').forEach(input=>input.checked=request.metrics.includes(input.value));
}
function updateSplit() {const train=Number($('train-share').value),val=Number($('val-share').value);$('test-share').textContent=number(100-train-val);document.querySelector('.split-bar .train').style.width=`${train}%`;document.querySelector('.split-bar .validation').style.width=`${val}%`;document.querySelector('.split-bar .test').style.width=`${100-train-val}%`;}

async function initialize() {
  state.catalogue=await api('/catalogue');
  $('model').innerHTML=state.catalogue.models.map(model=>`<option value="${esc(model.id)}">${esc(model.name)}</option>`).join('');$('model').value='ridge';showModelParameters();
  $('metric-options').innerHTML=state.catalogue.metrics.filter(metric=>metric.id!=='custom').map(metric=>`<label class="check" title="${esc(typeof metric.help==='string'?metric.help:metric.description)}"><input type="checkbox" value="${esc(metric.id)}" ${['mse','rmse','mae','r2'].includes(metric.id)?'checked':''}>${esc(metric.name)}</label>`).join('');
  $('ranking-metric').innerHTML=state.catalogue.metrics.map(metric=>`<option value="${esc(metric.id)}">${esc(metric.name)}</option>`).join('')+(state.catalogue.metrics.some(metric=>metric.id==='custom')?'':'<option value="custom">Своя формула</option>');$('ranking-metric').value='rmse';
  renderCatalogue();renderModels();renderLearning();
  document.querySelectorAll('[data-view]').forEach(button=>button.onclick=()=>switchView(button.dataset.view));
  document.querySelectorAll('[data-charts]').forEach(button=>button.onclick=safely(async()=>{state.chartTab=button.dataset.charts;document.querySelectorAll('.chart-tab').forEach(tab=>tab.classList.toggle('active',tab===button));await renderCharts();}));
  ['n-samples','n-features','noise','correlation','outliers','sparsity','scale-spread'].forEach(id=>$(id).oninput=()=>$(id+'-value').value=$(id).value);
  $('generate').onclick=safely(async()=>{await loadData(generatedSpec());await runModel();});
  $('browse-data').onclick=()=>switchView('data');$('models-compare').onclick=()=>{switchView('lab');$('compare').scrollIntoView({behavior:'smooth',block:'center'});};
  $('dataset-search').oninput=renderCatalogue;$('dataset-filter').onchange=renderCatalogue;$('model-search').oninput=renderModels;
  $('model').onchange=()=>{showModelParameters();scheduleFit();};$('target').onchange=()=>{refreshFeatures();scheduleFit();};
  ['scale','impute','degree','shuffle','cv','epochs'].forEach(id=>$(id).onchange=scheduleFit);
  ['train-share','val-share'].forEach(id=>$(id).onchange=()=>{updateSplit();scheduleFit();});
  $('run').onclick=safely(runModel);$('cancel').onclick=safely(async()=>{if(state.job)await api(`/jobs/${state.job}`,{method:'DELETE'});});
  $('all-metrics').onclick=()=>{const inputs=Array.from($('metric-options').querySelectorAll('input'));const all=inputs.every(input=>input.checked);inputs.forEach(input=>input.checked=!all);$('all-metrics').textContent=all?'Все':'Снять все';};
  $('play').onclick=play;$('timeline').oninput=safely(async()=>{stopPlayback();await showFrame(Number($('timeline').value));});$('step-back').onclick=safely(async()=>{stopPlayback();await showFrame(state.frame-1);});$('step-forward').onclick=safely(async()=>{stopPlayback();await showFrame(state.frame+1);});
  $('x-feature').onchange=safely(changeAxes);$('y-feature').onchange=safely(changeAxes);
  $('edit-points').onchange=safely(renderCharts);
  $('theme').onclick=safely(async()=>{document.body.dataset.theme=document.body.dataset.theme==='dark'?'light':'dark';localStorage.setItem('linear-lab-theme',document.body.dataset.theme);await renderCharts();});
  $('compare').onclick=safely(()=>compareModels(false));$('tune').onclick=safely(()=>compareModels(true));$('ranking-metric').onchange=comparisonTable;$('ranking-direction').onchange=comparisonTable;
  $('edit-data').onclick=safely(()=>openDataEditor());$('close-data').onclick=()=>$('data-dialog').close();$('apply-data').onclick=safely(applyData);
  $('add-row').onclick=()=>{const editor=state.editor,row={};editor.columns.forEach(column=>{const name=typeof column==='string'?column:column.name;row[name]=editor.rows.at(-1)?.[name]??0;});editor.additions.push(row);renderEditor();$('data-editor').scrollTop=$('data-editor').scrollHeight;};
  $('upload').onchange=safely(async()=>{const file=$('upload').files[0];if(!file)return;const body=new FormData();body.append('file',file);setDataset(await api('/datasets/upload',{method:'POST',body}));switchView('lab');toast('Таблица загружена. Выберите целевую величину.');});
  $('openml-load').onclick=safely(async()=>{const id=Number($('openml-id').value);if(!id)throw new Error('Введите числовой идентификатор OpenML.');$('openml-load').disabled=true;try{await loadData({kind:'openml',name:'fetch_openml',params:{data_id:id}});switchView('lab');}finally{$('openml-load').disabled=false;}});
  $('save').onclick=()=>{if(!state.result||!state.job){toast('Сначала обучите модель.');return;}$('experiment-name').value=`${state.result.model_name||selectedModel().name} · ${state.dataset.name}`;$('save-dialog').showModal();};$('close-save').onclick=()=>$('save-dialog').close();
  $('confirm-save').onclick=safely(async()=>{await api('/experiments',{method:'POST',body:{job_id:state.job,name:$('experiment-name').value}});$('save-dialog').close();toast('Эксперимент сохранен вместе с настройками и результатами.');});
  $('export').onchange=()=>{const format=$('export').value;if(format&&state.job){const a=document.createElement('a');a.href=`/api/jobs/${state.job}/export?format=${format}`;a.download='';a.click();}else if(format)toast('Сначала обучите модель.');$('export').value='';};
  $('reveal-test').onclick=safely(async()=>{if(!state.job)throw new Error('Сначала обучите модель.');const job=await api(`/jobs/${state.job}/reveal-test`,{method:'POST',body:{}});state.revealed=true;await setResult(job.result);toast('Итоговая проверка открыта. Следующее обучение снова скроет ее.');});
  $('predict').onclick=safely(async()=>{if(!state.job)throw new Error('Сначала обучите модель.');const row={};$('prediction-inputs').querySelectorAll('input').forEach(input=>row[input.dataset.predict]=input.type==='number'?Number(input.value):input.value);const result=await api('/predict',{method:'POST',body:{job_id:state.job,rows:[row]}});$('prediction-result').value=number(result.predictions[0]);});
  document.addEventListener('keydown',safely(async event=>{if(event.ctrlKey&&event.key==='Enter'){event.preventDefault();if(!state.busy)await runModel();}if((event.ctrlKey||event.metaKey)&&event.key.toLowerCase()==='s'){event.preventDefault();$('save').click();}if(event.key===' '&&state.view==='lab'&&!['INPUT','SELECT','TEXTAREA','BUTTON'].includes(event.target.tagName)){event.preventDefault();play();}}));
  document.body.dataset.theme=localStorage.getItem('linear-lab-theme')||'dark';
  await loadData(generatedSpec());await runModel();
  const initial=window.location.hash.slice(1);if(['data','models','learn','history'].includes(initial))switchView(initial);
}
initialize().catch(error=>{toast(error.message);$('job-status').textContent=error.message;console.error(error);});
