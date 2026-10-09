/** Train-only feature engineering controls and a preview of real transformed rows. */
import {installHelp, attachHelp} from './help.js';

const FEATURE = '#66e4bc';
const TARGET = '#ff927e';
const PREPROCESSING_KEYS = ['imputation', 'fill_value', 'knn_neighbors', 'scaler', 'numeric_transform', 'degree', 'interaction_only', 'clip_quantiles', 'missing_indicator', 'variance_threshold', 'selection', 'max_features'];
const SAMPLING_KEYS = ['method', 'focus', 'relevance_threshold', 'neighbors', 'perturbation', 'sampling', 'undersample'];
const DEFAULTS = {imputation:'median', fill_value:0, knn_neighbors:5, scaler:'standard', numeric_transform:'none', degree:1, interaction_only:false, clip_quantiles:null, missing_indicator:false, variance_threshold:null, selection:'none', max_features:null};
const SAMPLING_DEFAULTS = {method:'none', focus:'both', relevance_threshold:.5, neighbors:5, perturbation:.02, sampling:'balance', undersample:true};
const PRESETS = Object.freeze({
  standard:{name:'Обычные данные', text:'Заполнить пропуски медианой и привести признаки к одному масштабу. Начните с этого варианта.', settings:{...DEFAULTS}},
  robust:{name:'Есть крайние значения', text:'Использовать устойчивый масштаб и ограничить края каждого числового признака по обучающим квантилям 1% и 99%. Это меняет значения: сравните качество с обычным вариантом.', settings:{...DEFAULTS, scaler:'robust', clip_quantiles:[.01,.99]}},
  skewed:{name:'Длинный хвост распределения', text:'Yeo–Johnson сглаживает асимметричное распределение, затем применяется обычный масштаб. Подходит и для отрицательных чисел.', settings:{...DEFAULTS, numeric_transform:'yeo_johnson'}},
  nonlinear:{name:'Зависимость изгибается', text:'Добавить квадраты и произведения числовых признаков. Линейная модель получит больше входных столбцов и сможет описать изгиб.', settings:{...DEFAULTS, degree:2}},
});
const LABELS = {
  imputation:{median:'Медиана · середина значений', mean:'Среднее значение', most_frequent:'Самое частое значение', constant:'Заданное число', knn:'Похожие строки · KNN', none:'Оставить пропуски'},
  scaler:{standard:'Один масштаб · Standard', robust:'Устойчивый к выбросам · Robust', minmax:'В диапазон 0…1 · MinMax', maxabs:'Делить на наибольший модуль · MaxAbs', none:'Сохранить единицы измерения'},
  numeric_transform:{none:'Не менять форму', log1p:'Сжатие больших чисел · логарифм со знаком', sqrt:'Слабее сжимать большие числа · корень со знаком', yeo_johnson:'Уменьшить асимметрию · Yeo–Johnson', quantile_normal:'По рангам → нормальное распределение', quantile_uniform:'По рангам → равномерное распределение'},
  selection:{none:'Сохранить все столбцы', f_regression:'Выбрать по линейной связи с целью', mutual_info:'Выбрать по общей связи с целью'},
  method:{none:'Оставить обучающие строки', random_over:'Повторять редкие строки · over', random_under:'Удалять часть обычных строк · under', smoter:'Создавать между соседями · SMOTER', smogn:'Создавать между соседями и с шумом · SMOGN'},
  focus:{both:'Оба края: малые и большие ответы', high:'Большие ответы', low:'Малые ответы'},
  sampling:{balance:'Умеренно · balance', extreme:'Сильнее · extreme'},
};
const HELP = {
  fill_value:{title:'Число вместо пропуска', text:'Пустые числовые значения заменяются этим числом. Для категориальных столбцов используется отдельная категория «значение отсутствует». Константа может быть далеко от обычных значений — выбирайте её по смыслу данных.', example:'Ноль может означать отсутствие измерения, но не обязательно нулевой доход.', lesson:'22-missing-values', source:'https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html'},
  knn_neighbors:{title:'Похожие строки для заполнения', text:'KNN ищет это число похожих обучающих строк с известным значением и использует их для заполнения пропуска. Расстояния зависят от масштаба исходных чисел: разные единицы могут влиять на поиск соседей.', example:'При пяти соседях пропуск заполняется по доступным значениям пяти близких строк.', lesson:'22-missing-values', source:'https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html'},
  max_features:{title:'Сколько подготовленных признаков оставить', text:'Ограничение применяется после полиномов и кодирования категорий. Метод отбора вычисляет оценки по обучающей цели и оставляет не больше указанного числа столбцов. При перекрёстной проверке отбор выполняется заново в каждом обучающем разрезе.', example:'Из 20 подготовленных столбцов можно оставить 5 с наибольшей оценкой связи с целью.', lesson:'24-feature-engineering', source:'https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html'},
  undersample:{title:'Уменьшать обычную область в SMOGN', text:'Включает удаление части обычных обучающих строк вместе с созданием примеров для редкой области цели. Выключите, если хотите сохранять обычные строки. Проверочная и тестовая части не изменяются.', example:'Мало дорогих объектов: SMOGN может добавить примеры дорогих и уменьшить число обычных.', lesson:'28-rare-target-smoter', source:'https://proceedings.mlr.press/v74/branco17a.html'},
  preview:{title:'Проверить подготовку до обучения модели', text:'Сервис выделяет обучающую часть по текущей схеме разделения, обучает преобразования только на ней и показывает реальные подготовленные значения. Модель при этом не обучается. Если включено пересэмплирование, оно тоже выполняется только для обучающих строк.', example:'Выберите исходную площадь и подготовленную площадь, чтобы увидеть изменение единиц после Standard.', lesson:'25-data-leakage', source:'https://scikit-learn.org/stable/common_pitfalls.html'},
};

const esc = value => String(value ?? '').replace(/[&<>"']/g, char => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[char]));
const clone = value => structuredClone(value);
const number = value => Number.isFinite(Number(value)) && value !== null ? Number(value).toLocaleString('ru-RU',{maximumFractionDigits:4}) : '—';
const options = choices => Object.entries(choices).map(([value,label])=>`<option value="${esc(value)}">${esc(label)}</option>`).join('');
const field = (key, title, input, note='') => `<div class="pw-field"><label for="pw-${key}">${esc(title)}</label>${input}${note?`<p class="pw-note">${esc(note)}</p>`:''}</div>`;
const select = (key,title,choices,note='') => field(key,title,`<select id="pw-${key}" data-pw="${key}">${options(choices)}</select>`,note);
const numeric = (key,title,attrs,note='') => field(key,title,`<input id="pw-${key}" data-pw="${key}" type="number" ${attrs}>`,note);
const check = (key,title,note='') => `<div class="pw-field pw-checkbox"><label for="pw-${key}"><input id="pw-${key}" data-pw="${key}" type="checkbox"> ${esc(title)}</label>${note?`<p class="pw-note">${esc(note)}</p>`:''}</div>`;

export class PreprocessingWorkspace {
  constructor({api, getRequest, onChange=()=>{}, onError=()=>{}, openLesson=()=>{}}) {
    this.api=api; this.getRequest=getRequest; this.onChange=onChange; this.onError=onError; this.openLesson=openLesson;
    this.preprocessing={}; this.resampling={}; this.dataset=null; this.busy=false; this.previewBusy=false; this.sequence=0; this.preview=null;
  }

  mount(container) {
    if(!container)throw new Error('Не найден контейнер подготовки данных.');
    this.container=container;
    container.classList.add('pw-workspace');
    container.innerHTML=`<div class="pw-intro"><p>Создайте пригодные для модели признаки. Все вычисляемые настройки изучаются только на обучающих строках.</p></div>
      ${select('preset','Готовый рецепт', {custom:'Свои настройки',...Object.fromEntries(Object.entries(PRESETS).map(([id,preset])=>[id,preset.name]))})}
      <p class="pw-preset-note">Настройки по умолчанию: медиана для пропусков, общий масштаб, без новых столбцов.</p>
      <div class="pw-flow" aria-label="Порядок подготовки"><span>Числа</span><span>Пропуски</span><span>Форма</span><span>Новые признаки</span><span>Масштаб</span><span>Отбор</span></div>
      <details class="pw-settings"><summary>Точные настройки признаков</summary><div class="pw-settings-body">
        ${select('imputation','Как заполнить пропуски',LABELS.imputation)}
        <div data-pw-section="constant">${numeric('fill_value','Число вместо пропуска','step="any"')}</div>
        <div data-pw-section="knn">${numeric('knn_neighbors','Сколько похожих строк брать','min="1" max="100" step="1"')}</div>
        ${check('missing_indicator','Добавить флаги пропусков','Новый вход 0/1 сообщает, что исходное измерение отсутствовало.')}
        ${select('numeric_transform','Как изменить числовую форму',LABELS.numeric_transform)}
        ${select('degree','Добавить степени и сочетания',{1:'Без расширения · степень 1',2:'Квадраты и произведения · степень 2',3:'До кубов · степень 3',4:'До четвёртой степени',5:'До пятой степени'},'Каждый новый столбец получает свой коэффициент. Слишком много столбцов увеличивает время и риск переобучения.')}
        ${check('interaction_only','Оставить только сочетания разных признаков','Например x₁×x₂, без x₁² и x₂².')}
        ${select('scaler','Как привести признаки к одному масштабу',LABELS.scaler)}
        ${select('clipping','Ограничить крайние значения',{none:'Сохранить значения',p01:'Ограничить ниже 1% и выше 99%',p05:'Ограничить ниже 5% и выше 95%',custom:'Задать свои границы'},'Строки остаются; значения за границами заменяются границами. Квантили вычисляются по обучению.')}
        <div class="pw-pair" data-pw-section="clip-custom">${numeric('clip_lower','Нижний квантиль','min="0" max="1" step=".01"')}${numeric('clip_upper','Верхний квантиль','min="0" max="1" step=".01"')}</div>
        ${check('remove_constant','Удалить постоянные столбцы')}
        <div data-pw-section="variance">${numeric('variance_threshold','Минимальная дисперсия','min="0" step="any"','Ноль убирает постоянные столбцы. Больший порог зависит от масштаба после преобразований.')}</div>
        ${select('selection','Выбрать наиболее полезные признаки',LABELS.selection)}
        <div data-pw-section="selection">${numeric('max_features','Сколько столбцов оставить','min="1" max="2000" step="1"','Пустое поле сохраняет все столбцы. Оценка связи рассчитывается только по обучению.')}</div>
        <p class="pw-note">Категории автоматически становятся столбцами 0/1. Новые неизвестные категории при прогнозе игнорируются. Цель не преобразуется этими настройками.</p>
      </div></details>
      <details class="pw-settings"><summary>Редкие ответы: изменить состав обучения</summary><div class="pw-settings-body">
        ${select('method','Как менять обучающие строки',LABELS.method)}
        <div data-pw-section="sampling">
          ${select('focus','Какие края числовой цели важнее',LABELS.focus)}
          ${select('sampling','Насколько менять баланс',LABELS.sampling)}
          ${numeric('relevance_threshold','Порог важности редких ответов','min=".01" max=".99" step=".05"','Это порог функции важности области цели, а не вероятность класса.')}
          <div data-pw-section="neighbors">${numeric('neighbors','Соседи для новых примеров','min="1" max="100" step="1"')}</div>
          <div data-pw-section="smogn">${numeric('perturbation','Доля случайного возмущения','min=".001" max="1" step=".01"')}${check('undersample','Уменьшать также обычную область цели')}</div>
          <p class="pw-warning">Работает с числовой целью регрессии. SMOTER/SMOGN создают синтетические примеры, а не новые реальные наблюдения; для них нужны исходные числовые признаки. Для категорий выбирайте повторение или удаление целых строк.</p>
        </div>
      </div></details>
      <div class="pw-actions"><button type="button" class="button outlined" data-pw-action="preview">Посмотреть до / после</button><button type="button" class="text-button" data-pw-action="reset">Сбросить</button></div>
      <p class="pw-status" role="status" aria-live="polite"></p>
      <dialog class="pw-preview" aria-labelledby="pw-preview-title"><div class="pw-preview-header"><div><p class="eyebrow">Реальная подготовка данных</p><h2 id="pw-preview-title">До и после преобразований</h2></div><button type="button" class="icon-button" data-pw-action="close" aria-label="Закрыть просмотр">×</button></div><div class="pw-preview-body"></div></dialog>`;
    this.onInput=event=>this.handleChange(event);
    this.onClick=event=>{
      const action=event.target.closest('[data-pw-action]')?.dataset.pwAction;
      if(action==='preview')this.showPreview();
      if(action==='reset'){this.reset();this.onChange(this.getConfig());}
      if(action==='close')this.closePreview();
    };
    container.addEventListener('change',this.onInput);
    container.addEventListener('click',this.onClick);
    this.help=installHelp(container,{openLesson:this.openLesson});
    const keys={imputation:'imputation',fill_value:HELP.fill_value,knn_neighbors:HELP.knn_neighbors,missing_indicator:'missing_indicator',numeric_transform:'numeric_transform',degree:'degree',interaction_only:'interaction_only',scaler:'scaler',clipping:'clip_quantiles',clip_lower:'clip_quantiles',clip_upper:'clip_quantiles',remove_constant:'variance_threshold',variance_threshold:'variance_threshold',selection:'selection',max_features:HELP.max_features,method:'method',focus:'focus',sampling:'sampling',relevance_threshold:'relevance_threshold',neighbors:'neighbors',perturbation:'perturbation',undersample:HELP.undersample};
    for(const [key,help] of Object.entries(keys))attachHelp(this.help,this.field(key),help);
    attachHelp(this.help,container.querySelector('[data-pw-action="preview"]'),HELP.preview,{host:container.querySelector('.pw-actions')});
    this.refreshControls();this.setBusy(this.busy);
    return this;
  }

  field(key){return this.container?.querySelector(`[data-pw="${key}"]`);}
  baseConfig(){
    let request={};try{request=this.getRequest?.()||{};}catch{/* No dataset has been selected yet. */}
    const base=request.preprocessing||{};
    return {...DEFAULTS,...base,imputation:base.imputation??(base.impute===false?'none':'median'),scaler:base.scaler??(base.scale===false?'none':'standard'),...this.preprocessing};
  }
  getConfig(){return {preprocessing:clone(this.preprocessing),resampling:clone(this.resampling)};}

  setConfig(requestOrPreset={}){
    const source=requestOrPreset.preprocessing||{};
    this.preprocessing=Object.fromEntries(PREPROCESSING_KEYS.filter(key=>Object.hasOwn(source,key)).map(key=>[key,clone(source[key])]));
    this.resampling=Object.fromEntries(SAMPLING_KEYS.filter(key=>Object.hasOwn(requestOrPreset.resampling||{},key)).map(key=>[key,clone(requestOrPreset.resampling[key])]));
    this.invalidatePreview();this.refreshControls({...DEFAULTS,...source,imputation:source.imputation??(source.impute===false?'none':'median'),scaler:source.scaler??(source.scale===false?'none':'standard')});
    return this;
  }
  reset(){
    this.preprocessing={};this.resampling={};
    // Saved v1/v2 requests can populate legacy controls. They remain the base
    // request values, so reset them together with the visible workspace.
    const document=this.container?.ownerDocument;
    const degree=document?.getElementById('degree');if(degree)degree.value='1';
    for(const id of ['scale','impute']){const input=document?.getElementById(id);if(input)input.checked=true;}
    this.invalidatePreview();this.refreshControls({...DEFAULTS});
    this.status('Обычная подготовка: медиана, общий масштаб, без пересэмплирования.');return this;
  }
  setDataset(meta){this.dataset=meta;this.invalidatePreview();this.refreshControls();this.setBusy(this.busy);return this;}
  setBusy(busy){this.busy=Boolean(busy);if(!this.container)return this;this.container.querySelectorAll('[data-pw]').forEach(el=>el.disabled=this.busy||this.previewBusy);this.container.querySelector('[data-pw-action="reset"]').disabled=this.busy||this.previewBusy;this.container.querySelector('[data-pw-action="preview"]').disabled=this.busy||this.previewBusy||!this.dataset;this.refreshVisibility();return this;}
  status(message){const node=this.container?.querySelector('.pw-status');if(node)node.textContent=message;}

  refreshControls(effective){
    if(!this.container)return;
    const settings=effective||this.baseConfig(),sampling={...SAMPLING_DEFAULTS,...this.resampling};
    for(const key of PREPROCESSING_KEYS){const input=this.field(key);if(!input)continue;if(input.type==='checkbox')input.checked=Boolean(settings[key]);else input.value=settings[key]??'';}
    for(const key of SAMPLING_KEYS){const input=this.field(key);if(!input)continue;if(input.type==='checkbox')input.checked=Boolean(sampling[key]);else input.value=sampling[key]??'';}
    const clip=settings.clip_quantiles;
    this.field('clipping').value=!clip?'none':clip[0]===.01&&clip[1]===.99?'p01':clip[0]===.05&&clip[1]===.95?'p05':'custom';
    this.field('clip_lower').value=clip?.[0]??.01;this.field('clip_upper').value=clip?.[1]??.99;
    this.field('remove_constant').checked=settings.variance_threshold!==null&&settings.variance_threshold!==undefined;
    this.field('variance_threshold').value=settings.variance_threshold??0;
    const match=Object.entries(PRESETS).find(([,preset])=>PREPROCESSING_KEYS.every(key=>JSON.stringify(settings[key]??null)===JSON.stringify(preset.settings[key]??null)));
    this.field('preset').value=match?.[0]||'custom';
    this.container.querySelector('.pw-preset-note').textContent=match?.[1].text||'Точные настройки заданы вручную. Проверьте их на графиках и сравните качество на проверочных данных.';
    this.refreshVisibility();
  }
  refreshVisibility(){
    const imputation=this.field('imputation').value,method=this.field('method').value;
    const states={constant:imputation==='constant',knn:imputation==='knn','clip-custom':this.field('clipping').value==='custom',variance:this.field('remove_constant').checked,selection:this.field('selection').value!=='none',sampling:method!=='none',neighbors:['smoter','smogn'].includes(method),smogn:method==='smogn'};
    for(const [key,visible]of Object.entries(states))this.container.querySelector(`[data-pw-section="${key}"]`).hidden=!visible;
    this.field('missing_indicator').disabled=this.busy||this.previewBusy||imputation==='none';
    this.field('interaction_only').disabled=this.busy||this.previewBusy||Number(this.field('degree').value)===1;
  }
  handleChange(event){
    const key=event.target.dataset.pw;if(!key)return;
    if(key.startsWith('preview-')){this.renderPreviewPlots();return;}
    if(this.busy||this.previewBusy)return;
    if(key==='preset'){
      const preset=PRESETS[event.target.value];if(!preset)return;
      this.preprocessing=clone(preset.settings);this.refreshControls();
    }else if(SAMPLING_KEYS.includes(key)){
      this.resampling[key]=event.target.type==='checkbox'?event.target.checked:['relevance_threshold','neighbors','perturbation'].includes(key)?Number(event.target.value):event.target.value;
      this.refreshVisibility();
    }else{
      if(key==='clipping'){
        this.preprocessing.clip_quantiles=event.target.value==='none'?null:event.target.value==='p01'?[.01,.99]:event.target.value==='p05'?[.05,.95]:[Number(this.field('clip_lower').value),Number(this.field('clip_upper').value)];
      }else if(['clip_lower','clip_upper'].includes(key))this.preprocessing.clip_quantiles=[Number(this.field('clip_lower').value),Number(this.field('clip_upper').value)];
      else if(key==='remove_constant')this.preprocessing.variance_threshold=event.target.checked?Number(this.field('variance_threshold').value):null;
      else if(key==='max_features')this.preprocessing.max_features=event.target.value.trim()===''?null:Number(event.target.value);
      else if(event.target.type==='checkbox')this.preprocessing[key]=event.target.checked;
      else if(event.target.type==='number'||key==='degree')this.preprocessing[key]=Number(event.target.value);
      else this.preprocessing[key]=event.target.value;
      if(key==='imputation'&&event.target.value==='none')this.preprocessing.missing_indicator=false;
      if(key==='selection'&&event.target.value==='none')this.preprocessing.max_features=null;
      this.refreshControls();
    }
    this.invalidatePreview();this.status('Настройки изменены. Предпросмотр покажет новые преобразования; обучение использует их автоматически.');this.onChange(this.getConfig());
  }

  invalidatePreview(){this.sequence++;this.preview=null;this.closePreview();}
  closePreview(){const dialog=this.container?.querySelector('.pw-preview');if(!dialog)return;if(dialog.open){if(typeof dialog.close==='function')dialog.close();else dialog.removeAttribute('open');}}
  async showPreview(){
    if(this.busy||this.previewBusy)return;
    if(!this.dataset){this.onError(new Error('Сначала выберите или создайте набор данных.'));return;}
    let request;try{request=this.getRequest();}catch(error){this.onError(error);return;}
    const seq=++this.sequence,id=this.dataset.id;
    this.previewBusy=true;this.setBusy(this.busy);this.status('Вычисляем преобразования только по обучающей части…');
    try{
      const result=await this.api(`/datasets/${encodeURIComponent(id)}/preprocessing-preview`,{method:'POST',body:request});
      if(seq!==this.sequence||this.dataset?.id!==id)return;
      this.preview=result;this.renderPreview();
      const dialog=this.container.querySelector('.pw-preview');if(!dialog.open){if(typeof dialog.showModal==='function')dialog.showModal();else dialog.setAttribute('open','');}
      this.status('Показаны реальные обучающие значения. Модель не обучалась.');
      await this.renderPreviewPlots();
    }catch(error){if(seq===this.sequence){this.status(`Не удалось подготовить данные: ${error.message||error}`);this.onError(error);}}
    finally{this.previewBusy=false;this.setBusy(this.busy);this.refreshVisibility();}
  }

  targetData(){
    const response=this.preview||{};
    const target=response.target;
    let request={};try{request=this.getRequest?.()||{};}catch{}
    return {name:(target&&typeof target==='object'?target.name:null)||response.target_name||request.target||'Цель',values:(target&&typeof target==='object'?target.values:null)||response.target_values||[]};
  }
  originalRows(){
    const actual=this.preview?.original_rows;
    if(Array.isArray(actual))return {rows:actual,paired:true};
    return {rows:this.dataset?.preview||[],paired:false};
  }
  renderPreview(){
    const result=this.preview,features=result.features||[],target=this.targetData(),original=this.originalRows(),numericNames=(this.dataset?.columns||[]).filter(c=>c.numeric&&(result.original_features||[]).includes(c.name)).map(c=>c.name);
    const plotAxes={...Object.fromEntries(features.map(name=>[name,`Признак · ${name}`])),...(target.values.length?{'__target__':`Цель · ${target.name}`}:{})};
    const sampler=result.sampling;
    this.container.querySelector('.pw-preview-body').innerHTML=`<p class="pw-preview-note">${esc(result.note||'Преобразования обучены только на обучающей части. Показаны первые 100 обучающих строк; модель не обучалась.')}</p>
      <div class="pw-preview-stats"><div><strong>${number(result.train_rows)}</strong><span>Исходных строк обучения</span></div><div><strong>${number((result.original_features||[]).length)}</strong><span>Исходных признаков</span></div><div><strong>${number(features.length)}</strong><span>Подготовленных столбцов</span></div>${sampler?`<div><strong>${number(sampler.before)} → ${number(sampler.after)}</strong><span>Строк до / после пересэмплирования</span></div>`:''}</div>
      <div class="pw-preview-card"><h3>Что произошло со значениями</h3><p class="pw-note">${original.paired?'Одни и те же обучающие строки до и после подготовки.':'Слева — отдельный пример из первых строк исходного набора, справа — обучающие строки. Это разные выборки.'} Оси имеют свои единицы: распределения показаны рядом, а не поверх друг друга.</p><div class="pw-preview-controls">${select('preview-original','Исходный признак',Object.fromEntries(numericNames.map(name=>[name,name])))}${select('preview-distribution','Подготовленный признак',Object.fromEntries(features.map(name=>[name,name])))}</div><div class="pw-plot" data-pw-plot="distribution"></div></div>
      <div class="pw-preview-card"><h3><span class="pw-feature">Признаки</span> и <span class="pw-target">цель</span> после подготовки</h3><p class="pw-note">Точки — реальные обучающие строки после преобразований, до пересэмплирования. Цвет показывает числовую цель, если она доступна.</p><div class="pw-preview-controls pw-preview-axes">${select('preview-mode','Вид',{2:'2D · две оси',3:'3D · три оси'})}${select('preview-x','Ось X',plotAxes)}${select('preview-y','Ось Y',plotAxes)}${select('preview-z','Ось Z',plotAxes)}</div><div class="pw-plot pw-plot-tall" data-pw-plot="scatter"></div></div>
      ${sampler&&sampler.method!=='none'?`<div class="pw-preview-card"><h3>Как изменился состав обучения</h3><p class="pw-note">${sampler.synthetic?'Метод создаёт синтетические обучающие примеры.':'Метод повторяет или удаляет целые обучающие строки.'} Гистограммы показывают фактически возвращённые значения цели (до 2000 для каждого состояния). Проверочные строки не изменяются.</p><div class="pw-plot" data-pw-plot="sampling"></div></div>`:''}
      <details class="pw-preview-card"><summary>Точные значения и имена столбцов</summary><p class="pw-note">Среднее и стандартное отклонение рассчитаны по всей обучающей части после подготовки, до пересэмплирования. Пример строк ниже ограничен первыми 20 из предпросмотра.</p><div class="pw-table-scroll"><table><thead><tr><th scope="col">Подготовленный столбец</th><th scope="col">Среднее на обучении</th><th scope="col">Стандартное отклонение</th></tr></thead><tbody>${features.map((name,i)=>`<tr><th scope="row" class="pw-feature">${esc(name)}</th><td>${number(result.means?.[i])}</td><td>${number(result.std?.[i])}</td></tr>`).join('')}</tbody></table></div><div class="pw-table-scroll"><table><thead><tr><th scope="col">Исходная строка</th>${features.slice(0,20).map(name=>`<th scope="col" class="pw-feature">${esc(name)}</th>`).join('')}${target.values.length?`<th scope="col" class="pw-target">${esc(target.name)}</th>`:''}</tr></thead><tbody>${(result.rows||[]).slice(0,20).map((row,i)=>`<tr><th scope="row">${number(result.indices?.[i]??i)}</th>${row.slice(0,20).map(value=>`<td>${number(value)}</td>`).join('')}${target.values.length?`<td class="pw-target">${number(target.values[i])}</td>`:''}</tr>`).join('')}</tbody></table></div>${features.length>20?'<p class="pw-note">В таблице примеров показаны первые 20 столбцов. Все имена и обучающие статистики доступны в таблице выше.</p>':''}</details>`;
    if(features.length){this.field('preview-x').value=features[0];this.field('preview-y').value=target.values.length?'__target__':features[1]||features[0];this.field('preview-z').value=target.values.length?'__target__':features[2]||features[0];this.field('preview-distribution').value=features[0];}
    if(numericNames.length)this.field('preview-original').value=numericNames[0];
    this.field('preview-z').closest('.pw-field').hidden=true;
    for(const key of ['preview-original','preview-distribution','preview-x','preview-y','preview-z'])attachHelp(this.help,this.field(key),key==='preview-original'?'features':'features');
  }
  layout(){const light=this.container.ownerDocument.body.dataset.theme==='light',color=this.container.ownerDocument.defaultView.getComputedStyle(this.container).getPropertyValue('--text').trim()||(light?'#172e3b':'#edf5f4');return {paper_bgcolor:'transparent',plot_bgcolor:'transparent',font:{family:'Inter,Arial,sans-serif',size:14,color},margin:{l:55,r:30,t:40,b:65},hovermode:'closest',legend:{orientation:'h',y:-.18},uirevision:this.dataset?.id};}
  refreshTheme(){return this.preview&&this.container?.querySelector('.pw-preview')?.open?this.renderPreviewPlots():Promise.resolve();}
  plot(name,traces,layout={}){
    const node=this.container.querySelector(`[data-pw-plot="${name}"]`);if(!node)return Promise.resolve();
    if(!window.Plotly){node.innerHTML='<p class="pw-empty">Не удалось загрузить библиотеку графиков. Точные значения доступны в таблице.</p>';return Promise.resolve();}
    return Promise.resolve(window.Plotly.react(node,traces,{...this.layout(),...layout},{responsive:true,displaylogo:false,locale:'ru',toImageButtonOptions:{format:'png',filename:`linear-lab-${name}`},scrollZoom:false}));
  }
  async renderPreviewPlots(){
    if(!this.preview)return;
    const result=this.preview,features=result.features||[],rows=result.rows||[],target=this.targetData();
    const originalName=this.field('preview-original')?.value,transformedName=this.field('preview-distribution')?.value;
    const originalValues=this.originalRows().rows.map(row=>row[originalName]).filter(value=>typeof value==='number'&&Number.isFinite(value));
    const afterIndex=features.indexOf(transformedName),afterValues=rows.map(row=>row[afterIndex]).filter(Number.isFinite);
    const axis=title=>({title:{text:title},gridcolor:'rgba(148,168,179,.14)',zerolinecolor:'rgba(148,168,179,.3)',automargin:true});
    const tasks=[];
    tasks.push(this.plot('distribution',[{type:'histogram',x:originalValues,name:'До подготовки',marker:{color:'#a9a1ff'},xaxis:'x',yaxis:'y',hovertemplate:'Значение: %{x}<br>Строк: %{y}<extra>До</extra>'},{type:'histogram',x:afterValues,name:'После подготовки',marker:{color:FEATURE},xaxis:'x2',yaxis:'y2',hovertemplate:'Значение: %{x}<br>Строк: %{y}<extra>После</extra>'}],{xaxis:{...axis(originalName||'Нет числовых исходных признаков'),domain:[0,.44]},xaxis2:{...axis(transformedName||'Нет подготовленных признаков'),domain:[.56,1]},yaxis:{...axis('Строк в примере'),domain:[0,1]},yaxis2:{...axis('Строк в примере'),domain:[0,1],anchor:'x2'},annotations:[{text:'До',x:.22,y:1.15,xref:'paper',yref:'paper',showarrow:false},{text:'После',x:.78,y:1.15,xref:'paper',yref:'paper',showarrow:false}],barmode:'overlay'}));
    const mode=this.field('preview-mode').value,xName=this.field('preview-x').value,yName=this.field('preview-y').value,zName=this.field('preview-z').value;
    this.field('preview-z').closest('.pw-field').hidden=mode!=='3';
    const values=name=>name==='__target__'?target.values:rows.map(row=>row[features.indexOf(name)]);
    const label=name=>name==='__target__'?target.name:name;
    const x=values(xName),y=values(yName),z=values(zName);
    const indexes=rows.map((_,i)=>i).filter(i=>Number.isFinite(x[i])&&Number.isFinite(y[i])&&(mode!=='3'||Number.isFinite(z[i])));
    const colors=indexes.map(i=>target.values[i]);
    const hasColors=target.values.length>0&&colors.every(Number.isFinite);
    const marker={size:mode==='3'?4:7,opacity:.8,color:hasColors?colors:FEATURE};
    if(hasColors)Object.assign(marker,{colorscale:[[0,'#79b9ef'],[.5,'#c0d8c6'],[1,TARGET]],showscale:true,colorbar:{title:{text:target.name},thickness:12}});
    const trace={type:mode==='3'?'scatter3d':'scatter',mode:'markers',name:'Обучающие строки',x:indexes.map(i=>x[i]),y:indexes.map(i=>y[i]),marker,customdata:indexes.map(i=>[result.indices?.[i]??i,target.values[i]]),hovertemplate:`Строка %{customdata[0]}<br>${esc(label(xName))}: %{x}<br>${esc(label(yName))}: %{y}${mode==='3'?'<br>'+esc(label(zName))+': %{z}':''}${hasColors?'<br>Цель: %{customdata[1]}':''}<extra></extra>`};
    if(mode==='3')trace.z=indexes.map(i=>z[i]);
    const light=this.container.ownerDocument.body.dataset.theme==='light';
    const coloredAxis=name=>({...axis(label(name)),title:{text:label(name),font:{color:name==='__target__'?(light?'#a63d2a':TARGET):(light?'#076950':FEATURE)}}});
    tasks.push(this.plot('scatter',[trace],mode==='3'?{scene:{xaxis:coloredAxis(xName),yaxis:coloredAxis(yName),zaxis:coloredAxis(zName),bgcolor:'transparent',camera:{eye:{x:1.3,y:1.5,z:1}}},margin:{l:0,r:hasColors?65:0,t:10,b:15}}:{xaxis:coloredAxis(xName),yaxis:coloredAxis(yName),margin:{l:60,r:hasColors?90:20,t:15,b:65}}));
    if(result.sampling&&result.sampling.method!=='none'){
      const sample=result.sampling;
      tasks.push(this.plot('sampling',[{type:'histogram',x:sample.target_before||[],name:`До · ${number(sample.before)} строк`,histnorm:'probability',marker:{color:'#a9a1ff'},opacity:.65},{type:'histogram',x:sample.target_after||[],name:`После · ${number(sample.after)} строк`,histnorm:'probability',marker:{color:TARGET},opacity:.65}],{barmode:'overlay',xaxis:axis(target.name),yaxis:axis('Доля в показанном примере'),legend:{orientation:'h',y:-.2}}));
    }
    const checks=await Promise.allSettled(tasks);for(const check of checks)if(check.status==='rejected')this.onError(check.reason);
  }
  destroy(){this.sequence++;this.help?.destroy();this.container?.removeEventListener('change',this.onInput);this.container?.removeEventListener('click',this.onClick);if(window.Plotly)this.container?.querySelectorAll('[data-pw-plot]').forEach(node=>window.Plotly.purge(node));}
}
