/** Small, explicit choices which write the same authoritative model parameters. */
import {HELP_REGISTRY, MODEL_LESSONS} from './help.js';

const option = (label, value, detail) => ({label, value, detail});
export const BEGINNER_PRESETS = Object.freeze({
  l1_ratio: [option('Больше L2',0.2),option('Поровну',0.5),option('Больше L1',0.8)],
  n_nonzero_coefs: [option('Один',1),option('До трёх',3),option('До пяти',5)],
  max_features: [option('Только константа',0),option('До трёх',3),option('До пяти',5)],
  max_trials: [option('Быстро',50),option('Обычный бюджет',200),option('Больше попыток',1000)],
  max_subpopulation: [option('Быстро',100),option('Обычный бюджет',1000),option('Больше поднаборов',5000)],
  quantile: [option('Нижний',0.1),option('Медиана',0.5),option('Верхний',0.9)],
  power: [option('Нормальная ошибка',0),option('Пуассон',1),option('Между 1 и 2',1.5),option('Gamma',2)],
  group_size: [option('По одному',1),option('Пары',2),option('Тройки',3)],
  C: [option('Меньше цена ошибок',0.1),option('Исходная',1),option('Больше цена ошибок',10)],
  operator: [option('Величина весов','identity'),option('Соседние разности','first_difference'),option('Вторая разность','second_difference')],
  penalty: [option('Без штрафа','none'),option('L2','l2'),option('L1','l1'),option('Смесь','elasticnet')],
});

const PRIMARY = {
  ols:['fit_intercept'], nonnegative:['fit_intercept'],
  ridge:['alpha'],lasso:['alpha'],elasticnet:['alpha','l1_ratio'],
  lars:['n_nonzero_coefs'],lassolars:['alpha'],omp:['n_nonzero_coefs'],
  sgd:['alpha','penalty','eta0'],ransac:['max_trials','residual_threshold'],
  theilsen:['max_subpopulation'],huber:['epsilon','alpha'],
  bayesian_ridge:['fit_intercept'],ard:['threshold_lambda'],
  quantile:['quantile','alpha'],poisson:['alpha'],gamma:['alpha'],tweedie:['power','alpha'],
  linear_svr:['C','epsilon'],weighted_lasso:['alpha','weight_ratio'],
  adaptive_lasso:['alpha','gamma'],group_lasso:['alpha','group_size'],
  sparse_group_lasso:['alpha','l1_ratio','group_size'],fused_lasso:['alpha','fusion_strength'],
  tikhonov:['alpha','operator'],l0:['alpha','max_features'],scad:['alpha','gamma'],
  mcp:['alpha','gamma'],sqrt_lasso:['alpha'],
};

function numberLabel(value) {
  return new Intl.NumberFormat('ru-RU',{maximumSignificantDigits:5}).format(value);
}

function boundedOptions(definition, list) {
  const values = new Set();
  return list.filter(item => {
    if (definition.type === 'select') {
      const allowed = (definition.options || []).map(opt => typeof opt === 'object' ? opt.value : opt);
      if (!allowed.includes(item.value)) return false;
    } else if (definition.type !== 'bool') {
      if (typeof item.value !== 'number' || !Number.isFinite(item.value)) return false;
      if (definition.min != null && item.value < definition.min || definition.max != null && item.value > definition.max) return false;
      if (definition.type === 'int' && !Number.isInteger(item.value)) return false;
    }
    if (values.has(item.value)) return false;
    values.add(item.value);
    return true;
  });
}

export function presetOptions(model, definition, context = {}) {
  const key = definition.key;
  const base = Number(definition.default);
  let list;
  if (key === 'alpha' || key === 'fusion_strength') {
    const initial = base > 0 ? base : 0.1;
    list = [option('Слабее',initial / 10),option('Исходный',initial),option('Сильнее',initial * 10)];
    if (key === 'alpha' && ['ridge','poisson','gamma','tweedie','tikhonov','huber','l0'].includes(model.id)) list.unshift(option('Без этого штрафа',0));
  } else if (key === 'epsilon' && model.id === 'huber') {
    list = [option('Раньше ограничивать выбросы',1.1),option('Исходный',1.35),option('Позже ограничивать выбросы',2)];
  } else if (key === 'epsilon') {
    const initial = base > 0 ? base : 0.1;
    list = [option('Узкая полоса',initial / 10),option('Исходная',initial),option('Широкая полоса',initial * 10)];
  } else if (key === 'gamma') {
    list = model.id === 'scad' ? [option('3',3),option('Исходный: 3.7',3.7),option('6',6)] :
      model.id === 'mcp' ? [option('2',2),option('Исходный: 3',3),option('6',6)] :
      [option('Меньше различий',0.5),option('Исходный',1),option('Больше различий',2)];
  } else if (key === 'residual_threshold') {
    list = [option('Автоматически',0)];
    // Absolute residual units cannot be chosen honestly without target scale.
    if (Number.isFinite(context.targetScale) && context.targetScale > 0) list.push(
      option('Меньше допуск',context.targetScale * 0.5),option('Больше допуск',context.targetScale * 2));
  } else if (key === 'threshold_lambda') {
    list = [option('Раньше удалять',1000),option('Исходный',10000),option('Позже удалять',100000)];
  } else if (key === 'eta0') {
    list = [option('Меньший шаг',0.001),option('Исходный',0.01),option('Больший шаг',0.03)];
  } else if (key === 'weight_ratio') {
    list = [option('Одинаковый вес',1),option('Разница в 3 раза',3),option('Разница в 10 раз',10)];
  } else if (definition.type === 'bool') {
    list = [option('Включено',true),option('Выключено',false)];
  } else {
    list = BEGINNER_PRESETS[key] || [];
  }
  if (key === 'n_nonzero_coefs' && Number.isInteger(context.nFeatures)) list = list.filter(item => item.value <= context.nFeatures);
  return boundedOptions(definition,list);
}

/**
 * Render limited model choices without keeping a second source of parameter truth.
 * onChange receives a patch object, e.g. {alpha:0.1}; caller writes existing inputs.
 * Optional nFeatures is transformed feature count, targetScale is target units.
 */
export function renderBeginnerControls(container, model, options = {}) {
  if (!container) return {destroy(){},setParams(){}};
  const doc = container.ownerDocument;
  container.replaceChildren();
  const wrapper = doc.createElement('div');
  wrapper.className = 'beginner-controls';
  container.append(wrapper);
  const spec = typeof model === 'object' && model ? model : {id:String(model || ''),params:[]};
  let current = {...Object.fromEntries((spec.params || []).map(item => [item.key,item.default])),...options.params};
  const groups = [];
  const primary = PRIMARY[spec.id] || (spec.params || []).filter(item => !['max_iter','tol','solver'].includes(item.key)).slice(0,3).map(item => item.key);
  for (const key of primary) {
    const definition = spec.params?.find(item => item.key === key);
    if (!definition) continue;
    const choices = presetOptions(spec,definition,options);
    if (!choices.length) continue;
    const section = doc.createElement('div');
    section.className = 'beginner-setting';
    const label = doc.createElement('div');
    label.className = 'beginner-label';
    label.textContent = definition.label || key;
    label.dataset.help = HELP_REGISTRY[key] ? key : '';
    label.dataset.helpModelParam = key;
    if (!HELP_REGISTRY[key]) {
      label.dataset.helpText = definition.help || spec.description || '';
      label.dataset.helpTitle = definition.label || key;
      label.dataset.helpLesson = MODEL_LESSONS[spec.id] || '20-metrics-experiment';
      label.dataset.helpSource = spec.source || '';
    }
    section.append(label);
    const buttons = doc.createElement('div');
    buttons.className = 'beginner-options';
    buttons.setAttribute('role','group');
    buttons.setAttribute('aria-label',definition.label || key);
    const items = [];
    for (const choice of choices) {
      const button = doc.createElement('button');
      button.type = 'button';
      button.className = 'beginner-option';
      button.textContent = choice.label;
      button.dataset.presetKey = key;
      button.dataset.presetValue = String(choice.value);
      button.addEventListener('click', () => {
        current[key] = choice.value;
        update();
        options.onChange?.({[key]:choice.value});
      });
      buttons.append(button);
      items.push({button,choice});
    }
    section.append(buttons);
    const value = doc.createElement('div');
    value.className = 'beginner-value';
    value.setAttribute('aria-live','polite');
    section.append(value);
    wrapper.append(section);
    groups.push({key,definition,items,value});
  }
  const note = doc.createElement('p');
  note.className = 'beginner-note';
  if (spec.params?.some(item => item.key === 'alpha')) {
    note.textContent = 'Названия «слабее» и «сильнее» сравнивают варианты только внутри этой модели. Одинаковое α у Ridge и Lasso означает разное соотношение ошибки и штрафа. Начните с включённого масштабирования; лучший вариант выбирайте по проверке.';
  } else if (['bayesian_ridge','ard'].includes(spec.id)) {
    note.textContent = 'Эта модель сама оценивает силы сжатия по обучающим данным. Ручного α, как у Ridge, здесь нет.';
  } else if (spec.id === 'ransac') {
    note.textContent = 'Порог зависит от единиц цели. Автоматический вариант — начальная настройка; точное значение можно задать в расширенном режиме.';
  } else {
    note.textContent = 'Готовые варианты — отправная точка. Сравните проверочные метрики на одинаковых данных и разделении. Точные численные настройки доступны в расширенном режиме.';
  }
  wrapper.append(note);
  if (!groups.length) {
    const text = doc.createElement('p');
    text.className = 'beginner-note';
    text.textContent = 'У этой модели нет простых ручных настроек из каталога. Используйте исходные значения или откройте расширенный режим.';
    wrapper.prepend(text);
  }
  if (typeof options.onMode === 'function') {
    const advanced = doc.createElement('button');
    advanced.type = 'button';
    advanced.className = 'text-button';
    advanced.textContent = 'Перейти к точным настройкам';
    advanced.onclick = () => options.onMode('advanced');
    wrapper.append(advanced);
  }
  function update() {
    for (const group of groups) {
      const {key,definition,items,value} = group;
      let found = false;
      for (const item of items) {
        const matches = Object.is(current[key], item.choice.value);
        item.button.setAttribute('aria-pressed',String(matches));
        found ||= matches;
      }
      const shown = typeof current[key] === 'number' ? numberLabel(current[key]) : typeof current[key] === 'boolean' ? (current[key] ? 'да' : 'нет') : String(current[key]);
      value.textContent = `${definition.label || key}: ${shown}${found ? '' : ' · своё значение из расширенного режима'}`;
      value.classList.toggle('beginner-custom',!found);
    }
  }
  update();
  return {
    setParams(patch) {current = {...current,...patch}; update();},
    destroy() {wrapper.remove();},
  };
}
