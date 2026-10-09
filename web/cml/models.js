import { element, badge } from './dom.js';
import { action, field, heading, notice, select, setOptions } from './controls.js';
import { SchemaForm } from './schema-form.js';

export class ModelBuilder {
  constructor({ catalogue, onChange, onError }) {
    this.catalogue = catalogue; this.onChange = onChange; this.onError = onError;
    this.task = 'regression'; this.modelId = ''; this.savedParams = {}; this.family = '';
    this.parameterQuery = ''; this.advancedOpen = false;
  }

  mount(container) { this.container = container; this.render(); return this; }
  algorithms() { return (this.catalogue.algorithms || []).filter(model => model.tasks?.includes(this.task)); }
  selected() { return this.algorithms().find(model => model.id === this.modelId); }
  values() { return { algorithm_id: this.modelId, params: this.form?.values() || this.savedParams }; }

  setTask(task) {
    this.savedParams = this.form?.values() || this.savedParams;
    this.task = task; this.family = '';
    if (!this.algorithms().some(model => model.id === this.modelId)) { this.modelId = this.algorithms().find(model => model.available !== false)?.id || ''; this.savedParams = {}; }
    this.render();
  }

  setConfig(config = {}) {
    this.modelId = config.algorithm_id || this.modelId; this.family = '';
    this.savedParams = structuredClone(config.params || {});
    this.render();
    this.onChange(this.selected());
  }


  organizeParameters(container, schema) {
    const query = element('input', { type: 'search', value: this.parameterQuery, placeholder: 'Найти гиперпараметр: max_depth, alpha…', 'aria-label': 'Поиск гиперпараметров' });
    const status = element('p', { className: 'cml-note', role: 'status', 'aria-live': 'polite' });
    const basic = element('div', { className: 'cml-form-grid', 'data-parameter-group': 'basic' });
    const advancedGrid = element('div', { className: 'cml-form-grid', 'data-parameter-group': 'advanced' });
    const advancedSummary = element('summary', { text: 'Дополнительные параметры' });
    const advanced = element('details', { className: 'cml-parameter-advanced', open: this.advancedOpen }, [advancedSummary, advancedGrid]);
    const rows = [];
    for (const definition of schema) {
      const control = this.form.controls.get(definition.key)?.control;
      const wrapper = control?.closest('.cml-field');
      if (!wrapper) continue;
      const extra = definition.advanced === true;
      (extra ? advancedGrid : basic).append(wrapper);
      rows.push({ definition, wrapper, extra });
    }
    let applyingFilter = false;
    const applyFilter = () => {
      applyingFilter = true;
      const term = query.value.trim().toLocaleLowerCase('ru');
      let count = 0, extraCount = 0;
      for (const row of rows) {
        const searchable = `${row.definition.key} ${row.definition.help || ''} ${row.definition.description || ''}`.toLocaleLowerCase('ru');
        const visible = !term || searchable.includes(term);
        row.wrapper.hidden = !visible;
        row.wrapper.style.display = visible ? '' : 'none';
        if (visible) { count += 1; if (row.extra) extraCount += 1; }
      }
      status.textContent = `Показано параметров: ${count} из ${rows.length}. Скрытые поля сохраняют свои значения.`;
      advanced.hidden = !extraCount;
      advanced.style.display = extraCount ? '' : 'none';
      advancedSummary.textContent = `Дополнительные параметры (${extraCount})`;
      advanced.open = Boolean(term && extraCount) || this.advancedOpen;
      applyingFilter = false;
    };
    advanced.addEventListener('toggle', () => { if (!applyingFilter && !query.value.trim()) this.advancedOpen = advanced.open; });
    query.addEventListener('input', () => { this.parameterQuery = query.value; applyFilter(); });
    container.replaceChildren(query, status, basic, advanced);
    applyFilter();
  }

  render() {
    if (!this.container) return;
    const models = this.algorithms();
    if (!models.some(model => model.id === this.modelId)) this.modelId = models.find(model => model.available !== false)?.id || '';
    const families = [...new Map(models.map(model => [model.family, model.family_label || model.family])).entries()];
    if (this.family && !families.some(([family]) => family === this.family)) this.family = '';
    const filter = select([{ value: '', label: 'Все семейства' }, ...families.map(([value, label]) => ({ value, label }))], this.family);
    const modelChoice = select([], this.modelId);
    const fillModels = () => setOptions(modelChoice, models.filter(model => !filter.value || model.family === filter.value).map(model => ({ value: model.id, label: `${model.name}${model.available === false ? ' — недоступен' : ''}`, disabled: model.available === false })), this.modelId);
    fillModels();
    filter.addEventListener('change', () => { this.family = filter.value; fillModels(); if (modelChoice.value !== this.modelId) modelChange(); });
    const modelChange = () => { this.modelId = modelChoice.value; this.savedParams = {}; this.render(); this.onChange(this.selected()); };
    modelChoice.addEventListener('change', modelChange);
    const selected = this.selected();
    const controls = element('div');
    this.form = new SchemaForm({ onChange: () => this.onChange(this.selected()), definitions: this.catalogue.algorithms || [], task: this.task }).mount(controls, selected?.params || [], this.savedParams);
    this.organizeParameters(controls, selected?.params || []);
    const capabilityLabels = {
      predict_proba: 'Вероятности классов', decision_function: 'Оценка уверенности', coefficients: 'Коэффициенты', feature_importance: 'Важность признаков',
      partial_fit: 'Обучение порциями', staged_predict: 'Промежуточные прогнозы', transductive: 'Только исходные точки', predict: 'Прогноз для новых строк',
    };
    this.container.replaceChildren(element('div', { className: 'cml-workspace wide-form' }, [
      element('section', { className: 'cml-panel' }, [
        heading('Алгоритм', { help: 'Алгоритм описывает, как модель учится на данных. Готовый шаблон хранит его настройки, а обученная модель хранит найденные параметры.', lesson_id: selected?.lesson_id || '01-prediction' }),
        field('Семейство моделей', filter, { help: 'Семейство объединяет алгоритмы с похожим способом работы: деревья, соседи, линейные модели, ансамбли.', lesson_id: selected?.lesson_id || '01-prediction' }),
        field('Модель для выбранной задачи', modelChoice, { help: selected?.description || 'Выберите алгоритм, совместимый с задачей.', lesson_id: selected?.lesson_id || '01-prediction' }),
        element('p', { className: 'cml-note', text: selected?.description || '' }),
        element('div', { className: 'cml-badges' }, [badge(selected?.library || selected?.backend || 'scikit-learn'), ...Object.entries(selected?.capabilities || {}).filter(([key, enabled]) => enabled === true && capabilityLabels[key]).map(([key]) => badge(capabilityLabels[key]))]),
        ...(selected?.available === false ? [notice(selected.reason || 'Библиотека этого алгоритма не установлена.', true)] : []),
        notice('Начните с параметров по умолчанию. Меняйте одну настройку и сравнивайте результат. Подробное объяснение каждой настройки открывается по вопросительному знаку.'),
      ]),
      element('section', { className: 'cml-panel' }, [heading('Параметры модели', { help: 'Эти параметры задаются до обучения. Найденные моделью веса, деревья и центры появляются после расчета.', lesson_id: selected?.lesson_id || '27-hyperparameter-search' }), controls,
        action('Восстановить параметры по умолчанию', () => { this.savedParams = {}; this.render(); this.onChange(this.selected()); }),
      ]),
    ]));
  }
}
