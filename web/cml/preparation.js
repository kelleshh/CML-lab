import { element, formatNumber } from './dom.js';
import { action, field, heading, notice, select, setOptions } from './controls.js';
import { SchemaForm } from './schema-form.js';
import { helpButton } from './help.js';

export class PreparationBuilder {
  constructor({ api, getRequest, onChange, onError, catalogue = {} }) {
    this.api = api; this.getRequest = getRequest; this.onChange = onChange; this.onError = onError;
    this.catalogue = catalogue; this.steps = []; this.forms = []; this.dataset = null; this.task = 'regression';
    this.resampling = { method: 'none' };
  }

  mount(container) { this.container = container; this.render(); return this; }
  setCatalogue(catalogue) { this.catalogue = catalogue; this.render(); }
  setDataset(dataset) { this.capture(); this.dataset = dataset; this.render(); }
  setTask(task) { this.capture(); this.task = task; this.render(); }

  descriptors() { return this.catalogue.preprocessors || this.catalogue.stages || []; }
  samplerDescriptors() {
    return this.catalogue.samplers || [
      { id: 'none', name: 'Не менять количество строк', allowed_tasks: [] },
      { id: 'random_over', name: 'Повторять редкие строки', allowed_tasks: ['regression', 'classification'] },
      { id: 'random_under', name: 'Убирать часть частых строк', allowed_tasks: ['regression', 'classification'] },
      { id: 'smote', name: 'SMOTE: создавать редкие примеры', allowed_tasks: ['classification'] },
      { id: 'smoter', name: 'SMOTER: редкие значения цели', allowed_tasks: ['regression'] },
      { id: 'smogn', name: 'SMOGN: редкие значения и шум', allowed_tasks: ['regression'] },
    ];
  }

  values() {
    return { steps: this.steps.map((step, index) => ({ ...step, params: this.forms[index]?.values() || step.params || {} })), resampling: { ...this.resampling, ...(this.samplerForm?.values() || {}) } };
  }

  setConfig(config = {}) {
    this.steps = structuredClone(config.steps || []);
    this.resampling = structuredClone(config.resampling || { method: 'none' });
    this.render();
  }

  capture() { this.steps = this.values().steps; this.resampling = this.values().resampling; }

  render() {
    if (!this.container) return;
    const allowed = this.descriptors().filter(item => !item.allowed_tasks?.length || item.allowed_tasks.includes(this.task));
    const chooser = select(allowed.map(item => ({ value: item.id, label: `${item.name || item.id}${item.available === false ? ' — недоступен' : ''}`, disabled: item.available === false })), allowed.find(item => item.available !== false)?.id || '');
    const stageList = element('div', { className: 'cml-preparation-stages' });
    this.forms = [];
    this.steps.forEach((step, index) => {
      const descriptor = this.descriptors().find(item => item.id === step.adapter_id);
      const enabled = element('input', { type: 'checkbox', checked: step.enabled !== false, 'aria-label': `Включить этап ${index + 1}: ${descriptor?.name || step.adapter_id}` });
      enabled.addEventListener('change', () => { step.enabled = enabled.checked; this.onChange(); });
      const columns = element('div', { className: 'cml-column-checks' });
      const candidates = (this.dataset?.columns || []).filter(column => {
        const types = descriptor?.accepted_column_types;
        return !types?.length || types.includes('all') || types.includes(column.numeric ? 'numeric' : 'categorical') || types.includes(column.numeric ? 'number' : 'string');
      });
      for (const column of candidates) {
        const control = element('input', { type: 'checkbox', checked: (step.columns || []).includes(column.name) });
        control.addEventListener('change', () => {
          step.columns = [...columns.querySelectorAll('input:checked')].map(input => input.value);
          this.onChange();
        });
        control.value = column.name;
        columns.append(element('label', { className: 'cml-check' }, [control, element('span', { className: 'feature-label', text: column.name })]));
      }
      const formContainer = element('div');
      const schema = descriptor?.params || descriptor?.parameters || [];
      const form = new SchemaForm({ onChange: () => this.onChange() }).mount(formContainer, schema, step.params || {});
      this.forms[index] = form;
      const move = direction => {
        this.capture();
        const target = index + direction;
        [this.steps[index], this.steps[target]] = [this.steps[target], this.steps[index]];
        this.render(); this.onChange();
      };
      const actions = element('div', { className: 'cml-toolbar' }, [
        action('↑', () => move(-1), '', { disabled: index === 0, 'aria-label': 'Передвинуть этап выше' }),
        action('↓', () => move(1), '', { disabled: index === this.steps.length - 1, 'aria-label': 'Передвинуть этап ниже' }),
        action('Удалить этап', () => { this.capture(); this.steps.splice(index, 1); this.render(); this.onChange(); }, 'danger'),
      ]);
      const columnDetails = element('details', {}, [element('summary', { text: 'Выбрать столбцы; пустой выбор означает все подходящие' }), columns]);
      stageList.append(element('section', { className: 'cml-panel' }, [
        element('div', { className: 'cml-panel-heading' }, [element('label', { className: 'cml-check' }, [enabled, element('h3', { text: `${index + 1}. ${descriptor?.name || step.adapter_id}` })]), helpButton({ label: descriptor?.name || step.adapter_id, help: descriptor?.description, lesson_id: descriptor?.lesson_id })]),
        element('p', { className: 'cml-note', text: descriptor?.description || 'Настройте этап подготовки данных.' }), columnDetails, formContainer, actions,
        ...(descriptor?.available === false ? [notice(descriptor.reason || 'Этот адаптер недоступен.', true)] : []),
      ]));
    });
    const samplerDescriptors = this.samplerDescriptors();
    const samplers = samplerDescriptors.filter(item => !item.allowed_tasks?.length || item.allowed_tasks.includes(this.task));
    if (!samplers.some(item => item.id === this.resampling.method)) this.resampling = { method: 'none' };
    const sampler = select(samplers.map(item => ({ value: item.id, label: item.name || item.id, disabled: item.available === false })), this.resampling.method);
    const samplerFormContainer = element('div');
    const selectedSampler = samplers.find(item => item.id === this.resampling.method);
    this.samplerForm = new SchemaForm({ onChange: () => this.onChange() }).mount(samplerFormContainer, selectedSampler?.params || [], this.resampling);
    sampler.addEventListener('change', () => { this.capture(); this.resampling = { method: sampler.value }; this.render(); this.onChange(); });
    const preview = action('Посмотреть данные после подготовки', () => this.preview(), 'primary');
    this.previewButton = preview;
    this.previewContainer = element('div', { className: 'cml-panel' }, [heading('Предпросмотр подготовки', { help: 'Преобразования обучаются только на тренировочных строках. Проверочные строки не участвуют в расчете средних, категорий и новых примеров.', lesson_id: '25-data-leakage' }), element('p', { className: 'cml-empty', text: 'Добавьте этапы и нажмите кнопку предпросмотра. Здесь будут реальные преобразованные значения.' })]);
    this.container.replaceChildren(element('div', { className: 'cml-workspace wide-form' }, [
      element('div', {}, [
        element('section', { className: 'cml-panel' }, [heading('Конструктор подготовки', { help: 'Этапы выполняются по порядку. Сохраняйте удачную последовательность как препроцессор в библиотеке.', lesson_id: '24-feature-engineering' }),
          field('Добавить преобразование', chooser, { help: 'Каждый этап меняет определенные столбцы или отбирает признаки. Для категориального текста доступны свои адаптеры.', lesson_id: '24-feature-engineering' }),
          action('Добавить этап', () => { this.capture(); if (!chooser.value) return; this.steps.push({ adapter_id: chooser.value, enabled: true, columns: [], params: {} }); this.render(); this.onChange(); }),
          notice('Если этапы не добавлены, используются безопасные базовые преобразования. Точные правила показаны в предпросмотре и паспорте запуска.'),
        ]), stageList,
        element('section', { className: 'cml-panel' }, [heading('Баланс обучающих строк', { help: 'Добавление и удаление строк относится только к обучающей части. У проверочных данных сохраняется исходное распределение.', lesson_id: this.task === 'classification' ? 'classification-resampling' : '28-rare-target-smoter' }),
          field('Изменение количества строк', sampler, { help: 'Повторение, удаление или создание строк может помочь редким ответам. Сравните результат без изменения данных.', lesson_id: this.task === 'classification' ? 'classification-resampling' : '28-rare-target-smoter' }), samplerFormContainer,
        ]), element('div', { className: 'cml-toolbar' }, [preview, action('Очистить этапы', () => { this.setConfig({}); this.onChange(); })]),
      ]), this.previewContainer,
    ]));
  }

  async preview() {
    this.previewButton.disabled = true;
    try {
      const result = await this.api.request('/cml/preprocessing/preview', { method: 'POST', body: this.getRequest() });
      const columns = result.features || result.feature_names || result.columns || [];
      const rows = result.rows || result.preview?.rows || [];
      const table = element('table', { className: 'cml-table' }, [
        element('thead', {}, [element('tr', {}, columns.map(column => element('th', { text: typeof column === 'string' ? column : column.name })))]),
        element('tbody', {}, rows.slice(0, 100).map(row => element('tr', {}, (Array.isArray(row) ? row : columns.map(column => row[typeof column === 'string' ? column : column.name])).map(value => element('td', { text: typeof value === 'number' ? formatNumber(value) : String(value ?? '—') }))))),
      ]);
      this.previewContainer.replaceChildren(heading('После подготовки', { help: 'Таблица показывает настоящие преобразованные обучающие строки. Сохраняйте значение целевой переменной отдельно от признаков.', lesson_id: '24-feature-engineering' }),
        element('p', { className: 'cml-note', text: `${result.train_rows || result.n_samples || rows.length} обучающих строк · ${columns.length} преобразованных признаков` }),
        ...(result.note ? [notice(result.note)] : []), element('div', { className: 'cml-table-scroll' }, [table]));
      for (const warning of result.warnings || []) this.previewContainer.append(notice(warning));
    } catch (error) { this.onError(error); }
    finally { this.previewButton.disabled = false; }
  }
}
