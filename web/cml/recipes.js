import { element, formatNumber } from './dom.js';
import { action, field, notice, checkbox } from './controls.js';

const RESOURCES = {
  'model-recipes': { title: 'Шаблоны моделей', singular: 'Шаблон модели', description: 'Алгоритм и его настройки. Обученный результат хранится отдельно.' },
  'preprocessor-recipes': { title: 'Препроцессоры', singular: 'Препроцессор', description: 'Сохраненная последовательность подготовки данных.' },
  projects: { title: 'Проекты', singular: 'Проект', description: 'Данные, задача, модель, подготовка и правила проверки.' },
};

export class RecipeLibrary {
  constructor({ api, getConfig, applyConfig, notify, onError }) {
    this.api = api; this.getConfig = getConfig; this.applyConfig = applyConfig;
    this.notify = notify; this.onError = onError; this.kind = 'projects';
    this.selection = null; this.items = []; this.busy = false; this.copy = null; this.query = '';
  }

  mount(container) { this.container = container; this.render(); return this; }

  async open(kind) {
    this.kind = kind; this.selection = null; this.copy = null; this.query = '';
    await this.refresh();
  }

  async refresh() {
    const payload = await this.api.request(`/cml/${this.kind}`);
    this.items = Array.isArray(payload) ? payload : payload.items || [];
    if (this.selection) this.selection = this.items.find(item => item.id === this.selection.id) || null;
    this.render();
  }

  async guarded(callback) {
    if (this.busy) return;
    this.busy = true; this.container?.querySelectorAll('.cml-button').forEach(button => { button.disabled = true; });
    try { await callback(); } catch (error) { this.onError(error); }
    finally { this.busy = false; this.container?.querySelectorAll('.cml-button').forEach(button => { button.disabled = false; }); }
  }

  render() {
    if (!this.container) return;
    const resource = RESOURCES[this.kind];
    const search = element('input', { type: 'search', placeholder: 'Название или описание', value: this.query, 'aria-label': `Найти: ${resource.title}` });
    const list = element('div', { className: 'cml-entity-list' });
    const renderList = () => {
      const query = search.value.toLowerCase().trim();
      const items = this.items.filter(item => `${item.name} ${item.description}`.toLowerCase().includes(query));
      list.replaceChildren(...(items.length ? items.map(item => element('button', {
        className: 'cml-entity', 'aria-current': String(this.selection?.id === item.id), onclick: () => { this.selection = item; this.copy = null; this.render(); },
      }, [element('strong', { text: item.name }), element('small', { text: `${item.description || resource.singular} · редакция ${item.revision}` })])) : [element('p', { className: 'cml-empty', text: this.items.length ? 'Нет объектов, соответствующих поиску.' : 'Сохраненных объектов пока нет. Создайте объект из текущих настроек эксперимента.' })]));
    };
    search.addEventListener('input', () => { this.query = search.value; renderList(); });
    renderList();
    const name = element('input', { value: this.selection?.name || (this.copy ? `${this.copy.name} · копия` : ''), maxlength: 150, placeholder: resource.singular, required: true });
    const description = element('textarea', { value: this.selection?.description || this.copy?.description || '', rows: 3, maxlength: 2000 });
    const explanation = element('p', { className: 'cml-note', text: resource.description });
    const summary = element('div', { className: 'cml-code', text: this.selection || this.copy ? this.describe((this.selection || this.copy).config) : 'При создании сохранится текущая конфигурация в конструкторе эксперимента.' });
    const selectedId = this.selection?.id;
    const replaceConfig = checkbox('Заменить параметры текущими настройками конструктора', false, { help: 'Если выключено, изменение названия и описания сохраняет исходные настройки объекта. Если включено, параметры будут заменены настройками открытого эксперимента.', lesson_id: '29-model-export' });
    const save = action(selectedId ? 'Обновить объект' : 'Создать из текущих настроек', () => this.guarded(async () => {
      if (!name.reportValidity()) return;
      const config = selectedId && !replaceConfig.control.checked ? this.selection.config : this.copy?.config || this.getConfig(this.kind);
      const payload = { name: name.value.trim(), description: description.value.trim(), config };
      if (selectedId) payload.expected_revision = this.selection.revision;
      const item = await this.api.request(`/cml/${this.kind}${selectedId ? `/${encodeURIComponent(selectedId)}` : ''}`, { method: selectedId ? 'PATCH' : 'POST', body: payload });
      this.selection = item; this.copy = null; await this.refresh();
      this.notify(selectedId ? 'Объект обновлен. Предыдущая редакция сохранена.' : 'Объект создан.');
    }), 'primary');
    const buttons = [save];
    if (this.selection) {
      buttons.push(action('Применить в конструкторе', () => this.guarded(async () => { await this.applyConfig(this.kind, this.selection.config); this.notify('Настройки применены.'); })));
      buttons.push(action('Сохранить копию', () => { this.copy = structuredClone(this.selection); this.selection = null; this.render(); }));
      buttons.push(action('Удалить', () => this.guarded(async () => {
        if (!window.confirm(`Удалить «${this.selection.name}» из библиотеки? Снимки прошлых запусков сохранятся.`)) return;
        await this.api.request(`/cml/${this.kind}/${encodeURIComponent(this.selection.id)}`, { method: 'DELETE' });
        this.selection = null; this.copy = null; await this.refresh(); this.notify('Объект удален.');
      }), 'danger'));
    }
    this.container.replaceChildren(element('div', { className: 'cml-library' }, [
      element('section', { className: 'cml-panel' }, [
        element('div', { className: 'cml-panel-heading' }, [element('h2', { text: resource.title }), action('Новый', () => { this.selection = null; this.copy = null; this.render(); })]),
        field('Поиск', search, { help: 'Ищет по названию и описанию сохраненных объектов.', lesson_id: '29-model-export' }), list,
      ]),
      element('section', { className: 'cml-panel' }, [
        element('h2', { text: this.selection ? this.selection.name : `Новый объект: ${resource.singular.toLowerCase()}` }), explanation,
        field('Название', name, { help: 'Название помогает найти объект в своей библиотеке. Используйте задачу и смысл настроек.', lesson_id: '29-model-export' }),
        field('Описание', description, { help: 'Кратко запишите, зачем создан объект и где его применять.', lesson_id: '29-model-export' }),
        element('h3', { text: 'Конфигурация' }), summary,
        ...(this.selection ? [replaceConfig.wrapper] : []),
        notice(this.copy ? 'Копия содержит параметры выбранного объекта. Измените название и сохраните ее.' : 'Параметры редактируются в конструкторе. Чтобы открыть настройки объекта, нажмите «Применить в конструкторе». Обновление названия сохраняет параметры; их замена включается отдельным флажком.'),
        element('div', { className: 'cml-toolbar' }, buttons),
      ]),
    ]));
  }

  describe(config = {}) {
    if (this.kind === 'model-recipes') return [`Алгоритм: ${config.algorithm_id || 'не выбран'}`, ...Object.entries(config.params || {}).map(([key, value]) => `${key}: ${typeof value === 'object' ? JSON.stringify(value) : value}`)].join('\n');
    if (this.kind === 'preprocessor-recipes') return (config.steps || []).map((step, index) => `${index + 1}. ${step.adapter_id}${step.enabled === false ? ' (выключен)' : ''} · ${(step.columns || []).join(', ') || 'все подходящие столбцы'}`).join('\n') || 'Подготовка по умолчанию';
    return `Задача: ${config.task || 'regression'}\nДатасет: ${config.dataset_id || 'не выбран'}\nАлгоритм: ${config.algorithm_id || 'не выбран'}\nПризнаков: ${(config.features || []).length}\nОбучение: ${formatNumber((config.split?.train || .6) * 100)}%`;
  }
}
