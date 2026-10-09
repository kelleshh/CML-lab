import { element, formatNumber } from './dom.js';
import { action, field, heading, notice } from './controls.js';

const tasks = ['regression', 'classification', 'clustering', 'ranking', 'forecasting', 'panel', 'anomaly', 'reduction', 'other'];

/** Dataset descriptions are cheap metadata reads; editing creates a server-owned snapshot. */
export class DatasetLibrary {
  constructor({ api, onSelect = () => {}, onEdit = () => {}, onUpload = () => {}, onDeleted = () => {}, notify = () => {}, onError = () => {} } = {}) {
    Object.assign(this, { api, onSelect, onEdit, onUpload, onDeleted, notify, onError });
    this.items = []; this.selected = null; this.query = ''; this.task = ''; this.offset = 0; this.limit = 24; this.total = 0; this.busy = false; this.generation = 0;
  }
  mount(container) { this.container = container; this.render(); return this; }
  async open() { return this.refresh(); }
  async setTask(task) { this.task = task || ''; this.offset = 0; return this.refresh(); }
  async guarded(callback) {
    if (this.busy) return;
    this.busy = true;
    try { return await callback(); } catch (error) { this.onError(error); }
    finally { this.busy = false; }
  }
  async refresh() {
    const generation = ++this.generation;
    const query = new URLSearchParams({ query: this.query, offset: this.offset, limit: this.limit }); if (this.task) query.set('task', this.task);
    try {
      const response = await this.api.request(`/datasets/library?${query}`);
      if (generation !== this.generation) return;
      this.items = Array.isArray(response) ? response : response.items || []; this.total = response.total ?? this.items.length;
      if (this.selected) this.selected = this.items.find(item => item.id === this.selected.id) || this.selected;
      this.render();
    } catch (error) { if (generation === this.generation) this.onError(error); }
  }
  render() {
    if (!this.container) return;
    const search = element('input', { type: 'search', placeholder: 'Название, описание или tags', value: this.query, maxlength: 500, 'aria-label': 'Dataset search' });
    search.addEventListener('keydown', event => { if (event.key === 'Enter') { event.preventDefault(); this.query = search.value; this.offset = 0; this.refresh(); } });
    const task = element('select', { 'aria-label': 'Dataset task' }, [element('option', { value: '', text: 'All tasks' }), ...tasks.map(value => element('option', { value, text: value }))]); task.value = this.task;
    task.addEventListener('change', () => this.setTask(task.value));
    const list = element('div', { className: 'cml-entity-list cml-dataset-cards' }, this.items.map(item => this.card(item)));
    if (!this.items.length) list.append(notice(this.query || this.task ? 'Нет подходящих datasets. Измените фильтры.' : 'Datasets пока нет. Загрузите таблицу или создайте учебный набор.'));
    const previous = action('←', () => { this.offset = Math.max(0, this.offset - this.limit); this.refresh(); }, '', { disabled: this.offset === 0, 'aria-label': 'Предыдущая страница datasets' });
    const next = action('→', () => { this.offset += this.limit; this.refresh(); }, '', { disabled: this.offset + this.limit >= this.total, 'aria-label': 'Следующая страница datasets' });
    const browse = element('section', { className: 'cml-panel' }, [heading('Datasets', { help: 'Карточки содержат характеристики исходных данных. Изменение таблицы или описания создает новую версию; прошлые запуски сохраняют свой снимок.', lesson_id: 'dataset-sources' }), element('div', { className: 'cml-toolbar' }, [action('Upload / Create', () => this.onUpload()), action('Обновить', () => this.refresh())]), field('Search', search, { help: 'Enter или кнопка Search запускает фильтрацию по названию, описанию и меткам.', lesson_id: 'dataset-sources' }), field('task', task, { help: 'Фильтр по заявленной задаче набора; выбор цели и признаков выполняется в проекте.', lesson_id: 'dataset-sources' }), action('Search', () => { this.query = search.value; this.offset = 0; this.refresh(); }), list, element('div', { className: 'cml-toolbar' }, [previous, element('span', { text: `${this.total ? this.offset + 1 : 0}–${Math.min(this.offset + this.items.length, this.total)} / ${this.total}`, role: 'status' }), next])]);
    const details = element('section', { className: 'cml-panel' }, this.selected ? this.details(this.selected) : [notice('Выберите Dataset, чтобы изменить описание, открыть таблицу или использовать его в проекте.')]);
    this.container.replaceChildren(element('div', { className: 'cml-library' }, [browse, details]));
  }
  card(item) {
    const columns = item.columns || [], stats = item.stats || {};
    const missing = stats.missing_cells ?? columns.reduce((sum, column) => sum + (column.missing || 0), 0);
    return element('article', { className: 'cml-library-card', 'data-library-id': item.id }, [
      element('button', { type: 'button', className: 'cml-entity', 'aria-current': this.selected?.id === item.id, onclick: () => { this.selected = item; this.render(); } }, [element('strong', { text: item.name }), element('small', { text: `${item.task || (item.tasks || []).join(', ') || 'other'} · ${item.source || 'local'}` })]),
      element('p', { className: 'cml-library-card-summary', text: `${item.rows ?? stats.rows ?? '—'} rows · ${columns.length || stats.columns || 0} columns · ${missing} missing values` }),
      ...(item.description ? [element('p', { className: 'cml-library-card-summary', text: item.description.length > 180 ? `${item.description.slice(0, 177)}…` : item.description })] : []),
      element('div', { className: 'cml-toolbar' }, [action('Use in project', () => this.guarded(() => this.onSelect(item))), action('Download .csv', () => this.guarded(() => this.api.download(`/datasets/${encodeURIComponent(item.id)}/export`, 'dataset.csv')))]),
    ]);
  }
  details(item) {
    const name = element('input', { type: 'text', value: item.name, required: true, maxlength: 200 });
    const description = element('textarea', { value: item.description || '', maxlength: 5000, rows: 4 });
    const tags = element('input', { value: (item.tags || []).join(', ') });
    const task = element('select', { 'aria-label': 'Metadata task' }, tasks.map(value => element('option', { value, text: value }))); task.value = item.task || item.tasks?.[0] || 'other';
    const target = element('select', { 'aria-label': 'task_target' }, [element('option', { value: '', text: 'None' }), ...(item.columns || []).map(column => element('option', { value: column.name, text: column.name }))]); target.value = item.task_target || item.default_target || '';
    const stats = item.stats || {};
    const facts = element('dl', { className: 'cml-library-dataset-facts' }, [
      element('dt', { text: 'id' }), element('dd', { text: item.id }),
      element('dt', { text: 'rows' }), element('dd', { text: item.rows ?? stats.rows ?? '—' }),
      element('dt', { text: 'numeric columns' }), element('dd', { text: stats.numeric_columns ?? (item.columns || []).filter(column => column.numeric).length }),
      element('dt', { text: 'memory' }), element('dd', { text: stats.memory_bytes ? `${formatNumber(stats.memory_bytes / 1048576)} MiB` : '—' }),
      ...(item.parent_id ? [element('dt', { text: 'parent_id' }), element('dd', { text: item.parent_id })] : []),
    ]);
    const columns = element('table', { className: 'cml-library-metrics', 'aria-label': 'Dataset columns' }, [element('thead', {}, [element('tr', {}, ['column', 'dtype', 'missing', 'unique'].map(text => element('th', { scope: 'col', text })))]), element('tbody', {}, (item.columns || []).map(column => element('tr', {}, [column.name, column.dtype, column.missing ?? 0, column.unique ?? '—'].map(text => element('td', { text })))))]);
    return [heading(item.name, { help: 'Редактирование сохраняет новый dataset_id. Строки исходного dataset и снимки завершенных запусков остаются неизменными.', lesson_id: 'dataset-sources' }), facts,
      field('name', name, { help: 'Название новой версии dataset.', lesson_id: 'dataset-sources' }), field('description', description, { help: 'Источник, содержание и назначение данных.', lesson_id: 'dataset-sources' }), field('tags', tags, { help: 'Метки через запятую для поиска набора.', lesson_id: 'dataset-sources' }), field('task', task, { help: 'Основная задача набора. Задача проекта задается отдельно.', lesson_id: 'dataset-sources' }), field('task_target', target, { help: 'Целевой столбец. Для clustering, anomaly и reduction обычно None; цель classification может быть текстовой.', lesson_id: 'dataset-sources' }), notice('Сохранение создает новую версию dataset. Удаление затрагивает только выбранный dataset_id; сервер защищает наборы, используемые запусками и экспериментами.'),
      element('div', { className: 'cml-toolbar' }, [
        action('Use in project', () => this.guarded(() => this.onSelect(item)), 'primary'), action('Edit rows', () => this.guarded(() => this.onEdit(item))),
        action('Сохранить новую версию', () => this.guarded(async () => { if (!name.reportValidity()) return; const updated = await this.api.request(`/datasets/${encodeURIComponent(item.id)}/metadata`, { method: 'PATCH', body: { name: name.value.trim(), description: description.value.trim(), tags: tags.value.split(',').map(value => value.trim()).filter(Boolean), task: task.value, task_target: target.value || null } }); this.selected = updated; this.offset = 0; await this.refresh(); this.notify('Новая версия dataset сохранена.'); })),
        action('Download .csv', () => this.guarded(() => this.api.download(`/datasets/${encodeURIComponent(item.id)}/export`, 'dataset.csv'))),
        action('Удалить dataset', () => this.guarded(async () => { if (!window.confirm(`Удалить dataset «${item.name}»?`)) return; await this.api.request(`/datasets/${encodeURIComponent(item.id)}`, { method: 'DELETE' }); this.selected = null; await this.refresh(); await this.onDeleted(item); this.notify('Dataset удален.'); }), 'danger'),
      ]), columns];
  }
  destroy() { this.generation++; this.container?.replaceChildren(); this.container = null; }
}
