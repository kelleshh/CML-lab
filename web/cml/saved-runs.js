import { element, formatNumber } from './dom.js';
import { action, field, heading, notice } from './controls.js';

export class SavedRuns {
  constructor({ api, onOpen, onError, notify }) { this.api = api; this.onOpen = onOpen; this.onError = onError; this.notify = notify; this.kind = 'experiments'; this.selected = null; }
  mount(container) { this.container = container; return this; }
  async open(kind = 'experiments') { this.kind = kind; this.selected = null; await this.refresh(); }
  async refresh() {
    const response = await this.api.request(`/cml/${this.kind}`);
    this.items = Array.isArray(response) ? response : response.items || [];
    if (this.selected) this.selected = this.items.find(item => item.id === this.selected.id) || null;
    this.render();
  }
  async guarded(callback) { try { await callback(); } catch (error) { this.onError(error); } }
  render() {
    const artifacts = this.kind === 'models'; const liveRuns = this.kind === 'runs';
    const title = artifacts ? 'Обученные модели' : liveRuns ? 'Все запуски' : 'Сохраненные эксперименты';
    const list = element('div', { className: 'cml-entity-list' }, (this.items || []).map(item => element('button', { className: 'cml-entity', 'aria-current': String(this.selected?.id === item.id), onclick: () => { this.selected = item; this.render(); } }, [element('strong', { text: item.name || item.model_name || item.algorithm_id || item.id }), element('small', { text: `${item.task || ''} ${item.status || ''} · ${item.created_at ? new Date(typeof item.created_at === 'number' ? item.created_at * 1000 : item.created_at).toLocaleString('ru') : ''}` })])));
    if (!list.children.length) list.append(element('p', { className: 'cml-empty', text: artifacts ? 'Обученные модели появятся после завершенных расчетов.' : 'Здесь пока нет записей.' }));
    const details = element('section', { className: 'cml-panel' }, [heading(title, { help: artifacts ? 'Это обученные конвейеры с найденными параметрами. Настройки до обучения хранятся в шаблонах моделей.' : 'Запуск хранит снимок данных и настроек. Результат не меняется после редактирования шаблона.', lesson_id: '29-model-export' })]);
    if (this.selected) {
      const item = this.selected;
      const name = element('input', { value: item.name || item.model_name || '', maxlength: 150 });
      const description = element('textarea', { value: item.description || '', rows: 3 });
      details.append(field('Название', name, { help: 'Название записи в вашей библиотеке.', lesson_id: '29-model-export' }), field('Описание', description, { help: 'Запишите задачу, данные и смысл результата.', lesson_id: '29-model-export' }), element('p', { className: 'cml-note', text: `${item.task || ''} · ${item.algorithm_id || item.model_name || ''} · ${item.status || 'сохранен'}` }), element('div', { className: 'cml-toolbar' }, [
        action('Открыть результат', () => this.guarded(async () => { const runId = item.run_id || item.job_id || (liveRuns ? item.id : null); const payload = liveRuns || artifacts ? item : await this.api.request(`/cml/${this.kind}/${encodeURIComponent(item.id)}`); await this.onOpen(payload, runId || payload.run_id || payload.job_id); }), 'primary'),
        ...(!liveRuns ? [action('Обновить описание', () => this.guarded(async () => { await this.api.request(`/cml/${this.kind}/${encodeURIComponent(item.id)}`, { method: 'PATCH', body: { name: name.value.trim(), description: description.value.trim(), expected_revision: item.revision } }); await this.refresh(); this.notify('Описание обновлено.'); }))] : []),
        action(liveRuns && !['completed', 'done', 'failed', 'error', 'cancelled', 'interrupted'].includes(item.status) ? 'Остановить запуск' : 'Удалить запись', () => this.guarded(async () => { if (!window.confirm(`Удалить или остановить «${item.name || item.id}»?`)) return; await this.api.request(`/cml/${this.kind}/${encodeURIComponent(item.id)}`, { method: 'DELETE' }); this.selected = null; await this.refresh(); this.notify('Действие выполнено.'); }), 'danger'),
      ]));
      const metrics = item.metrics || item.result?.evaluations?.validation?.metrics;
      if (metrics) details.append(element('p', { className: 'cml-note', text: Object.entries(metrics).map(([key, value]) => `${key}: ${formatNumber(value)}`).join(' · ') }));
    } else details.append(notice('Выберите запись в списке.'));
    this.container.replaceChildren(element('div', { className: 'cml-library' }, [element('section', { className: 'cml-panel' }, [element('div', { className: 'cml-panel-heading' }, [element('h2', { text: title }), action('Обновить', () => this.guarded(() => this.refresh()))]), list]), details]));
  }
}
