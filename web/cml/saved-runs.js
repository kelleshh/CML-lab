import { element, formatNumber } from './dom.js';
import { action, field, heading, notice } from './controls.js';
import { helpButton } from './help.js';

export class SavedRuns {
  constructor({ api, onOpen, onError, notify, catalogue }) { this.api = api; this.onOpen = onOpen; this.onError = onError; this.notify = notify; this.catalogue = catalogue; this.kind = 'experiments'; this.selected = null; this.query = ''; this.busy = false; }
  mount(container) { this.container = container; return this; }
  async open(kind = 'experiments') { this.kind = kind; this.selected = null; this.query = ''; await this.refresh(); }
  async refresh() {
    const kind = this.kind, generation = this.refreshGeneration = (this.refreshGeneration || 0) + 1;
    const response = await this.api.request(`/cml/${kind}`);
    if (generation !== this.refreshGeneration || kind !== this.kind) return;
    this.items = Array.isArray(response) ? response : response.items || [];
    if (this.selected) this.selected = this.items.find(item => item.id === this.selected.id) || null;
    this.render();
  }
  async guarded(callback) { if (this.busy) return; this.busy = true; try { await callback(); } catch (error) { this.onError(error); } finally { this.busy = false; } }
  render() {
    const artifacts = this.kind === 'models'; const liveRuns = this.kind === 'runs';
    const title = artifacts ? 'Обученные модели' : liveRuns ? 'Все запуски' : 'Сохраненные эксперименты';
    const search = element('input', { type: 'search', value: this.query, 'aria-label': `Найти: ${title}`, placeholder: 'Название, task или algorithm' });
    const list = element('div', { className: 'cml-entity-list' });
    const renderList = () => {
      const query = this.query.toLowerCase().trim();
      const items = (this.items || []).filter(item => `${item.name || ''} ${item.model_name || ''} ${item.algorithm_id || ''} ${item.task || item.spec?.task || ''}`.toLowerCase().includes(query));
      list.replaceChildren(...items.map(item => this.card(item, { artifacts, liveRuns })));
      if (!list.children.length) list.append(element('p', { className: 'cml-empty', text: query ? 'Нет объектов, соответствующих поиску.' : artifacts ? 'Обученные модели появятся после завершенных расчетов.' : 'Здесь пока нет записей.' }));
    };
    search.addEventListener('input', () => { this.query = search.value; renderList(); }); renderList();
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
      if (metrics) details.append(this.metricsTable(metrics, item.task || item.spec?.task));
      if ((artifacts || liveRuns) && (item.status === 'completed' || artifacts)) details.append(this.exportChoices(item));
    } else details.append(notice('Выберите запись в списке.'));
    this.container.replaceChildren(element('div', { className: 'cml-library' }, [element('section', { className: 'cml-panel' }, [element('div', { className: 'cml-panel-heading' }, [element('h2', { text: title }), action('Обновить', () => this.guarded(() => this.refresh()))]), field('Поиск', search, { help: 'Фильтр по названию, задаче и алгоритму; сохраненные записи остаются в библиотеке.', lesson_id: '29-model-export' }), list]), details]));
  }

  metricsTable(metrics, task, maximum = Infinity) {
    const values = Object.entries(metrics).filter(([, value]) => typeof value === 'number' && Number.isFinite(value)).slice(0, maximum);
    if (!values.length) return element('p', { className: 'cml-note', text: 'Validation metrics недоступны для этой записи.' });
    return element('table', { className: 'cml-library-metrics', 'aria-label': 'Validation metrics' }, [element('caption', { text: 'Validation metrics' }), element('tbody', {}, values.map(([key, value]) => {
      const definition = this.catalogue?.metrics?.find(metric => metric.id === key && (!task || metric.task === task));
      return element('tr', {}, [element('th', { scope: 'row' }, [element('span', { text: definition?.name || key }), helpButton({ label: definition?.name || key, help: definition?.help || definition?.description || 'Метрика на validation. Значение характеризует выбранное разбиение; правила расчета и направление оптимизации описаны в учебнике.', lesson_id: definition?.lesson_id || '20-metrics-experiment' })]), element('td', { text: formatNumber(value) })]);
    }))]);
  }

  card(item, { artifacts, liveRuns }) {
    const task = item.task || item.spec?.task, algorithm = item.algorithm_id || item.spec?.algorithm_id || item.model_name;
    const select = element('button', { type: 'button', className: 'cml-entity', 'aria-current': String(this.selected?.id === item.id), onclick: () => { this.selected = item; this.render(); } }, [element('strong', { text: item.name || item.model_name || algorithm || item.id }), element('small', { text: `${task || ''} · ${algorithm || ''} · ${item.status || 'saved'}` })]);
    const card = element('article', { className: 'cml-library-card', 'data-library-id': item.id }, [select]);
    const metrics = item.metrics || item.result?.evaluations?.validation?.metrics;
    if (metrics) card.append(this.metricsTable(metrics, task, 4));
    if (item.created_at) card.append(element('p', { className: 'cml-library-card-summary', text: new Date(typeof item.created_at === 'number' ? item.created_at * 1000 : item.created_at).toLocaleString('ru') }));
    if ((artifacts || liveRuns) && (item.status === 'completed' || artifacts)) {
      const runId = item.run_id || item.job_id || item.id;
      card.append(action('Download .joblib', () => this.guarded(() => this.api.download(`/cml/runs/${encodeURIComponent(runId)}/export?format=joblib`, 'model.joblib'))));
    }
    return card;
  }

  exportChoices(item) {
    const runId = item.run_id || item.job_id || item.id;
    const choices = element('details', { className: 'cml-library-export-choices' }, [element('summary', { text: 'Другие форматы модели' })]);
    const body = element('div'); choices.append(body);
    let loaded = false;
    choices.addEventListener('toggle', async () => {
      if (!choices.open || loaded) return; loaded = true;
      body.replaceChildren(notice('Проверка доступных форматов…'));
      try {
        const result = await this.api.request(`/cml/runs/${encodeURIComponent(runId)}/export-capabilities`);
        body.replaceChildren(...(result.formats || []).map(format => element('div', { className: 'cml-library-export-format' }, [action(format.label || format.format, () => this.guarded(() => this.api.download(`/cml/runs/${encodeURIComponent(runId)}/export?format=${encodeURIComponent(format.format)}`, `model.${format.format}`)), '', { disabled: format.available === false }), ...(format.available === false ? [notice(format.reason || 'Формат недоступен для этой модели.')] : [])])));
      } catch (error) { loaded = false; body.replaceChildren(notice(error.message, true)); }
    });
    return choices;
  }
}
