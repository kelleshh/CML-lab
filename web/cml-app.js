import { ApiClient } from './cml/api.js';
import { element, badge } from './cml/dom.js';
import { field, notice } from './cml/controls.js';
import { lessonUrl } from './cml/help.js';
import { DataStep } from './cml/data.js';
import { PreparationBuilder } from './cml/preparation.js';
import { ModelBuilder } from './cml/models.js';
import { ValidationBuilder } from './cml/validation.js';
import { ResultsWorkspace } from './cml/results.js';
import { RecipeLibrary } from './cml/recipes.js';
import { SavedRuns } from './cml/saved-runs.js';
import { DatasetWorkspace } from './datasets-ui.js';
import { LiveTrainingView } from './cml/training.js';

const $ = id => document.getElementById(id);
const STEP_IDS = ['data', 'preparation', 'model', 'validation', 'results'];
const FINAL_STATES = new Set(['completed', 'done', 'failed', 'error', 'cancelled', 'interrupted', 'succeeded']);

export class CmlApplication {
  constructor(api = new ApiClient()) {
    this.api = api; this.step = 'data'; this.view = 'workflow'; this.busy = false;
    this.dataset = null; this.runId = null; this.result = null; this.timer = null;
  }

  notify(message) {
    $('toast').classList.remove('error'); $('toast').textContent = message; $('toast').hidden = false;
    clearTimeout(this.toastTimer);
    this.toastTimer = setTimeout(() => { $('toast').hidden = true; }, 6000);
  }

  error(error) { this.notify(error.message || String(error)); $('toast').classList.add('error'); }

  async init() {
    document.body.dataset.theme = localStorage.getItem('cml-lab-theme') || 'light';
    this.setSettingsWidth(localStorage.getItem('cml-settings-width') || 520);
    this.catalogue = await this.api.request('/cml/catalogue');
    const changed = () => this.changed();
    const onError = error => this.error(error);
    this.data = new DataStep({ api: this.api, catalogue: this.catalogue, onChange: changed, onTaskChange: task => this.setTask(task), onBrowse: () => this.showView('datasets'), onEdit: dataset => this.editData(dataset), onError });
    this.model = new ModelBuilder({ catalogue: this.catalogue, onChange: model => { this.validation?.setModel(model); this.changed(); }, onError });
    this.preparation = new PreparationBuilder({ api: this.api, catalogue: this.catalogue, getRequest: () => this.request(), onChange: changed, onError });
    this.validation = new ValidationBuilder({ catalogue: this.catalogue, onChange: changed, onError });
    this.results = new ResultsWorkspace({ api: this.api, onReveal: () => this.revealTest(), onSave: () => this.saveDialog('experiment'), onError });
    this.data.mount($('step-data'));
    this.preparation.mount($('step-preparation'));
    this.model.mount($('step-model'));
    this.validation.mount($('step-validation'));
    this.results.mount($('step-results'));
    this.liveTraining = new LiveTrainingView({ container: $('step-results') });
    this.validation.setModel(this.model.selected());
    this.datasets = new DatasetWorkspace({
      api: (path, options) => this.api.request(path, options),
      onSelect: (dataset, { intent } = {}) => { this.selectDataset(dataset); if (intent === 'train') { this.showView('workflow'); this.showStep('data'); } },
      onCreated: dataset => this.selectDataset(dataset),
      onEdit: point => this.editData(point.dataset || this.dataset, point.index ?? point.originalIndex),
      onHelp: id => window.open(lessonUrl(id), '_blank', 'noopener'),
      onDeleted: id => { if (this.dataset?.id === id) this.clearDataset(); }, onError,
    });
    this.datasets.mount($('dataset-workspace')).setCatalogue(this.catalogue);
    this.recipes = new RecipeLibrary({ api: this.api, getConfig: kind => this.configFor(kind), applyConfig: (kind, config) => this.applyConfig(kind, config), notify: message => this.notify(message), onError }).mount($('recipe-workspace'));
    const savedProps = { api: this.api, onOpen: (item, runId) => this.openSaved(item, runId), notify: message => this.notify(message), onError };
    this.artifacts = new SavedRuns(savedProps).mount($('artifact-workspace'));
    this.history = new SavedRuns(savedProps).mount($('history-workspace'));
    this.bind(); this.changed();
    if (localStorage.getItem('cml-lab-draft')) $('restore-draft').hidden = false;
    await this.renderLearning();
    const lesson = new URLSearchParams(location.search).get('lesson');
    if (lesson) await this.applyLesson(lesson);
  }

  request() {
    return { ...this.data.values(), ...this.model.values(), preprocessing: this.preparation.values(), ...this.validation.values() };
  }

  configFor(kind) {
    if (kind === 'model-recipes') return this.model.values();
    if (kind === 'preprocessor-recipes') return this.preparation.values();
    return this.request();
  }

  changed() {
    const task = this.catalogue?.tasks?.find(task => task.id === this.data?.taskId);
    $('run-summary').textContent = this.dataset ? `${this.dataset.name || this.dataset.label} · ${task?.label || this.data?.taskId} · ${this.model?.selected()?.name || 'выберите модель'}` : 'Выберите или загрузите датасет на первом шаге.';
    $('run').disabled = this.busy || !this.dataset || !this.model?.selected() || this.model.selected().available === false;
    if (!this.dataset || !this.validation?.train) return;
    clearTimeout(this.draftTimer);
    this.draftTimer = setTimeout(() => {
      try { localStorage.setItem('cml-lab-draft', JSON.stringify(this.request())); }
      catch { /* Incomplete forms remain editable and are not persisted. */ }
    }, 1000);
  }

  setTask(task) {
    this.model.setTask(task); this.preparation.setTask(task); this.validation.setTask(task);
    this.validation.setModel(this.model.selected()); this.changed();
  }

  selectDataset(dataset) {
    if (this.busy) throw new Error('Остановите текущий расчет перед сменой датасета.');
    this.dataset = dataset;
    this.data.setDataset(dataset); this.preparation.setDataset(dataset); this.validation.setDataset(dataset); this.datasets?.setDataset(dataset);
    this.changed(); this.notify('Датасет выбран. Проверьте задачу и роли столбцов.');
  }

  clearDataset() {
    this.dataset = null; this.data.dataset = null; this.data.saved = {}; this.data.render();
    this.preparation.setDataset(null); this.validation.setDataset(null);
    this.result = null; this.runId = null; this.results.empty(); this.changed();
  }

  async applyConfig(kind, config) {
    if (this.busy) throw new Error('Остановите текущий расчет перед сменой настроек.');
    if (kind === 'model-recipes') { this.model.setConfig(config); this.showView('workflow'); this.showStep('model'); }
    else if (kind === 'preprocessor-recipes') { this.preparation.setConfig(config); this.showView('workflow'); this.showStep('preparation'); }
    else {
      if (config.dataset_id) this.selectDataset(await this.api.request(`/datasets/${encodeURIComponent(config.dataset_id)}`));
      this.data.setConfig(config); this.setTask(config.task || 'regression'); this.model.setConfig(config);
      this.preparation.setConfig(config.preprocessing || {}); this.validation.setConfig(config);
      this.showView('workflow'); this.showStep('data');
    }
    this.changed();
  }

  showView(view) {
    this.view = view;
    document.querySelectorAll('.cml-view').forEach(node => { node.hidden = node.id !== `view-${view}`; });
    document.querySelectorAll('[data-view]').forEach(button => {
      if (button.dataset.view === view) button.setAttribute('aria-current', 'page'); else button.removeAttribute('aria-current');
    });
    if (view === 'library') this.openLibrary(document.querySelector('[data-library][aria-selected=true]')?.dataset.library || 'projects').catch(error => this.error(error));
    if (view === 'history') this.history.open(document.querySelector('[data-history][aria-selected=true]')?.dataset.history || 'experiments').catch(error => this.error(error));
    if (view === 'datasets') this.datasets.refreshLibrary().catch(error => this.error(error));
    if (view === 'workflow') { this.showStep(this.step); this.data.explore?.(); }
  }

  showStep(step) {
    this.step = step;
    document.querySelectorAll('[data-step]').forEach(button => { const selected = button.dataset.step === step; button.setAttribute('aria-selected', String(selected)); button.tabIndex = selected ? 0 : -1; });
    document.querySelectorAll('.cml-stage').forEach(node => { node.hidden = node.id !== `step-${step}`; });
    $('step-back').disabled = STEP_IDS.indexOf(step) === 0;
    $('step-next').disabled = STEP_IDS.indexOf(step) === STEP_IDS.length - 1;
    if (step === 'data') this.data.explore?.();
    if (globalThis.Plotly) document.querySelectorAll('.js-plotly-plot').forEach(plot => { if (plot.offsetParent !== null) Plotly.Plots.resize(plot); });
  }

  async openLibrary(kind) {
    document.querySelectorAll('[data-library]').forEach(button => { button.setAttribute('aria-selected', String(button.dataset.library === kind)); button.tabIndex = button.dataset.library === kind ? 0 : -1; });
    $('recipe-workspace').hidden = kind === 'models'; $('artifact-workspace').hidden = kind !== 'models';
    if (kind === 'models') await this.artifacts.open('models'); else await this.recipes.open(kind);
  }

  setBusy(busy) {
    this.busy = busy; $('run').disabled = busy || !this.dataset || !this.model?.selected() || this.model.selected().available === false; $('cancel-run').hidden = !busy; $('run-progress').hidden = !busy;
    this.datasets?.setBusy(busy); this.model?.form?.setDisabled(busy); $('save-project').disabled = busy;
  }

  async run() {
    if (this.busy) return;
    try {
      if (!this.dataset) throw new Error('Сначала выберите датасет.');
      if (!this.model.form.valid()) return;
      const request = this.request();
      if (request.search?.direction === 'auto') request.search.direction = this.validation.metrics().find(metric => metric.id === request.search.metric)?.direction || 'min';
      this.setBusy(true); $('run-status').textContent = 'Создается расчет…';
      this.results.tracePlayer?.destroy();
      this.liveTraining.begin(this.model.selected().name); this.showView('workflow'); this.showStep('results');
      const response = await this.api.request('/cml/runs', { method: 'POST', body: request });
      this.runId = response.id; this.result = null; await this.poll();
    } catch (error) { this.setBusy(false); this.error(error); }
  }

  async poll() {
    clearTimeout(this.timer);
    try {
      const job = await this.api.request(`/cml/runs/${encodeURIComponent(this.runId)}`);
      const event = job.events?.at(-1);
      const progress = typeof job.progress === 'number' ? job.progress : event?.progress ?? 0;
      $('run-progress').querySelector('span').style.width = `${Math.round(Math.min(1, Math.max(0, progress)) * 100)}%`;
      $('run-status').textContent = job.message || event?.message || `Состояние: ${job.status}`;
      if (!FINAL_STATES.has(job.status)) await this.liveTraining.update(job);
      if (FINAL_STATES.has(job.status)) {
        this.setBusy(false);
        if (job.result) {
          this.result = job.result; await this.results.setResult(job.result, this.runId);
          this.showView('workflow'); this.showStep('results'); $('run-status').textContent = 'Расчет завершен.';
        } else if (['failed', 'error'].includes(job.status)) throw new Error(typeof job.error === 'string' ? job.error : job.error?.message || 'Расчет завершился с ошибкой.');
        else this.notify(['cancelled', 'interrupted'].includes(job.status) ? 'Расчет остановлен.' : 'Расчет завершен без результата.');
      } else this.timer = setTimeout(() => this.poll(), 650);
    } catch (error) { this.setBusy(false); this.error(error); }
  }

  async cancel() {
    if (!this.runId) return;
    await this.api.request(`/cml/runs/${encodeURIComponent(this.runId)}`, { method: 'DELETE' });
    $('run-status').textContent = 'Останавливается…';
  }

  async revealTest() {
    try {
      const response = await this.api.request(`/cml/runs/${encodeURIComponent(this.runId)}/reveal-test`, { method: 'POST' });
      this.result = response.result || response; await this.results.setResult(this.result, this.runId);
    } catch (error) { this.error(error); }
  }

  async openSaved(item, runId) {
    let result = item.result;
    if (!result && runId) {
      const run = await this.api.request(`/cml/runs/${encodeURIComponent(runId)}`); result = run.result;
      if (!result && !FINAL_STATES.has(run.status)) { this.runId = runId; this.setBusy(true); this.showView('workflow'); await this.poll(); return; }
    }
    if (!result) throw new Error('Результат не найден. Запуск еще не завершен или модель удалена.');
    const config = result.effective_spec || item.effective_spec || item.request || item.spec;
    if (config) await this.applyConfig('projects', config);
    this.runId = runId || item.run_id || item.job_id; this.result = result;
    await this.results.setResult(result, this.runId); this.showView('workflow'); this.showStep('results');
  }

  saveDialog(kind) {
    if (kind === 'experiment' && !this.runId) return this.notify('Сначала завершите расчет.');
    const name = element('input', { required: true, maxlength: 150, value: `${this.dataset?.name || 'Эксперимент'} · ${this.model.selected()?.name || ''}` });
    const description = element('textarea', { rows: 3, maxlength: 2000 });
    $('save-title').textContent = kind === 'experiment' ? 'Сохранить эксперимент' : 'Сохранить проект';
    $('save-fields').replaceChildren(field('Название', name, { help: 'Название записи в вашей библиотеке.', lesson_id: '29-model-export' }), field('Описание', description, { help: 'Укажите смысл настроек и ожидаемое применение.', lesson_id: '29-model-export' }));
    $('confirm-save').onclick = async () => {
      if (!name.reportValidity()) return;
      try {
        const path = kind === 'experiment' ? `/cml/runs/${encodeURIComponent(this.runId)}/save` : '/cml/projects';
        await this.api.request(path, { method: 'POST', body: { name: name.value.trim(), description: description.value.trim(), ...(kind === 'experiment' ? {} : { config: this.request() }) } });
        $('save-dialog').close(); this.notify(kind === 'experiment' ? 'Эксперимент сохранен.' : 'Проект сохранен.');
      } catch (error) { this.error(error); }
    };
    $('save-dialog').showModal();
  }

  async editData(dataset = this.dataset, index = 0) {
    if (!dataset) return;
    const response = await this.api.request(`/datasets/${encodeURIComponent(dataset.id)}/rows?offset=${Math.max(0, Number(index) || 0)}&limit=100`);
    const rows = response.rows || [];
    const columns = response.columns?.map(column => typeof column === 'string' ? column : column.name) || dataset.columns.map(column => column.name);
    const offset = response.offset || Math.max(0, Number(index) || 0); const deleted = new Set(); const newRows = [];
    const table = element('table', { className: 'cml-table' });
    table.append(element('thead', {}, [element('tr', {}, [...columns.map(column => element('th', { text: column })), element('th', { text: 'Действие' })])]), element('tbody'));
    const body = table.querySelector('tbody');
    const append = (row, rowIndex, isNew = false) => {
      const record = Array.isArray(row) ? Object.fromEntries(columns.map((column, i) => [column, row[i]])) : row;
      const controls = columns.map(column => element('input', { value: record?.[column] ?? '', 'aria-label': `${column}, строка ${rowIndex + 1}` }));
      const tr = element('tr', {}, [...controls.map(control => element('td', {}, [control])), element('td', {}, [element('button', { className: 'cml-button danger', type: 'button', text: 'Удалить строку', onclick: () => { if (!isNew) deleted.add(rowIndex); else newRows.splice(newRows.findIndex(item => item.controls === controls), 1); tr.remove(); } })])]);
      tr._controls = controls; tr._index = rowIndex;
      if (isNew) newRows.push({ controls }); body.append(tr);
    };
    rows.forEach((row, i) => append(row, offset + i)); $('data-editor').replaceChildren(table);
    $('add-row').onclick = () => append({}, rows.length + newRows.length, true);
    $('apply-data').onclick = async () => {
      try {
        const convert = controls => Object.fromEntries(columns.map((column, i) => { const raw = controls[i].value; const numeric = dataset.columns.find(item => item.name === column)?.numeric; return [column, raw === '' ? null : numeric ? Number(raw) : raw]; }));
        const changes = [...body.children].filter(tr => !newRows.some(item => item.controls === tr._controls)).map(tr => ({ index: tr._index, values: convert(tr._controls) }));
        const metadata = await this.api.request(`/datasets/${encodeURIComponent(dataset.id)}/rows`, { method: 'PATCH', body: { changes, additions: newRows.map(item => convert(item.controls)), deletes: [...deleted] } });
        this.selectDataset(metadata); $('data-dialog').close(); this.notify('Сохранена новая редакция датасета.');
      } catch (error) { this.error(error); }
    };
    $('data-dialog').showModal();
  }

  async renderLearning() {
    try {
      const response = await this.api.request('/learning/lessons');
      const lessons = Array.isArray(response) ? response : response.items || response.lessons || [];
      const search = element('input', { type: 'search', placeholder: 'Задача, алгоритм или настройка', 'aria-label': 'Поиск учебных материалов' });
      const list = element('div', { className: 'cml-lesson-grid' });
      const render = () => list.replaceChildren(...lessons.filter(lesson => `${lesson.title} ${lesson.summary || ''} ${lesson.tasks?.join(' ') || ''}`.toLowerCase().includes(search.value.toLowerCase())).map(lesson => element('article', { className: 'cml-panel' }, [element('h2', {}, [element('a', { href: lessonUrl(lesson.id), target: '_blank', rel: 'noopener', text: lesson.title })]), element('p', { className: 'cml-note', text: lesson.summary || '' }), element('div', { className: 'cml-badges' }, (lesson.tasks || []).map(task => badge(task))), element('a', { href: lessonUrl(lesson.id), target: '_blank', rel: 'noopener', text: 'Прочитать материал ↗' })])));
      search.addEventListener('input', render); render(); $('learning-workspace').replaceChildren(search, list);
    } catch (error) { $('learning-workspace').replaceChildren(notice(`Материалы не загружены: ${error.message}`, true)); }
  }

  async applyLesson(id) {
    const lesson = await this.api.request(`/learning/lessons/${encodeURIComponent(id)}`); const preset = lesson.preset;
    if (!preset) { this.notify('У этого материала нет готового эксперимента. Выберите данные и настройки вручную.'); return; }
    if (preset.dataset) this.selectDataset(await this.api.request('/datasets/load', { method: 'POST', body: preset.dataset }));
    await this.applyConfig('projects', { ...preset, dataset_id: this.dataset?.id, algorithm_id: preset.algorithm_id || preset.model, preprocessing: preset.preprocessing || {} });
    this.notify('Учебный пример загружен. Проверьте настройки и запустите расчет.');
  }

  setSettingsWidth(value) {
    const width = Math.max(360, Math.min(760, Number(value) || 520));
    document.documentElement.style.setProperty('--settings-width', `${width}px`);
    $('settings-width').value = String(width); $('settings-width-value').textContent = `${width} px`;
    localStorage.setItem('cml-settings-width', String(width));
  }

  bind() {
    $('settings-width').addEventListener('input', event => { this.setSettingsWidth(event.target.value); });
    $('settings-width').addEventListener('change', () => { if (globalThis.Plotly) document.querySelectorAll('.js-plotly-plot').forEach(plot => { if (plot.offsetParent !== null) Plotly.Plots.resize(plot); }); });
    document.querySelectorAll('[data-view]').forEach(button => { button.onclick = () => this.showView(button.dataset.view); });
    document.querySelectorAll('[data-step]').forEach(button => { button.onclick = () => this.showStep(button.dataset.step); });
    document.querySelectorAll('[data-library]').forEach(button => { button.onclick = () => this.openLibrary(button.dataset.library).catch(error => this.error(error)); });
    document.querySelectorAll('[data-history]').forEach(button => { button.onclick = () => { document.querySelectorAll('[data-history]').forEach(item => { item.setAttribute('aria-selected', String(item === button)); item.tabIndex = item === button ? 0 : -1; }); this.history.open(button.dataset.history).catch(error => this.error(error)); }; });
    document.querySelectorAll('[role=tablist]').forEach(tablist => tablist.addEventListener('keydown', event => {
      if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
      const tabs = [...tablist.querySelectorAll('[role=tab]')]; const current = tabs.indexOf(event.target); if (current < 0) return;
      const next = event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1 : (current + (event.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length;
      event.preventDefault(); tabs[next].focus(); tabs[next].click();
    }));
    $('step-back').onclick = () => this.showStep(STEP_IDS[Math.max(0, STEP_IDS.indexOf(this.step) - 1)]);
    $('step-next').onclick = () => this.showStep(STEP_IDS[Math.min(STEP_IDS.length - 1, STEP_IDS.indexOf(this.step) + 1)]);
    $('run').onclick = () => this.run(); $('cancel-run').onclick = () => this.cancel().catch(error => this.error(error));
    $('save-project').onclick = () => this.saveDialog('project');
    $('restore-draft').onclick = async () => { try { await this.applyConfig('projects', JSON.parse(localStorage.getItem('cml-lab-draft'))); this.notify('Черновик восстановлен.'); } catch (error) { this.error(error); } };
    $('close-save').onclick = () => $('save-dialog').close(); $('close-data').onclick = () => $('data-dialog').close();
    $('theme').onclick = async () => { document.body.dataset.theme = document.body.dataset.theme === 'dark' ? 'light' : 'dark'; localStorage.setItem('cml-lab-theme', document.body.dataset.theme); await this.datasets.refreshTheme?.(); await this.data.explore?.(); if (this.result) await this.results.render(); };
    document.addEventListener('keydown', event => {
      if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') { event.preventDefault(); this.run(); }
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 's') { event.preventDefault(); this.saveDialog('project'); }
    });
  }
}

export const application = new CmlApplication();
application.init().catch(error => {
  $('app-error').hidden = false;
  $('app-error').replaceChildren(notice(`Не удалось загрузить лабораторию: ${error.message}. Перезапустите приложение и обновите страницу.`, true));
});
