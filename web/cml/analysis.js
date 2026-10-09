import { element, id, formatNumber } from './dom.js';
import { helpButton } from './help.js';

const LESSON = 'dataset-analysis';
const groups = { Distribution: 'Распределения', Relations: 'Связи признаков', 'Missing data': 'Пропуски', Categorical: 'Категории', Groups: 'Группы', Time: 'Время', Quality: 'Проверка данных' };

function choose(values, value = '', properties = {}) {
  const control = element('select', properties, values.map(item => element('option', { value: item.value, text: item.label })));
  control.value = value;
  return control;
}

function field(label, control, explanation) {
  control.id ||= id('analysis-control');
  return element('div', { className: 'cml-field' }, [
    element('label', { htmlFor: control.id, className: 'cml-field-heading' }, [element('span', { text: label }), helpButton({ label, help: explanation, lesson_id: LESSON })]), control,
  ]);
}

class AnalysisController {
  constructor(container, { api, dataset = null, onError = () => {} }) {
    this.container = container; this.api = api; this.onError = onError;
    this.sequence = 0; this.requests = new Set(); this.plotNodes = new Set(); this.disposed = false;
    if (globalThis.ResizeObserver) {
      this.observer = new ResizeObserver(() => this.resize());
      this.observer.observe(container);
    }
    if (!document.getElementById('cml-analysis-styles')) document.head.append(element('link', { id: 'cml-analysis-styles', rel: 'stylesheet', href: new URL('./analysis.css', import.meta.url).href }));
    this.ready = this.setDataset(dataset);
  }

  async request(body) {
    const controller = new AbortController(); this.requests.add(controller);
    try { return await this.api.request('/cml/analysis', { method: 'POST', body, signal: controller.signal }); }
    finally { this.requests.delete(controller); }
  }

  purge(node) {
    for (const plot of [...this.plotNodes]) if (!node || node === plot || node.contains(plot)) {
      if (globalThis.Plotly?.purge) globalThis.Plotly.purge(plot);
      this.plotNodes.delete(plot);
    }
  }

  resize() {
    if (!globalThis.Plotly?.Plots?.resize || this.disposed) return;
    for (const plot of this.plotNodes) if (plot.isConnected && plot.clientWidth > 0) globalThis.Plotly.Plots.resize(plot);
  }

  async setDataset(dataset) {
    if (dataset?.id && dataset.id === this.dataset?.id && this.overview && !this.disposed) return;
    const sequence = ++this.sequence;
    this.dataset = dataset; this.overview = null; this.purge();
    for (const request of this.requests) request.abort();
    this.requests.clear();
    this.container.replaceChildren();
    if (this.disposed) return;
    if (!dataset?.id) {
      this.container.append(element('p', { className: 'cml-empty', text: 'Выберите датасет для анализа исходной таблицы.' }));
      return;
    }
    this.status = element('p', { className: 'cml-note', role: 'status', 'aria-live': 'polite', text: 'Читаем исходную таблицу…' });
    this.container.append(this.status);
    try {
      const overview = await this.request({ dataset_id: dataset.id, kind: 'overview' });
      if (sequence !== this.sequence || this.disposed) return;
      this.overview = overview; this.build(overview);
    } catch (error) {
      if (sequence !== this.sequence || this.disposed || error.name === 'AbortError') return;
      this.status.textContent = error.message; this.status.setAttribute('role', 'alert'); this.onError(error);
    }
  }

  build(overview) {
    this.columns = overview.columns || [];
    this.definitions = overview.graph_types || [];
    this.category = choose([{ value: '', label: 'Все разделы' }, ...Object.entries(groups).map(([value, label]) => ({ value, label }))]);
    this.kind = choose(this.definitions.map(item => ({ value: item.id, label: item.name })), this.definitions[0]?.id || '');
    this.form = element('div', { className: 'cml-form-grid cml-analysis-form' });
    this.explanation = element('p', { className: 'cml-note' });
    this.generateButton = element('button', { type: 'button', className: 'cml-button primary', text: 'Построить график', onClick: () => this.generate() });
    this.generated = element('div', { className: 'cml-analysis-plots', 'aria-live': 'polite' });
    const main = element('div', { className: 'cml-analysis-plots cml-analysis-main' });
    const stats = overview.summary || {};
    this.status = element('p', { className: 'cml-note', role: 'status', 'aria-live': 'polite', text: 'Обзор готов. Дополнительные графики строятся только по кнопке.' });
    this.container.replaceChildren(
      element('section', { className: 'cml-panel cml-analysis-overview' }, [
        element('div', { className: 'cml-panel-heading' }, [element('h2', { text: `Анализ: ${this.dataset.name || this.dataset.id}` }), helpButton({ label: 'Dataset analysis', help: overview.explanation, lesson_id: LESSON })]),
        element('p', { className: 'cml-note', text: overview.explanation }),
        element('dl', { className: 'cml-analysis-statistics' }, [['rows', 'Строки'], ['columns', 'Столбцы'], ['numeric_columns', 'Числовые'], ['categorical_columns', 'Категории и bool'], ['missing_cells', 'Пропуски'], ['duplicate_rows', 'Дубли строк']].flatMap(([key, label]) => [element('dt', { text: label }), element('dd', { text: formatNumber(stats[key]) })])), main,
      ]),
      element('section', { className: 'cml-panel' }, [
        element('h2', { text: 'Дополнительные графики' }),
        element('p', { className: 'cml-note', text: 'Выберите вид графика и столбцы. Можно сравнивать разные пары признаков, не создавая сразу сотни графиков. Одновременно сохраняется до 12 дополнительных графиков.' }),
        element('div', { className: 'cml-form-grid' }, [field('Раздел анализа', this.category, 'Разделы группируют графики по вопросу: распределение, связь, пропуски, категории, группы или время.'), field('Вид графика', this.kind, 'Английское название соответствует общепринятому виду графика. Объяснение выбранного вида находится под настройками.')]),
        this.form, this.explanation, element('div', { className: 'cml-toolbar' }, [this.generateButton, element('button', { type: 'button', className: 'cml-button', text: 'Очистить дополнительные графики', onClick: () => { this.purge(this.generated); this.generated.replaceChildren(); this.status.textContent = 'Дополнительные графики очищены.'; } })]), this.status,
      ]), this.generated,
    );
    this.category.addEventListener('change', () => {
      const filtered = this.definitions.filter(item => !this.category.value || item.category === this.category.value);
      this.kind.replaceChildren(...filtered.map(item => element('option', { value: item.id, text: item.name })));
      this.buildForm();
    });
    this.kind.addEventListener('change', () => this.buildForm());
    this.buildForm();
    for (const plot of overview.plots || []) this.addPlot(main, plot, false);
  }

  buildForm() {
    const definition = this.definitions.find(item => item.id === this.kind.value);
    this.form.replaceChildren(); this.columnControls = []; this.optionControls = new Map();
    this.explanation.textContent = definition?.explanation || '';
    if (!definition) { this.generateButton.disabled = true; return; }
    this.generateButton.disabled = false;
    const used = new Set();
    const types = definition.column_types || [];
    if (!types.length) {
      const available = this.columns.filter(column => definition.id !== 'correlation' || column.numeric);
      const select = choose(available.map(column => ({ value: column.name, label: column.name })), undefined, { multiple: true, size: Math.min(8, Math.max(3, available.length)) });
      [...select.options].forEach((option, index) => { option.selected = index < Math.min(available.length, 10); });
      this.columnControls.push(select);
      this.form.append(field('Столбцы (до 20)', select, 'Удерживайте Ctrl или Cmd для выбора нескольких столбцов. Для correlation нужны числовые столбцы. Матрицы ограничены 20 столбцами, чтобы подписи читались.'));
      if (!available.length) this.generateButton.disabled = true;
    } else types.forEach((type, index) => {
      const available = this.columns.filter(column => type !== 'numeric' || column.numeric);
      const preferred = type === 'time' ? this.dataset.roles?.time_column || available.find(column => /datetime/i.test(column.dtype || '') || /^(time|date|datetime|timestamp)$/i.test(column.name))?.name : null;
      const value = (preferred && available.some(column => column.name === preferred) ? preferred : available.find(column => !used.has(column.name))?.name) || '';
      used.add(value);
      const control = choose(available.map(column => ({ value: column.name, label: column.name })), value);
      this.columnControls.push(control);
      this.form.append(field(type === 'time' ? 'Time column' : types.length === 1 ? 'Column' : `Column ${index + 1}${type === 'numeric' ? ' (numeric)' : ''}`, control,
        type === 'time' ? 'Столбец числового времени или дат. Анализ сортирует строки по времени; autocorrelation требует равномерную сетку и одну последовательность.' : type === 'numeric' ? 'Известное числовое значение. Пропуски исключаются только из нужных координат графика; исходный датасет не меняется.' : 'Категория или любой столбец для группировки. Числовое значение здесь интерпретируется как название категории.'));
      if (!value) this.generateButton.disabled = true;
    });
    const addNumber = (key, label, value, min, max, help) => {
      const control = element('input', { type: 'number', value, min, max, step: 1, required: true });
      this.optionControls.set(key, control); this.form.append(field(label, control, help));
    };
    addNumber('sample_size', 'sample_size', 2000, 1, 5000, 'Максимум отображаемых строк для точечных графиков. В графике подписано, использован весь набор или фиксированная выборка. Агрегированные частоты считаются по всем подходящим строкам.');
    if (['histogram', 'histogram2d'].includes(definition.id)) addNumber('bins', 'bins', 30, 2, 100, 'Число интервалов гистограммы по каждой оси. Большое значение показывает детали, но делает малые группы нестабильными.');
    if (['category_counts', 'category_crosstab', 'group_box', 'group_mean'].includes(definition.id)) addNumber('top_k', 'top_k', 15, 2, 20, 'Сколько самых частых категорий показать. Остальные агрегируются в category counts или явно исключаются в матрицах и группах.');
    if (definition.id === 'rolling_mean') addNumber('window', 'window', 10, 2, 1000, 'Число последних наблюдений в среднем. Ряд сортируется по времени; будущие наблюдения не входят. Это окно по строкам, а не число календарных дней.');
    if (definition.id === 'autocorrelation') addNumber('max_lag', 'max_lag', 30, 1, 100, 'Максимальный сдвиг в наблюдениях. Фактический максимум ограничен длиной ряда. Смешанные объекты панели нельзя считать одной последовательностью.');
    if (definition.id === 'correlation') {
      const control = choose([{ value: 'pearson', label: 'Pearson' }, { value: 'spearman', label: 'Spearman' }], 'pearson');
      this.optionControls.set('corr_method', control); this.form.append(field('corr_method', control, 'Pearson показывает линейную связь исходных чисел. Spearman использует ранги и показывает монотонную связь. Ни один из них не доказывает причинность.'));
    }
    this.explanation.replaceChildren(element('span', { text: definition.explanation }), helpButton({ label: definition.name, help: definition.explanation, lesson_id: definition.lesson_id || LESSON }));
  }

  async generate() {
    const sequence = this.sequence;
    if (!this.dataset?.id || this.disposed || this.generateButton.disabled) return;
    const controls = [...this.optionControls.values()];
    if (controls.some(control => !control.checkValidity())) { this.status.textContent = 'Проверьте диапазоны числовых параметров.'; return; }
    const columns = this.columnControls.flatMap(control => control.multiple ? [...control.selectedOptions].map(option => option.value) : [control.value]);
    if (columns.some(name => !name) || columns.length > 20 || new Set(columns).size !== columns.length) { this.status.textContent = 'Выберите разные столбцы, не более 20.'; return; }
    const options = Object.fromEntries([...this.optionControls].map(([name, control]) => [name, control.type === 'number' ? Number(control.value) : control.value]));
    const body = { dataset_id: this.dataset.id, kind: this.kind.value, columns, options };
    this.generateButton.disabled = true; this.status.textContent = 'Строим выбранный график…';
    try {
      const result = await this.request(body);
      if (sequence !== this.sequence || this.disposed) return;
      while (this.generated.children.length >= 12) { const first = this.generated.firstElementChild; this.purge(first); first.remove(); }
      await this.addPlot(this.generated, result, true);
      if (sequence !== this.sequence || this.disposed) return;
      this.status.textContent = 'График добавлен. Выберите другие столбцы или другой вид анализа.';
    } catch (error) {
      if (sequence !== this.sequence || this.disposed || error.name === 'AbortError') return;
      this.status.textContent = error.message; this.onError(error);
    } finally { if (sequence === this.sequence && !this.disposed) this.generateButton.disabled = false; }
  }

  async addPlot(parent, result, removable) {
    const sequence = this.sequence;
    const plot = element('div', { className: 'cml-analysis-plot', role: 'img', 'aria-label': result.title });
    const scope = result.sample_info || {};
    const caption = `${scope.used_rows ?? '?'} из ${scope.total_rows ?? '?'} строк${scope.sampled ? '; отображение ограничено выборкой' : '; все подходящие строки'}${scope.rendered_rows ? `; точек на графике: ${scope.rendered_rows}` : ''}.`;
    const card = element('article', { className: 'cml-panel cml-analysis-card' }, [
      element('div', { className: 'cml-panel-heading' }, [element('h3', { text: result.title }), helpButton({ label: result.title, help: result.explanation, lesson_id: result.lesson_id || LESSON }),
        ...(removable ? [element('button', { className: 'cml-icon', type: 'button', text: '×', 'aria-label': `Удалить график ${result.title}`, onClick: () => { this.purge(card); card.remove(); } })] : [])]),
      plot, element('p', { className: 'cml-note', text: result.explanation }), element('p', { className: 'cml-analysis-scope', text: caption }),
      element('details', {}, [element('summary', { text: 'Численные сведения и область анализа' }), element('pre', { className: 'cml-code', text: JSON.stringify({ statistics: result.statistics, sample_info: scope }, null, 2) })]),
    ]);
    parent.append(card); this.plotNodes.add(plot);
    if (globalThis.Plotly?.react) {
      const style = getComputedStyle(document.body);
      const color = name => style.getPropertyValue(name).trim();
      try { await globalThis.Plotly.react(plot, result.traces || [], { paper_bgcolor: color('--panel'), plot_bgcolor: color('--panel'), font: { color: color('--text'), family: 'Arial, sans-serif' }, ...result.layout }, { responsive: true, displaylogo: false }); }
      catch (error) { if (sequence === this.sequence && !this.disposed) { plot.textContent = error.message; this.onError(error); } }
    } else plot.textContent = 'Графическая библиотека загружается. Численные сведения доступны ниже.';
  }

  destroy() {
    this.disposed = true; this.sequence++; this.purge();
    this.observer?.disconnect();
    for (const request of this.requests) request.abort();
    this.requests.clear(); this.container.replaceChildren();
  }
}

export function renderAnalysis(container, options) { return new AnalysisController(container, options); }
