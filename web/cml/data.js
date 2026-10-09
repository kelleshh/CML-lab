import { element, badge, formatNumber } from './dom.js';
import { action, field, heading, notice, number, optionsFromColumns, select, setOptions, numericList } from './controls.js';
import { helpButton } from './help.js';

const TASK_EXPLANATIONS = {
  regression: 'Предсказать число: стоимость, расход, длительность.',
  classification: 'Предсказать класс: вид растения, тип события, наличие дефекта.',
  clustering: 'Найти группы похожих наблюдений без правильных ответов.',
  ranking: 'Упорядочить объекты внутри одного запроса по полезности.',
  forecasting: 'Предсказать будущие значения по прошлым наблюдениям.',
  panel: 'Работать с несколькими объектами, наблюдаемыми во времени.',
  anomaly: 'Найти наблюдения, которые отличаются от большинства.',
  reduction: 'Представить много признаков меньшим количеством координат.',
};

export class DataStep {
  constructor({ api, catalogue, onTaskChange, onChange, onBrowse, onEdit, onError }) {
    this.api = api; this.catalogue = catalogue; this.onTaskChange = onTaskChange; this.onChange = onChange;
    this.onBrowse = onBrowse; this.onEdit = onEdit; this.onError = onError;
    this.dataset = null; this.taskId = 'regression'; this.saved = {}; this.exploreSequence = 0;
  }

  mount(container) { this.container = container; this.render(); return this; }

  setDataset(dataset) {
    this.saved = this.values(); this.dataset = dataset;
    const aliases = { time_series: 'forecasting', dimensionality_reduction: 'reduction', anomaly_detection: 'anomaly' };
    const hintedTask = aliases[dataset.task] || dataset.task;
    if ((this.catalogue.tasks || []).some(task => (typeof task === 'string' ? task : task.id) === hintedTask)) this.taskId = hintedTask;
    this.saved.roles = structuredClone(dataset.roles || {});
    const target = dataset.task_target || dataset.default_target || '';
    this.saved.target = target;
    const excluded = new Set([target, ...(dataset.excluded_features || []), ...Object.values(dataset.roles || {}).filter(value => typeof value === 'string')]);
    this.saved.features = (dataset.columns || []).filter(column => !excluded.has(column.name)).map(column => column.name);
    this.render(); this.onTaskChange(this.taskId);
  }

  setTask(task) { this.saved = this.values(); this.taskId = task; this.render(); }

  setConfig(config) { this.taskId = config.task || this.taskId; this.saved = structuredClone(config); this.render(); }

  values() {
    if (!this.task) return this.saved;
    const plotFeatures = [this.plotX?.value, this.plotY?.value].filter(Boolean);
    return {
      ...(plotFeatures.length ? { plot_features: plotFeatures } : {}),
      task: this.taskId, dataset_id: this.dataset?.id, target: this.target.value || null,
      features: [...this.features.querySelectorAll('input:checked')].map(control => control.value),
      roles: { time_column: this.time.value || null, entity_column: this.entity.value || null, query_column: this.query.value || null, weight_column: this.weight.value || null, reference_target: this.reference.value || null },
      temporal: { lags: numericList(this.lags.value, 'Лаги'), rolling_windows: numericList(this.rolling.value, 'Окна'), horizon: Number(this.horizon.value) },
    };
  }

  render() {
    if (!this.container) return;
    const taskOptions = (this.catalogue.tasks || []).map(task => typeof task === 'string' ? { value: task, label: task } : { value: task.id, label: task.label || task.name || task.id });
    this.task = select(taskOptions, this.taskId);
    this.task.addEventListener('change', () => { this.saved = this.values(); this.taskId = this.task.value; this.render(); this.onTaskChange(this.taskId); });
    const supervised = !['clustering', 'anomaly', 'reduction'].includes(this.taskId);
    this.target = select(optionsFromColumns(this.dataset), this.saved.target || this.dataset?.task_target || this.dataset?.default_target || '');
    this.target.disabled = !supervised;
    if (!supervised) this.target.value = '';
    this.time = select(optionsFromColumns(this.dataset), this.saved.roles?.time_column || '');
    this.entity = select(optionsFromColumns(this.dataset), this.saved.roles?.entity_column || '');
    this.query = select(optionsFromColumns(this.dataset), this.saved.roles?.query_column || '');
    this.weight = select(optionsFromColumns(this.dataset, { numeric: true }), this.saved.roles?.weight_column || '');
    this.reference = select(optionsFromColumns(this.dataset), this.saved.roles?.reference_target || '');
    this.lags = element('input', { value: (this.saved.temporal?.lags || [1, 2, 3]).join(', '), placeholder: '1, 2, 3' });
    this.rolling = element('input', { value: (this.saved.temporal?.rolling_windows || [3]).join(', '), placeholder: '3, 7' });
    this.horizon = number(this.saved.temporal?.horizon || 1, { min: 1, max: 365, step: 1 });
    this.features = element('div', { className: 'cml-column-checks' });
    const featureNames = this.saved.features || (this.dataset?.columns || []).filter(column => column.name !== this.target.value).map(column => column.name);
    for (const column of this.dataset?.columns || []) {
      const control = element('input', { type: 'checkbox', value: column.name, checked: featureNames.includes(column.name), disabled: column.name === this.target.value });
      this.features.append(element('label', { className: 'cml-check' }, [control, element('span', { className: 'feature-label', text: column.name }), badge(column.numeric ? 'число' : 'категория')]));
    }
    const roleUpdate = () => {
      const roleNames = [this.target.value, this.time.value, this.entity.value, this.query.value, this.weight.value, this.reference.value].filter(Boolean);
      this.features.querySelectorAll('input').forEach(control => { control.disabled = roleNames.includes(control.value); if (control.disabled) control.checked = false; });
      this.updatePlotFeatures(); this.onChange(); this.renderExplorer();
    };
    [this.target, this.time, this.entity, this.query, this.weight, this.reference].forEach(control => control.addEventListener('change', roleUpdate));
    this.features.addEventListener('change', () => { this.updatePlotFeatures(); this.onChange(); });
    const temporal = ['forecasting', 'panel'].includes(this.taskId);
    this.plotX = this.plotY = null;
    const sliceFields = [];
    if (['regression', 'classification'].includes(this.taskId)) {
      this.plotX = select([{ value: '', label: 'Автоматически' }], '');
      this.plotY = select([{ value: '', label: 'Один признак: 2D' }], '');
      sliceFields.push(element('div', { className: 'cml-form-grid' }, [
        field('Признак 1 для графика модели', this.plotX, { help: 'После обучения график покажет прогноз по выбранному числовому признаку. Остальные числовые признаки фиксируются на медиане, категории — на самом частом значении обучающих данных. Это срез модели, а не весь многомерный датасет.', lesson_id: this.taskId === 'classification' ? 'classification-basics' : '01-prediction' }),
        field('Признак 2 для графика модели', this.plotY, { help: 'Добавляет второй числовой признак к срезу модели. Для регрессии появляется поверхность в 3D; для классификации — области предсказанных классов. Пустой выбор оставляет один признак.', lesson_id: this.taskId === 'classification' ? 'classification-basics' : '01-prediction' }),
      ]));
      this.updatePlotFeatures(this.saved.plot_features || []);
      this.plotX.addEventListener('change', () => { this.updatePlotFeatures(); this.onChange(); });
      this.plotY.addEventListener('change', () => this.onChange());
    }
    const roleFields = [
      field('Цель: что предсказываем', this.target, { help: supervised ? 'Правильный ответ, известный для обучающих строк. Целевая переменная исключается из признаков.' : 'Для этой задачи правильные ответы не нужны. Можно отдельно задать метки только для проверки найденных групп.', lesson_id: this.taskId === 'classification' ? 'classification-basics' : '01-prediction' }),
      ...(temporal ? [field('Столбец времени', this.time, { help: 'Время задает порядок наблюдений. Обучение использует прошлое, проверка использует более поздние строки.', lesson_id: 'forecasting-lags' })] : []),
      ...(this.taskId === 'panel' ? [field('Столбец объекта', this.entity, { help: 'Идентификатор компании, человека или другого объекта. Лаги и окна вычисляются отдельно для каждого объекта.', lesson_id: 'panel-groups' })] : []),
      ...(this.taskId === 'ranking' ? [field('Столбец запроса / группы ранжирования', this.query, { help: 'Модель сравнивает объекты внутри одного запроса. Целые запросы разделяются между обучением и проверкой.', lesson_id: 'ranking-groups' })] : []),
      ...(['clustering', 'anomaly'].includes(this.taskId) ? [field('Известные ответы только для оценки', this.reference, { help: 'Эти ответы не участвуют в обучении. Они нужны только для сравнения найденных групп или аномалий с известной разметкой.', lesson_id: 'clustering-basics' })] : []),
      field('Вес строки; необязательно', this.weight, { help: 'Вес меняет вклад строки в обучение. Не каждый алгоритм поддерживает веса; сервер проверяет совместимость.', lesson_id: '20-metrics-experiment' }),
    ];
    this.explorerContainer = element('section', { className: 'cml-panel' });
    const details = element('section', { className: 'cml-panel' }, [
      heading('Данные и постановка задачи', { help: 'Сначала выберите таблицу, затем определите задачу и роли столбцов. Только после этого выбирайте алгоритм.', lesson_id: this.taskId === 'classification' ? 'classification-basics' : '21-read-data' }),
      field('Задача машинного обучения', this.task, { help: TASK_EXPLANATIONS[this.taskId], lesson_id: this.taskId === 'classification' ? 'classification-basics' : this.taskId === 'clustering' ? 'clustering-basics' : '01-prediction' }),
      element('p', { className: 'cml-note', text: TASK_EXPLANATIONS[this.taskId] }),
      element('div', { className: 'cml-toolbar' }, [action('Выбрать или загрузить датасет', () => this.onBrowse()), ...(this.dataset ? [action('Изменить таблицу', () => this.onEdit(this.dataset))] : [])]),
      element('div', { className: 'cml-badges' }, this.dataset ? [badge(this.dataset.name || this.dataset.label || this.dataset.id), badge(`${this.dataset.shape?.[0] ?? this.dataset.rows ?? '?'} строк`)] : [badge('Датасет не выбран')]),
      element('div', { className: 'cml-form-grid single' }, roleFields),
      element('div', { className: 'cml-panel-heading' }, [element('h3', { className: 'feature-label', text: 'Признаки: по чему принимаем решение' }), helpButton({ label: 'Выбор признаков', help: 'Признаки доступны модели при прогнозе. Не включайте правильный ответ, идентификаторы и информацию из будущего.', lesson_id: '25-data-leakage' })]), this.features, ...sliceFields,
      ...(temporal ? [element('div', { className: 'cml-form-grid' }, [
        field('Лаги: прошлые значения цели', this.lags, { help: '1 означает предыдущее наблюдение; 2 — наблюдение перед ним. Все лаги строятся из прошлого.', lesson_id: 'forecasting-lags' }),
        field('Размеры скользящих окон', this.rolling, { help: 'Окно 3 усредняет три прошлых наблюдения. Текущее и будущие значения в окно не входят.', lesson_id: 'forecasting-lags' }),
        field('Горизонт прогноза', this.horizon, { help: 'Число шагов вперед. При горизонте 1 прогнозируется следующее наблюдение; при 3 — третье после текущего.', lesson_id: 'forecasting-lags' }),
      ])] : []),
    ]);
    this.container.replaceChildren(element('div', { className: 'cml-workspace' }, [details, this.explorerContainer]));
    roleUpdate();
  }

  updatePlotFeatures(saved) {
    if (!this.plotX || !this.plotY) return;
    const selected = new Set([...this.features.querySelectorAll('input:checked')].map(control => control.value));
    const choices = (this.dataset?.columns || []).filter(column => column.numeric && selected.has(column.name)).map(column => ({ value: column.name, label: column.name }));
    const x = saved?.[0] ?? this.plotX.value;
    setOptions(this.plotX, [{ value: '', label: 'Автоматически' }, ...choices], x);
    const y = saved?.[1] ?? this.plotY.value;
    setOptions(this.plotY, [{ value: '', label: 'Один признак: 2D' }, ...choices.filter(column => column.value !== this.plotX.value)], y);
    this.plotY.disabled = !this.plotX.value;
    if (this.plotY.disabled) this.plotY.value = '';
  }

  renderExplorer() {
    if (!this.explorerContainer) return;
    if (!this.dataset) { this.explorerContainer.replaceChildren(heading('Данные до обучения', { help: 'Смотрите распределения и связи до выбора алгоритма.', lesson_id: '21-read-data' }), element('p', { className: 'cml-empty', text: 'Выберите датасет. Здесь появятся исходные точки в 2D и 3D.' })); return; }
    const numeric = (this.dataset.columns || []).filter(column => column.numeric);
    if (!numeric.length) { this.explorerContainer.replaceChildren(notice('В наборе нет числовых столбцов для графика. Категории и текст можно исследовать в библиотеке и преобразовать на шаге подготовки.')); return; }
    this.xAxis = select(numeric.map(column => ({ value: column.name, label: column.name })), numeric[0].name);
    this.yAxis = select(numeric.map(column => ({ value: column.name, label: column.name })), this.target.value && numeric.some(column => column.name === this.target.value) ? this.target.value : numeric[1]?.name || numeric[0].name);
    this.zAxis = select([{ value: '', label: '2D: без третьей координаты' }, ...numeric.map(column => ({ value: column.name, label: column.name }))], '');
    this.color = select(optionsFromColumns(this.dataset), this.target.value || this.reference.value || '');
    this.rawPlot = element('div', { className: 'cml-plot large', id: `raw-explore-${++this.exploreSequence}` });
    this.plotStatus = element('p', { className: 'cml-note', 'aria-live': 'polite' });
    const table = element('div', { className: 'cml-table-scroll' });
    this.explorerContainer.replaceChildren(heading('Исходные данные: 2D и 3D', { help: 'Каждая точка соответствует строке исходного датасета. Выберите оси и цвет. Эти графики доступны до обучения.', lesson_id: '21-read-data' }),
      element('div', { className: 'cml-form-grid' }, [
        field('Горизонтальная ось', this.xAxis, { help: 'Числовой признак на горизонтальной оси.', lesson_id: '21-read-data' }),
        field('Вертикальная ось', this.yAxis, { help: 'Числовой признак или числовая цель. Для категориальной цели используйте цвет.', lesson_id: '21-read-data' }),
        field('Третья ось', this.zAxis, { help: 'Добавляет третью координату. 3D-график можно поворачивать мышью.', lesson_id: '21-read-data' }),
        field('Цвет точек', this.color, { help: 'Цвет помогает увидеть классы, группы или величину числовой цели. Значения берутся из выбранного столбца.', lesson_id: '21-read-data' }),
      ]), this.rawPlot, this.plotStatus, table);
    [this.xAxis, this.yAxis, this.zAxis, this.color].forEach(control => control.addEventListener('change', () => this.explore()));
    this.explore();
  }

  async explore() {
    if (!this.dataset || !this.rawPlot) return;
    const sequence = ++this.exploreSequence;
    const datasetId = this.dataset.id;
    const query = new URLSearchParams({ x: this.xAxis.value, y: this.yAxis.value, sample_size: '1500', seed: '42' });
    if (this.zAxis.value) query.set('z', this.zAxis.value);
    if (this.color.value) query.set('color', this.color.value);
    if (this.target.value) query.set('target', this.target.value);
    try {
      const result = await this.api.request(`/datasets/${encodeURIComponent(datasetId)}/explore?${query}`);
      if (sequence !== this.exploreSequence || this.dataset?.id !== datasetId) return;
      const points = result.points || {};
      const style = getComputedStyle(document.body);
      const color = style.getPropertyValue('--feature').trim();
      const categoryLabels = points.color_categories || [];
      const marker = { size: this.zAxis.value ? 4 : 7, color: points.color || color, opacity: .8, colorscale: [[0, '#315e54'], [.5, '#b8af9d'], [1, '#8d521d']], showscale: Boolean(points.color) && points.color_kind === 'numeric' };
      const dimensional = Boolean(this.zAxis.value);
      let traces;
      if (points.color_kind === 'categorical' && points.color) {
        const categories = [...new Set(points.color)];
        const palette = ['#315e54', '#8d521d', '#37624c', '#8b448a', '#437684', '#7a6233', '#6a6a6a', '#af4055'];
        traces = categories.map((category, categoryIndex) => {
          const positions = points.color.map((value, index) => value === category ? index : -1).filter(index => index >= 0);
          const label = Array.isArray(categoryLabels) ? categoryLabels[category] ?? category : categoryLabels[category] ?? category;
          return { type: dimensional ? 'scatter3d' : 'scatter', mode: 'markers', name: String(label), x: positions.map(index => points.x[index]), y: positions.map(index => points.y[index]), ...(dimensional ? { z: positions.map(index => points.z[index]) } : {}), marker: { ...marker, color: palette[categoryIndex % palette.length], showscale: false }, customdata: positions.map(index => points.indices[index]) };
        });
      } else traces = [{ type: dimensional ? 'scatter3d' : 'scatter', mode: 'markers', x: points.x, y: points.y, ...(dimensional ? { z: points.z } : {}), marker, customdata: points.indices, name: 'Строки датасета' }];
      const axis = name => ({ title: { text: name, font: { color: style.getPropertyValue(name === this.target.value || name === this.reference.value ? '--target' : '--feature').trim() } }, gridcolor: style.getPropertyValue('--line').trim() });
      const layout = { paper_bgcolor: style.getPropertyValue('--panel').trim(), plot_bgcolor: style.getPropertyValue('--panel').trim(), font: { color: style.getPropertyValue('--text').trim(), family: 'Arial, sans-serif', size: 13 }, margin: { t: 15, l: 55, r: 25, b: 55 }, xaxis: axis(this.xAxis.value), yaxis: axis(this.yAxis.value), scene: { xaxis: axis(this.xAxis.value), yaxis: axis(this.yAxis.value), zaxis: axis(this.zAxis.value) }, legend: { orientation: 'h' }, uirevision: `${datasetId}-${dimensional}` };
      if (globalThis.Plotly) await Plotly.react(this.rawPlot, traces, layout, { responsive: true, displaylogo: false });
      else this.rawPlot.textContent = 'Библиотека графиков загружается. Попробуйте изменить ось через несколько секунд.';
      this.plotStatus.textContent = `Показано ${points.indices?.length || 0} исходных строк. Зеленый: признаки. Коричневый: цель. Цвет точек: ${this.color.value || 'одинаковый'}.`;
    } catch (error) { if (sequence === this.exploreSequence) this.onError(error); }
  }
}
