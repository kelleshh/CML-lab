const PALETTES = {
  dark: {
    text: '#dce6f3', muted: '#8596ad', grid: '#263348', paper: '#111c23',
    train: '#63d9b6', validation: '#ab8cff', test: '#ffc773', actual: '#ff8a84',
    line: '#65dfc3', zero: '#6b7e97', surface: [[0, '#163753'], [.4, '#336c83'], [.72, '#69c4bd'], [1, '#e8ebac']],
  },
  light: {
    text: '#23324b', muted: '#64748b', grid: '#e1e7f0', paper: '#fff',
    train: '#128c72', validation: '#7953c5', test: '#b67615', actual: '#db625a',
    line: '#108a74', zero: '#8190a3', surface: [[0, '#e9f3f5'], [.4, '#82bcc6'], [.72, '#368f96'], [1, '#215771']],
  },
};
const COEFFICIENT_COLORS = ['#64d6b8', '#a88aff', '#ff9a8a', '#6fb8ff', '#e7c16a', '#eb8bc4', '#83c894', '#acb7ff'];
const IDS = ['fit', '3d', 'loss', 'coef', 'residual', 'prediction', 'path', 'correlation', 'surface', 'geometry', 'learning-curve', 'importance'];
const SPLIT_LABELS = {train: 'Обучение', validation: 'Выбор настроек', val: 'Выбор настроек', test: 'Итоговая проверка'};
const isNumber = value => typeof value === 'number' && Number.isFinite(value);
const escape = value => String(value ?? '').replace(/[&<>"']/g, char => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[char]));
const average = values => values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0;
const finiteValues = values => (values || []).filter(isNumber);
const vector = value => Array.isArray(value) ? value : [];
const range = (start, end, count) => Array.from({length: count}, (_, index) => start + (end - start) * index / (count - 1));
const shortName = value => String(value).length > 32 ? `${String(value).slice(0, 29)}…` : String(value);

function wrapNote(value, maximum) {
  const lines = [''];
  for (const word of String(value).split(/\s+/)) {
    if (lines[lines.length - 1].length + word.length > maximum && lines[lines.length - 1]) lines.push('');
    lines[lines.length - 1] += `${lines[lines.length - 1] ? ' ' : ''}${word}`;
  }
  return lines;
}

function bounds(values, padding = .08) {
  const clean = finiteValues(values);
  if (!clean.length) return [-1, 1];
  let min = Infinity, max = -Infinity;
  for (const value of clean) { min = Math.min(min, value); max = Math.max(max, value); }
  const span = max - min || Math.max(Math.abs(min) * .2, 1);
  return [min - span * padding, max + span * padding];
}

function sampledIndices(length, maximum = 2000) {
  if (length <= maximum) return Array.from({length}, (_, index) => index);
  const stride = length / maximum;
  return Array.from({length: maximum}, (_, index) => Math.floor(index * stride));
}

function rawData(result) {
  const data = result.plot_data || result.prediction_data || {};
  const names = vector(data.feature_names).length ? data.feature_names : vector(result.raw_feature_names).length ? result.raw_feature_names : result.feature_names || [];
  let matrix = data.X || [];
  if (!Array.isArray(matrix) && typeof matrix === 'object') {
    const columns = names.map(name => vector(matrix[name]));
    matrix = columns[0] ? columns[0].map((_, index) => columns.map(column => column[index])) : [];
  }
  return {X: vector(matrix), y: vector(data.y), split: vector(data.split), names};
}

function predictionData(result, frame) {
  const source = result.predictions || {};
  const actual = vector(source.actual);
  const framePredictions = frame?.predicted || frame?.predictions;
  const candidate = Array.isArray(framePredictions) ? framePredictions : framePredictions?.predicted;
  if (frame && (Object.hasOwn(frame, 'predicted') || Object.hasOwn(frame, 'predictions')) && vector(candidate).length !== actual.length) {
    return {actual, predicted: [], residual: [], split: vector(source.split), indices: vector(source.indices), unavailable: true};
  }
  const predicted = vector(Array.isArray(framePredictions) ? framePredictions : framePredictions?.predicted).length ?
    (Array.isArray(framePredictions) ? framePredictions : framePredictions.predicted) : vector(source.predicted);
  const residual = predicted.length === actual.length ? actual.map((value, index) => value - predicted[index]) : vector(source.residual);
  return {actual, predicted, residual, split: vector(source.split), indices: vector(source.indices)};
}

function correlationMatrix(matrix, count) {
  const columns = Array.from({length: count}, (_, index) => matrix.map(row => row[index]));
  return columns.map(left => columns.map(right => {
    const pairs = left.map((value, index) => [value, right[index]]).filter(pair => pair.every(isNumber));
    if (pairs.length < 2) return null;
    const ml = average(pairs.map(pair => pair[0])), mr = average(pairs.map(pair => pair[1]));
    let covariance = 0, varianceLeft = 0, varianceRight = 0;
    for (const [a, b] of pairs) {
      covariance += (a - ml) * (b - mr);
      varianceLeft += (a - ml) ** 2;
      varianceRight += (b - mr) ** 2;
    }
    return varianceLeft && varianceRight ? covariance / Math.sqrt(varianceLeft * varianceRight) : null;
  }));
}

/** Plotly views use recorded model results; animations never fabricate optimizer steps. */
export class ChartManager {
  constructor() {
    this.result = null;
    this.options = {theme: 'dark', xFeature: 0, yFeature: 1};
    this.frameIndex = null;
    this.activeFrame = null;
    this.renderVersion = 0;
    this.dataRevision = 0;
    this.listeners = new Map();
    this.pointEdit = null;
    this.programmaticRelayout = new Map();
    this.resizeObserver = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(entries => {
      if (!window.Plotly) return;
      for (const {target} of entries) if (target.data && target.clientWidth > 0) window.Plotly.Plots.resize(target);
    });
    this.observed = new Set();
  }

  get colors() { return PALETTES[this.options.theme] || PALETTES.dark; }

  config(name) {
    return {
      responsive: true, displaylogo: false, scrollZoom: true,
      displayModeBar: 'hover', locale: 'ru',
      toImageButtonOptions: {format: 'png', filename: `linear-lab-${name}`, scale: 2},
      modeBarButtonsToRemove: ['sendDataToCloud', 'autoScale2d'],
      editable: false,
      edits: {shapePosition: name === 'fit' && this.options.editPoints === true},
    };
  }

  layout(overrides = {}, note = '') {
    const c = this.colors;
    const axis = {gridcolor: c.grid, zerolinecolor: c.zero, linecolor: c.grid, tickfont: {color: c.muted, size: 10}, automargin: true};
    return {
      paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
      font: {family: 'Inter, ui-sans-serif, system-ui, sans-serif', size: 12, color: c.text},
      margin: {l: 54, r: 22, t: note ? 42 : 20, b: 52},
      hoverlabel: {bgcolor: c.paper, font: {color: c.text, size: 12}, bordercolor: c.grid},
      legend: {orientation: 'h', y: 1.02, yanchor: 'bottom', x: 0, font: {size: 10}, bgcolor: 'rgba(0,0,0,0)'},
      xaxis: {...axis}, yaxis: {...axis},
      colorway: COEFFICIENT_COLORS,
      uirevision: `linear-lab:${this.dataRevision}:${this.options.xFeature}:${this.options.yFeature}`,
      transition: {duration: this.activeFrame ? 120 : 240, easing: 'cubic-in-out'},
      annotations: note ? [{xref: 'paper', yref: 'paper', x: 0, y: 1, yshift: 25, xanchor: 'left', yanchor: 'bottom', showarrow: false, text: escape(note), align: 'left', font: {size: 10, color: c.muted}}] : [],
      ...overrides,
    };
  }

  async draw(name, traces, layout = {}, note = '') {
    const element = document.getElementById(`plot-${name}`);
    if (!element || !this.isVisible(name)) return;
    if (!window.Plotly) {
      element.textContent = 'Графики еще загружаются. Проверь локальный файл Plotly.';
      return;
    }
    const plotLayout = this.layout(layout, note);
    if (name === 'fit') {
      this.pointEdit = null;
      plotLayout.shapes = [];
    }
    if (note) {
      const lines = wrapNote(note, Math.max(36, Math.floor(((element.clientWidth || 600) - 85) / 5.8)));
      plotLayout.annotations[0].text = lines.map(escape).join('<br>');
      plotLayout.margin.t = Math.max(plotLayout.margin.t || 0, 38 + lines.length * 12);
    }
    await window.Plotly.react(element, traces, plotLayout, this.config(name));
    if (!this.observed.has(element)) { this.resizeObserver?.observe(element); this.observed.add(element); }
    if (name === 'fit') this.attachPointEvents(element);
  }

  isVisible(name) {
    const visible = this.options.visible;
    return !visible || (visible instanceof Set ? visible.has(name) || visible.has(`plot-${name}`) : visible.includes(name) || visible.includes(`plot-${name}`));
  }

  attachPointEvents(element) {
    const previous = this.listeners.get(element);
    if (previous) {
      element.removeListener?.('plotly_click', previous.click);
      element.removeListener?.('plotly_relayout', previous.relayout);
    }
    const click = event => {
      const point = event.points?.[0];
      if (!point || !isNumber(point.x) || !isNumber(point.y)) return;
      if (this.options.editPoints) {
        if (Number.isInteger(point.customdata)) this.selectEditablePoint(element, point).catch(error => console.error('Не удалось выбрать точку для перемещения', error));
        return;
      }
      this.options.onPoint?.(point.x, point.y, {index: point.customdata, source: point.data?.name, event});
    };
    const relayout = update => this.moveEditablePoint(element, update).catch(error => console.error('Не удалось переместить точку', error));
    element.on?.('plotly_click', click);
    element.on?.('plotly_relayout', relayout);
    this.listeners.set(element, {click, relayout});
  }

  async selectEditablePoint(element, point) {
    const data = rawData(this.result);
    const {x: selected} = this.selectedFeatures(data);
    const xRange = bounds((element.data || []).flatMap(trace => vector(trace.x)));
    const yRange = bounds((element.data || []).flatMap(trace => vector(trace.y)));
    const rx = (xRange[1] - xRange[0]) * .018;
    const ry = (yRange[1] - yRange[0]) * .025;
    const shape = {
      type: 'circle', xref: 'x', yref: 'y',
      x0: point.x - rx, x1: point.x + rx, y0: point.y - ry, y1: point.y + ry,
      line: {color: this.colors.line, width: 2.5},
      fillcolor: this.options.theme === 'light' ? 'rgba(18,140,114,.25)' : 'rgba(99,217,182,.25)',
      layer: 'above', editable: true,
    };
    this.pointEdit = {index: point.customdata, x: point.x, y: point.y, featureName: data.names[selected], shape};
    await this.replaceEditableShape(element, shape);
  }

  async replaceEditableShape(element, shape) {
    this.programmaticRelayout.set(element, (this.programmaticRelayout.get(element) || 0) + 1);
    try { await window.Plotly.relayout(element, {shapes: [shape]}); }
    finally {
      const pending = (this.programmaticRelayout.get(element) || 1) - 1;
      if (pending) this.programmaticRelayout.set(element, pending);
      else this.programmaticRelayout.delete(element);
    }
  }

  async moveEditablePoint(element, update) {
    if (!this.options.editPoints || !this.pointEdit || this.programmaticRelayout.has(element)) return;
    const complete = update?.shapes?.[0] || update?.['shapes[0]'];
    const fields = ['x0', 'x1', 'y0', 'y1'];
    const patch = {};
    for (const field of fields) {
      const key = `shapes[0].${field}`;
      if (Object.hasOwn(update || {}, key)) patch[field] = update[key];
    }
    if (!complete && !Object.keys(patch).length) return;
    const shape = {...this.pointEdit.shape, ...(complete || {}), ...patch};
    if (!fields.every(field => isNumber(shape[field]))) return;
    const x = (shape.x0 + shape.x1) / 2, y = (shape.y0 + shape.y1) / 2;
    const previous = this.pointEdit;
    const sameSize = ['x', 'y'].every(axis => {
      const size = shape[`${axis}1`] - shape[`${axis}0`];
      const original = previous.shape[`${axis}1`] - previous.shape[`${axis}0`];
      return Math.abs(size - original) <= 1e-9 * Math.max(1, Math.abs(original));
    });
    if (!sameSize) {
      await this.replaceEditableShape(element, previous.shape);
      return;
    }
    previous.shape = shape;
    const changed = Math.abs(x - previous.x) > 1e-10 * Math.max(1, Math.abs(previous.x)) || Math.abs(y - previous.y) > 1e-10 * Math.max(1, Math.abs(previous.y));
    if (!changed) return;
    previous.x = x;
    previous.y = y;
    // Public Plotly shape events carry data coordinates; no pixel/axis internals are used.
    await this.options.onPointMove?.({index: previous.index, x, y, featureName: previous.featureName});
  }

  async empty(name, message) {
    return this.draw(name, [], {
      xaxis: {visible: false}, yaxis: {visible: false}, showlegend: false,
      annotations: [{xref: 'paper', yref: 'paper', x: .5, y: .5, showarrow: false, text: wrapNote(message, 46).map(escape).join('<br>'), font: {color: this.colors.muted, size: 13}, align: 'center'}],
    });
  }

  async render(result, options = {}) {
    if (result !== this.result) this.dataRevision += 1;
    this.result = result || {};
    this.options = {...this.options, ...options};
    this.activeFrame = null;
    this.frameIndex = null;
    const version = ++this.renderVersion;
    const methods = ['fit', 'threeD', 'loss', 'coefficients', 'residuals', 'prediction', 'path', 'correlation', 'surface', 'geometry', 'learningCurve', 'importance'];
    const outcomes = await Promise.allSettled(methods.map(method => this[method]()));
    if (version !== this.renderVersion) return;
    outcomes.forEach((outcome, index) => {
      if (outcome.status === 'rejected') {
        console.error(`Ошибка графика ${IDS[index]}`, outcome.reason);
        this.empty(IDS[index], 'Не удалось построить этот график. Остальные панели доступны.');
      }
    });
  }

  selectedFeatures(data) {
    const x = Math.max(0, Math.min(Number(this.options.xFeature) || 0, data.names.length - 1));
    let y = Math.max(0, Math.min(Number(this.options.yFeature) || 0, data.names.length - 1));
    if (x === y && data.names.length > 1) y = (x + 1) % data.names.length;
    return {x, y};
  }

  splitTraces(values, makeTrace) {
    const labels = [...new Set(values.split.length ? values.split : ['train'])];
    return labels.filter(split => this.options.revealTest !== false || split !== 'test').map(split => {
      const indices = sampledIndices(values.length).filter(index => (values.split[index] || 'train') === split);
      return makeTrace(indices, split, SPLIT_LABELS[split] || split, this.colors[split === 'val' ? 'validation' : split] || this.colors.train);
    }).filter(trace => vector(trace.x).length > 0);
  }

  async fit() {
    const result = this.result, data = rawData(result);
    if (!data.X.length || !data.y.length || !data.names.length) return this.empty('fit', 'Выбери числовой признак и обучи модель: здесь появятся точки и предсказания.');
    const {x} = this.selectedFeatures(data), name = data.names[x];
    const traces = this.splitTraces({split: data.split, length: data.y.length}, (indices, split, label, color) => ({
      type: 'scattergl', mode: 'markers', name: label, x: indices.map(index => data.X[index]?.[x]), y: indices.map(index => data.y[index]),
      customdata: indices.map(index => result.predictions?.indices?.[index] ?? index), marker: {color: split === 'train' ? this.colors.actual : color, size: 6, opacity: .75},
      hovertemplate: `${escape(name)}: %{x:.4g}<br>Значение: %{y:.4g}<extra>${escape(label)}</extra>`,
    }));
    const grid = this.activeFrame ? this.activeFrame.grid || this.activeFrame.prediction_grid : result.prediction_grid;
    let note = data.y.length > 2000 ? `На экране до 2000 из ${data.y.length} точек; обучение использует полный набор.` : '';
    const axisIndex = this.gridFeatureIndex(grid, data, x);
    if (grid && axisIndex !== -1) {
      if (!vector(grid.y).length && axisIndex === 0 && vector(grid.z).length) {
        traces.push({type: 'scatter', mode: 'lines', name: 'Предсказание', x: grid.x, y: grid.z, line: {color: this.colors.line, width: 3}, hovertemplate: 'Признак: %{x:.4g}<br>Предсказание: %{y:.4g}<extra></extra>'});
        if (grid.note) note = grid.note;
      } else if (vector(grid.y).length && Array.isArray(grid.z?.[0])) {
        if (axisIndex === 0) {
          const middle = Math.floor(grid.y.length / 2);
          traces.push({type: 'scatter', mode: 'lines', name: 'Срез модели', x: grid.x, y: grid.z[middle], line: {color: this.colors.line, width: 3}});
          note = `Срез: второй признак = ${Number(grid.y[middle]).toPrecision(3)}; остальные фиксированы.`;
        } else {
          const middle = Math.floor(grid.x.length / 2);
          traces.push({type: 'scatter', mode: 'lines', name: 'Срез модели', x: grid.y, y: grid.z.map(row => row[middle]), line: {color: this.colors.line, width: 3}});
          note = `Срез: первый признак = ${Number(grid.x[middle]).toPrecision(3)}; остальные фиксированы.`;
        }
      }
    } else {
      const predictions = predictionData(result, this.activeFrame);
      const locations = predictions.actual.length === data.X.length ? predictions.actual.map((_, index) => index) : predictions.indices;
      const shown = sampledIndices(predictions.predicted.length).filter(index => this.options.revealTest !== false || predictions.split[index] !== 'test');
      if (predictions.predicted.length) traces.push({
        type: shown.length <= 1000 ? 'scatter' : 'scattergl', mode: 'markers', name: 'Предсказания',
        x: shown.map(index => data.X[locations[index]]?.[x]),
        y: shown.map(index => predictions.predicted[index]),
        marker: {color: this.colors.line, symbol: 'diamond-open', size: 5, opacity: .65},
      });
      note = 'Проекция на один признак. Предсказания учитывают все выбранные признаки.';
      if (predictions.unavailable) note = 'Предсказания кадра не совпадают с текущей выборкой. Показаны только исходные точки.';
    }
    if (this.activeFrame && !this.activeFrame.grid && !this.activeFrame.prediction_grid) {
      if (!this.activeFrame.predicted && !this.activeFrame.predictions) note = 'Здесь итоговая модель; реальные шаги показаны на графиках ошибки и коэффициентов.';
      else if (!predictionData(result, this.activeFrame).unavailable) note += ' Предсказания текущего записанного шага.';
    }
    if (this.options.editPoints) note = `Выберите точку, затем переместите круг. ${note}`;
    return this.draw('fit', traces, {xaxis: {...this.layout().xaxis, title: escape(name)}, yaxis: {...this.layout().yaxis, title: escape(result.target_name || 'Целевая величина')}}, note);
  }

  gridFeatureIndex(grid, data, featureIndex) {
    if (!grid) return -1;
    const features = grid.features || grid.feature_names;
    if (!features?.length) return featureIndex < 2 ? featureIndex : -1;
    return features.findIndex(feature => feature === featureIndex || feature === data.names[featureIndex]);
  }

  async threeD() {
    const result = this.result, data = rawData(result);
    if (data.names.length < 2 || !data.X.length) return this.empty('3d', 'Для 3D нужны два разных числовых признака. Добавь второй признак.');
    const {x, y} = this.selectedFeatures(data);
    const traces = this.splitTraces({split: data.split, length: data.y.length}, (indices, split, label, color) => ({
      type: 'scatter3d', mode: 'markers', name: label,
      x: indices.map(index => data.X[index]?.[x]), y: indices.map(index => data.X[index]?.[y]), z: indices.map(index => data.y[index]),
      marker: {color: split === 'train' ? this.colors.actual : color, size: 3, opacity: .75}, hovertemplate: `${escape(data.names[x])}: %{x:.3g}<br>${escape(data.names[y])}: %{y:.3g}<br>Значение: %{z:.3g}<extra>${escape(label)}</extra>`,
    }));
    const grid = this.activeFrame ? this.activeFrame.grid || this.activeFrame.prediction_grid : result.prediction_grid;
    const ix = this.gridFeatureIndex(grid, data, x), iy = this.gridFeatureIndex(grid, data, y);
    let note = 'Вращай мышью. Колесо меняет масштаб; двойной щелчок возвращает вид.';
    if (vector(grid?.y).length && Array.isArray(grid.z?.[0]) && ((ix === 0 && iy === 1) || (ix === 1 && iy === 0))) {
      traces.push({
        type: 'surface', name: 'Поверхность модели', x: ix === 0 ? grid.x : grid.y, y: ix === 0 ? grid.y : grid.x,
        z: ix === 0 ? grid.z : grid.x.map((_, index) => grid.y.map((__, second) => grid.z[second][index])),
        colorscale: [[0, this.colors.train], [1, this.colors.line]], opacity: .55, showscale: false,
        hovertemplate: 'Признак 1: %{x:.3g}<br>Признак 2: %{y:.3g}<br>Предсказание: %{z:.3g}<extra></extra>',
      });
      note = grid.note || 'Поверхность предсказания. Не показанные признаки фиксированы на опорных значениях.';
    } else {
      note = 'Точки в трех измерениях. Для этих осей поверхность предсказания не рассчитана.';
      const predictions = predictionData(result, this.activeFrame);
      if (predictions.predicted.length === data.X.length) {
        const indices = sampledIndices(data.X.length).filter(index => this.options.revealTest !== false || predictions.split[index] !== 'test');
        traces.push({type: 'scatter3d', mode: 'markers', name: this.activeFrame ? 'Предсказания шага' : 'Предсказания',
          x: indices.map(index => data.X[index]?.[x]), y: indices.map(index => data.X[index]?.[y]), z: indices.map(index => predictions.predicted[index]),
          marker: {color: this.colors.line, symbol: 'diamond', size: 2.5, opacity: .7},
          hovertemplate: 'Признак 1: %{x:.3g}<br>Признак 2: %{y:.3g}<br>Предсказание: %{z:.3g}<extra></extra>'});
        if (this.activeFrame) note = 'Реальные предсказания записанного шага; поверхность для кадра не рассчитана.';
      }
    }
    const axis = title => ({title: escape(title), gridcolor: this.colors.grid, zerolinecolor: this.colors.zero, backgroundcolor: 'rgba(0,0,0,0)', showbackground: false, tickfont: {size: 10, color: this.colors.muted}});
    return this.draw('3d', traces, {
      margin: {l: 0, r: 0, t: 42, b: 0}, scene: {xaxis: axis(data.names[x]), yaxis: axis(data.names[y]), zaxis: axis(result.target_name || 'Целевая величина'), camera: {eye: {x: 1.4, y: 1.5, z: 1.15}}, aspectmode: 'auto'},
      legend: {orientation: 'h', x: 0, y: 1, yanchor: 'bottom', font: {size: 10}},
    }, note);
  }

  visibleTrace() {
    const trace = vector(this.result.trace);
    if (this.frameIndex === null) return trace;
    return trace.slice(0, this.frameIndex + 1);
  }

  async loss() {
    const trace = this.visibleTrace();
    if (!trace.length) return this.empty('loss', 'Эта реализация возвращает итоговое решение без истории шагов. Выбери SGD для просмотра обучения.');
    const definitions = [['train_loss', 'Ошибка обучения', this.colors.train], ['validation_loss', 'Ошибка выбора настроек', this.colors.validation], ['objective', 'Ошибка + штраф', this.colors.actual]];
    const traces = definitions.map(([key, name, color]) => ({
      type: 'scatter', mode: trace.length === 1 ? 'markers' : 'lines', name,
      x: trace.map((frame, index) => frame.step ?? index), y: trace.map(frame => key === 'validation_loss' ? frame.validation_loss ?? frame.val_loss : frame[key]),
      connectgaps: false, line: {color, width: key === 'objective' ? 1.8 : 2.5, dash: key === 'objective' ? 'dot' : 'solid'},
      marker: {size: 7}, hovertemplate: 'Шаг: %{x}<br>Значение: %{y:.5g}<extra>%{fullData.name}</extra>',
    })).filter(trace => trace.y.some(isNumber));
    if (!traces.length) return this.empty('loss', 'В записанной истории нет значений ошибки. Коэффициенты доступны в соседней панели.');
    let note = this.result.trace_label || 'Реальные записанные состояния алгоритма.';
    if (this.result.trace_kind === 'active_set') note += ' Здесь шаг означает изменение активных признаков.';
    if (this.result.trace_kind === 'final' || trace.length === 1) note += ' Доступна только конечная точка.';
    const all = vector(this.result.trace);
    return this.draw('loss', traces, {
      xaxis: {...this.layout().xaxis, title: 'Записанный шаг', range: all.length > 1 ? bounds(all.map((frame, index) => frame.step ?? index), .025) : undefined},
      yaxis: {...this.layout().yaxis, title: this.result.trace_loss_label || 'MSE и полная цель', type: this.options.lossLog ? 'log' : 'linear'},
    }, note);
  }

  async coefficients() {
    const result = this.result, names = vector(result.feature_names);
    const trace = this.visibleTrace().filter(frame => Array.isArray(frame.coef));
    if (!names.length) return this.empty('coef', 'После обучения здесь появится вес каждого признака.');
    if (vector(result.trace).length > 1 && trace.length) {
      const count = Math.min(names.length, 16);
      const traces = names.slice(0, count).map((name, index) => ({
        type: 'scatter', mode: trace.length === 1 ? 'markers' : 'lines', name: shortName(name),
        x: trace.map((frame, position) => frame.step ?? position), y: trace.map(frame => frame.coef[index]),
        line: {color: COEFFICIENT_COLORS[index % COEFFICIENT_COLORS.length], width: 2},
        hovertemplate: `${escape(name)}<br>Шаг: %{x}<br>Коэффициент: %{y:.5g}<extra></extra>`,
      }));
      const all = vector(result.trace);
      return this.draw('coef', traces, {showlegend: count <= 8, xaxis: {...this.layout().xaxis, title: 'Записанный шаг', range: bounds(all.map((frame, index) => frame.step ?? index), .025)}, yaxis: {...this.layout().yaxis, title: 'Коэффициент'}}, `Коэффициенты после преобразования признаков.${count > 8 ? ' Названия показаны при наведении.' : ''}${names.length > 16 ? ' Показаны первые 16; полный список доступен в результатах.' : ''}`);
    }
    const coefficients = this.activeFrame?.coef || result.coefficients || [];
    const ranked = names.map((name, index) => ({name, value: coefficients[index]})).filter(item => isNumber(item.value)).sort((left, right) => Math.abs(right.value) - Math.abs(left.value)).slice(0, 24).reverse();
    return this.draw('coef', [{
      type: 'bar', orientation: 'h', x: ranked.map(item => item.value), y: ranked.map(item => shortName(item.name)),
      marker: {color: ranked.map(item => item.value >= 0 ? this.colors.train : this.colors.actual)},
      customdata: ranked.map(item => item.name), hovertemplate: '%{customdata}<br>Коэффициент: %{x:.5g}<extra></extra>',
    }], {showlegend: false, margin: {l: 110, r: 20, t: 40, b: 48}, xaxis: {...this.layout().xaxis, title: 'Коэффициент'}, yaxis: {...this.layout().yaxis, type: 'category'}}, 'Вес признака после преобразований. Сравнивай величины при одинаковом масштабе.');
  }

  async residuals() {
    const values = predictionData(this.result, this.activeFrame);
    if (values.unavailable) return this.empty('residual', 'Предсказания кадра не совпадают с текущей выборкой; остатки для этого шага недоступны.');
    if (!values.residual.length) return this.empty('residual', 'После обучения здесь будут остатки: реальное значение минус предсказание.');
    const traces = this.splitTraces({split: values.split, length: values.residual.length}, (indices, split, label, color) => ({
      type: indices.length <= 1000 ? 'scatter' : 'scattergl', mode: 'markers', name: label, x: indices.map(index => values.predicted[index]), y: indices.map(index => values.residual[index]),
      marker: {color, size: 5, opacity: .7}, hovertemplate: 'Предсказание: %{x:.4g}<br>Остаток: %{y:.4g}<extra>%{fullData.name}</extra>',
    }));
    return this.draw('residual', traces, {
      xaxis: {...this.layout().xaxis, title: 'Предсказание'}, yaxis: {...this.layout().yaxis, title: 'Реальное − предсказанное'},
      shapes: [{type: 'line', xref: 'paper', x0: 0, x1: 1, y0: 0, y1: 0, line: {color: this.colors.zero, dash: 'dot', width: 1.5}}],
    }, 'Систематическая форма остатков указывает на структуру, которую модель не описала.');
  }

  async prediction() {
    const values = predictionData(this.result, this.activeFrame);
    if (values.unavailable) return this.empty('prediction', 'Предсказания кадра не совпадают с текущей выборкой. Выбери другой шаг или обнови результат.');
    if (!values.predicted.length) return this.empty('prediction', 'Обучи модель, чтобы сравнить предсказания с реальными значениями.');
    const traces = this.splitTraces({split: values.split, length: values.predicted.length}, (indices, split, label, color) => ({
      type: indices.length <= 1000 ? 'scatter' : 'scattergl', mode: 'markers', name: label, x: indices.map(index => values.actual[index]), y: indices.map(index => values.predicted[index]),
      marker: {color, size: 5, opacity: .7}, hovertemplate: 'Реальное: %{x:.4g}<br>Предсказанное: %{y:.4g}<extra>%{fullData.name}</extra>',
    }));
    const shownIndices = values.actual.map((_, index) => index).filter(index => this.options.revealTest !== false || values.split[index] !== 'test');
    const interval = bounds(shownIndices.flatMap(index => [values.actual[index], values.predicted[index]]));
    traces.unshift({type: 'scatter', mode: 'lines', name: 'Точное совпадение', x: interval, y: interval, line: {color: this.colors.zero, dash: 'dash', width: 1.5}, hoverinfo: 'skip'});
    return this.draw('prediction', traces, {xaxis: {...this.layout().xaxis, title: 'Реальное значение'}, yaxis: {...this.layout().yaxis, title: 'Предсказанное значение'}}, 'Чем ближе точка к диагонали, тем меньше ошибка на этом объекте.');
  }

  async path() {
    const path = this.result.regularization_path;
    if (!path || !vector(path.alphas).length || !vector(path.coefficients).length) return this.empty('path', 'Путь по силе регуляризации доступен для моделей со штрафом. Включи расчет пути перед обучением.');
    const names = path.feature_names || this.result.feature_names || [];
    const rows = path.coefficients;
    const featureMajor = path.coefficient_axis === 'features' || (rows.length === names.length && rows.length !== path.alphas.length);
    const traces = names.slice(0, 16).map((name, index) => ({
      type: 'scatter', mode: 'lines', name: shortName(name), x: path.alphas,
      y: featureMajor ? rows[index] : rows.map(row => row[index]),
      line: {color: COEFFICIENT_COLORS[index % COEFFICIENT_COLORS.length], width: 2},
      hovertemplate: `${escape(name)}<br>α: %{x:.5g}<br>Коэффициент: %{y:.5g}<extra></extra>`,
    }));
    return this.draw('path', traces, {
      showlegend: names.length <= 8, xaxis: {...this.layout().xaxis, title: 'Сила штрафа α', type: path.alphas.every(value => value > 0) ? 'log' : 'linear'}, yaxis: {...this.layout().yaxis, title: 'Коэффициент'},
    }, 'Каждая α задает отдельную задачу. На графике решения отдельных задач, а не шаги одной оптимизации.');
  }

  async correlation() {
    const data = rawData(this.result);
    const diagnostics = this.result.diagnostics || {};
    const supplied = diagnostics.correlation;
    const names = supplied?.feature_names || supplied?.names || diagnostics.correlation_feature_names || data.names;
    const matrix = supplied?.matrix || supplied?.values || (Array.isArray(supplied) ? supplied : null) || (data.X.length ? correlationMatrix(data.X, Math.min(data.names.length, 25)) : []);
    if (!matrix.length || !names.length) return this.empty('correlation', 'Корреляции появятся после выбора числовых признаков.');
    const count = Math.min(names.length, matrix.length, 25);
    return this.draw('correlation', [{
      type: 'heatmap', x: names.slice(0, count).map(shortName), y: names.slice(0, count).map(shortName), z: matrix.slice(0, count).map(row => row.slice(0, count)),
      zmin: -1, zmax: 1, zmid: 0, colorscale: [[0, this.colors.validation], [.5, this.colors.paper], [1, this.colors.train]],
      colorbar: {thickness: 9, len: .8, tickfont: {size: 10}, title: {text: 'r', side: 'top'}},
      hovertemplate: '%{x} × %{y}<br>Корреляция: %{z:.3f}<extra></extra>',
    }], {margin: {l: 95, r: 35, t: 40, b: 100}, xaxis: {...this.layout().xaxis, tickangle: -40}, yaxis: {...this.layout().yaxis, autorange: 'reversed'}, showlegend: false}, `Корреляция Пирсона: от −1 до 1.${names.length > 25 ? ' Показаны первые 25 признаков.' : ''} Постоянные признаки имеют неопределенную корреляцию.`);
  }

  async learningCurve() {
    const curve = this.result.diagnostics?.learning_curve;
    if (!curve?.points?.length) return this.empty('learning-curve', curve?.reason || 'Включи «Кривая обучения» в настройках диагностики и запусти модель.');
    const points = curve.points;
    const traces = [
      {key: 'train_mse', name: 'Ошибка обучения', color: this.colors.train},
      {key: 'validation_mse', name: 'Ошибка выбора настроек', color: this.colors.validation},
    ].map(({key, name, color}) => ({
      type: 'scatter', mode: 'lines+markers', name,
      x: points.map(point => point.train_size), y: points.map(point => point[key]),
      customdata: points.map(point => point.reason || ''), connectgaps: false,
      marker: {size: 7, color}, line: {width: 2.5, color},
      hovertemplate: 'Обучающих строк: %{x}<br>MSE: %{y:.5g}<extra>%{fullData.name}</extra>',
    }));
    if (!traces.some(trace => trace.y.some(isNumber))) return this.empty('learning-curve', 'Не удалось обучить модели на выбранных подвыборках. Причины доступны в результатах диагностики.');
    const note = curve.note || 'Для каждого размера обучена отдельная модель; проверочная часть одна и та же.';
    return this.draw('learning-curve', traces, {
      xaxis: {...this.layout().xaxis, title: 'Количество строк для обучения'},
      yaxis: {...this.layout().yaxis, title: 'MSE: средняя квадратичная ошибка'},
    }, note);
  }

  async importance() {
    const importance = this.result.diagnostics?.permutation_importance;
    if (!importance?.feature_names?.length || !vector(importance.mean).length) return this.empty('importance', importance?.reason || 'Включи «Перестановочная важность» в настройках диагностики и запусти модель.');
    const items = importance.feature_names.map((name, index) => ({name, mean: importance.mean[index], std: importance.std?.[index]}))
      .filter(item => isNumber(item.mean)).sort((a, b) => b.mean - a.mean);
    const shown = items.slice(0, 24).reverse();
    const metric = importance.metric || 'выбранной метрике';
    return this.draw('importance', [{
      type: 'bar', orientation: 'h', x: shown.map(item => item.mean), y: shown.map(item => shortName(item.name)),
      customdata: shown.map(item => [item.name, item.std]),
      marker: {color: shown.map(item => item.mean >= 0 ? this.colors.train : this.colors.actual)},
      error_x: {type: 'data', array: shown.map(item => isNumber(item.std) ? item.std : 0), visible: true, color: this.colors.muted, thickness: 1.2, width: 3},
      hovertemplate: '%{customdata[0]}<br>Падение качества: %{x:.5g}<br>Разброс повторов: %{customdata[1]:.4g}<extra></extra>',
    }], {
      showlegend: false, margin: {l: 115, r: 25, t: 40, b: 50},
      xaxis: {...this.layout().xaxis, title: `Падение качества · ${escape(metric)}`}, yaxis: {...this.layout().yaxis, type: 'category'},
    }, `${importance.note || 'Признак перемешивается в проверочной части; положительное значение означает ухудшение качества.'} Полосы: стандартное отклонение повторов.${items.length > 24 ? ' Показаны 24 наибольших значения.' : ''}`);
  }

  illustrativeSurface() {
    const result = this.result;
    if (!['linear', 'ols', 'linear_regression', 'ridge', 'lasso', 'elasticnet', 'elastic_net'].includes(result.model)) return null;
    const data = rawData(result);
    if (data.names.length < 2 || !data.X.length) return null;
    const {x, y} = this.selectedFeatures(data);
    const selected = sampledIndices(data.X.length, 400).filter(index => !data.split.length || data.split[index] === 'train');
    const points = selected.map(index => [data.X[index][x], data.X[index][y], data.y[index]]).filter(row => row.every(isNumber));
    if (points.length < 3) return null;
    const means = [0, 1, 2].map(column => average(points.map(row => row[column])));
    const centered = points.map(row => row.map((value, index) => value - means[index]));
    let a = 0, b = 0, c = 0, d = 0, e = 0, targetSquare = 0;
    for (const [u, v, target] of centered) { a += u*u; b += u*v; c += v*v; d += u*target; e += v*target; targetSquare += target*target; }
    const determinant = a*c-b*b;
    const slopes = Math.abs(determinant) > 1e-12 ? [(d*c-e*b)/determinant, (e*a-d*b)/determinant] : [a ? d/a : 0, c ? e/c : 0];
    const limits = slopes.map((slope, index) => Math.max(Math.abs(slope)*1.8, Math.sqrt(targetSquare / ((index ? c : a) || 1)), .5));
    const bx = range(-limits[0], limits[0], 37), by = range(-limits[1], limits[1], 37);
    const params = result.params || result.model_params || {};
    const alpha = Math.max(0, Number(params.alpha) || 0), ratio = Number(params.l1_ratio ?? .5);
    const z = by.map(v => bx.map(u => {
      const mse = (targetSquare + a*u*u + c*v*v + 2*b*u*v - 2*d*u - 2*e*v)/points.length;
      const penalty = result.model === 'ridge' ? u*u+v*v : result.model === 'lasso' ? Math.abs(u)+Math.abs(v) :
        ['elasticnet', 'elastic_net'].includes(result.model) ? ratio*(Math.abs(u)+Math.abs(v))+(1-ratio)*(u*u+v*v)/2 : 0;
      return mse/2+alpha*penalty;
    }));
    return {x: bx, y: by, z, coef_names: [data.names[x], data.names[y]], exact: false, note: 'Учебная задача: ½ MSE + αP для двух исходных признаков; свободный член подбирается. Это не цель обученной модели.'};
  }

  surfaceStates() {
    const surface = this.result.objective_surface;
    if (!surface || typeof surface.exact !== 'boolean') return [];
    if (surface.exact && ['lassolars', 'lasso_lars'].includes(this.result.model) && this.result.trace_kind === 'active_set') return [];
    const univariate = vector(this.result.coefficients).length === 1;
    return this.visibleTrace().map(frame => {
      const x = frame.coef?.[0], y = univariate ? frame.intercept : frame.coef?.[1];
      const z = surface.exact ? frame.objective : frame.train_loss;
      return [x, y, z].every(isNumber) ? {x, y, z, step: frame.step} : null;
    });
  }

  async surface() {
    const supplied = this.result.objective_surface;
    const surface = supplied || this.illustrativeSurface();
    if (!surface?.x?.length || !surface?.y?.length || !Array.isArray(surface.z?.[0])) return this.empty('surface', 'Поверхность цели недоступна для этой модели. История реальной ошибки остается на графике обучения.');
    const names = surface.coef_names || surface.feature_names || ['β₁', 'β₂'];
    const traces = [{
      type: 'surface', x: surface.x, y: surface.y, z: surface.z, colorscale: this.colors.surface,
      opacity: .95, contours: {z: {show: true, usecolormap: true, highlightcolor: this.colors.line, project: {z: true}}},
      colorbar: {thickness: 9, len: .7, tickfont: {size: 10}, title: {text: 'Цель', side: 'top'}},
      hovertemplate: `${escape(names[0])}: %{x:.4g}<br>${escape(names[1])}: %{y:.4g}<br>Значение: %{z:.5g}<extra></extra>`,
    }];
    const point = (value, label, color) => {
      const coordinates = Array.isArray(value) ? value : value ? [value.x, value.y, value.z] : [];
      if (coordinates.length === 3 && !isNumber(coordinates[2]) && isNumber(coordinates[0]) && isNumber(coordinates[1])) {
        const ix = surface.x.findIndex(x => Math.abs(x - coordinates[0]) <= 1e-8 * Math.max(Math.abs(x), 1));
        const iy = surface.y.findIndex(y => Math.abs(y - coordinates[1]) <= 1e-8 * Math.max(Math.abs(y), 1));
        if (ix >= 0 && iy >= 0) coordinates[2] = surface.z[iy]?.[ix];
      }
      if (coordinates.length === 3 && coordinates.every(isNumber)) traces.push({type: 'scatter3d', mode: 'markers', name: label, x: [coordinates[0]], y: [coordinates[1]], z: [coordinates[2]], marker: {size: 6, color, line: {color: this.colors.paper, width: 1}}, hovertemplate: `${label}<br>${escape(names[0])}: %{x:.4g}<br>${escape(names[1])}: %{y:.4g}<br>Значение: %{z:.5g}<extra></extra>`});
    };
    point(surface.minimum, 'Минимум рассчитанной сетки', this.colors.validation);
    point(surface.current, 'Итоговые коэффициенты', this.activeFrame ? this.colors.zero : this.colors.actual);
    const states = this.surfaceStates(), validStates = states.filter(Boolean);
    if (validStates.length > 1) traces.push({
      type: 'scatter3d', mode: this.result.trace_kind === 'candidates' ? 'markers' : 'lines+markers',
      name: this.result.trace_kind === 'candidates' ? 'Проверенные кандидаты' : 'Записанные состояния',
      x: states.map(state => state?.x ?? null), y: states.map(state => state?.y ?? null), z: states.map(state => state?.z ?? null),
      customdata: states.map(state => state?.step ?? null), connectgaps: false,
      line: {color: this.colors.train, width: 3}, marker: {color: this.colors.train, size: 3, opacity: .8},
      hovertemplate: `Шаг: %{customdata}<br>${escape(names[0])}: %{x:.4g}<br>${escape(names[1])}: %{y:.4g}<br>Записанное значение: %{z:.5g}<extra></extra>`,
    });
    if (this.activeFrame) {
      const state = states[states.length - 1];
      if (state && state.step === this.activeFrame.step) point([state.x, state.y, state.z], 'Текущий записанный шаг', this.colors.actual);
    }
    const axis = title => ({title: escape(title), gridcolor: this.colors.grid, zerolinecolor: this.colors.zero, showbackground: false, tickfont: {size: 10, color: this.colors.muted}});
    let note = [surface.label, surface.note].filter(Boolean).join('. ') || (supplied ? 'Сечение функции по двум преобразованным признакам. Остальные коэффициенты фиксированы.' : 'Учебная иллюстрация для двух признаков.');
    if (validStates.length > 1 || this.activeFrame && validStates.length) note += ' Кривая содержит записанные значения цели. Срез фиксирует остальные параметры на финале; кадры могут менять их, поэтому точки могут находиться вне поверхности.';
    return this.draw('surface', traces, {margin: {l: 0, r: 15, t: 48, b: 0}, scene: {xaxis: axis(names[0]), yaxis: axis(names[1]), zaxis: axis(supplied && surface.exact === false ? 'Диагностическая MSE' : supplied ? 'Функция оптимизации' : 'Учебная цель'), camera: {eye: {x: 1.6, y: 1.6, z: 1.25}}, aspectmode: 'auto'}, legend: {orientation: 'h', x: 0, y: 1, yanchor: 'bottom', font: {size: 10}}}, note);
  }

  async geometry() {
    const result = this.result;
    const supplied = result.penalty_geometry;
    if (supplied?.kind === 'profile') {
      if (!vector(supplied.x).length || supplied.x.length !== vector(supplied.y).length) return this.empty('geometry', 'Профиль штрафа недоступен: координаты не совпадают.');
      return this.draw('geometry', [{
        type: 'scatter', mode: 'lines', x: supplied.x, y: supplied.y,
        name: supplied.label || 'Штраф', line: {color: this.colors.validation, width: 3},
        fill: 'tozeroy', fillcolor: this.options.theme === 'light' ? 'rgba(121,83,197,.10)' : 'rgba(171,140,255,.10)',
        connectgaps: false, hovertemplate: 'Коэффициент β: %{x:.4g}<br>Штраф: %{y:.5g}<extra></extra>',
      }], {
        showlegend: false, xaxis: {...this.layout().xaxis, title: 'Первый коэффициент β₁'}, yaxis: {...this.layout().yaxis, title: 'Значение штрафа P(β)'},
      }, [supplied.label, supplied.note].filter(Boolean).join('. '));
    }
    const params = result.params || result.model_params || {};
    const rawKind = String(supplied?.kind || result.penalty || params.penalty || result.model || '').toLowerCase();
    const kind = ['l1', 'lasso', 'lassolars', 'lasso_lars', 'adaptive_lasso'].includes(rawKind) ? 'l1' : ['l2', 'ridge', 'bayesian_ridge', 'bayesianridge'].includes(rawKind) ? 'l2' : ['elasticnet', 'elastic_net'].includes(rawKind) ? 'elasticnet' : rawKind;
    let x = supplied?.x, y = supplied?.y;
    let note = supplied?.note || '';
    if (!x?.length || !y?.length) {
      if (!['l1', 'l2', 'elasticnet'].includes(kind)) return this.empty('geometry', 'У этой модели нет единого штрафа L₁/L₂ или его двумерная геометрия здесь не рассчитана.');
      const ratio = kind === 'l1' ? 1 : kind === 'l2' ? 0 : Number(supplied?.l1_ratio ?? params.l1_ratio ?? .5);
      const angles = range(0, Math.PI*2, 241);
      const radii = angles.map(angle => {
        if (kind === 'l2') return 1;
        const linear = ratio*(Math.abs(Math.cos(angle))+Math.abs(Math.sin(angle)));
        const quadratic = (1-ratio)/2;
        return quadratic < 1e-12 ? 1/linear : 2/(linear+Math.sqrt(linear*linear+4*quadratic));
      });
      x = angles.map((angle, index) => radii[index]*Math.cos(angle));
      y = angles.map((angle, index) => radii[index]*Math.sin(angle));
      note = kind === 'l1' ? 'Граница |β₁| + |β₂| = 1. Углы на осях объясняют появление точных нулей.' : kind === 'l2' ?
        'Граница β₁² + β₂² = 1. Коэффициенты уменьшаются плавно.' : `Граница r‖β‖₁ + (1−r)‖β‖₂²/2 = 1, r = ${ratio.toFixed(2)}.`;
      note += ' α задает вес штрафа; эта граница служит иллюстрацией.';
    }
    const interval = bounds([...x, ...y], .12);
    return this.draw('geometry', [{
      type: 'scatter', mode: 'lines', name: 'Граница штрафа', x, y, fill: 'toself',
      fillcolor: this.options.theme === 'light' ? 'rgba(18,140,114,.10)' : 'rgba(99,217,182,.10)',
      line: {color: this.colors.train, width: 3}, hovertemplate: 'β₁: %{x:.3f}<br>β₂: %{y:.3f}<extra></extra>',
    }], {showlegend: false, xaxis: {...this.layout().xaxis, title: 'Коэффициент β₁', range: interval, constrain: 'domain'}, yaxis: {...this.layout().yaxis, title: 'Коэффициент β₂', range: interval, scaleanchor: 'x', scaleratio: 1}}, note);
  }

  async frame(frame, result = this.result) {
    if (!frame || !result) return;
    this.result = result;
    this.activeFrame = frame;
    const trace = vector(result.trace);
    let index = trace.indexOf(frame);
    if (index < 0) index = trace.findIndex(item => item.step === frame.step);
    this.frameIndex = index < 0 ? null : index;
    const work = [this.loss(), this.coefficients()];
    if (frame.grid || frame.prediction_grid || frame.predicted || frame.predictions) work.push(this.fit(), this.threeD(), this.residuals(), this.prediction());
    if (this.isVisible('surface') && this.surfaceStates().some(Boolean)) work.push(this.surface());
    await Promise.allSettled(work);
  }

  destroy() {
    this.renderVersion += 1;
    this.resizeObserver?.disconnect();
    this.observed.clear();
    for (const [element, listeners] of this.listeners) {
      element.removeListener?.('plotly_click', listeners.click);
      element.removeListener?.('plotly_relayout', listeners.relayout);
    }
    this.listeners.clear();
    this.programmaticRelayout.clear();
    this.pointEdit = null;
    for (const name of IDS) {
      const element = document.getElementById(`plot-${name}`);
      if (element?.data && window.Plotly) window.Plotly.purge(element);
    }
    this.result = null;
  }
}
