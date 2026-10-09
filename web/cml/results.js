import { element, formatNumber } from './dom.js';
import { action, heading, notice, select } from './controls.js';
import { TracePlayer } from './training.js';
import { renderPredictionView } from './prediction-plots.js';
import { addDiagnosticPlots } from './diagnostics-plots.js';

const TASK_LABELS = { regression: 'Регрессия', classification: 'Классификация', clustering: 'Кластеризация', ranking: 'Ранжирование', forecasting: 'Прогноз по времени', panel: 'Панельные данные', anomaly: 'Поиск аномалий', reduction: 'Сокращение размерности' };

export class ResultsWorkspace {
  constructor({ api, onReveal, onSave, onError }) { this.api = api; this.onReveal = onReveal; this.onSave = onSave; this.onError = onError; this.result = null; }
  mount(container) { this.container = container; this.empty(); return this; }
  empty() { this.tracePlayer?.destroy(); if (this.container) this.container.replaceChildren(element('section', { className: 'cml-panel' }, [heading('Результаты расчета', { help: 'Графики и метрики появятся после реального обучения.', lesson_id: '20-metrics-experiment' }), element('p', { className: 'cml-empty', text: 'Выберите данные, задачу и алгоритм. Затем нажмите «Запустить расчет».' })])); }

  async setResult(result, runId) { this.result = result; this.runId = runId; await this.render(); }

  layout(extra = {}) {
    const style = getComputedStyle(document.body);
    return { paper_bgcolor: style.getPropertyValue('--panel').trim(), plot_bgcolor: style.getPropertyValue('--panel').trim(), font: { color: style.getPropertyValue('--text').trim(), family: 'Arial, sans-serif', size: 13 }, margin: { t: 25, b: 60, l: 65, r: 25 }, xaxis: { gridcolor: style.getPropertyValue('--line').trim() }, yaxis: { gridcolor: style.getPropertyValue('--line').trim() }, ...extra };
  }

  async plot(container, traces, layout = {}) {
    if (globalThis.Plotly) await Plotly.react(container, traces, this.layout(layout), { responsive: true, displaylogo: false });
    else container.textContent = 'Графическая библиотека еще загружается.';
  }

  addPlot(title, traces, layout = {}) {
    const plot = element('div', { className: 'cml-plot' });
    this.plots.append(element('article', { className: 'cml-plot-panel' }, [element('h3', { text: title }), plot]));
    if (traces.length) this.plotJobs.push(this.plot(plot, traces, layout));
    return plot;
  }

  async render() {
    const result = this.result;
    if (!result) return this.empty();
    this.tracePlayer?.destroy();
    const evaluations = result.evaluations || {};
    const testHidden = result.test_hidden === true || evaluations.test?.hidden === true || !evaluations.test;
    const validation = evaluations.validation || evaluations.train || {};
    const metrics = validation.metrics || {};
    const cardMetrics = Object.entries(metrics).filter(([, value]) => typeof value === 'number' && Number.isFinite(value)).slice(0, 4);
    const exportSelect = select([{ value: '', label: 'Экспорт результата или модели…' }, ...['json', 'csv', 'joblib', 'pickle', 'skops', 'onnx', 'bundle', 'passport'].map(format => ({ value: format, label: { json: 'Результат JSON', csv: 'Предсказания CSV', joblib: 'Модель Joblib', pickle: 'Модель Pickle', skops: 'Модель Skops', onnx: 'Модель ONNX', bundle: 'Конвейер и паспорт ZIP', passport: 'Паспорт JSON' }[format] }))], '', { 'aria-label': 'Формат экспорта' });
    exportSelect.addEventListener('change', async () => {
      if (!exportSelect.value) return;
      const format = exportSelect.value; exportSelect.value = '';
      try { await this.api.download(`/cml/runs/${encodeURIComponent(this.runId)}/export?format=${format}`, `cml-lab-${this.runId}.${format}`); } catch (error) { this.onError(error); }
    });
    this.plots = element('div', { className: 'cml-plots' });
    this.plotJobs = [];
    const metricTable = this.metricsTable(evaluations);
    const rows = (validation.rows || []).map((row, index) => ({ ...row, ...(validation.classification?.probabilities?.[index] ? { probabilities: validation.classification.probabilities[index] } : {}) }));
    const diagnostics = result.diagnostics || {};
    const trace = Array.isArray(result.trace) ? result.trace : result.trace?.frames || result.trace?.history || [];
    const actions = element('div', { className: 'cml-toolbar' }, [
      action('Сохранить эксперимент', () => this.onSave()), exportSelect,
      ...(testHidden && !['clustering', 'anomaly', 'reduction'].includes(result.task) ? [action('Открыть итоговую проверку', () => this.onReveal())] : []),
    ]);
    this.container.replaceChildren(element('div', { className: 'cml-page-title' }, [element('div', {}, [element('h2', { text: `${TASK_LABELS[result.task] || result.task} · ${result.model_name || result.algorithm_id}` }), element('p', { className: 'cml-note', text: `Показатели по ${evaluations.validation ? 'проверочной' : 'обучающей'} части. ${testHidden ? 'Итоговая проверка скрыта.' : 'Итоговая проверка открыта.'}` })]), actions]),
      element('div', { className: 'cml-status-strip' }, cardMetrics.map(([key, value]) => element('div', { className: 'cml-stat' }, [element('span', { text: key }), element('strong', { text: formatNumber(value) })]))),
      this.plots,
      element('section', { className: 'cml-panel' }, [heading('Все метрики', { help: 'Сравнивайте обучение и проверку. Высокое качество на обучении при низком качестве на проверке может означать переобучение.', lesson_id: '20-metrics-experiment' }), metricTable]),
      element('section', { className: 'cml-panel' }, [heading('Строки результата', { help: 'Это настоящие ответы и прогнозы для строк выбранной проверочной части. Индексы относятся к исходному датасету.', lesson_id: '20-metrics-experiment' }), this.rowsTable(rows)]),
      element('details', { className: 'cml-panel' }, [element('summary', { text: 'Настройки и снимок запуска' }), element('pre', { className: 'cml-code', text: JSON.stringify(result.effective_spec || {}, null, 2) })]),
      ...(result.warnings || []).map(warning => notice(typeof warning === 'string' ? warning : warning.message || JSON.stringify(warning))),
      ...(diagnostics.scope ? [notice(diagnostics.scope)] : []),
    );
    const numericRows = rows.filter(row => Number.isFinite(Number(row.actual)) && Number.isFinite(Number(row.predicted)) && row.actual !== null && row.predicted !== null);
    if (['regression', 'forecasting', 'panel'].includes(result.task) && numericRows.length) {
      this.addPlot('Настоящий ответ и прогноз', [{ type: 'scatter', mode: 'markers', x: numericRows.map(row => row.actual), y: numericRows.map(row => row.predicted), marker: { color: '#315e54', size: 6, opacity: .7 }, name: 'Наблюдения' }], { xaxis: { title: 'Настоящий ответ' }, yaxis: { title: 'Прогноз' } });
      this.addPlot('Ошибки прогноза', [{ type: 'scatter', mode: 'markers', x: numericRows.map(row => row.predicted), y: numericRows.map(row => row.actual - row.predicted), marker: { color: '#8d521d', size: 6 }, name: 'Остаток' }], { xaxis: { title: 'Прогноз' }, yaxis: { title: 'Настоящий ответ − прогноз' } });
    }
    if (result.task === 'classification') {
      const confusion = validation.classification ? { matrix: validation.classification.confusion, labels: validation.classification.classes } : diagnostics.confusion_matrix || validation.plots?.confusion_matrix;
      if (confusion) {
        const matrix = Array.isArray(confusion) ? confusion : confusion.matrix || confusion.values || [];
        const labels = confusion.labels || diagnostics.classes || [...new Set(rows.map(row => row.actual))];
        this.addPlot('Матрица ошибок: настоящий класс и прогноз', [{ type: 'heatmap', z: matrix, x: labels, y: labels, colorscale: [[0, '#eee9dc'], [1, '#315e54']], showscale: true, hovertemplate: 'Настоящий: %{y}<br>Прогноз: %{x}<br>Строк: %{z}<extra></extra>' }], { xaxis: { title: 'Предсказанный класс' }, yaxis: { title: 'Настоящий класс' } });
      } else if (rows.length) this.addPlot('Правильные и ошибочные ответы', [{ type: 'bar', x: ['Верно', 'Ошибка'], y: [rows.filter(row => row.actual === row.predicted).length, rows.filter(row => row.actual !== row.predicted).length], marker: { color: ['#37624c', '#8d521d'] } }]);
      const roc = validation.classification?.roc || diagnostics.roc || validation.plots?.roc;
      if (roc?.fpr && roc?.tpr) this.addPlot('ROC: чувствительность к порогу', [{ type: 'scatter', mode: 'lines', x: roc.fpr, y: roc.tpr, line: { color: '#315e54' }, name: `AUC ${formatNumber(roc.auc)}` }], { xaxis: { title: 'Доля ложных срабатываний' }, yaxis: { title: 'Доля найденных положительных ответов' } });
    }
    if (['forecasting', 'panel'].includes(result.task) && rows.length) {
      const groups = result.task === 'panel' ? [...new Set(rows.map(row => row.entity ?? 'объект'))].slice(0, 8) : ['все'];
      const traces = groups.flatMap(group => {
        const chosen = result.task === 'panel' ? rows.filter(row => (row.entity ?? 'объект') === group) : rows;
        return [{ type: 'scatter', mode: 'lines+markers', x: chosen.map(row => row.time ?? row.index), y: chosen.map(row => row.actual), name: `${group}: факт`, line: { color: '#8d521d', width: 1.5 } }, { type: 'scatter', mode: 'lines+markers', x: chosen.map(row => row.time ?? row.index), y: chosen.map(row => row.predicted), name: `${group}: прогноз`, line: { color: '#315e54', width: 2 } }];
      });
      this.addPlot('Факт и прогноз по времени', traces, { xaxis: { title: 'Время' }, yaxis: { title: 'Целевая величина' } });
    }
    if (['clustering', 'anomaly', 'reduction'].includes(result.task)) {
      const projection = diagnostics.projection || validation.plots?.projection || diagnostics.embedding;
      if (projection) {
        const coordinates = projection.points || projection.coordinates || projection;
        const labels = projection.labels || coordinates.map(() => 0);
        if (Array.isArray(coordinates) && Array.isArray(coordinates[0])) this.addPlot(result.task === 'reduction' ? 'Новые координаты признаков' : 'Найденные группы и необычные наблюдения', [{ type: 'scatter', mode: 'markers', x: coordinates.map(point => point[0]), y: coordinates.map(point => point[1] ?? 0), text: labels.map(String), marker: { color: labels.every(label => typeof label === 'number') ? labels : '#315e54', colorscale: [[0, '#315e54'], [1, '#8d521d']], size: 7, opacity: .8 } }], { xaxis: { title: projection.feature_names?.[0] || 'Координата 1' }, yaxis: { title: projection.feature_names?.[1] || 'Координата 2' } });
      }
    }
    if (result.task === 'ranking' && rows.length) {
      const query = [...new Set(rows.map(row => row.query ?? row.query_group ?? 'запрос'))][0];
      const chosen = rows.filter(row => (row.query ?? row.query_group ?? 'запрос') === query).sort((a, b) => b.predicted - a.predicted).slice(0, 30);
      this.addPlot(`Порядок объектов: ${query}`, [{ type: 'bar', x: chosen.map(row => String(row.index)), y: chosen.map(row => row.actual), name: 'Настоящая полезность', marker: { color: '#8d521d' } }, { type: 'scatter', mode: 'lines+markers', x: chosen.map(row => String(row.index)), y: chosen.map(row => row.predicted), name: 'Оценка модели', line: { color: '#315e54' }, yaxis: 'y2' }], { xaxis: { title: 'Строки в порядке модели' }, yaxis: { title: 'Полезность' }, yaxis2: { title: 'Оценка модели', overlaying: 'y', side: 'right' } });
    }
    if (trace.length) {
      const keys = ['train_loss', 'validation_loss', 'loss', 'objective', 'metric'].filter(key => trace.some(frame => typeof frame[key] === 'number'));
      if (keys.length) {
        const plot = this.addPlot('Измерения во время обучения', [], { xaxis: { title: 'Реальный шаг / этап' }, yaxis: { title: 'Измеренное значение' } });
        const controls = element('div', { className: 'cml-trace-replay' });
        plot.parentElement.append(controls);
        this.tracePlayer = new TracePlayer({ trace, plot, container: controls, featureNames: diagnostics.preprocessing?.features || [] });
      }
    }
    if (diagnostics.cv?.folds?.length) {
      const reports = diagnostics.cv.folds;
      const keys = [...new Set(reports.flatMap(report => Object.keys(report.metrics || {})))];
      this.container.append(element('section', { className: 'cml-panel' }, [heading('Перекрестная проверка', { help: 'Каждая строка означает отдельное обучение на одной части и проверку на другой. Среднее показывает типичное качество, разброс — чувствительность к разделению.', lesson_id: '26-cross-validation' }), element('div', { className: 'cml-table-scroll' }, [element('table', { className: 'cml-table' }, [element('thead', {}, [element('tr', {}, ['Часть', 'Обучение, строк', 'Выбор, строк', ...keys].map(text => element('th', { text })))]), element('tbody', {}, reports.map(report => element('tr', {}, [report.fold, report.train_rows, report.validation_rows, ...keys.map(key => formatNumber(report.metrics[key]))].map(value => element('td', { text: String(value) }))))), element('tfoot', {}, [element('tr', {}, [element('th', { text: 'Среднее ± разброс', colspan: 3 }), ...keys.map(key => element('td', { text: `${formatNumber(diagnostics.cv.mean?.[key])} ± ${formatNumber(diagnostics.cv.std?.[key])}` }))])])])]) ]));
    }
    const importance = diagnostics.feature_importance || diagnostics.importance || diagnostics.coefficients;
    if (importance) {
      const names = importance.names || importance.features || [];
      const values = importance.values || importance.importances || importance.coefficients || [];
      if (names.length && values.length) {
        const classes = validation.classification?.classes || [];
        const traces = Array.isArray(values[0])
          ? values.map((row, index) => ({ type: 'bar', x: row, y: names, orientation: 'h', name: String(classes[values.length === 1 && classes.length === 2 ? 1 : index] || `Строка коэффициентов ${index + 1}`) }))
          : [{ type: 'bar', x: values, y: names, orientation: 'h', marker: { color: '#315e54' } }];
        this.addPlot(diagnostics.coefficients === importance ? 'Коэффициенты модели' : 'Важность признаков', traces, { margin: { l: 160, r: 25, t: 25, b: 60 } });
      }
    }
    renderPredictionView(this, diagnostics.prediction_view);
    addDiagnosticPlots(this, diagnostics);
    if (!this.plots.children.length) this.plots.append(notice('Этот алгоритм вернул метрики и строки результата. График истории обучения доступен только при наличии измеренных промежуточных шагов.'));
    await Promise.all(this.plotJobs);
    await this.updateExportOptions(exportSelect);
  }

  metricsTable(evaluations) {
    const keys = [...new Set(Object.values(evaluations).flatMap(evaluation => Object.keys(evaluation.metrics || {})))];
    const details = Object.entries(evaluations).flatMap(([part, evaluation]) => Object.entries(evaluation.metric_details || {}).filter(([, item]) => item?.reason || item?.message).map(([key, item]) => notice(`${part}: ${key} — ${item.reason || item.message}`)));
    return element('div', { className: 'cml-table-scroll' }, [element('table', { className: 'cml-table' }, [
      element('thead', {}, [element('tr', {}, ['Метрика', 'Обучение', 'Выбор', 'Итоговая проверка'].map(text => element('th', { text })))]),
      element('tbody', {}, keys.map(key => element('tr', {}, [element('td', { text: key }), ...['train', 'validation', 'test'].map(part => element('td', { text: evaluations[part]?.hidden === true || (!evaluations[part] && part === 'test') ? 'Скрыто' : evaluations[part] ? formatNumber(evaluations[part].metrics?.[key]) : '—' }))]))),
    ]), ...details]);
  }

  rowsTable(rows) {
    if (!rows.length) return element('p', { className: 'cml-empty', text: 'Строки прогноза недоступны для этой части.' });
    const columns = ['index', 'actual', 'predicted', ...['time', 'entity', 'query', 'query_group', 'score', 'probabilities'].filter(key => rows.some(row => row[key] !== undefined))];
    const labels = { index: 'Строка', actual: 'Настоящий ответ', predicted: 'Прогноз / группа', time: 'Время', entity: 'Объект', query: 'Запрос', query_group: 'Запрос', score: 'Оценка', probabilities: 'Вероятности' };
    return element('div', { className: 'cml-table-scroll' }, [element('table', { className: 'cml-table' }, [
      element('thead', {}, [element('tr', {}, columns.map(key => element('th', { text: labels[key] || key })))]),
      element('tbody', {}, rows.slice(0, 100).map(row => element('tr', {}, columns.map(key => element('td', { text: Array.isArray(row[key]) ? row[key].map(formatNumber).join('; ') : typeof row[key] === 'number' ? formatNumber(row[key]) : String(row[key] ?? '—') }))))),
    ]), element('p', { className: 'cml-note', text: `В таблице первые ${Math.min(100, rows.length)} из ${rows.length} строк. Полные прогнозы доступны в экспорте CSV.` })]);
  }

  async updateExportOptions(control) {
    try {
      const payload = await this.api.request(`/cml/runs/${encodeURIComponent(this.runId)}/export-capabilities`);
      const formats = Array.isArray(payload) ? payload : payload.formats || [];
      for (const item of formats) {
        const option = [...control.options].find(option => option.value === item.format || option.value === item.id);
        if (option) { option.disabled = item.available === false; option.title = item.reason || ''; }
      }
      const unavailable = formats.filter(item => item.available === false);
      if (unavailable.length) this.container.append(element('details', { className: 'cml-panel' }, [element('summary', { text: 'Совместимость форматов экспорта' }), ...unavailable.map(item => element('p', { className: 'cml-note', text: `${item.label || item.format}: ${item.reason}` }))]));
    } catch (error) { this.container.append(notice(`Список форматов не получен: ${error.message}`)); }
  }
}
