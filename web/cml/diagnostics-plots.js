import { element } from './dom.js';
import { notice } from './controls.js';

/** Render measured diagnostic reports without treating separate fits as training frames. */
export function addDiagnosticPlots(workspace, diagnostics = {}) {
  const style = getComputedStyle(document.body);
  const feature = style.getPropertyValue('--feature').trim() || '#315e54';
  const target = style.getPropertyValue('--target').trim() || '#8d521d';
  const messages = [];
  const report = (key, title) => {
    const value = diagnostics[key];
    if (!value) return null;
    if (value.reason) { messages.push(`${title}: ${value.reason}`); return null; }
    if (value.note) messages.push(`${title}: ${value.note}`);
    return value;
  };
  const path = report('regularization_path', 'Сила штрафа');
  if (path) {
    const points = (path.points || []).filter(point => Number.isFinite(point.alpha));
    const traces = (path.feature_names || []).map((name, column) => ({ type: 'scatter', mode: 'lines+markers', name, x: path.alphas || [], y: (path.coefficients || []).map(coef => Array.isArray(coef?.[0]) ? null : coef?.[column] ?? null), connectgaps: false }));
    if (traces.some(trace => trace.y.some(Number.isFinite))) workspace.addPlot('Коэффициенты при разных силах штрафа', traces, { xaxis: { title: 'Сила штрафа α', type: (path.alphas || []).every(alpha => alpha > 0) ? 'log' : 'linear' }, yaxis: { title: 'Коэффициент' } });
    if (points.length) workspace.addPlot('Качество при разных силах штрафа', [
      { type: 'scatter', mode: 'lines+markers', name: 'Обучение', x: points.map(point => point.alpha), y: points.map(point => point.train_score), line: { color: feature } },
      { type: 'scatter', mode: 'lines+markers', name: 'Выбор настроек', x: points.map(point => point.alpha), y: points.map(point => point.validation_score), line: { color: target } },
    ], { xaxis: { title: 'Сила штрафа α', type: points.every(point => point.alpha > 0) ? 'log' : 'linear' }, yaxis: { title: `${path.metric || 'Метрика'}; ${path.direction === 'max' ? 'больше лучше' : 'меньше лучше'}` } });
    if ((path.coefficients || []).some(coef => Array.isArray(coef?.[0]))) messages.push('Коэффициенты нескольких классов не сведены в одну линию. Качество разных сил штрафа показано отдельно.');
    points.filter(point => point.reason).forEach(point => messages.push(`α=${point.alpha}: ${point.reason}`));
  }
  const curve = report('learning_curve', 'Количество обучающих строк');
  if (curve?.points?.length) {
    const points = curve.points;
    workspace.addPlot('Качество при разном количестве обучающих строк', [
      { type: 'scatter', mode: 'lines+markers', name: 'Обучение', x: points.map(point => point.train_size), y: points.map(point => point.train_score), line: { color: feature } },
      { type: 'scatter', mode: 'lines+markers', name: 'Выбор настроек', x: points.map(point => point.train_size), y: points.map(point => point.validation_score), line: { color: target } },
    ], { xaxis: { title: 'Обучающих строк' }, yaxis: { title: `${curve.metric || 'Метрика'}; ${curve.direction === 'max' ? 'больше лучше' : 'меньше лучше'}` } });
    points.filter(point => point.reason).forEach(point => messages.push(`${point.train_size} строк: ${point.reason}`));
  }
  const importance = report('permutation_importance', 'Важность через перемешивание');
  if (importance?.feature_names?.length) workspace.addPlot('Изменение качества после перемешивания признака', [{ type: 'bar', orientation: 'h', y: importance.feature_names, x: importance.mean || [], error_x: { type: 'data', array: importance.std || [], visible: true }, marker: { color: feature }, hovertemplate: '%{y}<br>Изменение качества: %{x}<extra></extra>' }], { margin: { l: 160, r: 25, t: 25, b: 75 }, xaxis: { title: `${importance.metric || 'Метрика'}: падение качества; больше означает важнее` } });
  if (messages.length) workspace.container.append(element('section', { className: 'cml-panel' }, [element('h3', { text: 'Как читать дополнительные исследования' }), ...messages.map(message => notice(message))]));
}
