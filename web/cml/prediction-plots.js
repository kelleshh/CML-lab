import { element } from './dom.js';

const colors = ['#803748', '#315e54', '#8d521d', '#63568b', '#54738b', '#a96549'];

/** Plot an actual model slice; all data is the numerical adapter's public DTO. */
export function renderPredictionView(workspace, view) {
  if (!view?.available) {
    if (view?.reason) workspace.container.append(element('p', { className: 'cml-note', text: view.reason }));
    return;
  }
  const [first, second] = view.features;
  const [x, y] = view.axes;
  const points = view.points || { coordinates: [], actual: [] };
  workspace.container.append(element('section', { className: 'cml-panel' }, [
    element('h3', { text: 'Срез обученной модели' }),
    element('p', { className: 'cml-note', text: view.description }),
    element('p', { className: 'cml-note', text: Object.entries(view.fixed_features || {}).length
      ? `Закрепленные признаки: ${Object.entries(view.fixed_features).map(([name, value]) => `${name} = ${typeof value === 'number' ? value.toPrecision(4) : value ?? 'пропуск'}`).join('; ')}`
      : 'Все входные признаки показаны на осях.' }),
  ]));
  if (view.task === 'regression') {
    if (second) {
      workspace.addPlot(`3D: ${first}, ${second} и числовой ответ`, [
        { type: 'surface', x, y, z: view.predictions, name: 'Предсказания', opacity: .75, colorscale: [[0, '#eee9dc'], [1, '#803748']], showscale: false },
        { type: 'scatter3d', mode: 'markers', x: points.coordinates[0], y: points.coordinates[1], z: points.actual, name: 'Настоящие ответы: обучение и выбор', marker: { color: '#315e54', size: 3, opacity: .7 } },
      ], { scene: { xaxis: { title: first }, yaxis: { title: second }, zaxis: { title: 'Числовой ответ' } }, margin: { t: 15, b: 15, l: 15, r: 15 } });
      const middle = Math.floor(y.length / 2);
      workspace.addPlot(`2D-срез при ${second} = ${y[middle].toPrecision(4)}`, [
        { type: 'scatter', mode: 'lines', x, y: view.predictions[middle], name: 'Предсказания среза', line: { color: '#803748', width: 3 } },
        { type: 'scatter', mode: 'markers', x: points.coordinates[0], y: points.actual, name: 'Настоящие ответы при разных значениях остальных признаков', marker: { color: '#315e54', size: 5, opacity: .6 } },
      ], { xaxis: { title: first }, yaxis: { title: 'Числовой ответ' } });
    } else {
      workspace.addPlot(`Предсказание по ${first}`, [
        { type: 'scatter', mode: 'lines', x, y: view.predictions, name: 'Срез модели', line: { color: '#803748', width: 3 } },
        { type: 'scatter', mode: 'markers', x: points.coordinates[0], y: points.actual, name: 'Настоящие ответы', marker: { color: '#315e54', size: 5 } },
      ], { xaxis: { title: first }, yaxis: { title: 'Числовой ответ' } });
    }
  } else if (view.task === 'classification') {
    const labels = view.classes || [];
    const traces = second ? [{ type: 'heatmap', x, y, z: view.predictions, opacity: .35, zmin: 0, zmax: Math.max(1, labels.length - 1), colorscale: labels.flatMap((_, i) => {
      const n = labels.length;
      return [[i / n, colors[i % colors.length]], [(i + 1) / n, colors[i % colors.length]]];
    }), colorbar: { title: 'Класс', tickvals: labels.map((_, i) => i), ticktext: labels } }]
      : [{ type: 'scatter', mode: 'lines', x, y: view.predictions, name: 'Предсказанный класс', line: { color: '#803748', shape: 'hv' } }];
    labels.forEach((label, classIndex) => {
      const indices = points.actual.map((value, index) => value === classIndex || value === label ? index : -1).filter(index => index >= 0);
      traces.push({ type: 'scatter', mode: 'markers', x: indices.map(index => points.coordinates[0][index]), y: indices.map(index => second ? points.coordinates[1][index] : classIndex), name: String(label), marker: { color: colors[classIndex % colors.length], size: 6, line: { width: 1, color: '#fffdf5' } } });
    });
    workspace.addPlot('Области предсказанных классов', traces, { xaxis: { title: first }, yaxis: second ? { title: second } : { title: 'Класс', tickvals: labels.map((_, i) => i), ticktext: labels } });
  }
}

