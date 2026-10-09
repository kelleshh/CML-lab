/** Verify Plotly input against actual numerical fixtures; pixels need a browser. */
import assert from 'node:assert/strict';
import {test} from 'node:test';
import {readFileSync} from 'node:fs';
import {createRequire} from 'node:module';
const require = createRequire(import.meta.url);
let JSDOM;
try { ({JSDOM} = require('jsdom')); }
catch { ({JSDOM} = require(`${process.env.CODEX_PRIMARY_RUNTIME_NODE_MODULES}/jsdom`)); }
const {window} = new JSDOM('<!doctype html><body></body>');
Object.assign(globalThis, {document: window.document, getComputedStyle: window.getComputedStyle.bind(window)});
const {renderPredictionView} = await import('../web/cml/prediction-plots.js');
const {addDiagnosticPlots} = await import('../web/cml/diagnostics-plots.js');
const fixture = JSON.parse(readFileSync(new URL('./fixtures/cml-ui.json', import.meta.url), 'utf8'));
function workspace() {
  const plots=[];
  return {plots, container:document.createElement('div'), addPlot:(title,traces,layout)=>plots.push({title,traces,layout})};
}

test('regression surface and point heights are the actual visible numerical slice', () => {
  const view=fixture.results.regression.hidden.diagnostics.prediction_view;
  const subject=workspace();renderPredictionView(subject,view);
  const surface=subject.plots.flatMap(plot=>plot.traces).find(trace=>trace.type==='surface');
  assert.ok(surface);
  assert.deepEqual(surface.z,view.predictions);
  assert.equal(surface.z.length,view.axes[1].length);
  assert.equal(surface.z[0].length,view.axes[0].length);
  const scatter=subject.plots.flatMap(plot=>plot.traces).find(trace=>trace.type==='scatter3d');
  assert.deepEqual(scatter.z,view.points.actual);
  assert.ok(view.points.split.every(split=>['train','validation'].includes(split)));
  assert.match(subject.container.textContent,/медиан|частых/);
});

test('classification decision regions and class labels correspond to the saved classifier', () => {
  const view=fixture.results.classification.hidden.diagnostics.prediction_view;
  const subject=workspace();renderPredictionView(subject,view);
  const region=subject.plots[0].traces.find(trace=>trace.type==='heatmap');
  assert.deepEqual(region.z,view.predictions);
  assert.deepEqual(region.colorbar.ticktext,view.classes);
  assert.ok(region.z.flat().every(value=>Number.isInteger(value)&&value>=0&&value<view.classes.length));
  const classes=subject.plots[0].traces.filter(trace=>trace.type==='scatter');
  assert.deepEqual(classes.map(trace=>trace.name),view.classes);
  assert.equal(classes.reduce((total,trace)=>total+trace.x.length,0),view.points.actual.length);
});

test('measured regularization, sample-size and permutation arrays reach their exact charts', () => {
  const diagnostics=fixture.diagnostic_regression.diagnostics;
  const subject=workspace();addDiagnosticPlots(subject,diagnostics);
  assert.equal(subject.plots.length,4);
  const path=subject.plots.find(plot=>plot.title.startsWith('Качество при разных силах'));
  assert.deepEqual(path.traces[1].y,diagnostics.regularization_path.points.map(point=>point.validation_score));
  assert.match(path.layout.yaxis.title,/меньше лучше/);
  const curve=subject.plots.find(plot=>plot.title.includes('количестве'));
  assert.deepEqual(curve.traces[0].x,diagnostics.learning_curve.points.map(point=>point.train_size));
  const importance=subject.plots.find(plot=>plot.title.includes('перемешивания'));
  assert.deepEqual(importance.traces[0].x,diagnostics.permutation_importance.mean);
  assert.deepEqual(importance.traces[0].error_x.array,diagnostics.permutation_importance.std);
  assert.match(subject.container.textContent,/не.*шаг|разны|отдельн/);
});
