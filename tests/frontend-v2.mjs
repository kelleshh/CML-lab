/** End-to-end DOM flows through the actual v2 application modules.
 * HTTP responds with captured DataService / TrainingService / SearchService JSON;
 * Plotly rendering is replaced because jsdom has no WebGL canvas.
 */
import assert from 'node:assert/strict';
import {before, after, test} from 'node:test';
import {fixture, document, requests, plots, saved, $, event, posts, waitFor, clickAndWaitForJob,
  initializeApp, closeApp, getCurrentDataset} from './frontend-harness.mjs';

before(initializeApp);
after(closeApp);
const dw = key => document.querySelector(`[data-dw="${key}"]`);
const pw = key => document.querySelector(`[data-pw="${key}"]`);
const raw = key => document.querySelector(`[data-dw-plot="${key}"]`);
const button = selector => {
  const element = document.querySelector(selector);
  assert.ok(element, `Expected actionable element ${selector}`);
  return element;
};
const change = (element, value) => {
  assert.ok(element);
  if (element.type === 'checkbox') element.checked = value;
  else element.value = value;
  event(element);
};
const view = name => button(`[data-view="${name}"]`).click();

test('beginner penalty options write the exact parameter, advanced values survive and help opens a cited lesson', async () => {
  assert.equal($('parameter-mode').value, 'beginner');
  assert.equal($('model-params').hidden, true);
  const choices = [...$('model-presets').querySelectorAll('[data-preset-key="alpha"]')];
  assert.equal(choices.length, 4);
  const stronger = choices.find(item => item.textContent === 'Сильнее');
  assert.ok(stronger);
  stronger.click();
  assert.equal(stronger.getAttribute('aria-pressed'), 'true');
  let request = await clickAndWaitForJob($('run'));
  assert.equal(request.params.alpha, Number(stronger.dataset.presetValue));
  change($('parameter-mode'), 'advanced');
  assert.equal($('model-params').hidden, false);
  change($('model-params').querySelector('[data-param="alpha"]'), '2.7');
  change($('parameter-mode'), 'beginner');
  assert.match($('model-presets').textContent, /2,7.*своё значение/);
  request = await clickAndWaitForJob($('run'));
  assert.equal(request.params.alpha, 2.7);
  const question = $('model-presets').querySelector('.help-trigger');
  assert.ok(question, 'A novice needs help next to the actual penalty choice');
  question.focus();
  const panel = document.querySelector('.help-popover:not([hidden])');
  assert.ok(panel);
  assert.equal(question.getAttribute('aria-expanded'), 'true');
  assert.match(panel.textContent, /Ridge/);
  assert.match(panel.textContent, /α = n/);
  const source = [...panel.querySelectorAll('a')].find(link => link.textContent.includes('Первоисточник'));
  assert.match(source.href, /^https:\/\/scikit-learn\.org\//);
  panel.querySelector('a').click();
  assert.ok($('view-learn').classList.contains('active'));
  assert.match($('lesson-content').textContent, /Ridge/);
  assert.ok($('lesson-content').querySelector('.lesson-sources a[href^="https://"]'));
  view('lab');
});

test('dataset library shows task and size, and real target relationships render in 2D/3D before training', async () => {
  const before = posts('/api/jobs').length;
  view('data');
  await waitFor(() => document.querySelectorAll('.dw-dataset-card').length === 2, 'two saved datasets');
  const card = button(`[data-dw-select="${fixture.dataset.id}"]`).closest('article');
  assert.match(card.textContent, /Регрессия/);
  assert.match(card.textContent, /180/);
  assert.match(card.textContent, /Цель.*y/);
  card.querySelector('[data-dw-select]').click();
  await waitFor(() => raw('scatter2d')?.data?.length && raw('scatter3d')?.data?.length, 'raw plots');
  const captured2d = fixture.explorations[fixture.dataset.id + ':y'];
  const captured3d = fixture.explorations[fixture.dataset.id + ':x2'];
  assert.deepEqual(raw('scatter2d').data[0].x, captured2d.points.x);
  assert.deepEqual(raw('scatter2d').data[0].y, captured2d.points.y);
  assert.deepEqual(raw('scatter3d').data[0].x, captured3d.points.x);
  assert.deepEqual(raw('scatter3d').data[0].y, captured3d.points.y);
  assert.deepEqual(raw('scatter3d').data[0].z, captured3d.points.z);
  assert.deepEqual(raw('scatter2d').data[0].marker.color, captured2d.points.color);
  assert.equal(raw('scatter3d').data[0].type, 'scatter3d');
  assert.equal(dw('explore-y').classList.contains('dw-axis-target'), true);
  assert.equal(dw('explore-x').classList.contains('dw-axis-feature'), true);
  assert.notEqual(raw('scatter2d').layout.xaxis.title.font.color, raw('scatter2d').layout.yaxis.title.font.color);
  assert.match(document.querySelector('.dw-profile').textContent, /Медиана/);
  assert.equal(posts('/api/jobs').length, before, 'Selecting and exploring data must not train a model');
});

test('task filters distinguish regression from classification and categorical targets keep matching colours in 2D/3D', async () => {
  const before = posts('/api/jobs').length;
  change(dw('library-task'), 'classification');
  await waitFor(() => document.querySelectorAll('.dw-dataset-card').length === 1, 'classification library filter');
  assert.equal(button('.dw-dataset-card [data-dw-select]').dataset.dwSelect, fixture.classification.id);
  assert.ok(requests.some(item => item.path === '/api/datasets/library' && item.search.includes('task=classification')));
  button('.dw-dataset-card [data-dw-select]').click();
  await waitFor(() => raw('scatter2d')?.data?.length === 3 && raw('scatter3d')?.data?.length === 3, 'class colours');
  assert.equal(dw('explore-target').value, fixture.classification.task_target);
  for (const trace of raw('scatter2d').data) {
    const sameClass = raw('scatter3d').data.find(other => other.name === trace.name);
    assert.ok(sameClass);
    assert.equal(trace.marker.color, sameClass.marker.color);
    assert.ok(trace.x.length > 0);
    assert.ok(sameClass.z.length > 0);
  }
  button('[data-dw-tab="sources"]').click();
  change(dw('source-task'), 'regression');
  const sources = [...document.querySelectorAll('[data-dw-source]')];
  assert.ok(sources.length > 0);
  assert.ok(sources.every(item => fixture.catalogue.datasets.find(dataset => (dataset.id || dataset.name) === item.dataset.dwSource)?.tasks.includes('regression')));
  assert.ok(!sources.some(item => item.dataset.dwSource === 'load_iris'));
  assert.equal(posts('/api/jobs').length, before);
  change(dw('library-task'), '');
  button('[data-dw-tab="library"]').click();
  await waitFor(() => document.querySelector(`[data-dw-select="${fixture.dataset.id}"]`), 'regression card returns');
  button(`[data-dw-select="${fixture.dataset.id}"]`).click();
  await waitFor(() => getCurrentDataset().id === fixture.dataset.id && raw('scatter2d')?.data?.length === 1, 'regression selection restored');
});

test('dataset scenarios use bounded readable choices and send exact generator values without starting training', async () => {
  const before = posts('/api/jobs').length;
  button('[data-dw-tab="create"]').click();
  change(dw('scenario'), 'duplicates');
  assert.equal(dw('generator').value, 'correlated');
  assert.equal(dw('n-features').value, '4');
  assert.equal(dw('correlation-preset').value, '0.95');
  assert.match(document.querySelector('.dw-scenario-description').textContent, /Ridge/);
  change(dw('noise-preset'), '25');
  change(dw('outliers-preset'), '0.05');
  const count = posts('/api/datasets/load').length;
  button('[data-dw-action="create"]').click();
  await waitFor(() => posts('/api/datasets/load').length > count && getCurrentDataset().generator === 'correlated', 'created scenario');
  const request = posts('/api/datasets/load').at(-1).body;
  assert.equal(request.name, 'correlated');
  assert.equal(request.params.noise, 25);
  assert.equal(request.params.correlation, .95);
  assert.equal(request.params.outliers, .05);
  assert.equal(request.params.n_features, 4);
  assert.equal(posts('/api/jobs').length, before);
});

test('renaming and annotating a dataset persists library metadata while retaining its numerical training target', async () => {
  const before = requests.filter(item => item.method === 'PATCH').length;
  dw('metadata-name').value = 'Жильё для урока';
  dw('metadata-description').value = 'Площадь, этаж и цена';
  dw('metadata-tags').value = 'жильё, учебный пример';
  button('[data-dw-action="metadata"]').click();
  await waitFor(() => requests.filter(item => item.method === 'PATCH').length > before, 'metadata patch');
  const patched = requests.filter(item => item.method === 'PATCH').at(-1);
  assert.match(patched.path, /\/metadata$/);
  assert.deepEqual(patched.body.tags, ['жильё', 'учебный пример']);
  assert.equal(patched.body.name, 'Жильё для урока');
  assert.equal(patched.body.task_target, 'y');
  assert.equal(patched.body.default_target, 'y');
  await waitFor(() => document.querySelector('.dw-library-grid').textContent.includes('Жильё для урока'), 'updated saved card');
  assert.equal(getCurrentDataset().default_target, 'y');
  assert.notEqual(getCurrentDataset().id, fixture.dataset.id, 'Metadata updates preserve the previous dataset version');
});

test('feature engineering and regression sampling preview use real paired train rows then saved history restores all flags', async () => {
  view('lab');
  change(pw('preset'), 'robust');
  change(pw('degree'), '2');
  change(pw('interaction_only'), true);
  change(pw('missing_indicator'), true);
  change(pw('numeric_transform'), 'log1p');
  change(pw('selection'), 'mutual_info');
  change(pw('max_features'), '3');
  change(pw('method'), 'random_over');
  change(pw('focus'), 'high');
  button('[data-pw-action="preview"]').click();
  await waitFor(() => document.querySelector('.pw-preview').open && document.querySelector('[data-pw-plot="scatter"]')?.data, 'preprocessing preview');
  const request = posts(`/api/datasets/${getCurrentDataset().id}/preprocessing-preview`).at(-1).body;
  assert.equal(request.preprocessing.scaler, 'robust');
  assert.equal(request.preprocessing.numeric_transform, 'log1p');
  assert.equal(request.preprocessing.interaction_only, true);
  assert.equal(request.preprocessing.missing_indicator, true);
  assert.equal(request.preprocessing.max_features, 3);
  assert.equal(request.resampling.method, 'random_over');
  const real = fixture.preprocessingPreview;
  const trace = document.querySelector('[data-pw-plot="scatter"]').data[0];
  assert.deepEqual(trace.customdata.map(row => row[0]), real.indices);
  assert.deepEqual(trace.y, real.target.values);
  assert.deepEqual(trace.marker.color, real.target.values);
  assert.match(document.querySelector('.pw-preview-stats').textContent, /108/);
  assert.match(document.querySelector('.pw-preview-body').textContent, /только.*train/);
  button('[data-pw-action="close"]').click();
  const submitted = await clickAndWaitForJob($('run'));
  assert.deepEqual(submitted.preprocessing, request.preprocessing);
  assert.deepEqual(submitted.resampling, request.resampling);
  $('save').click();$('experiment-name').value = 'Prepared rows with rare targets';$('confirm-save').click();
  await waitFor(() => saved.size === 1, 'v2 experiment save');
  button('[data-pw-action="reset"]').click();
  assert.equal(pw('method').value, 'none');
  assert.equal(pw('interaction_only').checked, false);
  view('history');
  await waitFor(() => document.querySelector('[data-experiment]'), 'v2 history');
  button('[data-experiment]').click();
  await waitFor(() => $('view-lab').classList.contains('active') && pw('interaction_only').checked, 'all preparation flags restored');
  assert.equal(pw('scaler').value, 'robust');
  assert.equal(pw('degree').value, '2');
  assert.equal(pw('missing_indicator').checked, true);
  assert.equal(pw('selection').value, 'mutual_info');
  assert.equal(pw('max_features').value, '3');
  assert.equal(pw('method').value, 'random_over');
  assert.equal(pw('focus').value, 'high');
});

test('export options expose actual complete-pipeline format capabilities after training', () => {
  for (const item of fixture.exportCapabilities.formats) {
    const format = item.format === 'joblib' ? 'model' : item.format;
    const option = [...$('export').options].find(option => option.value === format);
    assert.ok(option, `An export choice is needed for ${format}`);
    assert.equal(option.disabled, !item.available);
    assert.match(option.textContent, new RegExp(item.label.split(' · ')[0]));
  }
  assert.match($('export-capabilities').textContent, /полный.*пайплайн/);
});

test('repeated cross-validation submits the chosen plan and renders the actual six fold scores and indices', async () => {
  button('[data-pw-action="reset"]').click();
  change($('vw-cv-method'), 'repeated_kfold');
  change($('vw-cv-folds'), '3');
  change($('vw-cv-repeats'), '2');
  change($('vw-parallelism'), '2');
  const request = await clickAndWaitForJob($('run'));
  assert.equal(request.preprocessing.degree, 1, 'Reset must clear a polynomial degree restored from an experiment');
  assert.deepEqual(request.cv_config, {strategy:'repeated_kfold',folds:3,repeats:2});
  assert.equal(request.n_jobs, 2);
  const cv = fixture.cvJob.result.cv;
  assert.equal($('vw-cv-fold-table').querySelectorAll('tbody tr').length, cv.folds.length);
  assert.deepEqual($('vw-cv-score-plot').data[0].y, cv.folds.map(fold => fold.metrics.rmse));
  assert.match($('cv-results').textContent, /6 \/ 6/);
  assert.match($('cv-results').textContent, /не доверительный интервал/);
  change($('vw-cv-result-metric'), 'mae');
  assert.deepEqual($('vw-cv-score-plot').data[0].y, cv.folds.map(fold => fold.metrics.mae));
  const details = document.querySelector('.vw-split-details');
  details.open = true;
  event(details, 'toggle');
  await waitFor(() => $('vw-cv-split-plot')?.data?.length, 'CV split map');
  const first = $('vw-cv-split-plot').data[0].z[0];
  for (const index of cv.folds[0].train_indices) assert.equal(first[index], 0);
  for (const index of cv.folds[0].validation_indices) assert.equal(first[index], 1);
  assert.equal($('vw-cv-method').options.length, fixture.catalogue.cv_strategies.length + 1);
});

test('real Optuna trials honour custom metric direction, expose exact scores and restore their search space from history', async () => {
  change($('vw-cv-method'), 'kfold');
  change($('vw-cv-folds'), '3');
  change($('vw-search-method'), 'optuna_tpe');
  change($('vw-search-trials'), '3');
  $('custom-metric').value = 'mean(abs(error))';
  change($('vw-search-metric'), 'custom');
  const row = button('[data-vw-param="alpha"]');
  change(row.querySelector('[data-vw-enabled]'), true);
  change(row.querySelector('[data-vw-space-kind]'), 'range');
  change(row.querySelector('[data-vw-low]'), '.1');
  change(row.querySelector('[data-vw-high]'), '10');
  change(row.querySelector('[data-vw-log]'), true);
  for (const direction of ['min', 'max']) {
    change($('vw-search-direction'), direction);
    assert.equal($('ranking-metric').value, 'custom');
    assert.equal($('ranking-direction').value, direction);
    const request = await clickAndWaitForJob($('vw-search-start'));
    assert.equal(request.custom_metric, 'mean(abs(error))');
    assert.equal(request.search.method, 'optuna_tpe');
    assert.equal(request.search.direction, direction);
    assert.equal(request.search.metric, 'custom');
    assert.equal(request.search.trials, 3);
    assert.deepEqual(request.search.param_space, {alpha:{type:'float',low:.1,high:10,log:true}});
    const real = fixture.searchJobs[direction].result.search;
    const scores = real.trials.map(trial => trial.score);
    assert.deepEqual($('vw-search-curve').data[0].y, scores);
    assert.deepEqual($('vw-search-curve').data[1].y, real.trials.map(trial => trial.best_score));
    assert.equal($('vw-search-trial-table').querySelectorAll('tbody tr').length, 3);
    assert.equal(Number($('vw-search-trial-table').querySelector('.vw-best-row td').textContent), real.best_trial);
    assert.equal(Number($('model-params').querySelector('[data-param="alpha"]').value), real.best_params.alpha);
    assert.equal(real.best_score, direction === 'min' ? Math.min(...scores) : Math.max(...scores));
    assert.match($('vw-search-result-note').textContent, /train/);
    assert.match($('vw-search-curve-note').textContent, /реальная комбинация/);
  }
  const count = saved.size;
  $('save').click();$('experiment-name').value = 'Optuna actual trials';$('confirm-save').click();
  await waitFor(() => saved.size === count + 1, 'search save');
  change($('vw-search-method'), 'random');
  change($('vw-search-trials'), '20');
  change($('vw-cv-method'), 'none');
  change($('vw-parallelism'), '1');
  view('history');
  const experiment = [...saved.values()].find(item => item.name === 'Optuna actual trials');
  await waitFor(() => document.querySelector(`[data-experiment="${experiment.id}"]`), 'saved search card');
  button(`[data-experiment="${experiment.id}"]`).click();
  await waitFor(() => $('view-lab').classList.contains('active') && $('vw-search-method').value === 'optuna_tpe', 'search restored');
  assert.equal($('vw-search-trials').value, '3');
  assert.equal($('vw-search-metric').value, 'custom');
  assert.equal($('vw-search-direction').value, 'max');
  assert.equal($('vw-cv-method').value, 'kfold');
  assert.equal($('vw-parallelism').value, '2');
  const restored = button('[data-vw-param="alpha"]');
  assert.equal(restored.querySelector('[data-vw-enabled]').checked, true);
  assert.equal(restored.querySelector('[data-vw-space-kind]').value, 'range');
  assert.equal(Number(restored.querySelector('[data-vw-low]').value), .1);
  assert.equal(Number(restored.querySelector('[data-vw-high]').value), 10);
  assert.equal(restored.querySelector('[data-vw-log]').checked, true);
});
