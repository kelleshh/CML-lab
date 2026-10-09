/** User journeys through real UI modules and captured service responses.
 * JSDOM verifies routing and actions; it cannot verify layout or WebGL pixels.
 */
import assert from 'node:assert/strict';
import {before, after, test} from 'node:test';
import {HELP_REGISTRY, MODEL_LESSONS} from '../web/help.js';
import {fixture, document, window, jobs, $, event, posts, waitFor,
  initializeApp, closeApp} from './frontend-harness.mjs';

before(initializeApp);
after(closeApp);
const lessons = new Map(fixture.catalogue.lessons.map(lesson => [lesson.id, lesson]));
const view = name => document.querySelector(`[data-view="${name}"]`).click();
const change = (field, value) => {field.value = value; event(field);};

function assertOpenedLesson(id) {
  assert.ok(lessons.has(id), `The link must refer to a published lesson: ${id}`);
  assert.equal($('view-learn').classList.contains('active'), true);
  assert.equal($('lesson-content').querySelector('h2').textContent, lessons.get(id).title,
    `Instruction navigation must open ${id}, without silently falling back to lesson 01`);
  assert.ok($('lesson-content').querySelector('.lesson-sources a[href^="https://"]'));
}

function openContextHelp(fieldId, expectedLesson) {
  const trigger = document.querySelector(`.help-trigger[data-help-for="${fieldId}"]`);
  assert.ok(trigger, `The active control needs contextual help: ${fieldId}`);
  for (let ancestor = trigger.parentElement; ancestor; ancestor = ancestor.parentElement) {
    if (ancestor.tagName === 'DETAILS') ancestor.open = true;
  }
  trigger.click();
  const panel = document.getElementById(trigger.getAttribute('aria-controls'));
  assert.equal(panel.hidden, false, `Question must open: ${fieldId}`);
  assert.ok(panel.textContent.length > 100, `Help needs an explanation, not only a label: ${fieldId}`);
  const link = panel.querySelector('a[href^="#lesson-"]');
  assert.ok(link);
  assert.equal(decodeURIComponent(link.hash.slice('#lesson-'.length)), expectedLesson);
  link.click();
  assertOpenedLesson(expectedLesson);
}

test('every registered context explanation and every model points to an existing lesson', () => {
  for (const [key, help] of Object.entries(HELP_REGISTRY)) {
    assert.ok(lessons.has(help.lesson), `Unknown lesson in ${key}: ${help.lesson}`);
    assert.ok(help.text.length > 80, `A novice needs a substantive explanation for ${key}`);
  }
  for (const model of fixture.catalogue.models) {
    assert.ok(lessons.has(MODEL_LESSONS[model.id]), `Missing model instruction: ${model.id}`);
  }
});

test('the visible sidebar shortcut opens the beginner generator directly without training a model', async () => {
  view('lab');
  const jobsBefore = posts('/api/jobs').length;
  assert.ok($('create-data-wizard'));
  $('create-data-wizard').click();
  assert.equal($('view-data').classList.contains('active'), true);
  const tab = document.querySelector('[data-dw-tab="create"]');
  assert.equal(tab.getAttribute('aria-selected'), 'true');
  assert.equal(document.getElementById(tab.getAttribute('aria-controls')).hidden, false);
  assert.match(document.querySelector('.dw-scenario-description').textContent, /похожих объектов/);
  assert.equal(posts('/api/jobs').length, jobsBefore);
  document.querySelector('[data-dw-action="use"]').click();
  assert.equal($('view-lab').classList.contains('active'), true);
  assert.equal(posts('/api/jobs').length, jobsBefore, 'Switching to the laboratory is not a request to fit');
  await waitFor(() => document.querySelector('[data-dw-plot="scatter2d"]')?.data?.length, 'raw dataset charts');
});

test('every dataset question opens the named instruction rather than the first lesson', () => {
  view('data');
  const links = [...document.querySelectorAll('#dataset-workspace [data-dw-help]')];
  assert.ok(links.length >= 12, 'Dataset creation and exploration need contextual explanations');
  const topics = {
    'Мои датасеты':'21-read-data', 'Задача набора':'21-read-data',
    'Цель — что хотим предсказывать':'01-prediction', 'Шум — случайная часть цели':'21-read-data',
    'Насколько признаки похожи':'05-scaling-correlation', 'Выбросы — необычные наблюдения':'21-read-data',
    'Зерно случайности':'21-read-data', 'Сколько точек рисовать':'21-read-data',
    'Что показывают оси':'21-read-data', 'Цвет точки':'21-read-data',
    'Распределение значений':'21-read-data', 'Корреляция числовых столбцов':'05-scaling-correlation',
  };
  for (const link of links) {
    const id = link.dataset.dwHelp;
    assert.ok(lessons.has(id), `Dataset help uses an unresolved alias: ${id}`);
    const question = link.closest('.dw-help-wrap').querySelector('.dw-help');
    const title = link.closest('.dw-help-pop').querySelector('strong').textContent;
    assert.equal(id, topics[title], `Instruction must describe this topic: ${title}`);
    const panel = question.closest('[data-dw-panel]');
    if (panel) document.querySelector(`[data-dw-tab="${panel.dataset.dwPanel}"]`).click();
    for (let ancestor = question.parentElement; ancestor; ancestor = ancestor.parentElement) {
      if (ancestor.tagName === 'DETAILS') ancestor.open = true;
    }
    question.click();
    assert.equal(link.closest('.dw-help-pop').hidden, false);
    link.click();
    assertOpenedLesson(id);
    view('data');
  }
});

test('preparation instructions cover missing values, feature engineering, train-only fitting and rare targets', () => {
  const routes = {
    'pw-imputation':'22-missing-values', 'pw-fill_value':'22-missing-values',
    'pw-knn_neighbors':'22-missing-values', 'pw-missing_indicator':'22-missing-values',
    'pw-degree':'24-feature-engineering', 'pw-selection':'24-feature-engineering',
    'pw-max_features':'24-feature-engineering', 'pw-method':'28-rare-target-smoter',
    'pw-undersample':'28-rare-target-smoter',
  };
  for (const [field, lesson] of Object.entries(routes)) {
    view('lab');
    if (field === 'pw-fill_value') change($('pw-imputation'), 'constant');
    if (field === 'pw-knn_neighbors') change($('pw-imputation'), 'knn');
    if (field === 'pw-max_features') change($('pw-selection'), 'f_regression');
    if (field === 'pw-undersample') change($('pw-method'), 'smogn');
    openContextHelp(field, lesson);
  }
  view('lab');
  const preview = document.querySelector('[data-pw-action="preview"]');
  const question = preview.parentElement.querySelector('.help-trigger');
  question.click();
  const link = document.getElementById(question.getAttribute('aria-controls')).querySelector('a[href^="#lesson-"]');
  link.click();
  assertOpenedLesson('25-data-leakage');
});

test('CV, time gaps, group identifiers and hyperparameter search link to their own instructions', () => {
  const routes = {
    'vw-cv-method':'26-cross-validation', 'vw-cv-gap':'26-cross-validation',
    'vw-cv-group':'26-cross-validation', 'vw-search-method':'27-hyperparameter-search',
    'vw-search-trials':'27-hyperparameter-search', 'vw-space-help':'27-hyperparameter-search',
    'vw-search-metric':'20-metrics-experiment',
  };
  for (const [field, lesson] of Object.entries(routes)) {
    view('lab');
    if (field === 'vw-cv-gap') change($('vw-cv-method'), 'timeseries');
    if (field === 'vw-cv-group') change($('vw-cv-method'), 'group_kfold');
    openContextHelp(field, lesson);
  }
});

test('all model parameter forms retain readable help and open the model-specific instruction', async () => {
  change($('parameter-mode'), 'advanced');
  for (const model of fixture.catalogue.models) {
    view('lab');
    change($('model'), model.id);
    await waitFor(() => $('model-params').querySelectorAll('.help-trigger').length === model.params.length, `help for ${model.id}`);
    for (const parameter of model.params) {
      view('lab');
      openContextHelp(parameter.key, MODEL_LESSONS[model.id]);
    }
  }
  view('lab');
  change($('model'), 'ridge');
  change($('parameter-mode'), 'beginner');
});

test('selecting an available model export initiates a download with the server filename and resets the selector', async () => {
  const originalFetch = globalThis.fetch;
  const anchorClick = window.HTMLAnchorElement.prototype.click;
  const downloads = [];
  const exports = [];
  const blob = new Blob(['{"model":"ridge"}'], {type:'application/json'});
  globalThis.fetch = async (address, options) => {
    const url = new URL(address, window.location.href);
    if (/^\/api\/jobs\/[^/]+\/export$/.test(url.pathname)) {
      exports.push(url);
      return {ok:true, async blob() {return blob;}, headers:new Headers({'Content-Disposition':'attachment; filename="ridge-passport.json"'})};
    }
    return originalFetch(address, options);
  };
  window.HTMLAnchorElement.prototype.click = function () {
    if (this.download) downloads.push({filename:this.download, href:this.href});
    else anchorClick.call(this);
  };
  try {
    const choice = [...$('export').options].find(option => option.value === 'passport');
    assert.equal(choice.disabled, false);
    change($('export'), 'passport');
    await waitFor(() => downloads.length === 1, 'model export download');
    assert.equal(exports[0].searchParams.get('format'), 'passport');
    const fittedJobId = [...jobs.keys()].at(-1);
    assert.ok(fittedJobId, 'A download must use a fitted model');
    assert.equal(exports[0].pathname, `/api/jobs/${fittedJobId}/export`);
    assert.equal(downloads[0].filename, 'ridge-passport.json');
    assert.match(downloads[0].href, /^blob:/);
    assert.equal($('export').value, '');
  } finally {
    globalThis.fetch = originalFetch;
    window.HTMLAnchorElement.prototype.click = anchorClick;
  }
});
