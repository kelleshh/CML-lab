/** Standalone instruction page behavior. JSDOM does not validate visual layout. */
import assert from 'node:assert/strict';
import {test} from 'node:test';
import {execFileSync} from 'node:child_process';
import {readFileSync} from 'node:fs';
import {createRequire} from 'node:module';
import {dirname, join} from 'node:path';
import {fileURLToPath} from 'node:url';

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const require = createRequire(import.meta.url);
let JSDOM;
try { ({JSDOM} = require('jsdom')); }
catch { ({JSDOM} = require(join(root, '..', 'build-tools', 'node_modules', 'jsdom'))); }
const fixtures = JSON.parse(execFileSync(join(root, '.venv', 'bin', 'python'), ['-c', `
import json
from cml_lab.contexts.learning.application import LearningService
from cml_lab.infrastructure.learning import InMemoryLearningRepository
service = LearningService(InMemoryLearningRepository())
print(json.dumps({"catalogue": service.list_lessons(), "lessons": {item["id"]: item for item in service.catalogue()}}, ensure_ascii=False))
`], {cwd: root, encoding: 'utf8'}));
const html = readFileSync(join(root, 'web', 'lesson.html'), 'utf8');
const script = readFileSync(join(root, 'web', 'lesson.js'), 'utf8');

async function page(identifier = '', transform = value => value) {
  const dom = new JSDOM(html, {url: `http://localhost/lesson${identifier ? `?id=${identifier}` : ''}`, runScripts: 'outside-only'});
  const {window} = dom;
  window.fetch = async url => {
    if (url === '/api/learning/lessons') return {ok: true, status: 200, json: async () => structuredClone(fixtures.catalogue)};
    const id = decodeURIComponent(url.split('/').pop());
    const lesson = fixtures.lessons[id];
    return {ok: Boolean(lesson), status: lesson ? 200 : 404, json: async () => transform(structuredClone(lesson))};
  };
  await window.eval(`(async () => {${script}\n})()`);
  return {dom, window, document: window.document};
}

test('question-mark URL opens its exact article and a form link without running training', async () => {
  const {dom, document} = await page('classification-basics');
  assert.equal(document.querySelector('h1').textContent, fixtures.lessons['classification-basics'].title);
  const link = document.querySelector('.lesson-open-preset');
  assert.equal(link.getAttribute('href'), '/?lesson=classification-basics');
  assert.match(document.querySelector('.lesson-actions').textContent, /своей кнопкой/);
  assert.equal(document.querySelector('[aria-current="page"]').getAttribute('href'), '/lesson?id=classification-basics');
  dom.window.close();
});

test('unknown material reports the missing article rather than showing an unrelated lesson', async () => {
  const {dom, document} = await page('unknown-lesson');
  assert.equal(document.querySelector('h1').textContent, 'Урок не открыт');
  assert.match(document.querySelector('.lesson-error').textContent, /Такого урока нет/);
  assert.equal(document.querySelector('.lesson-open-preset'), null);
  dom.window.close();
});

test('task filters and Russian search reduce the actual curriculum without discarding the open article', async () => {
  const {dom, document, window} = await page('panel-groups');
  const selector = document.getElementById('lesson-task');
  selector.value = 'ranking';
  selector.dispatchEvent(new window.Event('change'));
  let links = [...document.querySelectorAll('#lesson-list a')];
  assert.ok(links.length > 0);
  assert.ok(links.every(link => fixtures.lessons[new URL(link.href).searchParams.get('id')].tasks.includes('ranking')));
  const search = document.getElementById('lesson-search');
  search.value = 'NDCG';
  search.dispatchEvent(new window.Event('input'));
  links = [...document.querySelectorAll('#lesson-list a')];
  assert.ok(links.some(link => link.getAttribute('href') === '/lesson?id=ranking-metrics'));
  assert.equal(document.querySelector('h1').textContent, fixtures.lessons['panel-groups'].title);
  dom.window.close();
});

test('keyboard-operable catalogue navigation updates URL, focus, and selected article', async () => {
  const {dom, document, window} = await page('classification-basics');
  document.querySelector('#lesson-list a[href="/lesson?id=tree-depth"]').click();
  await new Promise(resolve => setTimeout(resolve, 0));
  assert.equal(new URL(window.location.href).searchParams.get('id'), 'tree-depth');
  assert.equal(document.querySelector('h1').textContent, fixtures.lessons['tree-depth'].title);
  assert.equal(document.activeElement.id, 'lesson-content');
  dom.window.close();
});

test('sources open safely in new tabs and formulas remain literal text', async () => {
  const {dom, document} = await page('03-least-squares');
  const sources = [...document.querySelectorAll('.lesson-sources a')];
  assert.ok(sources.length > 0);
  assert.ok(sources.every(link => link.target === '_blank' && /noopener/.test(link.rel)));
  assert.ok(document.querySelector('.lesson-formula').textContent.includes('SSE'));
  assert.match(document.querySelector('.lesson-sources').textContent, /английском/);
  dom.window.close();
});

test('article data is treated as text and cannot inject executable markup', async () => {
  const payload = '<img src=x onerror="window.pwned=1">';
  const {dom, document, window} = await page('tree-depth', lesson => {
    lesson.title = payload;
    lesson.sections[0].text = payload;
    return lesson;
  });
  assert.equal(document.querySelector('h1').textContent, payload);
  assert.equal(document.querySelector('#lesson-content img'), null);
  assert.equal(window.pwned, undefined);
  dom.window.close();
});

test('empty search and browser print expose clear user actions', async () => {
  const {dom, document, window} = await page();
  let printed = 0;
  window.print = () => {printed += 1;};
  document.getElementById('lesson-print').click();
  assert.equal(printed, 1);
  const search = document.getElementById('lesson-search');
  search.value = 'несуществующееслово';
  search.dispatchEvent(new window.Event('input'));
  assert.match(document.getElementById('lesson-list').textContent, /Ничего не найдено/);
  assert.equal(document.querySelector('h1').textContent, 'Учебник CML-lab');
  for (const field of document.querySelectorAll('input,select')) assert.ok(field.labels.length);
  dom.window.close();
});
