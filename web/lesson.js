const taskLabels = {
  regression: 'Регрессия', classification: 'Классификация', clustering: 'Кластеризация',
  ranking: 'Ранжирование', forecasting: 'Временной ряд', panel: 'Панельные ряды',
  anomaly: 'Поиск аномалий', reduction: 'Уменьшение размерности',
};

const content = document.getElementById('lesson-content');
const list = document.getElementById('lesson-list');
const search = document.getElementById('lesson-search');
const task = document.getElementById('lesson-task');
let catalogue = [];
let currentId = new URLSearchParams(location.search).get('id');
let requestNumber = 0;

function node(tag, text, className) {
  const element = document.createElement(tag);
  if (text !== undefined) element.textContent = text;
  if (className) element.className = className;
  return element;
}

async function get(url) {
  const response = await fetch(url, {headers: {Accept: 'application/json'}});
  if (!response.ok) {
    if (response.status === 404) throw new Error('Такого урока нет. Выбери материал в каталоге.');
    throw new Error('Не удалось прочитать учебник. Проверь, что CML-lab запущен, и обнови страницу.');
  }
  return response.json();
}

function paragraphs(text, parent) {
  for (const paragraph of String(text || '').split(/\n\s*\n/)) {
    if (paragraph.trim()) parent.append(node('p', paragraph));
  }
}

function renderList() {
  list.replaceChildren();
  const term = search.value.trim().toLocaleLowerCase('ru');
  const items = catalogue.filter(item => (!task.value || item.tasks?.includes(task.value)) &&
    (!term || `${item.title} ${item.summary}`.toLocaleLowerCase('ru').includes(term)));
  document.getElementById('lesson-count').textContent = `Найдено уроков: ${items.length}`;
  for (const item of items) {
    const link = node('a', item.title, 'lesson-item');
    link.href = `/lesson?id=${encodeURIComponent(item.id)}`;
    if (item.id === currentId) link.setAttribute('aria-current', 'page');
    link.addEventListener('click', event => {
      if (event.button || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
      event.preventDefault();
      currentId = item.id;
      history.pushState({}, '', link.href);
      loadLesson(true);
    });
    list.append(link);
  }
  if (!items.length) list.append(node('p', 'Ничего не найдено. Сократи запрос или выбери все задачи.', 'lesson-note'));
}

function renderLesson(lesson) {
  content.replaceChildren();
  const tags = node('div', undefined, 'lesson-tags');
  for (const key of lesson.tasks || []) tags.append(node('span', taskLabels[key] || key, 'lesson-tag'));
  content.append(tags, node('h1', lesson.title), node('p', lesson.summary, 'lesson-summary'));
  document.title = `${lesson.title} · CML-lab`;
  if (lesson.preset) {
    const actions = node('div', undefined, 'lesson-actions');
    const link = node('a', 'Открыть учебный пример в лаборатории', 'lesson-open-preset');
    link.href = `/?lesson=${encodeURIComponent(lesson.id)}`;
    actions.append(link, node('span', 'Откроется форма. Обучение запускаешь своей кнопкой.', 'lesson-note'));
    content.append(actions);
  }
  for (const section of lesson.sections || []) {
    const block = node('section');
    block.append(node('h2', section.title));
    paragraphs(section.text, block);
    if (section.formula) block.append(node('code', section.formula, 'lesson-formula'));
    content.append(block);
  }
  if (lesson.exercise) {
    const exercise = node('section', undefined, 'lesson-exercise');
    exercise.append(node('h2', 'Проверь сам'));
    paragraphs(lesson.exercise, exercise);
    content.append(exercise);
  }
  const sources = node('section');
  sources.append(node('h2', 'Источники и подробное чтение'));
  const links = node('ol', undefined, 'lesson-sources');
  for (const source of lesson.sources || []) {
    const item = node('li');
    const link = node('a', source.title);
    const url = new URL(source.url);
    if (url.protocol !== 'https:') continue;
    link.href = url.href;
    link.target = '_blank';
    link.rel = 'noopener noreferrer';
    item.append(link, node('small', `${source.language === 'ru' ? 'На русском' : 'Оригинал на английском; объяснение выше на русском'}${source.kind === 'supplementary' ? ' · дополнительный материал' : ''}`));
    links.append(item);
  }
  sources.append(links);
  content.append(sources);
}

async function loadLesson(focus = false) {
  const number = ++requestNumber;
  renderList();
  content.replaceChildren(node('p', 'Загружаем урок…'));
  try {
    if (!currentId) {
      content.replaceChildren(node('h1', 'Учебник CML-lab'), node('p', 'Выбери задачу и урок слева. В формах лаборатории знак ? открывает объяснение соответствующей настройки.'));
      return;
    }
    const lesson = await get(`/api/learning/lessons/${encodeURIComponent(currentId)}`);
    if (number !== requestNumber) return;
    renderLesson(lesson);
    if (focus) content.focus();
  } catch (error) {
    if (number === requestNumber) content.replaceChildren(node('h1', 'Урок не открыт'), node('p', error.message, 'lesson-error'));
  }
}

search.addEventListener('input', renderList);
task.addEventListener('change', renderList);
document.getElementById('lesson-print').addEventListener('click', () => window.print());
window.addEventListener('popstate', () => {
  currentId = new URLSearchParams(location.search).get('id');
  loadLesson();
});

try {
  const result = await get('/api/learning/lessons');
  catalogue = result.items || [];
  await loadLesson();
} catch (error) {
  content.replaceChildren(node('h1', 'Учебник не открыт'), node('p', error.message, 'lesson-error'));
}
