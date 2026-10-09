/** Dataset library and raw-data exploration, independent of training. */
const FEATURE = "#66e4bc";
const TARGET = "#ff927e";
const CATEGORY_COLORS = ["#66e4bc", "#ff927e", "#a9a1ff", "#ffd17c", "#6acbff", "#e38dc6", "#b5de82", "#e5b7a0"];
const TASKS = { regression: "Регрессия · предсказать число", classification: "Классификация · определить класс", clustering: "Кластеризация · найти группы", time_series: "Временные ряды · учитывать порядок", other: "Другая задача / цель еще не выбрана" };
const SOURCES = { synthetic: "Созданные здесь", sklearn: "Scikit-learn", upload: "Мои файлы", custom: "Ручной ввод", openml: "OpenML", builtin: "Встроенные", fetch: "Из интернета" };
const SYNTHETIC = { linear: "Прямая зависимость", correlated: "Признаки повторяют друг друга", sparse: "Много лишних признаков", nonlinear: "Изогнутая зависимость", heteroscedastic: "Разброс растет вместе с признаком", positive: "Только положительная цель", counts: "Число событий", grouped: "Группы полезных признаков", fused: "Похожие соседние коэффициенты" };
const SCENARIOS = {
  clean: { label: "Начать с понятной прямой", name: "linear", noise: 0, correlation: 0.1, outliers: 0, n_features: 2, description: "Все точки объясняются признаками. Хороший первый опыт: менять один признак и видеть, как меняется цель." },
  realistic: { label: "Добавить обычный разброс", name: "linear", noise: 8, correlation: 0.2, outliers: 0, n_features: 2, description: "У похожих объектов цель немного отличается. Прямая описывает общую тенденцию, а не каждую точку." },
  duplicates: { label: "Сравнить похожие признаки", name: "correlated", noise: 8, correlation: 0.95, outliers: 0, n_features: 4, description: "Признаки несут похожую информацию. Проверьте, как Ridge стабилизирует коэффициенты, а Lasso выбирает часть признаков." },
  unnecessary: { label: "Найти полезные признаки", name: "sparse", noise: 8, correlation: 0.2, outliers: 0, n_features: 8, description: "Большинство признаков не влияет на цель. Lasso помогает увидеть, какие коэффициенты можно обнулить." },
  outliers: { label: "Проверить устойчивость к выбросам", name: "linear", noise: 8, correlation: 0.2, outliers: 0.1, n_features: 2, description: "У 10% наблюдений большая ошибка цели. Сравните обычную регрессию с Huber и RANSAC." },
  curved: { label: "Исследовать изгиб", name: "nonlinear", noise: 8, correlation: 0.2, outliers: 0, n_features: 2, description: "Первый признак влияет на цель через квадрат. Затем попробуйте полиномиальные признаки." }
};
const HELP = {
  library: ["Мои датасеты", "Здесь остаются загруженные файлы, готовые наборы и созданные примеры. Это сами данные, а не обученные модели. У каждого набора можно выбрать задачу и цель, изменить название и выгрузить CSV.", "21-read-data"],
  task: ["Задача набора", "Регрессия предсказывает число: цену, длительность, температуру. Классификация предсказывает категорию: сорт растения или диагноз. Набор для классификации можно исследовать здесь; линейной регрессии нужна числовая цель.", "21-read-data"],
  target: ["Цель — что хотим предсказывать", "Цель, или таргет, — правильный ответ для каждого наблюдения. Например, цена квартиры. Признаки — известные сведения: площадь и этаж. Цель выделена коралловым, признаки — мятным. Выбор цели здесь управляет исследованием данных.", "01-prediction"],
  noise: ["Шум — случайная часть цели", "Шум добавляется к рассчитанному правильному ответу. Он создает разброс: даже одинаковые признаки не гарантируют одинаковую цель. «Нет» = 0; «Небольшой» = 3; «Обычный» = 8; «Сильный» = 25 в единицах цели. Это настройки генератора, а не ошибка уже обученной модели.", "21-read-data"],
  correlation: ["Насколько признаки похожи", "Если один признак растет вместе с другим, они положительно коррелируют. В этом генераторе настройка задает одинаковую попарную корреляцию: слабая = 0.1, средняя = 0.6, сильная = 0.95. Похожие признаки могут делать коэффициенты обычной регрессии нестабильными.", "05-scaling-correlation"],
  outliers: ["Выбросы — необычные наблюдения", "Генератор выбирает заданную долю строк и добавляет очень большую ошибку к цели. Значение 10% означает примерно одну строку из десяти. В реальной таблице необычная точка не обязательно ошибочна: выясните причину перед удалением.", "21-read-data"],
  seed: ["Зерно случайности", "Генератор использует псевдослучайные числа. Одинаковые настройки и одинаковое зерно дают одинаковую таблицу: опыт можно воспроизвести и честно сравнить модели.", "21-read-data"],
  sample: ["Сколько точек рисовать", "График показывает воспроизводимую выборку для быстрого отклика. Число строк исходного набора остается прежним. Пропуски в выбранных координатах не изображаются. Статистика таблицы и число отображенных точек могут различаться.", "21-read-data"],
  axes: ["Что показывают оси", "2D: один признак по горизонтали и цель по вертикали. 3D: два признака по горизонтальным осям и числовая цель по высоте. Можно выбрать другие столбцы. Все значения здесь исходные: модель и подготовка данных еще не применялись.", "21-read-data"],
  color: ["Цвет точки", "Для числового столбца цвет меняется непрерывно от малого значения к большому. Для категорий каждая группа получает свой цвет и подпись. Цвет помогает увидеть цель даже при просмотре сразу трех признаков.", "21-read-data"],
  histogram: ["Распределение значений", "Высота столбика — сколько наблюдений попало в этот диапазон. Так можно заметить длинный хвост, несколько групп, перекос или редкие значения. Пропуски не становятся нулем.", "21-read-data"],
  correlation_matrix: ["Корреляция числовых столбцов", "Коэффициент Пирсона от −1 до 1 описывает линейную связь. +1: растут вместе; −1: один растет, другой падает; около 0: нет заметной линейной связи. Нулевая корреляция не исключает изогнутую зависимость и не доказывает причинность.", "05-scaling-correlation"]
};
const esc = (value) => String(value ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const number = (value) => value == null || !Number.isFinite(Number(value)) ? "—" : new Intl.NumberFormat("ru", { maximumFractionDigits: 4 }).format(Number(value));
let tooltipSequence = 0;
let workspaceSequence = 0;
const METADATA_TASKS = ["regression", "classification", "clustering", "time_series", "other"];
const options = (values, selected = "") => values.map((value) => {
  const item = typeof value === "string" ? { value, label: value } : value;
  return `<option value="${esc(item.value)}" ${String(item.value) === String(selected) ? "selected" : ""}>${esc(item.label)}</option>`;
}).join("");
function help(key) {
  const [title, text, anchor] = HELP[key] || [key, key, "21-read-data"];
  const id = `dw-help-${++tooltipSequence}`;
  return `<span class="dw-help-wrap"><button type="button" class="dw-help" aria-label="Подробнее: ${esc(title)}" aria-haspopup="dialog" aria-expanded="false" aria-controls="${id}">?</button><span class="dw-help-pop" id="${id}" role="dialog" aria-modal="false" aria-labelledby="${id}-title" hidden><strong id="${id}-title">${esc(title)}</strong><span>${esc(text)}</span><button type="button" class="dw-help-close" aria-label="Закрыть пояснение">×</button><a href="#learn" data-dw-help="${esc(anchor)}">Открыть описание в инструкции ↗</a></span></span>`;
}
class DatasetWorkspace {
  constructor({ api, onSelect, onCreated, onEdit, onHelp, onError, onDeleted } = {}) {
    this.api = api || this.defaultApi;
    this.onSelect = onSelect || (() => {
    });
    this.onCreated = onCreated || this.onSelect;
    this.onEdit = onEdit || (() => {
    });
    this.onHelp = onHelp || (() => {
    });
    this.onError = onError || (() => {
    });
    this.onDeleted = onDeleted || (() => {
    });
    this.catalogue = [];
    this.dataset = null;
    this.library = [];
    this.tab = "library";
    this.exploreSequence = 0;
    this.librarySequence = 0;
    this.busy = false;
    this.externalBusy = false;
    this.sourceSearch = "";
    this.sourceTask = "";
    this.searchTimer = null;
    this.exploreTimer = null;
    this.accessibilityId = `dw-workspace-${++workspaceSequence}`;
  }
  async defaultApi(path, options2 = {}) {
    if (options2.body && !(options2.body instanceof FormData)) options2 = { ...options2, body: JSON.stringify(options2.body), headers: { "Content-Type": "application/json", ...options2.headers } };
    const response = await fetch(`/api${path}`, options2);
    const payload = await response.json();
    if (!response.ok) throw new Error(typeof payload.detail === "string" ? payload.detail : "Не удалось выполнить действие.");
    return payload;
  }
  mount(container) {
    this.container = typeof container === "string" ? document.querySelector(container) : container;
    if (!this.container) throw new Error("Не найден контейнер данных.");
    this.container.classList.add("dataset-workspace");
    this.container.innerHTML = `<div class="dw-header"><div><h2>Мои датасеты ${help("library")}</h2><p>Добавьте таблицу, исследуйте признаки и выберите, что предсказывать.</p></div><button class="dw-button dw-secondary" data-dw-action="refresh">↻ Обновить библиотеку</button></div>
      <div class="dw-tablist" role="tablist" aria-label="Работа с данными"><button role="tab" aria-selected="true" data-dw-tab="library">Мои датасеты</button><button role="tab" aria-selected="false" data-dw-tab="create">Создать пример</button><button role="tab" aria-selected="false" data-dw-tab="sources">Добавить из источника</button></div>
      <div class="dw-status" role="status" aria-live="polite"></div>
      <section class="dw-section" data-dw-panel="library">${this.libraryMarkup()}</section>
      <section class="dw-section" data-dw-panel="create" hidden>${this.createMarkup()}</section>
      <section class="dw-section" data-dw-panel="sources" hidden>${this.sourcesMarkup()}</section>
      <section class="dw-explorer" aria-label="Исследование исходных данных" hidden></section>`;
    for (const tab of this.container.querySelectorAll('[data-dw-tab]')) {
      const key = tab.dataset.dwTab;
      tab.id = `${this.accessibilityId}-tab-${key}`;
      tab.setAttribute('aria-controls', `${this.accessibilityId}-panel-${key}`);
      tab.tabIndex = key === this.tab ? 0 : -1;
      const panel = this.container.querySelector(`[data-dw-panel="${key}"]`);
      panel.id = `${this.accessibilityId}-panel-${key}`;
      panel.setAttribute('role', 'tabpanel');
      panel.setAttribute('aria-labelledby', tab.id);
      panel.tabIndex = 0;
    }
    this.associateControls();
    this.container.addEventListener("click", (event) => this.handleClick(event));
    this.container.addEventListener("change", (event) => this.handleChange(event));
    this.container.addEventListener("input", (event) => this.handleInput(event));
    this.container.addEventListener("keydown", (event) => this.handleKeyboard(event));
    this.container.addEventListener("focusin", (event) => {
      if (event.target.matches('.dw-help')) this.openHelp(event.target.closest('.dw-help-wrap'));
    });
    this.container.addEventListener("focusout", (event) => {
      const wrap = event.target.closest(".dw-help-wrap");
      if (wrap && !wrap.contains(event.relatedTarget)) {
        this.closeHelp(wrap);
        wrap.classList.remove('is-dismissed');
      }
    });
    this.container.addEventListener('pointerover', event => {
      const wrap = event.target.closest('.dw-help-wrap');
      if (wrap && !wrap.contains(event.relatedTarget)) {
        wrap.classList.remove('is-dismissed');
        this.openHelp(wrap);
      }
    });
    this.container.addEventListener('pointerout', event => {
      const wrap = event.target.closest('.dw-help-wrap');
      if (wrap && !wrap.contains(event.relatedTarget) && !wrap.classList.contains('is-pinned') && !wrap.contains(this.container.ownerDocument.activeElement)) this.closeHelp(wrap);
    });
    this.onOutsideHelp = event => {
      this.container.querySelectorAll('.dw-help-wrap.is-open').forEach(wrap => {
        if (!wrap.contains(event.target)) this.closeHelp(wrap);
      });
    };
    this.container.ownerDocument.addEventListener('pointerdown', this.onOutsideHelp);
    if (typeof ResizeObserver !== "undefined") {
      this.observer = new ResizeObserver(() => this.resize());
      this.observer.observe(this.container);
    }
    this.renderSources();
    this.renderScenario();
    this.refreshLibrary();
    return this;
  }
  associateControls(scope = this.container) {
    for (const field of scope.querySelectorAll('input,select,textarea')) {
      const label = field.closest('label');
      if (!label) continue;
      field.id ||= `${this.accessibilityId}-field-${field.dataset.dw}`;
      label.htmlFor = field.id;
      // Help buttons are labelable themselves. Explicit association keeps the
      // actual field named even when its question mark precedes it in the label.
      const text = label.cloneNode(true);
      text.querySelectorAll('.dw-help-wrap,input,select,textarea,button').forEach(node => node.remove());
      field.setAttribute('aria-label', text.textContent.replace(/\s+/g, ' ').trim());
    }
  }
  openHelp(wrap, pin = false) {
    if (!wrap || wrap.classList.contains('is-dismissed') && !pin) return;
    this.container.querySelectorAll('.dw-help-wrap.is-open').forEach(other => {if (other !== wrap) this.closeHelp(other);});
    wrap.classList.remove('is-dismissed');
    wrap.classList.add('is-open');
    if (pin) wrap.classList.add('is-pinned');
    wrap.querySelector('.dw-help').setAttribute('aria-expanded', 'true');
    wrap.querySelector('.dw-help-pop').hidden = false;
  }
  closeHelp(wrap, restoreFocus = false) {
    if (!wrap) return;
    wrap.classList.remove('is-open', 'is-pinned');
    wrap.querySelector('.dw-help').setAttribute('aria-expanded', 'false');
    wrap.querySelector('.dw-help-pop').hidden = true;
    if (restoreFocus) {
      wrap.classList.add('is-dismissed');
      wrap.querySelector('.dw-help').focus();
    }
  }
  handleKeyboard(event) {
    const tab = event.target.closest('[data-dw-tab]');
    if (tab && ['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) {
      const tabs = [...this.container.querySelectorAll('[data-dw-tab]')];
      const current = tabs.indexOf(tab);
      const index = event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1 : (current + (event.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length;
      event.preventDefault();
      this.switchTab(tabs[index].dataset.dwTab);
      tabs[index].focus();
      return;
    }
    const wrap = event.target.closest('.dw-help-wrap');
    if (wrap && event.target.matches('.dw-help') && event.key === 'ArrowDown') {
      event.preventDefault();
      this.openHelp(wrap, true);
      wrap.querySelector('a')?.focus();
    }
    if (event.key === 'Escape') {
      const open = [...this.container.querySelectorAll('.dw-help-wrap.is-open')];
      if (open.length) {
        event.preventDefault();
        for (const help of open) this.closeHelp(help, true);
      }
    }
  }
  libraryMarkup() {
    return `<div class="dw-toolbar"><label class="dw-search">Поиск<input data-dw="library-search" type="search" placeholder="Название, описание или метка"></label><label>Задача ${help("task")}<select data-dw="library-task"><option value="">Все задачи</option>${options(Object.entries(TASKS).map(([value, label]) => ({ value, label })))}</select></label><label>Источник<select data-dw="library-source"><option value="">Все источники</option>${options(Object.entries(SOURCES).filter(([key]) => !["builtin", "fetch"].includes(key)).map(([value, label]) => ({ value, label })))}</select></label><label>Порядок<select data-dw="library-sort"><option value="recent">Последние добавленные</option><option value="name">По названию</option><option value="rows">Больше строк сначала</option></select></label></div><div class="dw-library-summary"></div><div class="dw-library-grid"></div>`;
  }
  createMarkup() {
    return `<div class="dw-creation-grid"><div><h3>1. Что хотите увидеть?</h3><label>Учебный сценарий<select data-dw="scenario">${options(Object.entries(SCENARIOS).map(([value, item]) => ({ value, label: item.label })), "realistic")}</select></label><p class="dw-scenario-description"></p><label>Тип зависимости<select data-dw="generator">${options(Object.entries(SYNTHETIC).map(([value, label]) => ({ value, label })), "linear")}</select></label><p class="dw-generator-note"></p></div><div><h3>2. Настройте таблицу</h3><div class="dw-two"><label>Наблюдений, строк<input data-dw="n-samples" type="number" min="12" max="100000" step="10" value="180"></label><label><span class="dw-feature">Признаков</span>, столбцов<input data-dw="n-features" type="number" min="1" max="100" value="2"></label></div><label>Разброс <span class="dw-target">цели</span> ${help("noise")}<select data-dw="noise-preset"><option value="0">Нет — точная зависимость</option><option value="3">Небольшой — точки близко</option><option value="8" selected>Обычный — заметный разброс</option><option value="25">Сильный — зависимость труднее увидеть</option></select></label><label>Похожесть <span class="dw-feature">признаков</span> ${help("correlation")}<select data-dw="correlation-preset"><option value="0.1">Слабая — разная информация</option><option value="0.6">Средняя — часть информации общая</option><option value="0.95">Сильная — почти дубликаты</option></select></label><label>Необычные точки ${help("outliers")}<select data-dw="outliers-preset"><option value="0">Без выбросов</option><option value="0.05">Немного — 5% строк</option><option value="0.1">Заметно — 10% строк</option><option value="0.25">Много — 25% строк</option></select></label></div></div><details class="dw-advanced"><summary>Точные значения и дополнительные настройки</summary><p>Эти поля переопределяют варианты выше. Для первого опыта достаточно готовых вариантов.</p><div class="dw-four"><label>Шум, стандартное отклонение<input data-dw="noise-exact" type="number" min="0" max="10000" step="any" placeholder="Из варианта выше"></label><label>Корреляция признаков<input data-dw="correlation-exact" type="number" min="-0.99" max="0.9999" step="0.01" placeholder="Из варианта выше"></label><label>Доля лишних признаков<input data-dw="sparsity" type="number" min="0" max="1" step="0.05" value="0.7"></label><label>Разница масштабов, порядки<input data-dw="scale-spread" type="number" min="0" max="6" step="0.5" value="0"></label></div><p>Доля лишних признаков действует в сценарии поиска полезных признаков. Разница масштабов 3 дает отношение до 1000 раз.</p></details><div class="dw-create-footer"><label>Зерно случайности ${help("seed")}<input data-dw="seed" type="number" min="0" max="1000000" value="42"></label><button class="dw-button dw-primary" data-dw-action="create">Создать и исследовать</button></div>`;
  }
  sourcesMarkup() {
    return `<div class="dw-import-grid"><div class="dw-import-card"><h3>Своя таблица</h3><p>CSV, TSV или Excel · до 25 МБ. Столбцы станут признаками; цель выберете после загрузки.</p><label class="dw-file-label">Выберите файл<input data-dw="upload" type="file" accept=".csv,.tsv,.xlsx,.xls"></label></div><div class="dw-import-card"><h3>OpenML</h3><p>Открытые данные по числовому ID или имени. Для загрузки нужен интернет.</p><div class="dw-two"><label>ID набора<input data-dw="openml-id" type="number" min="1" placeholder="531"></label><label>или название<input data-dw="openml-name" type="text" placeholder="boston"></label></div><button class="dw-button dw-secondary" data-dw-action="openml">Добавить в библиотеку</button></div></div><div class="dw-toolbar"><label class="dw-search">Каталог scikit-learn<input data-dw="source-search" type="search" placeholder="Ирисы, жилье, make_regression…"></label><label>Задача<select data-dw="source-task"><option value="">Все задачи</option>${options(Object.entries(TASKS).map(([value, label]) => ({ value, label })))}</select></label><label>Доступность<select data-dw="source-support"><option value="supported">Можно загрузить</option><option value="all">Показать весь каталог</option></select></label></div><p class="dw-note">Задача описывает исходный набор. Классы можно раскрасить на графике. Для обучения линейной регрессии выберите числовую цель.</p><div class="dw-source-grid"></div>`;
  }
  setCatalogue(catalogue) {
    this.catalogue = Array.isArray(catalogue) ? catalogue : catalogue?.datasets || [];
    if (this.container) this.renderSources();
    return this;
  }
  setDataset(metadata) {
    if (!metadata) return this;
    if (this.dataset?.id === metadata.id && this.dataset.default_target === metadata.default_target && this.dataset.task_target === metadata.task_target && this.container?.querySelector('[data-dw="explore-target"]')) return this;
    this.dataset = metadata;
    if (this.container) {
      this.renderExplorer();
      this.refreshLibrary();
    }
    return this;
  }
  field(key) {
    return this.container.querySelector(`[data-dw="${key}"]`);
  }
  status(message, error = false) {
    const node = this.container?.querySelector(".dw-status");
    if (node) {
      node.textContent = message;
      node.classList.toggle("is-error", error);
      node.hidden = !message;
    }
  }
  async guarded(work) {
    if (this.externalBusy) {
      this.status("Сначала дождитесь завершения обучения или остановите расчет.", true);
      return;
    }
    if (this.busy) return;
    this.busy = true;
    this.container.setAttribute("aria-busy", "true");
    this.container.querySelectorAll('[data-dw-action="create"],[data-dw-action="openml"]').forEach((el) => el.disabled = true);
    try {
      await work();
    } catch (error) {
      this.status(error.message, true);
      this.onError(error);
    } finally {
      this.busy = false;
      this.container.removeAttribute("aria-busy");
      this.container.querySelectorAll('[data-dw-action="create"],[data-dw-action="openml"]').forEach((el) => el.disabled = this.externalBusy);
      this.setBusy(this.externalBusy);
    }
  }
  async refreshLibrary() {
    if (!this.container) return;
    const seq = ++this.librarySequence;
    const search = new URLSearchParams();
    for (const [key, value] of [["query", this.field("library-search")?.value], ["task", this.field("library-task")?.value], ["source", this.field("library-source")?.value]]) if (value) search.set(key, value);
    try {
      const payload = await this.api(`/datasets/library${search.size ? "?" + search : ""}`);
      if (seq !== this.librarySequence) return;
      this.library = Array.isArray(payload) ? payload : payload.items || payload.datasets || [];
      this.renderLibrary();
    } catch (error) {
      if (seq !== this.librarySequence) return;
      this.container.querySelector(".dw-library-grid").innerHTML = `<div class="dw-empty"><h3>Библиотека пока недоступна</h3><p>${esc(error.message)}</p><button class="dw-button dw-secondary" data-dw-action="refresh">Попробовать снова</button></div>`;
    }
  }
  renderLibrary() {
    const grid = this.container.querySelector(".dw-library-grid");
    const query = (this.field("library-search")?.value || "").toLocaleLowerCase("ru");
    const task = this.field("library-task")?.value, source = this.field("library-source")?.value;
    let items = this.library.filter((item) => (!task || (item.tasks || [item.task]).includes(task)) && (!source || item.source === source) && (!query || [item.name, item.description, ...item.tags || []].join(" ").toLocaleLowerCase("ru").includes(query)));
    const sort = this.field("library-sort")?.value;
    if (sort === "name") items.sort((a, b) => a.name.localeCompare(b.name, "ru"));
    else if (sort === "rows") items.sort((a, b) => b.rows - a.rows);
    this.container.querySelector(".dw-library-summary").textContent = `Наборов: ${items.length}${this.dataset ? " · выбран: " + this.dataset.name : ""}`;
    if (!items.length) {
      grid.innerHTML = `<div class="dw-empty"><h3>${this.library.length ? "По этим условиям ничего не найдено" : "Здесь появятся ваши данные"}</h3><p>${this.library.length ? "Уберите фильтр или попробуйте другое название." : "Создайте учебный пример, загрузите файл или добавьте готовый набор. Он останется в этой библиотеке."}</p><button class="dw-button dw-primary" data-dw-tab="create">Создать пример</button><button class="dw-button dw-secondary" data-dw-tab="sources">Загрузить данные</button></div>`;
      return;
    }
    grid.innerHTML = items.map((item) => {
      const cols = Array.isArray(item.columns) ? item.columns.length : Number(item.stats?.columns || item.columns_count || item.column_count || item.columns || 0);
      const missing = Number(item.stats?.missing_cells ?? item.missing ?? item.missing_cells ?? (Array.isArray(item.columns) ? item.columns.reduce((s, c) => s + (c.missing || 0), 0) : 0));
      const selected = this.dataset?.id === item.id;
      return `<article class="dw-dataset-card ${selected ? "is-selected" : ""}"><div class="dw-card-top"><span class="dw-task-chip">${esc(TASKS[item.task] || item.task || TASKS.other)}</span>${selected ? '<span class="dw-selected-label">Выбран</span>' : ""}</div><h3>${esc(item.name || "Без названия")}</h3><p class="dw-card-description">${esc(item.description || SOURCES[item.source] || "Табличные данные")}</p><dl class="dw-card-stats"><div><dt>Строк</dt><dd>${number(item.rows)}</dd></div><div><dt>Столбцов</dt><dd>${number(cols)}</dd></div><div><dt>Пропусков</dt><dd>${number(missing)}</dd></div></dl><div class="dw-card-target"><span class="dw-target">Цель</span>: ${esc(item.task_target || item.default_target || item.stats?.target || item.target || "еще не выбрана")}</div><div class="dw-card-tags">${(item.tags || []).slice(0, 5).map((tag) => `<span>${esc(tag)}</span>`).join("")}</div><footer><span>${esc(SOURCES[item.source] || item.source || "Локальный набор")}</span><button class="dw-button dw-secondary" data-dw-select="${esc(item.id)}">${selected ? "Исследовать" : "Выбрать и исследовать"}</button></footer></article>`;
    }).join("");
    this.setBusy(this.externalBusy);
  }
  renderSources() {
    if (!this.container) return;
    const search = (this.field("source-search")?.value || "").toLowerCase();
    const task = this.field("source-task")?.value;
    const supported = this.field("source-support")?.value !== "all";
    const items = this.catalogue.filter((item) => !(item.kind === "synthetic" && !item.name?.startsWith("make_")) && (!supported || item.supported) && (!task || (item.tasks || [item.task]).includes(task)) && (!search || [item.label, item.name, item.description].join(" ").toLowerCase().includes(search)));
    const groups = /* @__PURE__ */ new Map();
    for (const item of items) {
      const key = item.task || "unknown";
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(item);
    }
    this.container.querySelector(".dw-source-grid").innerHTML = [...groups].map(([task2, list]) => `<section class="dw-source-group"><h3>${esc(TASKS[task2] || task2)} <span>${list.length}</span></h3><div class="dw-source-cards">${list.map((item) => `<article class="dw-source-card"><div class="dw-card-top"><code>${esc(item.name)}</code><span class="dw-support ${item.supported ? "" : "is-unavailable"}">${item.supported ? "Доступен" : "Нужен адаптер"}</span></div><h4>${esc(item.label || item.name)}</h4><p>${esc(item.description)}</p><footer><span>${item.requires_network ? "↓ Нужен интернет" : "На этом компьютере"}</span><button class="dw-button dw-secondary" data-dw-source="${esc(item.id || item.name)}" ${item.supported ? "" : "disabled"}>${item.kind === "synthetic" ? "Создать" : "Загрузить"}</button></footer>${!item.supported ? `<p class="dw-note">${esc(item.reason || "Этот формат пока нельзя преобразовать в таблицу.")}</p>` : ""}</article>`).join("")}</div></section>`).join("") || '<div class="dw-empty"><h3>Наборы не найдены</h3><p>Попробуйте другое название или снимите фильтр задачи.</p></div>';
    this.setBusy(this.externalBusy);
  }
  renderScenario() {
    const scenario = SCENARIOS[this.field("scenario").value] || SCENARIOS.realistic;
    this.field("generator").value = scenario.name;
    this.field("n-features").value = scenario.n_features;
    this.field("noise-preset").value = scenario.noise;
    this.field("correlation-preset").value = scenario.correlation === 0.2 ? 0.1 : scenario.correlation;
    this.field("outliers-preset").value = scenario.outliers;
    this.field("noise-exact").value = "";
    this.field("correlation-exact").value = "";
    this.container.querySelector(".dw-scenario-description").textContent = scenario.description;
    this.renderGeneratorNote();
  }
  renderGeneratorNote() {
    const name = this.field("generator").value;
    const catalogue = this.catalogue.find((item) => item.name === name);
    this.container.querySelector(".dw-generator-note").textContent = catalogue?.description || ({ positive: "Цель строго положительная: можно проверить Gamma и Tweedie.", counts: "Цель — неотрицательное число событий: подойдет регрессия Пуассона.", nonlinear: "Добавлен квадрат первого признака: исследуйте кривую в 2D." }[name] || "Цель получается из вкладов признаков и случайного разброса.");
  }
  createSpec() {
    const exact = (key, fallback) => {
      const value = this.field(key).value;
      return value.trim() === "" ? Number(this.field(fallback).value) : Number(value);
    };
    return { kind: "synthetic", name: this.field("generator").value, params: { n_samples: Number(this.field("n-samples").value), n_features: Number(this.field("n-features").value), noise: exact("noise-exact", "noise-preset"), correlation: exact("correlation-exact", "correlation-preset"), outliers: Number(this.field("outliers-preset").value), sparsity: Number(this.field("sparsity").value), scale_spread: Number(this.field("scale-spread").value), seed: Number(this.field("seed").value) } };
  }
  async create() {
    await this.guarded(async () => {
      this.status("Создаем таблицу…");
      const spec = this.createSpec();
      if (!Number.isInteger(spec.params.n_samples) || !Number.isInteger(spec.params.n_features)) throw new Error("Количество строк и признаков должно быть целым.");
      const meta = await this.api("/datasets/load", { method: "POST", body: spec });
      this.setDataset(meta);
      await this.onCreated(meta);
      this.status(`Создан набор «${meta.name}». Исследуйте исходные данные ниже.`);
      this.scrollExplorer();
    });
  }
  async select(id) {
    await this.guarded(async () => {
      this.status("Открываем набор…");
      const meta = await this.api(`/datasets/${encodeURIComponent(id)}`);
      this.setDataset(meta);
      await this.onSelect(meta);
      this.status(`Выбран набор «${meta.name}».`);
      this.scrollExplorer();
    });
  }
  async upload(file) {
    if (!file) return;
    await this.guarded(async () => {
      if (file.size > 25 * 1024 * 1024) throw new Error("Файл больше 25 МБ. Уменьшите таблицу перед загрузкой.");
      this.status(`Загружаем ${file.name}…`);
      const form = new FormData();
      form.append("file", file);
      const meta = await this.api("/datasets/upload", { method: "POST", body: form });
      this.setDataset(meta);
      await this.onCreated(meta);
      this.status(`Файл загружен. Выберите цель и проверьте столбцы.`);
      this.scrollExplorer();
    });
    this.field("upload").value = "";
  }
  async openml() {
    await this.guarded(async () => {
      const id = this.field("openml-id").value.trim(), name = this.field("openml-name").value.trim();
      if (!id && !name) throw new Error("Введите ID или название набора OpenML.");
      if (id && name) throw new Error("Оставьте один способ поиска: ID или название.");
      this.status("Загружаем OpenML. Первый запрос может занять время…");
      const meta = await this.api("/datasets/load", { method: "POST", body: { kind: "openml", name: name || "fetch_openml", params: id ? { data_id: Number(id) } : { name } } });
      this.setDataset(meta);
      await this.onCreated(meta);
      this.status("Набор OpenML добавлен в библиотеку.");
      this.scrollExplorer();
    });
  }
  async loadSource(id) {
    const source = this.catalogue.find((item) => (item.id || item.name) === id);
    if (!source || !source.supported) return;
    if (source.name === "fetch_openml") {
      this.field("openml-id").focus();
      this.status("Для OpenML укажите ID или название выше.");
      return;
    }
    await this.guarded(async () => {
      this.status(source.requires_network ? "Загружаем данные из интернета…" : "Подготавливаем данные…");
      const meta = await this.api("/datasets/load", { method: "POST", body: { kind: source.kind, name: source.name, params: source.kind === "synthetic" ? { n_samples: 180, n_features: source.name === "make_friedman1" ? 5 : 4, seed: 42 } : {} } });
      this.setDataset(meta);
      await this.onCreated(meta);
      this.status(`«${meta.name}» добавлен в библиотеку.`);
      this.scrollExplorer();
    });
  }
  switchTab(tab) {
    if (!this.container.querySelector(`[data-dw-panel="${tab}"]`)) return;
    this.container.querySelectorAll('.dw-help-wrap.is-open').forEach(wrap => this.closeHelp(wrap));
    this.tab = tab;
    this.container.querySelectorAll("[data-dw-panel]").forEach((el) => el.hidden = el.dataset.dwPanel !== tab);
    this.container.querySelectorAll('[role="tab"]').forEach((el) => {
      const selected = el.dataset.dwTab === tab;
      el.setAttribute("aria-selected", String(selected));
      el.tabIndex = selected ? 0 : -1;
    });
    if (tab === "library") this.refreshLibrary();
  }
  scrollExplorer() {
    this.container.querySelector(".dw-explorer")?.scrollIntoView?.({ behavior: window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" });
  }
  renderExplorer() {
    const meta = this.dataset;
    if (!meta) return;
    this.destroyPlots();
    this.exploreSequence++;
    const root = this.container.querySelector(".dw-explorer");
    root.hidden = false;
    const cols = meta.columns || [];
    const all = cols.map((c) => c.name);
    const numeric = cols.filter((c) => c.numeric).map((c) => c.name);
    const target = meta.task_target || meta.default_target || all[all.length - 1] || "";
    const features = numeric.filter((name) => name !== target);
    const x = features[0] || numeric[0] || all[0] || "";
    const second = features[1] || features[0] || numeric[1] || numeric[0] || "";
    const z = numeric.includes(target) ? target : numeric.find((c) => ![x, second].includes(c)) || numeric[numeric.length - 1] || "";
    root.innerHTML = `<div class="dw-explorer-header"><div><div class="dw-eyebrow">ИСХОДНЫЕ ДАННЫЕ · ДО ОБУЧЕНИЯ</div><h2>${esc(meta.name)}</h2><p>${number(meta.rows)} строк · ${cols.length} столбцов · ${esc(TASKS[meta.task] || meta.task || TASKS.other)}</p></div><div class="dw-explorer-actions"><button class="dw-button dw-secondary" data-dw-action="use">Использовать для обучения</button><a class="dw-button dw-secondary" href="/api/datasets/${encodeURIComponent(meta.id)}/export?format=csv" download>Скачать CSV ↓</a><button class="dw-button dw-secondary" data-dw-action="edit">Редактировать строки</button></div></div>
      <div class="dw-role-legend"><span><i style="background:${FEATURE}"></i><strong class="dw-feature">Признак</strong> — сведения для предсказания</span><span><i style="background:${TARGET}"></i><strong class="dw-target">Цель</strong> — ответ, который изучаем</span></div>
      <div class="dw-explorer-controls"><label><span class="dw-target">Цель</span> ${help("target")}<select data-dw="explore-target">${options(all, target)}</select></label><label><span class="dw-feature">Ось X</span> ${help("axes")}<select data-dw="explore-x">${options(all, x)}</select></label><label>Ось Y в 2D<select data-dw="explore-y">${options(all, target)}</select></label><label><span class="dw-feature">Второй признак в 3D</span><select data-dw="explore-second">${options(numeric, second)}</select></label><label>Высота в 3D<select data-dw="explore-z">${options(numeric, z)}</select></label><label>Окрасить по ${help("color")}<select data-dw="explore-color"><option value="">Без окраски</option>${options(all, target)}</select></label><label>Показать точек ${help("sample")}<select data-dw="explore-sample"><option value="500">500 · быстро</option><option value="1000">1 000 · средний обзор</option><option value="2000" selected>2 000 · подробнее</option></select></label></div>
      <p class="dw-explore-description">В 2D: <span class="dw-feature">признак</span> → <span class="dw-target">цель</span>. В 3D: два признака → числовая цель. Цвет показывает выбранный столбец. Наведите на точку, чтобы увидеть исходную строку.</p>
      <div class="dw-explore-state" role="status" aria-live="polite">Подготовка графиков…</div>
      <div class="dw-raw-plots"><article><div class="dw-plot-heading"><h3>Признак и цель · 2D</h3><span>Масштабируйте колесом, выделяйте область</span></div><div class="dw-plot" data-dw-plot="scatter2d"></div></article><article><div class="dw-plot-heading"><h3>Два признака и цель · 3D</h3><span>Поворачивайте мышью</span></div><div class="dw-plot" data-dw-plot="scatter3d"></div></article><article><div class="dw-plot-heading"><h3>Распределения ${help("histogram")}</h3><span>Где сосредоточены значения</span></div><div class="dw-plot" data-dw-plot="histogram"></div></article><article><div class="dw-plot-heading"><h3>Связи столбцов ${help("correlation_matrix")}</h3><span>Корреляция числовых значений</span></div><div class="dw-plot" data-dw-plot="correlation"></div></article></div>
      <div class="dw-profile"></div><details class="dw-preview"><summary>Первые строки таблицы · оригинальные значения</summary><div class="dw-table-scroll">${this.previewMarkup(meta)}</div></details><details class="dw-metadata"><summary>Название, задача и метки набора</summary><div class="dw-metadata-grid"><label>Название<input data-dw="metadata-name" value="${esc(meta.name)}" maxlength="200"></label><label>Задача ${help("task")}<select data-dw="metadata-task">${options(METADATA_TASKS.map((value) => ({ value, label: TASKS[value] })), METADATA_TASKS.includes(meta.task) ? meta.task : "other")}</select></label><label>Описание<textarea data-dw="metadata-description" maxlength="2000">${esc(meta.description || "")}</textarea></label><label>Метки через запятую<input data-dw="metadata-tags" value="${esc((meta.tags || []).join(", "))}" placeholder="жилье, учебный пример"></label></div><div class="dw-metadata-footer"><button class="dw-button dw-secondary" data-dw-action="metadata">Сохранить описание</button><button class="dw-button dw-danger" data-dw-action="delete-request">Удалить набор…</button></div><div class="dw-delete-confirm" hidden><p>Удалить «${esc(meta.name)}» из библиотеки? Набор, используемый сохраненным экспериментом, удалить нельзя. Остальные снимки данных останутся.</p><button class="dw-button dw-danger" data-dw-action="delete">Да, удалить этот набор</button><button class="dw-button dw-secondary" data-dw-action="delete-cancel">Оставить</button></div></details>`;
    this.associateControls(root);
    this.renderAxisRoles();
    this.setBusy(this.externalBusy);
    this.explore();
  }
  previewMarkup(meta) {
    const cols = (meta.columns || []).slice(0, 15);
    const target = meta.task_target || meta.default_target;
    return `<table><caption>${Math.min(meta.preview?.length || 0, 20)} первых строк${meta.columns.length > 15 ? " · показаны первые 15 столбцов" : ""}</caption><thead><tr><th scope="col">Строка</th>${cols.map((c) => `<th scope="col" class="${c.name === target ? "dw-target" : "dw-feature"}">${esc(c.name)}</th>`).join("")}</tr></thead><tbody>${(meta.preview || []).slice(0, 20).map((row, index) => `<tr><th scope="row">${index + 1}</th>${cols.map((c) => `<td>${row[c.name] == null ? '<span class="dw-missing">пропуск</span>' : esc(typeof row[c.name] === "number" ? number(row[c.name]) : row[c.name])}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
  }
  renderAxisRoles() {
    const target = this.field("explore-target")?.value;
    for (const key of ["explore-x", "explore-y", "explore-second", "explore-z", "explore-color"]) {
      const el = this.field(key);
      if (el) {
        el.classList.toggle("dw-axis-target", el.value === target);
        el.classList.toggle("dw-axis-feature", el.value !== target && !!el.value);
      }
    }
  }
  async explore() {
    if (!this.dataset || !this.container) return;
    const seq = ++this.exploreSequence;
    const id = this.dataset.id;
    const values = { x: this.field("explore-x").value, y: this.field("explore-y").value, z: this.field("explore-z").value, color: this.field("explore-color").value, target: this.field("explore-target").value, sample_size: this.field("explore-sample").value };
    const query = new URLSearchParams(Object.entries(values).filter(([, value]) => value !== ""));
    const state = this.container.querySelector(".dw-explore-state");
    state.classList.remove("is-error");
    state.textContent = "Рассчитываем обзор исходных значений…";
    try {
      const payload = await this.api(`/datasets/${encodeURIComponent(id)}/explore?${query}`);
      if (seq !== this.exploreSequence || id !== this.dataset.id) return;
      this.exploration = payload;
      this.renderExploration(payload, values);
      const indices = payload.points?.indices || [];
      const sample = indices.length || payload.sample_size || payload.sample_rows || 0;
      state.textContent = `На графике: ${number(sample)} из ${number(this.dataset.rows)} строк. ${payload.note || ""}`;
      this.renderProfile(payload.profile || payload.stats || {}, payload.columns || this.dataset.columns);
    } catch (error) {
      if (seq !== this.exploreSequence) return;
      state.textContent = error.message;
      state.classList.add("is-error");
      this.onError(error);
    }
  }
  plotNode(key) {
    return this.container.querySelector(`[data-dw-plot="${key}"]`);
  }
  layout(extra = {}) {
    const light = document.body.dataset.theme === "light";
    return { paper_bgcolor: "transparent", plot_bgcolor: "transparent", font: { family: "Inter, system-ui, sans-serif", size: 14, color: light ? "#283d49" : "#c3d2dd" }, margin: { l: 60, r: 35, t: 15, b: 55 }, autosize: true, hovermode: "closest", uirevision: this.dataset?.id, ...extra };
  }
  roleColor(color = FEATURE) {
    if (this.container.ownerDocument.body.dataset.theme !== 'light') return color;
    return color === FEATURE ? '#076950' : color === TARGET ? '#a63d2a' : color;
  }
  axis(label, color) {
    return { title: { text: label, font: { color: this.roleColor(color || FEATURE), size: 14 } }, gridcolor: document.body.dataset.theme === "light" ? "#dee8ee" : "#2a3a49", zerolinecolor: "#41536a", automargin: true };
  }
  refreshTheme() {
    if (!window.Plotly?.relayout || !this.container) return Promise.resolve();
    const target = this.field('explore-target')?.value;
    const grid = this.container.ownerDocument.body.dataset.theme === 'light' ? '#dee8ee' : '#2a3a49';
    const updates = [...this.container.querySelectorAll('[data-dw-plot]')].filter(node => node.data?.length).map(node => {
      const patch = {'font.color': this.layout().font.color, 'font.size': 14};
      for (const key of ['xaxis', 'yaxis', 'xaxis2', 'yaxis2', 'scene.xaxis', 'scene.yaxis', 'scene.zaxis']) {
        const axis = key.split('.').reduce((value, part) => value?.[part], node.layout);
        if (!axis) continue;
        patch[`${key}.gridcolor`] = grid;
        if (axis.title?.font?.color) patch[`${key}.title.font.color`] = this.roleColor(axis.title.text === target ? TARGET : FEATURE);
      }
      const operations = [window.Plotly.relayout(node, patch)];
      if (window.Plotly.restyle) node.data.forEach((trace, index) => {
        const title = trace.marker?.colorbar?.title;
        if (title?.font?.color) operations.push(window.Plotly.restyle(node, {'marker.colorbar.title.font.color': this.roleColor(title.text === target ? TARGET : FEATURE)}, [index]));
      });
      return Promise.all(operations);
    });
    return Promise.all(updates);
  }
  plot(key, traces, layout) {
    const node = this.plotNode(key);
    if (!window.Plotly) {
      node.innerHTML = '<p class="dw-empty">Графическая библиотека еще загружается. Измените выбор осей, чтобы повторить.</p>';
      return;
    }
    window.Plotly.react(node, traces, this.layout(layout), { responsive: true, displaylogo: false, locale: "ru", modeBarButtonsToRemove: ["lasso2d", "select2d"], toImageButtonOptions: { format: "png", filename: "исходные-данные", scale: 2 } });
    if (!node._datasetWorkspaceClick && typeof node.on === "function") {
      node.on("plotly_click", (event) => {
        const index = event.points?.[0]?.customdata?.[0];
        if (Number.isInteger(index) && !this.externalBusy) this.onEdit({ dataset: this.dataset, index, originalIndex: index, source: "plot", point: event.points[0] });
      });
      node._datasetWorkspaceClick = true;
    }
  }
  renderExploration(payload, values) {
    const p = payload.points || {};
    const indices = p.indices || [];
    const xs = p.x || [], ys = p.y || [], zs = p.z || [], colors = p.color || [];
    const target = values.target;
    const isNumericColor = (p.color_kind === "numeric" || p.color_kind === "continuous" || this.dataset.columns.find((c2) => c2.name === values.color)?.numeric) && values.color;
    const valid = xs.map((_, i) => i).filter((i) => xs[i] != null && ys[i] != null);
    const makeTrace = (indexes, name, color) => ({ type: "scattergl", mode: "markers", name, x: indexes.map((i) => xs[i]), y: indexes.map((i) => ys[i]), customdata: indexes.map((i) => [Number(indices[i] ?? i), colors[i]]), hovertemplate: `Строка %{customdata[0]}<br>${esc(values.x)}: %{x}<br>${esc(values.y)}: %{y}${values.color ? "<br>" + esc(values.color) + ": %{customdata[1]}" : ""}<extra></extra>`, marker: { size: 7, opacity: 0.76, color, line: { width: 0 } } });
    let traces;
    if (values.color && !isNumericColor) {
      const categories = p.color_categories?.length ? [...p.color_categories, ...colors.includes(null) ? ["Пропуск"] : []] : [...new Set(colors.map((value) => String(value ?? "Пропуск")))];
      traces = categories.map((category, index) => makeTrace(valid.filter((i) => String(colors[i] ?? "Пропуск") === category), category, CATEGORY_COLORS[index % CATEGORY_COLORS.length]));
    } else {
      const markerColor = isNumericColor ? valid.map((i) => colors[i]) : FEATURE;
      const trace = makeTrace(valid, values.color || "Наблюдения", markerColor);
      if (isNumericColor) Object.assign(trace.marker, { colorscale: [[0, "#79b9ef"], [0.5, "#b2d7ca"], [1, TARGET]], showscale: true, colorbar: { title: { text: values.color, font: { color: this.roleColor(values.color === target ? TARGET : FEATURE), size: 14 } }, thickness: 12 } });
      traces = [trace];
    }
    this.plot("scatter2d", traces, { xaxis: this.axis(values.x, values.x === target ? TARGET : FEATURE), yaxis: this.axis(values.y, values.y === target ? TARGET : FEATURE), legend: { orientation: "h", y: -0.25 }, margin: { l: 65, r: isNumericColor ? 70 : 25, t: 10, b: 70 } });
    this.render3D(payload, values);
    const hist = this.normalizeHistograms(payload.histograms, values, p);
    const htraces = hist.map((item, index) => item.counts ? { type: "bar", name: item.name, x: item.centers || item.labels || [], y: item.counts, ...item.width ? { width: item.width } : {}, marker: { color: item.name === target ? TARGET : FEATURE }, xaxis: index ? "x2" : "x", yaxis: index ? "y2" : "y" } : { type: "histogram", name: item.name, x: item.values, marker: { color: item.name === target ? TARGET : FEATURE }, xaxis: index ? "x2" : "x", yaxis: index ? "y2" : "y" });
    this.plot("histogram", htraces, { grid: { rows: 1, columns: Math.max(1, hist.length), pattern: "independent" }, xaxis: this.axis(hist[0]?.name || values.x, hist[0]?.name === target ? TARGET : FEATURE), xaxis2: this.axis(hist[1]?.name || values.y, hist[1]?.name === target ? TARGET : FEATURE), yaxis: { title: "Наблюдений", gridcolor: "#2a3a49" }, yaxis2: { title: "Наблюдений", gridcolor: "#2a3a49" }, showlegend: false, bargap: 0.08 });
    const c = payload.correlation || {};
    const names = c.columns || c.features || c.names || c.labels || [];
    const matrix = c.matrix || c.values || c.data || [];
    if (names.length && matrix.length) this.plot("correlation", [{ type: "heatmap", x: names, y: names, z: matrix, zmin: -1, zmax: 1, zmid: 0, colorscale: [[0, "#6d9fe2"], [0.5, "#202e3b"], [1, TARGET]], colorbar: { title: "r", thickness: 12 }, hovertemplate: "%{x} ↔ %{y}<br>Корреляция: %{z:.3f}<extra></extra>" }], { xaxis: { tickangle: -35, automargin: true }, yaxis: { automargin: true }, margin: { l: 95, r: 55, t: 10, b: 80 } });
    else this.plotNode("correlation").innerHTML = '<p class="dw-empty">Для корреляции нужны хотя бы два числовых столбца с непостоянными значениями.</p>';
  }
  async render3D(payload, values) {
    const x = values.x, second = this.field("explore-second").value, z = values.z, target = values.target;
    const nums = this.dataset.columns.filter((c) => c.numeric).map((c) => c.name);
    if (!nums.includes(x) || !nums.includes(second) || !nums.includes(z)) {
      this.plotNode("scatter3d").innerHTML = '<p class="dw-empty">Для 3D выберите три числовые оси. Категориальную цель удобно показывать цветом.</p>';
      return;
    }
    if (new Set([x, second, z]).size < 3) {
      this.plotNode("scatter3d").innerHTML = '<p class="dw-empty">Для 3D нужны три разных числовых столбца: два признака и цель. В этом наборе используйте 2D или создайте пример с двумя признаками.</p>';
      return;
    }
    const id = this.dataset.id, seq = this.exploreSequence;
    let result = payload;
    if (second !== values.y) {
      try {
        const query = new URLSearchParams({ ...values, y: second });
        result = await this.api(`/datasets/${encodeURIComponent(id)}/explore?${query}`);
        if (seq !== this.exploreSequence || this.dataset.id !== id) return;
      } catch (error) {
        this.plotNode("scatter3d").innerHTML = `<p class="dw-empty">${esc(error.message)}</p>`;
        return;
      }
    }
    const p = result.points || {}, indices = p.indices || [], colors = p.color || [], numericColor = this.dataset.columns.find((c) => c.name === values.color)?.numeric;
    const valid = (p.x || []).map((_, i) => i).filter((i) => p.x[i] != null && p.y?.[i] != null && p.z?.[i] != null);
    const make = (subset, name, color) => ({ type: "scatter3d", mode: "markers", name, x: subset.map((i) => p.x[i]), y: subset.map((i) => p.y[i]), z: subset.map((i) => p.z[i]), customdata: subset.map((i) => [Number(indices[i] ?? i), colors[i]]), marker: { size: 3.6, opacity: 0.8, color }, hovertemplate: `Строка %{customdata[0]}<br>${esc(x)}: %{x}<br>${esc(second)}: %{y}<br>${esc(z)}: %{z}${values.color ? "<br>" + esc(values.color) + ": %{customdata[1]}" : ""}<extra></extra>` });
    let traces;
    if (values.color && !numericColor) {
      const categories = p.color_categories?.length ? [...p.color_categories, ...colors.includes(null) ? ["Пропуск"] : []] : [...new Set(colors.map((value) => String(value ?? "Пропуск")))];
      traces = categories.map((name, j) => make(valid.filter((i) => String(colors[i] ?? "Пропуск") === name), name, CATEGORY_COLORS[j % CATEGORY_COLORS.length]));
    } else {
      const trace = make(valid, values.color || "Наблюдения", numericColor ? valid.map((i) => colors[i]) : FEATURE);
      if (numericColor && values.color) Object.assign(trace.marker, { colorscale: [[0, "#79b9ef"], [0.5, "#b2d7ca"], [1, TARGET]], showscale: true, colorbar: { title: values.color, thickness: 12 } });
      traces = [trace];
    }
    this.plot("scatter3d", traces, { scene: { xaxis: this.axis(x, x === target ? TARGET : FEATURE), yaxis: this.axis(second, second === target ? TARGET : FEATURE), zaxis: this.axis(z, z === target ? TARGET : FEATURE), bgcolor: "transparent", camera: { eye: { x: 1.4, y: 1.5, z: 1 } } }, legend: { orientation: "h", y: -0.12 }, margin: { l: 0, r: numericColor ? 50 : 0, t: 0, b: 25 } });
  }
  normalizeHistograms(histograms, values, points) {
    const names = [values.x, values.y].filter((name, index, all) => name && all.indexOf(name) === index);
    return names.map((name) => {
      const item = Array.isArray(histograms) ? histograms.find((h) => (h.name || h.column) === name) : histograms?.[name];
      if (item) {
        const counts = item.counts || item.y;
        const edges = item.bin_edges || item.edges;
        return { name, counts, centers: item.centers || item.x || (edges ? edges.slice(0, -1).map((v, i) => (v + edges[i + 1]) / 2) : null), labels: item.labels || item.categories, values: item.values, width: edges?.length === 2 && edges[0] === edges[1] ? Math.max(Math.abs(edges[0]) * 0.02, 1) : null };
      }
      return { name, values: name === values.x ? points.x : points.y };
    });
  }
  renderProfile(profile, columns) {
    const list = Array.isArray(profile) ? profile : Array.isArray(profile.columns) ? profile.columns : columns.map((column) => ({ ...column, ...profile[column.name] || {} }));
    const target = this.field("explore-target").value;
    const rowCount = this.dataset.rows;
    this.container.querySelector(".dw-profile").innerHTML = `<div class="dw-profile-heading"><h3>Паспорт столбцов</h3><p>Типы, пропуски и разброс исходных значений. Подготовка данных выполняется отдельно внутри обучения.</p></div><div class="dw-table-scroll"><table><thead><tr><th scope="col">Столбец</th><th scope="col">Роль</th><th scope="col">Тип</th><th scope="col">Пропусков</th><th scope="col">Разных значений</th><th scope="col">Минимум</th><th scope="col">Медиана</th><th scope="col">Максимум</th></tr></thead><tbody>${list.map((c) => {
      const stats = c.stats || c;
      const missing = c.missing ?? stats.missing ?? 0;
      return `<tr><th scope="row" class="${c.name === target ? "dw-target" : "dw-feature"}">${esc(c.name)}</th><td>${c.name === target ? "Цель" : "Признак"}</td><td>${c.numeric ? "Число" : "Категория"}</td><td>${number(missing)}${rowCount ? ` <span class="dw-muted">(${number(100 * missing / rowCount)}%)</span>` : ""}</td><td>${number(c.unique ?? stats.unique ?? stats.nunique)}</td><td>${number(stats.min)}</td><td>${number(stats.median ?? stats.quantiles?.q50 ?? stats["50%"])}</td><td>${number(stats.max)}</td></tr>`;
    }).join("")}</tbody></table></div>`;
  }
  async saveMetadata() {
    await this.guarded(async () => {
      const name = this.field("metadata-name").value.trim();
      if (!name) throw new Error("Введите название набора.");
      const body = { name, task: this.field("metadata-task").value, description: this.field("metadata-description").value.trim(), tags: this.field("metadata-tags").value.split(",").map((tag) => tag.trim()).filter(Boolean), task_target: this.field("explore-target").value };
      if (this.dataset.targets?.includes(body.task_target)) body.default_target = body.task_target;
      const meta = await this.api(`/datasets/${encodeURIComponent(this.dataset.id)}/metadata`, { method: "PATCH", body });
      this.setDataset(meta);
      await this.onSelect(meta);
      this.status("Сохранена новая версия с описанием и выбранной целью. Предыдущая версия осталась в библиотеке.");
    });
  }
  async deleteDataset() {
    await this.guarded(async () => {
      const id = this.dataset.id;
      await this.api(`/datasets/${encodeURIComponent(id)}`, { method: "DELETE" });
      this.destroyPlots();
      this.dataset = null;
      this.exploreSequence++;
      this.container.querySelector(".dw-explorer").hidden = true;
      await this.refreshLibrary();
      await this.onDeleted(id);
      this.status("Набор удален из библиотеки.");
    });
  }
  handleClick(event) {
    const el = event.target.closest("button,a");
    if (!el || !this.container.contains(el)) return;
    if (el.classList.contains("dw-help")) {
      event.preventDefault();
      const wrap = el.closest(".dw-help-wrap");
      if (wrap.classList.contains('is-pinned')) this.closeHelp(wrap, true);
      else this.openHelp(wrap, true);
      return;
    }
    if (el.classList.contains('dw-help-close')) {
      event.preventDefault();
      this.closeHelp(el.closest('.dw-help-wrap'), true);
      return;
    }
    if (el.dataset.dwHelp) {
      event.preventDefault();
      this.closeHelp(el.closest('.dw-help-wrap'));
      this.onHelp(el.dataset.dwHelp);
      return;
    }
    if (el.dataset.dwTab) {
      this.switchTab(el.dataset.dwTab);
      return;
    }
    if (el.dataset.dwSelect) {
      this.select(el.dataset.dwSelect);
      return;
    }
    if (el.dataset.dwSource) {
      this.loadSource(el.dataset.dwSource);
      return;
    }
    const action = el.dataset.dwAction;
    if (action === "refresh") this.refreshLibrary();
    if (action === "create") this.create();
    if (action === "openml") this.openml();
    if (action === "edit" && !this.externalBusy) this.onEdit({ dataset: this.dataset, source: "table" });
    if (action === "use" && !this.externalBusy) {
      const target = this.field("explore-target").value;
      this.onSelect({ ...this.dataset, default_target: this.dataset.targets?.includes(target) ? target : this.dataset.default_target, task_target: target }, { intent: "train" });
    }
    if (action === "metadata") this.saveMetadata();
    if (action === "delete-request") this.container.querySelector(".dw-delete-confirm").hidden = false;
    if (action === "delete-cancel") this.container.querySelector(".dw-delete-confirm").hidden = true;
    if (action === "delete") this.deleteDataset();
  }
  handleChange(event) {
    const key = event.target.dataset.dw;
    if (!key) return;
    if (key === "scenario") this.renderScenario();
    if (key === "generator") this.renderGeneratorNote();
    if (key === "upload") this.upload(event.target.files[0]);
    if (key.startsWith("library-")) {
      if (key === "library-sort") this.renderLibrary();
      else this.refreshLibrary();
    }
    if (key.startsWith("source-")) this.renderSources();
    if (key.startsWith("explore-")) {
      if (key === "explore-target") {
        const target = event.target.value;
        this.field("explore-y").value = target;
        this.field("explore-color").value = target;
        if (this.dataset.columns.find((c) => c.name === target)?.numeric) this.field("explore-z").value = target;
      }
      this.renderAxisRoles();
      clearTimeout(this.exploreTimer);
      this.exploreTimer = setTimeout(() => this.explore(), 120);
    }
  }
  handleInput(event) {
    const key = event.target.dataset.dw;
    if (key === "library-search") {
      clearTimeout(this.searchTimer);
      this.searchTimer = setTimeout(() => this.refreshLibrary(), 200);
    }
    if (key === "source-search") this.renderSources();
  }
  setBusy(busy) {
    this.externalBusy = Boolean(busy);
    if (!this.container) return this;
    this.container.querySelectorAll('[data-dw-select],[data-dw-source],[data-dw-action="create"],[data-dw-action="openml"],[data-dw-action="use"],[data-dw-action="edit"],[data-dw-action="metadata"],[data-dw-action="delete"],[data-dw-action="delete-request"],[data-dw="upload"]').forEach((el) => {
      const unavailable = el.dataset.dwSource && this.catalogue.find((item) => (item.id || item.name) === el.dataset.dwSource)?.supported === false;
      el.disabled = this.externalBusy || this.busy || Boolean(unavailable);
    });
    return this;
  }
  setTarget(target) {
    if (!this.container || !this.dataset?.columns.some((c) => c.name === target)) return this;
    this.field("explore-target").value = target;
    this.field("explore-y").value = target;
    this.field("explore-color").value = target;
    if (this.dataset.columns.find((c) => c.name === target)?.numeric) this.field("explore-z").value = target;
    this.renderAxisRoles();
    this.explore();
    return this;
  }
  resize() {
    if (!window.Plotly || !this.container?.offsetWidth) return;
    this.container.querySelectorAll("[data-dw-plot]").forEach((node) => {
      if (node.data?.length) window.Plotly.Plots.resize(node);
    });
  }
  destroyPlots() {
    if (window.Plotly) this.container?.querySelectorAll("[data-dw-plot]").forEach((node) => window.Plotly.purge(node));
  }
  destroy() {
    clearTimeout(this.searchTimer);
    clearTimeout(this.exploreTimer);
    this.exploreSequence++;
    this.librarySequence++;
    this.observer?.disconnect();
    this.container?.ownerDocument.removeEventListener('pointerdown', this.onOutsideHelp);
    this.destroyPlots();
  }
}
export {
  DatasetWorkspace
};
