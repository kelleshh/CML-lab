/**
 * Cross-validation and parameter search controls.
 *
 * This module only builds validated requests and displays server reports. It
 * never approximates model quality, fits a model, or selects a winner locally.
 */

const CV_METHODS = [
  {id: 'none', name: 'Одна проверочная часть · без CV'},
  {id: 'kfold', name: 'KFold · проверять каждую часть по очереди'},
  {id: 'repeated_kfold', name: 'Repeated KFold · повторить разбиение'},
  {id: 'shuffle_split', name: 'ShuffleSplit · случайные проверочные части'},
  {id: 'timeseries', name: 'TimeSeriesSplit · прошлое → будущее'},
  {id: 'group_kfold', name: 'Group KFold · целые группы отдельно'},
  {id: 'group_shuffle_split', name: 'Group ShuffleSplit · случайные группы'},
  {id: 'leave_one_group_out', name: 'Leave One Group Out · оставить одну группу'},
  {id: 'leave_one_out', name: 'Leave One Out · оставить одну строку'},
  {id: 'stratified_bins', name: 'Диапазоны цели · выравнивать разрезы'},
];
const SEARCH_METHODS = [
  {id: 'optuna_tpe', name: 'Optuna TPE · использовать прошлые результаты'},
  {id: 'random', name: 'Случайный поиск · пробовать разные настройки'},
  {id: 'grid', name: 'Сетка · проверить все заданные варианты'},
  {id: 'optuna_random', name: 'Optuna Random · случайные испытания'},
  {id: 'halving_grid', name: 'Сокращение сетки · дать больше данных лучшим'},
  {id: 'halving_random', name: 'Сокращение случайных кандидатов'},
];
const GROUP_METHODS = new Set(['group_kfold', 'group_shuffle_split', 'leave_one_group_out']);
const ESCAPES = {'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'};
const escape = value => String(value ?? '').replace(/[&<>"']/g, character => ESCAPES[character]);
const format = value => value == null || !Number.isFinite(Number(value)) ? '—' : new Intl.NumberFormat('ru', {maximumFractionDigits: 5}).format(Number(value));
const numeric = value => Number.isFinite(Number(value)) && value !== '';
const clone = value => value == null ? value : structuredClone(value);
const finite = value => typeof value === 'number' && Number.isFinite(value);
const option = (value, label, selected) => `<option value="${escape(value)}"${String(value) === String(selected) ? ' selected' : ''}>${escape(label)}</option>`;
const options = (values, selected) => values.map(item => option(item.id ?? item.value, item.name ?? item.label, selected)).join('');

const HELP = {
  strategy: ['Как проверять качество', 'CV, или перекрестная проверка, несколько раз обучает модель на одной части обучающих строк и проверяет на другой. Среднее показывает обычное качество, разброс — зависимость от разбиения. Для независимых строк подходит KFold. Повторные наблюдения одного объекта делите по группам, данные во времени — по хронологии.', '26-cross-validation', 'https://scikit-learn.org/stable/modules/cross_validation.html'],
  folds: ['Число проверочных разрезов', 'В KFold с пятью частями данные разбивают на пять блоков. Каждый блок один раз служит проверкой; для каждой проверки модель обучается заново на остальных блоках. В ShuffleSplit это число случайных разрезов. Больше частей требует больше обучений; допустимо от 2 до 20.', '26-cross-validation', 'https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html'],
  repeats: ['Повторить KFold с новым разбиением', 'При 5 частях и 3 повторах модель обучится 15 раз. Каждый повтор перемешивает обучающие строки заново. Это помогает увидеть, насколько результат зависит от случайного разбиения, но не делает связанные строки независимыми.', '26-cross-validation', 'https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RepeatedKFold.html'],
  shuffle: ['Перемешивание внутри CV', 'Перемешивание распределяет независимые объекты случайно по частям. Для временных рядов оно выключено: будущие ответы не должны помогать предсказывать прошлое. В GroupKFold перемешиваются группы, а не отдельные наблюдения.', '26-cross-validation', 'https://scikit-learn.org/stable/modules/cross_validation.html'],
  fraction: ['Размер проверочной части', 'В ShuffleSplit значение 20% означает: на каждом разрезе примерно 20% внешнего обучения откладывается для проверки. Проверочные части разных разрезов могут пересекаться. В GroupShuffleSplit доля относится к числу групп, поэтому доля строк может отличаться.', '26-cross-validation', 'https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html'],
  group: ['Какой объект нельзя разрывать', 'Укажите колонку с идентификатором пациента, устройства, покупателя или другого объекта, у которого несколько строк. Все наблюдения одного объекта попадут по одну сторону разделения. Колонка группы автоматически исключается из признаков модели. Цель не может быть идентификатором группы; пропуски не допускаются.', '26-cross-validation', 'https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html'],
  gap: ['Разрыв между прошлым и проверкой', 'Это число строк, которые пропускаются между концом обучения и началом проверки внутри каждого разреза CV. Например, разрыв 7 оставляет семь наблюдений между прошлым и будущим. Он полезен при задержке появления ответа. Между внешними train, validation и test дополнительного разрыва нет. Строки должны уже идти по времени; числу строк нельзя приписывать дни без знания частоты измерений.', '26-cross-validation', 'https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html'],
  window: ['Сколько прошлого помнить', 'Пустое поле означает растущее окно: модель получает все доступное прошлое. Число 100 ограничивает обучение последними 100 строками перед проверкой. Так можно сравнить длинную историю с короткой, когда зависимость со временем меняется.', '26-cross-validation', 'https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html'],
  horizon: ['Сколько будущих строк проверять', 'Пустое поле оставляет стандартный размер TimeSeriesSplit. Число 20 задает двадцать следующих наблюдений на каждый проверочный разрез. Убедитесь, что набора хватит на все разрезы и разрывы. Порядок исходных строк — порядок времени.', '26-cross-validation', 'https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html'],
  bins: ['Выравнивать диапазоны цели', 'Числовую цель внутри внешнего обучения разбивают по квантилям. Части CV получают примерно одинаковую долю каждого диапазона. Это экспериментальная стратификация для регрессии: она не заменяет разделение по времени или группам. В каждом диапазоне должно хватить строк на все части.', '26-cross-validation', 'https://scikit-learn.org/stable/modules/cross_validation.html#stratification'],
  parallel: ['Одновременно обучать несколько разрезов', 'Работники — параллельно выполняемые обучения. Доступно от одного до четырех. Внутренние библиотеки ограничиваются одним потоком, чтобы вложенная параллельность не перегружала компьютер. Чем больше работников, тем больше расход памяти. В Optuna предложения идут последовательно для воспроизводимости; разрезы проверяются параллельно.', '26-cross-validation', 'https://scikit-learn.org/stable/computing/parallelism.html'],
  method: ['Как искать настройки', 'Сетка проверяет каждую комбинацию из списка. Случайный поиск выбирает комбинации из диапазонов. Optuna TPE использует оценки прошлых попыток, чтобы предложить следующие. Последовательное сокращение сначала дает кандидатам меньше обучающих строк, затем увеличивает объем у лучших. Победителя выбирает сервер по CV внутри внешнего обучения.', '27-hyperparameter-search', 'https://scikit-learn.org/stable/modules/grid_search.html'],
  trials: ['Сколько вариантов можно попробовать', 'Одна попытка — одна комбинация параметров. При пяти частях CV двадцать попыток требуют до ста обучений плюс переобучение победителя. Максимум 100 попыток и 500 обучений CV. Сетка не обрывается на произвольном кандидате: число ее комбинаций должно помещаться в бюджет.', '27-hyperparameter-search', 'https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study.optimize'],
  metric: ['По чему выбирать победителя', 'Выберите одну главную метрику поиска. RMSE измеряет типичный размер ошибки в единицах цели и сильнее реагирует на большие промахи. MAE измеряет средний модуль ошибки. Для R² обычно больше лучше. Остальные выбранные метрики сохраняются в отчете; выбирать победителя одновременно по «всем метрикам» без отдельного правила нельзя.', '20-metrics-experiment', 'https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics'],
  direction: ['Какой результат лучше', 'Для ошибок MAE, MSE и RMSE обычно меньше лучше. Для R² больше лучше. Автоматическое направление берется из каталога метрик. Для собственной формулы укажите направление явно: например, mean(abs(error)) нужно уменьшать, а -mean(abs(error)) увеличивать.', '20-metrics-experiment', 'https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.create_study.html'],
  factor: ['Насколько сокращать кандидатов', 'Фактор 3 оставляет примерно треть кандидатов на следующий этап и увеличивает число обучающих строк. Оценки разных этапов получены на разных объемах данных: их нельзя сравнивать как оценки на одинаковой проверке. Победитель выбирается только на последнем этапе.', '27-hyperparameter-search', 'https://scikit-learn.org/stable/modules/grid_search.html#searching-for-optimal-parameters-with-successive-halving'],
  space: ['Какие параметры разрешено менять', 'Отметьте параметры для поиска. Список — конкретные значения; диапазон — нижняя и верхняя границы. Логарифмический диапазон ищет по порядкам величин, например 0.001 → 0.01 → 0.1. Для сетки диапазон превращается в указанное число точек. Остальные параметры сохраняют текущие значения модели. Числа в списке разделяйте точкой с запятой, десятичную часть записывайте через точку.', '27-hyperparameter-search', 'https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html'],
};

function integer(input, title, low, high) {
  const value = Number(input?.value);
  if (!Number.isInteger(value) || value < low || value > high) throw new Error(`${title}: нужно целое число от ${low} до ${high}.`);
  return value;
}

function displayParameter(value) {
  return typeof value === 'boolean' ? (value ? 'да' : 'нет') : typeof value === 'number' ? format(value) : String(value ?? '');
}

function boundedValues(values, spec) {
  return [...new Set(values.filter(value => finite(value) && (spec.min == null || value >= spec.min) && (spec.max == null || value <= spec.max)).map(value => spec.type === 'int' ? Math.round(value) : value))];
}

function defaultDefinition(spec, model, featureCount) {
  if (spec.type === 'bool') return {enabled: false, kind: 'list', choices: [true, false]};
  if (spec.type === 'select') return {enabled: false, kind: 'list', choices: clone(spec.options || [spec.default])};
  let values;
  if (spec.key === 'alpha') values = model.id === 'ridge' ? [.001, .01, .1, 1, 10, 100] : [.0001, .001, .01, .1, 1, 10];
  else if (spec.key === 'l1_ratio') values = [.1, .5, .9];
  else if (spec.key === 'quantile') values = [.1, .5, .9];
  else if (spec.key === 'epsilon' && model.id === 'huber') values = [1.05, 1.35, 2];
  else if (spec.key === 'n_nonzero_coefs') values = [1, 2, 3, 5].filter(value => value <= Math.max(1, featureCount));
  else if (spec.key === 'tol') values = [1e-6, 1e-4, .01];
  else if (spec.key === 'max_iter') values = [100, 500, 2000];
  else if (spec.key === 'max_trials') values = [20, 50, 100];
  else if (spec.key === 'group_size') values = [1, 2, 4].filter(value => value <= Math.max(1, featureCount));
  else if (spec.key === 'min_samples' || spec.key === 'residual_threshold') values = [0, 2, 5];
  else if (spec.key === 'power') values = [0, 1, 1.5, 2];
  else {
    const base = finite(spec.default) ? spec.default : 1;
    values = base === 0 ? [0, .1, 1] : base > 0 ? [base / 3, base, base * 3] : [base * 3, base, base / 3];
  }
  values = boundedValues(values, spec);
  if (!values.length) values = [Number(spec.default ?? spec.min ?? 0)];
  const positive = values.filter(value => value > 0);
  const low = Math.min(...(spec.key === 'alpha' && positive.length ? positive : values));
  let high = Math.max(...values);
  if (low === high) high = Math.min(spec.max ?? high + 1, high + (spec.type === 'int' ? 1 : Math.max(Math.abs(high), 1)));
  return {enabled: false, kind: 'list', choices: values, low, high, log: low > 0 && high / low >= 50, points: 3};
}

/** One instance per laboratory; external callbacks own the actual job lifecycle. */
export class ValidationWorkspace {
  constructor({getRequest, onChange, onSearch, onError, openLesson} = {}) {
    this.getRequest = getRequest || (() => ({}));
    this.onChange = onChange || (() => {});
    this.onSearch = onSearch || (() => {});
    this.onError = onError || (() => {});
    this.openLesson = openLesson || (() => {});
    this.catalogue = {};
    this.dataset = null;
    this.model = null;
    this.space = new Map();
    this.busy = false;
    this.progressRecords = [];
    this.result = null;
    this.listeners = [];
  }

  mount(container) {
    this.container = typeof container === 'string' ? document.querySelector(container) : container;
    if (!this.container) throw new Error('Не найден контейнер проверки и поиска.');
    this.doc = this.container.ownerDocument;
    this.container.classList.add('validation-workspace');
    this.container.innerHTML = `
      <section class="vw-section" aria-label="Перекрестная проверка">
        <label>Схема проверки<select id="vw-cv-method">${options(CV_METHODS, 'none')}</select></label>
        <div class="vw-quick" role="group" aria-label="Готовые схемы проверки">
          <button type="button" data-vw-cv="kfold">Независимые строки</button>
          <button type="button" data-vw-cv="timeseries">По времени</button>
          <button type="button" data-vw-cv="group_kfold">По группам</button>
        </div>
        <p class="vw-note" id="vw-cv-description"></p>
        <div data-vw-field="folds" class="vw-field"><label>Число частей / разрезов<input id="vw-cv-folds" type="number" min="2" max="20" step="1" value="5"></label><div class="vw-quick vw-small" role="group" aria-label="Выбрать число частей"><button type="button" data-vw-folds="3">3</button><button type="button" data-vw-folds="5">5</button><button type="button" data-vw-folds="10">10</button></div></div>
        <label data-vw-field="repeats">Повторить разбиение<input id="vw-cv-repeats" type="number" min="1" max="10" step="1" value="2"></label>
        <label data-vw-field="shuffle" class="check"><input id="vw-cv-shuffle" type="checkbox" checked>Перемешивать внутри CV</label>
        <label data-vw-field="fraction">Проверочная доля<select id="vw-cv-fraction">${options([.1, .2, .25, .3, .4, .5].map(value => ({id: value, name: `${value * 100}%`})), .2)}</select></label>
        <label data-vw-field="group">Колонка группы<select id="vw-cv-group"><option value="">Сначала выберите набор</option></select></label>
        <div data-vw-field="time" class="vw-time">
          <label>Разрыв, строк<input id="vw-cv-gap" type="number" min="0" step="1" value="0"></label>
          <label>Окно обучения, строк<input id="vw-cv-window" type="number" min="2" step="1" placeholder="Все прошлое"></label>
          <label>Горизонт проверки, строк<input id="vw-cv-horizon" type="number" min="1" step="1" placeholder="Автоматически"></label>
        </div>
        <label data-vw-field="bins">Диапазонов числовой цели<input id="vw-cv-bins" type="number" min="2" max="20" step="1" value="5"></label>
        <label>Параллельные работники<select id="vw-parallelism">${options([1, 2, 3, 4].map(value => ({id: value, name: `${value}${value === 1 ? ' · меньше памяти' : value === 4 ? ' · больше памяти' : ''}`})), 1)}</select></label>
        <p class="vw-note" id="vw-cv-budget" role="status"></p>
      </section>
      <details class="vw-search">
        <summary>Поиск гиперпараметров · Optuna и другие</summary>
        <p class="vw-note">Гиперпараметры — настройки, которые выбирают до обучения: например, сила штрафа. Поиск сравнивает их на разрезах обучения. Итоговый тест остается закрытым.</p>
        <label>Метод поиска<select id="vw-search-method">${options(SEARCH_METHODS, 'optuna_tpe')}</select></label>
        <label>Максимум попыток<input id="vw-search-trials" type="number" min="1" max="100" step="1" value="20"></label>
        <label>Главная метрика<select id="vw-search-metric"><option value="rmse">RMSE · ошибка в единицах цели</option></select></label>
        <label>Куда улучшать<select id="vw-search-direction">${options([{id: 'auto', name: 'По правилу метрики'}, {id: 'min', name: 'Меньше — лучше'}, {id: 'max', name: 'Больше — лучше'}], 'auto')}</select></label>
        <p class="vw-note" id="vw-custom-note" hidden>Формула берется из поля «Своя метрика». Для собственной формулы выберите «Меньше» или «Больше».</p>
        <label id="vw-factor-field" hidden>Фактор сокращения<select id="vw-search-factor">${options([2, 3, 4, 5].map(value => ({id: value, name: `Оставить примерно 1/${value} кандидатов`})), 3)}</select></label>
        <p class="vw-note" id="vw-search-description"></p>
        <div class="vw-quick" role="group" aria-label="Готовые пространства поиска"><button type="button" data-vw-space-preset="basic">Один важный параметр</button><button type="button" data-vw-space-preset="extended">Несколько параметров</button></div>
        <details class="vw-space"><summary><span id="vw-space-heading">Какие параметры менять</span></summary><p class="vw-note" id="vw-space-help">Отметьте параметры и задайте варианты. Остальные настройки остаются как у выбранной модели.</p><div id="vw-parameter-space"></div></details>
        <p class="vw-budget" id="vw-search-budget" role="status"></p>
        <button type="button" id="vw-search-start" class="button primary">Запустить подбор настроек</button>
      </details>
      <div class="vw-live" id="vw-live" role="status" aria-live="polite" hidden></div>`;
    this.annotateHelp();
    this.container.addEventListener('click', this.handleClick = event => this.clicked(event));
    this.container.addEventListener('change', this.handleChange = event => this.changed(event));
    this.container.addEventListener('input', this.handleInput = event => {
      if (event.target.matches('input[type="number"], [data-vw-values]')) this.updateBudget();
    });
    this.listen(this.doc.getElementById('model'), 'change', () => this.setModel(this.catalogue.models?.find(model => model.id === this.doc.getElementById('model').value)));
    this.listen(this.doc.getElementById('target'), 'change', () => this.updateGroups());
    this.listen(this.doc.getElementById('ranking-metric'), 'change', () => this.syncRanking());
    this.listen(this.doc.getElementById('ranking-direction'), 'change', () => this.syncRanking());
    this.listen(this.doc.getElementById('train-share'), 'change', () => this.updateBudget());
    const reports = new Set(['cv-results', 'search-results'].map(id => this.element(id)?.closest('details')).filter(Boolean));
    for (const report of reports) this.listen(report, 'toggle', () => {
      if (!report.open) return;
      for (const plot of report.querySelectorAll('.vw-plot')) {
        if (plot.offsetParent !== null) this.doc.defaultView.Plotly?.Plots?.resize?.(plot);
      }
    });
    this.updateFields();
    return this;
  }

  listen(element, type, listener) {
    if (!element) return;
    element.addEventListener(type, listener);
    this.listeners.push([element, type, listener]);
  }

  element(id) { return this.doc?.getElementById(id); }

  annotate(element, key, extra = '') {
    const help = HELP[key];
    if (!element || !help) return;
    element.dataset.help = `vw-${key}`;
    element.dataset.helpTitle = help[0];
    element.dataset.helpText = help[1] + extra;
    element.dataset.helpLesson = help[2];
    element.dataset.helpSource = help[3];
  }

  annotateHelp() {
    const fields = {'vw-cv-method': 'strategy', 'vw-cv-folds': 'folds', 'vw-cv-repeats': 'repeats', 'vw-cv-shuffle': 'shuffle', 'vw-cv-fraction': 'fraction', 'vw-cv-group': 'group', 'vw-cv-gap': 'gap', 'vw-cv-window': 'window', 'vw-cv-horizon': 'horizon', 'vw-cv-bins': 'bins', 'vw-parallelism': 'parallel', 'vw-search-method': 'method', 'vw-search-trials': 'trials', 'vw-search-metric': 'metric', 'vw-search-direction': 'direction', 'vw-search-factor': 'factor', 'vw-space-help': 'space'};
    for (const [id, key] of Object.entries(fields)) this.annotate(this.element(id), key);
  }

  call(callback, ...args) {
    try {
      const result = callback(...args);
      if (result?.catch) result.catch(error => this.onError(error));
    } catch (error) { this.onError(error); }
  }

  setCatalogue(catalogue = {}) {
    this.catalogue = catalogue;
    if (!this.container) return;
    const cvMethods = catalogue.cv_strategies?.length ? [CV_METHODS[0], ...catalogue.cv_strategies] : CV_METHODS;
    const current = this.element('vw-cv-method').value;
    this.element('vw-cv-method').innerHTML = options(cvMethods, current);
    if (!this.element('vw-cv-method').value) this.element('vw-cv-method').value = 'none';
    const metric = this.element('vw-search-metric').value || 'rmse';
    const metrics = (catalogue.metrics || []).filter(item => item.id !== 'all');
    if (!metrics.some(item => item.id === 'custom')) metrics.push({id: 'custom', name: 'Своя формула'});
    this.element('vw-search-metric').innerHTML = options(metrics.length > 1 ? metrics : [{id: 'rmse', name: 'RMSE'}, ...metrics], metric);
    const method = this.element('vw-search-method').value;
    this.element('vw-search-method').innerHTML = options(catalogue.search_methods?.length ? catalogue.search_methods : SEARCH_METHODS, method);
    this.annotateHelp();
    const selected = this.element('model')?.value;
    this.setModel(catalogue.models?.find(item => item.id === selected) || catalogue.models?.find(item => item.id === 'ridge') || catalogue.models?.[0]);
    this.syncRanking();
    this.updateFields();
  }

  setDataset(metadata) {
    this.dataset = metadata || null;
    if (!this.container) return;
    this.updateGroups();
    if (this.model) { this.captureSpace(); this.renderSpace(false); }
    this.updateBudget();
    this.clearResults();
  }

  updateGroups() {
    const select = this.element('vw-cv-group');
    if (!select) return;
    const previous = select.value;
    const target = this.element('target')?.value || this.dataset?.default_target;
    const columns = (this.dataset?.columns || []).filter(column => column.name !== target);
    const suggested = columns.find(column => /group|patient|subject|device|user|client|пациент|групп/i.test(column.name));
    select.innerHTML = option('', 'Выберите идентификатор группы', '') + columns.map(column => option(column.name, `${column.name} · ${column.unique ?? '?'} разных значений`, previous || suggested?.name)).join('');
    this.updateBudget();
  }

  setModel(model) {
    if (typeof model === 'string') model = this.catalogue.models?.find(item => item.id === model);
    if (!model || !this.container) return;
    const changed = this.model?.id !== model.id;
    this.model = model;
    if (changed || !this.space.size) {
      this.space.clear();
      const count = this.featureCount();
      for (const spec of model.params || []) this.space.set(spec.key, defaultDefinition(spec, model, count));
      this.applySpacePreset('basic', false);
    }
    this.updateBudget();
  }

  featureCount() {
    const selected = this.element('features')?.querySelectorAll('input:checked').length;
    return selected || this.dataset?.feature_names?.length || Math.max((this.dataset?.columns?.length || 3) - 1, 1);
  }

  blockedParameter(spec) {
    return this.model?.id === 'sgd' && ['max_iter', 'tol'].includes(spec.key);
  }

  applySpacePreset(kind, render = true) {
    if (!this.model) return;
    if (render) this.captureSpace();
    const params = this.model.params || [];
    const preferred = ['alpha', 'C', 'n_nonzero_coefs', 'epsilon', 'quantile', 'l1_ratio', 'fit_intercept'];
    const eligible = params.filter(spec => !this.blockedParameter(spec));
    const primary = preferred.map(key => eligible.find(spec => spec.key === key)).find(Boolean) || eligible[0];
    const selected = new Set(primary ? [primary.key] : []);
    if (kind === 'extended') {
      const secondary = preferred.filter(key => key !== primary?.key).map(key => eligible.find(spec => spec.key === key)).filter(Boolean);
      secondary.slice(0, 2).forEach(spec => selected.add(spec.key));
    }
    for (const [key, definition] of this.space) definition.enabled = selected.has(key);
    this.renderSpace(false);
  }

  renderSpace(capture = true) {
    if (!this.container || !this.model) return;
    if (capture) this.captureSpace();
    const rows = (this.model.params || []).map(spec => {
      const definition = this.space.get(spec.key) || defaultDefinition(spec, this.model, this.featureCount());
      this.space.set(spec.key, definition);
      const blocked = this.blockedParameter(spec);
      const categorical = spec.type === 'select' || spec.type === 'bool';
      const choiceControls = categorical ? `<div class="vw-choices">${(spec.type === 'bool' ? [true, false] : spec.options || []).map(value => `<label class="check"><input type="checkbox" data-vw-choice value="${escape(String(value))}"${definition.choices?.includes(value) ? ' checked' : ''}>${escape(spec.option_labels?.[value] || displayParameter(value))}</label>`).join('')}</div>` : `
        <label>Как задавать<select data-vw-space-kind>${options([{id: 'list', name: 'Готовые значения'}, {id: 'range', name: 'Диапазон'}], definition.kind)}</select></label>
        <label data-vw-list${definition.kind === 'range' ? ' hidden' : ''}>Значения через ;<input type="text" data-vw-values value="${escape(definition.rawValues ?? definition.choices?.join('; ') ?? '')}" placeholder="0.001; 0.01; 0.1"></label>
        <div data-vw-range class="vw-range"${definition.kind !== 'range' ? ' hidden' : ''}>
          <label>От<input data-vw-low type="number" step="${spec.type === 'int' ? '1' : 'any'}"${spec.min == null ? '' : ` min="${escape(spec.min)}"`}${spec.max == null ? '' : ` max="${escape(spec.max)}"`} value="${escape(definition.low)}"></label>
          <label>До<input data-vw-high type="number" step="${spec.type === 'int' ? '1' : 'any'}"${spec.min == null ? '' : ` min="${escape(spec.min)}"`}${spec.max == null ? '' : ` max="${escape(spec.max)}"`} value="${escape(definition.high)}"></label>
          <label class="check vw-wide"><input data-vw-log type="checkbox"${definition.log ? ' checked' : ''}>По порядкам величин</label>
          <label class="vw-wide" data-vw-grid-points>Точек для сетки<select data-vw-points>${options([2, 3, 4, 5, 6, 8, 10].map(value => ({id: value, name: String(value)})), definition.points || 3)}</select></label>
        </div>`;
      return `<fieldset class="vw-param" data-vw-param="${escape(spec.key)}"${blocked ? ' data-vw-blocked="true"' : ''}>
        <legend><label class="check"><input data-vw-enabled type="checkbox"${definition.enabled ? ' checked' : ''}${blocked ? ' disabled' : ''}><span>${escape(spec.label || spec.key)}</span></label></legend>
        <p class="vw-note vw-param-help" data-vw-param-description>${escape(spec.help || `Параметр ${spec.key} выбранной модели.`)}</p>
        ${blocked ? '<p class="vw-note">В интерактивном SGD длительность задается числом эпох. Этот параметр не участвует в поиске.</p>' : choiceControls}
      </fieldset>`;
    }).join('');
    this.element('vw-parameter-space').innerHTML = rows || '<p class="vw-note">У модели нет настраиваемых параметров.</p>';
    for (const row of this.element('vw-parameter-space').querySelectorAll('[data-vw-param]')) {
      const spec = this.model.params.find(item => item.key === row.dataset.vwParam);
      const help = row.querySelector('[data-vw-param-description]');
      this.annotate(help, 'space', ` Параметр «${spec.label}»: ${spec.help || 'значения проверяются по ограничениям модели.'}`);
      help.dataset.helpTitle = `Поиск: ${spec.label}`;
      if (this.model.source) help.dataset.helpSource = this.model.source;
    }
    this.updateEnabled();
    this.updateBudget();
  }

  captureSpace() {
    for (const row of this.element('vw-parameter-space')?.querySelectorAll('[data-vw-param]') || []) {
      const key = row.dataset.vwParam;
      const spec = this.model?.params.find(item => item.key === key);
      const definition = this.space.get(key);
      if (!spec || !definition) continue;
      if (this.blockedParameter(spec)) { definition.enabled = false; continue; }
      definition.enabled = row.querySelector('[data-vw-enabled]').checked && !this.blockedParameter(spec);
      if (spec.type === 'bool' || spec.type === 'select') {
        definition.choices = [...row.querySelectorAll('[data-vw-choice]:checked')].map(input => spec.type === 'bool' ? input.value === 'true' : input.value);
      } else {
        definition.kind = row.querySelector('[data-vw-space-kind]').value;
        // Keep the original text until request validation; do not silently drop
        // a misspelled number and search a different space.
        definition.rawValues = row.querySelector('[data-vw-values]').value;
        definition.low = Number(row.querySelector('[data-vw-low]').value);
        definition.high = Number(row.querySelector('[data-vw-high]').value);
        definition.log = row.querySelector('[data-vw-log]').checked;
        definition.points = Number(row.querySelector('[data-vw-points]').value);
      }
    }
  }

  updateEnabled() {
    for (const row of this.element('vw-parameter-space')?.querySelectorAll('[data-vw-param]') || []) {
      const blocked = row.dataset.vwBlocked === 'true';
      const enabled = row.querySelector('[data-vw-enabled]');
      enabled.disabled = this.busy || blocked;
      row.classList.toggle('vw-param-selected', enabled.checked && !blocked);
      for (const input of row.querySelectorAll('input, select')) if (input !== enabled) input.disabled = this.busy || blocked || !enabled.checked;
      const mode = row.querySelector('[data-vw-space-kind]')?.value;
      if (row.querySelector('[data-vw-list]')) row.querySelector('[data-vw-list]').hidden = mode === 'range';
      if (row.querySelector('[data-vw-range]')) row.querySelector('[data-vw-range]').hidden = mode !== 'range';
      if (row.querySelector('[data-vw-grid-points]')) row.querySelector('[data-vw-grid-points]').hidden = !this.element('vw-search-method').value.includes('grid');
    }
    const selected = [...this.space.values()].filter(definition => definition.enabled).length;
    const heading = this.element('vw-space-heading');
    if (heading) heading.textContent = `${this.model?.name || 'Модель'} · менять ${selected} параметр${selected === 1 ? '' : selected > 1 && selected < 5 ? 'а' : 'ов'}`;
  }

  clicked(event) {
    const button = event.target.closest('button');
    if (!button || !this.container.contains(button)) return;
    if (button.dataset.vwCv) {
      this.element('vw-cv-method').value = button.dataset.vwCv;
      this.element('vw-cv-folds').value = 5;
      if (button.dataset.vwCv === 'timeseries') this.disableOuterShuffle();
      this.updateFields();
      this.call(this.onChange);
    } else if (button.dataset.vwFolds) {
      this.element('vw-cv-folds').value = button.dataset.vwFolds;
      this.updateBudget();
      this.call(this.onChange);
    } else if (button.dataset.vwSpacePreset) {
      this.applySpacePreset(button.dataset.vwSpacePreset);
    } else if (button.id === 'vw-search-start') {
      if (!this.busy) this.call(this.onSearch);
    }
  }

  changed(event) {
    const target = event.target;
    if (target.closest('[data-vw-param]')) {
      this.captureSpace();
      this.updateEnabled();
      this.updateBudget();
      return;
    }
    if (target.id === 'vw-search-metric' || target.id === 'vw-search-direction') this.syncRanking(true);
    if (target.id === 'vw-cv-method' && target.value === 'timeseries') this.disableOuterShuffle();
    this.updateFields();
    if (target.id.startsWith('vw-cv-') || target.id === 'vw-parallelism') this.call(this.onChange);
  }

  disableOuterShuffle() {
    const checkbox = this.element('shuffle');
    if (checkbox) checkbox.checked = false;
  }

  syncRanking(toRoot = false) {
    const changed = [];
    for (const [localId, rootId] of [['vw-search-metric', 'ranking-metric'], ['vw-search-direction', 'ranking-direction']]) {
      const local = this.element(localId), root = this.element(rootId);
      if (!local || !root) continue;
      if (toRoot && [...root.options].some(item => item.value === local.value)) {
        root.value = local.value;
        changed.push(root);
      } else if (!toRoot && [...local.options].some(item => item.value === root.value)) local.value = root.value;
    }
    // Commit both controls before dispatch: a root listener synchronizes the
    // pair and must not overwrite the second local value halfway through.
    for (const root of changed) root.dispatchEvent(new this.doc.defaultView.Event('change', {bubbles: true}));
    const note = this.element('vw-custom-note');
    if (note) note.hidden = this.element('vw-search-metric')?.value !== 'custom';
  }

  updateFields() {
    if (!this.container) return;
    const strategy = this.element('vw-cv-method').value;
    const enabled = strategy !== 'none';
    const descriptions = {
      none: 'Модель учится один раз. Часть «Выбор» служит проверкой; итоговый тест откроете после выбора настроек.',
      kfold: 'Для независимых наблюдений. Каждый блок по очереди служит проверкой; подготовка данных обучается заново внутри каждого разреза.',
      repeated_kfold: 'Повторяем KFold с новыми случайными разбиениями. Смотрите среднее и разброс, а не только лучший разрез.',
      shuffle_split: 'Повторяем случайный разрез. В разных проверках некоторые строки могут встречаться несколько раз.',
      timeseries: 'Строки должны идти по времени. Обучение получает прошлое, проверка — следующие строки. Внешнее разделение тоже сохраняет порядок.',
      group_kfold: 'Все строки одного пациента, устройства или другого объекта остаются вместе. Колонка группы исключается из признаков.',
      group_shuffle_split: 'Каждый раз случайно выбираем целые группы для проверки. В разных проверках группы могут повторяться.',
      leave_one_group_out: 'Каждый раз проверяем на одной полностью отложенной группе. Число обучений равно числу групп во внешнем обучении.',
      leave_one_out: 'Каждый раз проверяем на одной строке. Доступно до 100 обучающих строк. R² на одной строке не определен: выбирайте MAE или MSE.',
      stratified_bins: 'Экспериментальная схема для независимых строк. Выравнивает доли диапазонов числовой цели; не исправляет зависимость групп или времени.',
    };
    this.element('vw-cv-description').textContent = descriptions[strategy] || '';
    const visible = {folds: enabled && !['leave_one_out', 'leave_one_group_out'].includes(strategy), repeats: strategy === 'repeated_kfold', shuffle: ['kfold', 'group_kfold', 'stratified_bins'].includes(strategy), fraction: ['shuffle_split', 'group_shuffle_split'].includes(strategy), group: GROUP_METHODS.has(strategy), time: strategy === 'timeseries', bins: strategy === 'stratified_bins'};
    for (const field of this.container.querySelectorAll('[data-vw-field]')) field.hidden = !visible[field.dataset.vwField];
    const method = this.element('vw-search-method').value;
    this.element('vw-factor-field').hidden = !method.startsWith('halving');
    const searchDescriptions = {
      grid: 'Все комбинации из отмеченных вариантов. Большая сетка быстро растет: 5 значений × 5 значений = 25 кандидатов.',
      random: 'Независимые случайные комбинации. Широкий диапазон по порядкам величин помогает начать поиск силы штрафа.',
      optuna_tpe: 'Первые испытания исследуют пространство. Затем настоящий Optuna TPE предлагает настройки по предыдущим результатам. Улучшение не гарантировано.',
      optuna_random: 'Настоящий Optuna Study со случайным генератором предложений. Удобно сравнивать с TPE при одинаковом бюджете.',
      halving_grid: 'Сначала кандидаты получают меньше строк. Лучшие переходят на следующий этап. Доступно для KFold, повторного KFold и ShuffleSplit.',
      halving_random: 'Начинаем со случайных кандидатов, затем оставляем лучших и увеличиваем объем данных. Для групп и времени выберите обычный поиск.',
    };
    this.element('vw-search-description').textContent = searchDescriptions[method] || '';
    this.syncRanking();
    this.updateEnabled();
    this.updateBudget();
  }

  getCVConfig() {
    if (!this.container) return {strategy: 'none'};
    const strategy = this.element('vw-cv-method').value;
    if (strategy === 'none') return {strategy: 'none'};
    const config = {strategy, folds: integer(this.element('vw-cv-folds'), 'Число частей', 2, 20)};
    if (strategy === 'repeated_kfold') {
      config.repeats = integer(this.element('vw-cv-repeats'), 'Число повторов', 1, 10);
      if (config.folds * config.repeats > 100) throw new Error('Перекрестная проверка требует больше 100 обучений. Уменьшите части или повторы.');
    }
    if (['kfold', 'group_kfold', 'stratified_bins'].includes(strategy)) config.shuffle = this.element('vw-cv-shuffle').checked;
    if (['shuffle_split', 'group_shuffle_split'].includes(strategy)) {
      const fraction = Number(this.element('vw-cv-fraction').value);
      if (!finite(fraction) || fraction <= 0 || fraction >= 1) throw new Error('Проверочная доля должна быть между 0 и 1.');
      config.test_size = fraction;
    }
    if (GROUP_METHODS.has(strategy)) {
      config.group_column = this.element('vw-cv-group').value;
      if (!config.group_column) throw new Error('Выберите колонку с идентификатором группы.');
      if (config.group_column === this.element('target')?.value) throw new Error('Целевая колонка не может служить идентификатором группы.');
    }
    if (strategy === 'timeseries') {
      const rows = Math.max(Number(this.dataset?.rows) || 100000, 2);
      config.shuffle = false;
      config.gap = integer(this.element('vw-cv-gap'), 'Разрыв', 0, rows);
      if (this.element('vw-cv-window').value !== '') config.max_train_size = integer(this.element('vw-cv-window'), 'Окно обучения', 2, rows);
      if (this.element('vw-cv-horizon').value !== '') config.test_size = integer(this.element('vw-cv-horizon'), 'Горизонт проверки', 1, rows);
    }
    if (strategy === 'stratified_bins') config.bins = integer(this.element('vw-cv-bins'), 'Диапазоны цели', 2, 20);
    return config;
  }

  getParallelism() {
    return this.container ? integer(this.element('vw-parallelism'), 'Число работников', 1, 4) : 1;
  }

  numericChoices(definition, spec, key) {
    const tokens = String(definition.rawValues ?? definition.choices?.join(';') ?? '').trim().split(/[;,\s]+/).filter(Boolean);
    if (!tokens.length || tokens.length > 100) throw new Error(`${spec.label}: укажите от 1 до 100 значений.`);
    if (tokens.some(token => !numeric(token))) throw new Error(`${spec.label}: значение должно быть числом. Разделяйте варианты знаком ; и используйте точку для десятичной части.`);
    const values = [...new Set(tokens.map(Number))];
    if (values.some(value => spec.type === 'int' && !Number.isInteger(value))) throw new Error(`${spec.label}: нужны целые значения.`);
    if (values.some(value => (spec.min != null && value < spec.min) || (spec.max != null && value > spec.max))) throw new Error(`${spec.label}: значения должны быть в пределах ${spec.min ?? '−∞'}…${spec.max ?? '∞'}.`);
    if (!values.length) throw new Error(`Не заданы варианты для ${key}.`);
    return values;
  }

  readSpace(method) {
    this.captureSpace();
    const space = {};
    for (const [key, definition] of this.space) {
      if (!definition.enabled) continue;
      const spec = this.model.params.find(item => item.key === key);
      if (spec.type === 'bool' || spec.type === 'select') {
        if (!definition.choices?.length) throw new Error(`Отметьте хотя бы один вариант «${spec.label}».`);
        space[key] = clone(definition.choices);
      } else if (definition.kind === 'list') space[key] = this.numericChoices(definition, spec, key);
      else {
        const {low, high, log} = definition;
        if (!finite(low) || !finite(high) || low >= high) throw new Error(`${spec.label}: нижняя граница должна быть меньше верхней.`);
        if (spec.type === 'int' && (!Number.isInteger(low) || !Number.isInteger(high))) throw new Error(`${spec.label}: границы должны быть целыми.`);
        if ((spec.min != null && low < spec.min) || (spec.max != null && high > spec.max)) throw new Error(`${spec.label}: диапазон должен находиться в пределах ${spec.min ?? '−∞'}…${spec.max ?? '∞'}.`);
        if (log && low <= 0) throw new Error(`${spec.label}: диапазон по порядкам величин должен быть положительным.`);
        if (method.includes('grid')) {
          const points = Number(definition.points);
          if (!Number.isInteger(points) || points < 2 || points > 10) throw new Error('Для сетки выберите от 2 до 10 точек диапазона.');
          const values = Array.from({length: points}, (_, index) => index === points - 1 ? high : log ? Math.exp(Math.log(low) + (Math.log(high) - Math.log(low)) * index / (points - 1)) : low + (high - low) * index / (points - 1));
          space[key] = [...new Set(values.map(value => spec.type === 'int' ? Math.round(value) : Number(value.toPrecision(12))))];
        } else {
          if (spec.type === 'int' && log && ['random', 'halving_random'].includes(method)) throw new Error(`${spec.label}: для целочисленного диапазона по порядкам величин выберите Optuna или готовый список значений.`);
          space[key] = {type: spec.type, low, high, log: !!log};
        }
      }
    }
    const count = Object.keys(space).length;
    if (!count) throw new Error('Отметьте хотя бы один параметр для поиска.');
    if (count > 12) throw new Error('Для одного поиска отметьте не больше 12 параметров.');
    return space;
  }

  estimatedFolds(config, search = false) {
    const train = Number(this.element('train-share')?.value || 60) / 100;
    const rows = this.dataset?.rows ? Math.max(2, Math.floor(this.dataset.rows * train)) : null;
    if (config.strategy === 'none') return search ? 3 : 0;
    if (config.strategy === 'leave_one_out') return rows;
    if (config.strategy === 'leave_one_group_out') {
      // Exact training groups are known only after the server builds holdout.
      return null;
    }
    return config.folds * (config.strategy === 'repeated_kfold' ? config.repeats : 1);
  }

  getSearchConfig(model = this.model) {
    if (!this.container) return null;
    this.setModel(model);
    const method = this.element('vw-search-method').value;
    const trials = integer(this.element('vw-search-trials'), 'Бюджет поиска', 1, 100);
    const param_space = this.readSpace(method);
    const cv = this.getCVConfig();
    const folds = this.estimatedFolds(cv, true);
    const candidates = method.includes('grid') ? Object.values(param_space).reduce((count, values) => count * values.length, 1) : trials;
    if (method.includes('grid') && candidates > trials) throw new Error(`Сетка содержит ${candidates} комбинаций, бюджет ${trials}. Увеличьте бюджет или сократите варианты.`);
    if (method.startsWith('halving') && !['none', 'kfold', 'repeated_kfold', 'shuffle_split'].includes(cv.strategy)) throw new Error('Последовательное сокращение доступно для KFold, повторного KFold и ShuffleSplit. Для групп, времени и диапазонов цели выберите сетку, случайный поиск или Optuna.');
    const factor = integer(this.element('vw-search-factor'), 'Фактор сокращения', 2, 5);
    let fits = folds == null ? null : folds * candidates;
    if (method.startsWith('halving') && fits != null && this.dataset?.rows) {
      const trainRows = Math.floor(this.dataset.rows * Number(this.element('train-share')?.value || 60) / 100);
      const levels = 1 + Math.floor(Math.log(Math.max(trainRows / Math.max(10, folds * 4), 1)) / Math.log(factor));
      fits *= levels;
    }
    if (fits != null && fits > 500) throw new Error(`Поиск может потребовать ${fits} обучений CV; предел 500. Уменьшите бюджет, число частей или параметры сетки.`);
    const metric = this.element('ranking-metric')?.value || this.element('vw-search-metric').value;
    const requestedDirection = this.element('ranking-direction')?.value || this.element('vw-search-direction').value;
    const direction = requestedDirection === 'auto' ? this.catalogue.metrics?.find(item => item.id === metric)?.direction || 'min' : requestedDirection;
    if (metric === 'custom' && !(this.element('custom-metric')?.value || '').trim()) throw new Error('Для поиска по собственной метрике задайте формулу в поле «Своя метрика».');
    const config = {method, trials, metric, direction, param_space, n_jobs: this.getParallelism()};
    if (method.startsWith('halving')) config.factor = factor;
    return config;
  }

  updateBudget() {
    if (!this.container) return;
    try {
      const config = this.getCVConfig();
      const folds = this.estimatedFolds(config);
      const text = config.strategy === 'none' ? 'Для обычного обучения CV выключена. Поиск все равно использует 3 внутренних разреза.' : folds == null ? 'Число обучений определяется группами во внешней обучающей части. Предел — 100 разрезов.' : `Около ${folds} обучений CV внутри обучающей части. Подготовка данных не видит ответы отложенных строк.`;
      this.element('vw-cv-budget').textContent = text;
      const cvWarning = folds != null && folds > 100;
      this.element('vw-cv-budget').classList.toggle('vw-invalid', cvWarning);
      this.captureSpace();
      const method = this.element('vw-search-method').value;
      const trials = Number(this.element('vw-search-trials').value);
      const selected = [...this.space.values()].filter(item => item.enabled);
      const candidates = method.includes('grid') ? selected.reduce((total, item) => total * (item.kind === 'range' ? Number(item.points || 3) : item.rawValues != null ? item.rawValues.trim().split(/[;,\s]+/).filter(Boolean).length : item.choices?.length || 0), 1) : trials;
      const searchFolds = this.estimatedFolds(config, true);
      const fits = searchFolds == null ? null : candidates * searchFolds;
      const textSearch = `${selected.length} параметр${selected.length === 1 ? '' : selected.length < 5 ? 'а' : 'ов'} · ${candidates} кандидат${candidates === 1 ? '' : candidates < 5 ? 'а' : 'ов'}${fits == null ? ' · число обучений зависит от групп' : ` × ${searchFolds} разрез${searchFolds === 1 ? '' : searchFolds < 5 ? 'а' : 'ов'} = ${fits} обучений`}${method.startsWith('halving') ? ' на первом этапе; следующие этапы добавят обучения' : ''}.`;
      this.element('vw-search-budget').textContent = textSearch + (method.includes('grid') && candidates > trials ? ' Сетка не помещается в бюджет.' : fits != null && fits > 500 ? ' Превышен предел 500 обучений.' : '');
      this.element('vw-search-budget').classList.toggle('vw-invalid', !selected.length || (method.includes('grid') && candidates > trials) || (fits != null && fits > 500));
    } catch (error) {
      this.element('vw-cv-budget').textContent = error.message;
      this.element('vw-cv-budget').classList.add('vw-invalid');
      this.element('vw-search-budget').textContent = 'Сначала исправьте настройки проверки.';
    }
  }

  setConfig(request = {}) {
    if (!this.container) return;
    const cv = request.cv_config || (request.cv ? {strategy: 'kfold', folds: request.cv} : {strategy: 'none'});
    const mapping = {strategy: 'vw-cv-method', folds: 'vw-cv-folds', n_splits: 'vw-cv-folds', repeats: 'vw-cv-repeats', group_column: 'vw-cv-group', gap: 'vw-cv-gap', max_train_size: 'vw-cv-window', bins: 'vw-cv-bins'};
    for (const [key, id] of Object.entries(mapping)) if (cv[key] != null) this.element(id).value = String(cv[key]);
    if (cv.max_train_size == null) this.element('vw-cv-window').value = '';
    if (cv.strategy === 'timeseries') this.element('vw-cv-horizon').value = cv.test_size ?? '';
    else if (cv.test_size != null) {
      const value = String(cv.test_size);
      const select = this.element('vw-cv-fraction');
      if (![...select.options].some(item => item.value === value)) select.add(new this.doc.defaultView.Option(`${Number(value) * 100}%`, value));
      select.value = value;
    }
    this.element('vw-cv-shuffle').checked = cv.shuffle ?? true;
    this.element('vw-parallelism').value = String(request.n_jobs || request.search?.n_jobs || 1);
    const model = this.catalogue.models?.find(item => item.id === request.model);
    if (model) this.setModel(model);
    const search = request.search;
    if (search) {
      this.element('vw-search-method').value = search.method || 'optuna_tpe';
      this.element('vw-search-trials').value = search.trials || 20;
      this.element('vw-search-factor').value = search.factor || 3;
      if (search.metric) this.element('vw-search-metric').value = search.metric;
      if (search.direction) this.element('vw-search-direction').value = search.direction;
      if (search.param_space) {
        for (const definition of this.space.values()) definition.enabled = false;
        for (const [key, definition] of Object.entries(search.param_space)) {
          if (!this.space.has(key)) continue;
          const item = this.space.get(key);
          const categorical = Array.isArray(definition) || ['categorical', 'select', 'bool'].includes(definition.type);
          Object.assign(item, categorical ? {enabled: true, kind: 'list', choices: clone(Array.isArray(definition) ? definition : definition.choices || definition.values || []), rawValues: undefined} : {enabled: true, kind: 'range', low: definition.low, high: definition.high, log: !!definition.log});
        }
        this.renderSpace(false);
      }
      this.syncRanking(true);
    }
    if (cv.strategy === 'timeseries') this.disableOuterShuffle();
    this.updateFields();
  }

  reset() {
    if (!this.container) return;
    this.element('vw-cv-method').value = 'none';
    this.element('vw-cv-folds').value = 5;
    this.element('vw-cv-repeats').value = 2;
    this.element('vw-cv-shuffle').checked = true;
    this.element('vw-cv-fraction').value = .2;
    this.element('vw-cv-gap').value = 0;
    this.element('vw-cv-window').value = '';
    this.element('vw-cv-horizon').value = '';
    this.element('vw-cv-bins').value = 5;
    this.element('vw-parallelism').value = 1;
    this.element('vw-search-method').value = 'optuna_tpe';
    this.element('vw-search-trials').value = 20;
    this.element('vw-search-factor').value = 3;
    if (this.model) {
      this.space.clear();
      for (const spec of this.model.params || []) this.space.set(spec.key, defaultDefinition(spec, this.model, this.featureCount()));
      this.applySpacePreset('basic', false);
    }
    this.clearResults();
    this.updateFields();
  }

  setBusy(busy) {
    const started = busy && !this.busy;
    this.busy = !!busy;
    if (!this.container) return;
    for (const element of this.container.querySelectorAll('input, select, button:not(.help-trigger)')) element.disabled = this.busy;
    this.updateEnabled();
    if (started) {
      this.progressRecords = [];
      this.element('vw-live').hidden = true;
      this.clearResults();
    }
  }

  clearResults() {
    this.result = null;
    this.progressRecords = [];
    for (const id of ['cv-results', 'search-results']) {
      const target = this.element(id);
      if (!target) continue;
      for (const plot of target.querySelectorAll('.vw-plot')) this.doc.defaultView.Plotly?.purge?.(plot);
      target.replaceChildren();
    }
    if (this.element('vw-live')) this.element('vw-live').hidden = true;
  }

  metricName(id) { return this.catalogue.metrics?.find(item => item.id === id)?.name || id; }

  plot(target, traces, layout = {}) {
    if (!target || !this.doc.defaultView.Plotly?.react) return;
    const theme = this.doc.defaultView.getComputedStyle(this.doc.documentElement);
    const color = theme.getPropertyValue('--text').trim() || '#edf5f4';
    const grid = theme.getPropertyValue('--line').trim() || '#273740';
    const settings = {paper_bgcolor: 'transparent', plot_bgcolor: 'transparent', font: {family: 'Inter, Arial, sans-serif', color, size: 12}, margin: {l: 62, r: 20, t: 34, b: 58}, autosize: true, legend: {orientation: 'h', y: 1.12}, xaxis: {gridcolor: grid, zerolinecolor: grid}, yaxis: {gridcolor: grid, zerolinecolor: grid}, ...layout};
    Promise.resolve(this.doc.defaultView.Plotly.react(target, traces, settings, {responsive: true, displaylogo: false, locale: 'ru', modeBarButtonsToRemove: ['select2d', 'lasso2d']})).catch(error => this.onError(error));
  }

  renderResults(result = {}) {
    this.result = result;
    this.element('vw-live').hidden = true;
    const cvTarget = this.element('cv-results');
    const cv = result.cv;
    if (cvTarget) {
      for (const plot of cvTarget.querySelectorAll('.vw-plot')) this.doc.defaultView.Plotly?.purge?.(plot);
      if (cv?.summary && cv.folds?.length) {
        const metrics = Object.keys(cv.summary);
        const main = this.element('ranking-metric')?.value || 'rmse';
        const selected = metrics.includes(main) ? main : metrics[0];
        cvTarget.innerHTML = `<section class="vw-report"><h3>Перекрестная проверка · ${escape(CV_METHODS.find(item => item.id === cv.kind)?.name || cv.kind)}</h3><p class="vw-note">${escape(cv.note || 'Разрезы построены внутри внешнего обучения. Среднее и разброс рассчитаны по реальным проверкам.')}</p><div class="vw-table-scroll"><table><thead><tr><th>Метрика</th><th>Среднее</th><th>Разброс (σ)</th><th>Определена на частях</th></tr></thead><tbody>${metrics.map(metric => `<tr><th>${escape(this.metricName(metric))}</th><td>${format(cv.summary[metric].mean)}</td><td>${format(cv.summary[metric].std)}</td><td>${cv.summary[metric].valid_folds} / ${cv.folds.length}</td></tr>`).join('')}</tbody></table></div><p class="vw-note">σ — стандартное отклонение между разрезами, а не доверительный интервал. «—» означает, что метрика для этих данных не определена.</p><label class="vw-result-metric">Показать по частям<select id="vw-cv-result-metric">${options(metrics.map(id => ({id, name: this.metricName(id)})), selected)}</select></label><div id="vw-cv-score-plot" class="vw-plot" aria-label="Реальные метрики на разрезах CV"></div><div id="vw-cv-fold-table" class="vw-table-scroll"></div><details class="vw-split-details"><summary>Какие строки использованы в разрезах</summary><p class="vw-note">Номера здесь относятся к внешней обучающей части. Зеленое — обучение, сиреневое — проверка, пустое — строка не использовалась на этом разрезе. График доступен, если сервер вернул индексы.</p><div id="vw-cv-split-plot" class="vw-plot" aria-label="Реальные обучающие и проверочные индексы CV"></div></details></section>`;
        const update = () => this.renderFoldMetrics(cv, this.element('vw-cv-result-metric').value);
        this.element('vw-cv-result-metric').addEventListener('change', update);
        update();
        const details = cvTarget.querySelector('.vw-split-details');
        details.addEventListener('toggle', () => { if (details.open) this.renderSplitMap(cv); });
      } else cvTarget.replaceChildren();
    }
    if (result.search?.trials) {
      const report = this.element('search-results')?.closest('details');
      if (report) report.open = true;
      this.renderSearch(result.search, false);
    }
    else this.element('search-results')?.replaceChildren();
  }

  renderFoldMetrics(cv, metric) {
    const scores = cv.folds.map(fold => fold.metrics?.[metric] ?? null);
    this.plot(this.element('vw-cv-score-plot'), [{type: 'bar', x: cv.folds.map(fold => `Часть ${fold.fold}`), y: scores, name: this.metricName(metric), marker: {color: '#b7a4ff'}, hovertemplate: '%{x}<br>%{y:.5g}<extra></extra>'}], {xaxis: {title: {text: 'Проверочный разрез'}}, yaxis: {title: {text: this.metricName(metric)}}, showlegend: false});
    this.element('vw-cv-fold-table').innerHTML = `<table><thead><tr><th>Часть</th><th>Строк обучения</th><th>Строк проверки</th><th>${escape(this.metricName(metric))}</th><th>Время, с</th></tr></thead><tbody>${cv.folds.map(fold => `<tr><td>${fold.fold}</td><td>${format(fold.train_size)}</td><td>${format(fold.validation_size)}</td><td>${format(fold.metrics?.[metric])}${fold.metric_details?.[metric]?.reason ? `<small class="vw-metric-reason">${escape(fold.metric_details[metric].reason)}</small>` : ''}</td><td>${format(fold.seconds)}</td></tr>`).join('')}</tbody></table>`;
  }

  renderSplitMap(cv) {
    const target = this.element('vw-cv-split-plot');
    if (!target) return;
    if (!cv.folds.every(fold => Array.isArray(fold.train_indices) && Array.isArray(fold.validation_indices))) {
      target.innerHTML = '<p class="plot-empty">Для больших разрезов индексы не передаются. Число обучающих и проверочных строк показано в таблице.</p>';
      return;
    }
    const max = cv.folds.reduce((largest, fold) => [...fold.train_indices, ...fold.validation_indices].reduce((current, index) => Math.max(current, index), largest), -1);
    const z = cv.folds.map(fold => {
      const values = Array(max + 1).fill(null);
      for (const index of fold.train_indices) values[index] = 0;
      for (const index of fold.validation_indices) values[index] = 1;
      return values;
    });
    this.plot(target, [{type: 'heatmap', x: Array.from({length: max + 1}, (_, index) => index), y: cv.folds.map(fold => `Часть ${fold.fold}`), z, zmin: 0, zmax: 1, colorscale: [[0, '#66e4bc'], [.5, '#66e4bc'], [.5, '#b7a4ff'], [1, '#b7a4ff']], colorbar: {tickvals: [0, 1], ticktext: ['Обучение', 'Проверка'], len: .7}, hovertemplate: '%{y}<br>Строка %{x}<extra></extra>'}], {xaxis: {title: {text: 'Индекс строки внутри внешнего обучения'}}, yaxis: {autorange: 'reversed'}, margin: {l: 72, r: 110, t: 20, b: 52}});
  }

  renderSearch(search, live) {
    const target = this.element('search-results');
    if (!target) return;
    const method = SEARCH_METHODS.find(item => item.id === search.method)?.name || search.method || 'Подбор параметров';
    if (!target.querySelector('#vw-search-curve')) target.innerHTML = `<section class="vw-report"><h3 id="vw-search-result-title"></h3><p class="vw-note" id="vw-search-result-note"></p><div id="vw-search-winner" class="vw-winner"></div><div id="vw-search-curve" class="vw-plot" aria-label="Реальные оценки испытаний поиска"></div><p class="vw-note" id="vw-search-curve-note"></p><details${live ? '' : ' open'}><summary>Все испытания и параметры</summary><div id="vw-search-trial-table" class="vw-table-scroll"></div></details></section>`;
    this.element('vw-search-result-title').textContent = `${live ? 'Поиск идет' : 'Подбор завершен'} · ${method}`;
    this.element('vw-search-result-note').textContent = search.note || 'Кандидаты оцениваются на внутренних разрезах внешнего обучения. Тест не участвует в выборе.';
    const metric = search.metric || this.element('ranking-metric')?.value || this.element('vw-search-metric').value;
    const records = search.trials || [];
    const winner = this.element('vw-search-winner');
    winner.innerHTML = live ? `<span>Завершено попыток: ${records.length}</span>` : `<strong>${escape(this.metricName(metric))}: ${format(search.best_score)}</strong><span>Победитель · попытка ${format(search.best_trial)} · ${search.folds || '?'} разреза CV</span><div>${Object.entries(search.best_params || {}).map(([key, value]) => `<span class="vw-param-chip">${escape(this.model?.params?.find(item => item.key === key)?.label || key)}: ${escape(displayParameter(value))}</span>`).join('')}</div>`;
    const hasStages = records.some(record => record.iteration != null);
    this.element('vw-search-curve-note').textContent = hasStages ? 'Этапы используют разные объемы данных. Лучший результат считается отдельно для каждого этапа; победитель выбран на последнем. График не является историей градиентного спуска.' : 'Одна точка — одна реальная комбинация параметров. Оценка — среднее по внутренним разрезам CV. Линия лучшего результата показывает достигнутую оценку, а не шаги обучения одной модели.';
    const label = record => record.iteration == null ? `Попытка ${record.trial}` : `Попытка ${record.trial}, этап ${record.iteration + 1}, строк ${record.resources}`;
    const curve = search.curve || records;
    const traces = [{type: 'scatter', mode: 'markers', x: curve.map(record => record.trial), y: curve.map(record => record.score ?? null), text: curve.map(label), name: 'Среднее CV', marker: {color: '#b7a4ff', size: 8}, error_y: {type: 'data', array: curve.map(record => record.std ?? 0), visible: curve.some(record => finite(record.std))}, hovertemplate: '%{text}<br>%{y:.5g}<extra></extra>'}];
    // Do not join winners across successive-halving stages with incomparable
    // resource budgets. Null breaks are real boundaries, not fabricated scores.
    const bestX = [], bestY = [];
    curve.forEach((record, index) => {
      if (index && record.iteration !== curve[index - 1].iteration) {bestX.push(null); bestY.push(null);}
      bestX.push(record.trial); bestY.push(record.best_score ?? null);
    });
    traces.push({type: 'scatter', mode: 'lines', line: {color: '#66e4bc', shape: 'hv', width: 2}, x: bestX, y: bestY, name: hasStages ? 'Лучшее на этапе' : 'Лучшее к этой попытке', connectgaps: false, hovertemplate: 'После попытки %{x}<br>%{y:.5g}<extra></extra>'});
    this.plot(this.element('vw-search-curve'), traces, {xaxis: {title: {text: 'Завершенная попытка'}, dtick: records.length > 20 ? undefined : 1}, yaxis: {title: {text: this.metricName(metric)}}});
    this.element('vw-search-trial-table').innerHTML = `<table><thead><tr><th>Попытка</th><th>Настройки</th><th>Среднее CV</th><th>σ</th>${hasStages ? '<th>Этап / строк</th>' : ''}<th>Время, с</th><th>Результат</th></tr></thead><tbody>${records.map(record => `<tr${record.trial === search.best_trial ? ' class="vw-best-row"' : ''}><td>${format(record.trial)}</td><td>${Object.entries(record.params || {}).map(([key, value]) => `<span class="vw-param-chip">${escape(this.model?.params?.find(item => item.key === key)?.label || key)}: ${escape(displayParameter(value))}</span>`).join('')}</td><td>${format(record.score)}</td><td>${format(record.std)}</td>${hasStages ? `<td>${format(Number(record.iteration) + 1)} / ${format(record.resources)}</td>` : ''}<td>${format(record.seconds)}</td><td>${escape(record.status === 'complete' ? 'Оценен' : record.error || 'Метрика не определена')}</td></tr>`).join('')}</tbody></table>`;
  }

  renderProgress(events = []) {
    if (!this.container || !Array.isArray(events)) return;
    const cvFolds = events.filter(event => event.cv_fold);
    const searchEvents = events.filter(event => event.search_trial);
    if (!cvFolds.length && !searchEvents.length) return;
    const live = this.element('vw-live');
    live.hidden = false;
    if (searchEvents.length) {
      if (!this.progressRecords.length) {
        const report = this.element('search-results')?.closest('details');
        if (report) report.open = true;
      }
      const byTrial = new Map();
      for (const event of searchEvents) byTrial.set(`${event.search_trial.iteration ?? 0}:${event.search_trial.trial}`, event.search_trial);
      this.progressRecords = [...byTrial.values()];
      const last = searchEvents.at(-1);
      live.textContent = last.message || `Подбор: завершено ${this.progressRecords.length} попыток.`;
      this.renderSearch({method: this.element('vw-search-method').value, metric: this.element('ranking-metric')?.value || this.element('vw-search-metric').value, trials: this.progressRecords, curve: last.search_curve || this.progressRecords}, true);
    } else {
      const last = cvFolds.at(-1);
      live.textContent = last.message || `Готово частей CV: ${cvFolds.length}.`;
    }
  }

  destroy() {
    if (!this.container) return;
    for (const [element, type, listener] of this.listeners) element.removeEventListener(type, listener);
    this.listeners = [];
    this.container.removeEventListener('click', this.handleClick);
    this.container.removeEventListener('change', this.handleChange);
    this.container.removeEventListener('input', this.handleInput);
    this.clearResults();
  }
}
