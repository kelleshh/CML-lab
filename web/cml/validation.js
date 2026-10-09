import { element } from './dom.js';
import { action, checkbox, field, heading, notice, number, select, setOptions } from './controls.js';

const STRATEGIES = [
  { value: 'none', label: 'Одна обучающая и одна проверочная часть' },
  { value: 'kfold', label: 'K-fold: несколько частей' },
  { value: 'stratified_kfold', label: 'Stratified K-fold: сохранить доли классов' },
  { value: 'repeated_kfold', label: 'Повторить K-fold с разными разделениями' },
  { value: 'shuffle_split', label: 'Несколько случайных разделений' },
  { value: 'group_kfold', label: 'K-fold по целым группам' },
  { value: 'group_shuffle_split', label: 'Случайное разделение целых групп' },
  { value: 'leave_one_group_out', label: 'По очереди отложить каждую группу' },
  { value: 'timeseries', label: 'По времени: обучаться на прошлом' },
  { value: 'leave_one_out', label: 'По очереди отложить каждую строку' },
  { value: 'stratified_bins', label: 'Сохранить области числовой цели' },
];

export class ValidationBuilder {
  constructor({ catalogue, onChange, onError }) {
    this.catalogue = catalogue; this.onChange = onChange; this.onError = onError;
    this.task = 'regression'; this.model = null; this.saved = {}; this.searchRows = [];
  }

  mount(container) { this.container = container; this.render(); return this; }
  setCatalogue(catalogue) { this.catalogue = catalogue; this.render(); }
  setTask(task) {
    const previous = this.task;
    this.saved = this.values(); this.task = task;
    if (previous !== task) { delete this.saved.metrics; this.saved.search = null; this.saved.validation = {}; this.saved.custom_metric = ''; this.searchRows = []; }
    this.render();
  }
  setModel(model) { this.model = model; this.renderSearchRows(); }
  metrics() { return (this.catalogue.metrics || []).filter(metric => metric.tasks?.length ? metric.tasks.includes(this.task) : metric.task ? metric.task === this.task : true); }

  values() {
    if (!this.train) return this.saved;
    const validation = { strategy: this.strategy.value, folds: Number(this.folds.value), repeats: Number(this.repeats.value), gap: Number(this.gap.value) };
    if (this.group.value) validation.group_column = this.group.value;
    const split = { train: Number(this.train.value) / 100, validation: Number(this.val.value) / 100, test: Math.max(0, 1 - (Number(this.train.value) + Number(this.val.value)) / 100), shuffle: this.shuffle.checked };
    const metrics = [...this.metricList.querySelectorAll('input:checked')].map(control => control.value);
    const search = this.searchEnabled.checked ? {
      method: this.searchMethod.value, trials: Number(this.trials.value), metric: this.searchMetric.value,
      direction: this.direction.value, param_space: Object.fromEntries(this.searchRows.filter(row => row.key.value).map(row => {
        const raw = row.values.value.split(/[,;]+/).map(value => value.trim()).filter(Boolean);
        const schema = (this.model?.params || []).find(param => param.key === row.key.value);
        const values = raw.map(value => ['int', 'integer', 'float', 'number'].includes(schema?.type) ? Number(value) : ['bool', 'boolean'].includes(schema?.type) ? value === 'true' : value);
        return [row.key.value, values];
      })),
    } : null;
    return { regularization_path: this.diagnostics?.regularization_path?.checked || false, learning_curve: this.diagnostics?.learning_curve?.checked || false, permutation_importance: this.diagnostics?.permutation_importance?.checked || false, split, validation, search, metrics, seed: Number(this.seed.value), n_jobs: Number(this.jobs.value), custom_metric: this.custom.value.trim() || undefined };
  }

  setConfig(config = {}) {
    this.saved = structuredClone(config);
    this.render();
    if (config.search?.param_space) {
      this.searchRows = Object.entries(config.search.param_space).map(([key, values]) => ({ savedKey: key, savedValues: Array.isArray(values) ? values.join(', ') : `${values.low}, ${values.high}` }));
      this.renderSearchRows();
    }
  }

  setDataset(dataset) {
    this.dataset = dataset;
    if (this.group) setOptions(this.group, [{ value: '', label: 'Не используется' }, ...(dataset?.columns || []).map(column => ({ value: column.name, label: column.name }))], this.saved.validation?.group_column);
  }

  render() {
    if (!this.container) return;
    const config = this.saved || {};
    const temporal = ['forecasting', 'panel'].includes(this.task);
    const unsupervised = ['clustering', 'anomaly', 'reduction'].includes(this.task);
    this.train = number(Math.round((config.split?.train ?? .6) * 100), { min: 20, max: 90, step: 1 });
    this.val = number(Math.round((config.split?.validation ?? .2) * 100), { min: 5, max: 60, step: 1 });
    this.test = element('output', { className: 'cml-code', text: `${100 - Number(this.train.value) - Number(this.val.value)}%` });
    const shareUpdate = () => { this.test.textContent = `${100 - Number(this.train.value) - Number(this.val.value)}%`; this.onChange(); };
    this.train.addEventListener('change', shareUpdate); this.val.addEventListener('change', shareUpdate);
    const shuffle = checkbox('Перемешать перед разделением', temporal ? false : config.split?.shuffle ?? true, { help: 'Перемешивание допустимо для независимых наблюдений. Прогноз по времени требует последовательного разделения; ранжирование разделяется целыми запросами.', lesson_id: temporal ? 'forecasting-lags' : '26-cross-validation' });
    this.shuffle = shuffle.control; this.shuffle.disabled = temporal;
    const strategies = STRATEGIES.filter(item => temporal ? ['none', 'timeseries'].includes(item.value) : this.task === 'classification' ? item.value !== 'stratified_bins' : item.value !== 'stratified_kfold');
    this.strategy = select(strategies, config.validation?.strategy || (temporal ? 'timeseries' : 'none'));
    this.folds = number(config.validation?.folds ?? 3, { min: 2, max: 10, step: 1 });
    this.repeats = number(config.validation?.repeats ?? 2, { min: 1, max: 5, step: 1 });
    this.gap = number(config.validation?.gap ?? 0, { min: 0, max: 1000, step: 1 });
    this.group = select([{ value: '', label: 'Не используется' }, ...(this.dataset?.columns || []).map(column => ({ value: column.name, label: column.name }))], config.validation?.group_column || '');
    this.seed = number(config.seed ?? 42, { min: 0, max: 4294967295, step: 1 });
    this.jobs = select([{ value: 1, label: '1 поток' }, { value: 2, label: '2 потока' }, { value: 4, label: '4 потока' }], config.n_jobs || 1);
    const metrics = this.metrics();
    const defaults = { regression: ['rmse', 'mae', 'r2'], classification: ['accuracy', 'f1_macro', 'balanced_accuracy'], clustering: ['silhouette', 'calinski_harabasz', 'davies_bouldin'], ranking: ['ndcg', 'map', 'mrr'], forecasting: ['rmse', 'mae', 'smape'], panel: ['rmse', 'mae', 'smape'], anomaly: ['anomaly_fraction'], reduction: ['explained_variance', 'reconstruction_mse'] };
    const selectedMetrics = config.metrics || defaults[this.task] || [];
    this.metricList = element('div', { className: 'cml-form-grid' }, metrics.map(metric => {
      const control = checkbox(metric.name || metric.label || metric.id, selectedMetrics.includes(metric.id), { help: metric.help || metric.description || `Метрика ${metric.name || metric.id}. ${metric.direction === 'max' ? 'Большие значения лучше.' : 'Маленькие значения лучше.'}`, lesson_id: metric.lesson_id || '20-metrics-experiment' });
      control.control.value = metric.id;
      control.control.addEventListener('change', () => this.onChange());
      return control.wrapper;
    }));
    this.custom = element('input', { value: config.custom_metric || '', placeholder: 'mean(abs(error))', spellcheck: false });
    this.custom.disabled = !['regression', 'forecasting', 'panel', 'ranking'].includes(this.task);
    const searchEnabled = checkbox('Подбирать параметры автоматически', Boolean(config.search), { help: 'Поиск сравнивает комбинации настроек внутри обучающих данных. Итоговая проверочная часть остается закрытой.', lesson_id: '27-hyperparameter-search' });
    this.searchEnabled = searchEnabled.control;
    this.searchEnabled.disabled = unsupervised;
    if (unsupervised) this.searchEnabled.checked = false;
    this.searchMethod = select([
      { value: 'grid', label: 'Перебрать все комбинации' }, { value: 'random', label: 'Случайно выбрать комбинации' },
      { value: 'optuna_tpe', label: 'Optuna TPE: использовать прошлые результаты' }, { value: 'optuna_random', label: 'Optuna: случайный поиск' },
      { value: 'halving_grid', label: 'Поэтапно сокращать сетку кандидатов' }, { value: 'halving_random', label: 'Поэтапно сокращать случайные кандидаты' },
    ], config.search?.method || 'optuna_tpe');
    this.trials = number(config.search?.trials ?? 12, { min: 1, max: 100, step: 1 });
    this.searchMetric = select(metrics.map(metric => ({ value: metric.id, label: metric.name || metric.label || metric.id })), config.search?.metric || selectedMetrics[0]);
    this.direction = select([{ value: 'auto', label: 'Взять из описания метрики' }, { value: 'min', label: 'Меньше — лучше' }, { value: 'max', label: 'Больше — лучше' }], config.search?.direction || 'auto');
    this.searchRowsContainer = element('div');
    this.renderSearchRows();
    const validationPanel = element('section', { className: 'cml-panel' }, [
      heading('Как проверять качество', { help: 'Разделение имитирует будущие неизвестные данные. Независимые строки, группы и время требуют разных правил.', lesson_id: '26-cross-validation' }),
      element('div', { className: 'cml-form-grid' }, [
        field('Обучение, %', this.train, { help: 'Эти строки используются для подбора коэффициентов, деревьев и других внутренних параметров.', lesson_id: '25-data-leakage' }),
        field('Выбор настроек, %', this.val, { help: 'Эти строки сравнивают выбранные настройки. Повторный ручной выбор по ним постепенно подстраивается под эту часть.', lesson_id: '25-data-leakage' }),
        field('Итоговая проверка, %', this.test, { help: 'Итоговая проверка скрыта до завершения выбора модели. Откройте ее один раз после выбора.', lesson_id: '25-data-leakage' }),
        field('Схема перекрестной проверки', this.strategy, { help: 'Несколько разделений показывают устойчивость качества. Каждая часть заново обучает подготовку данных.', lesson_id: '26-cross-validation' }),
        field('Количество частей', this.folds, { help: 'При трех частях модель обучается три раза. Большое число частей увеличивает время расчета.', lesson_id: '26-cross-validation' }),
        field('Количество повторений', this.repeats, { help: 'Повторения создают новые разделения K-fold для оценки разброса качества.', lesson_id: '26-cross-validation' }),
        field('Промежуток между частями по времени', this.gap, { help: 'Gap исключает соседние наблюдения между обучающей и проверочной частью внутри перекрестной проверки по времени.', lesson_id: '26-cross-validation' }),
        field('Столбец группы для проверки', this.group, { help: 'Строки одной группы проверяются вместе. Это нужно, когда несколько строк относятся к одному человеку, компании или объекту.', lesson_id: 'panel-groups' }),
        field('Зерно случайности', this.seed, { help: 'Одинаковое число помогает повторить случайное разделение и обучение. Изменение числа проверяет чувствительность результата.', lesson_id: '26-cross-validation' }),
        field('Параллельные вычисления', this.jobs, { help: 'Несколько независимых расчетов могут выполняться одновременно. Для маленьких данных один поток иногда быстрее.', lesson_id: '27-hyperparameter-search' }),
      ]), shuffle.wrapper,
    ]);
    if (unsupervised) validationPanel.replaceChildren(heading('Расчет без учителя', { help: 'Алгоритм работает с выбранным рабочим датасетом. Известные классы могут использоваться только для независимой оценки найденных групп.', lesson_id: this.task === 'clustering' ? 'clustering-basics' : this.task === 'anomaly' ? 'anomaly-detection' : 'dimensionality-reduction' }), notice('Метрики относятся к рабочему датасету. Отдельной проверки качества на новых объектах у этого расчета нет.'), field('Зерно случайности', this.seed, { help: 'Помогает воспроизвести начальные условия алгоритма.', lesson_id: '26-cross-validation' }), field('Параллельные вычисления', this.jobs, { help: 'Количество вычислительных потоков.', lesson_id: '27-hyperparameter-search' }));
    const searchPanel = element('section', { className: 'cml-panel' }, [
      heading('Подбор гиперпараметров', { help: 'Гиперпараметры задают поведение алгоритма до обучения: глубину дерева, силу штрафа, количество соседей.', lesson_id: '27-hyperparameter-search' }), searchEnabled.wrapper,
      element('div', { className: 'cml-form-grid' }, [
        field('Метод поиска', this.searchMethod, { help: 'Сетка проверяет перечисленные значения. Случайный поиск пробует часть комбинаций. Optuna использует предыдущие результаты.', lesson_id: '27-hyperparameter-search' }),
        field('Количество проб', this.trials, { help: 'Одна проба означает одну комбинацию настроек. С перекрестной проверкой каждая проба обучает несколько моделей.', lesson_id: '27-hyperparameter-search' }),
        field('Главная метрика', this.searchMetric, { help: 'По этой метрике выбирается победитель. Другие метрики сохраняются для сравнения.', lesson_id: '20-metrics-experiment' }),
        field('Направление выбора', this.direction, { help: 'Ошибки уменьшают, долю верных ответов увеличивают. Автоматический режим берет направление из каталога.', lesson_id: '20-metrics-experiment' }),
      ]), this.searchRowsContainer,
      action('Добавить параметр для поиска', () => { this.searchRows.push({ savedKey: '', savedValues: '' }); this.renderSearchRows(); }),
      notice('Значения задаются в форме через запятую. Итоговая тестовая часть не участвует в поиске.'),
    ]);
    if (unsupervised) searchPanel.replaceChildren(heading('Подбор параметров', { help: 'Для поиска без учителя нужен явный внешний критерий. Он не эквивалентен обычной перекрестной проверке с правильными ответами.', lesson_id: 'clustering-metrics' }), notice('Автоматический подбор для задач без учителя отключен. Меняйте параметры модели вручную и изучайте найденные группы, устойчивость и соответствующие метрики.'));
    const metricsPanel = element('section', { className: 'cml-panel' }, [
      heading('Метрики результата', { help: 'Метрики измеряют разные виды ошибок. Выбирайте показатели по задаче и смыслу ошибок.', lesson_id: '20-metrics-experiment' }),
      action('Выбрать все метрики', () => { this.metricList.querySelectorAll('input').forEach(control => { control.checked = true; }); this.onChange(); }), this.metricList,
      field('Своя формула метрики', this.custom, { help: 'y — настоящий ответ; pred — прогноз; error = y − pred. Для числовой задачи доступны mean, abs, sqrt, sum, min, max, log, exp, clip.', lesson_id: '20-metrics-experiment' }),
    ]);
    this.diagnostics = {};
    const diagnosticDefinitions = [
      ['regularization_path', 'Как сила штрафа меняет модель', 'Обучает отдельную модель для каждой силы штрафа. Показывает коэффициенты и качество. Это разные задачи оптимизации, а не шаги обучения одной модели.', '06-ridge'],
      ['learning_curve', 'Как количество данных меняет качество', 'Повторяет обучение на частях обучающих строк разного размера. Проверочная часть остается одной и той же. Помогает оценить, стоит ли собирать больше данных.', '20-metrics-experiment'],
      ['permutation_importance', 'Важность признаков через перемешивание', 'По очереди перемешивает один исходный признак в проверочных строках и измеряет изменение качества. Это влияние на данную модель, а не доказательство причинной связи.', '24-feature-engineering'],
    ];
    const diagnosticPanel = element('section', { className: 'cml-panel' }, [heading('Дополнительные исследования', { help: 'Эти расчеты требуют дополнительных обучений или прогнозов. Тестовая часть остается закрытой. Сервер явно объясняет недоступные исследования для неподходящих моделей и разбиений.', lesson_id: '20-metrics-experiment' })]);
    for (const [key, label, help, lesson_id] of diagnosticDefinitions) {
      const control = checkbox(label, Boolean(config[key]), { help, lesson_id });
      this.diagnostics[key] = control.control; control.control.dataset.diagnostic = key;
      diagnosticPanel.append(control.wrapper);
    }
    diagnosticPanel.append(notice('Исследования поддерживают независимые строки регрессии и классификации. Для временных рядов, связанных групп и других задач требуется отдельная постановка; ее ограничения показываются в результате. Расчет может занять больше времени.'));
    this.container.replaceChildren(element('div', { className: 'cml-workspace wide-form' }, [element('div', {}, [validationPanel, diagnosticPanel]), element('div', {}, [metricsPanel, searchPanel])]));
    this.container.querySelectorAll('input,select').forEach(control => control.addEventListener('change', () => this.onChange()));
  }

  renderSearchRows() {
    if (!this.searchRowsContainer) return;
    const parameters = (this.model?.params || []).filter(param => ['int', 'integer', 'float', 'number', 'select', 'choice', 'bool', 'boolean'].includes(param.type));
    this.searchRowsContainer.replaceChildren(...this.searchRows.map((row, index) => {
      row.savedKey = row.key?.value ?? row.savedKey;
      row.savedValues = row.values?.value ?? row.savedValues;
      row.key = select([{ value: '', label: 'Выберите параметр' }, ...parameters.map(param => ({ value: param.key, label: param.label || param.key }))], row.savedKey);
      row.values = element('input', { value: row.savedValues || '', placeholder: '0.1, 1, 10', 'aria-label': 'Значения параметра через запятую' });
      return element('div', { className: 'cml-search-row' }, [
        field('Параметр', row.key, { help: 'Выберите настройку, которую поиск будет менять.', lesson_id: '27-hyperparameter-search' }),
        field('Проверяемые значения', row.values, { help: 'Для чисел: 0.1, 1, 10. Для переключателей: true, false. Для списков используйте точные значения из описания модели.', lesson_id: '27-hyperparameter-search' }),
        action('Удалить', () => { this.searchRows.splice(index, 1); this.renderSearchRows(); }),
      ]);
    }));
  }
}
