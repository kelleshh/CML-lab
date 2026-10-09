# Подготовка данных CML-lab

Рецепт подготовки — сохраняемый набор этапов и их параметров. Изменение рецепта не обучает модель и не перезаписывает исходный датасет. Конфигурация запущенного эксперимента сохраняется собственным снимком; последующая правка рецепта не меняет завершённый запуск.

## Действующий интерфейс

`PreparationApplication` в контексте данных возвращает каталог этапов, проверяет сохраняемую конфигурацию и запускает предпросмотр через порт `PreparationGateway`. Домены и application не импортируют pandas/sklearn. Адаптеры `preparation.py` и `preparation_preview.py` создают обычные sklearn Pipeline/ColumnTransformer. Они не читают HTTP-запросы и не хранят рецепты в базе.

- `GET /api/cml/preprocessing/stages` — этапы и пересэмплировщики, со схемами параметров и ссылками на уроки.
- `POST /api/cml/preprocessing/preview` — полный запрос опыта; свежая подготовка train без обучения выбранного алгоритма.
- CRUD `/api/cml/preprocessor-recipes` — создание, чтение, обновление и удаление рецептов. `expected_revision` защищает от потери параллельного изменения.

Рецепт содержит не более 32 этапов. У этапа: `adapter_id`, `columns`, `params`, `enabled`. `columns:[]` выбирает все подходящие столбцы; для текста и удаления столбцов выбор должен быть явным. Последовательность этапов одной ветки исполняется в указанном порядке. Их selectors должны совпадать; разные ветки используют непересекающиеся столбцы. Глобальные отбор/разложение работают со всеми подготовленными столбцами и требуют `columns:[]`. Неизвестные настройки, несовместимая задача и пересекающиеся разные ветки отклоняются.

```json
{
  "steps": [
    {"adapter_id":"numeric.impute","columns":[],"params":{"strategy":"median","add_indicator":true},"enabled":true},
    {"adapter_id":"numeric.scale","columns":[],"params":{"method":"robust"},"enabled":true},
    {"adapter_id":"categorical.onehot","columns":[],"params":{"min_frequency":3,"handle_unknown":"infrequent_if_exist"},"enabled":true}
  ],
  "resampling":{"method":"none"}
}
```

В рецепте `steps:[]` числовая ветка получает медиану и StandardScaler; категориальная — наиболее частое значение и one-hot. Не выбранные отдельным этапом столбцы остаются входами с этим автоматическим рецептом. `columns.drop` исключает столбцы. Даты datetime получают календарные признаки. Строки дат разбираются только при явном `datetime.extract`, обычная строка автоматически считается категорией.

## Реализованные этапы

Каталог содержит 25 адаптеров. Один адаптер может предлагать несколько настоящих библиотечных методов.

| ID | Результат и ограничения |
|---|---|
| `columns.drop` | Исключает явно выбранные столбцы; исходная таблица сохраняется |
| `numeric` | Совместимый составной рецепт прежней версии; нельзя объединять с другими числовыми этапами той же ветки |
| `numeric.impute` | Среднее, медиана, частое значение, константа, KNN или IterativeImputer; индикаторы пропусков; этот этап первым в числовой ветке |
| `numeric.scale` | Standard, Robust, MinMax, MaxAbs, нормирование строк или отключение; MinMax clip задаётся явно |
| `numeric.power` | Signed log1p/sqrt, Yeo-Johnson, Box-Cox, квантильное normal/uniform; Box-Cox требует положительные значения |
| `numeric.clip` | Квантили train ограничивают крайние значения; строки holdout не удаляются |
| `numeric.polynomial` | Степени и взаимодействия 1–5; размер оценивается до создания матрицы |
| `numeric.bin` | KBins uniform/quantile/kmeans, ordinal или плотный one-hot; размер проверяется до создания выходной матрицы |
| `numeric.spline` | B-сплайн базис sklearn: 2–30 узлов, степень 1–5, uniform/quantile, явное поведение вне train-диапазона; узлы учатся на train |
| `categorical.onehot` | Редкие категории, ограничение их числа, неизвестные значения, sparse/dense |
| `categorical.ordinal` | Заданный или автоматический порядок; неизвестная категория −1; автоматический порядок не имеет предметного смысла |
| `categorical.frequency` | Частота категории в train; неизвестная категория 0 |
| `categorical.hash` | Фиксированные хеш-корзины, возможные коллизии и выбор знака |
| `categorical.target` | Средняя цель со сглаживанием и внутренним cross-fitting; только regression/classification; отключён для временных/групповых разделений |
| `text.tfidf`, `text.count`, `text.hash` | Словесные/символьные цепочки, ограниченный словарь/корзины, разреженная матрица; английские стоп-слова автоматически не применяются |
| `datetime.extract` | Календарные поля и sine/cosine циклов в заданном часовом поясе; ошибки разбора заполняются как пропуски |
| `selection.variance` | Отбор без цели по дисперсии train |
| `selection.univariate` | F/MI по задаче или classification chi2; chi2 требует неотрицательные входы; точное число k; тип признаков MI задается как auto/continuous/discrete |
| `selection.model` | SelectFromModel с ограниченным ExtraTrees; классификатор или регрессор по задаче |
| `reduction.pca`, `reduction.svd`, `reduction.ica`, `reduction.nmf` | Компоненты вместо исходных признаков; PCA/ICA требуют dense; SVD поддерживает sparse; NMF требует неотрицательные входы |

One-hot/TF-IDF словари, медианы, квантили, масштаб, отбор и компоненты обучаются на train соответствующего fold. Для target encoder обучающий путь использует `fit_transform`, а не `fit().transform()`. В sklearn 1.8 это разные результаты: первый путь не кодирует строку её собственной целью. Целочисленная цель регрессии принудительно получает `target_type="continuous"`, чтобы auto не считал каждое число отдельным классом.

### Итеративное заполнение и сплайны

`numeric.impute` с `strategy: "iterative"` использует экспериментальный `sklearn.impute.IterativeImputer` с BayesianRidge, медианой для начального заполнения, `skip_complete=True` и фиксированным seed. Один столбец с пропусками предсказывается по другим обучающим столбцам; цикл повторяется до сходимости или `max_iter` (1–30). `n_nearest_features` (1–30) ограничивает число используемых других признаков. Полностью пустые train-столбцы сохраняются и заполняются нулем. Если train-столбец не имел пропусков, новый пропуск на прогнозе заполняется начальной медианой: `skip_complete=True` не обучает модель для такого столбца. Это одна заполненная таблица, а не статистическое множественное заполнение. Цель эксперимента не используется. Бюджет: 10 000 строк, 100 столбцов и 500 000 ячеек до заполнения.

`numeric.spline` заменяет каждое числовое измерение локальными полиномиальными функциями. Например, 4 узла и степень 3 создают 5 столбцов на один вход: `include_bias=False` убирает избыточный базис. `quantile` ставит узлы по распределению train, `uniform` между минимумом и максимумом train. Сохраненный рецепт применяет те же узлы к новым строкам. `constant` продолжает крайним значением, `linear` линейно, `continue` крайним полиномом, `error` запрещает значения вне train-диапазона. Сплайн не создает взаимодействий между разными исходными столбцами. Для фиксированного календарного периода используйте `datetime.extract` с циклическими признаками; автоматический train-диапазон сплайна не является известным годовым/суточным периодом.

Для взаимной информации `discrete_features=auto` следует соглашению sklearn: dense вход считается непрерывным, sparse — дискретным. Смешанная sparse матрица с непрерывными числовыми, календарными, частотными или целевыми кодировками отклоняется: иначе sklearn принял бы каждое измерение за отдельную дискретную категорию. Используйте плотный one-hot, предварительное SVD или сознательно заданный `discrete` для настоящих дискретных счетчиков. Вручную заданные порядки категорий используют то же строковое представление, что входные категории; повтор после преобразования (например, `10` и `"10"`) запрещен.

## Пересэмплирование

Каталог содержит 19 позиций, включая отключение и два явно недоступных метода. Реально работают 12 способов классификации: RandomOver/Under, SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE, KMeansSMOTE, TomekLinks, ENN, NearMiss, SMOTEENN, SMOTETomek. В регрессии работают копирование/уменьшение редких областей цели и исправленные SMOTER/SMOGN. Версии, лицензии и изменения исследовательских модулей сохранены в [PROVENANCE.md](../linear_lab/_vendor/PROVENANCE.md).

Самплер получает только train. Validation/test и исходные train-метрики сохраняют исходные строки. Обычные классификационные/регрессионные самплеры запрещены для ranking, forecasting, panel, clustering, anomaly и reduction. Веса строк нельзя объединить с самплером без правила переноса весов.

SMOTE/ADASYN и варианты генерации/очистки в этом маршруте требуют исходные числовые признаки. При закодированных категориях допускается только случайное копирование/уменьшение целых строк. SMOTENC/SMOTEN перечислены с `available:false`: необходим отдельный план, сохраняющий категории до синтеза; обычный SMOTE не выдаётся за такой план. Не соседей по всей таблице, а размер редкого класса внутри каждого train ограничивает параметр neighbours.

## Ряды и панели

`build_temporal_frame` возвращает `TemporalDataset`: X, y, origins, targets, entities, row_indices, history_config. Время разбирается и сортируется внутри объекта. Каждая пара объект/время уникальна. Требуется регулярная временная сетка; пропущенные даты не молча превращаются в соседние периоды. Месячная календарная частота допускает месяцы разной длины.

Origin t — первая неизвестная точка. `lag1=y(t−1)`. Горизонт h предсказывает `y(t+h−1)`; rolling mean/std/min/max сначала сдвигают цель на один шаг. В панели история одного объекта не пересекает другой. Внешние признаки на origin должны быть действительно доступны к прогнозу. Для разделения engine исключает train-метки, чьё время достигает начала validation, и validation-метки, достигающие начала test.

Проверка движка использует обновляемую историю прошлых наблюдений: перед следующим origin уже известно фактическое предыдущее значение. Это отдельный режим от одновременного прогноза всего будущего блока. `TemporalArtifact.forecast` для horizon=1 строит до 20 будущих шагов на объект, добавляя в историю собственные предсказания. Значения target из будущей таблицы игнорируются. Для прямой модели h>1 допустим один origin на объект; результат содержит и origin, и время целевой метки. Выбранные будущие внешние признаки передаются явно. Пропуск будущих шагов и неизвестный объект отклоняются.

## Предпросмотр и бюджеты

Предпросмотр заново применяет то же разбиение и компилятор, что обучение. До/после показываются по одним исходным индексам. Сэмплированная таблица показывается отдельно: у синтетики нет исходного индекса. Для обучения без учителя scope=`working_dataset`: это явно вся выбранная таблица и не holdout-оценка.

В ответе до 100 строк и 100 столбцов для таблиц/графиков; `features` описывает показанный срез, `all_features` и `feature_count` — фактическую полную матрицу. Статистики вычисляются по train, гистограммы подготовленного среза используют не более 2000 строк. Разреженные preview срезы не превращают весь датасет в dense.

Бюджеты: исходные данные до 5 млн ячеек; dense до 2000 выходных признаков/15 млн ячеек; sparse до 20 000 признаков/15 млн ненулевых значений. KNN-заполнение до 20 000 строк. Итеративное заполнение до 10 000 строк/100 столбцов/500 000 ячеек. Самплеры до 10 000 входных строк/200 признаков/1 млн ячеек; выход до 100 000 строк/15 млн ячеек. План классификационного копирования/генерации проверяется до выделения выходной матрицы. One-hot, полином, сплайн, календарные признаки и интервализация проверяют размер перед созданием плотного результата. При прогнозе полином/сплайн/интервалы/компоненты также ограничивают выходное число ячеек. Модель может иметь более строгий собственный бюджет.

Нельзя автоматически объявлять MinMax совместимым с неотрицательной моделью: новые значения ниже train minimum могут стать отрицательными без clip. Hashing с alternate_sign=true также даёт отрицательные входы. Ограничения NB/NMF/chi2 проверяются по реальной преобразованной матрице. Sparse сохраняется для совместимых алгоритмов; dense требуется явному ограниченному маршруту.

## Исследовательский каталог для расширения

Следующая таблица содержит изученные кандидаты и приоритеты; доступность определяется только действующим каталогом API выше. RFE/RFECV, дополнительные under-samplers, kernel features, UMAP, joins и target transform в этом рецепте не реализованы. IterativeImputer и SplineTransformer доступны в описанном выше ограниченном маршруте.


Приоритет A: обязательная реализация для работы нескольких задач. B: расширение адаптерами после проверок. C: исследовательские/дорогие методы с отдельными бюджетами. Все обучаемые методы находятся внутри fold.

| Семейство | Методы и библиотека | Приоритет | Ограничения |
|---|---|---|---|
| Пропуски numeric | SimpleImputer mean/median/most_frequent/constant; MissingIndicator; KNNImputer | A | Сохранить пустые столбцы, реальные feature names; KNN бюджет по строкам/расстояниям |
| Пропуски multivariate | IterativeImputer + BayesianRidge/forest | B | Experimental sklearn API, max_iter/n_nearest_features бюджет; не выдавать single imputation за multiple imputation |
| Масштаб numeric | StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler, Normalizer | A | Normalizer нормирует строку, остальные столбец; center=False на sparse |
| Распределение | Signed log1p/sqrt, PowerTransformer Yeo-Johnson/Box-Cox, QuantileTransformer uniform/normal | A/B | Box-Cox строго positive; quantile finite, отдельные n_quantiles/subsample; смысл единиц меняется |
| Выбросы | Train-quantile winsorization; IQR bounds; IsolationForest detector | A/B | Обрезка и train-only row rejection различаются; holdout строки не удаляются для красивой метрики |
| Категории | OneHotEncoder min_frequency/max_categories/unknown; OrdinalEncoder fixed-order/unknown/missing; TargetEncoder crossfit | A/B | Unknown explicit; target encoder supervised и restricted split |
| Редкие категории | OneHotEncoder infrequent category; frequency encoding | A/B | Counts учатся только train; frequency не целевая кодировка |
| Текст | CountVectorizer, TfidfVectorizer word/char ngrams; HashingVectorizer | A/B | Vocabulary/IDF train-only; sparse, max_features/buckets; русский текст не удаляется английскими stop_words |
| Даты | year/month/day/weekday/hour, elapsed time, sine/cosine cycles | A/B | Explicit parse format/timezone; weekday category vs numeric distances объяснить |
| Взаимодействия | PolynomialFeatures interaction_only, bounded ratios/differences; SplineTransformer | A/B | Не eval; строго ограниченные операции; denominator zero policy и имя источников |
| Биннинг | KBinsDiscretizer uniform/quantile/kmeans; Binarizer | A/B | Bins выучены train; ordinal bins не интерпретировать как одинаковые расстояния |
| Отбор без y | VarianceThreshold, correlation pruning | A/B | Train-only; constant empty outputs rejected; threshold зависит от scale |
| Отбор с y | SelectKBest/Percentile f_regression/f_classif/MI/chi2; SelectFromModel | A/B | Score по task; chi2 только неотрицательные count-like values; selector fit fold-only |
| Wrapper отбор | RFE, RFECV, SequentialFeatureSelector | B/C | Много fit; вложенная CV и общий fit budget; не добавить как одну дешёвую галочку |
| Снижение размерности | PCA, TruncatedSVD, FastICA, NMF, IncrementalPCA, random projections | A/B | PCA centers; SVD sparse; NMF nonnegative; компоненты вместо исходных признаков объясняются |
| Kernel features | RBFSampler, Nystroem, PolynomialCountSketch | B | Матрицы размера и kernel domain budget; learned landmark sampling train-only |
| Manifold просмотр | t-SNE, Isomap, MDS, UMAP optional | B/C | t-SNE/MDS no generic out-of-sample transform; отдельный EDA инструмент, не ложный pipeline |
| Class over-sampling | RandomOverSampler, SMOTE, SMOTENC, SMOTEN, ADASYN, BorderlineSMOTE, SVMSMOTE, KMeansSMOTE | A/B | Число соседей меньше размера класса в каждом fold; категории требуют правильного варианта |
| Class under-sampling | RandomUnderSampler, TomekLinks, ENN/RepeatedENN/AllKNN, NearMiss, ClusterCentroids, CondensedNN, OneSidedSelection, NeighborhoodCleaningRule, InstanceHardnessThreshold | A/B/C | Cleaning меняет границу; centroid только continuous numeric; pairwise/inner estimator budgets |
| Class combine | SMOTEENN, SMOTETomek | B | Настройки компонентов явные; class-only и numeric-only исходный маршрут |
| Regression sampling | Existing random_over/random_under, corrected SMOTER/SMOGN | A | Сохранить vendor provenance, licenses, affine/determinism tests; ≤10k rows/200 features/1m cells input |
| Ряд/panel | positive lag, shifted rolling mean/std/min/max/median/ewm; entity grouping; differencing; seasonal indicators | A/B | Forecast origin/horizon/history port; проверять fixed-origin vs rolling protocol |
| Табличные join | Cardinality checks, backward merge_asof, entity train-history aggregation | B/C | Внешние данные versioned и timestamp availability; many-to-many blowup guard |
| Target transform | TransformedTargetRegressor log/quantile/standard | B | Only regression; inverse_transform реальные predictions, metrics в исходной единице; domain positive |
| Weighting | sample_weight/class_weight choice | A/B | Model capability; original evaluation distribution; weight не меняет число строк |

Каталог не является обещанием безопасно соединять любой метод с любым другим. Ошибка совместимости возвращает stage_id, code, понятное русское сообщение и ссылку на конкретный урок.


## Источники


Основные документы относятся к pin sklearn 1.8.0. External optional libraries требуют версионной проверки перед включением.

- https://scikit-learn.org/1.8/common_pitfalls.html — split, fit/transform, leakage.
- https://scikit-learn.org/1.8/modules/compose.html — Pipeline и ColumnTransformer.
- https://scikit-learn.org/1.8/modules/preprocessing.html — scaler/encoder/power/bins.
- https://scikit-learn.org/1.8/modules/impute.html — Simple/KNN/Iterative и empty features.
- https://scikit-learn.org/1.8/modules/generated/sklearn.impute.IterativeImputer.html — экспериментальный API, вычислительная стоимость и skip_complete.
- https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.SplineTransformer.html — размер B-сплайн базиса и экстраполяция.
- https://scikit-learn.org/1.8/modules/generated/sklearn.feature_selection.mutual_info_classif.html — discrete_features и ограничения sparse.
- https://feat.engineering/ — открытая книга Kuhn и Johnson: представления признаков, выбор и воспроизводимые проверки.
- https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.TargetEncoder.html — cross-fitting и innercv ограничения.
- https://scikit-learn.org/1.8/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html — vocabulary/IDF/options.
- https://scikit-learn.org/1.8/modules/generated/sklearn.feature_extraction.text.HashingVectorizer.html — collisions/alternate_sign/stateless.
- https://scikit-learn.org/1.8/modules/generated/sklearn.feature_selection.SelectKBest.html — supervised task scores.
- https://scikit-learn.org/1.8/modules/decomposition.html — PCA/SVD/ICA/NMF.
- https://scikit-learn.org/1.8/auto_examples/applications/plot_time_series_lagged_features.html — shift before rolling и optimistic shuffled validation.
- https://scikit-learn.org/1.8/auto_examples/applications/plot_cyclical_feature_engineering.html — period encoding.
- https://imbalanced-learn.org/stable/common_pitfalls.html — fold-local sampling.
- https://imbalanced-learn.org/stable/references/generated/imblearn.pipeline.Pipeline.html — sampler Pipeline contract.
- https://imbalanced-learn.org/stable/over_sampling.html — class algorithms/type constraints.
- https://imbalanced-learn.org/stable/under_sampling.html — cleaning/selection/generation distinction.
- https://imbalanced-learn.org/stable/combine.html — SMOTEENN/SMOTETomek.
- https://feature-engine.trainindata.com/en/latest/user_guide/datetime/index.html — datetime adapters optional.
- https://feature-engine.trainindata.com/en/1.6.x/user_guide/timeseries/forecasting/WindowFeatures.html — shifted windows optional.
- https://pandas.pydata.org/pandas-docs/version/2.2/reference/api/pandas.core.groupby.DataFrameGroupBy.shift.html — entity-local lag.
- https://skforecast.org/latest/user_guides/table-of-contents — forecasting adapters and backtest candidates.

