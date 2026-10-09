# CML-lab: контракт ML-каталога

Статус: предложение ML-контекста. Список ниже задает объем реализации и проверки, а не объявляет уже существующие функции. В рабочий каталог включается только алгоритм с реальной фабрикой, проверенным обучением и понятным ограничением входных данных.

## Задача, семейство и реализация

Задачи имеют ключи `regression`, `classification`, `clustering`, `ranking`, `forecasting`, `panel`. Семейство описывает устройство алгоритма, задача описывает смысл ответа. Один регрессор можно использовать в обычной регрессии и в прогнозировании через адаптер лагов. Разделение по задачам не требует копировать фабрику модели.

Алгоритмы каталога неизменяемы. Пользователь создает рецепт: имя, задача, идентификатор алгоритма, параметры и ссылка на версию препроцессора. Обучение создает отдельный артефакт обученной модели. Обновление рецепта не меняет завершенный эксперимент.

```json
{
  "id": "random_forest_classifier",
  "name": "Случайный лес · классификация",
  "tasks": ["classification"],
  "family": "forest",
  "family_label": "Леса",
  "backend": "sklearn",
  "available": true,
  "availability_reason": null,
  "description": "Деревья голосуют за класс строки.",
  "source": "https://scikit-learn.org/1.8/modules/generated/sklearn.ensemble.RandomForestClassifier.html",
  "lesson_id": "cml-random-forest",
  "params": [{
    "key": "n_estimators", "label": "Количество деревьев", "type": "int",
    "default": 100, "min": 1, "max": 1000, "step": 1,
    "help": "Больше деревьев обычно делает результат устойчивее и увеличивает время обучения.",
    "help_key": "model.n_estimators", "lesson_id": "cml-random-forest"
  }],
  "capabilities": {
    "predict": true, "predict_proba": true, "decision_function": false,
    "fit_predict": false, "partial_fit": false, "staged_predict": false,
    "coefficients": false, "feature_importance": true, "transductive": false,
    "input_domain": "real", "requires_dense": false,
    "allows_missing": true, "native_categorical": false
  }
}
```

Схема параметров сохраняет ключи прежнего интерфейса. `type` принимает `int`, `float`, `bool`, `select`; значения проверяются до создания estimator. Произвольный импорт, Python-код и `eval` в рецепте не нужны. Неизвестные параметры отклоняются. Для каждого поля обязательны русское название, пример смысла, `help_key` и учебная ссылка.

`allows_missing` описывает библиотечный алгоритм; действующий рецепт все равно проверяет результат своего препроцессора. Масштабирование, заполнение пропусков и выбор признаков обучаются внутри каждой обучающей части. Возможность рассчитывать коэффициенты, вероятности или историю обучения задается отдельно: дереву не приписывается `coef_`, DBSCAN не приписывается `predict`.

## Реализуемый каталог

29 существующих идентификаторов линейной регрессии сохраняются. Следующие алгоритмы можно добавить на установленном sklearn 1.8 без новой ML-библиотеки.

| Семейство | Регрессия | Классификация |
|---|---|---|
| Линейные | существующие 29 моделей; `KernelRidge`, `PLSRegression` | `LogisticRegression`, `RidgeClassifier`, `SGDClassifier`, `Perceptron` |
| Деревья | `DecisionTreeRegressor`, `ExtraTreeRegressor` | `DecisionTreeClassifier`, `ExtraTreeClassifier` |
| Леса | `RandomForestRegressor`, `ExtraTreesRegressor` | `RandomForestClassifier`, `ExtraTreesClassifier` |
| Бустинг | `GradientBoostingRegressor`, `HistGradientBoostingRegressor`, `AdaBoostRegressor` | `GradientBoostingClassifier`, `HistGradientBoostingClassifier`, `AdaBoostClassifier` |
| Ансамбли | `BaggingRegressor`, `VotingRegressor`, `StackingRegressor` | `BaggingClassifier`, `VotingClassifier`, `StackingClassifier` |
| Соседи | `KNeighborsRegressor`, `RadiusNeighborsRegressor` | `KNeighborsClassifier`, `RadiusNeighborsClassifier`, `NearestCentroid` |
| Метод опорных векторов | `SVR`, `NuSVR` и существующий `LinearSVR` | `SVC`, `NuSVC`, `LinearSVC` |
| Байесовские | существующие `BayesianRidge`, `ARDRegression` | `GaussianNB`, `MultinomialNB`, `ComplementNB`, `BernoulliNB` |
| Дискриминантный анализ | — | `LinearDiscriminantAnalysis`, `QuadraticDiscriminantAnalysis` |
| Гауссовские процессы | `GaussianProcessRegressor` | `GaussianProcessClassifier` |
| Контрольный ответ | `DummyRegressor` | `DummyClassifier` |

Кластеризация: `KMeans`, `MiniBatchKMeans`, `BisectingKMeans`, `AgglomerativeClustering`, `DBSCAN`, `HDBSCAN`, `OPTICS`, `MeanShift`, `AffinityPropagation`, `SpectralClustering`, `Birch`, `GaussianMixture`, `BayesianGaussianMixture`.

Для `RadiusNeighbors*` строки без соседей требуют выбранной политики, иначе возвращается объяснение ошибки. `CategoricalNB` не следует автоматически применять к стандартному one-hot-преобразованию: нужен отдельный проверенный ordinal-рецепт. Gaussian process и методы с полной матрицей попарных расстояний получают явный предел строк; разрешить их на 100 000 строках по умолчанию нельзя.

Дополнения XGBoost, LightGBM и CatBoost предоставляют classifier, regressor и ranker. Они устанавливаются отдельным набором зависимостей. Каталог показывает реальную доступность импорта и причину отказа, а обязательные sklearn-бустинги остаются работоспособны без дополнений. CPU-вариант XGBoost экономит место; LightGBM на macOS может потребовать OpenMP. Наличие пакета проверяется настоящим импортом, поскольку найденный модуль еще не означает загрузку нативной библиотеки.

Ансамбль хранит вложенные рецепты, а не сериализованные estimator. Ограничения: белый список компонентов, задача каждого компонента совпадает, глубина не более 3, не более 8 компонентов, отсутствие циклов. Soft voting требует `predict_proba` у каждого компонента. Stacking обучает метамодель на out-of-fold прогнозах; прогнозы той же обучающей строки не заменяют OOF.

## Данные, результаты и оценка

| Задача | Роли столбцов | Обучение и разбиение | Результаты |
|---|---|---|---|
| Регрессия | обязательная числовая цель; признаки | train/validation/test; положительная цель для подходящих GLM | прогнозы, ошибки, MAE/RMSE/R² и существующие метрики |
| Классификация | обязательная категориальная цель; признаки | стратификация при достаточном числе строк каждого класса; encoder только train | матрица ошибок, прогнозы классов, доступные вероятности/оценки |
| Кластеризация | признаки; необязательная эталонная метка | без цели в `fit`; явно различать исследование полного набора и out-of-sample оценку | номер группы, шум `-1`, размеры групп, центры при наличии |
| Ранжирование | признаки, релевантность, обязательный `query_group` | запрос целиком входит в один split; сортировка строк по запросу перед fit | оценки релевантности, порядок внутри запроса, метрики по запросам |
| Временной ряд | признаки, цель, время, лаги, горизонт | хронологически; признак использует только доступное прошлое | прогноз по времени, фактическая цель, ошибки по горизонту |
| Панель | те же роли и `entity` | лаги внутри сущности; выбор сценария прогноза известных сущностей или переноса на новые | прогноз и ошибки по сущности и времени |

`target`, `query_group`, `time`, `entity`, `sample_weight` исключаются из признаков, если специально не создан безопасный производный признак. Для классификации доступны accuracy, balanced accuracy, macro/weighted F1, precision/recall, MCC. Log loss и многоклассовый ROC AUC требуют вероятностей; бинарный ROC AUC также допускает decision scores. Отсутствующая способность отключает метрику с причиной.

Для кластеров доступны silhouette, Calinski–Harabasz и Davies–Bouldin при допустимом числе групп. Метка шума исключается из внутренних метрик, а число исключенных строк показывается. ARI, AMI/NMI, homogeneity, completeness, V-measure используют эталонную метку только после обучения. Объединение всех строк в одну группу не превращается в silhouette=0: метрика недоступна с причиной. Предсказание новой строки не создается искусственно для трансдуктивного метода.

Для ранжирования доступны NDCG@k, MAP@k и MRR@k, сначала внутри каждого запроса, затем среднее по запросам. Нельзя считать sklearn `ndcg_score` по всем строкам как по одному запросу. Смысл gain должен совпадать с выбранной библиотекой: линейная релевантность и `2**relevance-1` дают разные числа. MAP/MRR используют явно заданное определение релевантного документа. Все нулевые и одноэлементные запросы учитываются отдельно и объясняются.

Для forecasting/panel MAE, RMSE, sMAPE и MASE рассчитываются на исходном масштабе цели. Знаменатель MASE получают только из обучения. Нулевой знаменатель делает метрику недоступной. `horizon=1` в walk-forward означает, что после наблюдения предыдущего шага можно использовать его факт как лаг. Прогноз всех будущих шагов из одной точки требует рекурсивных собственных прогнозов или отдельных direct-моделей; эти режимы не смешиваются.

Панель имеет два разных проверяемых сценария: будущее известных сущностей (хронологическое разбиение каждого ряда) и новые сущности (entity holdout). Второй сценарий не обещает прогноз без необходимой начальной истории; требуемый объем истории показывается в форме.

## История и экспорт

Настоящие шаги SGD берутся из `partial_fit`; бустинг использует staged predictions или библиотечный eval history. Обычный `fit` без публичной истории дает один факт завершения, время и финальные метрики. Плавная анимация не создает вымышленные итерации. Набор изображений коэффициентов и поверхности потерь остается дополнительным режимом линейной модели.

Joblib/Pickle сохраняют полный собственный обученный конвейер. Skops и ONNX проверяют совместимость каждого преобразования и estimator; отсутствие converter возвращает причину. Нативные JSON/UBJ XGBoost, текст LightGBM и CBM CatBoost сохраняют сам estimator; для работы с исходными данными рядом необходимы препроцессор, схема ролей, версия рецепта и зависимости. Экспорт отдельного estimator не называется полным raw-input конвейером.

## Основные источники

- [sklearn 1.8: задачи и алгоритмы с учителем](https://scikit-learn.org/1.8/supervised_learning.html)
- [sklearn 1.8: кластеризация, predict и ограничения оценки](https://scikit-learn.org/1.8/modules/clustering.html)
- [sklearn: метрики и необходимые ответы модели](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [sklearn 1.8: лаговые признаки и временное разбиение](https://scikit-learn.org/1.8/auto_examples/applications/plot_time_series_lagged_features.html)
- [XGBoost: learning to rank, qid, групповые метрики](https://xgboost.readthedocs.io/en/stable/tutorials/learning_to_rank.html)
- [LightGBM: LGBMRanker и размеры групп](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRanker.html)
- [CatBoost: CatBoostRanker.fit и group_id](https://catboost.ai/docs/en/concepts/python-reference_catboostranker_fit)
- [XGBoost: установка CPU-пакета](https://xgboost.readthedocs.io/en/stable/install.html)
- [LightGBM: установка и OpenMP](https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html)

Исследованы официальные API и установленный sklearn 1.8.0. Это проверяемый набор classic ML; слова «весь интернет» и «абсолютно все алгоритмы» не описывают достижимый критерий готовности.
