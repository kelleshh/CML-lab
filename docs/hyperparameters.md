# Параметры реальных исполнителей

CML-lab показывает имена классов и аргументов Python API без переименования: `RandomForestClassifier`, `max_depth`, `min_samples_leaf`, `hidden_layer_sizes`. Русские объяснения находятся в подсказках и учебнике. Параметр передается реальному конструктору выбранной библиотеки.

Схема модели строится из сигнатуры и `get_params(deep=False)` установленного исполнителя. У sklearn варианты и числовые ограничения берутся из объявленных ограничений параметров. XGBoost и LightGBM дополнительно получают именованные параметры бустера, принимаемые через `**kwargs`. Весь бесконечный набор произвольных имен `**kwargs` не считается параметрами модели: неизвестное имя отвергается.

Каждый параметр содержит точное имя, тип формы, выбранное значение, библиотечное значение `native_default`, исходную ссылку, описание и собственный адрес урока. Выбранное значение может отличаться от библиотечного: лаборатория сохраняет воспроизводимые начальные настройки старых проектов. `native_default` позволяет увидеть эту разницу.

## Значения в формах

| Вид значения | Как задать |
|---|---|
| Число | `0.1` или `100` в числовом поле |
| Логическое значение | Checkbox для обычного `bool`; `true`, `false` или `null` в JSON для объединенных типов |
| Официальное имя варианта | Выбрать вариант; при JSON ввести строку в кавычках: `"balanced"` |
| Автоматический выбор | Пустое nullable поле либо `null` в JSON |
| Размеры слоев MLP | `[64, 32]` |
| Веса именованных компонентов | `[2, 1]` |
| Веса кодов классов | `{"0": 1, "1": 3}`; числовые ключи преобразуются в коды классов |
| Бесконечный библиотечный предел | `{"special": "inf"}`, например `max_eps` OPTICS |

Не все сочетания допустимы. Например, `Ridge(solver="lbfgs")` требует `positive=True`, а `SVC(probability=False)` не возвращает вероятности. Конструктор и библиотечная проверка сообщают ошибку для несовместимых значений; успешное создание объекта не гарантирует, что параметр подходит размеру конкретного датасета.

`random_state` и `random_seed` принимают целое число или `null`. Явное число сохраняется. `null` получает `seed` проекта. `n_jobs`, `thread_count` и `num_threads` принимают от 1 до 8; `null` получает лимит запуска. Параметры воспроизводимости не являются способом улучшить качество перебором test.

## Вложенные модели

`estimator` и `final_estimator` задаются декларацией из каталога:

```json
{"algorithm_id": "ridge", "params": {"alpha": 0.5, "fit_intercept": true}}
```

`estimators` для Voting и Stacking задается списком именованных компонентов:

```json
[
  {"name": "tree", "algorithm_id": "decision_tree_classifier", "params": {"max_depth": 3}},
  {"name": "linear", "algorithm_id": "logistic_regression", "params": {"C": 0.5}}
]
```

Допускается от двух до восьми компонентов и три уровня вложенности. Имена уникальны и не содержат `__`. Вложенная модель должна поддерживать задачу проекта. `voting="soft"` требует `predict_proba` у каждого компонента. Stacking работает с независимыми строками; временные и групповые схемы проверки для него отклоняются, в том числе внутри другого ансамбля.

`covariance_estimator` выбирает `EmpiricalCovariance`, `ShrunkCovariance`, `LedoitWolf`, `OAS`, `GraphicalLasso`, `GraphicalLassoCV` или `MinCovDet`:

```json
{"class": "LedoitWolf", "params": {"assume_centered": false}}
```

## Ядра GaussianProcess

Имена: `RBF`, `Matern`, `DotProduct`, `WhiteKernel`, `ConstantKernel`, `RationalQuadratic`, `ExpSineSquared`. У каждого ядра собственные аргументы Python API. Сложение и умножение сохраняют библиотечную семантику:

```json
{
  "op": "sum",
  "left": {"class": "RBF", "params": {"length_scale": 1.0}},
  "right": {"class": "WhiteKernel", "params": {"noise_level": 0.1}}
}
```

`length_scale_bounds` и подобные границы записываются массивом из двух чисел либо официальным вариантом `"fixed"`. Глубина выражения ограничена десятью уровнями.

## Функции и callbacks

JSON-рецепт содержит данные. Строки не выполняются через `eval` или `exec`; произвольные импорты не разрешены. Параметр, поддерживающий функцию, может использовать зарегистрированную декларацию `{"function": "sklearn.metrics.pairwise.linear_kernel"}`. Реестр включает функции NumPy `mean`, `median`, `sum`, `min`, `max`; sklearn metrics `accuracy_score`, `r2_score`, `mean_squared_error`, `mean_absolute_error`; pairwise `linear_kernel`, `rbf_kernel`, `polynomial_kernel`, `sigmoid_kernel`, `cosine_similarity`. Сигнатура функции должна соответствовать конкретному параметру.

XGBoost `callbacks` принимает список деклараций `EarlyStopping`, `EvaluationMonitor`, `LearningRateScheduler` с `class` и `params`. Например, `LearningRateScheduler` может получать список скоростей. Callback с early stopping требует контрольного набора внутри обучения; он не дает доступа к финальному test. Произвольный custom objective, custom callback или Python объект через JSON не загружается. В этих случаях требуется отдельный проверенный адаптер, а не текст программы в числовом поле.

## Совместимость сохраненных рецептов

Старые поля `hidden_units`, `components`, `base_estimators` читаются как aliases, но формы показывают `hidden_layer_sizes` и `estimators`. Старые `max_features="all"`, `class_weight="none"`, sklearn `max_depth=0` преобразуются в native значения. CatBoost и LightGBM проверяют группы официальных aliases: два разных измененных синонима одной настройки вызывают ошибку. Выбор alias в форме с остальными неизмененными defaults действительно меняет модель.

## Проверка

```bash
python -m pytest tests/test_cml_catalogue.py tests/test_cml_hyperparameters.py -q
```

Проверки сопоставляют все опубликованные поля с реальными конструкторами, обучают продвинутые конфигурации, сериализуют MLP с несколькими слоями, проверяют именованные ансамбли, составные ядра, параметры библиотек бустинга и запрет неизвестных исполняемых деклараций.

Первичные справочники: [scikit-learn API](https://scikit-learn.org/1.8/api/index.html), [XGBoost parameters](https://xgboost.readthedocs.io/en/stable/parameter.html), [LightGBM parameters](https://lightgbm.readthedocs.io/en/stable/Parameters.html), [CatBoost parameters](https://catboost.ai/docs/en/references/training-parameters/).
