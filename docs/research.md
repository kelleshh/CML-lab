# Основания моделей и визуализаций

Документ описывает математические соглашения `linear_lab/models.py` и ограничения библиотечных интерфейсов. Исследование проведено 8 октября 2026 года. Проверяемая среда: scikit-learn 1.8.0, skglm 0.5, CVXPY 1.9.3, SciPy 1.17.0. Ссылки `stable` могут позже вести на другую версию; воспроизводимость определяется зависимостями проекта.

## Похожие инструменты

| Инструмент | Что показывает его официальная страница | Что полезно для этой лаборатории |
| --- | --- | --- |
| [Scienxlab Machine Learning Playground](https://playground.scienxlab.org/) | Скорость обучения, штраф, шум, свои JSON, обучение по шагам, ошибка на обучении и проверке; без скрытых слоев представлены линейные варианты | Отдельные настройки ошибки, штрафа и алгоритма; короткие интерактивные эксперименты |
| [EngineersOfAI Regression Explorer](https://engineersofai.com/playground/regression-explorer) | Одновременное сравнение OLS/Ridge/Lasso, изменение λ, перетаскивание точек, остатки и коэффициенты | Сравнение на одинаковых строках и явные остатки |
| [Interactive ML: Regularization](https://www.interactive-ml.com/regularization.html) | Крутилки силы штрафа, шума и числа признаков, настоящие и оцененные коэффициенты | Видимые последствия регуляризации и шумовых признаков |
| [Orange](https://orangedatamining.com/) | Визуальное соединение компонентов, загрузка данных, интерактивные диаграммы и расширения | Настройка эксперимента через интерфейс и библиотечные компоненты |

Эти страницы проверены как источники функций и идей интерфейса. Проверка их исходного кода, математической точности всех визуализаций и полноты каталогов не проводилась. Существование этих инструментов само по себе не доказывает отсутствие другого готового решения со всеми запрошенными функциями.

## Соглашение о функции оптимизации

Обозначения: `n` — число обучающих строк, `w` — коэффициенты преобразованных признаков, `b` — свободный член, `e=y−Xw−b`, `RSS=Σe²`. Если не оговорено иначе, свободный член не штрафуется. Формулы и `ModelRegistry.objective` относятся к одинаковым весам строк; веса отдельных наблюдений пока не входят в контракт обучения.

| Модель | Функция или процедура | Реализация |
| --- | --- | --- |
| МНК | `RSS` | sklearn LinearRegression |
| Ridge | `RSS + alpha·||w||²` | sklearn Ridge |
| Lasso / LassoLars | `RSS/(2n) + alpha·||w||₁` | sklearn, разные решатели |
| ElasticNet | `RSS/(2n) + alpha·r·||w||₁ + alpha·(1−r)·||w||²/2` | sklearn ElasticNet |
| LARS | Последовательный путь с равными углами к активным столбцам | sklearn Lars |
| OMP | Жадное приближение МНК с пределом числа активных признаков | sklearn OrthogonalMatchingPursuit |
| SGD | Средняя выбранная ошибка + выбранный штраф | sklearn SGDRegressor |
| RANSAC | Случайные кандидаты, выбор согласованного множества, повторное обучение на нем | sklearn RANSACRegressor |
| Theil–Sen | Пространственная медиана оценок МНК на подмножествах строк | sklearn TheilSenRegressor |
| Huber | `n·σ + Σσ·h(e/σ) + alpha·||w||²`; `h(u)=u²` внутри порога, иначе `2ε|u|−ε²` | sklearn HuberRegressor |
| Quantile | Средняя асимметричная ошибка квантиля + `alpha·||w||₁` | sklearn QuantileRegressor, SciPy HiGHS |
| BayesianRidge / ARD | Байесовская оценка с гауссовским распределением коэффициентов и оценкой точностей; максимизация критерия evidence | sklearn |
| Неотрицательный МНК | `RSS` при `w≥0` | sklearn LinearRegression(positive=True), SciPy NNLS |
| LinearSVR | `||w||²/2 + b²/2 + C·Σ max(|e|−ε,0)` при линейной ошибке и `intercept_scaling=1` | sklearn LinearSVR, LIBLINEAR |

Нельзя сравнивать число `alpha=1` у Ridge и Lasso как одинаковую силу штрафа. Для общей записи `RSS/(2n)+λ||w||²/2` параметр sklearn Ridge равен `n·λ`. Для SGD квадратичная ошибка равна `e²/2`; L2 штраф равен `alpha·||w||²/2`. У LinearSVR искусственный признак свободного члена тоже штрафуется; при квадратичной ошибке за пределами полосы соответствующее превышение возводится в квадрат.

Источники точных соглашений:

- [Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html), [Lasso](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html), [ElasticNet](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html).
- [Lars](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lars.html), [OMP и return_path](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.orthogonal_mp.html), [SGDRegressor и partial_fit](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html).
- [HuberRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.HuberRegressor.html), [RANSACRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RANSACRegressor.html), [TheilSenRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TheilSenRegressor.html), [QuantileRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.QuantileRegressor.html).
- [BayesianRidge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html), [ARDRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ARDRegression.html), [LinearSVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html).

## Структурные и невыпуклые штрафы

| Штраф | Смысл и конкретное соглашение | Библиотека и источник |
| --- | --- | --- |
| Weighted Lasso | `RSS/(2n)+alpha·Σaⱼ|wⱼ|`; профили весов задаются по порядку преобразованных признаков | [skglm WeightedLasso](https://contrib.scikit-learn.org/skglm/generated/skglm.WeightedLasso.html) |
| Adaptive Lasso | Ridge на обучающих строках, затем `aⱼ=(|w_pilot,j|+ε)^−γ` и Weighted Lasso | [Zou, 2006](https://pages.cs.wisc.edu/~shao/stat992/zou2006.pdf) |
| Group Lasso | `RSS/(2n)+alpha·Σ√|g|·||w_g||₂`; последовательные непересекающиеся группы | [skglm GroupLasso](https://contrib.scikit-learn.org/skglm/generated/skglm.GroupLasso.html) |
| Sparse Group Lasso | Смесь L1 и взвешенной суммы длин групп | [Официальный пример skglm](https://contrib.scikit-learn.org/skglm/auto_examples/plot_sparse_group_lasso.html), [Friedman, Hastie, Tibshirani, 2010](https://arxiv.org/abs/1001.0736) |
| Fused Lasso | `RSS/(2n)+alpha·||w||₁+fusion_strength·Σ|wⱼ₊₁−wⱼ|` | [Tibshirani и др., 2005](https://academic.oup.com/jrsssb/article/67/1/91/7110658); задача выражена в CVXPY |
| Тихонов | `RSS/(2n)+alpha·||Dw||²/2`; D — единичный оператор либо первая/вторая разность | [CVXPY least squares](https://www.cvxpy.org/examples/basic/least_squares.html), [параметры решателей](https://www.cvxpy.org/tutorial/solvers/index.html) |
| SCAD | Производная штрафа постоянна возле нуля, уменьшается и становится нулевой у больших коэффициентов; γ>2 | [skglm SCAD](https://contrib.scikit-learn.org/skglm/generated/skglm.penalties.SCAD.html) |
| MCP | `p(t)=alpha·t−t²/(2γ)` до `t=alpha·γ`, затем постоянный штраф | [skglm MCPRegression](https://contrib.scikit-learn.org/skglm/generated/skglm.MCPRegression.html) |
| Square-root Lasso | Нормировка установленного skglm: `||e||₂+alpha·||w||₁`; для `||e||₂/√n+λ||w||₁` нужно `alpha=√n·λ` | [skglm experimental.SqrtLasso](https://contrib.scikit-learn.org/skglm/generated/skglm.experimental.SqrtLasso.html); проверен исходный код установленного SqrtQuadratic |
| L0 | `RSS/(2n)+alpha·||w||₀`, максимум k выбранных признаков, полный перебор при p≤12 | МНК подзадачи решаются numpy.linalg.lstsq; [L0Learn: масштабируемая альтернатива](https://arxiv.org/abs/2202.04820) |

Добавление ε в Adaptive Lasso — явно выбранная конечная стабилизация исходных весов `1/|w_pilot|^γ`; асимптотические свойства оригинального метода не заявляются автоматически для этой конкретной настройки. Группы, профили весов, Fused Lasso и разностный оператор Тихонова зависят от порядка **признаков**, а не от порядка наблюдений. После создания полиномиальных признаков это порядок преобразованной матрицы.

SCAD и MCP невыпуклы, поэтому решатель не гарантирует глобальный минимум. Точный L0 здесь использует до 4096 подмножеств и ограничен 12 преобразованными признаками. При больших p доступен OMP с явной подписью жадного приближения. [L0Learn](https://tnonet.github.io/L0Learn/tutorial.html) предлагает библиотечные приближенные L0/L0L1/L0L2 алгоритмы; он не выдается за установленную зависимость или точный глобальный решатель.

## История обучения

| Интерфейс | Что реально доступно | Как трактовать |
| --- | --- | --- |
| SGDRegressor.partial_fit | Модель после каждого вызова, одна эпоха на переданных строках | История реального обучения; внешняя остановка контролируется сервисом |
| Lars/LassoLars.coef_path_ | Узлы пути коэффициентов | Реальные шаги активного множества, не эпохи SGD |
| orthogonal_mp(return_path=True) | Коэффициенты по мере включения столбцов | Реальный путь жадного отбора |
| RANSAC is_model_valid | Коэффициенты каждого проверяемого кандидата через публичный callback | История кандидатов; выбранная итоговая модель и ее маска показываются отдельно |
| BayesianRidge/ARD compute_score=True | scores_ критерия максимизации | История evidence; промежуточных коэффициентов API не возвращает |
| Lasso/ElasticNet path | Итоговые коэффициенты для разных alpha | Решения разных задач регуляризации |
| Прочие перечисленные fit | Итог, иногда число итераций или дополнительные диагностические атрибуты | Отсутствующие промежуточные коэффициенты не восстанавливаются вымышленной интерполяцией |

В просмотренной документации scikit-learn 1.9 появился [экспериментальный callback API](https://scikit-learn.org/stable/callbacks.html). Официальный список совместимых компонентов содержит LogisticRegression, методы поиска по параметрам, Pipeline и StandardScaler; требуемые регрессоры в нем не перечислены. У установленной версии 1.8 эта новая возможность не используется.

Повторные `fit(max_iter=1, warm_start=True)` — отдельные запуски, которые могут менять внутренние счетчики и процедуру остановки. Они не объявляются точной записью одного обычного `fit`. CVXPY `solver_stats` содержит итоговую статистику; общая история коэффициентов решателя из него не следует.

В сервисе обучения callback `is_model_valid` для RANSAC записывает кандидатов и возвращает `True`, сохраняя стандартное принятие моделей. Он не сообщает, какой кандидат в этот момент стал лучшим по согласованному множеству. Перед сохранением модели callback возвращается к исходному значению, чтобы локальная функция не попала в сериализованный объект.

## Целевая переменная и метрики

Обычные регрессоры требуют числовую конечную целевую переменную; произвольные номера классов не становятся осмысленной величиной для регрессии автоматически. У обобщенных линейных моделей дополнительно действуют области определения:

- [PoissonRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PoissonRegressor.html): y≥0, предсказания >0.
- [GammaRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.GammaRegressor.html): y>0.
- [TweedieRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TweedieRegressor.html): p=0 допускает любые числовые y; 0<p<1 не определен; 1≤p<2 требует y≥0; p≥2 требует y>0. При `link='auto'` p>0 использует экспоненту линейного предиктора, поэтому прогноз не является плоскостью в исходных координатах.

Метрика проверки и функция обучения — отдельные настройки. Выбор MAE для сравнения не заменяет МНК внутри Ridge. Результаты должны явно показывать направление улучшения и область определения метрик. MAPE проблематична возле нуля; логарифмические ошибки не допускают отрицательные значения; девиансы требуют соответствующей области y и предсказаний. Каталог и проверки метрик принадлежат `linear_lab/metrics.py`.

## Подготовка, проверка и поиск

Препроцессор использует sklearn `SimpleImputer`/`KNNImputer`, `PowerTransformer`, `QuantileTransformer`, `PolynomialFeatures`, масштабирующие преобразования, `OneHotEncoder`, `VarianceThreshold` и `SelectKBest`. Состояние этих объектов обучается по текущему train. Отбор по цели также находится внутри конвейера: создание списка полезных признаков до CV стало бы утечкой.

Порядок подготовки и пересэмплирования закреплен в `RegressionPipeline`: обучение преобразований → преобразованный train → пересэмплирование train → fit регрессора. В validation/test и при новом прогнозе пересэмплирование не вызывается. После пересэмплирования `n` в функции модели означает число **фактически поданных обучающих строк**. MSE на исходном train и ошибка на пересэмплированном fit различаются; записанные состояния содержат `train_loss` и `fit_loss` отдельно.

Источники и решения:

| Тема | Первоисточник | Применение в лаборатории |
|---|---|---|
| Обучаемые преобразования и утечки | [sklearn 1.8: Common pitfalls](https://scikit-learn.org/1.8/common_pitfalls.html) | Медианы, квантили, масштаб и отбор обучаются заново в каждой части |
| План CV | [sklearn 1.8: Cross-validation](https://scikit-learn.org/1.8/modules/cross_validation.html) | 9 схем; группы отдельно, время последовательно; диапазоны y подписаны экспериментальным приемом |
| Сетка и случайные параметры | [sklearn 1.8: Tuning hyperparameters](https://scikit-learn.org/1.8/modules/grid_search.html) | ParameterGrid/ParameterSampler, общий бюджет и одинаковые CV-разбиения кандидатов |
| Сокращение по ресурсам | [sklearn 1.8: Successive halving](https://scikit-learn.org/1.8/modules/grid_search.html#searching-for-optimal-parameters-with-successive-halving) | Библиотечные HalvingGridSearchCV/HalvingRandomSearchCV по числу строк; выбор на последнем этапе |
| Адаптивные предложения | [Optuna: Efficient optimization algorithms](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html) | Настоящий Study с TPE/Random; предложения последовательны, независимые CV-части могут работать параллельно |
| SMOTER | [Torgo и др., 2013: SMOTE for Regression](https://researchcommons.waikato.ac.nz/entities/publication/dbeb406b-0c2e-47d4-9910-0485a22c68e8) | Редкость определяется областью непрерывной цели, а не классовой меткой |
| SMOGN | [Branco, Torgo, Ribeiro, 2017](https://proceedings.mlr.press/v74/branco17a.html) | Интерполяция и возмущение с возможным уменьшением обычной области |

Стратегии CV не являются универсальным способом доказать качество. KFold предполагает подходящую независимость строк; группам и временным данным нужны соответствующие разбиения. Стратификация диапазонов y не устраняет зависимости. При Leave One Out R² на одной проверочной строке не определен.

Поиск оценивает кандидатов только внутри внешнего train. Optuna TPE использует законченные оценки для следующих предложений; реализации прерывания неудачного trial по промежуточным эпохам здесь нет. Невалидная метрика или упавший fit получают соответствующий статус, а не нулевую ошибку. Проверяются все части; отсутствие оценки на одной части не позволяет кандидату победить по оставшимся легким частям.

Сокращение допускает KFold, Repeated KFold и ShuffleSplit. Оценки разных этапов получены на разных бюджетах строк, поэтому общий минимум по всем этапам не выбирается как победитель. Для групп, времени, Leave One Out и диапазонов y доступны обычные методы поиска.

## Реализации пересэмплирования

Используются `ImbalancedLearningRegression==0.0.2` и `smogn==0.1.2`. Функции важности и случайное повторение/удаление вызывают установленные пакеты. Для SMOTER/SMOGN четыре исходных модуля включены в `linear_lab/_vendor` по GPL-3.0 с полными лицензиями, исходными SHA-256 и ссылками на репозитории авторов.

Математические проверки обнаружили проблемы в вычислении синтетической цели: неинициализированная память, расстояние с еще не записанным target, некорректное усреднение и обрезка отрицательных стандартизованных признаков. Включенные модули исправляют эти места. Расстояния для интерполяции цели рассчитываются в пространстве признаков; совпадающие позиции используют среднее двух целей. Знаковые признаки сохраняют отрицательные значения. Импорты выбирают конкретные модули без изменения глобальных функций NumPy или установленных пакетов.

Это исправления открытой библиотечной реализации, а не независимое изобретение алгоритма. Список изменений и их границы — [PROVENANCE.md](../linear_lab/_vendor/PROVENANCE.md). Численные тесты включают известную аффинную зависимость, постоянные столбцы, воспроизводимость и независимость от внешних holdout-ответов.

Синтетика строится после обученной подготовки и только по числовым исходным признакам. Интерполяция one-hot категорий дала бы значения между 0 и 1, не соответствующие категории. Поэтому для категорий доступны методы, сохраняющие целые строки. При временном разделении все методы пересэмплирования отвергаются.

## Экспорт полного прогноза

[Руководство sklearn по сохранению моделей](https://scikit-learn.org/1.8/model_persistence.html) различает перенос Python-объекта и отдельного вычислительного графа. Joblib/Pickle/Skops лаборатории сохраняют обученные преобразования вместе с регрессором. ZIP добавляет код пользовательских классов, версии и схему входов. Паспорт JSON описывает модель, но сам не выполняет прогноз.

ONNX-конвертация использует [sklearn-onnx: Dataframe as input](https://onnx.ai/sklearn-onnx/auto_tutorial/plot_gbegin_dataframe.html) и [документированные пользовательские конвертеры](https://onnx.ai/sklearn-onnx/auto_tutorial/plot_jfunction_transformer.html). Экспорт строит полный граф исходных столбцов, а не один регрессор после невидимой подготовки. Доступность проверяется для конкретного конвейера. ONNX Runtime сверяет прогнозы с Python на исходных строках, в том числе числовых NaN/±inf при включенном заполнении; допуск `rtol=atol=2e-5`. Неподдержанный шаг или несовпадение отключает формат с причиной.

ONNX использует числа float64 по одному входному столбцу. Категориальные входы должны быть строками без null/NaN; пропуски категорий требуют Python-формата. KNN-заполнение в ONNX не поддержано. Телеметрия Runtime отключается до импортов через `ORT_DISABLE_TELEMETRY=1` и вызовом `disable_telemetry_events()` перед сессией; основания настройки — [документ авторов Runtime](https://github.com/microsoft/onnxruntime/blob/main/docs/Privacy.md).

Python-формат требует совместимой среды. Загрузка Joblib/Pickle из недоверенного источника может исполнить код; Skops требует отдельной проверки типов. Сервис не принимает загружаемые обученные модели и экспортирует только собственные готовые артефакты.

## Источники объяснений и производительность

Учебные статьи лаборатории используют собственные небольшие численные примеры и ссылки на русские учебники, ODS, ISL/ESL, Wikipedia и ML Wiki. Точные формулы конкретного класса сверяются с первичной документацией и установленными пакетами. Открытый список чтения и замеченные ограничения источников — [learning-sources.md](learning-sources.md).

Научные страницы CV, поиска, SMOGN, сохранения моделей и ONNX повторно открыты 9 октября 2026 года. Эта проверка подтверждает содержимое указанных источников, а не математическую корректность каждого внешнего набора или каждой статьи из всего каталога.

Параллелизация независимых fit использует joblib с ограничением вложенных BLAS/OpenMP-пулов. Расчеты NumPy/SciPy/sklearn/skglm используют готовые BLAS/LAPACK, C/Cython и Numba. Отдельный C++-модуль не добавляется без измеренного узкого места. Реальные замеры, где Lasso ускоряется, а Ridge замедляется, — [performance.md](performance.md).
