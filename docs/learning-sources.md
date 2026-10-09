# Источники учебника CML-lab

Основные объяснения написаны по-русски. Под каждой статьей есть ссылки на документацию конкретного исполнителя и открытые учебники. Сначала выполни маленький пример из статьи, затем читай источник по нужному параметру. Английские названия классов сохранены, чтобы находить их в документации.

## Открытые учебники

| Источник | Что читать | Связанные темы |
|---|---|---|
| [ISLP: An Introduction to Statistical Learning](https://www.statlearning.com/) | Начальная постановка; главы 3–6, 8–9 и 12 | Регрессия, классификация, проверка, штрафы, деревья, SVM, обучение без учителя |
| [The Elements of Statistical Learning](https://hastie.su.domains/ElemStatLearn/download.html) | Математические основания после вводного ISLP; главы 3, 5, 7, 8–10, 14 | Линейные методы, сплайны, оценка качества, ансамбли, бустинг, кластеризация |
| [Forecasting: Principles and Practice](https://otexts.com/fpp3/) | Разделы о точности, временной проверке и лаговых входах | Последовательный прогноз, MAE/RMSE/MASE, горизонты |
| [Учебник Яндекса](https://education.yandex.ru/handbook/ml) | Русские объяснения постановки и семейств алгоритмов | Решающие деревья, линейные модели, ансамбли |

ISLP — введение в статистическое обучение с Python. ESL — более подробный математический учебник. FPP — учебник прогнозирования временных рядов. Подборочная картинка из t-SNE или найденные кластеры сами по себе не подтверждают качество прогноза.

## Документация исполнителей

| Вопрос | Первичный источник |
|---|---|
| Точная функция штрафа и допустимые параметры класса | [scikit-learn 1.8: руководство](https://scikit-learn.org/1.8/user_guide.html) и карточка конкретного класса |
| Где возникает утечка и почему шаги обучаются по train | [Типичные ошибки scikit-learn](https://scikit-learn.org/1.8/common_pitfalls.html) |
| Разбиения по времени, объектам и классам | [Перекрестная проверка](https://scikit-learn.org/1.8/modules/cross_validation.html) |
| Вероятность, порог и калибровка | [Порог решения](https://scikit-learn.org/1.8/modules/classification_threshold.html), [калибровка](https://scikit-learn.org/1.8/modules/calibration.html) |
| AP, AUC, MCC, каппа, Brier и log loss | [Метрики sklearn](https://scikit-learn.org/1.8/modules/model_evaluation.html) |
| Что означает тип признака при MI | [mutual_info_regression](https://scikit-learn.org/1.8/modules/generated/sklearn.feature_selection.mutual_info_regression.html) |
| Узлы, степень и поведение сплайна за диапазоном | [SplineTransformer](https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.SplineTransformer.html) |
| Итеративное заполнение, пределы итераций и skip_complete | [IterativeImputer](https://scikit-learn.org/1.8/modules/generated/sklearn.impute.IterativeImputer.html) |
| Байесовские счета и плотности | [Naive Bayes](https://scikit-learn.org/1.8/modules/naive_bayes.html), [GaussianNB](https://scikit-learn.org/1.8/modules/generated/sklearn.naive_bayes.GaussianNB.html) |
| Бустинг и группы ранжирования | [XGBoost](https://xgboost.readthedocs.io/en/stable/tutorials/learning_to_rank.html), [LightGBM](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRanker.html), [CatBoost](https://catboost.ai/docs/en/concepts/python-reference_catboostranker_fit) |
| Редкие классы и выборка внутри проверки | [imbalanced-learn: ошибки применения](https://imbalanced-learn.org/stable/common_pitfalls.html) |
| Что отличает pooled regression от фиксированных эффектов | [linearmodels: примеры панельных моделей](https://bashtage.github.io/linearmodels/panel/examples/examples.html) |

Наличие статьи о библиотеке не означает доступности всех ее классов в GUI. Карточка алгоритма показывает установленную библиотеку и реальный доступный адаптер. Панельный регрессор с лагами не объявляется оценкой фиксированных эффектов. Ссылки на `stable` у дополнительных библиотек могут вести на более новую версию; точные параметры всегда сверяй с версией установленного исполнителя.

## Дополнительное чтение на русском

[Открытый курс ODS: линейные модели](https://habr.com/ru/companies/ods/articles/323890/) полезен для другого объяснения регрессии и штрафов. [Wikipedia: Linear regression](https://en.wikipedia.org/wiki/Linear_regression) и [конспект ML Wiki автора](https://github.com/alexeygrigorev/mlwiki.org/blob/main/index.php/Linear_Regression.md) помогают искать термины и первичные работы. Статьи и энциклопедии являются дополнительными источниками: нормировку, допустимые данные и поведение API проверяем по документации класса. Не переносим параметры из старой статьи в современный исполнитель без проверки.

## Формулы, которые нельзя смешивать

Ridge sklearn минимизирует `SSE + alpha × сумму квадратов коэффициентов`, Lasso — `SSE/(2n) + alpha × сумму модулей`. Одинаковая alpha у этих классов задает разное относительное ограничение. У SVM уменьшение C усиливает регуляризацию. У GaussianNB alpha не используется; var_smoothing умножает наибольшую обучающую дисперсию признаков и добавляет результат к дисперсиям классов. У MultinomialNB alpha означает псевдосчета.

В CML-lab NDCG использует `2^relevance−1`. sklearn `ndcg_score` сам не делает это превращение: адаптер передает ему уже преобразованную полезность. Линейная полезность дала бы другое число для той же сортировки. У MAP/MRR релевантность положительна при `relevance>0`; правила k, равных оценок и запросов без релевантных объектов указаны в статье `ranking-metrics`.

SMOTER и SMOGN относятся к непрерывной цели, SMOTE — к классам. [Torgo и соавторы: SMOTE for Regression, 2013](https://researchcommons.waikato.ac.nz/entities/publication/dbeb406b-0c2e-47d4-9910-0485a22c68e8) и [Branco, Torgo, Ribeiro: SMOGN, 2017](https://proceedings.mlr.press/v74/branco17a.html) описывают исходные методы. Все пересэмплирование выполняется только по обучающим строкам текущего разреза. Веса и синтетика меняют обучающее распределение, поэтому вероятности проверяют на исходном потоке.

Руководство по действиям: [cml-learning.md](cml-learning.md), [guide.md](guide.md). Учебные ссылки проверяются через каталог и тесты; доступность каждого внешнего сайта во все будущие моменты не гарантируется.
