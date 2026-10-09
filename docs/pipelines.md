# Декларативные preprocessing Pipeline

`Pipeline` задает преобразования фич перед тюнингом и обучением модели. Он состоит из настоящих объектов sklearn: `Pipeline`, `ColumnTransformer`, `FeatureUnion`, преобразователей и вложенных моделей для отбора фич. Названия классов и параметров совпадают с установленной версией sklearn. Модель, которую обучает проект, выбирается отдельно.

В редакторе код и GUI описывают одну структуру. При проверке сервер разбирает Python, возвращает дерево и направленный граф. GUI меняет дерево и получает из него обычную Python декларацию. Граф показывает последовательные шаги, входы выбранных столбцов, параллельные ветки и соединение их выходных фич. Исходный код с комментариями сохраняется до изменения структуры через GUI.

## Входы и выходы

`Pipeline` передает выход предыдущего шага следующему. `ColumnTransformer` направляет конкретные исходные столбцы в разные ветки, затем соединяет результаты по столбцам. `FeatureUnion` передает весь вход каждой ветке и соединяет их результаты. Любую из этих композиций можно вложить в другую.

`ColumnTransformer` принимает имя столбца, индекс, список имен/индексов, список bool, `slice(...)` или `make_column_selector(...)`. Скалярное имя, например `"description"`, передает одномерный поток документов в `TfidfVectorizer`; `["description"]` передает двумерную таблицу и для этого преобразователя не подходит. `remainder="drop"` исключает невыбранные столбцы; `"passthrough"` сохраняет их. Одну фичу можно отправить в несколько веток.

```python
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

numeric = Pipeline([
    ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
    ("scale", StandardScaler()),
])
categorical = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
    ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])
pipeline = ColumnTransformer([
    ("numeric", numeric, make_column_selector(dtype_include="number")),
    ("categorical", categorical, make_column_selector(dtype_exclude="number")),
], remainder="drop")
```

Медиана, масштаб и категории в этом примере изучаются только на `train`. Неизвестная категория после обучения дает нулевые индикаторы. Если `categorical` содержит пропуски `None`, убедитесь, что `missing_values` соответствует реальному представлению пропуска: sklearn не считает все виды пропусков взаимозаменяемыми.

## Декларативный Python

Допустимы разрешенные imports, присваивания, литералы, вложенные конструкторы, `slice`, `numpy.array` из литералов и известные ссылки на функции numpy. Результат называется `pipeline`. Можно использовать вспомогательные переменные и import aliases. `set_params(scale__with_mean=False)` меняет явно именованные параметры вложенного шага. `set_output(transform="pandas")` и `set_output(transform="default")` выбирают контейнер результата там, где это поддерживает преобразователь.

```python
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

pipeline = Pipeline([
    ("log", FunctionTransformer(
        func=np.log1p,
        inverse_func=np.expm1,
        feature_names_out="one-to-one",
    )),
    ("scale", StandardScaler()),
])
```

`log1p` требует входа больше `-1`; для отрицательных значений нельзя выбирать его без проверки смысла фичи. Ссылку `np.log1p` передают параметру `func`; вызывать функцию над данными в определении Pipeline не нужно.

Сервер не вызывает `eval` или `exec` для пользовательской декларации. Циклы, функции, классы, lambda, произвольные imports, `fit`, `transform`, файловый/сетевой ввод, пользовательские estimators, распаковка `*args`/`**kwargs` и изменения атрибутов не поддерживаются. `Pipeline.memory` и `transform_input` принимают только `None`; `prefit=True` у селекторов запрещен. Ошибка содержит строку и столбец. Обычный полный Python остается доступен пользователю в экспортированном проекте, который он запускает самостоятельно.

## Наш формат и экспорт

`.py` содержит imports и присваивание `pipeline`. Формат `cml.pipeline` хранит ту же декларацию в JSON:

```json
{
  "format": "cml.pipeline",
  "version": 1,
  "source": "from sklearn.preprocessing import StandardScaler\npipeline = StandardScaler()\n"
}
```

Проект хранит это значение в `preprocessing.declarative_pipeline`. Сервер принимает также `{ "format": "cml.pipeline", "version": 1, "tree": ... }` от GUI и сериализует дерево в Python. JSON дерево хранит tuple как `{ "$tuple": [1, 2] }`, ссылки как `{ "$ref": "numpy.log1p" }`, slices как `{ "$slice": [0, 3, null] }`, массивы как `{ "$array": [[0], [1]], "dtype": "float64" }`. Это сохраняет различия типов, которые важны для sklearn, например tuple у `ngram_range`.

В Python API доступны:

```python
from cml_lab.infrastructure.ml.pipelines import (
    compile_pipeline,
    describe_pipeline,
    normalize_pipeline,
    prepare_pipeline_for_fit,
)

definition = {"format": "cml.pipeline", "version": 1, "source": source}
stored = normalize_pipeline(definition)
description = describe_pipeline(stored)  # source, tree, graph, warnings
unfitted = compile_pipeline(stored)      # новые объекты sklearn; fit не вызывается
train_transformer = prepare_pipeline_for_fit(
    unfitted, task="regression", split_kind="holdout"
)
```

`compile_pipeline` не читает датасет и не вычисляет статистики. Каждый обучающий запуск и каждый fold получают новый преобразователь. Validation и test вызывают только `transform` уже обученного объекта. Сериализация сохраняет обученное состояние, чтобы новые строки проходили те же преобразования.

## Совместимость и ограничения

Каталог содержит композиции, числовые преобразования, imputation, categorical encoding, text vectorization, selection, decomposition, kernel approximation, random projection и transformer методы manifold learning. Вложенные модели доступны для `SelectFromModel`, `RFE` и `IterativeImputer`, но не могут заменить preprocessing объект без `transform`.

Для `PLSRegression`, `PLSCanonical` и `CCA` лаборатория передает дальше только компоненты `X`. Сам sklearn при `fit_transform(X, y)` возвращает пару `(X_scores, y_scores)`; компоненты `y` не становятся фичами модели. Экспортированный проект вызывает адаптер лаборатории и сохраняет это правило. Если отдельно использовать обычную sklearn декларацию `.py`, учитывайте стандартный тип результата sklearn. `DictVectorizer` и `FeatureHasher` требуют соответствующего потока словарей/записей; обычная таблица сама по себе не превращается в такой поток.

`TargetEncoder`, `RFECV` и `SequentialFeatureSelector` имеют свою внутреннюю CV. Лаборатория не передает в нее time/group разбиение, поэтому эти шаги запрещены в ordered, forecasting, panel и grouped задачах. Обычный `SelectKBest` или `SelectFromModel(prefit=False)` обучается на конкретной train части и не имеет такого внутреннего смешивания.

Определение ограничено 100 000 байт, 12 000 AST узлов, 200 конструкторами и 40 уровнями вложенности. Литеральный массив содержит не более 10 000 элементов и занимает до 10 млн байт; тип и длина строк проверяются до выделения numpy массива. Перед созданием расширенных матриц проверяется бюджет: до 2000 плотных или 20 000 разреженных фич и 15 млн ячеек. `PolynomialFeatures`, `SplineTransformer` и `OneHotEncoder` проверяют размеры до transform. `IterativeImputer` ограничен 100 входными фичами, 500 000 ячейками и 30 итерациями. Kernel/manifold методы ограничивают попарную train матрицу 15 млн расстояний. Остальные ограничения конструктора sklearn проверяются по его установленной версии. Некоторые стандартные настройки, например миллион hash фич, превышают бюджет лаборатории; уменьшите `n_features`.

Контекстное меню «Исходники» показывает настоящий установленный Python исходник выбранного класса sklearn, его путь внутри пакета и номер строки. Оно принимает только имя из каталога и не читает произвольный путь. Исходники распространяются по BSD-3-Clause; показанный код может отличаться от новой версии на сайте sklearn.

## Первичные источники

- [sklearn 1.8: Pipeline and composite estimators](https://scikit-learn.org/1.8/modules/compose.html)
- [sklearn 1.8: ColumnTransformer](https://scikit-learn.org/1.8/modules/generated/sklearn.compose.ColumnTransformer.html)
- [sklearn 1.8: FeatureUnion](https://scikit-learn.org/1.8/modules/generated/sklearn.pipeline.FeatureUnion.html)
- [sklearn 1.8: FunctionTransformer](https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html)
