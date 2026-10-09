# Контракты API и расчетных модулей

Локальный HTTP API имеет префикс `/api`. Каталоги возвращают допустимые модели, параметры и настройки; интерфейс не держит отдельную математическую реализацию. Файлы состояния находятся в `LINEAR_LAB_DATA` или `~/.linear-lab`. Идентификаторы наборов, расчетов и экспериментов выдает сервер. Клиент не задает путь артефакта. Объекты настроек должны быть JSON-объектами; `features` — null или непустой список имен, `metrics` — имя либо непустой список. `seed` — целое 0…4294967295. Ошибки этой формы отклоняются до запуска процесса.

## HTTP API

Запросы JSON имеют `Content-Type: application/json`. Ошибки входных значений — HTTP 422, отсутствующие объекты — 404, ответ содержит `{"detail":"Пояснение"}`. Изменяющий запрос с `Origin` чужого сайта отклоняется 403. Исключение внутри запущенного расчета записывается в его `status="error"` и `error`; POST расчета сам по себе не означает успешного обучения.

| Метод и путь | Назначение / параметры |
|---|---|
| `GET /api/health` | `{status:"ok",version:"2.0.0"}` |
| `GET /api/catalogue` | `models`, `datasets`, `metrics`, `lessons`, `glossary`, `cv_strategies` |
| `GET /api/datasets/library` | Поиск: `query`, `task`, `source`, `offset`, `limit`; ответ `{items,total,offset,limit,tasks}` |
| `POST /api/datasets/load` | Создать или загрузить набор из спецификации |
| `POST /api/datasets/upload` | Multipart с полем `file`; до 25 МБ |
| `GET /api/datasets/{id}` | Полные метаданные выбранной версии |
| `GET /api/datasets/{id}/rows` | Страница: `offset`, `limit` |
| `PATCH /api/datasets/{id}/rows` | Новая версия из `changes`, `additions`, `deletes` |
| `PATCH /api/datasets/{id}/metadata` | Новый снимок с названием, описанием, метками, задачей и целью |
| `DELETE /api/datasets/{id}` | Удалить один снимок; ссылка сохраненного эксперимента блокирует удаление |
| `GET /api/datasets/{id}/explore` | Исходные точки, распределения, профили и корреляции; `x,y,z,color,target,sample_size,seed` |
| `GET /api/datasets/{id}/export` | Исходная таблица CSV |
| `POST /api/datasets/{id}/preprocessing-preview` | Подготовка и пересэмплирование текущего train без обучения регрессора |
| `POST /api/jobs` | Запуск одиночного обучения или поиска; ответ `{id}` |
| `GET /api/jobs/{id}` | Состояние, события, результат либо ошибка |
| `DELETE /api/jobs/{id}` | Отменить активный процесс |
| `POST /api/jobs/{id}/reveal-test` | Явно открыть итоговые тестовые метрики и строки |
| `GET /api/jobs/{id}/grid` | Новый срез готовой модели: `x_feature`, необязательный `y_feature` |
| `GET /api/jobs/{id}/export-capabilities` | `{formats:[{format,label,available,reason}]}` после завершения расчета |
| `GET /api/jobs/{id}/export` | `format=json,csv,model,joblib,pickle,skops,onnx,bundle,passport` |
| `POST /api/predict` | `{job_id,rows:[{feature:value}]}`; от 1 до 1000 строк; ответ `{predictions}` |
| `POST /api/metrics/preview` | `{expression,actual,predicted}`; ответ `{value}` |
| `GET /api/experiments` | Список сохраненных опытов |
| `POST /api/experiments` | `{job_id,name}`; сохраняет завершенный расчет |
| `GET /api/experiments/{id}` | Настройки и сохраненный результат |
| `DELETE /api/experiments/{id}` | Удалить сохраненный опыт |

### Набор данных

Пример создания без внешней сети:

```json
{
  "kind": "synthetic",
  "name": "correlated",
  "params": {
    "n_samples": 180,
    "n_features": 4,
    "noise": 8,
    "correlation": 0.95,
    "outliers": 0,
    "seed": 42
  }
}
```

Поддержанные `kind`: `synthetic`, `builtin`, `fetch`, `openml`, `custom`. Для sklearn `name` содержит имя функции из каталога; для OpenML ID или имя и версия передаются в `params`. Ручной набор: `{kind:"custom",name,rows:[{x:1,y:2}],target:"y"}`. Адаптеры и ограничения источников — [datasets.md](datasets.md).

Метаданные плоские: `id`, `name`, `rows`, `columns:[{name,dtype,numeric,missing}]`, `preview`, `targets`, `default_target`, `task_target`, `task`, `tasks`, `modality`, `source`, `description`, `tags`, `stats`. `default_target` — допустимая числовая цель регрессии; `task_target` — цель исходной задачи, которая может быть классом. `excluded_features` и `excluded_targets` препятствуют автоматическому использованию меток класса. Клиент читает эти поля, а не угадывает роль по префиксу имени столбца.

Редактор использует **абсолютный нулевой индекс** строки исходной версии:

```json
{
  "changes": [{"index":125,"values":{"x1":11,"y":22}}],
  "additions": [{"x1":3,"x2":4,"y":20}],
  "deletes": [4]
}
```

Индексы `changes`/`deletes` относятся к исходной таблице до удаления. До 500 изменений, 500 удалений и 100 добавлений; строки вне патча сохраняются. Возвращаются метаданные нового ID. PATCH метаданных допускает только `name`, `description`, `tags`, `task`, `default_target`, `task_target`.

`explore` выбирает до 2000 строк воспроизводимо и сохраняет исходные индексы. Корреляции/гистограммы/профиль используют полную таблицу; матрица — до 50 числовых столбцов. Категории возвращают подписи и коды окраски; код цвета не служит числовой целью.

### Полный запрос обучения

Замените `DATASET_ID` на ID из ответа загрузки. Все необязательные поля можно опустить; этот пример показывает действующие настройки, а не обязательную сложность первого опыта.

```json
{
  "dataset_id": "DATASET_ID",
  "target": "y",
  "features": ["x1", "x2", "x3", "x4"],
  "model": "elasticnet",
  "params": {"alpha": 0.1, "l1_ratio": 0.5},
  "seed": 42,
  "split": {"train":0.6,"validation":0.2,"test":0.2,"shuffle":true},
  "preprocessing": {
    "imputation": "median",
    "fill_value": 0,
    "knn_neighbors": 5,
    "missing_indicator": false,
    "numeric_transform": "none",
    "degree": 2,
    "interaction_only": false,
    "scaler": "standard",
    "clip_quantiles": null,
    "variance_threshold": null,
    "selection": "none",
    "max_features": null
  },
  "resampling": {"method":"none"},
  "metrics": ["rmse", "mae", "r2", "custom"],
  "custom_metric": "mean(abs(error))",
  "metric_params": {"quantile":0.5,"power":1.5},
  "epochs": 100,
  "cv_config": {"strategy":"kfold","folds":5,"shuffle":true},
  "n_jobs": 1,
  "regularization_path": true,
  "learning_curve": false,
  "permutation_importance": false
}
```

`features:null` выбирает допустимые входы автоматически. `metrics:"all"` или список с `"all"` включает стандартные метрики; `custom` требует формулу. Старые `preprocessing.scale`, `impute` и `cv` сохраняют совместимость; явные `scaler`, `imputation`, `cv_config` имеют приоритет.

До общего обучения происходит внешнее разделение. Медианы, квантили, масштаб, категории и отбор обучаются на train. Затем пересэмплируется только подготовленный train. Предсказания и метрики исходных train/validation/test сохраняют исходные строки; синтетические строки не становятся тестом. Каждое разбиение CV заново обучает весь конвейер.

Допустимые значения `preprocessing`:

| Поле | Значения |
|---|---|
| `imputation` | `median,mean,most_frequent,constant,knn,none` |
| `scaler` | `standard,robust,minmax,maxabs,none` |
| `numeric_transform` | `none,log1p,sqrt,yeo_johnson,quantile_normal,quantile_uniform` |
| `degree` | Целое 1–5 |
| `interaction_only`, `missing_indicator` | Boolean |
| `clip_quantiles` | `null` или два числа `0 ≤ low < high ≤ 1` |
| `variance_threshold` | `null` или конечное неотрицательное число |
| `selection` | `none,f_regression,mutual_info` |
| `max_features` | `null` или целое 1–2000; нужен включенный отбор |

### Пересэмплирование

```json
{
  "method": "smogn",
  "focus": "high",
  "sampling": "balance",
  "relevance_threshold": 0.5,
  "neighbors": 5,
  "perturbation": 0.02,
  "undersample": true
}
```

`method`: `none,random_over,random_under,smoter,smogn`. `focus`: `both,high,low`. `sampling`: `balance,extreme`. `relevance_threshold` строго между 0 и 1; соседей 1–100, меньше числа обучающих строк; возмущение SMOGN `0 < perturbation ≤ 1`. SMOTER/SMOGN запрещены при исходных категориях. Все методы запрещены для временных схем/разделения без перемешивания.

Ручное задание важности доступно через API: `relevance:"manual",control_points:[[y,importance,derivative],...]`. Нужны минимум три конечные точки с возрастающим y и важностью 0…1. Интерфейс показывает автоматическую важность, настройку ее порога и края цели; редактора ручных точек в GUI нет.

Вход пересэмплирования — до 10 000 строк, 200 подготовленных признаков и 1 млн ячеек. Выход — до 100 000 строк и 15 млн ячеек. Требуются хотя бы три разных значения цели. Ответ с некорректной синтетикой отклоняется. Используются зафиксированные сторонние пакеты и исправленные модули из [PROVENANCE.md](../linear_lab/_vendor/PROVENANCE.md).

`preprocessing-preview` принимает поля запроса обучения кроме обязательных модели/метрик; ID берет из URL. Возвращает `fitted_on:"train"`, `train_rows`, `original_features`, `features`, `rows` первых 100 строк, их `indices`, `original_rows`, `target`, `means`, `std`, `sampling`. В `sampling` находятся фактические `before/after`, до 2000 значений цели каждого состояния и первые 100 пересэмплированных строк.

### Перекрестная проверка и поиск

`cv_config.strategy`: `none,kfold,repeated_kfold,shuffle_split,timeseries,group_kfold,group_shuffle_split,leave_one_group_out,leave_one_out,stratified_bins`.

Дополнительные поля: `folds` 2–20, `repeats` 1–10, `shuffle`, `test_size` (доля для ShuffleSplit, число строк для времени), `gap`, `max_train_size`, `group_column`, `bins` 2–20. Обычная CV ограничена 100 обучениями и 3 млн суммарных обучающих строк. `n_jobs` — целое 1–4. Групповой столбец исключается из входов; внешнее разделение также держит группы отдельно. Временные строки должны быть упорядочены заранее.

Поиск запускается тем же `POST /api/jobs`, если в запросе есть `search`:

```json
{
  "method": "optuna_tpe",
  "trials": 20,
  "metric": "custom",
  "direction": "min",
  "n_jobs": 2,
  "param_space": {
    "alpha": {"type":"float","low":0.0001,"high":100,"log":true},
    "l1_ratio": [0.2, 0.5, 0.8]
  }
}
```

Методы: `grid,random,optuna_tpe,optuna_random,halving_grid,halving_random`. Сетка требует списки; случайный поиск/Optuna допускают `float,int,categorical`, `low/high/log` или `choices`. До 12 параметров, 100 значений в списке, 100 попыток и 500 обучений кандидатов. Направление `min/max`; `all` не является единой метрикой выбора. Для сокращения есть `factor` 2–5; минимальный объем строк вычисляет сервис. Сокращение допускает KFold, Repeated KFold и ShuffleSplit; другие схемы требуют сетку, случайный поиск или Optuna. Все диапазоны сверяются с схемой модели.

При выключенной CV поиск создает три KFold-разбиения либо три временных разреза без перемешивания. Предложения Optuna последовательны, части CV параллельны. Каждый кандидат обязан дать допустимую главную метрику на всех частях. Победитель сокращения выбирается только на последнем ресурсе. Итоговая модель обучается на внешнем train; внешние validation/test не участвуют в поиске.

### Состояние и результат

Состояния: `running,completed,error,cancelled,interrupted`. В ответе: `id,status,progress,message,events`, после успеха `result`, после ошибки `error`. `progress` 0…1 отражает завершенные этапы, а не искусственную оценку качества. После перезапуска активный незавершенный расчет отмечается `interrupted`.

Результат содержит `metrics`, `metric_details`, `coefficients`, `intercept`, `feature_names`, `trace`, `trace_kind`, `trace_label`, `predictions`, `plot_data`, `prediction_grid`, `regularization_path`, `objective_surface`, `penalty_geometry`, `diagnostics`, `preprocessing`, `resampling`, `split`, `timing`, `warnings`; `cv`, `diagnostics.learning_curve`, `diagnostics.permutation_importance`, а после поиска — `search` и `effective_request`. Неподходящая метрика имеет `null` и причину.

До `reveal-test` метрики/строки теста, их отображаемые точки и соответствующие предсказания кадров скрыты в публичном результате и выгрузках JSON/CSV. Это режим использования итоговой проверки, а не защита данных от владельца локального компьютера. Изменение ответа/экспорта не изменяет сохраненный внутренний расчет.

### Экспорт

`model` — совместимый алиас `joblib`. Остальные варианты модели: `pickle,skops,onnx,bundle,passport`. Экспортируется обученный полный конвейер от исходной строки до прогноза. Skops возвращает требуемые доверенные типы в паспорте. ONNX доступен после конвертации всех шагов и проверки через ONNX Runtime с `rtol=atol=2e-5`; неподдерживаемый шаг возвращает причину в `export-capabilities`.

ZIP содержит Joblib, паспорт, зависимости, пример `predict.py`, исходники пользовательских классов/функций с лицензиями и доступные Skops/ONNX. Исходные таблицы и посторонние файлы сервиса в ZIP модели не входят. Паспорт содержит схему входов, выбранную цель, параметры, преобразования, метрики, версии и источники.

Python-модель ожидает pandas.DataFrame с исходными именами столбцов. ONNX принимает отдельные столбцы формы `[None,1]`, числа `float64` и строки категорий без `null/NaN`. При включенном числовом заполнении NaN/±inf обрабатываются выученным заполнителем. Пересэмплирование при предсказании не выполняется.

## Ответственность модулей

| Модуль | Контракт |
|---|---|
| `datasets.py` | `DataService`: адаптеры, валидация таблицы, версии, библиотека, исследование; `DataBundle` содержит DataFrame X, y, имена и метаданные |
| `models.py` | `ModelRegistry`: допустимые параметры, создание библиотечной модели и точная функция для диагностики |
| `preprocessing.py` | Обучаемый sklearn-препроцессор, отбор, `ResamplingService` и `RegressionSampler` |
| `pipeline.py` | `RegressionPipeline`: clone/fit/predict с пересэмплированием только при fit |
| `training.py` | Внешнее разделение, реальное обучение/история, диагностика и сохранение готовой модели |
| `validation.py` | Проверяемые планы разбиений и отдельный fit полного конвейера в каждой части |
| `search.py` | Библиотечные сетка/случайный/Optuna/halving, общий бюджет и повторное обучение победителя |
| `exports.py` | Форматы, схема входов, паспорт, проверка конвертации полного конвейера |
| `jobs.py` | Один процесс на опыт, очередь событий, отмена, сохранение и список экспериментов |
| `metrics.py` | Каталог метрик и ограниченный AST-интерпретатор числовых выражений без `eval` |
| `teaching.py` | 32 урока с формулами/заданиями/пресетами/источниками, словарь и объяснения метрик |
| `app.py` | Сборка FastAPI, публичное скрытие теста, HTTP и локальные статические файлы |
| `web/app.js` | Состояние интерфейса и действия; расчет выполняет API |
| `web/datasets-ui.js` | Библиотека и исходные 2D/3D до обучения |
| `web/preprocessing-ui.js` | Настройки признаков/сэмплирования и настоящий предпросмотр |
| `web/validation-ui.js` | Настройки CV/поиска, таблицы частей/попыток, графические отчеты |
| `web/help.js`, `beginner.js` | Доступные пояснения и ограниченные варианты реальных параметров |
| `web/charts.js` | 12 панелей Plotly и воспроизведение реальных кадров |

`ChartManager.render(result,options)`, `frame(frame,result)`, `destroy()` используют локальный Plotly. Изменение осей готовой модели запрашивает `/grid`, не вызывает fit. Кадры SGD/Lars/OMP/RANSAC получены из публичных библиотечных интерфейсов; отсутствующая история не интерполируется как история решателя. Ограничения математических срезов — [research.md](research.md).
