# CML-lab: договор команды, версия 3

Основной пакет `cml_lab`, приложение `cml_lab.presentation.http.api:create_app`.
Модульный монолит: contexts/data, recipes, experiments, execution, learning.
В каждом контексте domain/application/ports. Домен и application не импортируют
pandas, sklearn, FastAPI, joblib, sqlite3. Адаптеры находятся в infrastructure.
Существующий linear_lab — совместимый вычислительный адаптер для ранее сохраненных
классов и проверенных линейных решателей. Новый HTTP/UI не вызывает его приватные методы.

## Общие идентификаторы

Tasks: regression, classification, clustering, ranking, forecasting, panel,
anomaly, reduction. Поле algorithm.tasks — список задач. Алгоритм != пользовательский
рецепт != обученный артефакт. Все идентификаторы выдаются сервером.

CRUD `/api/cml/{model-recipes,preprocessor-recipes,projects}`:
GET список `{items,total}`, POST объект; GET/PATCH/DELETE `/{id}`.
Объект `{id,kind,name,description,revision,config,created_at,updated_at}`.
PATCH принимает `name,description,config,expected_revision`; optimistic lock =>409.
Обновление сохраняет предыдущую ревизию. DELETE закрывает пользовательский head;
эксперимент хранит разрешенный снимок, не читает живой рецепт после запуска.

Model config `{algorithm_id,params}`; preparation config `{steps:[{adapter_id,columns,params,enabled}],resampling:{method}}`;
project config — полный запрос расчета либо ссылки `model_recipe_id,preprocessor_recipe_id`.

Данные сохраняют прежний HTTP-контракт `/api/datasets/*`, но новый HTTP вызывает
application use cases через порт к совместимому DataService. Классификационная
цель остается категорией, не принудительно превращается в float.

## Каталог и запуск

GET `/api/cml/catalogue` -> `{tasks,algorithms,datasets,preprocessors,metrics,lessons}`.
Algorithm `{id,name,family,tasks,library,available,reason,params,capabilities,lesson_id}`.
Параметр `{key,label,type,default,min,max,step,options,help,help_key,lesson_id}`.
Preparation stage имеет ту же схему плюс allowed_tasks.

POST `/api/cml/runs` -> `{id,status}`; GET/DELETE `/api/cml/runs/{id}`.
GET `/api/cml/runs` -> список запусков; POST `/{id}/reveal-test` открывает test.
POST `/api/cml/runs/{id}/save` сохраняет запуск как именованный эксперимент.
GET/PATCH/DELETE `/api/cml/experiments/{id}`, GET `/api/cml/experiments`.
GET `/api/cml/models` -> обученные артефакты; PATCH/DELETE `/{id}` name/description/delete.
GET `/api/cml/runs/{id}/export?format=joblib|pickle|skops|onnx|bundle|passport|json|csv`;
GET `/{id}/export-capabilities`; POST `/api/cml/predict` `{run_id,rows}`.
GET `/api/cml/preprocessing/stages`; POST `/api/cml/preprocessing/preview` полный запрос.

Запрос расчета:

```json
{
  "task":"classification", "dataset_id":"SERVER_ID", "target":"target",
  "features":["x1","x2"], "algorithm_id":"logistic_regression", "params":{},
  "preprocessing":{"steps":[],"resampling":{"method":"none"}},
  "split":{"train":0.6,"validation":0.2,"test":0.2,"shuffle":true},
  "validation":{"strategy":"stratified_kfold","folds":3},
  "search":null, "metrics":["accuracy","f1_macro"], "seed":42,"n_jobs":1,
  "roles":{"time_column":null,"entity_column":null,"query_column":null,"weight_column":null,"reference_target":null},
  "temporal":{"lags":[1,2,3],"rolling_windows":[3],"horizon":1}
}
```

Config recipes IDs разрешаются application до запуска; effective_spec сохраняет
полный снимок. Служебные роли не становятся признаками. Ranking split целыми query;
forecast/panel сортируются по времени, лаги panel вычисляются внутри entity.

RunResult `{task,algorithm_id,model_name,evaluations:{train,validation,test},diagnostics,trace,events,effective_spec,artifact_id,timing,warnings}`.
Каждая evaluation `{metrics,metric_details,rows:[{index,actual,predicted,...}],plots,...}`.
Вся test evaluation скрывается одной domain visibility policy до reveal.
Глобальные diagnostics используют train/validation и не содержат скрытых test rows.
Нельзя давать coefficients/proba/forecast функции моделям без соответствующих capabilities.

Engine port `run(spec,progress,cancelled)->RunResult`; adapter сохраняет fitted artifact
в переданный execution-owned временный путь; успешный процесс атомарно commit.
В engine только передаются готовые plain DTO, без HTTP/request/БД объектов.

## Учебник

GET `/api/learning/lessons`, GET `/api/learning/lessons/{id}`,
GET `/api/learning/help/{key}`. HTML `/lesson?id=stable-id` открывается отдельной
вкладкой; `?` имеет доступное название и ссылку `target=_blank rel=noopener`.
У каждого нового алгоритма/stage/настройки есть lesson_id и содержательное help.

## Владение

Архитектор: shared + contexts/{data,recipes,experiments,execution}/domain/application/ports.
ML-эксперт: infrastructure/ml/catalogue.py + tests реестра.
Data Engineers: infrastructure/ml/preparation.py + temporal.py + tests.
Учебный отдел: contexts/learning + web/lesson.*.
UI/UX: web/index.html, web/cml-app.js, web/cml/*, web/cml.css.
Тимлид: composition, infrastructure storage/workers/engines/exports, presentation/http,
интеграция, обязательные проверки и выпуск. Согласованные контракты меняем сообщением.
