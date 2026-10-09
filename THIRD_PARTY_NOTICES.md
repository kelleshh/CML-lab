# Используемые библиотеки

Интерфейс включает локальные версии Plotly.js (из Python plotly 7.1.0) и KaTeX. Plotly.js распространяется по MIT: https://github.com/plotly/plotly.js/blob/master/LICENSE. Заголовок лицензии сохранен в `web/vendor/plotly.min.js`. Лицензия KaTeX находится в `web/vendor/katex/LICENSE`.

Расчетное ядро использует NumPy, SciPy, pandas, scikit-learn, joblib, skglm и CVXPY; HTTP API использует FastAPI, Starlette, Uvicorn и python-multipart. Зависимости устанавливаются из публичного Python-реестра; их лицензии находятся в соответствующих установленных пакетах. Книга данных каждого внешнего набора и его лицензия принадлежат источнику набора.

Источники математических соглашений и библиотечных реализаций перечислены в `docs/research.md`.
