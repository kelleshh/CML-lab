# Используемые библиотеки

Интерфейс включает локальные версии Plotly.js (из Python plotly 7.1.0) и KaTeX. Plotly.js распространяется по MIT: https://github.com/plotly/plotly.js/blob/master/LICENSE. Заголовок лицензии сохранен в `web/vendor/plotly.min.js`. Лицензия KaTeX находится в `web/vendor/katex/LICENSE`.

Расчетное ядро использует NumPy, SciPy, pandas, scikit-learn, joblib, skglm и CVXPY; HTTP API использует FastAPI, Starlette, Uvicorn и python-multipart. Зависимости устанавливаются из публичного Python-реестра; их лицензии находятся в соответствующих установленных пакетах. Книга данных каждого внешнего набора и его лицензия принадлежат источнику набора.

Источники математических соглашений и библиотечных реализаций перечислены в `docs/research.md`.

Экспорт моделей использует Skops (MIT, https://github.com/skops-dev/skops/blob/main/LICENSE), sklearn-onnx / skl2onnx (Apache-2.0, https://github.com/onnx/sklearn-onnx/blob/main/LICENSE), ONNX (Apache-2.0, https://github.com/onnx/onnx/blob/main/LICENSE) и ONNX Runtime (MIT, https://github.com/microsoft/onnxruntime/blob/main/LICENSE). Установленные пакеты содержат тексты лицензий. Пользовательские адаптеры преобразований в ONNX реализованы в `linear_lab/exports.py`; сторонний код конвертеров не копировался.

## Corrected regression sampler sources

The files in `linear_lab/_vendor` derive from `smogn==0.1.2` and `ImbalancedLearningRegression==0.0.2` (GPL-3.0). Their full licenses, original source hashes, upstream URLs and the numerical changes are in `linear_lab/_vendor/PROVENANCE.md`. The upstream interpolation implementation is preserved with corrections for uninitialized targets, feature-space distance weights, mean parentheses and signed-feature clipping.
