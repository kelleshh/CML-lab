"""Fit one complete fold. Statistics and samplers see only its training rows."""
from __future__ import annotations

import inspect
from copy import deepcopy
import numpy as np
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.utils import get_tags
from .artifacts import FittedArtifact
from .matrices import for_estimator



def _estimator_components(model):
    """Walk instantiated nested estimators, including ensemble components."""
    yield model
    for value in model.get_params(deep=False).values():
        if hasattr(value, "get_params"):
            yield from _estimator_components(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, tuple) and len(item) == 2 and hasattr(item[1], "get_params"):
                    yield from _estimator_components(item[1])


def _check_stacking_context(model, spec, groups):
    dependent = groups is not None or spec["task"] in {"forecasting", "panel"} or (spec.get("split") or {}).get("shuffle") is False
    if dependent and any(isinstance(item, (StackingClassifier, StackingRegressor)) for item in _estimator_components(model)):
        raise ValueError("Стекинг, включая вложенный, пока поддерживает только независимые строки. "
                         "Его внутренняя CV не сохраняет группы и хронологию. Выберите голосование, "
                         "лес или бустинг; для временного стекинга нужен отдельный исторический OOF-конвейер.")

def fit_artifact(spec, X, y, catalogue, *, label_encoder=None, groups=None,
                 weights=None, progress=None, cancelled=None, validation=None):
    from .preparation import build_preprocessor, fit_resample
    task = spec["task"]
    seed = spec.get("seed", 42)
    prep_config = deepcopy(spec.get("preprocessing") or {})
    if groups is not None: prep_config["split_kind"]="group"
    elif task in {"forecasting","panel"}: prep_config["split_kind"]="timeseries"
    elif spec.get("split", {}).get("shuffle") is False: prep_config["split_kind"]="ordered"
    parameters = dict(spec.get('params') or {})
    for key in list(parameters):
        if key.startswith('pipeline__'):
            prep_config.setdefault('pipeline_params', {})[key[len('pipeline__'):]] = parameters.pop(key)
    model = catalogue.build(task, spec["algorithm_id"], parameters, seed, spec.get("n_jobs", 1))
    _check_stacking_context(model, spec, groups)
    preprocessing = build_preprocessor(X, prep_config, task=task, seed=seed)
    capabilities = dict(catalogue.descriptor(spec["algorithm_id"]).get("capabilities", {}))
    if hasattr(catalogue, 'effective_capabilities'):
        capabilities = catalogue.effective_capabilities(model, capabilities)
    # Ensemble recipes can contain dense-only children even when the default
    # recipe accepts sparse input. Persist the actual input contract for inference.
    if any(not get_tags(item).input_tags.sparse for item in _estimator_components(model)):
        capabilities["requires_dense"] = True
    Xt = for_estimator(preprocessing.fit_transform(X, y),capabilities)
    categorical = any(not is_numeric_dtype(dtype) for dtype in X.dtypes)
    sampling = prep_config.get("resampling") or spec.get("resampling") or {"method":"none"}
    if weights is not None and sampling.get("method", "none") != "none":
        raise ValueError("Веса строк и пересэмплирование нельзя объединить без правила переноса весов.")
    Xfit, yfit, sampling_summary = fit_resample(Xt, y, sampling, task=task, seed=seed,
                                              categorical=categorical, temporal=task in {"forecasting", "panel"})
    domain=capabilities.get("target_domain")
    if domain=="positive" and np.any(np.asarray(yfit)<=0):
        raise ValueError("Gamma-регрессия требует строго положительную цель.")
    if domain=="nonnegative" and np.any(np.asarray(yfit)<0):
        raise ValueError("Poisson-регрессия требует неотрицательную цель.")
    if spec["algorithm_id"]=="tweedie":
        power=(spec.get("params") or {}).get("power",0)
        if 0<power<1:raise ValueError("Tweedie не определён для power между 0 и 1.")
        if power>=2 and np.any(np.asarray(yfit)<=0):raise ValueError("Tweedie с power≥2 требует строго положительную цель.")
        if 1<=power<2 and np.any(np.asarray(yfit)<0):raise ValueError("Tweedie с 1≤power<2 требует неотрицательную цель.")
    trace = []
    def check():
        if cancelled and cancelled(): raise InterruptedError("Расчет отменен.")
    check()
    if task == "ranking":
        if groups is None: raise ValueError("Для ранжирования нужны группы запросов.")
        if weights is not None: raise ValueError("Для ранжирования нужны отдельные веса запросов. Веса отдельных строк здесь не поддерживаются.")
        names, encoded = np.unique(np.asarray(groups).astype(str), return_inverse=True)
        order = np.argsort(encoded, kind="stable")
        Xrank, yrank = Xfit[order], np.asarray(yfit)[order]
        library = type(model).__module__.split(".")[0]
        if library == "xgboost":
            model.fit(Xrank, yrank, qid=encoded[order])
        elif library == "lightgbm":
            model.fit(Xrank, yrank, group=np.bincount(encoded).tolist())
        elif library == "catboost":
            model.fit(Xrank, yrank, group_id=encoded[order])
        else:
            raise ValueError("Алгоритм ранжирования должен поддерживать группы запросов.")
    elif hasattr(model, "partial_fit") and spec["algorithm_id"] in {"sgd", "sgd_classifier", "perceptron", "passive_aggressive_classifier", "passive_aggressive_regressor"}:
        epochs = int(spec.get("epochs", 50))
        if not 1 <= epochs <= 500: raise ValueError("Число эпох: от 1 до 500.")
        for epoch in range(epochs):
            check()
            kwargs = {}
            if task == "classification": kwargs["classes"] = np.arange(len(label_encoder.classes_))
            if weights is not None: kwargs["sample_weight"] = weights
            model.partial_fit(Xfit, yfit, **kwargs)
            prediction = model.predict(Xt)
            loss = float(np.mean(prediction != y)) if task == "classification" else float(mean_squared_error(y, prediction))
            point = {"step":epoch+1, "train_loss":loss, "label":"Доля ошибок" if task == "classification" else "MSE"}
            if validation is not None:
                vx, vy = validation
                vp = model.predict(for_estimator(preprocessing.transform(vx),capabilities))
                point["validation_loss"] = float(np.mean(vp != vy)) if task == "classification" else float(mean_squared_error(vy, vp))
            trace.append(point)
            if progress: progress({"type":"frame", "progress":.2+.45*(epoch+1)/epochs, "frame":point, "message":f"Настоящая эпоха {epoch+1}/{epochs}"})
    else:
        kwargs = {}
        if weights is not None:
            if "sample_weight" not in inspect.signature(model.fit).parameters:
                raise ValueError("Этот алгоритм не принимает веса строк.")
            kwargs["sample_weight"] = weights
        model.fit(Xfit, yfit, **kwargs)
    check()
    artifact = FittedArtifact(task, preprocessing, model, list(X.columns), label_encoder,input_capabilities=capabilities)
    from .feature_names import feature_names
    artifact.transformed_features = feature_names(preprocessing, Xt.shape[1])
    if not trace and hasattr(model, "staged_predict"):
        for step, prediction in enumerate(model.staged_predict(Xt), 1):
            check()
            loss = float(np.mean(prediction != y)) if task == "classification" else float(mean_squared_error(y, prediction))
            trace.append({"step":step,"train_loss":loss,"label":"Доля ошибок" if task == "classification" else "MSE"})
        if len(trace) > 100:
            indices = np.unique(np.linspace(0, len(trace)-1, 100).astype(int))
            trace = [trace[i] for i in indices]
    return artifact, sampling_summary, trace
