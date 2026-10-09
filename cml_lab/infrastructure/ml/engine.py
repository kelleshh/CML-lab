"""Task runners compose preparation, fit, evaluation and validation adapters."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve

from .artifacts import FittedArtifact
from .fitting import fit_artifact
from .metrics import TaskMetrics, REGRESSION_TASKS
from .splitting import holdout, folds
from .matrices import dense, for_estimator
from .feature_names import feature_names


def plain(value):
    if isinstance(value, dict): return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [plain(v) for v in value]
    if isinstance(value, np.ndarray): return plain(value.tolist())
    if isinstance(value, np.generic): return plain(value.item())
    if isinstance(value, (pd.Timestamp, np.datetime64)): return str(value)
    if isinstance(value,float) and not np.isfinite(value): return None
    return value


class ExperimentEngine:
    def __init__(self, data_gateway, catalogue):
        self.data = data_gateway
        self.catalogue = catalogue
        self.metrics = TaskMetrics()

    def run(self, spec, progress=lambda event:None, cancelled=lambda:False):
        started=time.perf_counter()
        spec=deepcopy(spec)
        X,y,context=self.prepare_data(spec)
        descriptor=self.catalogue.descriptor(spec["algorithm_id"])
        if len(X)*len(X.columns)>5_000_000: raise ValueError("Опыт ограничен пятью миллионами исходных ячеек.")
        max_rows=descriptor.get("capabilities",{}).get("max_rows") or (3000 if any(token in descriptor["id"] for token in ("gaussian_process","spectral","affinity","tsne")) else 100000)
        if len(X)>max_rows: raise ValueError(f"Этот алгоритм ограничен {max_rows} строками в интерактивной лаборатории.")
        progress({"type":"progress","progress":.08,"message":"Проверены данные, задача и роли столбцов"})
        if spec["task"] in {"clustering","anomaly","reduction"}:
            result,artifact=self.run_unsupervised(spec,X,context,progress,cancelled)
        else:
            result,artifact=self.run_supervised(spec,X,y,context,progress,cancelled)
        if cancelled(): raise InterruptedError("Расчет отменен.")
        artifact_path=spec.pop("artifact_path",None)
        if artifact_path:
            path=Path(artifact_path);path.parent.mkdir(parents=True,exist_ok=True)
            temporary=path.with_suffix(".tmp")
            joblib.dump(artifact,temporary)
            temporary.replace(path)
        result.update(task=spec["task"],algorithm_id=spec["algorithm_id"],model_name=descriptor["name"],
                      effective_spec=spec,timing={"seconds":time.perf_counter()-started},warnings=context.get("warnings",[]))
        progress({"type":"progress","progress":.98,"message":"Сохранен полный обученный конвейер"})
        return plain(result)

    def prepare_data(self,spec):
        frame=self.data.frame(spec["dataset_id"])
        roles=spec.get("roles") or {}
        target=spec.get("target")
        task=spec["task"]
        features=spec.get("features")
        if features is None:features=[c for c in frame.columns if c!=target and c not in roles.values()]
        if task in {"forecasting","panel"}:
            from .temporal import build_temporal_frame
            settings=spec.get("temporal") or {}
            temporal=build_temporal_frame(frame,target,roles.get("time_column"),roles.get("entity_column") if task=="panel" else None,
                                          lags=settings.get("lags",[1,2,3]),rolling=settings.get("rolling_windows",[3]),horizon=settings.get("horizon",1),features=features)
            return temporal.X,temporal.y.to_numpy(),{"origins":np.asarray(temporal.origins),"target_times":np.asarray(temporal.targets),
                    "entities":np.asarray(temporal.entities),"indices":np.asarray(temporal.row_indices),"temporal":temporal.history_config,
                    "warnings":["Проверка ряда использует доступные прошлые наблюдения для каждого нового момента; это не одновременный прогноз всего будущего блока."]}
        mask=frame[target].notna() if target and task not in {"clustering","anomaly","reduction"} else np.ones(len(frame),dtype=bool)
        original=np.flatnonzero(mask)
        data=frame.loc[mask].reset_index(drop=True)
        X=data[features].copy()
        if not len(features): raise ValueError("Не осталось признаков для обучения.")
        if len(X)<12: raise ValueError("Для опыта нужно минимум 12 наблюдений.")
        y=None
        if task=="classification": y=data[target].astype(str).to_numpy()
        elif task in REGRESSION_TASKS or task=="ranking":
            y=pd.to_numeric(data[target],errors="raise").to_numpy(dtype=float)
            if not np.isfinite(y).all(): raise ValueError("Числовая цель должна быть конечной.")
        if task=="ranking" and (np.any(y<0) or np.any(y>30) or not np.equal(y,np.floor(y)).all()):
            raise ValueError("Релевантность ранжирования: целые оценки от 0 до 30.")
        grouping=roles.get("query_column") if task=="ranking" else (spec.get("validation") or {}).get("group_column")
        if grouping and data[grouping].isna().any():
            raise ValueError("Идентификатор группы или запроса не может быть пропущен.")
        groups=data[grouping].astype(str).to_numpy() if grouping else None
        reference=data[roles["reference_target"]].astype(str).to_numpy() if roles.get("reference_target") else None
        weights=pd.to_numeric(data[roles["weight_column"]],errors="raise").to_numpy(dtype=float) if roles.get("weight_column") else None
        if weights is not None and (not np.isfinite(weights).all() or np.any(weights<0) or not np.any(weights>0)):
            raise ValueError("Веса строк должны быть конечными, неотрицательными и хотя бы один положительным.")
        return X,y,{"groups":groups,"reference":reference,"weights":weights,"indices":original,
                    "warnings":[f"Исключены {len(frame)-len(data)} строки без известной цели."] if len(data)!=len(frame) else []}

    def run_supervised(self,spec,X,y,context,progress,cancelled):
        parts=holdout(spec,len(X),y,context.get("groups"),context.get("origins"),context.get("target_times"))
        encoder=None
        if spec["task"]=="classification":
            encoder=LabelEncoder().fit(y[parts["train"]])
            if len(encoder.classes_)<2: raise ValueError("В обучении должны быть хотя бы два класса.")
            try:y=encoder.transform(y)
            except ValueError as error:raise ValueError("Проверка содержит класс, отсутствующий в обучении. Используй стратифицированное разделение или добавь данные.") from error
        train=parts["train"]
        groups=context.get("groups");weights=context.get("weights")
        inner=folds(spec,y[train],None if groups is None else groups[train],
                    None if context.get("origins") is None else context["origins"][train],
                    None if context.get("target_times") is None else context["target_times"][train],force=bool(spec.get("search")))
        cv=None;search=None
        if spec.get("search"):
            from .tuning import tune
            best,search=tune(spec,X.iloc[train],y[train],self.catalogue,self.metrics,inner,
                             label_encoder=encoder,groups=None if groups is None else groups[train],
                             weights=None if weights is None else weights[train],progress=progress,cancelled=cancelled)
            spec["params"]={key: value for key, value in best.items() if not key.startswith('pipeline__')}
            pipeline_params = {key[len('pipeline__'):]: value for key, value in best.items() if key.startswith('pipeline__')}
            if pipeline_params: spec['preprocessing'].setdefault('pipeline_params', {}).update(pipeline_params)
        if inner:
            cv=self.evaluate_folds(spec,X.iloc[train],y[train],inner,encoder,None if groups is None else groups[train],
                                    None if weights is None else weights[train],cancelled)
        artifact,sampling,trace=fit_artifact(spec,X.iloc[train],y[train],self.catalogue,label_encoder=encoder,
                                            groups=None if groups is None else groups[train],weights=None if weights is None else weights[train],
                                            progress=progress,cancelled=cancelled,validation=(X.iloc[parts["validation"]],y[parts["validation"]]))
        artifact.temporal=context.get("temporal") or {}
        evaluations={}
        for name,indices in parts.items():
            evaluations[name]=self.evaluate_artifact(spec,artifact,X.iloc[indices],y[indices],context["indices"][indices],
                                                     None if groups is None else groups[indices])
        names=artifact.feature_names
        model=artifact.estimator
        diagnostics={"preprocessing":{"features":names,"original_features":list(X.columns),"n_features":len(names)},
                     "sampling":sampling,"cv":cv,"search":search,"split_indices":{k:context["indices"][v[:1000]].tolist() for k,v in parts.items() if k!="test"}}
        from .diagnostics import build_diagnostics
        extra, diagnostic_warnings = build_diagnostics(
            spec, artifact, X.iloc[train], y[train], X.iloc[parts["validation"]], y[parts["validation"]],
            self.catalogue, self.metrics, groups=groups,
            weights=None if weights is None else weights[train], progress=progress, cancelled=cancelled)
        diagnostics.update(extra)
        if spec["task"] in {"regression", "classification"}:
            from .prediction_views import build_prediction_view
            diagnostics["prediction_view"] = build_prediction_view(
                artifact, X, y, train, parts["validation"], features=spec.get("plot_features"))
        context.setdefault("warnings", []).extend(diagnostic_warnings)
        if hasattr(model,"feature_importances_"):
            diagnostics["importance"]={"names":names,"values":model.feature_importances_.tolist()}
        if hasattr(model,"coef_"):
            diagnostics["coefficients"]={"names":names,"values":np.asarray(model.coef_).tolist(),"intercept":np.asarray(model.intercept_).tolist()}
        if task:=spec["task"]:
            if task in {"forecasting","panel"}:
                for name,indices in parts.items():
                    evaluations[name]["time"]={"origins":context["origins"][indices[:2000]].astype(str).tolist(),
                        "targets":context["target_times"][indices[:2000]].astype(str).tolist(),"entities":context["entities"][indices[:2000]].astype(str).tolist()}
                    for row,origin,target_time,entity in zip(evaluations[name]["rows"],context["origins"][indices[:2000]],context["target_times"][indices[:2000]],context["entities"][indices[:2000]]):
                        row.update(time=str(target_time),origin=str(origin),entity=str(entity))
        return {"evaluations":evaluations,"diagnostics":diagnostics,"trace":trace},artifact

    def evaluate_artifact(self,spec,artifact,X,y,indices=None,groups=None):
        pred=artifact.predict_encoded(X)
        proba=artifact.predict_proba(X) if spec["task"]=="classification" and hasattr(artifact.estimator,"predict_proba") else None
        nc=len(artifact.label_encoder.classes_) if artifact.label_encoder is not None else None
        values,details=self.metrics.evaluate(spec["task"],y,pred,selection=spec.get("metrics"),probabilities=proba,n_classes=nc,
                                             groups=groups,expression=spec.get("custom_metric"),metric_params=spec.get("metric_params"))
        encoded_y=artifact.label_encoder.inverse_transform(y.astype(int)) if nc else y
        displayed_pred=artifact.label_encoder.inverse_transform(pred.astype(int)) if nc else pred
        indices=np.arange(len(y)) if indices is None else indices
        rows=[{"index":int(i),"actual":plain(a),"predicted":plain(p)} for i,a,p in zip(indices[:2000],encoded_y[:2000],displayed_pred[:2000])]
        result={"metrics":values,"metric_details":details,"rows":rows,"n_rows":len(y)}
        if groups is not None:
            for row,group in zip(rows,groups[:2000]):row["query"]=str(group)
        if nc:
            result["classification"]={"classes":artifact.label_encoder.classes_.tolist(),"confusion":confusion_matrix(y,pred,labels=np.arange(nc)).tolist()}
            if proba is not None:
                result["classification"]["probabilities"]=proba[:2000].tolist()
                if nc==2 and len(np.unique(y))==2:
                    fpr,tpr,_=roc_curve(y,proba[:,1]);precision,recall,_=precision_recall_curve(y,proba[:,1])
                    result["classification"].update(roc={"fpr":fpr.tolist(),"tpr":tpr.tolist()},pr={"precision":precision.tolist(),"recall":recall.tolist()})
        return result

    def evaluate_folds(self,spec,X,y,plans,encoder,groups,weights,cancelled):
        from joblib import Parallel, delayed
        fold_spec={**spec,"n_jobs":1}
        def evaluate(index,tr,va):
            start=time.perf_counter()
            artifact,_,_=fit_artifact(fold_spec,X.iloc[tr],y[tr],self.catalogue,label_encoder=encoder,
                                      groups=None if groups is None else groups[tr],weights=None if weights is None else weights[tr],cancelled=cancelled)
            evaluation=self.evaluate_artifact(spec,artifact,X.iloc[va],y[va],groups=None if groups is None else groups[va])
            return {"fold":index+1,"train_rows":len(tr),"validation_rows":len(va),"metrics":evaluation["metrics"],"seconds":time.perf_counter()-start}
        reports=Parallel(n_jobs=spec.get("n_jobs",1),prefer="threads")(delayed(evaluate)(i,tr,va) for i,(tr,va) in enumerate(plans))
        keys=set().union(*(r["metrics"] for r in reports))
        means={};std={}
        for key in keys:
            values=[r["metrics"][key] for r in reports if r["metrics"].get(key) is not None]
            means[key]=float(np.mean(values)) if values else None;std[key]=float(np.std(values)) if values else None
        return {"folds":reports,"mean":means,"std":std,"n_splits":len(reports)}

    def run_unsupervised(self,spec,X,context,progress,cancelled):
        from .preparation import build_preprocessor
        if spec.get("search"):raise ValueError("Подбор для обучения без учителя пока требует явного внешнего критерия; обычный supervised поиск сюда не применяется.")
        preprocessing=build_preprocessor(X,spec.get("preprocessing") or {},task=spec["task"],seed=spec.get("seed",42))
        capabilities=self.catalogue.descriptor(spec["algorithm_id"]).get("capabilities",{})
        model=self.catalogue.build(spec["task"],spec["algorithm_id"],spec.get("params"),spec.get("seed",42),spec.get("n_jobs",1))
        if hasattr(self.catalogue, 'effective_capabilities'): capabilities=self.catalogue.effective_capabilities(model,capabilities)
        Xt=for_estimator(preprocessing.fit_transform(X),capabilities)
        progress({"type":"progress","progress":.2,"message":"Подготовлена рабочая таблица обучения без учителя"})
        if cancelled():raise InterruptedError("Расчет отменен.")
        if spec["task"]=="reduction": predictions=dense(model.fit_transform(Xt))
        elif capabilities.get('fit_predict'): predictions=np.asarray(model.fit_predict(Xt))
        elif capabilities.get('predict'):
            model.fit(Xt)
            predictions=np.asarray(model.predict(Xt))
        else: raise ValueError('Настроенная модель не поддерживает fit_predict или predict для этой задачи.')
        values,details=self.metrics.evaluate(spec["task"],None,predictions,selection=spec.get("metrics"),X=Xt,reference=context.get("reference"),model=model)
        if spec["task"]=="reduction":coordinates=predictions
        else:coordinates=dense(Xt[:,:min(3,Xt.shape[1])])
        embedding={"coordinates":coordinates[:2000].tolist(),"indices":context["indices"][:2000].tolist(),
                   "labels":predictions[:2000].tolist() if spec["task"]!="reduction" else None}
        rows=[{"index":int(index),"predicted":plain(prediction),"coordinates":plain(coordinate)} for index,prediction,coordinate in zip(context["indices"][:2000],predictions[:2000],coordinates[:2000])]
        names = feature_names(preprocessing, Xt.shape[1])
        result={"evaluations":{"train":{"metrics":values,"metric_details":details,"n_rows":len(X),"rows":rows}},
                "diagnostics":{"embedding":embedding,"scope":"Вся выбранная рабочая таблица; отдельной оценки новых объектов нет.",
                    "preprocessing":{"features":names,"original_features":list(X.columns)}},"trace":[]}
        from .diagnostics import build_diagnostics
        extra, diagnostic_warnings = build_diagnostics(
            spec, None, X, None, X, None, self.catalogue, self.metrics, progress=progress, cancelled=cancelled)
        result["diagnostics"].update(extra)
        context.setdefault("warnings", []).extend(diagnostic_warnings)
        if spec["task"]=="clustering":
            unique,counts=np.unique(predictions,return_counts=True)
            result["diagnostics"]["clusters"]={"labels":unique.tolist(),"counts":counts.tolist(),"noise_rows":int(np.sum(predictions==-1))}
        if spec["task"]=="anomaly":
            scores=-np.asarray(model.score_samples(Xt)) if hasattr(model,"score_samples") else -np.asarray(getattr(model,"negative_outlier_factor_",np.zeros(len(Xt))))
            result["diagnostics"]["anomaly_scores"]=scores[:2000].tolist()
        return result,FittedArtifact(spec["task"],preprocessing,model,list(X.columns),input_capabilities=capabilities,transformed_features=names)
