"""Outer holdout and inner fold plans with task-aware grouping and chronology."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit, StratifiedKFold, RepeatedStratifiedKFold, StratifiedShuffleSplit, GroupKFold, TimeSeriesSplit
from linear_lab.validation import build_cv


def holdout(spec, n, y=None, groups=None, origins=None, target_times=None):
    settings = spec.get("split") or {}
    fractions = [float(settings.get(key, default)) for key, default in (("train",.6),("validation",.2),("test",.2))]
    if not all(np.isfinite(fractions)) or min(fractions) <= 0 or abs(sum(fractions)-1) > 1e-6:
        raise ValueError("Доли обучения, выбора и теста должны быть положительными и в сумме равны 1.")
    indices = np.arange(n)
    seed = spec.get("seed",42)
    if origins is not None:
        times = np.asarray(origins)
        unique = np.unique(times)
        if len(unique) < 10: raise ValueError("Для временного опыта нужны хотя бы десять разных дат после лагов.")
        a, b = int(len(unique)*fractions[0]), int(len(unique)*(fractions[0]+fractions[1]))
        val_start, test_start = unique[a], unique[b]
        targets = times if target_times is None else np.asarray(target_times)
        parts = {"train":indices[(times<val_start)&(targets<val_start)],
                 "validation":indices[(times>=val_start)&(times<test_start)&(targets<test_start)],
                 "test":indices[times>=test_start]}
    elif groups is not None:
        values = np.asarray(groups).astype(str)
        if len(np.unique(values)) < 5: raise ValueError("Для группового holdout нужны хотя бы пять разных групп.")
        tr, rest = next(GroupShuffleSplit(1,train_size=fractions[0],random_state=seed).split(indices,groups=values))
        vl, te = next(GroupShuffleSplit(1,train_size=fractions[1]/(1-fractions[0]),random_state=seed+1).split(rest,groups=values[rest]))
        parts = {"train":tr,"validation":rest[vl],"test":rest[te]}
    elif settings.get("shuffle",True) is False:
        a,b=int(n*fractions[0]),int(n*(fractions[0]+fractions[1]))
        parts={"train":indices[:a],"validation":indices[a:b],"test":indices[b:]}
    else:
        strata = y if spec["task"] == "classification" else None
        tr,rest=train_test_split(indices,train_size=fractions[0],random_state=seed,stratify=strata)
        vl,te=train_test_split(rest,train_size=fractions[1]/(1-fractions[0]),random_state=seed+1,stratify=np.asarray(y)[rest] if strata is not None else None)
        parts={"train":tr,"validation":vl,"test":te}
    if min(map(len,parts.values())) < 2: raise ValueError("После разбиения в каждой части должно остаться хотя бы два наблюдения.")
    return parts


def folds(spec, y, groups=None, origins=None, target_times=None, force=False):
    config=dict(spec.get("validation") or {})
    strategy=config.get("strategy","none")
    strategy={"time_series":"timeseries", "rolling_origin":"timeseries", "expanding_window":"timeseries"}.get(strategy, strategy)
    if strategy == "none" and not force: return []
    if strategy == "none": strategy="stratified_kfold" if spec["task"]=="classification" else "kfold"
    count=int(config.get("folds",3))
    if not 2 <= count <= 20: raise ValueError("Частей CV: от 2 до 20.")
    n=len(y)
    if origins is not None:
        if strategy not in {"timeseries","time_series","none","kfold"}:
            raise ValueError("Временные данные проверяются только прошлое → будущее.")
        times=np.asarray(origins);unique=np.unique(times)
        splitter=TimeSeriesSplit(count,gap=int(config.get("gap",0)),max_train_size=config.get("max_train_size"),test_size=config.get("test_size"))
        plans=[]
        for tr,va in splitter.split(unique):
            ti=np.flatnonzero(np.isin(times,unique[tr]));vi=np.flatnonzero(np.isin(times,unique[va]))
            if target_times is not None: ti=ti[np.asarray(target_times)[ti] < unique[va[0]]]
            if len(ti)<2 or len(vi)<2: raise ValueError("После временного разрыва недостаточно наблюдений в части CV.")
            plans.append((ti,vi))
        return plans
    if groups is not None:
        if strategy not in {"group_kfold","group_shuffle_split","leave_one_group_out","kfold","stratified_kfold"}:
            raise ValueError("Для групповых данных выбери групповую CV.")
        config["strategy"]="group_kfold" if strategy in {"kfold","stratified_kfold"} else strategy
        return list(build_cv(config,n,y,groups,seed=spec.get("seed",42)).splits)
    if strategy in {"stratified_kfold","repeated_stratified_kfold","stratified_shuffle_split"}:
        if spec["task"] != "classification": raise ValueError("Стратификация по классам относится к классификации.")
        if strategy == "repeated_stratified_kfold":
            repeats=int(config.get("repeats",2))
            if not 1 <= repeats <= 10 or count*repeats>100:
                raise ValueError("Повторов CV: от 1 до 10, всего не более 100 обучений.")
            splitter=RepeatedStratifiedKFold(n_splits=count,n_repeats=repeats,random_state=spec.get("seed",42))
        elif strategy == "stratified_shuffle_split":
            splitter=StratifiedShuffleSplit(n_splits=count,test_size=config.get("test_size",.2),random_state=spec.get("seed",42))
        else:
            shuffle=bool(config.get("shuffle",True));splitter=StratifiedKFold(count,shuffle=shuffle,random_state=spec.get("seed",42) if shuffle else None)
        plans=list(splitter.split(np.arange(n),y))
    else:
        config["strategy"]=strategy
        plans=list(build_cv(config,n,y,seed=spec.get("seed",42)).splits)
    if len(plans)>100: raise ValueError("CV ограничена ста обучениями.")
    return plans
