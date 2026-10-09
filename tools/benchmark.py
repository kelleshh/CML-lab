"""Reproducible local timings; never an assertion that more workers are faster.

Run from the project directory: .venv/bin/python tools/benchmark.py --profile
"""

from __future__ import annotations

import argparse
import cProfile
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import pstats
import statistics
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from linear_lab.datasets import DataService
from linear_lab.models import ModelRegistry
from linear_lab.training import TrainingService


def timed(function, repeats):
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        samples.append(time.perf_counter() - started)
    return {"seconds": samples, "median_seconds": statistics.median(samples)}


def run(args):
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    results = []
    profiled = None
    with tempfile.TemporaryDirectory(prefix="linear-lab-benchmark-") as temporary:
        data = DataService(Path(temporary) / "data")
        trainer = TrainingService(data, ModelRegistry())
        metadata = data.load({"kind": "synthetic", "name": "correlated", "params": {
            "n_samples": args.rows, "n_features": args.features,
            "noise": 2., "correlation": .6, "seed": 417,
        }})
        request = {"dataset_id": metadata["id"], "seed": 417,
                   "preprocessing": {"scale": True, "degree": args.degree, "impute": True},
                   "cv_config": {"strategy": "kfold", "folds": args.folds},
                   "metrics": ["mse", "rmse", "mae", "r2"],
                   "regularization_path": True, "permutation_importance": True,
                   "artifact_path": str(Path(temporary) / "model.joblib")}
        _, X, _, y, groups = trainer._dataset_context(request)
        parts = trainer._partition(X, y, groups, request)
        preprocessor, _, _ = trainer._preprocessor(X, request["preprocessing"])
        Xt = preprocessor.fit_transform(X.iloc[parts["train"]], y[parts["train"]])
        names = preprocessor.get_feature_names_out().tolist()
        for model_id in args.models:
            params = {"alpha": 1.} if model_id == "ridge" else {"alpha": .1}
            reference = None
            # Alternate configurations on the same generated data; threadpoolctl
            # caps the native pools for serial and parallel timings alike.
            for jobs in args.jobs:
                config = dict(request, model=model_id, params=params, n_jobs=jobs)
                estimator = trainer.model_registry.create(model_id, params, 417)
                def alpha_path():
                    return trainer._regularization_path(config, model_id, estimator, Xt, y[parts["train"]], names, lambda event: None, lambda: False, [])
                with threadpool_limits(limits=1):
                    # Warm imports, estimators and compiled numerical routines.
                    path = alpha_path()
                    trainer.run(config)
                    if reference is None:
                        reference = path
                    np.testing.assert_allclose(path["coefficients"], reference["coefficients"], rtol=1e-10, atol=1e-10)
                    np.testing.assert_array_equal(path["alphas"], reference["alphas"])
                    path_timing = timed(alpha_path, args.repeats)
                    experiment_timing = timed(lambda: trainer.run(config), args.repeats)
                    if args.profile and profiled is None:
                        profiler = cProfile.Profile()
                        profiler.runcall(trainer.run, config)
                        stream = io.StringIO()
                        pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumulative").print_stats(25)
                        profiled = stream.getvalue()
                results.append({"model": model_id, "n_jobs": jobs, "alpha_path": path_timing,
                                "experiment": experiment_timing})
        report = {"hardware": {"platform": platform.platform(), "machine": platform.machine(),
                   "processor": platform.processor(), "logical_cpus": os.cpu_count()},
                  "versions": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn", "joblib", "threadpoolctl")},
                  "workload": {"rows": args.rows, "features": args.features, "degree": args.degree,
                               "train_shape": list(Xt.shape), "cv_folds": args.folds, "repeats": args.repeats,
                               "seed": 417, "blas_threads": 1},
                  "native_pools": threadpool_info(), "results": results,
                  "note": "Local warm timings; scheduling, hardware and workload change the result. Independent alpha-path coefficients were checked equal. The full experiment includes preprocessing, diagnostics, alpha path, CV, permutation importance and export.",
                  "profile": profiled}
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument("--features", type=int, default=10)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--jobs", nargs="+", type=int, default=[1, 2, 4], choices=[1, 2, 3, 4])
    parser.add_argument("--models", nargs="+", default=["ridge", "lasso"], choices=["ridge", "lasso", "elasticnet"])
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    encoded = json.dumps(run(args), ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
