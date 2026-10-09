"""Training-time budgets and split compatibility for declarative transformers.

Compilation itself stays an exact, unfitted sklearn definition. The application
uses a fresh guarded clone for each train fit, including every CV fold.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import sparse
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.cross_decomposition import CCA, PLSCanonical, PLSRegression
from sklearn.decomposition import KernelPCA
from sklearn.impute import IterativeImputer
from sklearn.manifold import Isomap, LocallyLinearEmbedding
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, SplineTransformer

from .compiler import PipelineDefinitionError, validate_pipeline_object


MAX_DENSE_FEATURES = 2000
MAX_SPARSE_FEATURES = 20_000
MAX_CELLS = 15_000_000


def _shape(X):
    shape = getattr(X, "shape", None)
    if shape is None or len(shape) != 2:
        raise ValueError("Этот преобразователь требует двумерную матрицу фич.")
    return shape


def _budget(rows, features, *, sparse_output=False):
    maximum = MAX_SPARSE_FEATURES if sparse_output else MAX_DENSE_FEATURES
    if features > maximum or rows * features > MAX_CELLS:
        raise ValueError(f"Pipeline создаст более {maximum} фич или 15 млн ячеек. Уменьшите расширение или число исходных фич.")


class BudgetedPolynomialFeatures(PolynomialFeatures):
    def fit(self, X, y=None):
        rows, count = _shape(X)
        low, high = self.degree if isinstance(self.degree, tuple) else (0, self.degree)
        low = max(1, low)
        if self.interaction_only:
            features = sum(math.comb(count, degree) for degree in range(low, min(high, count) + 1))
        else:
            features = math.comb(count + high, high) - math.comb(count + low - 1, low - 1)
        features += int(self.include_bias)
        _budget(rows, features, sparse_output=sparse.issparse(X))
        return super().fit(X, y)

    def transform(self, X):
        if hasattr(self, "n_output_features_"):
            _budget(_shape(X)[0], self.n_output_features_, sparse_output=sparse.issparse(X))
        return super().transform(X)


class BudgetedSplineTransformer(SplineTransformer):
    def fit(self, X, y=None, sample_weight=None):
        rows, count = _shape(X)
        knots = self.n_knots if isinstance(self.knots, str) else np.asarray(self.knots).shape[0]
        bases = knots - 1 if self.extrapolation == "periodic" else knots + self.degree - 1
        _budget(rows, count * (bases - int(not self.include_bias)), sparse_output=self.sparse_output)
        return super().fit(X, y, sample_weight=sample_weight)

    def transform(self, X):
        if hasattr(self, "n_features_out_"):
            _budget(_shape(X)[0], self.n_features_out_, sparse_output=self.sparse_output)
        return super().transform(X)


class BudgetedOneHotEncoder(OneHotEncoder):
    def fit(self, X, y=None):
        result = super().fit(X, y)
        self._check_output(X)
        return result

    def _check_output(self, X):
        # _n_features_outs counts infrequent/drop categories exactly, without
        # allocating a potentially enormous array of feature-name strings.
        count = sum(self._n_features_outs)
        _budget(_shape(X)[0], count, sparse_output=self.sparse_output)

    def transform(self, X):
        if hasattr(self, "_n_features_outs"):
            self._check_output(X)
        return super().transform(X)


class BudgetedIterativeImputer(IterativeImputer):
    def fit_transform(self, X, y=None, **params):
        rows, features = _shape(X)
        if features > 100 or rows * features > 500_000 or self.max_iter > 30:
            raise ValueError("IterativeImputer ограничен 100 фичами, 500 000 ячейками и 30 итерациями.")
        return super().fit_transform(X, y, **params)


class _PairwiseBudget:
    def _check_pairwise(self, X):
        rows, _ = _shape(X)
        if rows * rows > MAX_CELLS:
            raise ValueError("Kernel/manifold Pipeline превысит 15 млн попарных train расстояний. Уменьшите число строк.")

    def fit(self, X, y=None):
        self._check_pairwise(X)
        return super().fit(X, y)

    def fit_transform(self, X, y=None, **fit_params):
        self._check_pairwise(X)
        return super().fit_transform(X, y, **fit_params)


class BudgetedKernelPCA(_PairwiseBudget, KernelPCA):
    pass


class BudgetedIsomap(_PairwiseBudget, Isomap):
    pass


class BudgetedLocallyLinearEmbedding(_PairwiseBudget, LocallyLinearEmbedding):
    pass


class _FeatureScoresOnly:
    def fit_transform(self, X, y=None, **fit_params):
        # sklearn cross-decomposition returns (X_scores, y_scores) when y is
        # supplied. Preprocessing must pass only X scores to the next step;
        # y stays owned by the experiment's task, never becomes an input feature.
        return self.fit(X, y, **fit_params).transform(X)


class FeaturePLSRegression(_FeatureScoresOnly, PLSRegression):
    pass


class FeaturePLSCanonical(_FeatureScoresOnly, PLSCanonical):
    pass


class FeatureCCA(_FeatureScoresOnly, CCA):
    pass


BOUNDED = {
    PolynomialFeatures: BudgetedPolynomialFeatures,
    SplineTransformer: BudgetedSplineTransformer,
    OneHotEncoder: BudgetedOneHotEncoder,
    IterativeImputer: BudgetedIterativeImputer,
    KernelPCA: BudgetedKernelPCA,
    Isomap: BudgetedIsomap,
    LocallyLinearEmbedding: BudgetedLocallyLinearEmbedding,
    PLSRegression: FeaturePLSRegression,
    PLSCanonical: FeaturePLSCanonical,
    CCA: FeatureCCA,
}


def prepare_pipeline_for_fit(transformer, task="regression", split_kind="holdout"):
    """Clone, enforce ordered/grouped CV rules and add preallocation guards."""
    validate_pipeline_object(transformer)
    ordered_or_grouped = task in ("forecasting", "panel") or split_kind in (
        "time_series", "timeseries", "blocked", "blocked_time", "expanding", "expanding_window",
        "group", "group_kfold", "group_shuffle", "group_shuffle_split", "leave_one_group_out", "purged_time_series", "ordered",
    )
    forbidden = {"TargetEncoder", "RFECV", "SequentialFeatureSelector"}

    def convert(value):
        name = type(value).__name__
        if ordered_or_grouped and name in forbidden:
            raise PipelineDefinitionError(f"{name} использует внутреннюю CV без time/group информации. Для этой задачи выберите другой шаг Pipeline.")
        result = clone(value)
        if isinstance(result, Pipeline):
            result.steps = [(step_name, convert(step) if hasattr(step, "get_params") else step) for step_name, step in result.steps]
        elif isinstance(result, ColumnTransformer):
            result.transformers = [(step_name, convert(step) if hasattr(step, "get_params") else step, columns) for step_name, step, columns in result.transformers]
            if hasattr(result.remainder, "get_params"):
                result.remainder = convert(result.remainder)
        elif isinstance(result, FeatureUnion):
            result.transformer_list = [(step_name, convert(step) if hasattr(step, "get_params") else step) for step_name, step in result.transformer_list]
        elif type(result) in BOUNDED:
            output = getattr(result, "_sklearn_output_config", {}).get("transform")
            result = BOUNDED[type(result)](**result.get_params(deep=False))
            if output:
                result.set_output(transform=output)
        # Selectors and iterative imputers may contain an estimator. Inspecting
        # every child protects nested compositions without fitting anything.
        for key, child in list(result.get_params(deep=False).items()):
            if key not in ("steps", "transformers", "transformer_list", "remainder") and hasattr(child, "get_params"):
                result.set_params(**{key: convert(child)})
        return result

    return convert(transformer)
