"""Обучение с пересэмплированием; предсказание сохраняет исходные строки."""

from __future__ import annotations

from sklearn.pipeline import Pipeline


class RegressionPipeline(Pipeline):
    """Fit transformations and the sampler independently in every training fold.

    Sampling changes only the data passed to ``model.fit``. It never runs during
    ``predict`` and cannot duplicate validation or test observations.
    """

    def __init__(self, steps, *, resampling=None, seed=42):
        super().__init__(steps)
        self.resampling = resampling
        self.seed = seed

    def fit(self, X, y=None, **params):
        if params:
            raise ValueError('Передавайте настройки модели через set_params, а не fit.')
        from .preprocessing import ResamplingService

        preprocessor, model = self.steps[0][1], self.steps[-1][1]
        transformed = preprocessor.fit_transform(X, y)
        numeric_only = len(X.select_dtypes(exclude='number').columns) == 0
        sampled, target, summary = ResamplingService().fit_resample(
            transformed, y, self.resampling or {}, seed=self.seed,
            categorical=not numeric_only,
        )
        from sklearn.linear_model import SGDRegressor
        if isinstance(model, SGDRegressor):
            epochs = model.get_params().get('max_iter', 100)
            if not 1 <= epochs <= 1000:
                raise ValueError('Число эпох SGD должно быть от 1 до 1000.')
            for _ in range(epochs):
                model.partial_fit(sampled, target)
        else:
            model.fit(sampled, target)
        self.resampling_summary_ = summary
        return self
