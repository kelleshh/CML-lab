"""Portable fitted adapters: raw rows in, task-specific predictions out."""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from .matrices import dense, for_estimator


@dataclass
class FittedArtifact:
    task: str
    preprocessing: object
    estimator: object
    features: list[str]
    label_encoder: object = None
    temporal: dict = field(default_factory=dict)
    input_capabilities: dict = field(default_factory=dict)
    transformed_features: list[str] = field(default_factory=list)

    @property
    def feature_names(self):
        from .feature_names import feature_names
        stored = getattr(self, 'transformed_features', None)
        return stored or feature_names(self.preprocessing, getattr(self.estimator, 'n_features_in_', len(self.features)))

    def transform(self, rows):
        frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
        missing = set(self.features) - set(frame.columns)
        if missing:
            raise ValueError("Не хватает исходных признаков: " + ", ".join(sorted(missing)))
        return for_estimator(self.preprocessing.transform(frame[self.features]), self.input_capabilities)

    def predict_encoded(self, rows):
        if not hasattr(self.estimator, "predict"):
            raise ValueError("Этот метод размечает только обучающую таблицу; повторно обучи его для новых объектов.")
        return np.asarray(self.estimator.predict(self.transform(rows))).reshape(-1)

    def predict(self, rows):
        result = self.predict_encoded(rows)
        if self.label_encoder is not None:
            return self.label_encoder.inverse_transform(result.astype(int))
        return result

    def predict_proba(self, rows):
        if not hasattr(self.estimator, "predict_proba"):
            raise ValueError("Выбранная модель не возвращает вероятности классов.")
        probabilities = np.asarray(self.estimator.predict_proba(self.transform(rows)))
        if self.label_encoder is None:
            return probabilities
        # A group fold can lack a class present in the outer training table.
        # Columns must still correspond to that table's stable encoded labels.
        classes = np.asarray(self.estimator.classes_, dtype=int)
        aligned = np.zeros((len(probabilities), len(self.label_encoder.classes_)), dtype=probabilities.dtype)
        aligned[:, classes] = probabilities
        return aligned

    @property
    def classes_(self):
        return self.label_encoder.classes_ if self.label_encoder is not None else getattr(self.estimator, "classes_", None)

    def reduce(self, rows):
        if not hasattr(self.estimator, "transform"):
            raise ValueError("Этот метод не переносит новые объекты в готовое пространство; нужен новый fit.")
        return dense(self.estimator.transform(self.transform(rows)))
