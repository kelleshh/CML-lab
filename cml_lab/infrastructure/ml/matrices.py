"""Keep sparse inputs sparse and bound deliberate dense conversions."""
import numpy as np
from scipy import sparse

MAX_DENSE_CELLS = 5_000_000


def dense(matrix):
    if matrix.shape[0] * matrix.shape[1] > MAX_DENSE_CELLS:
        raise ValueError("Плотная матрица ограничена пятью миллионами ячеек. Уменьшите словарь или число признаков.")
    return matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)


def for_estimator(matrix, capabilities):
    if len(matrix.shape) != 2:
        raise ValueError("Подготовка должна вернуть двумерную матрицу признаков.")
    if capabilities.get("requires_dense"):
        matrix = dense(matrix)
    values = matrix.data if sparse.issparse(matrix) else np.asarray(matrix)
    if not capabilities.get("allows_missing") and not np.isfinite(values).all():
        raise ValueError("Эта модель требует конечные признаки. Заполните пропуски и проверьте преобразования.")
    if np.isinf(values).any():
        raise ValueError("Признаки не могут содержать бесконечности.")
    if capabilities.get("input_domain") == "nonnegative" and np.any(values < 0):
        raise ValueError("Этому алгоритму нужны неотрицательные признаки. Выберите MinMax с clip или отключите центрирование.")
    return matrix
