"""Numerically defined target interpolation for the upstream SMOTER algorithm."""
import numpy as np


def interpolate_target(data, first, second, features, ranges):
    """Interpolate y with inverse distances in feature space, excluding y itself."""
    left = data.iloc[first, :-1].to_numpy(dtype=float)
    right = data.iloc[second, :-1].to_numpy(dtype=float)
    spans = np.asarray(ranges, dtype=float)
    variable = spans > 0
    if variable.any():
        a = float(np.linalg.norm((left[variable] - features[variable]) / spans[variable]))
        b = float(np.linalg.norm((right[variable] - features[variable]) / spans[variable]))
    else:
        a = b = 0.0
    y_left, y_right = float(data.iloc[first, -1]), float(data.iloc[second, -1])
    if a + b == 0:
        return (y_left + y_right) / 2
    return (b * y_left + a * y_right) / (a + b)
