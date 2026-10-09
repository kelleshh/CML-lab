# Regression sampler sources

The four Python files are derived from PyPI `smogn==0.1.2` and
`ImbalancedLearningRegression==0.0.2`, under their GPL-3.0 licenses.
The original license texts and source SHA-256 values are included here.

Sources: https://github.com/nickkunz/smogn and
https://github.com/paobranco/ImbalancedLearningRegression .
Algorithm: https://proceedings.mlr.press/v74/branco17a.html .

Local changes:

- Synthetic matrices are initialized, so a target never depends on uninitialized memory.
- Target interpolation uses aggregate distances in **feature space**, excluding the not-yet-written target. Equal positions use the arithmetic mean of both targets.
- Clipping negative values applies only to columns whose whole sampled domain is nonnegative; signed standardized features keep their negative values.
- Constant target columns remain in the last position, while constant input features are restored as upstream intended.
- Imports select these fixed modules directly, without replacing global functions in NumPy or installed packages.
- Console progress bars and a global Pandas option mutation are removed. Seed isolation remains in ResamplingService.

Public sampling APIs and relevance functions otherwise use the pinned upstream packages.
Numerical regression tests include signed affine relationships, target interpolation,
repeatability and independence from external holdout data.
