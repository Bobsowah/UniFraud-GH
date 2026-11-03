"""Prevalence-aware metrics and corrected statistics (paper Secs. VII-F, IX).

* ``ap_skill``     (AP - pi) / (1 - pi): 0 = no skill, 1 = perfect, comparable across sectors.
* ``expected_cost``/``cost_threshold``: Prop. cost-sensitive Bayes threshold and iso-cost view.
* ``ece``, ``precision_at_k``, ``bootstrap_ci``.
* ``corrected_resampled_t``: Nadeau-Bengio corrected test (variance factor 1/K + n_te/n_tr).
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score


def ap_skill(y, score) -> float:
    y = np.asarray(y).astype(int)
    pi = y.mean()
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    return float((average_precision_score(y, score) - pi) / (1 - pi))


def auc(y, score) -> float:
    y = np.asarray(y).astype(int)
    return float("nan") if y.sum() in (0, len(y)) else float(roc_auc_score(y, score))


def mcc(y, pred) -> float:
    return float(matthews_corrcoef(np.asarray(y).astype(int), np.asarray(pred).astype(int)))


def bayes_threshold(c_fp: float, c_fn: float) -> float:
    """tau* = c_FP / (c_FP + c_FN): predict fraud iff calibrated posterior >= tau* (Prop. cost)."""
    return c_fp / (c_fp + c_fn)


def expected_cost(y, pred, c_fp: float, c_fn: float) -> float:
    y = np.asarray(y).astype(int)
    pred = np.asarray(pred).astype(int)
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return (c_fp * fp + c_fn * fn) / len(y)


def cost_threshold(y_val, p_val, c_fp: float, c_fn: float, grid=None) -> float:
    """Empirical cost-minimising threshold on VALIDATION data (never on test)."""
    grid = np.linspace(0.005, 0.995, 199) if grid is None else grid
    costs = [expected_cost(y_val, np.asarray(p_val) >= t, c_fp, c_fn) for t in grid]
    return float(grid[int(np.argmin(costs))])


def ece(y, p, bins: int = 10) -> float:
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    out = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            out += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(out)


def brier(y, p) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def precision_at_k(y, score, kappa: float) -> float:
    y = np.asarray(y).astype(int)
    k = max(1, int(np.ceil(kappa * len(y))))
    top = np.argsort(-np.asarray(score), kind="stable")[:k]
    return float(y[top].mean())


def bootstrap_ci(values, B: int = 1000, alpha: float = 0.05, seed: int = 0):
    v = np.asarray([x for x in values if x == x], float)
    if len(v) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(B, len(v)), replace=True).mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def corrected_resampled_t(d, n_train: int, n_test: int):
    """Nadeau-Bengio corrected paired t-test on K per-split differences ``d``.

    t = mean(d) / sqrt((1/K + n_test/n_train) * var(d)),  df = K-1.
    Returns (t, two-sided p).
    """
    d = np.asarray(d, float)
    K = len(d)
    if K < 2:
        return float("nan"), float("nan")
    var = d.var(ddof=1)
    if var == 0:
        return (0.0 if d.mean() == 0 else float("inf")), (1.0 if d.mean() == 0 else 0.0)
    t = d.mean() / np.sqrt((1.0 / K + n_test / n_train) * var)
    return float(t), float(2 * stats.t.sf(abs(t), df=K - 1))


def holm(pvals, alpha: float = 0.05):
    """Holm step-down: returns a boolean array of rejections in the original order."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    rej = np.zeros(m, dtype=bool)
    for rank, i in enumerate(order):
        if p[i] <= alpha / (m - rank):
            rej[i] = True
        else:
            break
    return rej
