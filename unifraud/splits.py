"""Leakage-aware data splits.

* ``chronological_split``: for CC (transaction time) and MM (simulation step) - train < val < test
  in time, no shuffling, so drift is part of the evaluation.
* ``grouped_stratified_split``: for HI - all claims of a patient (group) land in one partition;
  class balance is approximated with StratifiedGroupKFold.
* ``stratified_split``: plain stratified fallback.
All functions return index arrays (positional) for train / val / test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


def _check_ratios(train: float, val: float, test: float) -> None:
    if min(train, val, test) <= 0 or abs(train + val + test - 1.0) > 1e-9:
        raise ValueError("ratios must be positive and sum to 1")


def chronological_split(time: pd.Series | np.ndarray, train=0.70, val=0.15, test=0.15):
    _check_ratios(train, val, test)
    t = np.asarray(time)
    order = np.argsort(t, kind="stable")
    n = len(t)
    a, b = int(round(n * train)), int(round(n * (train + val)))
    return order[:a], order[a:b], order[b:]


def grouped_stratified_split(y, groups, train=0.70, val=0.15, test=0.15, seed=42):
    """Group-disjoint split with approximately stratified labels."""
    _check_ratios(train, val, test)
    y = np.asarray(y)
    groups = np.asarray(groups)
    idx = np.arange(len(y))

    def carve(pool, frac):
        k = max(2, int(round(1.0 / frac)))
        sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
        rest, part = next(iter(sgkf.split(pool, y[pool], groups[pool])))
        return pool[rest], pool[part]

    tv, te = carve(idx, test)
    tr, va = carve(tv, val / (train + val))
    return tr, va, te


def stratified_split(y, train=0.70, val=0.15, test=0.15, seed=42):
    _check_ratios(train, val, test)
    y = np.asarray(y)
    idx = np.arange(len(y))
    tv, te = train_test_split(idx, test_size=test, stratify=y, random_state=seed)
    tr, va = train_test_split(tv, test_size=val / (train + val), stratify=y[tv], random_state=seed)
    return tr, va, te


def assert_disjoint(*parts) -> None:
    seen: set[int] = set()
    for p in parts:
        s = set(np.asarray(p).tolist())
        if seen & s:
            raise AssertionError("partitions overlap")
        seen |= s
