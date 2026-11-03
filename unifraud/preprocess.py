"""Train-only-fit preprocessing.

Every transformer learns its statistics in ``fit`` from the training partition only; the
validation/test partitions are transformed with those frozen statistics. This replaces the
original notebook-02 pipeline, which computed quantile caps, scalers, imputations, target
encodings and (in notebook 03) feature selection on the entire dataset before splitting.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


# ----------------------------------------------------------------------------- stateless steps
def add_time_features(df: pd.DataFrame, time_col: str | None = None, step_col: str | None = None,
                      peak_hours=((7, 10), (12, 15), (17, 21))) -> pd.DataFrame:
    """Deterministic calendar features. Parses the timestamp *before* any encoding
    (the original pipeline target-encoded the timestamp to a float first, so every parsed
    date collapsed to 1970-01-01 and all time features were constant)."""
    out = df.copy()
    if time_col is not None:
        ts = pd.to_datetime(out[time_col], errors="coerce")
        hour = ts.dt.hour
        out["time_epoch_s"] = (ts - pd.Timestamp("1970-01-01")).dt.total_seconds()
    elif step_col is not None:
        step = pd.to_numeric(out[step_col], errors="coerce")
        hour = step % 24
        out["day_of_sim"] = step // 24
        out["time_epoch_s"] = step * 3600.0
        ts = None
    else:
        return out
    out["hour_of_day"] = hour
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    peak = np.zeros(len(out), dtype=bool)
    for lo, hi in peak_hours:
        peak |= ((hour >= lo) & (hour < hi)).to_numpy()
    out["unusual_hour"] = (~peak).astype(int)
    if ts is not None:
        out["day_of_week"] = ts.dt.dayofweek
        out["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
    return out


# ----------------------------------------------------------------------------- fitted steps
class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip numeric columns at the ``upper_q`` (and optionally ``lower_q``) quantile of the TRAIN data."""

    def __init__(self, upper_q: float = 0.99, lower_q: float | None = None):
        self.upper_q = upper_q
        self.lower_q = lower_q

    def fit(self, X, y=None):
        A = np.asarray(X, dtype=float)
        self.upper_ = np.nanquantile(A, self.upper_q, axis=0)
        self.lower_ = None if self.lower_q is None else np.nanquantile(A, self.lower_q, axis=0)
        self.n_features_in_ = A.shape[1]
        return self

    def transform(self, X):
        A = np.asarray(X, dtype=float)
        A = np.minimum(A, self.upper_)
        if self.lower_ is not None:
            A = np.maximum(A, self.lower_)
        return A

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


class OOFTargetEncoder(BaseEstimator, TransformerMixin):
    """Smoothed target encoding that never lets a row see its own label.

    ``fit_transform`` (used on the training partition) returns *out-of-fold* encodings: each row
    is encoded with statistics from the other folds. ``transform`` (validation/test) uses
    statistics from the whole training partition; unseen categories get the global rate.
    ``enc(c) = (n_c * mean_c + m * global) / (n_c + m)``.
    """

    def __init__(self, smoothing: float = 20.0, n_splits: int = 5, random_state: int = 42):
        self.smoothing = smoothing
        self.n_splits = n_splits
        self.random_state = random_state

    # -- helpers
    @staticmethod
    def _stats(col: pd.Series, y: pd.Series):
        g = y.groupby(col, observed=True).agg(["sum", "count"])
        return g["sum"], g["count"]

    def _encode(self, col: pd.Series, sums, counts, prior: float) -> np.ndarray:
        s = col.map(sums)
        n = col.map(counts)
        return ((s.fillna(0.0) + self.smoothing * prior) / (n.fillna(0.0) + self.smoothing)).to_numpy(float)

    def fit(self, X, y):
        X = pd.DataFrame(X).reset_index(drop=True)
        y = pd.Series(np.asarray(y, dtype=float))
        self.columns_ = list(X.columns)
        self.prior_ = float(y.mean())
        self.maps_ = {c: self._stats(X[c], y) for c in self.columns_}
        return self

    def transform(self, X):
        X = pd.DataFrame(X).reset_index(drop=True)
        out = np.column_stack([self._encode(X[c], *self.maps_[c], self.prior_) for c in self.columns_])
        return out

    def fit_transform(self, X, y=None, **kw):
        X = pd.DataFrame(X).reset_index(drop=True)
        y = pd.Series(np.asarray(y, dtype=float))
        self.fit(X, y)
        out = np.zeros((len(X), len(self.columns_)))
        kf = KFold(self.n_splits, shuffle=True, random_state=self.random_state)
        for tr, va in kf.split(X):
            ytr = y.iloc[tr]
            prior = float(ytr.mean())
            for j, c in enumerate(self.columns_):
                sums, counts = self._stats(X[c].iloc[tr], ytr)
                out[va, j] = self._encode(X[c].iloc[va], sums, counts, prior)
        return out

    def get_feature_names_out(self, input_features=None):
        return np.asarray([f"{c}__te" for c in self.columns_], dtype=object)


# ----------------------------------------------------------------------------- assembly
def split_columns(df: pd.DataFrame, low_card_max: int = 10):
    """Partition columns of a *training* frame into numeric / low-cardinality / high-cardinality categorical."""
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
    cat = [c for c in df.columns if c not in num]
    low = [c for c in cat if df[c].nunique(dropna=True) <= low_card_max]
    high = [c for c in cat if c not in low]
    return num, low, high


def build_preprocessor(num_cols, low_cols, high_cols, *, winsor_q: float = 0.99,
                       te_smoothing: float = 20.0, random_state: int = 42) -> ColumnTransformer:
    """ColumnTransformer: median-impute -> winsorise -> robust-scale numerics; one-hot low-card
    categoricals; OOF target-encode high-card categoricals. All fit on TRAIN only."""
    transformers = []
    if num_cols:
        transformers.append(("num", Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("winsor", Winsorizer(upper_q=winsor_q)),
            ("scale", RobustScaler()),
        ]), list(num_cols)))
    if low_cols:
        transformers.append(("low", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), list(low_cols)))
    if high_cols:
        transformers.append(("high", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="__missing__")),
            ("te", OOFTargetEncoder(smoothing=te_smoothing, random_state=random_state)),
        ]), list(high_cols)))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def fit_transform_partitions(X: pd.DataFrame, y: pd.Series, tr_idx, va_idx, te_idx, *,
                             low_card_max: int = 10, **kw):
    """Fit on ``tr_idx`` only; return transformed train/val/test arrays and the fitted transformer."""
    Xtr, ytr = X.iloc[tr_idx], y.iloc[tr_idx]
    num, low, high = split_columns(Xtr, low_card_max)
    pre = build_preprocessor(num, low, high, **kw)
    A_tr = pre.fit_transform(Xtr, ytr)
    A_va = pre.transform(X.iloc[va_idx])
    A_te = pre.transform(X.iloc[te_idx])
    return A_tr, A_va, A_te, pre
