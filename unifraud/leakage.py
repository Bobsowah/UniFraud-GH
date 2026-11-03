"""Leakage registry and automatic checks.

Three leakage channels were found in the original pipeline and are guarded here:
  1. features algebraically derived from the inputs of the rule-generated HI label,
  2. quasi-identifiers (patient folder / insurance id / account ids) that let the same entity
     appear in train and test and that target-encoding turns into near-labels,
  3. any feature that predicts the label (near) perfectly on its own - a generic tripwire.
"""
from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd

# Substrings (case-insensitive) of HI columns that are rule inputs or functions of them.
HI_RULE_DERIVED_PATTERNS = [
    r"^service$", r"^medical$", r"^total$", r"approved_cost", r"tariff",
    r"^dev_", r"overbill", r"billing_variab", r"billing_inconsistency", r"impossible_amount",
    r"max_deviation", r"multi_field", r"is_overbilled", r"service_to_total", r"is_service_dominant",
    r"drg_fraud_rate",                       # label-derived per-DRG rate
    r"^amount_deviation", r"^amount_zscore", r"^is_large_trans", r"^is_micro_trans", r"^is_high_value",
    r"^r[1-4]_", r"^dq_",
]
# Identifier-like columns (all sectors). Case-insensitive exact/regex matches.
IDENTIFIER_PATTERNS = [
    r"^folder", r"nhis.*id", r"^no\.?$", r"^cc_num$", r"^trans_num$", r"^first$", r"^last$",
    r"^street$", r"^nameorig$", r"^namedest$", r"^initiator$", r"^recipient$", r"^source_file$",
]


def _match(columns: Iterable[str], patterns: list[str]) -> list[str]:
    rx = [re.compile(p, re.I) for p in patterns]
    return [c for c in columns if any(r.search(str(c).strip()) for r in rx)]


def forbidden_columns(columns: Iterable[str], domain: str) -> list[str]:
    """Columns that must not be used as features for ``domain`` ('CC', 'MM' or 'HI')."""
    cols = list(columns)
    bad = set(_match(cols, IDENTIFIER_PATTERNS))
    if domain.upper() in ("HI", "HEALTH INSURANCE"):
        bad |= set(_match(cols, HI_RULE_DERIVED_PATTERNS))
    return [c for c in cols if c in bad]


def drop_forbidden(df: pd.DataFrame, domain: str, *, keep: Iterable[str] = ()) -> pd.DataFrame:
    keep = set(keep)
    drop = [c for c in forbidden_columns(df.columns, domain) if c not in keep]
    return df.drop(columns=drop)


def single_feature_auc(x: pd.Series, y: pd.Series) -> float:
    """Rank-based AUC of one numeric feature (sign-agnostic, max(AUC, 1-AUC))."""
    m = x.notna() & y.notna()
    x, y = x[m].astype(float), y[m].astype(int)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0 or x.nunique() < 2:
        return 0.5
    r = x.rank(method="average")
    auc = (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return float(max(auc, 1 - auc))


def audit_features(X: pd.DataFrame, y: pd.Series, *, domain: str, auc_threshold: float = 0.98) -> pd.DataFrame:
    """Report, per feature, whether it is forbidden or near-perfectly predictive on its own."""
    rows = []
    forb = set(forbidden_columns(X.columns, domain))
    for c in X.columns:
        auc = single_feature_auc(X[c], y) if pd.api.types.is_numeric_dtype(X[c]) else np.nan
        rows.append({"feature": c, "forbidden": c in forb, "single_feature_auc": auc,
                     "suspicious_auc": bool(auc >= auc_threshold) if auc == auc else False})
    return pd.DataFrame(rows)


def assert_no_leakage(X: pd.DataFrame, y: pd.Series, *, domain: str, auc_threshold: float = 0.98) -> None:
    """Raise ``ValueError`` listing every offending feature. Call before any model is fit."""
    rep = audit_features(X, y, domain=domain, auc_threshold=auc_threshold)
    bad = rep[rep.forbidden | rep.suspicious_auc]
    if len(bad):
        raise ValueError("Leakage check failed for %s:\n%s" % (domain, bad.to_string(index=False)))
