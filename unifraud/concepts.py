"""Semantic concept alignment (paper Sec. V-B, Eq. (concept), Prop. 1).

Each sector's schema is mapped to three shared fraud concepts

    amount_anomaly, counterparty_risk, structural_distance

by an *oriented z-score average*: for concept c with feature group G_c and orientations sigma_j,

    phi(x)_c = mean_{j in G_c} sigma_j * (x_j - mu_j) / s_j ,         (0 if G_c is empty)

where mu_j, s_j are TRAIN statistics. Properties (proved in the paper): invariant to positive
affine changes of units, zero training mean, monotone in a latent risk when all oriented
features are. The specs below are PROVISIONAL expert hypotheses: validate orientations with
``concept_validity`` on the training partition before trusting any transfer result.
"""
from __future__ import annotations

import re
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

CONCEPTS = ("amount_anomaly", "counterparty_risk", "structural_distance")

# concept -> list of (regex matched against cleaned column names, orientation sigma in {-1,+1})
Spec = Mapping[str, Sequence[tuple[str, int]]]

DEFAULT_SPECS: dict[str, Spec] = {
    "CC": {
        "amount_anomaly": [(r"^amt$", 1), (r"^amount_(zscore|deviation)", 1), (r"^is_large_transaction$", 1)],
        "counterparty_risk": [(r"^(merchant|category)(__te)?$", 1)],
        "structural_distance": [(r"geo_distance", 1), (r"^unusual_hour$", 1)],
    },
    "MM": {
        "amount_anomaly": [(r"^amount$", 1), (r"^amount_(zscore|deviation)", 1), (r"^is_large_transaction$", 1)],
        "counterparty_risk": [(r"^balance_zeroed_dest$", 1), (r"^is_agent_proxy$", 1)],
        "structural_distance": [(r"^balance_error_(orig|dest)$", 1), (r"^unusual_hour$", 1)],
    },
    # HI: the natural amount-anomaly features are rule inputs (circular) -> deliberately EMPTY.
    "HI": {
        "amount_anomaly": [],
        "counterparty_risk": [(r"^drg_claim_volume$", 1)],
        "structural_distance": [(r"^length_of_stay$", 1), (r"^attendance_(month|dow)$", 1)],
    },
}


def signed_auc(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> float:
    """Rank AUC of ``x`` for label ``y`` (0.5 = none; >0.5 means larger x -> more fraud)."""
    x = pd.Series(np.asarray(x, dtype=float))
    y = pd.Series(np.asarray(y)).astype(int)
    m = x.notna() & y.notna()
    x, y = x[m], y[m]
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0 or x.nunique() < 2:
        return 0.5
    r = x.rank(method="average")
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


class ConceptMap:
    """Fit on TRAIN only; ``transform`` returns a DataFrame with the three concept columns."""

    def __init__(self, spec: Spec, name: str = ""):
        self.spec = spec
        self.name = name

    def fit(self, X: pd.DataFrame) -> "ConceptMap":
        self.groups_: dict[str, list[tuple[str, int]]] = {}
        self.mu_: dict[str, float] = {}
        self.sd_: dict[str, float] = {}
        for c in CONCEPTS:
            chosen: list[tuple[str, int]] = []
            seen: set[str] = set()
            for pat, sg in self.spec.get(c, []):
                rx = re.compile(pat, re.I)
                for col in X.columns:
                    if col in seen or not rx.search(str(col)):
                        continue
                    if not pd.api.types.is_numeric_dtype(X[col]):
                        continue
                    seen.add(col)
                    chosen.append((col, int(sg)))
                    v = pd.to_numeric(X[col], errors="coerce")
                    self.mu_[col] = float(v.mean())
                    sd = float(v.std(ddof=0))
                    self.sd_[col] = sd if sd > 0 else 1.0
            self.groups_[c] = chosen
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for c in CONCEPTS:
            g = self.groups_[c]
            if not g:
                out[c] = np.zeros(len(X))
                continue
            zs = [sg * (pd.to_numeric(X[col], errors="coerce").to_numpy(float) - self.mu_[col]) / self.sd_[col]
                  for col, sg in g]
            out[c] = np.nanmean(np.vstack(zs), axis=0)
        Z = pd.DataFrame(out, index=X.index)
        return Z.fillna(0.0)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)

    def report(self) -> dict:
        """Which columns fed each concept (store in the run manifest)."""
        return {c: [col for col, _ in g] for c, g in self.groups_.items()}


def concept_validity(data: Mapping[str, tuple[pd.DataFrame, pd.Series]]) -> pd.DataFrame:
    """Per sector and concept: signed AUC of the concept coordinate, plus a cross-sector
    sign-consistency flag (all non-empty sectors on the same side of 0.5).

    ``data[sector] = (Z_train, y_train)`` with Z from :class:`ConceptMap`. Empty concepts
    (all-zero columns) are reported as NaN and excluded from the consistency check.
    """
    rows = []
    for sector, (Z, y) in data.items():
        for c in CONCEPTS:
            col = Z[c]
            empty = (col.abs().max() == 0) if len(col) else True
            rows.append({"sector": sector, "concept": c,
                         "auc": np.nan if empty else signed_auc(col, y)})
    df = pd.DataFrame(rows)
    cons = {}
    for c, g in df.groupby("concept"):
        a = g["auc"].dropna()
        cons[c] = bool(len(a) >= 2 and ((a > 0.5).all() or (a < 0.5).all()))
    df["sign_consistent"] = df["concept"].map(cons)
    return df
