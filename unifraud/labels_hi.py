"""Health-insurance rule labels (G-DRG tariff compliance) with correct missing-data handling.

Fixes relative to notebook 01:
  * a *missing* drug (Medical) charge is no longer treated as an "impossible amount"
    (the old ``fillna(0) <= 0`` rule fired on 632 claims, ~629 of which were simply missing);
  * the total-consistency rule is evaluated only when all of S, M, T are observed;
  * claims whose G-DRG code has no tariff cannot be assessed against rules 1-3 and are
    labelled ``NaN`` by default instead of silently "legitimate";
  * rules and data-quality flags are returned separately so results can be reported
    with and without each.

Rules (Sowah et al., 2019; see paper Eq. (rules)), with S = Service, M = Medical (drug),
T = Total, A = approved tariff:
    r1: S > A          r2: M > 0.5 A        r3: T > A
    r4: |T - (S + M)| > tol   (only if S, M, T all observed)
    r5 (data-quality, NOT fraud): any *observed* bill <= 0
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RULE_COLUMNS = ["r1_service_over_tariff", "r2_drug_over_half_tariff",
                "r3_total_over_tariff", "r4_total_inconsistent"]
DQ_COLUMNS = ["dq_missing_service", "dq_missing_medical", "dq_missing_total",
              "dq_nonpositive_observed", "dq_no_tariff_match"]

#: Columns that define (or are algebraically derived from) the rule label. They must never
#: be used as model features for HI; see :mod:`unifraud.leakage`.
RULE_INPUT_COLUMNS = ["Service", "Medical", "Total", "Approved_Cost", "GDRG_Code"]


def load_tariff(gd: pd.DataFrame) -> pd.DataFrame:
    gd = gd.copy()
    gd.columns = gd.columns.str.strip()
    gd = gd.rename(columns={"Code": "GDRG_Code", "Cost": "Approved_Cost"})
    gd["GDRG_Code"] = gd["GDRG_Code"].astype(str).str.strip().str.upper()
    gd["Approved_Cost"] = pd.to_numeric(gd["Approved_Cost"], errors="coerce")
    return gd.drop_duplicates("GDRG_Code")


def merge_tariff(claims: pd.DataFrame, tariff: pd.DataFrame, drg_col: str = "G-DRG") -> pd.DataFrame:
    out = claims.copy()
    out["GDRG_Code"] = out[drg_col].astype(str).str.strip().str.upper()
    return out.merge(tariff[["GDRG_Code", "Approved_Cost"]], on="GDRG_Code", how="left")


def compute_rule_flags(df: pd.DataFrame, *, service="Service", medical="Medical", total="Total",
                       tariff="Approved_Cost", tol: float = 1.0) -> pd.DataFrame:
    """Return a frame of rule flags (0/1) and data-quality flags. Index matches ``df``.

    Rules that need the tariff are 0 where the tariff is missing (they are *not evaluable*;
    use :func:`assign_labels` to turn that into NaN labels).
    """
    S = pd.to_numeric(df[service], errors="coerce")
    M = pd.to_numeric(df[medical], errors="coerce")
    T = pd.to_numeric(df[total], errors="coerce")
    A = pd.to_numeric(df[tariff], errors="coerce")
    has_A = A.notna()

    f = pd.DataFrame(index=df.index)
    f["r1_service_over_tariff"] = ((S > A) & has_A).astype(int)
    f["r2_drug_over_half_tariff"] = ((M > 0.5 * A) & has_A).astype(int)
    f["r3_total_over_tariff"] = ((T > A) & has_A).astype(int)
    all_obs = S.notna() & M.notna() & T.notna()
    f["r4_total_inconsistent"] = (all_obs & ((T - (S + M)).abs() > tol)).astype(int)

    f["dq_missing_service"] = S.isna().astype(int)
    f["dq_missing_medical"] = M.isna().astype(int)
    f["dq_missing_total"] = T.isna().astype(int)
    nonpos = (S <= 0) | (M <= 0) | (T <= 0)          # NaN comparisons are False
    f["dq_nonpositive_observed"] = nonpos.fillna(False).astype(int)
    f["dq_no_tariff_match"] = (~has_A).astype(int)
    return f


def assign_labels(flags: pd.DataFrame, *, include_unmatched: bool = False,
                  include_nonpositive: bool = False) -> pd.Series:
    """Combine rule flags into ``fraud_label`` (float with NaN for 'cannot be assessed').

    * default: fraud = any of r1..r4; claims with no tariff match are NaN (excluded), except
      that if r4 fires the claim is labelled 1 because r4 does not need the tariff.
    * ``include_unmatched=True``: unmatched claims are labelled by r4 only (0 if it is quiet).
    * ``include_nonpositive=True``: observed non-positive bills also count as fraud
      (reproduces the *intent* of the old r5 without the missing-data artefact).
    """
    fraud = flags[RULE_COLUMNS].sum(axis=1) > 0
    if include_nonpositive:
        fraud = fraud | (flags["dq_nonpositive_observed"] > 0)
    label = fraud.astype(float)
    if not include_unmatched:
        unassessable = (flags["dq_no_tariff_match"] > 0) & (~fraud)
        label[unassessable] = np.nan
    return label.rename("fraud_label")


def audit_report(flags: pd.DataFrame, label: pd.Series) -> dict:
    """Counts needed for the paper's label-audit table."""
    return {
        "n_claims": int(len(flags)),
        **{c: int(flags[c].sum()) for c in RULE_COLUMNS + DQ_COLUMNS},
        "n_labelled": int(label.notna().sum()),
        "n_unassessable": int(label.isna().sum()),
        "fraud_rate_labelled": float(label.dropna().mean()) if label.notna().any() else float("nan"),
    }
