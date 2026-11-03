#!/usr/bin/env python
"""Leakage-safe data preparation (replaces notebooks 01-03's label/preprocessing logic).

Usage:  python scripts/prepare_data.py --data data --out outputs/clean [--domains HI CC MM]

For each domain this: builds labels (HI: fixed rules), removes identifier / rule-derived columns,
creates time features, splits (HI grouped by patient, CC/MM chronological), fits the preprocessor on
TRAIN only, runs the leakage tripwire, and writes  {domain}_{train,val,test}.csv  plus
{domain}_manifest.json (counts, fraud rates, dropped columns, label audit).

NOTE: written against the column names visible in the notebooks. It has been exercised on synthetic
data only (tests/test_pipeline.py, tests/test_prepare_smoke.py) - run it on the real files and inspect
the manifest before trusting any downstream number.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from unifraud import labels_hi, leakage, preprocess, splits  # noqa: E402

TARGET = "fraud_label"


def _finish(name, X, y, tr, va, te, out, manifest, domain_key, *, drop_log):
    """Fit preprocessing on TRAIN only and save the three partitions."""
    leakage.assert_no_leakage(X.iloc[tr].select_dtypes("number"), y.iloc[tr], domain=domain_key)
    A_tr, A_va, A_te, pre = preprocess.fit_transform_partitions(X, y, tr, va, te)
    names = list(pre.get_feature_names_out())
    os.makedirs(out, exist_ok=True)
    for tag, A, idx in (("train", A_tr, tr), ("val", A_va, va), ("test", A_te, te)):
        df = pd.DataFrame(A, columns=names)
        df[TARGET] = y.iloc[idx].to_numpy()
        df.to_csv(os.path.join(out, f"{name}_{tag}.csv"), index=False)
    manifest.update({
        "domain": name, "n": int(len(y)), "n_features_out": len(names),
        "sizes": {"train": len(tr), "val": len(va), "test": len(te)},
        "fraud_rate": {k: float(y.iloc[i].mean()) for k, i in (("train", tr), ("val", va), ("test", te))},
        "dropped_columns": drop_log,
    })
    with open(os.path.join(out, f"{name}_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    return manifest


def prepare_hi(data_dir, out_dir, *, group_col="NHIS ID No", include_unmatched=False, include_nonpositive=False):
    gd = labels_hi.load_tariff(pd.read_csv(os.path.join(data_dir, "GD.csv")))
    parts = []
    for fn in sorted(os.listdir(data_dir)):
        if fn.startswith("ReData") and fn.endswith(".csv"):
            parts.append(pd.read_csv(os.path.join(data_dir, fn)))
    claims = pd.concat(parts, ignore_index=True)
    claims.columns = claims.columns.str.strip()
    df = labels_hi.merge_tariff(claims, gd)
    flags = labels_hi.compute_rule_flags(df)
    y = labels_hi.assign_labels(flags, include_unmatched=include_unmatched, include_nonpositive=include_nonpositive)
    audit = labels_hi.audit_report(flags, y)

    keep = y.notna().to_numpy()
    df, y = df[keep].reset_index(drop=True), y[keep].astype(int).reset_index(drop=True)
    groups = df[group_col].astype(str).to_numpy() if group_col in df.columns else np.arange(len(df))

    # derive date features first (pure), then drop everything the label (or identity) depends on
    for c in [c for c in df.columns if "date" in c.lower()]:
        df[c] = pd.to_datetime(df[c], dayfirst=True, errors="coerce")
    if {"Attendance Date", "Discharge Date"} <= set(df.columns):
        df["length_of_stay"] = (df["Discharge Date"] - df["Attendance Date"]).dt.days.clip(lower=0)
        df["attendance_month"] = df["Attendance Date"].dt.month
        df["attendance_dow"] = df["Attendance Date"].dt.dayofweek
    df = df.drop(columns=[c for c in df.columns if "date" in c.lower()])
    forbidden = leakage.forbidden_columns(df.columns, "HI")
    X = df.drop(columns=forbidden)
    if "Age" in X:
        X["Age"] = pd.to_numeric(X["Age"], errors="coerce")
    tr, va, te = splits.grouped_stratified_split(y.to_numpy(), groups)
    splits.assert_disjoint(tr, va, te)
    return _finish("health_insurance", X, y, tr, va, te, out_dir, {"label_audit": audit}, "HI", drop_log=forbidden)


def _prepare_simple(name, key, df, label_col, out_dir, *, time_col=None, step_col=None):
    df = df.rename(columns={label_col: TARGET}).dropna(subset=[TARGET]).reset_index(drop=True)
    y = df.pop(TARGET).astype(int)
    df = preprocess.add_time_features(df, time_col=time_col, step_col=step_col)
    order_key = df["time_epoch_s"] if "time_epoch_s" in df else pd.Series(np.arange(len(df)))
    tr, va, te = splits.chronological_split(order_key)
    forbidden = leakage.forbidden_columns(df.columns, key)
    drop = list(dict.fromkeys(forbidden + [c for c in (time_col,) if c]))
    X = df.drop(columns=drop)
    # raw dates / unused unix time never enter the model; time_epoch_s is dropped from features
    X = X.drop(columns=[c for c in ("time_epoch_s", "unix_time", "dob") if c in X.columns])
    return _finish(name, X, y, tr, va, te, out_dir, {"split": "chronological"}, key, drop_log=drop)


def prepare_cc(data_dir, out_dir):
    parts = [pd.read_csv(os.path.join(data_dir, f), index_col=0) for f in ("fraudTrain.csv", "fraudTest.csv")]
    return _prepare_simple("credit_card", "CC", pd.concat(parts, ignore_index=True), "is_fraud", out_dir,
                           time_col="trans_date_trans_time")


def prepare_mm(data_dir, out_dir):
    path = next(os.path.join(data_dir, f) for f in ("MoMTSim_transactions.csv", "paysim.csv")
                if os.path.exists(os.path.join(data_dir, f)))
    df = pd.read_csv(path)
    label = "isFraud" if "isFraud" in df.columns else "is_fraud"
    return _prepare_simple("mobile_money", "MM", df, label, out_dir, step_col="step" if "step" in df.columns else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="outputs/clean")
    ap.add_argument("--domains", nargs="+", default=["HI", "CC", "MM"])
    a = ap.parse_args()
    fn = {"HI": prepare_hi, "CC": prepare_cc, "MM": prepare_mm}
    for d in a.domains:
        m = fn[d](a.data, a.out)
        print(d, json.dumps({k: m[k] for k in ("sizes", "fraud_rate", "n_features_out")}))


if __name__ == "__main__":
    main()
