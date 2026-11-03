#!/usr/bin/env python
"""Concept-space transfer baselines for all six directed sector pairs.

Usage:  python scripts/run_transfer_baselines.py --clean outputs/clean --out outputs/transfer \
            [--budgets 0 10 50 100 500] [--seeds 20] [--model logreg|hgb]

Inputs : outputs/clean/{credit_card,mobile_money,health_insurance}_{train,val,test}.csv  (prepare_data.py)
Outputs: results.csv (per pair/method/budget/seed), gains.csv (transfer gain vs target-only),
         concept_validity.csv, manifest.json (which columns fed each concept).

Source examples are capped (stratified) for speed. Everything is fit on TRAIN partitions; the target
TEST set is only used for the final metrics. NOT yet run on real data - inspect concept_validity.csv
and the manifest's concept membership before trusting any number.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from unifraud import concepts as C  # noqa: E402
from unifraud import transfer as T  # noqa: E402

SECTORS = {"CC": "credit_card", "MM": "mobile_money", "HI": "health_insurance"}
TARGET = "fraud_label"


def _load(clean_dir, key):
    name = SECTORS[key]
    return {part: pd.read_csv(os.path.join(clean_dir, f"{name}_{part}.csv")) for part in ("train", "val", "test")}


def _cap(Z, y, n, seed=0):
    if len(y) <= n:
        return Z, y
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    npos = min(len(pos), max(1, int(n * y.mean())))
    idx = np.concatenate([rng.choice(pos, npos, replace=False), rng.choice(neg, min(len(neg), n - npos), replace=False)])
    return Z.iloc[idx].reset_index(drop=True), y[idx]


def run_all(clean_dir, out_dir, budgets=(0, 10, 50, 100, 500), seeds=20, kind="logreg",
            specs=None, max_source=200_000, c_fp=1.0, c_fn=10.0):
    specs = specs or C.DEFAULT_SPECS
    os.makedirs(out_dir, exist_ok=True)
    data, Z, y, manifest = {}, {}, {}, {"concept_columns": {}, "model": kind, "seeds": seeds, "budgets": list(budgets)}
    for k in SECTORS:
        d = _load(clean_dir, k)
        cm = C.ConceptMap(specs[k], k).fit(d["train"].drop(columns=[TARGET]))
        Z[k] = {p: cm.transform(d[p].drop(columns=[TARGET])) for p in d}
        y[k] = {p: d[p][TARGET].to_numpy().astype(int) for p in d}
        manifest["concept_columns"][k] = cm.report()
    validity = C.concept_validity({k: (Z[k]["train"], y[k]["train"]) for k in SECTORS})
    validity.to_csv(os.path.join(out_dir, "concept_validity.csv"), index=False)

    rows = []
    for s_key in SECTORS:
        for t_key in SECTORS:
            if s_key == t_key:
                continue
            Zs, ys = _cap(Z[s_key]["train"], y[s_key]["train"], max_source)
            for b in budgets:
                for seed in range(seeds if b > 0 else 1):
                    for r in T.run_pair(Zs, ys, Z[t_key]["train"], y[t_key]["train"], Z[t_key]["test"],
                                        y[t_key]["test"], b=b, seed=seed, kind=kind, c_fp=c_fp, c_fn=c_fn):
                        rows.append({"source": s_key, "target": t_key, **r.__dict__})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(out_dir, "results.csv"), index=False)
    gains = []
    for (s_key, t_key), g in res.groupby(["source", "target"]):
        gg = T.transfer_gain(g.drop(columns=["source", "target"]))
        gg.insert(0, "target", t_key)
        gg.insert(0, "source", s_key)
        gains.append(gg)
    if gains:
        pd.concat(gains).to_csv(os.path.join(out_dir, "gains.csv"), index=False)
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return res, validity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", default="outputs/clean")
    ap.add_argument("--out", default="outputs/transfer")
    ap.add_argument("--budgets", nargs="+", type=int, default=[0, 10, 50, 100, 500])
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--model", default="logreg", choices=["logreg", "hgb"])
    a = ap.parse_args()
    res, val = run_all(a.clean, a.out, a.budgets, a.seeds, a.model)
    print(val.to_string(index=False))
    print(res.groupby(["source", "target", "method", "b"])[["ap_skill", "auc", "mcc"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
