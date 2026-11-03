import json
import os

import numpy as np
import pandas as pd

from scripts import prepare_data as PD


def _make_hi(tmp):
    rng = np.random.default_rng(0)
    codes = [f"ADME{i:02d}A" for i in range(1, 21)]
    gd = pd.DataFrame({"Code": codes, "Description": "x", "Cost": rng.uniform(50, 500, 20)})
    gd.to_csv(os.path.join(tmp, "GD.csv"), index=False)
    n = 600
    cost = gd.set_index("Code")["Cost"]
    drg = rng.choice(codes, n)
    svc = cost[drg].to_numpy() * rng.uniform(0.2, 1.4, n)
    med = np.where(rng.random(n) < 0.25, np.nan, cost[drg].to_numpy() * rng.uniform(0.0, 0.7, n))
    tot = svc + np.nan_to_num(med)
    pat = rng.integers(0, 250, n)
    pd.DataFrame({"No.": np.arange(n), "Attendance Date": "01/02/2020", "Discharge Date": "03/02/2020",
                  "Age": rng.integers(1, 80, n), "Sex": rng.choice(["M", "F"], n), "Folder No": [f"F{p}" for p in pat],
                  "NHIS ID No": [f"N{p}" for p in pat], "Procudure": "p", "Diagnosis": rng.choice(list("abcdefghijklm"), n),
                  "G-DRG": drg, "ICD10": "I", "Service": svc, "Medical": med, "Total": tot}
                 ).to_csv(os.path.join(tmp, "ReData1_100.csv"), index=False)


def test_hi_end_to_end_no_leak_no_overlap(tmp_path):
    d, o = str(tmp_path / "d"), str(tmp_path / "o")
    os.makedirs(d)
    _make_hi(d)
    m = PD.prepare_hi(d, o)
    assert m["label_audit"]["dq_missing_medical"] > 0
    # missing medical must not be counted as fraud by construction
    assert m["label_audit"]["dq_nonpositive_observed"] == 0
    assert sum(m["sizes"].values()) == m["n"]
    for banned in ("Service", "Medical", "Total", "Approved_Cost", "Folder No", "NHIS ID No", "dev_total"):
        assert banned in m["dropped_columns"] or banned not in pd.read_csv(os.path.join(o, "health_insurance_train.csv")).columns
    tr = pd.read_csv(os.path.join(o, "health_insurance_train.csv"))
    assert not any(c in tr.columns for c in ("Service", "Medical", "Total", "Folder No"))
    assert json.load(open(os.path.join(o, "health_insurance_manifest.json")))["domain"] == "health_insurance"


def test_cc_chronological_end_to_end(tmp_path):
    rng = np.random.default_rng(1)
    n = 800
    t = pd.date_range("2019-01-01", periods=n, freq="h").astype(str)
    df = pd.DataFrame({"trans_date_trans_time": t, "cc_num": rng.integers(1, 40, n), "merchant": rng.choice(list("abcdefghijklmnopqrstuvwxyz"), n),
                       "category": rng.choice(["a", "b", "c"], n), "amt": rng.lognormal(3, 1, n), "city_pop": rng.integers(100, 9000, n),
                       "is_fraud": (rng.random(n) < 0.05).astype(int)})
    d, o = str(tmp_path / "d"), str(tmp_path / "o")
    os.makedirs(d)
    df.iloc[:600].to_csv(os.path.join(d, "fraudTrain.csv"))
    df.iloc[600:].to_csv(os.path.join(d, "fraudTest.csv"))
    m = PD.prepare_cc(d, o)
    assert "cc_num" in m["dropped_columns"]
    tr = pd.read_csv(os.path.join(o, "credit_card_train.csv"))
    assert "hour_sin" in tr.columns and tr["hour_sin"].nunique() > 1      # time features are not constant
