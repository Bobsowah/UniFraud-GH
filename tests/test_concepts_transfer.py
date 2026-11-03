import json
import os

import numpy as np
import pandas as pd
import pytest

from tests import synth
from unifraud import concepts as C
from unifraud import metrics as M
from unifraud import transfer as T


def _Z(df, key):
    cm = C.ConceptMap(C.DEFAULT_SPECS[key], key).fit(df.drop(columns=["fraud_label"]))
    return cm, cm.transform(df.drop(columns=["fraud_label"]))


# ------------------------------------------------------------------ concept map properties (Prop. 1)
def test_concept_map_is_unit_invariant():
    df = synth.make_sector("CC", 1500, 0.1, 0)
    _, Z1 = _Z(df, "CC")
    df2 = df.copy()
    df2["amt"] = df2["amt"] * 15.8 + 3.0                              # USD -> GHS style affine change
    df2["geo_distance"] = df2["geo_distance"] * 1.609
    _, Z2 = _Z(df2, "CC")
    np.testing.assert_allclose(Z1.to_numpy(), Z2.to_numpy(), atol=1e-8)


def test_orientation_sign_flips_concept():
    df = synth.make_sector("MM", 1500, 0.3, 1)
    spec = {"amount_anomaly": [(r"^amount$", 1)], "counterparty_risk": [], "structural_distance": []}
    flipped = {"amount_anomaly": [(r"^amount$", -1)], "counterparty_risk": [], "structural_distance": []}
    Za = C.ConceptMap(spec).fit_transform(df.drop(columns=["fraud_label"]))
    Zb = C.ConceptMap(flipped).fit_transform(df.drop(columns=["fraud_label"]))
    np.testing.assert_allclose(Za["amount_anomaly"], -Zb["amount_anomaly"])


def test_train_mean_zero_and_empty_hi_amount_is_zero():
    df = synth.make_sector("HI", 1500, 0.4, 2)
    cm, Z = _Z(df, "HI")
    assert (Z["amount_anomaly"] == 0).all()                            # HI amount deliberately empty
    assert abs(Z["structural_distance"].mean()) < 0.2                  # month feature has a different mean; loose
    assert cm.report()["amount_anomaly"] == []


def test_test_statistics_do_not_leak_into_concept_map():
    df = synth.make_sector("CC", 1200, 0.1, 3)
    tr, te = df.iloc[:800], df.iloc[800:].copy()
    cm = C.ConceptMap(C.DEFAULT_SPECS["CC"]).fit(tr.drop(columns=["fraud_label"]))
    z1 = cm.transform(te.drop(columns=["fraud_label"]))
    te["amt"] = te["amt"] * 1e6
    cm2 = C.ConceptMap(C.DEFAULT_SPECS["CC"]).fit(tr.drop(columns=["fraud_label"]))
    assert cm.mu_ == cm2.mu_ and cm.sd_ == cm2.sd_                      # statistics are train-only
    assert not np.allclose(z1["amount_anomaly"], cm.transform(te.drop(columns=["fraud_label"]))["amount_anomaly"])


def test_concept_validity_is_sign_consistent_on_shared_latent_risk():
    data = synth.make_all(seed=5)
    parts = {}
    for k, df in data.items():
        _, Z = _Z(df, k)
        parts[k] = (Z, df["fraud_label"])
    v = C.concept_validity(parts)
    assert v[v.concept == "structural_distance"].sign_consistent.all()
    assert np.isnan(v[(v.sector == "HI") & (v.concept == "amount_anomaly")].auc.iloc[0])
    assert (v.dropna(subset=["auc"]).auc > 0.55).all()


# ------------------------------------------------------------------ prior correction
def test_saerens_scales_odds():
    p = np.array([0.1, 0.5, 0.9])
    q = T.saerens_correct(p, 0.02, 0.5)
    ratio = (0.5 / 0.5) / (0.02 / 0.98)
    np.testing.assert_allclose(q / (1 - q), p / (1 - p) * ratio, rtol=1e-6)
    np.testing.assert_allclose(T.saerens_correct(p, 0.3, 0.3), p, atol=1e-9)


def test_bbse_recovers_shifted_prior():
    rng = np.random.default_rng(0)

    def draw(n, pi):
        y = (rng.random(n) < pi).astype(int)
        x = rng.normal(loc=1.5 * y, scale=1.0)                         # same p(x|y) in both domains
        return x.reshape(-1, 1), y
    Xs, ys = draw(20000, 0.2)
    Xt, yt = draw(20000, 0.5)
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression().fit(Xs[:10000], ys[:10000])
    pi_hat = T.bbse_prior(ys[10000:], clf.predict(Xs[10000:]), clf.predict(Xt))
    assert abs(pi_hat - 0.5) < 0.04


# ------------------------------------------------------------------ metrics
def test_ap_skill_zero_for_random_and_one_for_perfect():
    rng = np.random.default_rng(1)
    y = (rng.random(20000) < 0.02).astype(int)
    assert abs(M.ap_skill(y, rng.random(20000))) < 0.03
    assert M.ap_skill(y, y.astype(float) + 0.001 * rng.random(20000)) > 0.99


def test_corrected_t_uses_nadeau_bengio_factor():
    d = np.array([0.02, 0.01, 0.03, 0.015, 0.025] * 4)               # K = 20
    t, p = M.corrected_resampled_t(d, n_train=70, n_test=15)
    naive = d.mean() / np.sqrt(d.var(ddof=1) / len(d))
    factor = np.sqrt((1 / 20) / (1 / 20 + 15 / 70))
    assert t == pytest.approx(naive * factor)
    assert 0 <= p <= 1 and p > 0                                       # more conservative than the naive test


def test_holm_and_cost_threshold():
    assert M.holm([0.001, 0.04, 0.2], 0.05).tolist() == [True, False, False]
    assert M.bayes_threshold(1, 9) == pytest.approx(0.1)
    rng = np.random.default_rng(2)
    y = (rng.random(5000) < 0.3).astype(int)
    p = np.clip(0.5 * y + 0.25 + 0.2 * rng.normal(size=5000), 0, 1)
    t = M.cost_threshold(y, p, 1.0, 5.0)
    assert t < 0.5                                                      # false negatives cost more -> lower threshold


# ------------------------------------------------------------------ transfer end to end (synthetic)
def test_concept_map_is_stable_while_positional_pairing_depends_on_column_order():
    """Negative control A1. Positional pairing of raw columns is not necessarily *worse* on average
    (all informative features here increase with risk), but its result depends on an arbitrary column
    order; the concept map does not."""
    data = synth.make_all(seed=7, n=4000)
    from sklearn.linear_model import LogisticRegression
    ys, yt = data["MM"].fraud_label.to_numpy(), data["CC"].fraud_label.to_numpy()

    def concept_auc(cc_df):
        Zs = C.ConceptMap(C.DEFAULT_SPECS["MM"]).fit_transform(data["MM"].drop(columns="fraud_label"))
        Zt = C.ConceptMap(C.DEFAULT_SPECS["CC"]).fit_transform(cc_df.drop(columns="fraud_label"))
        return M.auc(yt, LogisticRegression(max_iter=500).fit(Zs, ys).predict_proba(Zt)[:, 1])

    base = concept_auc(data["CC"])
    assert base > 0.85
    cols = [c for c in data["CC"].columns if c != "fraud_label"]
    rng = np.random.default_rng(0)
    shuffled = data["CC"][list(rng.permutation(cols)) + ["fraud_label"]]
    assert concept_auc(shuffled) == pytest.approx(base, abs=1e-9)       # invariant to column order

    Xs = data["MM"].drop(columns="fraud_label").to_numpy()[:, :5]
    Xs = (Xs - Xs.mean(0)) / Xs.std(0)
    m = LogisticRegression(max_iter=500).fit(Xs, ys)
    Xt = data["CC"].drop(columns="fraud_label").to_numpy()[:, :5]
    Xt = (Xt - Xt.mean(0)) / Xt.std(0)
    aucs = [M.auc(yt, m.predict_proba(Xt[:, rng.permutation(5)])[:, 1]) for _ in range(30)]
    assert max(aucs) - min(aucs) > 0.15                                  # order-dependent, unstable
    assert base > np.mean(aucs)                                          # and worse on average here


def test_run_pair_methods_and_gain():
    data = synth.make_all(seed=11, n=4000)
    Zs, ys = _Z(data["MM"], "MM")[1], data["MM"].fraud_label.to_numpy()
    cmt = C.ConceptMap(C.DEFAULT_SPECS["HI"]).fit(data["HI"].drop(columns="fraud_label"))
    Zt = cmt.transform(data["HI"].drop(columns="fraud_label"))
    yt = data["HI"].fraud_label.to_numpy()
    tr, te = slice(0, 2500), slice(2500, 4000)
    rows = []
    for seed in range(3):
        for r in T.run_pair(Zs, ys, Zt.iloc[tr], yt[tr], Zt.iloc[te], yt[te], b=50, seed=seed):
            rows.append(r.__dict__)
    df = pd.DataFrame(rows)
    assert set(df.method) == {"source_only", "target_only", "pooled"}
    g = T.transfer_gain(df)
    assert {"gain", "ref"} <= set(g.columns) and len(g) == 6           # 2 methods x 3 seeds
    assert df[df.method == "source_only"].pi_hat.notna().all()


def test_runner_smoke(tmp_path):
    from scripts import run_transfer_baselines as R
    data = synth.make_all(seed=3, n=1500)
    clean = tmp_path / "clean"
    clean.mkdir()
    names = {"CC": "credit_card", "MM": "mobile_money", "HI": "health_insurance"}
    for k, df in data.items():
        n = len(df)
        parts = {"train": df.iloc[: int(.7 * n)], "val": df.iloc[int(.7 * n): int(.85 * n)], "test": df.iloc[int(.85 * n):]}
        for p, d in parts.items():
            d.to_csv(clean / f"{names[k]}_{p}.csv", index=False)
    res, val = R.run_all(str(clean), str(tmp_path / "out"), budgets=(0, 10), seeds=2)
    assert not res.empty and set(res.source) == {"CC", "MM", "HI"}
    for f in ("results.csv", "gains.csv", "concept_validity.csv", "manifest.json"):
        assert os.path.exists(tmp_path / "out" / f)
    m = json.load(open(tmp_path / "out" / "manifest.json"))
    assert m["concept_columns"]["HI"]["amount_anomaly"] == []
