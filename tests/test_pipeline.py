import numpy as np
import pandas as pd
import pytest

from unifraud import labels_hi as L
from unifraud import leakage as LK
from unifraud import preprocess as P
from unifraud import splits as SP


# ------------------------------------------------------------------ HI labels
def _claims(**cols):
    return pd.DataFrame(cols)


def test_missing_medical_is_not_fraud():
    df = _claims(Service=[10.0, 10.0], Medical=[np.nan, 5.0], Total=[10.0, 15.0], Approved_Cost=[100.0, 100.0])
    f = L.compute_rule_flags(df)
    y = L.assign_labels(f)
    assert f["dq_missing_medical"].tolist() == [1, 0]
    assert f["dq_nonpositive_observed"].sum() == 0            # old rule would have flagged claim 0
    assert y.tolist() == [0.0, 0.0]


def test_overbilling_rules_fire():
    df = _claims(Service=[150.0, 10.0, 10.0], Medical=[1.0, 60.0, 1.0], Total=[151.0, 70.0, 150.0],
                 Approved_Cost=[100.0, 100.0, 100.0])
    f = L.compute_rule_flags(df)
    assert f["r1_service_over_tariff"].tolist() == [1, 0, 0]
    assert f["r2_drug_over_half_tariff"].tolist() == [0, 1, 0]
    assert f["r3_total_over_tariff"].tolist() == [1, 0, 1]
    assert L.assign_labels(f).tolist() == [1.0, 1.0, 1.0]


def test_r4_requires_all_observed_and_detects_inconsistency():
    df = _claims(Service=[10.0, 10.0, 10.0], Medical=[5.0, 5.0, np.nan], Total=[15.0, 30.0, 30.0],
                 Approved_Cost=[100.0] * 3)
    f = L.compute_rule_flags(df)
    assert f["r4_total_inconsistent"].tolist() == [0, 1, 0]


def test_unmatched_tariff_is_unassessable_not_legitimate():
    df = _claims(Service=[10.0, 10.0], Medical=[1.0, 1.0], Total=[11.0, 11.0], Approved_Cost=[np.nan, 100.0])
    y = L.assign_labels(L.compute_rule_flags(df))
    assert np.isnan(y.iloc[0]) and y.iloc[1] == 0.0
    y2 = L.assign_labels(L.compute_rule_flags(df), include_unmatched=True)
    assert y2.iloc[0] == 0.0


def test_total_strings_are_parsed():
    df = _claims(Service=[10.0], Medical=[1.0], Total=["11.0"], Approved_Cost=[100.0])
    assert L.compute_rule_flags(df)["r4_total_inconsistent"].iloc[0] == 0
    df2 = _claims(Service=[10.0], Medical=[1.0], Total=["n/a"], Approved_Cost=[100.0])
    assert L.compute_rule_flags(df2)["dq_missing_total"].iloc[0] == 1


def test_observed_nonpositive_is_only_dq_unless_requested():
    df = _claims(Service=[0.0], Medical=[1.0], Total=[1.0], Approved_Cost=[100.0])
    f = L.compute_rule_flags(df)
    assert f["dq_nonpositive_observed"].iloc[0] == 1
    assert L.assign_labels(f).iloc[0] == 0.0
    assert L.assign_labels(f, include_nonpositive=True).iloc[0] == 1.0


# ------------------------------------------------------------------ leakage guards
def test_forbidden_columns_hi():
    cols = ["Age", "Sex", "dev_total", "overbill_drug", "Folder No", "NHIS ID No", "drg_fraud_rate",
            "Total", "Service", "Medical", "Approved_Cost", "MDC_Specialty", "length_of_stay", "r1_service_over_tariff"]
    bad = set(LK.forbidden_columns(cols, "HI"))
    assert bad == {"dev_total", "overbill_drug", "Folder No", "NHIS ID No", "drg_fraud_rate", "Total",
                   "Service", "Medical", "Approved_Cost", "r1_service_over_tariff"}


def test_assert_no_leakage_flags_label_copy():
    rng = np.random.default_rng(0)
    y = pd.Series(rng.integers(0, 2, 500))
    X = pd.DataFrame({"noise": rng.normal(size=500), "copy": y * 3.0 + 1})
    with pytest.raises(ValueError, match="copy"):
        LK.assert_no_leakage(X, y, domain="CC")
    LK.assert_no_leakage(X[["noise"]], y, domain="CC")


# ------------------------------------------------------------------ splits
def test_chronological_split_is_ordered_and_disjoint():
    t = pd.Series(np.random.default_rng(1).permutation(1000))
    tr, va, te = SP.chronological_split(t)
    SP.assert_disjoint(tr, va, te)
    assert t.iloc[tr].max() < t.iloc[va].min() and t.iloc[va].max() < t.iloc[te].min()
    assert abs(len(tr) / 1000 - 0.70) < 0.01


def test_grouped_split_has_no_group_overlap():
    rng = np.random.default_rng(2)
    groups = rng.integers(0, 300, size=1500)
    y = (rng.random(1500) < 0.4).astype(int)
    tr, va, te = SP.grouped_stratified_split(y, groups)
    SP.assert_disjoint(tr, va, te)
    g = [set(groups[i]) for i in (tr, va, te)]
    assert not (g[0] & g[1]) and not (g[0] & g[2]) and not (g[1] & g[2])
    assert len(tr) + len(va) + len(te) == 1500


# ------------------------------------------------------------------ preprocessing
def test_oof_target_encoder_own_label_does_not_leak():
    rng = np.random.default_rng(3)
    n = 400
    X = pd.DataFrame({"id": [f"u{i}" for i in range(n)]})          # every row its own category
    y = pd.Series(rng.integers(0, 2, n))
    enc = P.OOFTargetEncoder(smoothing=1.0, n_splits=5, random_state=0)
    a = enc.fit_transform(X, y)
    # every category is a singleton: a leaky encoder would copy the row's own label into its encoding,
    # an out-of-fold encoder can only reflect the other rows' (independent) labels
    corr = np.corrcoef(a[:, 0], y.to_numpy())[0, 1]
    assert abs(corr) < 0.15


def test_target_encoder_unseen_category_gets_prior():
    X = pd.DataFrame({"c": ["a", "a", "b", "b"]})
    y = pd.Series([1, 1, 0, 0])
    enc = P.OOFTargetEncoder(smoothing=1.0).fit(X, y)
    out = enc.transform(pd.DataFrame({"c": ["zzz"]}))
    assert out[0, 0] == pytest.approx(0.5)


def test_winsorizer_uses_train_quantile_only():
    train = np.arange(100, dtype=float).reshape(-1, 1)
    w = P.Winsorizer(upper_q=0.9).fit(train)
    cap = w.upper_[0]
    out = w.transform(np.array([[1e9], [1.0]]))
    assert out[0, 0] == pytest.approx(cap) and out[1, 0] == 1.0


def test_preprocessing_statistics_independent_of_test_data():
    rng = np.random.default_rng(4)
    n = 600
    X = pd.DataFrame({"amt": rng.lognormal(3, 1, n), "cat": rng.choice(list("abcdefghijklmnop"), n),
                      "kind": rng.choice(["x", "y"], n)})
    y = pd.Series((rng.random(n) < 0.2).astype(int))
    tr, va, te = np.arange(0, 400), np.arange(400, 500), np.arange(500, 600)
    A1 = P.fit_transform_partitions(X, y, tr, va, te)[0]
    X2 = X.copy()
    X2.loc[te, "amt"] = 1e12                         # corrupt the test partition
    X2.loc[te, "cat"] = "zzz"
    y2 = y.copy()
    y2.iloc[te] = 1 - y2.iloc[te]
    A2 = P.fit_transform_partitions(X2, y2, tr, va, te)[0]
    np.testing.assert_allclose(A1, A2)               # train features identical -> nothing learned from test


def test_time_features_parse_before_encoding():
    df = pd.DataFrame({"trans_date_trans_time": ["2020-06-21 12:14:25", "2020-06-21 03:00:00"]})
    out = P.add_time_features(df, time_col="trans_date_trans_time")
    assert out["hour_of_day"].tolist() == [12, 3]
    assert out["unusual_hour"].tolist() == [0, 1]
    assert out["time_epoch_s"].iloc[0] > 1.5e9                   # not 1970


def test_sim_step_time_features():
    df = pd.DataFrame({"step": [0, 25, 100]})
    out = P.add_time_features(df, step_col="step")
    assert out["hour_of_day"].tolist() == [0, 1, 4]
    assert out["day_of_sim"].tolist() == [0, 1, 4]
