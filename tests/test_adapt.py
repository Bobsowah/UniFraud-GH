import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from unifraud import adapt as A
from unifraud import metrics as M


def _fd(f, P, key, eps=1e-6):
    """Finite-difference gradient of scalar f(P) wrt P[key]."""
    out = np.zeros_like(P[key], dtype=float)
    it = np.nditer(P[key], flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        old = P[key][idx]
        P[key][idx] = old + eps
        fp = f(P)
        P[key][idx] = old - eps
        fm = f(P)
        P[key][idx] = old
        out[idx] = (fp - fm) / (2 * eps)
    return out


def _data(seed=0, ns=40, nt=30, d=3):
    rng = np.random.default_rng(seed)
    Zs = rng.normal(size=(ns, d))
    ys = (Zs[:, 0] + 0.5 * Zs[:, 1] + 0.3 * rng.normal(size=ns) > 0).astype(float)
    Zt = rng.normal(size=(nt, d)) + 0.7
    return Zs, ys, Zt


@pytest.mark.parametrize("key", A.PARAMS)
def test_gradient_label_matches_finite_differences(key):
    P = A.init_params(3, 5, seed=1)
    Zs, ys, _ = _data()
    _, g = A.grad_label(P, Zs, ys, omega=3.0)
    num = _fd(lambda Q: A.grad_label(Q, Zs, ys, 3.0)[0], P, key)
    np.testing.assert_allclose(g[key], num, atol=1e-6, rtol=1e-4)


@pytest.mark.parametrize("key", A.PARAMS)
def test_gradient_domain_matches_finite_differences(key):
    P = A.init_params(3, 5, seed=2)
    Zs, _, Zt = _data()
    _, g = A.grad_domain(P, Zs, Zt)
    num = _fd(lambda Q: A.grad_domain(Q, Zs, Zt)[0], P, key)
    np.testing.assert_allclose(g[key], num, atol=1e-6, rtol=1e-4)


@pytest.mark.parametrize("key", ["W1", "b1"])
def test_gradient_lmmd_matches_finite_differences(key):
    P = A.init_params(3, 5, seed=3)
    Zs, ys, Zt = _data()
    pt = np.clip(np.random.default_rng(4).random(len(Zt)), 0.05, 0.95)
    med2 = 0.7                                         # fixed bandwidth reference so the loss is smooth in P
    _, g = A.grad_lmmd(P, Zs, ys, Zt, pt, med2=med2)
    num = _fd(lambda Q: A.grad_lmmd(Q, Zs, ys, Zt, pt, med2=med2)[0], P, key)
    np.testing.assert_allclose(g[key], num, atol=1e-6, rtol=1e-4)


def test_lmmd_is_zero_for_identical_samples_and_positive_otherwise():
    rng = np.random.default_rng(0)
    h = np.tanh(rng.normal(size=(30, 4)))
    y = (rng.random(30) > 0.5).astype(float)
    same, _, _ = A.lmmd_loss_and_dh(h, h.copy(), y, y.copy())
    assert same < 1e-10
    diff, _, _ = A.lmmd_loss_and_dh(h, h + 0.5, y, y.copy())
    assert diff > 1e-3


def test_calibrate_undoes_positive_weight():
    m = A.AdaptModel(3, 4)
    m.omega_ = 9.0
    p = np.array([0.5])
    q = m.calibrate(p)
    assert q[0] == pytest.approx(0.1)                  # odds 1 / 9


def _shifted_problem(seed=0, n=1500):
    rng = np.random.default_rng(seed)
    Zs = rng.normal(size=(n, 3))
    ys = (Zs[:, 0] + 0.6 * Zs[:, 1] + 0.4 * rng.normal(size=n) > 1.0).astype(int)
    Zt = rng.normal(size=(n, 3)) * np.array([1.0, 1.0, 1.0]) + np.array([1.5, 0.0, 0.0])
    yt = (Zt[:, 0] - 1.5 + 0.6 * Zt[:, 1] + 0.4 * rng.normal(size=n) > 1.0).astype(int)
    return Zs, ys, Zt, yt


def test_dann_reduces_domain_discriminability_of_the_features():
    Zs, ys, Zt, _ = _shifted_problem()
    acc = {}
    for name in ("source", "dann"):
        m = A.AdaptModel(3, 16, seed=0)
        m.fit_adapt(Zs, ys, Zt, method=name, iters=600, lr=0.02)
        Hs, Ht = A.encode(m.P, Zs), A.encode(m.P, Zt)
        X = np.vstack([Hs, Ht])
        d = np.r_[np.zeros(len(Hs)), np.ones(len(Ht))]
        acc[name] = cross_val_score(LogisticRegression(max_iter=500), X, d, cv=3).mean()
    assert acc["dann"] < acc["source"]


def test_lmmd_adaptation_reduces_feature_discrepancy():
    Zs, ys, Zt, _ = _shifted_problem(seed=1)
    out = {}
    for name in ("source", "lmmd"):
        m = A.AdaptModel(3, 16, seed=0)
        m.fit_adapt(Zs, ys, Zt, method=name, iters=250, lr=0.02, alpha=5.0, batch=64)
        Hs, Ht = A.encode(m.P, Zs[:300]), A.encode(m.P, Zt[:300])
        pt = A._sigmoid(Ht @ m.P["wy"] + m.P["by"])
        out[name] = A.lmmd_loss_and_dh(Hs, Ht, ys[:300].astype(float), pt)[0]
    assert out["lmmd"] < out["source"]


def test_l2sp_keeps_weights_closer_and_labels_help():
    Zs, ys, Zt, yt = _shifted_problem(seed=2)
    base = A.AdaptModel(3, 16, seed=0).fit_source(Zs, ys, iters=500)
    theta_s = base.snapshot()
    auc0 = M.auc(yt, base.predict_proba(Zt))
    rng = np.random.default_rng(0)
    idx = np.r_[rng.choice(np.where(yt == 1)[0], 25, replace=False), rng.choice(np.where(yt == 0)[0], 75, replace=False)]
    dist = {}
    aucs = {}
    for eta in (0.0, 5.0):
        m = A.AdaptModel(3, 16, seed=0)
        m.P = {k: v.copy() for k, v in theta_s.items()}
        m.omega_ = base.omega_
        m.finetune(Zt[idx], yt[idx], eta=eta, iters=300, lr=0.01)
        dist[eta] = m.distance_from(theta_s)
        aucs[eta] = M.auc(yt, m.predict_proba(Zt))
    assert dist[5.0] < dist[0.0]
    assert aucs[0.0] >= auc0 - 0.02                    # labelled target data do not hurt
