"""First transfer baselines in concept space (non-neural; DANN/LMMD/gated net come later).

Methods (paper Sec. VII, Alg. eval):
  target_only   fit on the b labelled target examples only            (the fair reference)
  source_only   fit on source, apply to target (zero-shot)            (+ optional prior correction)
  pooled        source + b target examples, target examples up-weighted by ``target_weight``
                (Prop. mix: when domains differ the optimal weight on target labels exceeds
                their sample share)

Transfer gain  Delta_m(b) = M(m) - M(target_only at the same b)  isolates transfer from supervision.
All thresholds come from calibrated posteriors and costs (Prop. cost), never from target-test labels.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from . import metrics as M


# ----------------------------------------------------------------------------- prior handling
def saerens_correct(p, pi_s: float, pi_t: float):
    """Posterior under label shift: odds scale by [pi_t/(1-pi_t)] / [pi_s/(1-pi_s)] (Prop. saerens)."""
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    ratio = (pi_t / (1 - pi_t)) / (pi_s / (1 - pi_s))
    odds = p / (1 - p) * ratio
    return odds / (1 + odds)


def bbse_prior(y_src_val, yhat_src_val, yhat_tgt, pi_s: float | None = None) -> float:
    """Black-box shift estimate of the target fraud prior (Lipton et al. 2018; Prop. bbse).

    Solves  mu = C w  with C_ij = P_s(yhat=i, y=j), mu_i = P_t(yhat=i); returns pi_t = w_1 * pi_s.
    Falls back to the source prior if C is singular.
    """
    y = np.asarray(y_src_val).astype(int)
    yh = np.asarray(yhat_src_val).astype(int)
    yt = np.asarray(yhat_tgt).astype(int)
    pi_s = float(y.mean()) if pi_s is None else pi_s
    C = np.array([[np.mean((yh == i) & (y == j)) for j in (0, 1)] for i in (0, 1)])
    mu = np.array([np.mean(yt == 0), np.mean(yt == 1)])
    try:
        w = np.linalg.solve(C, mu)
    except np.linalg.LinAlgError:
        return pi_s
    w = np.clip(w, 0, None)
    return float(np.clip(w[1] * pi_s, 1e-6, 1 - 1e-6))


# ----------------------------------------------------------------------------- models
def make_model(kind: str = "logreg", seed: int = 0):
    if kind == "logreg":
        return LogisticRegression(max_iter=1000, random_state=seed)
    if kind == "hgb":
        return HistGradientBoostingClassifier(max_depth=3, max_iter=100, learning_rate=0.1, random_state=seed)
    raise ValueError(kind)


def _fit(X, y, kind, seed, sample_weight=None):
    m = make_model(kind, seed)
    if len(np.unique(y)) < 2:
        return None
    m.fit(X, y, sample_weight=sample_weight)
    return m


def _proba(m, X, fallback: float):
    return np.full(len(X), fallback) if m is None else m.predict_proba(X)[:, 1]


# ----------------------------------------------------------------------------- one pair
@dataclass
class PairResult:
    method: str
    b: int
    seed: int
    ap_skill: float
    auc: float
    mcc: float
    cost: float
    ece: float
    prec_at_1pct: float
    pi_hat: float | None = None


def _score(y, p, tau, c_fp, c_fn, method, b, seed, pi_hat=None) -> PairResult:
    pred = (p >= tau).astype(int)
    return PairResult(method, b, seed, M.ap_skill(y, p), M.auc(y, p), M.mcc(y, pred),
                      M.expected_cost(y, pred, c_fp, c_fn), M.ece(y, p), M.precision_at_k(y, p, 0.01), pi_hat)


def sample_budget(y_train: np.ndarray, b: int, seed: int) -> np.ndarray:
    """Stratified draw of ``b`` labelled target examples (at least one per class when possible)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y_train).astype(int)
    idx1, idx0 = np.where(y == 1)[0], np.where(y == 0)[0]
    n1 = int(min(len(idx1), max(1, round(b * y.mean())))) if len(idx1) else 0
    n0 = int(min(len(idx0), max(0, b - n1)))
    return np.concatenate([rng.choice(idx1, n1, replace=False) if n1 else [],
                           rng.choice(idx0, n0, replace=False) if n0 else []]).astype(int)


def run_pair(Zs, ys, Zt_train, yt_train, Zt_test, yt_test, *, b: int, seed: int,
             kind: str = "logreg", c_fp: float = 1.0, c_fn: float = 10.0,
             target_weight: float | None = None, prior_correct: bool = True,
             da_methods: Sequence[str] = (), da_iters: int = 300):
    """Evaluate target_only / source_only / pooled at budget ``b`` on the fixed target TEST set.

    Zs, Zt_*: DataFrames/arrays in the shared concept space. Returns a list of PairResult.
    """
    Zs, Zt_train, Zt_test = (np.asarray(a, float) for a in (Zs, Zt_train, Zt_test))
    ys = np.asarray(ys).astype(int)
    yt_train = np.asarray(yt_train).astype(int)
    yt_test = np.asarray(yt_test).astype(int)
    pi_s = float(ys.mean())
    tau = M.bayes_threshold(c_fp, c_fn)
    out: list[PairResult] = []

    # source-only (zero-shot). Fit on 70% of the source and keep 30% as held-out *source validation*
    # so the BBSE confusion matrix is not computed in-sample; no target labels are used.
    from sklearn.model_selection import train_test_split
    try:
        i_fit, i_val = train_test_split(np.arange(len(ys)), test_size=0.3, stratify=ys, random_state=seed)
    except ValueError:                                                # too few positives to stratify
        i_fit, i_val = np.arange(len(ys)), np.arange(len(ys))
    ms = _fit(Zs[i_fit], ys[i_fit], kind, seed)
    p_src = _proba(ms, Zt_test, pi_s)
    pi_hat = None
    if prior_correct and ms is not None:
        yhat_val = (_proba(ms, Zs[i_val], pi_s) >= 0.5).astype(int)
        pi_hat = bbse_prior(ys[i_val], yhat_val, (p_src >= 0.5).astype(int), pi_s)
        p_src = saerens_correct(p_src, pi_s, pi_hat)
    out.append(_score(yt_test, p_src, tau, c_fp, c_fn, "source_only", b, seed, pi_hat))

    # domain-adaptation baselines (NumPy models, unlabelled target features only; opt-in)
    if da_methods:
        from . import adapt as AD
        for meth in da_methods:
            if meth == "ft_l2sp":
                continue                                           # handled below (needs target labels)
            am = AD.AdaptModel(Zs.shape[1], 16, seed=seed)
            am.fit_adapt(Zs[i_fit], ys[i_fit], Zt_train, method=meth, iters=da_iters, batch=128)
            p = am.calibrate(am.predict_proba(Zt_test))
            pi_h = None
            if prior_correct:
                yhat_val = (am.calibrate(am.predict_proba(Zs[i_val])) >= 0.5).astype(int)
                pi_h = bbse_prior(ys[i_val], yhat_val, (p >= 0.5).astype(int), pi_s)
                p = saerens_correct(p, pi_s, pi_h)
            out.append(_score(yt_test, p, tau, c_fp, c_fn, meth, b, seed, pi_h))

    if b > 0:
        sel = sample_budget(yt_train, b, seed)
        Zb, yb = Zt_train[sel], yt_train[sel]
        pi_b = float(yb.mean()) if len(yb) else pi_s
        # target-only
        mt = _fit(Zb, yb, kind, seed)
        out.append(_score(yt_test, _proba(mt, Zt_test, pi_b), tau, c_fp, c_fn, "target_only", b, seed))
        # pooled with up-weighted target (Prop. mix): default weight = source size / target size share
        w_t = target_weight if target_weight is not None else max(1.0, 0.5 * len(Zs) / max(len(Zb), 1))
        X = np.vstack([Zs, Zb])
        y = np.concatenate([ys, yb])
        sw = np.concatenate([np.ones(len(Zs)), np.full(len(Zb), w_t)])
        mp = _fit(X, y, kind, seed, sample_weight=sw)
        p_pool = _proba(mp, Zt_test, pi_b)
        if prior_correct and mp is not None and len(yb):
            # re-anchor to the labelled-target prior: pooled model was trained under a mixed prior
            pi_pool = float(np.average(y, weights=sw))
            p_pool = saerens_correct(p_pool, pi_pool, pi_b)
        out.append(_score(yt_test, p_pool, tau, c_fp, c_fn, "pooled", b, seed))
        if "ft_l2sp" in da_methods and len(np.unique(yb)) == 2:
            from . import adapt as AD
            base = AD.AdaptModel(Zs.shape[1], 16, seed=seed).fit_source(Zs[i_fit], ys[i_fit], iters=da_iters)
            base.finetune(Zb, yb, eta=1.0, iters=150)
            p_ft = base.calibrate(base.predict_proba(Zt_test))
            out.append(_score(yt_test, p_ft, tau, c_fp, c_fn, "ft_l2sp", b, seed))
    return out


def transfer_gain(results: pd.DataFrame, metric: str = "ap_skill") -> pd.DataFrame:
    """Delta_m(b) = metric(m) - metric(target_only) per (b, seed); source_only is compared at each b."""
    df = results.copy()
    ref = df[df.method == "target_only"].set_index(["b", "seed"])[metric].rename("ref")
    sub = df[df.method != "target_only"].join(ref, on=["b", "seed"])
    sub["gain"] = sub[metric] - sub["ref"]
    return sub
