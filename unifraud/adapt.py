"""Domain-adaptation models in the shared concept space, implemented in NumPy with analytic gradients.

Why NumPy: the concept space is 3-dimensional (paper Sec. V-B), so a one-hidden-layer encoder is enough, the
whole thing runs on CPU in seconds, and every gradient can be checked against finite differences
(``tests/test_adapt.py``). PyTorch versions (and the attention-gated variant) can be dropped in later.

Model: encoder  h = tanh(Z W1 + b1)  (m units); label head  p = sigmoid(h wy + by);  domain head  q = sigmoid(h wd + bd).

Objectives (paper Eq. (unified)):
  L_y    weighted logistic loss on labelled source data (positive weight omega, Prop. weight)
  L_d    domain loss, source = 0 / target = 1;  DANN reversal: the encoder receives  -lambda * dL_d/dh  (Prop. grl),
         the domain head descends on L_d;  lambda_p = 2 / (1 + exp(-10 p)) - 1
  L_lmmd class-conditional (local) multi-bandwidth MMD between source features (true labels) and target features
         (soft predictions as class weights), Gaussian kernels with sigma = multiples of the median distance
  fine-tuning on b labelled target examples with an L2-SP anchor  (eta/2)||theta - theta_s||^2  (Eq. l2sp)
"""
from __future__ import annotations

import numpy as np

PARAMS = ("W1", "b1", "wy", "by", "wd", "bd")


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def _bce_logits(logit, y, w=None):
    """Mean weighted binary cross-entropy from logits (stable)."""
    l = np.maximum(logit, 0) - logit * y + np.log1p(np.exp(-np.abs(logit)))
    return float(np.mean(l if w is None else w * l))


# ----------------------------------------------------------------------------- model and component gradients
def init_params(d: int, m: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return {"W1": rng.normal(scale=1.0 / np.sqrt(d), size=(d, m)), "b1": np.zeros(m),
            "wy": rng.normal(scale=1.0 / np.sqrt(m), size=m), "by": np.zeros(()),
            "wd": rng.normal(scale=1.0 / np.sqrt(m), size=m), "bd": np.zeros(())}


def encode(P, Z):
    return np.tanh(Z @ P["W1"] + P["b1"])


def _zero_grads(P):
    return {k: np.zeros_like(v) for k, v in P.items()}


def _backprop_encoder(P, Z, h, dh, g):
    dpre = dh * (1.0 - h ** 2)
    g["W1"] += Z.T @ dpre
    g["b1"] += dpre.sum(axis=0)


def grad_label(P, Z, y, omega: float = 1.0):
    """Weighted source loss L_y and its gradient wrt all parameters (wd, bd get zero)."""
    h = encode(P, Z)
    logit = h @ P["wy"] + P["by"]
    w = np.where(y == 1, omega, 1.0)
    loss = _bce_logits(logit, y, w)
    dl = w * (_sigmoid(logit) - y) / len(y)
    g = _zero_grads(P)
    g["wy"] += h.T @ dl
    g["by"] += dl.sum()
    _backprop_encoder(P, Z, h, np.outer(dl, P["wy"]), g)
    return loss, g


def grad_domain(P, Zs, Zt):
    """Domain loss L_d = 0.5*[mean BCE(source,0) + mean BCE(target,1)] and its gradient (true gradient, no reversal)."""
    hs, ht = encode(P, Zs), encode(P, Zt)
    ls = hs @ P["wd"] + P["bd"]
    lt = ht @ P["wd"] + P["bd"]
    loss = 0.5 * (_bce_logits(ls, np.zeros(len(ls))) + _bce_logits(lt, np.ones(len(lt))))
    dls = _sigmoid(ls) / (2 * len(ls))
    dlt = (_sigmoid(lt) - 1.0) / (2 * len(lt))
    g = _zero_grads(P)
    g["wd"] += hs.T @ dls + ht.T @ dlt
    g["bd"] += dls.sum() + dlt.sum()
    _backprop_encoder(P, Zs, hs, np.outer(dls, P["wd"]), g)
    g2 = _zero_grads(P)
    _backprop_encoder(P, Zt, ht, np.outer(dlt, P["wd"]), g2)
    for k in ("W1", "b1"):
        g[k] += g2[k]
    return loss, g


def _median_dist2(H):
    d2 = ((H[:, None, :] - H[None, :, :]) ** 2).sum(-1)
    iu = np.triu_indices(len(H), 1)
    v = d2[iu]
    return float(np.median(v)) if len(v) and np.median(v) > 0 else 1.0


def lmmd_loss_and_dh(hs, ht, ys, pt, sigma_scales=(0.5, 1.0, 2.0, 4.0), med2=None):
    """Class-conditional multi-bandwidth MMD: (1/C) sum_c sum_u a_c^T K_u a_c with a_c = [w_s^c ; -w_t^c].

    ``ys`` source labels in {0,1}; ``pt`` target P(y=1) (treated as constants). Returns (loss, dh_s, dh_t).
    ``med2`` fixes the bandwidth reference (median squared distance) so finite-difference checks are exact.
    """
    ns, nt = len(hs), len(ht)
    H = np.vstack([hs, ht])
    med2 = _median_dist2(H) if med2 is None else med2
    D = H[:, None, :] - H[None, :, :]
    d2 = (D ** 2).sum(-1)
    # class weights over the concatenated sample
    ws = np.stack([(ys == 0).astype(float), (ys == 1).astype(float)])          # (2, ns)
    ws = ws / np.maximum(ws.sum(1, keepdims=True), 1e-12)
    wt = np.stack([1.0 - pt, pt])
    wt = wt / np.maximum(wt.sum(1, keepdims=True), 1e-12)
    A = np.concatenate([ws, -wt], axis=1)                                          # (2, ns+nt)
    loss = 0.0
    dH = np.zeros_like(H)
    for sc in sigma_scales:
        sig2 = sc * med2
        K = np.exp(-d2 / (2 * sig2))
        for c in range(A.shape[0]):
            a = A[c]
            loss += float(a @ K @ a) / A.shape[0]
            # d(a^T K a)/dh_i = 2 a_i sum_j a_j dk_ij/dh_i,  dk_ij/dh_i = -k_ij (h_i - h_j)/sig2
            coef = (a[:, None] * a[None, :]) * K                                  # (n, n)
            dH += (2.0 / A.shape[0]) * (-(coef[:, :, None] * D).sum(1) / sig2)
    return loss, dH[:ns], dH[ns:]


def grad_lmmd(P, Zs, ys, Zt, pt, sigma_scales=(0.5, 1.0, 2.0, 4.0), med2=None):
    hs, ht = encode(P, Zs), encode(P, Zt)
    loss, dhs, dht = lmmd_loss_and_dh(hs, ht, ys, pt, sigma_scales, med2)
    g = _zero_grads(P)
    _backprop_encoder(P, Zs, hs, dhs, g)
    g2 = _zero_grads(P)
    _backprop_encoder(P, Zt, ht, dht, g2)
    for k in ("W1", "b1"):
        g[k] += g2[k]
    return loss, g


# ----------------------------------------------------------------------------- optimiser
class Adam:
    def __init__(self, P, lr=0.01, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps, self.t = lr, b1, b2, eps, 0
        self.m = {k: np.zeros_like(v) for k, v in P.items()}
        self.v = {k: np.zeros_like(v) for k, v in P.items()}

    def step(self, P, g):
        self.t += 1
        for k in P:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g[k] ** 2
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            P[k] = P[k] - self.lr * mh / (np.sqrt(vh) + self.eps)


# ----------------------------------------------------------------------------- the model
class AdaptModel:
    """One-hidden-layer concept-space classifier with source training, DANN / LMMD adaptation and L2-SP fine-tuning."""

    def __init__(self, d: int = 3, hidden: int = 16, seed: int = 0):
        self.P = init_params(d, hidden, seed)
        self.seed = seed
        self.omega_ = 1.0

    # -- helpers
    @staticmethod
    def _balanced_omega(y, cap=50.0):
        n1 = max(int((y == 1).sum()), 1)
        return float(min((len(y) - n1) / n1, cap))

    def predict_logit(self, Z):
        return encode(self.P, np.asarray(Z, float)) @ self.P["wy"] + self.P["by"]

    def predict_proba(self, Z):
        """Posterior under the *weighted* training prior (apply ``calibrate`` to undo the positive-class weight)."""
        return _sigmoid(self.predict_logit(Z))

    def calibrate(self, p):
        """Undo the positive-class weight (Prop. weight): odds divided by omega."""
        p = np.clip(p, 1e-12, 1 - 1e-12)
        o = p / (1 - p) / self.omega_
        return o / (1 + o)

    def _batch(self, rng, n, B):
        return rng.choice(n, size=min(B, n), replace=False)

    # -- training
    def fit_source(self, Zs, ys, iters=400, lr=0.02, batch=256, omega=None):
        Zs, ys = np.asarray(Zs, float), np.asarray(ys).astype(float)
        self.omega_ = self._balanced_omega(ys) if omega is None else omega
        rng = np.random.default_rng(self.seed)
        opt = Adam(self.P, lr)
        for _ in range(iters):
            i = self._batch(rng, len(ys), batch)
            _, g = grad_label(self.P, Zs[i], ys[i], self.omega_)
            opt.step(self.P, g)
        return self

    def fit_adapt(self, Zs, ys, Zt, method="dann", iters=400, lr=0.02, batch=256, omega=None,
                  lam_max=1.0, alpha=1.0, gamma=10.0):
        """Adapt using labelled source and UNLABELLED target. method in {'source','dann','lmmd','dann_lmmd'}."""
        Zs, ys, Zt = np.asarray(Zs, float), np.asarray(ys).astype(float), np.asarray(Zt, float)
        self.omega_ = self._balanced_omega(ys) if omega is None else omega
        rng = np.random.default_rng(self.seed)
        opt = Adam(self.P, lr)
        use_d = method in ("dann", "dann_lmmd")
        use_m = method in ("lmmd", "dann_lmmd")
        for it in range(iters):
            p = it / max(iters - 1, 1)
            lam = lam_max * (2.0 / (1.0 + np.exp(-gamma * p)) - 1.0)
            i, j = self._batch(rng, len(ys), batch), self._batch(rng, len(Zt), batch)
            _, g = grad_label(self.P, Zs[i], ys[i], self.omega_)
            if use_d:
                _, gd = grad_domain(self.P, Zs[i], Zt[j])
                for k in ("wd", "bd"):                      # discriminator descends on L_d
                    g[k] += gd[k]
                for k in ("W1", "b1"):                      # encoder reverses it (Prop. grl): applied ONCE
                    g[k] -= lam * gd[k]
            if use_m:
                pt = _sigmoid(encode(self.P, Zt[j]) @ self.P["wy"] + self.P["by"])
                _, gm = grad_lmmd(self.P, Zs[i], ys[i], Zt[j], pt)
                for k in ("W1", "b1"):
                    g[k] += alpha * gm[k]
            opt.step(self.P, g)
        return self

    def finetune(self, Zb, yb, eta=1.0, iters=200, lr=0.005, omega=None):
        """Fine-tune on b labelled target examples with an L2-SP anchor to the current (source-trained) weights."""
        Zb, yb = np.asarray(Zb, float), np.asarray(yb).astype(float)
        anchor = {k: v.copy() for k, v in self.P.items()}
        om = self._balanced_omega(yb) if omega is None else omega
        self.omega_ = om
        opt = Adam(self.P, lr)
        keys = ("W1", "b1", "wy", "by")
        for _ in range(iters):
            _, g = grad_label(self.P, Zb, yb, om)
            for k in keys:
                g[k] = g[k] + eta * (self.P[k] - anchor[k])
            for k in ("wd", "bd"):
                g[k] = np.zeros_like(g[k])
            opt.step(self.P, g)
        return self

    def distance_from(self, other_params):
        return float(np.sqrt(sum(((self.P[k] - other_params[k]) ** 2).sum() for k in ("W1", "b1", "wy", "by"))))

    def snapshot(self):
        return {k: v.copy() for k, v in self.P.items()}
