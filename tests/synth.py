"""Synthetic three-sector data sharing a latent risk r but with different schemas, units and priors."""
import numpy as np
import pandas as pd


def _labels(r, prior, rng, k=2.5):
    # choose offset c by bisection so that the mean label rate equals `prior`
    lo, hi = -6.0, 8.0
    for _ in range(60):
        c = (lo + hi) / 2
        rate = (1 / (1 + np.exp(-k * (r - c)))).mean()
        lo, hi = (c, hi) if rate > prior else (lo, c)
    p = 1 / (1 + np.exp(-k * (r - c)))
    return (rng.random(len(r)) < p).astype(int)


def make_sector(key, n, prior, seed):
    rng = np.random.default_rng(seed)
    r = rng.normal(size=n)
    y = _labels(r, prior, rng)
    nz = lambda s=0.6: rng.normal(scale=s, size=n)
    if key == "CC":
        df = pd.DataFrame({
            "amt": np.exp(0.8 * r + nz()) * 80,                       # USD-like units
            "geo_distance": 20 + 6 * r + 4 * nz(),
            "unusual_hour": (rng.random(n) < 1 / (1 + np.exp(-(r - 0.5)))).astype(int),
            "merchant__te": 0.01 + 0.02 / (1 + np.exp(-r)) + 0.004 * nz(),
            "noise_a": rng.normal(size=n), "noise_b": rng.normal(size=n)})
    elif key == "MM":
        df = pd.DataFrame({
            "noise_a": rng.normal(size=n),                            # different column order / meaning
            "amount": np.exp(0.7 * r + nz()) * 5000,                  # local-currency units
            "balance_error_orig": np.abs(1.5 * r + nz(1.0)),
            "balance_zeroed_dest": (rng.random(n) < 1 / (1 + np.exp(-r))).astype(int),
            "unusual_hour": (rng.random(n) < 1 / (1 + np.exp(-(r - 0.3)))).astype(int)})
    else:  # HI
        df = pd.DataFrame({
            "length_of_stay": np.clip(2 + 1.2 * r + nz(), 0, None),
            "drg_claim_volume": 50 + 10 * r + 6 * nz(),
            "attendance_month": rng.integers(1, 13, n).astype(float),
            "noise_a": rng.normal(size=n)})
    df["fraud_label"] = y
    return df


def make_all(seed=0, n=3000, priors=(0.02, 0.5, 0.4)):
    return {k: make_sector(k, n, p, seed + i) for i, (k, p) in enumerate(zip(("CC", "MM", "HI"), priors))}
