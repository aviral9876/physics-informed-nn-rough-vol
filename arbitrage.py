"""
Static-arbitrage and data-quality diagnostics on the BTC surface.
=================================================================

CNSNS revision: Reviewer 1 item 6 and Reviewer 2 item 5.

Reviewer 2: "No calendar- or butterfly-arbitrage check is reported for the
calibrated implied-volatility surface. This is standard practice in the
calibration literature and should be added or its absence justified."

Reviewer 1: crypto surfaces are "wide, thinly traded in the wings, and prone to
static arbitrage violations in raw mid quotes", and asks for the diagnostics
after filtering plus a fit restricted to a liquid moneyness core.

WHY THIS MATTERS FOR THE PAPER'S ARGUMENT. The manuscript attributes the coarse
496 vol bp fit to the expressiveness of rough Heston on wide crypto smiles. That
reading is only defensible if the target surface is itself arbitrage-consistent:
if the market quotes contain butterfly or calendar violations, then NO
arbitrage-free model can fit them, and part of the 496 is the data's fault
rather than the model's. This script measures which.

THREE TESTS, all on undiscounted calls with r = 0 (so the discount factor is 1
and the forward is the spot):

  butterfly  For K1 < K2 < K3 at one maturity, convexity of C in K:
                 w1*C(K1) + w3*C(K3) - C(K2) >= 0,
             with w1 = (K3-K2)/(K3-K1), w3 = (K2-K1)/(K3-K1). A negative value
             is a genuine arbitrage (buy the butterfly for a negative price).

  vertical   Monotonicity and the slope bound, -1 <= dC/dK <= 0.

  calendar   Total implied variance w(k,T) = sigma^2 T must be non-decreasing in
             T at FIXED log-moneyness k (not fixed strike). Checked on the k
             range shared by each adjacent maturity pair.

Put-call parity is tested separately on the RAW chain, because surface.py
ASSUMES parity when it keeps only the out-of-the-money leg -- an assumption the
manuscript previously described as an inference. Testing it is the honest
alternative.

Run:  python arbitrage.py      (seconds)
Writes results/arbitrage.json
"""

import json

import numpy as np
import pandas as pd
from scipy.stats import norm

from resultio import dump
from surface import build_surface, implied_vol

R = 0.0


# ------------------------------------------------------------------ helpers
def bs_call(F, K, T, sig):
    """Undiscounted Black-Scholes call on a forward, r = 0."""
    sig = np.asarray(sig, dtype=float)
    K = np.asarray(K, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = sig * np.sqrt(T)
        d1 = (np.log(F / K) + 0.5 * v ** 2) / v
        d2 = d1 - v
        out = F * norm.cdf(d1) - K * norm.cdf(d2)
    return np.where(v > 0, out, np.maximum(F - K, 0.0))


def call_curve(g):
    """
    Undiscounted call prices across strikes at one maturity.

    The surface keeps only the OTM leg, so the put quotes are converted with
    put-call parity (r = 0):  C = P + F - K. This is the same parity the
    surface construction implicitly relies on.
    """
    F = float(g["F"].iloc[0])
    K = g["K"].values.astype(float)
    C = bs_call(F, K, float(g["T"].iloc[0]), g["iv"].values)
    return F, K, C


# ------------------------------------------------------------------- tests
def butterfly(surf, label):
    """Convexity of C in K, per maturity. Returns (violations, tested)."""
    # track the true MINIMUM margin, not just violations: 'no violations with
    # a smallest margin of x' is a far stronger statement than 'no violations'.
    viol, tested, worst = [], 0, np.inf
    for T, g in surf.groupby("T"):
        g = g.sort_values("K")
        if len(g) < 3:
            continue
        F, K, C = call_curve(g)
        for i in range(len(K) - 2):
            K1, K2, K3 = K[i], K[i + 1], K[i + 2]
            if not (K1 < K2 < K3):
                continue
            w1 = (K3 - K2) / (K3 - K1)
            w3 = (K2 - K1) / (K3 - K1)
            val = w1 * C[i] + w3 * C[i + 2] - C[i + 1]
            tested += 1
            # normalise by spot so the tolerance is in px bp
            v_bp = 1e4 * val / F
            worst = min(worst, v_bp)
            if v_bp < 0.0:
                viol.append(dict(T=float(T), K=[float(K1), float(K2), float(K3)],
                                 value_px_bp=float(v_bp)))
    print(f"  [{label}] butterfly : {len(viol)}/{tested} violations, "
          f"min margin {worst:+.2f} px bp")
    return dict(n_tested=tested, n_violations=len(viol),
                worst_px_bp=float(worst), violations=viol)


def vertical(surf, label):
    """-1 <= dC/dK <= 0 between adjacent strikes, per maturity."""
    viol, tested, worst_hi, worst_lo = [], 0, -np.inf, np.inf
    for T, g in surf.groupby("T"):
        g = g.sort_values("K")
        if len(g) < 2:
            continue
        F, K, C = call_curve(g)
        slope = np.diff(C) / np.diff(K)
        for i, s in enumerate(slope):
            tested += 1
            worst_hi = max(worst_hi, s)          # should be <= 0
            worst_lo = min(worst_lo, s + 1.0)    # s >= -1  =>  s+1 >= 0
            if s > 0.0 or s < -1.0:
                viol.append(dict(T=float(T), K=[float(K[i]), float(K[i + 1])],
                                 slope=float(s)))
    print(f"  [{label}] vertical  : {len(viol)}/{tested} violations, "
          f"max slope {worst_hi:+.4f} (need <= 0), min slope+1 {worst_lo:+.4f}")
    return dict(n_tested=tested, n_violations=len(viol),
                max_slope=float(worst_hi), min_slope_plus_one=float(worst_lo),
                violations=viol)


def calendar(surf, label, n_grid=25):
    """Total implied variance non-decreasing in T at fixed log-moneyness."""
    Ts = np.sort(surf["T"].unique())
    viol, tested, worst = [], 0, np.inf
    for Ta, Tb in zip(Ts[:-1], Ts[1:]):
        ga = surf[surf["T"] == Ta].sort_values("k")
        gb = surf[surf["T"] == Tb].sort_values("k")
        if len(ga) < 2 or len(gb) < 2:
            continue
        lo = max(ga["k"].min(), gb["k"].min())
        hi = min(ga["k"].max(), gb["k"].max())
        if not (hi > lo):
            continue                      # no shared moneyness range
        ks = np.linspace(lo, hi, n_grid)
        wa = np.interp(ks, ga["k"], ga["iv"] ** 2 * Ta)
        wb = np.interp(ks, gb["k"], gb["iv"] ** 2 * Tb)
        d = wb - wa                       # must be >= 0
        tested += len(ks)
        worst = min(worst, float(d.min()))
        for kk, dd in zip(ks, d):
            if dd < 0.0:
                viol.append(dict(T_from=float(Ta), T_to=float(Tb),
                                 k=float(kk), dw=float(dd)))
    print(f"  [{label}] calendar  : {len(viol)}/{tested} violations, "
          f"min margin dw {worst:+.6f} (total variance units)")
    return dict(n_tested=tested, n_violations=len(viol),
                worst_dw=float(worst), violations=viol[:50])


def parity_on_raw(df):
    """
    Put-call parity on the RAW chain, where both legs are quoted two-sided.
    C - P = F - K with r = 0. Reported relative to the quoted spread, since a
    violation inside the spread is not tradable and therefore not an arbitrage.
    """
    df = df.copy()
    df["mid"] = np.where((df["bid"] > 0) & (df["ask"] > df["bid"]),
                         0.5 * (df["bid"] + df["ask"]), np.nan)
    df["spread"] = np.where((df["bid"] > 0) & (df["ask"] > df["bid"]),
                            df["ask"] - df["bid"], np.nan)
    rows, outside = [], 0
    for (T, K), g in df.groupby(["T", "strike"]):
        c = g[g["type"] == "C"]
        p = g[g["type"] == "P"]
        if len(c) != 1 or len(p) != 1:
            continue
        cm, pm = float(c["mid"].iloc[0]), float(p["mid"].iloc[0])
        if not (np.isfinite(cm) and np.isfinite(pm)):
            continue
        F = float(c["spot"].iloc[0])
        resid = (cm - pm) - (F - float(K))
        tol = 0.5 * (float(c["spread"].iloc[0]) + float(p["spread"].iloc[0]))
        rows.append(dict(T=float(T), K=float(K), resid=resid, tol=float(tol),
                         resid_px_bp=1e4 * resid / F))
        if abs(resid) > tol:
            outside += 1
    res = np.array([r["resid_px_bp"] for r in rows]) if rows else np.array([])
    print(f"  [raw chain] parity  : {len(rows)} strikes with both legs quoted; "
          f"{outside} outside the quoted spread")
    if len(res):
        print(f"                        residual RMS {np.sqrt(np.mean(res**2)):.1f} "
              f"px bp, max |resid| {np.abs(res).max():.1f} px bp")
    return dict(n_pairs=len(rows), n_outside_spread=outside,
                rms_resid_px_bp=(float(np.sqrt(np.mean(res ** 2)))
                                 if len(res) else None),
                max_abs_resid_px_bp=(float(np.abs(res).max())
                                     if len(res) else None))


def data_quality(df, fitted):
    """Counts the manuscript must disclose (open interest is NOT available)."""
    raw = df.rename(columns={"strike": "K"})
    m = fitted.merge(raw[["T", "K", "type", "bid", "ask", "volume"]],
                     on=["T", "K", "type"], how="left")
    two_sided = (m["bid"] > 0) & (m["ask"] > m["bid"])
    q = dict(
        n_raw_instruments=int(len(df)),
        n_fitted=int(len(m)),
        n_two_sided=int(two_sided.sum()),
        n_mark_fallback=int((~two_sided).sum()),
        n_zero_volume=int((m["volume"].fillna(0) == 0).sum()),
        open_interest_available=False,
        moneyness_min=float((fitted["K"] / fitted["F"]).min()),
        moneyness_max=float((fitted["K"] / fitted["F"]).max()),
        n_maturities=int(fitted["T"].nunique()),
    )
    print(f"  [data]  {q['n_fitted']} fitted pts | "
          f"{q['n_mark_fallback']} on mark price | "
          f"{q['n_zero_volume']} zero volume | "
          f"K/F in [{q['moneyness_min']:.3f}, {q['moneyness_max']:.3f}]")
    print("          open interest: NOT returned by the Deribit book-summary "
          "endpoint")
    return q


def model_surface(fitted, cal):
    """The calibrated model's own surface, for the same three tests."""
    from calibrate import _model_ivs
    vec = [cal["H"], cal["V0"], cal["theta"], cal["lam"], cal["nu"], cal["rho"]]
    ivs = _model_ivs(vec, fitted, R)
    out = fitted.copy()
    out["iv"] = ivs
    return out[np.isfinite(out["iv"])].reset_index(drop=True)


if __name__ == "__main__":
    df = pd.read_csv("data/deribit_chain.csv")
    cal = json.load(open("calib_real.json"))

    from generate_calib import cap_points_per_maturity
    full = build_surface(df, r=R)
    fitted = cap_points_per_maturity(full)

    print("Data quality")
    dq = data_quality(df, fitted)

    print("\nStatic arbitrage -- MARKET surface (the calibration target)")
    mkt = dict(butterfly=butterfly(fitted, "market"),
               vertical=vertical(fitted, "market"),
               calendar=calendar(fitted, "market"))
    par = parity_on_raw(df)

    print("\nStatic arbitrage -- CALIBRATED MODEL surface")
    mod_surf = model_surface(fitted, cal)
    mod = dict(butterfly=butterfly(mod_surf, "model"),
               vertical=vertical(mod_surf, "model"),
               calendar=calendar(mod_surf, "model"))

    print("\nStatic arbitrage -- MARKET surface, liquid core |k| <= 0.10")
    core = fitted[np.abs(fitted["k"]) <= 0.10].reset_index(drop=True)
    print(f"  core has {len(core)} of {len(fitted)} points, "
          f"{core['T'].nunique()} maturities")
    corr = dict(n_points=int(len(core)),
                n_maturities=int(core["T"].nunique()),
                butterfly=butterfly(core, "core"),
                vertical=vertical(core, "core"),
                calendar=calendar(core, "core"))

    dump("arbitrage", dict(data_quality=dq, market=mkt, model=mod,
                           parity_raw=par, liquid_core=corr,
                           calibrated_params=cal))
