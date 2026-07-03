"""
Stage 2 - Clean the chain and build the implied-volatility surface.
===================================================================

WHY THESE STEPS
---------------
Raw option chains are noisy: deep ITM options are illiquid and their quoted IVs
are unreliable; puts and calls of the same strike should give the same IV under
put-call parity, so we keep the more liquid OTM leg on each side; zero/absurd
IVs must be dropped. What the calibrator actually needs is a clean map

        (log-moneyness k = log(K/F),  maturity T)  ->  implied vol.

We recompute IV ourselves from mid prices with a robust Black-Scholes inverter
rather than trusting the exchange's field, so the target surface is internally
consistent with the pricer we calibrate.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq


def bs_price(S, K, T, sig, r, typ):
    if sig <= 0 or T <= 0:
        fwd = S - K*np.exp(-r*T)
        return max(fwd, 0.0) if typ == "C" else max(-fwd, 0.0)
    d1 = (np.log(S/K) + (r + 0.5*sig**2)*T) / (sig*np.sqrt(T))
    d2 = d1 - sig*np.sqrt(T)
    if typ == "C":
        return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)


def implied_vol(price, S, K, T, r, typ):
    """Robust BS implied vol via Brent; returns nan if no arbitrage-free root."""
    intrinsic = max(S - K*np.exp(-r*T), 0.0) if typ == "C" \
        else max(K*np.exp(-r*T) - S, 0.0)
    if price < intrinsic - 1e-8 or price <= 0:
        return np.nan
    try:
        return brentq(lambda s: bs_price(S, K, T, s, r, typ) - price,
                      1e-4, 5.0, maxiter=100, xtol=1e-8)
    except ValueError:
        return np.nan


def build_surface(df, r=0.0, min_price=1.0, moneyness_window=0.30):
    """
    Clean the chain and return a surface DataFrame with columns
        ['T','K','F','k','iv','mid','type','weight'].

    Works for any source with the common schema; the exchange's own IV column
    may be named 'iv_nse' or 'iv_mkt' (or absent) - we recompute IV ourselves
    from mid prices regardless, so it does not matter.

    Defaults: r=0 suits crypto (no clean risk-free/carry; use forward from
    quotes). moneyness_window widened to 0.30 because crypto smiles are wide.

    Rules:
      - mid = (bid+ask)/2 when both present and ask>bid, else ltp.
      - keep OTM legs: calls for K>=F, puts for K<F (more liquid, tighter).
      - drop |k| > moneyness_window (deep wings: unreliable).
      - drop mid < min_price (too cheap -> IV noise dominates).
      - recompute IV from mid; drop nan.
      - vega weight so ATM points (where IV is well-determined) count more.
    """
    df = df.copy()
    df["mid"] = np.where(
        (df["bid"] > 0) & (df["ask"] > df["bid"]),
        0.5*(df["bid"] + df["ask"]),
        df["ltp"])

    out = []
    for T, g in df.groupby("T"):
        spot = float(g["spot"].iloc[0])
        F = spot*np.exp(r*T)
        for _, row in g.iterrows():
            K = row["strike"]; typ = row["type"]
            k = np.log(K/F)
            # keep OTM leg only
            if (typ == "C" and K < F) or (typ == "P" and K >= F):
                continue
            if abs(k) > moneyness_window:
                continue
            mid = row["mid"]
            if not np.isfinite(mid) or mid < min_price:
                continue
            iv = implied_vol(mid, spot, K, T, r, typ)
            if not np.isfinite(iv) or iv < 0.01 or iv > 2.0:
                continue
            # BS vega for weighting
            d1 = (np.log(spot/K) + (r + 0.5*iv**2)*T)/(iv*np.sqrt(T))
            vega = spot*norm.pdf(d1)*np.sqrt(T)
            out.append(dict(T=T, K=K, F=F, k=k, iv=iv, mid=mid,
                            type=typ, spot=spot, weight=max(vega, 1e-6)))
    surf = pd.DataFrame(out).sort_values(["T", "k"]).reset_index(drop=True)
    return surf


if __name__ == "__main__":
    from data_nse import load_chain
    df, src = load_chain("NIFTY", allow_network=True)
    surf = build_surface(df)
    print(f"source={src}  raw={len(df)}  clean surface points={len(surf)}")
    for T, g in surf.groupby("T"):
        print(f"  T={T:.4f}  n={len(g):2d}  "
              f"ATM_iv~{g['iv'].iloc[len(g)//2]:.4f}  "
              f"k in [{g['k'].min():+.3f},{g['k'].max():+.3f}]")
    surf.to_csv("data/iv_surface.csv", index=False)
    print("saved -> data/iv_surface.csv")
