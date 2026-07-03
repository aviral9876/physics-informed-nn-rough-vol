"""
Stage 3 - Calibrate rough Heston to the market IV surface.
==========================================================

We fit the rough Heston parameters (H, V0, theta, lam, nu, rho) so the model's
implied vols match the cleaned market surface. The model prices come from the
VALIDATED fractional-Riccati Fourier pricer (rough_heston_fourier.py); the loss
is vega-weighted squared IV error, which is the industry-standard objective
because traders care about vol points, not price.

WHY THIS PRICER IN THE LOOP
---------------------------
Calibration evaluates the model hundreds of times. The Fourier pricer costs
~0.4s for a whole smile and is essentially exact, so it is the right engine for
the optimiser. (Monte Carlo would be far too slow and noisy inside a loop; the
PINN is not yet trained. Once trained, the parametric PINN can REPLACE this
pricer for instant recalibration - that is one of the thesis's selling points.)

Optimiser: differential evolution for a robust global search over the small
6-D parameter box, then a local polish. H is bounded in (0.02, 0.5) to stay in
the rough regime.
"""

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize

from rough_heston_fourier import price_european_fourier
from surface import implied_vol


PARAM_NAMES = ["H", "V0", "theta", "lam", "nu", "rho"]
BOUNDS = [(0.02, 0.45),    # H  (rough regime)
          (0.005, 0.20),   # V0
          (0.005, 0.20),   # theta
          (0.05, 3.0),     # lam (mean reversion speed)
          (0.05, 1.5),     # nu  (vol of vol)
          (-0.95, -0.05)]  # rho (leverage, negative for equity)


def _model_ivs(params_vec, surf, r):
    """Compute model implied vols at every surface point for one parameter set."""
    H, V0, theta, lam, nu, rho = params_vec
    p = dict(V0=V0, theta=theta, lam=lam, nu=nu, rho=rho)
    ivs = np.full(len(surf), np.nan)
    # group by maturity so each Fourier solve prices a whole smile at once
    for T, g in surf.groupby("T"):
        idx = g.index.values
        spot = float(g["spot"].iloc[0])
        Ks = g["K"].values.astype(float)
        try:
            prices = price_european_fourier(spot, Ks, T, r, p, H,
                                            N=120, u_max=100, n_u=800,
                                            option="call")
            prices = np.atleast_1d(prices)
        except Exception:
            continue
        for j, (ii, K, px) in enumerate(zip(idx, Ks, prices)):
            # price is a CALL on forward measure; convert to IV
            iv = implied_vol(px, spot, K, T, r, "C")
            ivs[ii] = iv
    return ivs


def calibrate(surf, r=0.065, maxiter=25, popsize=12, seed=0, polish=True,
              verbose=True):
    """
    Calibrate rough Heston to the surface. Returns (params_dict, diagnostics).
    """
    w = surf["weight"].values
    mkt = surf["iv"].values
    wnorm = w / w.sum()

    n_eval = {"k": 0}

    def loss(vec):
        n_eval["k"] += 1
        model = _model_ivs(vec, surf, r)
        mask = np.isfinite(model)
        if mask.sum() < 0.5*len(surf):
            return 1e3
        err = (model[mask] - mkt[mask])
        val = np.sqrt(np.sum(wnorm[mask]*err**2) / wnorm[mask].sum())
        return val

    if verbose:
        print("Calibrating rough Heston (differential evolution)...")
    result = differential_evolution(
        loss, BOUNDS, maxiter=maxiter, popsize=popsize, seed=seed,
        tol=1e-4, mutation=(0.5, 1.0), recombination=0.7,
        polish=False, disp=verbose)

    best = result.x
    if polish:
        if verbose:
            print("Local polish (Nelder-Mead)...")
        loc = minimize(loss, best, method="Nelder-Mead",
                       options=dict(maxiter=200, xatol=1e-4, fatol=1e-6))
        if loc.fun < result.fun:
            best = loc.x

    params = dict(zip(PARAM_NAMES, best))
    model = _model_ivs(best, surf, r)
    mask = np.isfinite(model)
    rmse_bp = 1e4*np.sqrt(np.mean((model[mask]-mkt[mask])**2))
    diagnostics = dict(rmse_vol_bp=float(rmse_bp),
                       n_points=int(mask.sum()),
                       n_evals=n_eval["k"],
                       final_loss=float(result.fun),
                       model_iv=model, market_iv=mkt)
    if verbose:
        print(f"Done. Fit RMSE = {rmse_bp:.1f} vol bp over {mask.sum()} points.")
        for name, v in params.items():
            print(f"   {name:6s} = {v:.4f}")
    return params, diagnostics


if __name__ == "__main__":
    from data_nse import load_chain
    from surface import build_surface

    df, src = load_chain("NIFTY", allow_network=True)
    surf = build_surface(df)
    print(f"[calibrate] source={src}, {len(surf)} surface points")
    params, diag = calibrate(surf, maxiter=15, popsize=10, seed=1)
    pd.Series({**params, **{k: diag[k] for k in
              ['rmse_vol_bp', 'n_points', 'n_evals']}}
              ).to_csv("data/calibrated_params.csv")
    print("saved -> data/calibrated_params.csv")
