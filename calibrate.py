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
from functools import partial
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


IV_FLOOR, IV_CEIL = 1e-4, 5.0     # the Brent bracket in surface.implied_vol


def _model_ivs(params_vec, surf, r):
    """Model implied vols at every surface point for one parameter set.

    Two things here were wrong in the submitted version and are corrected for
    the revision. (1) The Lewis integrand decays like exp(-V T u^2 / 2) at short
    maturity, so the truncation frequency has to scale with 1/sqrt(V T); a fixed
    u_max = 100 mis-prices options of a few days badly (IV 0.64 where the fine
    inversion gives 0.31 at T = 0.002). (2) A point the model prices at or below
    intrinsic returned NaN and was silently DROPPED by the objective, which let
    the optimiser lower the loss by moving to parameters where badly-fit points
    vanished. Such a point is now floored at the Brent bracket's lower bound, so
    its full error counts; a genuine pricer failure stays NaN and the caller
    treats the vector as infeasible.
    """
    H, V0, theta, lam, nu, rho = params_vec
    p = dict(V0=V0, theta=theta, lam=lam, nu=nu, rho=rho)
    ivs = np.full(len(surf), np.nan)
    # group by maturity so each Fourier solve prices a whole smile at once
    for T, g in surf.groupby("T"):
        idx = g.index.values
        spot = float(g["spot"].iloc[0])
        Ks = g["K"].values.astype(float)
        u_max = float(np.clip(8.0 / np.sqrt(max(V0, 1e-4) * T), 100.0, 1000.0))
        n_u = int(max(800, 6 * u_max))
        try:
            prices = price_european_fourier(spot, Ks, T, r, p, H,
                                            N=120, u_max=u_max, n_u=n_u,
                                            option="call")
            prices = np.atleast_1d(prices)
        except Exception:
            continue                                   # NaN -> infeasible
        for ii, K, px in zip(idx, Ks, prices):
            if not np.isfinite(px):
                continue                               # NaN -> infeasible
            iv = implied_vol(px, spot, K, T, r, "C")
            if not np.isfinite(iv):
                intrinsic = max(spot - K * np.exp(-r * T), 0.0)
                iv = IV_FLOOR if px <= intrinsic + 1e-10 else IV_CEIL
            ivs[ii] = iv
    return ivs


def _loss_vec(vec, surf, r, wnorm, mkt):
    """Vega-weighted IV RMSE; module-level so it can be pickled to DE workers."""
    model = _model_ivs(vec, surf, r)
    if not np.all(np.isfinite(model)):
        return 1e3                        # pricer failure: infeasible, never masked
    err = model - mkt
    return float(np.sqrt(np.sum(wnorm * err**2)))


def calibrate(surf, r=0.065, maxiter=25, popsize=12, seed=0, polish=True,
              verbose=True, bounded_polish=False, bounds=None, workers=1):
    """
    Calibrate rough Heston to the surface. Returns (params_dict, diagnostics).

    bounded_polish : confine the Nelder-Mead polish to the box. False reproduces
        the canonical calib_real.json, whose theta escaped it (see below).
    bounds : override the module-level BOUNDS. Used by refit_canonical.py to test
        whether the canonical optimum survives a box wide enough to contain it.
    """
    bnds = BOUNDS if bounds is None else bounds
    w = surf["weight"].values
    mkt = surf["iv"].values
    wnorm = w / w.sum()

    n_eval = {"k": 0}
    base = partial(_loss_vec, surf=surf, r=r, wnorm=wnorm, mkt=mkt)

    def loss(vec):
        n_eval["k"] += 1
        return base(vec)

    if verbose:
        print("Calibrating rough Heston (differential evolution, %d worker%s)..."
              % (workers, "" if workers == 1 else "s"))
    # workers > 1 evaluates a whole generation in parallel, which requires
    # deferred population updating; the evolutionary path then differs from
    # the serial (immediate-update) run at the same seed. Both are full-budget.
    result = differential_evolution(
        base if workers != 1 else loss, bnds, maxiter=maxiter, popsize=popsize, seed=seed,
        tol=1e-4, mutation=(0.5, 1.0), recombination=0.7,
        polish=False, disp=verbose, workers=workers,
        updating="deferred" if workers != 1 else "immediate")
    if workers != 1:
        n_eval["k"] = int(result.nfev)

    best = result.x
    best_loss = float(result.fun)
    if polish:
        if verbose:
            print("Local polish (Nelder-Mead)...")
        # NOTE: historically this polish ran UNBOUNDED, and on the canonical BTC
        # fit it walked theta to 0.2525 -- outside the (0.005, 0.20) box that the
        # global search actually explored. The canonical calib_real.json was
        # produced that way and is kept for continuity (the escape improves the
        # fit, and theta ~ 0.25 is economically reasonable for crypto, where the
        # long-run vol level is well above 45%). bounded_polish=True confines the
        # polish to BOUNDS; the manuscript discloses which was used.
        loc = minimize(loss, best, method="Nelder-Mead",
                       bounds=(bnds if bounded_polish else None),
                       options=dict(maxiter=200, xatol=1e-4, fatol=1e-6))
        if loc.fun < best_loss:
            best = loc.x
            best_loss = float(loc.fun)

    params = dict(zip(PARAM_NAMES, best))
    model = _model_ivs(best, surf, r)
    mask = np.isfinite(model)
    rmse_bp = 1e4*np.sqrt(np.mean((model[mask]-mkt[mask])**2))
    diagnostics = dict(rmse_vol_bp=float(rmse_bp),
                       n_points=int(mask.sum()),
                       n_evals=n_eval["k"],
                       final_loss=best_loss,
                       de_loss=float(result.fun),
                       bounded_polish=bool(bounded_polish),
                       bounds=[list(b) for b in bnds],
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
