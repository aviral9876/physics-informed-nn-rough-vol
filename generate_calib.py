"""
Regenerate calib_real.json - rough-Heston fit to the real Deribit BTC surface.

This is the CANONICAL calibration: the real Deribit BTC chain
(data/deribit_chain.csv) fit with the validated Fourier-in-the-loop calibrator,
r=0 (crypto convention). Full budget: differential evolution over many
generations with a local Nelder-Mead polish, all 13 maturities, up to
MAX_PER_MAT points per maturity spread across moneyness (keeps both wings and
ATM so the skew is constrained, not just the level).
"""
import json
import numpy as np
import pandas as pd

from surface import build_surface
from calibrate import calibrate

MAX_PER_MAT = 8
DE_MAXITER = 60
DE_POPSIZE = 12
SEED = 1


def cap_points_per_maturity(surf, max_per_mat=MAX_PER_MAT):
    """Keep <= max_per_mat points per maturity, spread evenly across log-moneyness
    k (both wings + ATM) so every maturity contributes and the smile shape is
    constrained rather than the ATM level alone."""
    keep = []
    for _, g in surf.groupby("T"):
        g = g.sort_values("k")
        if len(g) <= max_per_mat:
            keep.append(g)
        else:
            idx = np.unique(np.linspace(0, len(g) - 1, max_per_mat).round().astype(int))
            keep.append(g.iloc[idx])
    return pd.concat(keep).sort_values(["T", "k"]).reset_index(drop=True)


if __name__ == "__main__":
    df = pd.read_csv("data/deribit_chain.csv")
    surf_full = build_surface(df, r=0.0)
    surf = cap_points_per_maturity(surf_full)
    print(f"[calib] surface {len(surf_full)} -> {len(surf)} points "
          f"(<= {MAX_PER_MAT}/maturity) over {surf['T'].nunique()} maturities")
    for T, g in surf.groupby("T"):
        print(f"   T={T:.4f}  n={len(g):2d}")

    params, diag = calibrate(surf, r=0.0, maxiter=DE_MAXITER, popsize=DE_POPSIZE,
                             seed=SEED, polish=True, verbose=True)
    out = {**params,
           "r": 0.0,
           "calib_rmse_vol_bp": diag["rmse_vol_bp"],
           "n_points": diag["n_points"],
           "n_maturities": int(surf["T"].nunique()),
           "max_points_per_maturity": MAX_PER_MAT,
           "de_maxiter": DE_MAXITER,
           "de_popsize": DE_POPSIZE,
           "spot": float(surf["spot"].iloc[0]),
           "maturities": sorted(surf["T"].round(6).unique().tolist())}
    json.dump(out, open("calib_real.json", "w"), indent=2)
    print("saved -> calib_real.json")
    print(json.dumps(out, indent=2))
