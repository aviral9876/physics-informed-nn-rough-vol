"""
Regenerate calib_real.json - rough-Heston fit to the real Deribit BTC surface.

CLAUDE.md references calib_real.json (calibrated params on real BTC) but the file
was missing from the repo, so this reconstructs it from data/deribit_chain.csv
using the validated Fourier-in-the-loop calibrator. r=0 (crypto convention).
"""
import json
import numpy as np
import pandas as pd

from surface import build_surface
from calibrate import calibrate

if __name__ == "__main__":
    df = pd.read_csv("data/deribit_chain.csv")
    surf = build_surface(df, r=0.0)
    print(f"[calib] surface points={len(surf)}  "
          f"maturities={sorted(surf['T'].round(4).unique().tolist())}")
    params, diag = calibrate(surf, r=0.0, maxiter=20, popsize=8, seed=1,
                             verbose=True)
    out = {**params,
           "r": 0.0,
           "calib_rmse_vol_bp": diag["rmse_vol_bp"],
           "n_points": diag["n_points"],
           "spot": float(surf["spot"].iloc[0]),
           "maturities": sorted(surf["T"].round(6).unique().tolist())}
    json.dump(out, open("calib_real.json", "w"), indent=2)
    print("saved -> calib_real.json")
    print(json.dumps(out, indent=2))
