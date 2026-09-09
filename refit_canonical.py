r"""
Does the canonical fit survive a box wide enough to contain it?
===============================================================

The canonical calib_real.json was produced with an UNBOUNDED Nelder-Mead polish,
and theta walked to 0.2525 -- outside the (0.005, 0.20) box the global search
actually explored. Reporting a parameter outside its own stated search box is the
same class of defect as the three data-section errors we self-found (C2, C3, C5),
and a second-round referee would be right to flag it.

The fix is not to force theta back inside a box that was simply too tight: crypto
long-run variance really is high (theta = 0.2525 is a long-run vol of ~50%, which
is unremarkable for BTC). The fix is to widen the box to an economically honest
range and re-run the GLOBAL search inside it, with the polish bounded.

Two outcomes, both fine:
  * The optimum lands where it already is -> calib_real.json stands unchanged, the
    escape defect disappears, and no downstream number moves. (Expected: the
    unbounded polish already found this point, so it is a genuine local optimum.)
  * The optimum moves -> every downstream number moves, and that has to be known
    BEFORE the overnight jobs are re-run rather than after.

Writes calib_real_wide.json. It does NOT overwrite calib_real.json; the comparison
is made first.
"""
import json, time
import numpy as np, pandas as pd

from surface import build_surface
from calibrate import calibrate
from generate_calib import cap_points_per_maturity, MAX_PER_MAT, DE_MAXITER, DE_POPSIZE, SEED

# Widened box. Only theta and nu move; H, lam, rho already had room.
#   theta: 0.20 -> 0.45  (long-run vol up to ~67%, the range BTC actually trades)
#   V0   : 0.20 -> 0.45  (same scale as theta; V0=0.103 was interior anyway)
#   nu   : 1.5 -> 2.0    (headroom so vol-of-vol is not the binding constraint)
WIDE = [(0.02, 0.45),    # H
        (0.005, 0.45),   # V0
        (0.005, 0.45),   # theta
        (0.05, 3.0),     # lam
        (0.05, 2.0),     # nu
        (-0.95, -0.05)]  # rho

if __name__ == "__main__":
    df = pd.read_csv("data/deribit_chain.csv")
    surf = cap_points_per_maturity(build_surface(df, r=0.0))
    print(f"[refit] {len(surf)} points over {surf['T'].nunique()} maturities", flush=True)

    t0 = time.time()
    params, diag = calibrate(surf, r=0.0, maxiter=DE_MAXITER, popsize=DE_POPSIZE,
                             seed=SEED, polish=True, verbose=True,
                             bounded_polish=True, bounds=WIDE)
    secs = time.time() - t0

    old = json.load(open("calib_real.json"))
    print(f"\n[refit] done in {secs/60:.1f} min")
    print(f"{'param':8s}{'canonical':>12s}{'wide-box':>12s}{'delta':>12s}")
    for k in ["H", "V0", "theta", "lam", "nu", "rho"]:
        print(f"{k:8s}{old[k]:>12.4f}{params[k]:>12.4f}{params[k]-old[k]:>+12.4f}")
    print(f"{'rmse_bp':8s}{old['calib_rmse_vol_bp']:>12.1f}{diag['rmse_vol_bp']:>12.1f}"
          f"{diag['rmse_vol_bp']-old['calib_rmse_vol_bp']:>+12.1f}")

    out = {**params, "r": 0.0,
           "calib_rmse_vol_bp": diag["rmse_vol_bp"],
           "n_points": diag["n_points"],
           "n_maturities": int(surf["T"].nunique()),
           "max_points_per_maturity": MAX_PER_MAT,
           "de_maxiter": DE_MAXITER, "de_popsize": DE_POPSIZE,
           "bounded_polish": True, "bounds": [list(b) for b in WIDE],
           "spot": float(surf["spot"].iloc[0]),
           "maturities": sorted(surf["T"].round(6).unique().tolist()),
           "fit_seconds": secs}
    json.dump(out, open("calib_real_wide.json", "w"), indent=2)
    from resultio import dump
    dump("refit_wide_box", dict(canonical=old, wide=out,
                                deltas={k: float(params[k] - old[k])
                                        for k in ["H","V0","theta","lam","nu","rho"]}))
    print("\nsaved -> calib_real_wide.json  (calib_real.json untouched)")

# The corrected-objective fit pinned lambda at the WIDE box's upper bound of 3.0
# (2026-09-09), so the box was constraining the answer rather than containing it.
# WIDER raises lambda and theta; every other edge stayed interior.
WIDER = [(0.02, 0.45), (0.005, 0.60), (0.005, 0.60), (0.05, 12.0), (0.05, 2.0), (-0.95, -0.05)]
