r"""
Liquid-core refit: is the 496 vol bp fit -- and the recovered H -- a wing
artefact? (Reviewer 1, item 6)
=========================================================================

The referee asks whether the coarse fit is driven by illiquid far-wing quotes.
We refit on the liquid core only, |k| <= 0.10 (and, as an intermediate, 0.20),
and report fit quality and parameters against the full-surface fit.

Because the full-surface loss has a flat ridge in H (diagnose_calib_uncertainty),
a single refit would land at an arbitrary point on whatever ridge the core has.
So each restricted surface is fitted by Nelder-Mead from BOTH ends of the ridge
(canonical H~0.09 and wide-box H~0.25), and we also evaluate the two unrefitted
full-surface vectors on the core. That answers two separable questions:

  * Does removing the wings improve the fit?  Compare the refitted core loss
    with the full-surface vectors' loss on the same core points.
  * Does the core prefer one end of the ridge?  Compare the two refits' losses.

Runs in ~20 min on one core.
"""
import json, os, time
import numpy as np, pandas as pd
from scipy.optimize import minimize

from surface import build_surface
from generate_calib import cap_points_per_maturity
from calibrate import PARAM_NAMES
from refit_canonical import WIDE
from diagnose_calib_uncertainty import make_losses, NM

WINDOWS = [0.10, 0.20]


def fit(surf, start):
    both = make_losses(surf)
    lo = np.array([b[0] for b in WIDE]); hi = np.array([b[1] for b in WIDE])

    def obj(v):
        return both(np.clip(v, lo, hi))[0]

    t0 = time.time()
    res = minimize(obj, np.array(start, float), method="Nelder-Mead",
                   bounds=list(zip(lo, hi)), options=NM)
    v = np.clip(res.x, lo, hi)
    w, u = both(v)
    return dict(weighted_vol_bp=1e4 * w, unweighted_vol_bp=1e4 * u, nfev=int(res.nfev),
                secs=time.time() - t0, params={k: float(x) for k, x in zip(PARAM_NAMES, v)})


if __name__ == "__main__":
    full = cap_points_per_maturity(build_surface(pd.read_csv("data/deribit_chain.csv"), r=0.0))
    starts = {"canonical": json.load(open("calib_real.json"))}
    if os.path.exists("calib_real_wide.json"):
        starts["wide"] = json.load(open("calib_real_wide.json"))
    starts = {k: [float(v[p]) for p in PARAM_NAMES] for k, v in starts.items()}

    out = dict(full_n=int(len(full)), windows={})
    print(f"full surface: {len(full)} points, {full['T'].nunique()} maturities")
    for kmax in [None] + WINDOWS:
        surf = full if kmax is None else full[full["k"].abs() <= kmax].reset_index(drop=True)
        label = "full" if kmax is None else f"|k|<={kmax:.2f}"
        both = make_losses(surf)
        print(f"\n== {label}: {len(surf)} points, {surf['T'].nunique()} maturities ==", flush=True)
        block = dict(n_points=int(len(surf)), n_maturities=int(surf["T"].nunique()),
                     unrefitted={}, refitted={})
        for name, v in starts.items():
            w, u = both(v)
            block["unrefitted"][name] = dict(weighted_vol_bp=1e4 * w, unweighted_vol_bp=1e4 * u)
            print(f"  {name:9s} vector, no refit : weighted {1e4*w:6.1f}  unweighted {1e4*u:6.1f} vol bp", flush=True)
        if kmax is not None:
            for name, v in starts.items():
                f = fit(surf, v)
                block["refitted"][name] = f
                p = f["params"]
                print(f"  refit from {name:9s}: weighted {f['weighted_vol_bp']:6.1f}  unweighted {f['unweighted_vol_bp']:6.1f}"
                      f"  H={p['H']:.3f} rho={p['rho']:+.3f} nu={p['nu']:.3f}  ({f['secs']:.0f}s)", flush=True)
        out["windows"][label] = block

    from resultio import dump
    dump("liquid_core", dict(starts=starts, windows_k=WINDOWS, nelder_mead=NM, **out))
    print("\nsaved -> results/liquid_core.json")
