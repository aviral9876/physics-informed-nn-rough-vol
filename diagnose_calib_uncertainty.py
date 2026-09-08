r"""
How well is H actually identified? (Reviewer 2 item 11; bears on Reviewer 1 item 5)
==================================================================================

This diagnostic exists because of an accident. While sizing the budget for the
pre/post-ETF regime study we ran the SAME light optimiser budget on our own
headline snapshot, whose full-budget answer is known (H=0.0899). It returned
H=0.1741 -- nearly double -- at an unweighted RMSE of 495 vol bp against the
canonical 496. Two parameter vectors whose Hurst exponents differ by a factor of
two fit the same surface equally well.

If that is real, then "H = 0.090, firmly rough" is an overstatement of what one
option surface can pin down, and any pre/post-ETF comparison of H is measuring
optimiser noise unless the budget is large enough to resolve the flat direction.
Both consequences matter more than the number itself, so we measure it directly
rather than inferring it from two runs.

METHOD. Profile likelihood, which is the honest tool here: fix H on a grid, and
for each fixed H re-optimise the remaining five parameters against the true
vega-weighted objective. The resulting curve is the calibration's own statement
about how much the fit degrades when H is forced away from its optimum. A flat
profile means H is unidentified; a sharp one means it is pinned. We report the
same for rho, since the two are the parameters the paper's economic claims rest
on.

We report the profile in the WEIGHTED objective actually optimised and in the
UNWEIGHTED RMSE the paper quotes, because the asymmetry between them is itself
something the manuscript now discloses.
"""
import json, time
import numpy as np, pandas as pd
from scipy.optimize import minimize

from surface import build_surface
from generate_calib import cap_points_per_maturity
from calibrate import _model_ivs, PARAM_NAMES
from refit_canonical import WIDE

H_GRID = [0.04, 0.06, 0.09, 0.12, 0.15, 0.18, 0.22, 0.26, 0.30]
RHO_GRID = [-0.95, -0.90, -0.85, -0.79, -0.70, -0.60, -0.45, -0.30, -0.15]
NM = dict(maxiter=300, xatol=1e-4, fatol=1e-7)


def make_losses(surf, r=0.0):
    w = surf["weight"].values
    mkt = surf["iv"].values
    wn = w / w.sum()

    def both(vec):
        model = _model_ivs(np.asarray(vec, float), surf, r)
        m = np.isfinite(model)
        if m.sum() < 0.5 * len(surf):
            return 1e3, 1e3
        err = model[m] - mkt[m]
        weighted = np.sqrt(np.sum(wn[m] * err ** 2) / wn[m].sum())
        unweighted = np.sqrt(np.mean(err ** 2))
        return float(weighted), float(unweighted)
    return both


def profile(surf, idx, grid, start, both):
    """Fix parameter `idx` at each grid value; re-optimise the other five."""
    free = [i for i in range(6) if i != idx]
    lo = np.array([WIDE[i][0] for i in free]); hi = np.array([WIDE[i][1] for i in free])
    out = []
    for g in grid:
        def obj(free_vec):
            v = np.array(start, float)
            v[idx] = g
            v[free] = np.clip(free_vec, lo, hi)
            return both(v)[0]
        t0 = time.time()
        res = minimize(obj, np.array(start, float)[free], method="Nelder-Mead",
                       bounds=list(zip(lo, hi)), options=NM)
        v = np.array(start, float); v[idx] = g; v[free] = res.x
        wl, ul = both(v)
        out.append(dict(fixed=float(g), weighted=wl, unweighted_vol_bp=1e4 * ul,
                        params={k: float(x) for k, x in zip(PARAM_NAMES, v)},
                        secs=time.time() - t0))
        print(f"    {PARAM_NAMES[idx]}={g:+.3f}  weighted={1e4*wl:7.1f}  "
              f"unweighted={1e4*ul:7.1f} vol bp  ({time.time()-t0:.0f}s)", flush=True)
    return out


if __name__ == "__main__":
    surf = cap_points_per_maturity(build_surface(pd.read_csv("data/deribit_chain.csv"), r=0.0))
    cal = json.load(open("calib_real.json"))
    start = [cal[k] for k in PARAM_NAMES]
    both = make_losses(surf)
    w0, u0 = both(start)
    print(f"canonical: weighted {1e4*w0:.1f}, unweighted {1e4*u0:.1f} vol bp "
          f"over {len(surf)} points\n", flush=True)

    print("  profile in H:", flush=True)
    prof_H = profile(surf, 0, H_GRID, start, both)
    print("\n  profile in rho:", flush=True)
    prof_rho = profile(surf, 5, RHO_GRID, start, both)

    # A confidence set by the standard profile rule: keep every fixed value whose
    # re-optimised fit is within a tolerance of the best. We use 5% relative
    # degradation of the weighted objective -- deliberately strict, since the
    # quantity of interest is whether H is pinned at all.
    def interval(prof, tol=0.05):
        best = min(p["weighted"] for p in prof)
        ok = [p["fixed"] for p in prof if p["weighted"] <= best * (1 + tol)]
        return (min(ok), max(ok)), best

    (hlo, hhi), hbest = interval(prof_H)
    (rlo, rhi), rbest = interval(prof_rho)
    print(f"\n==== profile-likelihood intervals (5% degradation of the weighted objective) ====")
    print(f"  H   in [{hlo:.3f}, {hhi:.3f}]   (canonical {cal['H']:.4f})")
    print(f"  rho in [{rlo:.3f}, {rhi:.3f}]   (canonical {cal['rho']:.4f})")
    print(f"  H identified as ROUGH (whole interval < 0.5): {hhi < 0.5}")

    from resultio import dump
    dump("calib_uncertainty", dict(
        canonical=dict(params={k: cal[k] for k in PARAM_NAMES},
                       weighted_vol_bp=1e4 * w0, unweighted_vol_bp=1e4 * u0),
        n_points=int(len(surf)), n_maturities=int(surf["T"].nunique()),
        profile_H=prof_H, profile_rho=prof_rho,
        interval_H=[hlo, hhi], interval_rho=[rlo, rhi], tol_rel=0.05,
        nelder_mead=NM, bounds=[list(b) for b in WIDE]))
    print("\nsaved -> results/calib_uncertainty.json")
