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
import json, os, time
import numpy as np, pandas as pd
from scipy.optimize import minimize

from surface import build_surface
from generate_calib import cap_points_per_maturity
from calibrate import _model_ivs, PARAM_NAMES
from refit_canonical import WIDE

# The first run was still falling at H=0.22, so the grid now runs to the edge of
# the wide box (0.45) to find where the profile actually turns.
H_GRID = [0.04, 0.06, 0.09, 0.12, 0.15, 0.18, 0.22, 0.26, 0.30, 0.35, 0.40, 0.45]
RHO_GRID = [-0.95, -0.90, -0.85, -0.79, -0.70, -0.60, -0.45, -0.30, -0.15]
NM = dict(maxiter=300, xatol=1e-4, fatol=1e-7)
CKPT = "results/_profile_ckpt.jsonl"


def make_losses(surf, r=0.0):
    w = surf["weight"].values
    mkt = surf["iv"].values
    wn = w / w.sum()

    def both(vec):
        model = _model_ivs(np.asarray(vec, float), surf, r)
        if not np.all(np.isfinite(model)):
            return 1e3, 1e3               # pricer failure: infeasible, never masked
        err = model - mkt
        weighted = np.sqrt(np.sum(wn * err ** 2))
        unweighted = np.sqrt(np.mean(err ** 2))
        return float(weighted), float(unweighted)
    return both


def profile(surf, idx, grid, starts, both):
    """Fix parameter `idx` at each grid value; re-optimise the other five.

    `starts` is a list of warm-start vectors. Nelder-Mead is local, and the loss
    surface has (at least) two basins -- the canonical H~0.09 fit and the wide-box
    H~0.25 fit -- so a single start from the canonical point would report a
    spuriously high profile at large H. We run from every start and keep the
    best, which is what a profile likelihood is supposed to be.
    """
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
        best_v, best_w = None, np.inf
        for start in starts:
            res = minimize(obj, np.array(start, float)[free], method="Nelder-Mead",
                           bounds=list(zip(lo, hi)), options=NM)
            # obj() clips into the box before evaluating, so the loss NM minimised is
            # the loss at the CLIPPED point. Re-evaluating at the raw res.x would
            # report a different (and possibly worse) number than the optimiser found
            # -- which is exactly what happened on the first run, where the profile at
            # the canonical H came out above the canonical loss itself.
            v = np.array(start, float); v[idx] = g; v[free] = np.clip(res.x, lo, hi)
            wv = both(v)[0]
            if wv < best_w:
                best_v, best_w = v, wv
        v = best_v
        wl, ul = both(v)
        out.append(dict(fixed=float(g), weighted=wl, unweighted_vol_bp=1e4 * ul,
                        params={k: float(x) for k, x in zip(PARAM_NAMES, v)},
                        secs=time.time() - t0))
        print(f"    {PARAM_NAMES[idx]}={g:+.3f}  weighted={1e4*wl:7.1f}  "
              f"unweighted={1e4*ul:7.1f} vol bp  ({time.time()-t0:.0f}s)", flush=True)
        with open(CKPT, "a") as f:
            f.write(json.dumps(dict(param=PARAM_NAMES[idx], **out[-1])) + "\n")
    return out


if __name__ == "__main__":
    surf = cap_points_per_maturity(build_surface(pd.read_csv("data/deribit_chain.csv"), r=0.0))
    cal = json.load(open("calib_real.json"))
    start = [cal[k] for k in PARAM_NAMES]
    starts = [start]
    if os.path.exists("calib_real_wide.json"):
        wide = json.load(open("calib_real_wide.json"))
        starts.append([wide[k] for k in PARAM_NAMES])
        print(f"warm starts: canonical (H={cal['H']:.3f}) and wide-box (H={wide['H']:.3f})")
    both = make_losses(surf)
    w0, u0 = both(start)
    os.makedirs("results", exist_ok=True)
    open(CKPT, "w").close()
    print(f"canonical: weighted {1e4*w0:.1f}, unweighted {1e4*u0:.1f} vol bp "
          f"over {len(surf)} points\n", flush=True)

    print("  profile in H:", flush=True)
    prof_H = profile(surf, 0, H_GRID, starts, both)
    print("\n  profile in rho:", flush=True)
    prof_rho = profile(surf, 5, RHO_GRID, starts, both)

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
