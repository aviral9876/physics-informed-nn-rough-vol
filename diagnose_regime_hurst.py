r"""
Option-implied H before and after the January 2024 spot-ETF approval.
(Reviewer 1, item 5 -- the option-implied half; the physical half is
diagnose_physical_hurst.py)
=====================================================================

DATA. Weekly (Wednesday) BTC implied-vol surfaces built by hist_surface.py from
the author's own Deribit trade-tape panel: 52 pre-ETF weeks (calendar 2023) and
47 post-ETF weeks (February-December 2024). January 2024 is excluded entirely as
the event window. The panel is TRADE-based whereas the headline snapshot is
QUOTE-based; the comparison is internally consistent but not a like-for-like
extension of the headline fit, and the manuscript says so.

DESIGN, and why it changed. The first version of this script ran a light
differential-evolution fit on every week. Then the profile-likelihood study
(diagnose_calib_uncertainty.py) showed that a single surface does not pin H: the
loss has a long, nearly flat ridge from the canonical H~0.09 fit to the wide-box
H~0.25 fit. A light global optimiser lands at a seed-dependent point on that
ridge, so week-to-week variation in its H would be optimiser noise, not market
information, and a pre/post test on it would be meaningless.

So every week is now fitted TWICE by local Nelder-Mead, warm-started from the
two ends of the ridge -- the canonical vector and the wide-box vector -- with
the other five parameters free inside the wide box. Nothing is warm-started
from the previous week: sequential warm-starting biases toward continuity,
which is the null being tested. Each fit records H, rho, nu and the loss. We
then ask three questions of the pre/post split:

  1. Does H move, holding the basin fixed?  (H from each start, pre vs post)
  2. Does the market change WHICH basin it prefers?  The sign and size of
     loss(wide start) - loss(canonical start), pre vs post.
  3. Do the parameters the surface DOES identify move?  rho and nu, pre vs post.

Tests: Welch t, Mann-Whitney, and a 4-week block bootstrap on the difference in
means (weekly fits are persistent; an i.i.d. resample would understate the SE).
Every fit is checkpointed to results/_regime_ckpt.jsonl as it completes.

Usage:  REGIME_WORKERS=4 python diagnose_regime_hurst.py [limit]
"""
import json, os, sys, time
import numpy as np, pandas as pd
from multiprocessing import Pool
from scipy.optimize import minimize

from hist_surface import load_panel, surface_for_date, trading_dates
from calibrate import PARAM_NAMES
from refit_canonical import WIDE as HEADLINE_BOX

# The headline box binds on these surfaces, so the weekly fits get a wider one.
# The first full pass (86 weeks, 2026-09-09) showed rho CENSORED: 27% of weekly
# fits sat exactly on the -0.05 cap and none went below -0.250, i.e. the
# trade-tape surfaces want a correlation near zero and the box would not let
# them have it. Since rho is the parameter this surface actually identifies
# (the profile pins it to -0.45 on the headline snapshot while H is only
# bounded to [0.04, 0.18]), censoring it would destroy the one meaningful
# pre/post test. rho is therefore allowed to cross zero. H keeps its range so
# the two basins mean the same thing.
WIDE = [(0.02, 0.45), (0.005, 1.0), (0.005, 1.0), (0.05, 8.0), (0.05, 3.0), (-0.95, 0.50)]
from diagnose_calib_uncertainty import make_losses, NM

PRE = ("2023-01-01", "2023-12-31")
POST = ("2024-02-01", "2024-12-31")
CKPT = "results/_regime_ckpt.jsonl"
STARTS = dict(canonical="calib_real.json", wide="calib_real_wide.json")


def load_starts():
    out = {}
    for name, path in STARTS.items():
        if os.path.exists(path):
            d = json.load(open(path))
            out[name] = [float(d[k]) for k in PARAM_NAMES]
    return out


def fit_one(args):
    date, surf_records, starts = args
    surf = pd.DataFrame(surf_records)
    both = make_losses(surf)
    lo = np.array([b[0] for b in WIDE]); hi = np.array([b[1] for b in WIDE])
    t0 = time.time()
    fits = {}
    try:
        for name, start in starts.items():
            def obj(v):
                return both(np.clip(v, lo, hi))[0]
            res = minimize(obj, np.array(start, float), method="Nelder-Mead",
                           bounds=list(zip(lo, hi)), options=NM)
            v = np.clip(res.x, lo, hi)
            w, u = both(v)
            fits[name] = dict(weighted_vol_bp=1e4 * w, unweighted_vol_bp=1e4 * u,
                              nfev=int(res.nfev), **{k: float(x) for k, x in zip(PARAM_NAMES, v)})
    except Exception as e:
        return dict(date=str(date), ok=False, error=repr(e))
    return dict(date=str(date), ok=True, secs=time.time() - t0,
                n_points=int(len(surf)), n_maturities=int(surf["T"].nunique()),
                F=float(surf["F"].iloc[0]), fits=fits)


def build_jobs(panel, window, starts):
    jobs = []
    for d in trading_dates(panel, *window):
        s = surface_for_date(panel, d)
        if s is not None:
            jobs.append((d.date(), s.to_dict("records"), starts))
    return jobs


def compare(label, a, b, unit="", rng_seed=0, B=10000, blk=4):
    """Pre vs post: Welch, Mann-Whitney, 4-week block bootstrap on the mean difference."""
    from scipy import stats
    a, b = np.asarray(a, float), np.asarray(b, float)
    tt = stats.ttest_ind(a, b, equal_var=False)
    mw = stats.mannwhitneyu(a, b, alternative="two-sided")
    rng = np.random.default_rng(rng_seed)

    def resample(x):
        nb = int(np.ceil(len(x) / blk))
        st = rng.integers(0, max(len(x) - blk, 1), nb)
        return np.concatenate([x[s:s + blk] for s in st])[:len(x)]

    diffs = np.array([resample(a).mean() - resample(b).mean() for _ in range(B)])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    differs = bool(lo > 0 or hi < 0)
    print(f"\n  {label}")
    print(f"    pre  n={len(a):3d}  mean {a.mean():+.4f} +- {a.std(ddof=1):.4f}  median {np.median(a):+.4f}")
    print(f"    post n={len(b):3d}  mean {b.mean():+.4f} +- {b.std(ddof=1):.4f}  median {np.median(b):+.4f}")
    print(f"    pre - post = {a.mean()-b.mean():+.4f} {unit}   Welch p = {tt.pvalue:.4f}   "
          f"MW p = {mw.pvalue:.4f}   block-bootstrap 95% CI [{lo:+.4f}, {hi:+.4f}]"
          f"  -> {'DIFFERS' if differs else 'stable'}")
    return dict(n_pre=int(len(a)), n_post=int(len(b)),
                pre_mean=float(a.mean()), pre_sd=float(a.std(ddof=1)), pre_median=float(np.median(a)),
                post_mean=float(b.mean()), post_sd=float(b.std(ddof=1)), post_median=float(np.median(b)),
                diff=float(a.mean() - b.mean()), welch_t=float(tt.statistic), welch_p=float(tt.pvalue),
                mannwhitney_p=float(mw.pvalue), boot_ci95=[float(lo), float(hi)], differs=differs)


if __name__ == "__main__":
    nproc = int(os.environ.get("REGIME_WORKERS", "4"))
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    starts = load_starts()
    assert "canonical" in starts, "calib_real.json missing"
    print("warm starts: " + ", ".join(f"{k} (H={v[0]:.3f})" for k, v in starts.items()), flush=True)

    panel = load_panel()
    jobs = [("pre", j) for j in build_jobs(panel, PRE, starts)] + \
           [("post", j) for j in build_jobs(panel, POST, starts)]
    if limit:
        jobs = jobs[:limit]
    print(f"[regime] {len(jobs)} weeks ({sum(r=='pre' for r,_ in jobs)} pre, "
          f"{sum(r=='post' for r,_ in jobs)} post) x {len(starts)} starts on {nproc} workers", flush=True)

    os.makedirs("results", exist_ok=True)
    open(CKPT, "w").close()
    rows, t0 = [], time.time()
    with Pool(nproc) as pool:
        for (regime, _), res in zip(jobs, pool.imap(fit_one, [j for _, j in jobs])):
            res["regime"] = regime
            rows.append(res)
            with open(CKPT, "a") as fh:      # checkpoint every fit; J1 taught us
                fh.write(json.dumps(res) + "\n")
            if res["ok"]:
                f = res["fits"]
                msg = "  ".join(f"{k}: H={v['H']:.3f} rho={v['rho']:+.2f} nu={v['nu']:.2f} "
                                f"w={v['weighted_vol_bp']:.0f}" for k, v in f.items())
                print(f"  {res['date']} {regime:4s} {msg}  ({res['secs']:.0f}s) [{len(rows)}/{len(jobs)}]", flush=True)
            else:
                print(f"  {res['date']} {regime:4s} FAILED {res['error'][:80]}", flush=True)
    print(f"[regime] {len(rows)} weeks in {(time.time()-t0)/60:.1f} min", flush=True)

    ok = [r for r in rows if r["ok"]]
    flat = []
    for r in ok:
        row = dict(date=r["date"], regime=r["regime"], n_points=r["n_points"], F=r["F"])
        for k, v in r["fits"].items():
            row.update({f"{k}_{kk}": vv for kk, vv in v.items()})
        if "canonical" in r["fits"] and "wide" in r["fits"]:
            row["dloss_wide_minus_can"] = r["fits"]["wide"]["weighted_vol_bp"] - r["fits"]["canonical"]["weighted_vol_bp"]
        flat.append(row)
    df = pd.DataFrame(flat)
    df.to_csv("results/regime_fits.csv", index=False)
    pre, post = df[df.regime == "pre"], df[df.regime == "post"]

    print("\n==== pre-ETF (2023) vs post-ETF (Feb-Dec 2024) ====")
    summary = {}
    for name in starts:
        summary[f"H_{name}"] = compare(f"H, {name} start", pre[f"{name}_H"], post[f"{name}_H"])
    summary["rho_canonical"] = compare("rho, canonical start", pre["canonical_rho"], post["canonical_rho"])
    summary["nu_canonical"] = compare("nu, canonical start", pre["canonical_nu"], post["canonical_nu"])
    summary["fit_canonical"] = compare("weighted fit (vol bp), canonical start",
                                       pre["canonical_weighted_vol_bp"], post["canonical_weighted_vol_bp"], "vol bp")
    if "dloss_wide_minus_can" in df:
        summary["basin"] = compare("loss(wide start) - loss(canonical start), vol bp; <0 prefers the smoother basin",
                                   pre["dloss_wide_minus_can"], post["dloss_wide_minus_can"], "vol bp")
        summary["basin"]["frac_prefer_wide_pre"] = float((pre["dloss_wide_minus_can"] < 0).mean())
        summary["basin"]["frac_prefer_wide_post"] = float((post["dloss_wide_minus_can"] < 0).mean())
        print(f"    weeks preferring the wide (smoother-H) basin: pre {100*summary['basin']['frac_prefer_wide_pre']:.0f}%, "
              f"post {100*summary['basin']['frac_prefer_wide_post']:.0f}%")

    from resultio import dump
    dump("regime_hurst", dict(
        config=dict(pre=PRE, post=POST, starts=starts, nelder_mead=NM,
                    bounds=[list(x) for x in WIDE],
                    source="C:/Data/Options/data/surfaces/BTC_slices.parquet",
                    provenance="Deribit trade tape (author's own public-API collection)",
                    n_bootstrap=10000, block_weeks=4),
        fits=rows, summary=summary))
    print("\nsaved -> results/regime_hurst.json, results/regime_fits.csv")
