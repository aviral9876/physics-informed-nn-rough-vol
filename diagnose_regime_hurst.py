r"""
Is the recovered Hurst exponent stable across market regimes? (Reviewer 1, item 5)
==================================================================================

The referee asked us to recalibrate on clearly separated periods before and after
the 11 January 2024 US spot-ETF approval and report whether H is stable. The plan
recorded this as item D1, NOT SOLVABLE: Deribit's public endpoint serves only live
snapshots and historical chains are a paid vendor product.

It is solvable. C:\Data\Options\data\surfaces\BTC_slices.parquet holds a daily BTC
IV panel from 2016-12-22 to 2026-08-13, built from the Deribit trade tape by the
sibling Options project (author's own collection, public API). See hist_surface.py
for the schema map and -- importantly -- for the quote-vs-trade provenance caveat
that the manuscript must disclose.

DESIGN CHOICES, and why
-----------------------
1. WEEKLY, NOT DAILY. 52 pre + 47 post surfaces is ample for a two-sample
   comparison and keeps the run overnight-sized. Daily would multiply cost by 5
   for a variance reduction that autocorrelation would mostly eat anyway.

2. EVERY FIT INDEPENDENT -- no sequential warm-starting. Warm-starting each date
   from the previous one is the cheap way to run a calibration time series, but it
   biases the estimate toward continuity, which is precisely the null we are
   testing. An identical fixed-budget global search on every date cannot smooth the
   transition it is being used to detect.

3. JANUARY 2024 EXCLUDED ENTIRELY. The approval week is the event, not a sample
   from either regime.

4. The comparison is reported as a Welch t-test AND a Mann-Whitney U, because H
   across dates is not obviously normal and the two tests fail differently. A
   block bootstrap over 4-week blocks handles the autocorrelation that both
   ignore.
"""
import json, os, sys, time
import numpy as np, pandas as pd
from multiprocessing import Pool

from hist_surface import load_panel, surface_for_date, trading_dates
from calibrate import calibrate
from refit_canonical import WIDE

PRE = ("2023-01-01", "2023-12-31")
POST = ("2024-02-01", "2024-12-31")
DE_MAXITER, DE_POPSIZE, DE_SEED = 10, 8, 1
CKPT = "results/_regime_ckpt.jsonl"


def fit_one(args):
    date, surf_records = args
    surf = pd.DataFrame(surf_records)
    t0 = time.time()
    try:
        p, d = calibrate(surf, r=0.0, maxiter=DE_MAXITER, popsize=DE_POPSIZE,
                         seed=DE_SEED, polish=True, verbose=False,
                         bounded_polish=True, bounds=WIDE)
    except Exception as e:
        return dict(date=str(date), ok=False, error=repr(e))
    return dict(date=str(date), ok=True, secs=time.time() - t0,
                n_points=int(d["n_points"]), n_maturities=int(surf["T"].nunique()),
                F=float(surf["F"].iloc[0]), rmse_vol_bp=float(d["rmse_vol_bp"]),
                **{k: float(v) for k, v in p.items()})


def build_jobs(panel, window):
    jobs = []
    for d in trading_dates(panel, *window):
        s = surface_for_date(panel, d)
        if s is not None:
            jobs.append((d.date(), s.to_dict("records")))
    return jobs


if __name__ == "__main__":
    nproc = int(os.environ.get("REGIME_WORKERS", "4"))
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    panel = load_panel()
    jobs = [("pre", j) for j in build_jobs(panel, PRE)] + \
           [("post", j) for j in build_jobs(panel, POST)]
    if limit:
        jobs = jobs[:limit]
    print(f"[regime] {len(jobs)} fits ({sum(r=='pre' for r,_ in jobs)} pre, "
          f"{sum(r=='post' for r,_ in jobs)} post) on {nproc} workers", flush=True)

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
                print(f"  {res['date']} {regime:4s} H={res['H']:.4f} rho={res['rho']:+.3f} "
                      f"nu={res['nu']:.3f} rmse={res['rmse_vol_bp']:.0f}bp "
                      f"({res['secs']:.0f}s)  [{len(rows)}/{len(jobs)}]", flush=True)
            else:
                print(f"  {res['date']} {regime:4s} FAILED {res['error'][:80]}", flush=True)
    print(f"[regime] {len(rows)} fits in {(time.time()-t0)/60:.1f} min", flush=True)

    df = pd.DataFrame([r for r in rows if r["ok"]])
    df.to_csv("results/regime_fits.csv", index=False)

    from scipy import stats
    a = df[df.regime == "pre"]["H"].values
    b = df[df.regime == "post"]["H"].values
    tt = stats.ttest_ind(a, b, equal_var=False)
    mw = stats.mannwhitneyu(a, b, alternative="two-sided")

    # 4-week block bootstrap on the difference in means: weekly H is persistent,
    # so an i.i.d. resample would understate the standard error.
    rng = np.random.default_rng(0)
    B, blk, diffs = 10000, 4, []
    for _ in range(B):
        def resample(x):
            nb = int(np.ceil(len(x) / blk))
            st = rng.integers(0, max(len(x) - blk, 1), nb)
            return np.concatenate([x[s:s + blk] for s in st])[:len(x)]
        diffs.append(resample(a).mean() - resample(b).mean())
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])

    print("\n==== H by regime ====")
    for lbl, x in [("pre-ETF (2023)", a), ("post-ETF (2024)", b)]:
        print(f"  {lbl:16s} n={len(x):3d}  H = {x.mean():.4f} +- {x.std(ddof=1):.4f}"
              f"  median {np.median(x):.4f}  [{x.min():.3f}, {x.max():.3f}]")
    print(f"  difference (pre - post) = {a.mean()-b.mean():+.4f}")
    print(f"  Welch t = {tt.statistic:+.3f}, p = {tt.pvalue:.4f}")
    print(f"  Mann-Whitney U p = {mw.pvalue:.4f}")
    print(f"  4-week block bootstrap 95% CI on the difference: [{lo:+.4f}, {hi:+.4f}]")
    print(f"  => H {'DIFFERS' if (lo>0 or hi<0) else 'is STABLE'} across the regimes "
          f"at the 5% level (bootstrap CI {'excludes' if (lo>0 or hi<0) else 'includes'} 0)")

    from resultio import dump
    dump("regime_hurst", dict(
        config=dict(pre=PRE, post=POST, de_maxiter=DE_MAXITER, de_popsize=DE_POPSIZE,
                    de_seed=DE_SEED, bounds=[list(x) for x in WIDE],
                    source="C:/Data/Options/data/surfaces/BTC_slices.parquet",
                    provenance="Deribit trade tape (author's own public-API collection)",
                    n_bootstrap=B, block_weeks=blk),
        fits=rows,
        summary=dict(
            pre=dict(n=len(a), H_mean=float(a.mean()), H_sd=float(a.std(ddof=1)),
                     H_median=float(np.median(a))),
            post=dict(n=len(b), H_mean=float(b.mean()), H_sd=float(b.std(ddof=1)),
                      H_median=float(np.median(b))),
            diff=float(a.mean() - b.mean()),
            welch_t=float(tt.statistic), welch_p=float(tt.pvalue),
            mannwhitney_p=float(mw.pvalue),
            boot_ci95=[float(lo), float(hi)],
            differs=bool(lo > 0 or hi < 0))))
    print("\nsaved -> results/regime_hurst.json, results/regime_fits.csv")
