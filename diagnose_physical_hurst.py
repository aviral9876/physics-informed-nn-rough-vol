r"""
Hurst exponent under the PHYSICAL measure, from realised volatility. (R1.5)
===========================================================================

The referee's item 5 has two halves. The option-implied half -- recalibrating on
pre- and post-ETF surfaces -- is diagnose_regime_hurst.py. This is the other
half, and it is worth doing on its own merits: it estimates H from the realised
volatility of the BTC price series alone, with NO option data and NO calibration.

That independence is the point. Our option-implied H comes from a six-parameter
inverse problem whose identifiability we measure separately (see
diagnose_calib_uncertainty.py, which finds the profile in H far from sharp). An
estimate that never touches the surface is therefore not a nicety -- it is the
only way to tell whether H~0.09 is a property of Bitcoin or an artefact of the
calibration's flat directions.

METHOD (Gatheral, Jaisson & Rosenbaum 2018; applied to BTC by Takaishi 2020).
Build daily realised variance from 5-minute log returns, take sigma_t = sqrt(RV_t),
and form the q-th absolute moment of log-volatility increments at lag Delta,

    m(q, Delta) = mean over t of |log sigma_{t+Delta} - log sigma_t|^q.

For a fractional process, m(q, Delta) ~ Delta^{q H}. So for each q we regress
log m on log Delta to get a slope zeta(q), then regress zeta(q) on q through the
origin; that second slope is H. Rough volatility is H < 0.5, and the striking
empirical fact this literature reports is H ~ 0.1 across essentially every liquid
asset.

DATA. C:\Data\Options\data\bars\BTC-perp-5min.parquet -- 841k five-minute bars of
the Deribit BTC perpetual, 2018-08 to 2026-08 (author's own public-API collection).
Crypto trades continuously, so a day is a clean 288 bars with no overnight gap,
which removes the usual main headache in realised-variance estimation.
"""
import numpy as np, pandas as pd

BARS = r"C:\Data\Options\data\bars\BTC-perp-5min.parquet"
QS = [0.5, 1.0, 1.5, 2.0, 3.0]
LAGS = [1, 2, 3, 4, 5, 7, 10, 14, 20, 30, 45]
MIN_BARS_PER_DAY = 240          # of 288; drop days with a data outage


def realised_vol(path=BARS, start=None, end=None):
    df = pd.read_parquet(path, columns=["ts", "close"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    if start is not None:
        df = df[df["ts"] >= pd.Timestamp(start, tz="UTC")]
    if end is not None:
        df = df[df["ts"] <= pd.Timestamp(end, tz="UTC")]
    df = df.sort_values("ts")
    df["r"] = np.log(df["close"]).diff()
    df["day"] = df["ts"].dt.floor("D")
    g = df.dropna(subset=["r"]).groupby("day")["r"]
    rv = g.apply(lambda x: float(np.sum(x.values ** 2)))
    cnt = g.size()
    rv = rv[(cnt >= MIN_BARS_PER_DAY) & (rv > 0)]
    return np.sqrt(rv * 288.0)          # daily-scaled volatility


def hurst(sig, qs=QS, lags=LAGS):
    """Return H, the per-q slopes, and the raw moment surface."""
    ls = np.log(sig.values)
    rows, zetas = [], []
    for q in qs:
        xs, ys = [], []
        for d in lags:
            if d >= len(ls) - 5:
                continue
            inc = np.abs(ls[d:] - ls[:-d]) ** q
            m = float(np.mean(inc))
            if m > 0:
                xs.append(np.log(d)); ys.append(np.log(m))
                rows.append(dict(q=float(q), lag=int(d), log_m=float(np.log(m))))
        A = np.vstack([np.ones(len(xs)), np.array(xs)]).T
        beta, *_ = np.linalg.lstsq(A, np.array(ys), rcond=None)
        zetas.append(float(beta[1]))
    qs_a, z_a = np.array(qs, float), np.array(zetas)
    H = float(np.sum(qs_a * z_a) / np.sum(qs_a ** 2))     # through the origin
    resid = z_a - H * qs_a
    dof = max(len(qs_a) - 1, 1)
    se = float(np.sqrt(np.sum(resid ** 2) / dof / np.sum(qs_a ** 2)))
    return H, se, dict(zip([str(q) for q in qs], zetas)), rows


def boot_hurst(sig, n_boot=400, block=60, seed=0):
    """Moving-block bootstrap CI for H.

    The analytic SE from the zeta(q)-on-q regression is badly optimistic: it
    propagates only the scatter of the five slopes about the fitted line, and
    ignores both the sampling error in each moment m(q,Delta) and the very strong
    persistence of log-volatility. On our data it returns an SE near 1e-4 for a
    window whose point estimate sits 0.02 away from its neighbour's, which is
    self-evidently too small. Resampling contiguous blocks of days preserves the
    persistence and gives an interval worth quoting.
    """
    rng = np.random.default_rng(seed)
    v = sig.values
    n = len(v)
    nb = int(np.ceil(n / block))
    out = []
    for _ in range(n_boot):
        st = rng.integers(0, max(n - block, 1), nb)
        samp = np.concatenate([v[i:i + block] for i in st])[:n]
        try:
            H, _, _, _ = hurst(pd.Series(samp))
            if np.isfinite(H):
                out.append(H)
        except Exception:
            pass
    a = np.array(out)
    return (float(a.std(ddof=1)),
            [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))])


if __name__ == "__main__":
    windows = [("full sample", None, None),
               ("pre-ETF 2023", "2023-01-01", "2023-12-31"),
               ("post-ETF 2024", "2024-02-01", "2024-12-31"),
               ("2025-2026", "2025-01-01", None)]
    out = {}
    print(f"{'window':16s}{'days':>7}{'H':>9}{'bootSE':>9}   95% CI (block bootstrap)")
    for lbl, a, b in windows:
        sig = realised_vol(start=a, end=b)
        if len(sig) < 120:
            print(f"{lbl:16s}{len(sig):>7}   too short, skipped"); continue
        H, se, zet, rows = hurst(sig)
        bse, bci = boot_hurst(sig)
        out[lbl] = dict(n_days=int(len(sig)), H=H, H_se_analytic=se,
                        H_se_boot=bse, H_ci95_boot=bci, zetas=zet,
                        start=str(sig.index.min().date()), end=str(sig.index.max().date()),
                        mean_ann_vol=float(np.mean(sig.values) * np.sqrt(365)))
        print(f"{lbl:16s}{len(sig):>7}{H:>9.4f}{bse:>9.4f}   "
              f"[{bci[0]:.3f}, {bci[1]:.3f}]   (analytic SE {se:.4f}, optimistic)")

    f = out.get("full sample")
    if f:
        lo, hi = f["H_ci95_boot"]
        print("")
        print(f"full-sample H = {f['H']:.4f}  95% CI [{lo:.3f}, {hi:.3f}]  "
              f"({f['n_days']} days, {f['start']} to {f['end']})")
        print(f"  rough (upper CI < 0.5): {hi < 0.5}")
        print("  NOTE: this estimator is biased low on short windows, so the")
        print("  full sample is not comparable to the regime windows; only the")
        print("  two near-equal-length regime windows are compared below.")
    if "pre-ETF 2023" in out and "post-ETF 2024" in out:
        a, b = out["pre-ETF 2023"], out["post-ETF 2024"]
        d = a["H"] - b["H"]; sd = float(np.hypot(a["H_se_boot"], b["H_se_boot"]))
        print("")
        print(f"pre-ETF  H = {a['H']:.4f}  95% CI [{a['H_ci95_boot'][0]:.3f}, "
              f"{a['H_ci95_boot'][1]:.3f}]  ({a['n_days']} days)")
        print(f"post-ETF H = {b['H']:.4f}  95% CI [{b['H_ci95_boot'][0]:.3f}, "
              f"{b['H_ci95_boot'][1]:.3f}]  ({b['n_days']} days)")
        print(f"  difference {d:+.4f}, combined bootstrap SE {sd:.4f}, z = {d/sd:+.2f}")
        stable = abs(d) <= 2 * sd
        print(f"  => physical-measure H is {'STABLE' if stable else 'DIFFERENT'} "
              f"across the regimes")
        out["regime_test"] = dict(diff=float(d), combined_se=sd, z=float(d/sd),
                                  stable=bool(stable))
    from resultio import dump
    dump("physical_hurst", dict(
        source=BARS, method="Gatheral-Jaisson-Rosenbaum moment scaling on daily RV",
        qs=QS, lags=LAGS, min_bars_per_day=MIN_BARS_PER_DAY, windows=out))
    print("\nsaved -> results/physical_hurst.json")
