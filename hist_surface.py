r"""
Historical BTC IV surfaces -> our calibration schema.
=====================================================

Reviewer 1 item 5 asks whether the recovered Hurst exponent is stable across
market regimes, specifically before and after the 11 Jan 2024 US spot-ETF
approval. The revision plan recorded this as NOT SOLVABLE (item D1): the public
Deribit endpoint serves only live snapshots, and historical chains are a paid
vendor product.

That is no longer true. C:\Data\Options\data\surfaces\BTC_slices.parquet is a
daily BTC implied-volatility panel covering 2016-12-22 to 2026-08-13 (3,449
days), built from the Deribit trade tape (41.5M option trades) by the sibling
Options project. Its schema is nearly ours already:

    slices:  k, iv, T, F, weight, date, expiration_timestamp
    ours  :  T, K, F, k, iv, mid, type, spot, weight

PROVENANCE DIFFERENCE, and it must be disclosed in the manuscript. Our headline
snapshot is QUOTE-based: mid of the two-sided book where one exists, and Deribit's
mark price for the 11 points that lack one. This panel is TRADE-based: every point
is a completed transaction, so there is no bid/ask and the two-sided-quote filter
cannot be applied. Neither is strictly better -- trades are real prints but arrive
irregularly and can be stale; quotes are continuous but the mark is an exchange
model price. The regime comparison below is internally consistent (every surface
built the same way), which is what the comparison needs; it is NOT a like-for-like
extension of the headline snapshot, and the paper should not present it as one.
"""
import numpy as np
import pandas as pd

SLICES = r"C:\Data\Options\data\surfaces\BTC_slices.parquet"

# Filters chosen to match the headline pipeline's *effect*, not its mechanism.
# The headline fitted surface spans K/F in [0.77, 1.35], i.e. k in [-0.26, 0.30],
# and 13 maturities out to T~0.9. The panel carries k out to +-1.2, which is far
# into the illiquid wing where a single stale print sets the IV. K_ABS caps that.
K_ABS = 0.35
T_MIN, T_MAX = 0.02, 1.0
IV_LO, IV_HI = 0.01, 2.0
MAX_PER_MAT = 8
MIN_MATURITIES = 4
MIN_POINTS = 25


def load_panel(path=SLICES):
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_convert(None).dt.normalize()
    return df


def surface_for_date(panel, date, max_per_mat=MAX_PER_MAT):
    """One day of the panel -> a surface DataFrame that calibrate() accepts.

    Returns None when the day is too thin to carry a 6-parameter fit.
    """
    g = panel[panel["date"] == pd.Timestamp(date)]
    if g.empty:
        return None
    g = g[(g["iv"] > IV_LO) & (g["iv"] < IV_HI)
          & (g["k"].abs() <= K_ABS)
          & (g["T"] >= T_MIN) & (g["T"] <= T_MAX)]
    if len(g) < MIN_POINTS or g["T"].nunique() < MIN_MATURITIES:
        return None

    out = g.copy()
    out["K"] = out["F"] * np.exp(out["k"])
    # r = 0 by the project's crypto convention, so the forward IS the spot.
    out["spot"] = out["F"]
    out["type"] = np.where(out["k"] >= 0.0, "C", "P")   # OTM leg, as in surface.py
    out["mid"] = np.nan                                  # not carried by the panel

    # Same even-in-k thinning as generate_calib.cap_points_per_maturity, so each
    # maturity contributes both wings and ATM rather than whatever traded most.
    keep = []
    for _, h in out.groupby("T"):
        h = h.sort_values("k")
        if len(h) > max_per_mat:
            idx = np.unique(np.linspace(0, len(h) - 1, max_per_mat).round().astype(int))
            h = h.iloc[idx]
        keep.append(h)
    surf = pd.concat(keep).sort_values(["T", "k"]).reset_index(drop=True)
    return surf[["T", "K", "F", "k", "iv", "mid", "type", "spot", "weight"]]


def trading_dates(panel, start, end, freq="W-WED"):
    """Panel dates closest to each grid point in [start, end]."""
    have = pd.DatetimeIndex(sorted(pd.to_datetime(panel["date"].unique())))
    have = have[(have >= pd.Timestamp(start)) & (have <= pd.Timestamp(end))]
    if freq is None:
        return list(have)
    have = have.values
    grid = pd.date_range(start, end, freq=freq)
    picked = []
    for t in grid:
        d = have[np.abs(have - np.datetime64(t)) <= np.timedelta64(3, "D")]
        if len(d):
            picked.append(pd.Timestamp(d[np.argmin(np.abs(d - np.datetime64(t)))]))
    return sorted(set(picked))


if __name__ == "__main__":
    p = load_panel()
    print(f"panel: {len(p):,} rows, {p['date'].nunique():,} days, "
          f"{p['date'].min().date()} -> {p['date'].max().date()}")
    for lbl, a, b in [("pre-ETF", "2023-01-01", "2023-12-31"),
                      ("post-ETF", "2024-02-01", "2024-12-31")]:
        ds = trading_dates(p, a, b)
        ok = [d for d in ds if surface_for_date(p, d) is not None]
        n = [len(surface_for_date(p, d)) for d in ok]
        m = [surface_for_date(p, d)["T"].nunique() for d in ok]
        print(f"{lbl:9s}: {len(ds)} weekly dates, {len(ok)} usable | "
              f"points/day med {np.median(n):.0f} [{min(n)}-{max(n)}] | "
              f"maturities med {np.median(m):.0f}")
