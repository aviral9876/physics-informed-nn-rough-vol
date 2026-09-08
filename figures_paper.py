"""
Regenerate the paper figures from COMMITTED results.
====================================================

Every number here now comes from results/<name>.json written by the script that
measured it (see resultio.py), NOT from a literal transcribed by hand. The
literals that used to sit here survive only as FALLBACKS, tagged with the commit
they came from, so the figures still build on a fresh checkout before the
experiments are re-run -- and so the de-hardcoding could be verified by
reproducing the committed PNGs before any new run changed the numbers.

Each build prints its provenance ("json:<name>" or "fallback"). If a panel says
fallback for an experiment you just ran, the dump did not land.

Fig 1 (n-convergence): results/n_convergence.json <- diagnose_robustness_sweep.py
Fig 2 (calibration fit): calib_real.json, priced by the Fourier pricer
Fig 3 (hybrid calibration): results/task5_hybrid.json <- demo_task5_hybrid.py
"""
import json, os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from surface import build_surface, implied_vol
from rough_heston_fourier import price_european_fourier
from generate_calib import cap_points_per_maturity   # same <=8 pts/mat as the fit

_PROV = {}


def from_json(name, extract, fallback, label):
    """Measured value from results/<name>.json, else the committed fallback.

    Any failure -- file missing, key renamed, run incomplete -- falls back rather
    than raising, because a figure drawn from the last good run beats no figure.
    """
    try:
        with open(os.path.join("results", name + ".json")) as fh:
            v = extract(json.load(fh))
        if v is None:
            raise ValueError("extractor returned None")
        _PROV[label] = "json:" + name
        return v
    except Exception as e:
        _PROV[label] = "fallback (%s)" % type(e).__name__
        return fallback


# ---------- Fig 1: n-convergence ----------
# Fallbacks: the 3-seed run at 433afb6 (+ c86a976 for train time).
_rows = from_json("n_convergence", lambda d: d["rows"], None, "fig1")
if _rows:
    n = np.array([r["n"] for r in _rows])
    solve_mean = np.array([r["solve_mean"] for r in _rows])
    solve_std = np.array([r["solve_std"] for r in _rows])
    lift = np.array([r["lift"] for r in _rows])
    nseed = len(_rows[0].get("solves", [])) or 3
else:
    n = np.array([4, 8, 16, 32])
    solve_mean = np.array([5.4, 6.1, 8.8, 10.7]); solve_std = np.array([1.0, 1.4, 1.6, 2.0])
    lift = np.array([16.4, 7.9, 3.5, 1.7]); nseed = 3
# train time comes from diagnose_n_convergence.py (single seed), not the sweep
tsec = from_json("n_cost", lambda d: np.array(d["train_seconds"]),
                 np.array([397, 386, 591, 457]), "fig1_time")
fd = 50.0 ** (n + 1)
fig, axL = plt.subplots(figsize=(8.4, 5.2))
hL, = axL.plot(n, lift, "o-", color="tab:red", lw=2, label="lift-error (MC vs Fourier)")
hS = axL.errorbar(n, solve_mean, yerr=solve_std, fmt="s-", color="tab:blue", lw=2,
                  capsize=4, label=f"PINN solve-error (vs lifted MC, {nseed}-seed)")
axL.set_yscale("log"); axL.set_xscale("log", base=2); axL.set_xticks(n); axL.set_xticklabels(n)
axL.set_xlabel("number of lift factors  $n$"); axL.set_ylabel("price RMSE (bp of spot, log)")
axL.grid(alpha=.3, which="both")
axR = axL.twinx(); hT, = axR.plot(n, tsec, "^--", color="tab:green", lw=1.4, label="PINN train time (s)")
axR.set_ylabel("wall-clock to plateau (s), CPU", color="tab:green")
axR.tick_params(axis="y", colors="tab:green"); axR.set_ylim(0, 800)
axL.text(0.03, 0.03, "naive FD grid $50^{(n+1)}$:\n" +
         "\n".join(f"  $n{{=}}{ni}$: {f:.0e}" for ni, f in zip(n, fd)) + "\n(infeasible by $n{=}8$)",
         transform=axL.transAxes, fontsize=8, va="bottom",
         bbox=dict(boxstyle="round", fc="wheat", alpha=.6))
axL.legend([hL, hS, hT], [h.get_label() for h in (hL, hS, hT)], loc="upper right", fontsize=9)
fig.tight_layout(); fig.savefig("figures/fig1_n_convergence.png", dpi=140); plt.close(fig)

# ---------- Fig 2: BTC calibration fit (committed calib: 1e3913f) ----------
cal = json.load(open("calib_real.json"))
p = dict(V0=cal["V0"], theta=cal["theta"], lam=cal["lam"], nu=cal["nu"], rho=cal["rho"]); H = cal["H"]
surf = cap_points_per_maturity(build_surface(pd.read_csv("data/deribit_chain.csv"), r=0.0))
per = {}                                       # T -> (k, mkt_iv, mdl_iv)
mk, mdl, mats = [], [], []
for T, g in surf.groupby("T"):
    kk = g["k"].values; K = np.exp(kk)
    px = np.atleast_1d(price_european_fourier(1.0, K, T, 0.0, p, H, N=200, u_max=100, n_u=1000))
    mi = np.array([implied_vol(px[i], 1.0, K[i], T, 0.0, "C") for i in range(len(K))])
    ok = np.isfinite(mi)
    per[T] = (kk[ok], g["iv"].values[ok], mi[ok])
    mk += list(g["iv"].values[ok]); mdl += list(mi[ok]); mats += [T] * ok.sum()
mk, mdl, mats = np.array(mk), np.array(mdl), np.array(mats)
rmse = 1e4 * np.sqrt(np.mean((mdl - mk) ** 2))
fig, (a0, a1) = plt.subplots(1, 2, figsize=(11.2, 4.9))
# (a) market vs model scatter
sc = a0.scatter(100 * mk, 100 * mdl, c=mats, cmap="viridis", s=20, alpha=.8)
lim = [100 * min(mk.min(), mdl.min()) - 2, 100 * max(mk.max(), mdl.max()) + 2]
a0.plot(lim, lim, "k--", lw=1, alpha=.6); a0.set_xlim(lim); a0.set_ylim(lim)
a0.set_xlabel("market IV (%)"); a0.set_ylabel("model IV (%)"); a0.grid(alpha=.3)
a0.set_title(f"(a) all points: RMSE {rmse:.0f} vol bp ({len(mk)} pts)")
plt.colorbar(sc, ax=a0, label="maturity T (yr)")
# (b) per-maturity smile overlay for 4 representative maturities
Ts = sorted(per); sel = [Ts[1], Ts[6], Ts[9], Ts[12]]
cols = plt.cm.viridis(np.linspace(0.05, 0.9, len(sel)))
for T, c in zip(sel, cols):
    kk, miv, dv = per[T]; o = np.argsort(kk)
    a1.plot(kk[o], 100 * miv[o], "o", color=c, ms=5, label=f"T={T:.3f}")
    a1.plot(kk[o], 100 * dv[o], "--", color=c, lw=1.6)
a1.set_xlabel("log-moneyness  k = log(K/F)"); a1.set_ylabel("implied vol (%)")
a1.set_title("(b) smiles: market (o) vs model (--)"); a1.legend(fontsize=8); a1.grid(alpha=.3)
fig.suptitle(f"BTC calibration fit  (H={H:.3f}, $\\rho$={cal['rho']:.2f})", y=1.02)
fig.tight_layout(); fig.savefig("figures/fig2_calibration_fit.png", dpi=140, bbox_inches="tight"); plt.close(fig)

# ---------- Fig 3: hybrid calibration (fallbacks committed at 8aba282) ----------
meth = ["Fourier\nfrom scratch", "PINN-only\n(single-shot)", "Hybrid\n(PINN+polish)"]
_h = from_json("task5_hybrid", lambda d: d["summary"], None, "fig3")
if _h:
    rmse3 = [_h["scratch"]["rmse"], _h["ai_only"]["rmse"], _h["hybrid"]["rmse"]]
    err3 = [_h["scratch"]["rmse_sd"], _h["ai_only"]["rmse_sd"], _h["hybrid"]["rmse_sd"]]
    evals3 = [_h["scratch"]["evals"], _h["ai_only"]["evals"], _h["hybrid"]["evals"]]
    ev_err = [_h["scratch"]["evals_sd"], _h["ai_only"]["evals_sd"], _h["hybrid"]["evals_sd"]]
else:
    rmse3 = [23.7, 222.0, 29.4]; err3 = [0, 20, 7.1]
    evals3 = [2018, 0, 237]; ev_err = [0, 0, 46]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.4))
cols = ["tab:gray", "tab:orange", "tab:blue"]
a1.bar(meth, rmse3, yerr=err3, color=cols, capsize=4)
a1.axhline(rmse3[0], color="k", ls=":", lw=1); a1.set_ylabel("fit RMSE (vol bp, true model)")
a1.set_title("Calibration quality"); a1.grid(axis="y", alpha=.3)
a2.bar(meth, evals3, yerr=ev_err, color=cols, capsize=4)
a2.set_ylabel("Fourier evaluations"); a2.set_title("Fourier-eval cost (0 for PINN)")
a2.grid(axis="y", alpha=.3)
fig.tight_layout(); fig.savefig("figures/fig3_hybrid.png", dpi=140); plt.close(fig)

print(f"saved fig1_n_convergence, fig2_calibration_fit (RMSE {rmse:.0f} bp, {len(mk)} pts), fig3_hybrid")
print("provenance:")
for _k, _v in _PROV.items():
    print("   %-12s %s" % (_k, _v))
