"""
Regenerate the paper figures from COMMITTED results.
====================================================

Fig 1 (n-convergence): 3-seed solve-error (diagnose_robustness_sweep.py, 433afb6)
   + lift-error and FD grid (diagnose_n_convergence.py, c86a976).
Fig 2 (calibration fit): market vs model IV at the canonical BTC calibration
   (calib_real.json, 1e3913f), model priced by the Fourier pricer.
Fig 3 (hybrid calibration): the three methods (demo_task5_hybrid.py, 8aba282).
Numbers below are transcribed from those committed artifacts (source tagged).
"""
import json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from surface import build_surface, implied_vol
from rough_heston_fourier import price_european_fourier
from generate_calib import cap_points_per_maturity   # same <=8 pts/mat as the fit

# ---------- Fig 1: n-convergence (committed: 433afb6 + c86a976) ----------
n = np.array([4, 8, 16, 32])
solve_mean = np.array([5.4, 6.1, 8.8, 10.7]); solve_std = np.array([1.0, 1.4, 1.6, 2.0])
lift = np.array([16.4, 7.9, 3.5, 1.7]); tsec = np.array([397, 386, 591, 457])
fd = 50.0 ** (n + 1)
fig, axL = plt.subplots(figsize=(8.4, 5.2))
hL, = axL.plot(n, lift, "o-", color="tab:red", lw=2, label="lift-error (MC vs Fourier)")
hS = axL.errorbar(n, solve_mean, yerr=solve_std, fmt="s-", color="tab:blue", lw=2,
                  capsize=4, label="PINN solve-error (vs lifted MC, 3-seed)")
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
mk, mdl, mats = [], [], []
for T, g in surf.groupby("T"):
    K = np.exp(g["k"].values)
    px = np.atleast_1d(price_european_fourier(1.0, K, T, 0.0, p, H, N=200, u_max=100, n_u=1000))
    for i in range(len(K)):
        iv = implied_vol(px[i], 1.0, K[i], T, 0.0, "C")
        if np.isfinite(iv):
            mk.append(g["iv"].values[i]); mdl.append(iv); mats.append(T)
mk, mdl, mats = np.array(mk), np.array(mdl), np.array(mats)
rmse = 1e4 * np.sqrt(np.mean((mdl - mk) ** 2))
fig, ax = plt.subplots(figsize=(6.2, 5.6))
sc = ax.scatter(100 * mk, 100 * mdl, c=mats, cmap="viridis", s=22, alpha=.8)
lim = [100 * min(mk.min(), mdl.min()) - 2, 100 * max(mk.max(), mdl.max()) + 2]
ax.plot(lim, lim, "k--", lw=1, alpha=.6); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("market IV (%)"); ax.set_ylabel("model IV (%)")
ax.set_title(f"BTC calibration fit  (H={H:.3f}, $\\rho$={cal['rho']:.2f};  "
             f"RMSE {rmse:.0f} vol bp, {len(mk)} pts)")
plt.colorbar(sc, label="maturity T (yr)"); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig("figures/fig2_calibration_fit.png", dpi=140); plt.close(fig)

# ---------- Fig 3: hybrid calibration (committed: 8aba282) ----------
meth = ["Fourier\nfrom scratch", "PINN-only\n(single-shot)", "Hybrid\n(PINN+polish)"]
rmse3 = [23.7, 222.0, 29.4]; err3 = [0, 20, 7.1]; evals3 = [2018, 0, 237]; ev_err = [0, 0, 46]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.4))
cols = ["tab:gray", "tab:orange", "tab:blue"]
a1.bar(meth, rmse3, yerr=err3, color=cols, capsize=4)
a1.axhline(23.7, color="k", ls=":", lw=1); a1.set_ylabel("fit RMSE (vol bp, true model)")
a1.set_title("Calibration quality"); a1.grid(axis="y", alpha=.3)
a2.bar(meth, evals3, yerr=ev_err, color=cols, capsize=4)
a2.set_ylabel("Fourier evaluations"); a2.set_title("Fourier-eval cost (0 for PINN)")
a2.grid(axis="y", alpha=.3)
fig.tight_layout(); fig.savefig("figures/fig3_hybrid.png", dpi=140); plt.close(fig)

print(f"saved fig1_n_convergence, fig2_calibration_fit (RMSE {rmse:.0f} bp, {len(mk)} pts), fig3_hybrid")
