"""
Task 4a: training-time vs capacity diagnostic (BEFORE scaling the network).
===========================================================================

Trains the CURRENT architecture (width=64, depth=4) for 20k iters with the
Task-3 settings (adaptive weights, tau-curriculum, MC factor sampling) on the
canonical BTC calib at T=0.15, and asks whether the residual smile error is a
training-time problem or a capacity problem:

  - IV RMSE vs Fourier on a held-out smile grid at iters 2.5k/5k/10k/15k/20k;
  - final error decomposed by moneyness: OTM puts (k<-0.05), ATM (|k|<=0.05),
    OTM calls (k>0.05).

Finding (seed 0): the curve PLATEAUS by ~iter 10k (curriculum reaches full T at
10k) and is flat/noisy ~120 bp thereafter -- NOT training-time-limited, so 40k
would not help. But the error is NOT strike-uniform: a systematic under-skew
(negative bias everywhere, ~-175 bp on the OTM-put wing vs ~-29 bp on the OTM
calls). So it fails the strike-uniform precondition for a pure capacity limit;
the residual is a left-wing skew under-fit, which argues for skew-AWARE next
steps (wing-focused collocation/loss) rather than naive width/depth scaling.
"""
import json, numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rough_heston_lift import lift_weights_geometric
from rough_heston_fourier import price_european_fourier
from lifted_mc import simulate_factor_stats
from surface import implied_vol
from pinn import RoughHestonPINN

cal = json.load(open("calib_real.json"))
params = dict(V0=cal["V0"], theta=cal["theta"], lam=cal["lam"], nu=cal["nu"], rho=cal["rho"])
H = cal["H"]; r = cal.get("r", 0.0); T = 0.15; n = 4
c, x = lift_weights_geometric(H + 0.5, n)
print(f"calib H={H:.4f} V0={params['V0']:.4f} theta={params['theta']:.4f} "
      f"lam={params['lam']:.3f} nu={params['nu']:.3f} rho={params['rho']:.3f}  T={T}")

# denser held-out smile grid: k = log(K/F), F=1 (r=0, S0=1)
kk = np.linspace(-0.18, 0.18, 13)
K = np.exp(kk)
fpx = np.atleast_1d(price_european_fourier(1.0, K, T, r, params, H, N=300, u_max=120, n_u=1500))
ivf = np.array([implied_vol(fpx[i], 1.0, K[i], T, r, "C") for i in range(len(K))])
good = np.isfinite(ivf)
print(f"held-out grid: {len(K)} strikes k in [{kk.min():.2f},{kk.max():.2f}]  "
      f"({good.sum()} with finite Fourier IV)")

def pinn_ivs(p):
    hom = K * p.price(1.0 / K, tau=T)                 # homogeneity: C(1,K)=K*C(1/K,1)
    return np.array([implied_vol(max(hom[i], 1e-8), 1.0, K[i], T, r, "C")
                     for i in range(len(K))])

def rmse_bp(ivp, mask):
    return 1e4 * np.sqrt(np.nanmean((ivp[mask] - ivf[mask]) ** 2))

torch.manual_seed(0); np.random.seed(0)
p = RoughHestonPINN(params, (c, x), K=1.0, r=r, T=T, width=64, depth=4, x_halfwidth=1.2)
u_mean, u_std, _, _ = simulate_factor_stats(1.0, T, params, (c, x), n_paths=2000, n_steps=300, seed=1)
p.set_factor_sampling(u_mean, u_std)

ITERS = 20000
curve = []
def probe(it, net):
    curve.append((it, rmse_bp(pinn_ivs(net), good)))

p.train(iters=ITERS, n_col=2000, n_bnd=500, n_bc=400, lr=1e-3, log_every=500, on_log=probe)

curve = np.array(curve)
def at(target):
    j = np.argmin(np.abs(curve[:, 0] - target)); return curve[j, 0], curve[j, 1]
print("\n=== IV RMSE convergence (held-out smile grid) ===")
for tgt in [2500, 5000, 10000, 15000, 20000]:
    it, v = at(tgt)
    frac = min(1.0, it / (0.5 * ITERS)); taumax = T * (0.15 + 0.85 * frac)
    print(f"  iter {int(it):6d}  IV RMSE = {v:7.1f} bp   (tau_max={taumax:.3f} of T={T})")

# final decomposed error by moneyness bucket
ivp = pinn_ivs(p)
err = 1e4 * (ivp - ivf)
buckets = {"OTM puts (k<-0.05)": kk < -0.05,
           "ATM (|k|<=0.05)":    np.abs(kk) <= 0.05,
           "OTM calls (k>0.05)": kk > 0.05}
print("\n=== final decomposed smile error (signed bp: PINN-Fourier) ===")
for name, m in buckets.items():
    mm = m & good
    rm = np.sqrt(np.nanmean(err[mm] ** 2)); bias = np.nanmean(err[mm])
    print(f"  {name:22s} n={mm.sum()}  RMSE={rm:6.1f} bp  mean(bias)={bias:+6.1f} bp")
print(f"  {'ALL':22s} n={good.sum()}  RMSE={rmse_bp(ivp,good):6.1f} bp")

fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
ax[0].plot(curve[:, 0], curve[:, 1], "-o", ms=3)
for xc in [10000]:
    ax[0].axvline(xc, color="0.6", ls=":", label="curriculum reaches full T")
ax[0].set_xlabel("iteration"); ax[0].set_ylabel("held-out IV RMSE (bp)")
ax[0].set_title(f"Task 4a convergence (width=64, depth=4, MC factors, T={T})")
ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].plot(kk[good], 100 * ivf[good], "o-", color="k", label="Fourier truth")
ax[1].plot(kk[good], 100 * ivp[good], "^--", color="tab:blue", label="PINN @20k")
for b in [-0.05, 0.05]:
    ax[1].axvline(b, color="0.7", ls=":")
ax[1].set_xlabel("log-moneyness k"); ax[1].set_ylabel("IV (%)")
ax[1].set_title("Final smile & bucket boundaries"); ax[1].legend(); ax[1].grid(alpha=.3)
fig.tight_layout(); fig.savefig("figures/pinn_task4a_diagnostic.png", dpi=130)
print("\nsaved -> figures/pinn_task4a_diagnostic.png")
