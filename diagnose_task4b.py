"""
Task 4b: skew-mechanism diagnosis (NOT a width scale-up).
========================================================

The 4a decomposition showed a monotonic downward bias deepening toward the put
wing -- the signature of the rho*nu*V cross-derivative (skew) term being under-
represented. Two controlled tests isolate the cause, each retraining the CURRENT
architecture (width=64, depth=4) 20k iters, seed 0, on the canonical BTC calib
at T=0.15, decomposed error tail-averaged over the plateau (last 15% of iters):

  Test 1 -- is it the correlation term?  Retrain with rho=0 (other params fixed).
    If the bias becomes wing-symmetric, the net can fit a symmetric smile and the
    put-wing error is specifically the cross term under strong rho.
  Test 2 -- is it factor-tail coverage?  Keep rho=-0.79 but widen the LOW-variance
    factor tail to 3x std and raise the tail fraction 20%->35%. If the OTM-put
    bias shrinks materially, it is a sampling-coverage problem.

RESULT (signed bias bp, PINN-Fourier):
                       OTMputs    ATM   OTMcalls
  baseline (rho-0.79)   -175     -97      -32     <- strong tilt toward puts
  Test 1 (rho=0)        -140    -128     -135     <- tilt GONE, wing-symmetric
  Test 2 (asym tails)   -168    -117      -39     <- ~unchanged, RMSE 122.6 = base
CONCLUSION: the put-wing skew error IS the rho cross-derivative term (Test 1
removes the asymmetry), and is NOT factor-tail coverage (Test 2 doesn't help).
A separate uniform ~-100..-135 bp under-bias persists regardless of rho.
Do NOT scale width; target the cross-derivative term.

NOTE: the task referred to "rho=-0.94", but the canonical calib / 4a baseline is
rho=-0.792; this uses -0.792 so the comparison is apples-to-apples with 4a.
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
BASE = dict(V0=cal["V0"], theta=cal["theta"], lam=cal["lam"], nu=cal["nu"], rho=cal["rho"])
H = cal["H"]; r = cal.get("r", 0.0); T = 0.15; n = 4
c, x = lift_weights_geometric(H + 0.5, n)
kk = np.linspace(-0.18, 0.18, 13); K = np.exp(kk)
ITERS = 20000
buckets = {"OTM puts (k<-0.05)": kk < -0.05,
           "ATM (|k|<=0.05)":    np.abs(kk) <= 0.05,
           "OTM calls (k>0.05)": kk > 0.05}

def fourier_iv(params):
    fpx = np.atleast_1d(price_european_fourier(1.0, K, T, r, params, H, N=300, u_max=120, n_u=1500))
    return np.array([implied_vol(fpx[i], 1.0, K[i], T, r, "C") for i in range(len(K))])

def run(params, tag, tail_scale_low=None, tail_frac=0.20):
    ivf = fourier_iv(params); good = np.isfinite(ivf)
    torch.manual_seed(0); np.random.seed(0)
    p = RoughHestonPINN(params, (c, x), K=1.0, r=r, T=T, width=64, depth=4, x_halfwidth=1.2)
    u_mean, u_std, _, _ = simulate_factor_stats(1.0, T, params, (c, x), n_paths=2000, n_steps=300, seed=1)
    p.set_factor_sampling(u_mean, u_std, tail_frac=tail_frac, tail_scale=2.0,
                          tail_scale_low=tail_scale_low)
    def pinn_ivs(net):
        hom = K * net.price(1.0 / K, tau=T)
        return np.array([implied_vol(max(hom[i], 1e-8), 1.0, K[i], T, r, "C") for i in range(len(K))])
    stack = []
    def probe(it, net):
        if it >= 0.85 * ITERS:          # average smile over the plateau tail
            stack.append(pinn_ivs(net))
    p.train(iters=ITERS, n_col=2000, n_bnd=500, n_bc=400, lr=1e-3, log_every=500, on_log=probe)
    ivp = np.nanmean(np.array(stack), axis=0)          # stable (tail-averaged) smile
    err = 1e4 * (ivp - ivf)
    print(f"\n=== {tag}  (rho={params['rho']:+.3f}, tail_frac={tail_frac}, "
          f"tail_low={tail_scale_low}) ===")
    rows = {}
    for name, m in buckets.items():
        mm = m & good
        rm = np.sqrt(np.nanmean(err[mm] ** 2)); bias = np.nanmean(err[mm])
        rows[name] = (rm, bias)
        print(f"  {name:22s} n={mm.sum()}  RMSE={rm:6.1f} bp  bias={bias:+7.1f} bp")
    allrm = np.sqrt(np.nanmean(err[good] ** 2))
    print(f"  {'ALL':22s} n={good.sum()}  RMSE={allrm:6.1f} bp")
    return dict(tag=tag, ivf=ivf, ivp=ivp, good=good, rows=rows, allrmse=allrm)

R = {}
R["base"] = run(BASE, "BASELINE (4a repro)")
R["rho0"] = run({**BASE, "rho": 0.0}, "TEST 1: rho=0 (symmetric truth)")
R["asym"] = run(BASE, "TEST 2: asym tails (low side 3x std, tail_frac 35%)",
                tail_scale_low=3.0, tail_frac=0.35)

print("\n\n================ SUMMARY: signed bias by wing (bp) ================")
print(f"{'config':<34}{'OTMputs':>10}{'ATM':>9}{'OTMcalls':>10}{'ALLrmse':>10}")
for key in ["base", "rho0", "asym"]:
    d = R[key]; rw = d["rows"]
    print(f"{d['tag']:<34}"
          f"{rw['OTM puts (k<-0.05)'][1]:>+10.1f}"
          f"{rw['ATM (|k|<=0.05)'][1]:>+9.1f}"
          f"{rw['OTM calls (k>0.05)'][1]:>+10.1f}"
          f"{d['allrmse']:>10.1f}")

fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
for a, key, ttl in zip(ax, ["base", "rho0", "asym"],
                       ["baseline (rho=-0.79)", "Test 1: rho=0", "Test 2: asym tails"]):
    d = R[key]; g = d["good"]
    a.plot(kk[g], 100*d["ivf"][g], "o-", color="k", label="Fourier")
    a.plot(kk[g], 100*d["ivp"][g], "^--", color="tab:blue", label="PINN")
    for b in [-0.05, 0.05]:
        a.axvline(b, color="0.8", ls=":")
    a.set_title(ttl); a.set_xlabel("log-moneyness k"); a.set_ylabel("IV (%)")
    a.legend(); a.grid(alpha=.3)
fig.tight_layout(); fig.savefig("figures/pinn_task4b_skew.png", dpi=130)
print("\nsaved -> figures/pinn_task4b_skew.png")
