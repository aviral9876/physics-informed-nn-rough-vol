"""
Task 5 hybrid: PINN warm-start + Fourier local polish (the C11 result).
======================================================================

Resolves the parametric-PINN calibration quality gap. Take the PINN's single-shot
gradient calibration as the INITIAL GUESS and hand it to a LOCAL Fourier polish
(Nelder-Mead), counting Fourier evaluations, vs two baselines on the BTC T~0.15
slice (all judged by the TRUE Fourier model; all over the same economically-valid
region -- note the slice optimum lies OUTSIDE the parametric PINN's training box).

RESULT (3 PINN seeds; fit = vega-weighted IV RMSE under the true model):
    method                          fit RMSE(true)   wall-clock   Fourier evals
    (1) Fourier-loop from scratch      23.7 bp          168 s          2018
    (2) PINN-only (single-shot)        222 +- 20 bp      ~2 s             0
    (3) HYBRID (PINN + polish)         29.4 +- 7.1 bp     23 s         237 +- 46

=> The warm-start reaches NEAR from-scratch quality (29+-7 vs 24 bp -- comparable
   on a 34-52% IV smile; the ~6 bp gap is local-optimum variation on a flat
   multimodal landscape) at ~9x FEWER Fourier evals (237 vs 2018) and ~7x faster.
   PINN-only alone is fast but poor (222 bp); the hybrid gets both speed and
   quality. So C11 is an experimental result, not future work: the parametric
   PINN's applied value is as a calibration WARM-START, not a standalone pricer.
Requires outputs/param_net_seed{0,1,2}.pt (run diagnose_task5_parametric.py first).
"""
import json, time, numpy as np, pandas as pd, torch
from scipy.optimize import differential_evolution, minimize
from surface import build_surface, bs_price, implied_vol
from rough_heston_fourier import price_european_fourier
from parametric_pinn import ParametricPINN, DEFAULT_BOX, PNAMES

SAVE = "outputs"
cal = json.load(open("calib_real.json")); V0 = cal["V0"]; r = 0.0
surf = build_surface(pd.read_csv("data/deribit_chain.csv"), r=0.0)
Tsel = min(surf["T"].unique(), key=lambda t: abs(t - 0.15)); g = surf[surf["T"] == Tsel]
K = np.exp(g["k"].values); ivm = g["iv"].values; vega = g["weight"].values; w = vega / vega.sum()
Tp = 0.15; bounds = [DEFAULT_BOX[p] for p in PNAMES]
VALID = [(0.02, 0.45), (0.02, 1.2), (-0.99, -0.01), (0.05, 5.0), (0.02, 0.55)]

_nf = [0]
def fourier_iv(vec):
    p = dict(V0=V0, theta=vec[4], lam=vec[3], nu=vec[1], rho=vec[2])
    px = np.atleast_1d(price_european_fourier(1.0, K, Tsel, r, p, vec[0], N=120, u_max=100, n_u=800))
    return np.array([implied_vol(px[i], 1.0, K[i], Tsel, r, "C") for i in range(len(K))])
def wrmse(iv):
    m = np.isfinite(iv); return 1e4 * np.sqrt(np.sum(w[m] * (iv[m] - ivm[m]) ** 2) / w[m].sum())
def floss(vec):
    _nf[0] += 1
    if any(v < lo or v > hi for v, (lo, hi) in zip(vec, VALID)):
        return 1e3
    iv = fourier_iv(vec); m = np.isfinite(iv)
    return 1e3 if m.sum() < 0.5 * len(K) else np.sqrt(np.sum(w[m] * (iv[m] - ivm[m]) ** 2) / w[m].sum())

# ---- (1) Fourier-loop from scratch (DE + polish) -- over the SAME wider valid
#         region as the hybrid polish, so the head-to-head is fair. (The T=0.15
#         slice optimum lies OUTSIDE the parametric PINN's training box.)
_nf[0] = 0; t0 = time.perf_counter()
resF = differential_evolution(floss, VALID, maxiter=30, popsize=10, seed=1, tol=1e-4, polish=True)
tF, nF = time.perf_counter() - t0, _nf[0]
rmseF = wrmse(fourier_iv(resF.x))

# ---- (2)+(3) PINN single-shot init -> local Fourier polish, per seed ----
lo = torch.tensor([b[0] for b in bounds]); hi = torch.tensor([b[1] for b in bounds])
pmkt = torch.tensor([bs_price(1.0, K[i], Tsel, ivm[i], r, "C") for i in range(len(K))])
vg = torch.tensor(vega)
def pinn_init(seed):
    m = ParametricPINN(V0=V0, T=Tp, width=80, depth=4, n=10)
    m.net.load_state_dict(torch.load(f"{SAVE}/param_net_seed{seed}.pt"))
    t0 = time.perf_counter()
    raw = torch.zeros(5, requires_grad=True); opt = torch.optim.Adam([raw], lr=0.05)
    for step in range(400):
        opt.zero_grad()
        pv = lo + (hi - lo) * torch.sigmoid(raw)
        (((m.price_smile_grad(K, Tp, pv) - pmkt) ** 2 / vg).sum()).backward(); opt.step()
    return (lo + (hi - lo) * torch.sigmoid(raw)).detach().numpy(), time.perf_counter() - t0

Ponly, Hfit, Hev, Htime = [], [], [], []
for s in (0, 1, 2):
    p0, tP = pinn_init(s)
    Ponly.append(wrmse(fourier_iv(p0)))
    _nf[0] = 0; t0 = time.perf_counter()
    resH = minimize(floss, p0, method="Nelder-Mead", options=dict(maxiter=120, xatol=1e-4, fatol=1e-6))
    Hfit.append(wrmse(fourier_iv(resH.x))); Hev.append(_nf[0]); Htime.append(tP + time.perf_counter() - t0)
Ponly, Hfit, Hev, Htime = map(np.array, (Ponly, Hfit, Hev, Htime))

print("================ C11 HYBRID TEST (BTC T~0.15 slice, 3 seeds) ================")
print(f"{'method':<34}{'fit RMSE(true)':>18}{'wall-clock':>13}{'Fourier evals':>15}")
print(f"{'(1) Fourier-loop from scratch':<34}{rmseF:>16.1f}bp{tF:>11.1f}s{nF:>15d}")
print(f"{'(2) PINN-only (single-shot)':<34}{Ponly.mean():>10.0f}+-{Ponly.std():<4.0f}bp"
      f"{'~2':>11}s{0:>15d}")
print(f"{'(3) HYBRID: PINN + Fourier polish':<34}{Hfit.mean():>10.1f}+-{Hfit.std():<4.1f}bp"
      f"{Htime.mean():>11.1f}s{int(Hev.mean()):>11d}+-{int(Hev.std()):<3d}")
print(f"\n=> hybrid {Hfit.mean():.0f}+-{Hfit.std():.0f} bp (~ from-scratch {rmseF:.0f}) in "
      f"{int(Hev.mean())}+-{int(Hev.std())} Fourier evals vs {nF} "
      f"({nF/max(Hev.mean(),1):.0f}x fewer); {Htime.mean():.0f}s vs {tF:.0f}s")
