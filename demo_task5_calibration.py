"""
Task 5 Part 3 (headline): recalibrate the BTC T~0.15 slice two ways.
===================================================================

(a) Fourier-in-the-loop differential evolution (the existing pricer).
(b) Gradient descent on the 5 params using the parametric PINN as pricer (zero
    Fourier calls), multi-restart across the 3 trained nets for a fair global
    search. V0 fixed in both; both fit the same slice (35 pts).

RESULT (fit RMSE = vega-weighted IV RMSE vs market):
    Fourier-loop DE : 44.9 vol bp | ~180 s | 2030 Fourier evals
    PINN gradient    : self-fit 88.6 vol bp | 15.6 s | 0 Fourier evals  (x24 restarts)
                       BUT priced with the TRUE Fourier model, the PINN's fitted
                       params give 362 vol bp -- the net calibrates to its OWN
                       (imperfect) prices, so its self-fit is optimistic and the
                       chosen params are actually poor.

HONEST VERDICT: the speedup is real and large (~11x here at 24 restarts, ~245x
single-shot), but fit quality is NOT comparable yet -- the calibration inherits
the net's solve-error (worst in the rough low-H corner where the BTC optimum
sits). Closing the gap needs a better-trained/wider parametric net (an amortised,
one-time cost). Speed: yes; production-grade fit: not with this 20k/width-80 net.
Requires outputs/param_net_seed{0,1,2}.pt (run diagnose_task5_parametric.py first).
"""
import json, time, numpy as np, pandas as pd, torch
from scipy.optimize import differential_evolution
from surface import build_surface, bs_price, implied_vol
from rough_heston_fourier import price_european_fourier
from parametric_pinn import ParametricPINN, DEFAULT_BOX, PNAMES

SAVE = "outputs"
cal = json.load(open("calib_real.json")); V0 = cal["V0"]; r = 0.0
surf = build_surface(pd.read_csv("data/deribit_chain.csv"), r=0.0)
Tsel = min(surf["T"].unique(), key=lambda t: abs(t - 0.15)); g = surf[surf["T"] == Tsel]
K = np.exp(g["k"].values); ivm = g["iv"].values; vega = g["weight"].values; w = vega / vega.sum()
Tp = 0.15; bounds = [DEFAULT_BOX[p] for p in PNAMES]
def wrmse(iv):
    m = np.isfinite(iv); return 1e4 * np.sqrt(np.sum(w[m] * (iv[m] - ivm[m]) ** 2) / w[m].sum())

# ---- Fourier-loop DE (baseline) ----
def fiv(vec):
    p = dict(V0=V0, theta=vec[4], lam=vec[3], nu=vec[1], rho=vec[2])
    px = np.atleast_1d(price_european_fourier(1.0, K, Tsel, r, p, vec[0], N=120, u_max=100, n_u=800))
    return np.array([implied_vol(px[i], 1.0, K[i], Tsel, r, "C") for i in range(len(K))])
def floss(vec):
    iv = fiv(vec); m = np.isfinite(iv)
    return 1e3 if m.sum() < 0.5 * len(K) else np.sqrt(np.sum(w[m] * (iv[m] - ivm[m]) ** 2) / w[m].sum())
t0 = time.perf_counter()
resF = differential_evolution(floss, bounds, maxiter=30, popsize=10, seed=1, tol=1e-4, polish=True)
tF = time.perf_counter() - t0
print(f"[Fourier-loop DE]  fit RMSE = {wrmse(fiv(resF.x)):.1f} vol bp | {tF:.1f} s | {resF.nfev} evals")
print("   ", dict(zip(PNAMES, np.round(resF.x, 4))))

# ---- PINN gradient, multi-restart x multi-seed (fair global search) ----
lo = torch.tensor([b[0] for b in bounds]); hi = torch.tensor([b[1] for b in bounds])
pmkt = torch.tensor([bs_price(1.0, K[i], Tsel, ivm[i], r, "C") for i in range(len(K))])
vg = torch.tensor(vega)
nets = []
for s in (0, 1, 2):
    m = ParametricPINN(V0=V0, T=Tp, width=80, depth=4, n=10)
    m.net.load_state_dict(torch.load(f"{SAVE}/param_net_seed{s}.pt")); nets.append(m)

def pinn_iv(m, pv):
    return np.array([implied_vol(max(m.smile(np.array([K[i]]), Tp, pv)[0], 1e-12), 1.0, K[i], Tsel, r, "C")
                     for i in range(len(K))])
t0 = time.perf_counter(); best = (1e9, None, None); rng = np.random.default_rng(0)
for m in nets:
    for rs in range(8):
        raw = torch.tensor(rng.normal(0, 1.2, 5), requires_grad=True)
        opt = torch.optim.Adam([raw], lr=0.05)
        for step in range(500):
            opt.zero_grad()
            pv = lo + (hi - lo) * torch.sigmoid(raw)
            loss = ((m.price_smile_grad(K, Tp, pv) - pmkt) ** 2 / vg).sum()
            loss.backward(); opt.step()
        pv = (lo + (hi - lo) * torch.sigmoid(raw)).detach().numpy()
        rm = wrmse(pinn_iv(m, pv))
        if rm < best[0]: best = (rm, pv, m)
tP = time.perf_counter() - t0
print(f"\n[PINN gradient x24 restarts]  best fit RMSE = {best[0]:.1f} vol bp | {tP:.1f} s | 0 Fourier evals")
print("   ", dict(zip(PNAMES, np.round(best[1], 4))))
# what does Fourier think of the PINN's fitted params? (does the net mis-price there?)
print(f"   Fourier-IV RMSE at PINN's params = {wrmse(fiv(best[1])):.1f} vol bp (net's own solve error inflates its fit)")
print(f"\n==> speedup {tF/tP:.0f}x ({tF:.0f}s -> {tP:.1f}s);  fit Fourier {wrmse(fiv(resF.x)):.0f} vs PINN {best[0]:.0f} vol bp")
