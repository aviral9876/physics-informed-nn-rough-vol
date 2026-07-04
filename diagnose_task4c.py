"""
Task 4c: diagnose the uniform ~-115 bp level bias that persists even at rho=0.
=============================================================================

Three ordered checks on the rho=0 case (canonical BTC calib, T=0.15), stopping
when one explains the parallel downward price offset. Reproduce: run this file.

FINDINGS:
  CHECK 1 (baseline) -- FIRES, it is the root cause. p._bs_baseline matches
    scipy BS to 1e-16 (no discount/forward/formula bug), BUT it is anchored at
    the SPOT variance V0=0.103 (sig=32.17%) while the forward/effective variance
    over [0,T] is E[int V]/T = 0.150 (sig=38.78%, ~ the Fourier ATM IV 39.73%).
    Because theta=0.25 >> V0 and mean-reversion is fast (lam=2.57), the baseline
    is ~756 bp of IV too low (-116 bp*S at ATM). The network is then asked to
    supply a huge +756 bp correction and falls ~127 bp short.
  CHECK 2 (terminal condition) -- CLEAN. P(tau->0)-intrinsic = O(tau) -> 0
    (1.3e-5 at tau=1e-8); the tau-ansatz pins the initial condition correctly.
  CHECK 3 (residual vs price) -- CONFIRMS. held-out PDE residual L2 ~ 4.6e-3
    (mean-sq 2e-5) while IV bias is -127 bp: the net satisfies the PDE but
    converges to an under-corrected solution -- the near-zero residual does not
    penalise the remaining level gap (under-constrained by a poor anchor).

CONCLUSION: the level bias is a BASELINE-ANCHOR problem, not a convention bug
and not the skew term. Fix (next step, not done here): replace sqrt(V0) in
_bs_baseline with sqrt(E[int V]/T) -- the deterministic forward variance of the
lifted model (available from the same MC pre-flight, or the factor-mean ODE) --
so the baseline sits ~39% near truth and the correction shrinks to near zero.
"""
import json, numpy as np, torch
from scipy.stats import norm
from rough_heston_lift import lift_weights_geometric
from rough_heston_fourier import price_european_fourier
from lifted_mc import simulate_factor_stats
from surface import implied_vol
from pinn import RoughHestonPINN

cal = json.load(open("calib_real.json"))
P0 = dict(V0=cal["V0"], theta=cal["theta"], lam=cal["lam"], nu=cal["nu"], rho=0.0)  # rho=0
H = cal["H"]; r = 0.0; T = 0.15; n = 4
c, x = lift_weights_geometric(H + 0.5, n)
kk = np.linspace(-0.18, 0.18, 13); K = np.exp(kk)

def bs_call(S, Kk, T, sig):
    d1 = (np.log(S/Kk) + 0.5*sig*sig*T)/(sig*np.sqrt(T)); d2 = d1 - sig*np.sqrt(T)
    return S*norm.cdf(d1) - Kk*norm.cdf(d2)

fpx = np.atleast_1d(price_european_fourier(1.0, K, T, r, P0, H, N=300, u_max=120, n_u=1500))
ivf = np.array([implied_vol(fpx[i], 1.0, K[i], T, r, "C") for i in range(len(K))])
p = RoughHestonPINN(P0, (c, x), K=1.0, r=r, T=T, width=64, depth=4, x_halfwidth=1.2)

print("="*72)
print("CHECK 1 - baseline consistency (network zeroed => P = BS_baseline(V0))")
print("="*72)
# baseline smile via the SAME homogeneity remap used for the PINN: C(1,K)=K*C(1/K,1)
xb = torch.tensor(np.log(1.0/K)).reshape(-1,1); tb = torch.full_like(xb, T)
base = p._bs_baseline(tb, xb).detach().numpy().ravel()
base_smile = K * base                                   # baseline C(1,K)
# independent BS(sqrt(V0)) via homogeneity, as a formula cross-check
sigV0 = np.sqrt(P0["V0"])
base_ref = K * bs_call(1.0/K, 1.0, T, sigV0)
ivb = np.array([implied_vol(max(base_smile[i],1e-12),1.0,K[i],T,r,"C") for i in range(len(K))])
print(f"  sqrt(V0) = {sigV0*100:.2f}%   (baseline flat vol)")
print(f"  formula check |P._bs_baseline - scipy BS| max = {np.max(np.abs(base_smile-base_ref)):.2e}")
print(f"  {'k':>7}{'Fourier':>10}{'baseline':>10}{'dPrice(bp*S)':>13}{'IVf%':>8}{'IVbase%':>9}{'dIV bp':>9}")
for i in range(len(K)):
    print(f"  {kk[i]:>+7.3f}{fpx[i]:>10.5f}{base_smile[i]:>10.5f}"
          f"{1e4*(base_smile[i]-fpx[i]):>+13.1f}{100*ivf[i]:>8.2f}{100*ivb[i]:>9.2f}"
          f"{1e4*(ivb[i]-ivf[i]):>+9.1f}")
# effective (time-averaged) variance from MC vs V0
mean, std, smean, sstd = simulate_factor_stats(1.0, T, P0, (c, x), n_paths=4000, n_steps=300, seed=1)
EV_t = P0["V0"] + smean @ c                              # E[V_t] per step
EV_avg = float(EV_t.mean())                             # (1/T) int E[V] dt
print(f"\n  E[V]_timeavg = {EV_avg:.4f} -> eff vol = {100*np.sqrt(EV_avg):.2f}%  "
      f"(vs baseline sqrt(V0)={sigV0*100:.2f}%, Fourier ATM IV={100*ivf[6]:.2f}%)")
eff_smile = K * bs_call(1.0/K, 1.0, T, np.sqrt(EV_avg))
iveff = np.array([implied_vol(max(eff_smile[i],1e-12),1.0,K[i],T,r,"C") for i in range(len(K))])
print(f"  BS(eff vol) ATM dIV vs Fourier = {1e4*(iveff[6]-ivf[6]):+.1f} bp  "
      f"(baseline ATM dIV = {1e4*(ivb[6]-ivf[6]):+.1f} bp)")

print("\n" + "="*72)
print("CHECK 2 - terminal condition: P(tau->0) vs intrinsic payoff")
print("="*72)
for tau0 in [1e-8, 1e-4]:
    Sg = K.copy()                                       # native coords: spot=strike, K=1
    xg = torch.tensor(np.log(Sg)).reshape(-1,1); tg = torch.full_like(xg, tau0)
    ug = torch.zeros(len(Sg), n)
    Pg = p.price_ansatz(tg, xg, ug).detach().numpy().ravel()
    intrinsic = np.maximum(Sg - 1.0, 0.0)
    print(f"  tau={tau0:.0e}: max|P - intrinsic| = {np.max(np.abs(Pg-intrinsic)):.3e}")

print("\n" + "="*72)
print("CHECK 3 - residual vs price coupling (train rho=0, 15k iters)")
print("="*72)
torch.manual_seed(0); np.random.seed(0)
p3 = RoughHestonPINN(P0, (c, x), K=1.0, r=r, T=T, width=64, depth=4, x_halfwidth=1.2)
p3.set_factor_sampling(mean, std)
val_pts = p3.fixed_val_set(n=3000)
def pinn_ivs(net):
    hom = K * net.price(1.0/K, tau=T)
    return np.array([implied_vol(max(hom[i],1e-8),1.0,K[i],T,r,"C") for i in range(len(K))])
p3.train(iters=15000, n_col=2000, n_bnd=500, n_bc=400, lr=1e-3, log_every=1500, val_pts=val_pts)
res_final = p3.val_history[-1][1]                        # held-out mean residual^2
ivp = pinn_ivs(p3)
ivrmse = 1e4*np.sqrt(np.nanmean((ivp-ivf)**2))
biasbp = 1e4*np.nanmean(ivp-ivf)
print(f"  held-out PDE residual (mean sq) = {res_final:.3e}  (L2 = {np.sqrt(res_final):.3e})")
print(f"  IV RMSE = {ivrmse:.1f} bp   mean IV bias = {biasbp:+.1f} bp")
p_atm = float(K[6] * p3.price(1.0/K[6], tau=T)[0])
print(f"  price(ATM) PINN={p_atm:.5f}  Fourier={fpx[6]:.5f}  "
      f"dPrice={1e4*(p_atm-fpx[6]):+.1f} bp*S")
