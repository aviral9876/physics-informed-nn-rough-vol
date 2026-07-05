"""
Task 4c fix confirmation: re-anchor BS baseline to lifted E[int V dt], and a
key reframing of what the "level bias" actually is.
======================================================================

The fix (pinn.set_baseline_term_structure): anchor the BS baseline at the lifted
model's expected integrated variance w(tau)=int_0^tau E[V_s] ds instead of the
spot variance V0. Baseline sanity (rho=0, small nu): the term-structure baseline
matches the LIFTED MC (the model the PINN actually solves) to ~13 bp RMSE, vs
~780 bp for the V0 anchor -- so the fix is CORRECT.

BUT the retrain (canonical rho=-0.79, T=0.15, 3 seeds, 15k) reveals the "-127 bp
level bias vs Fourier" is NOT the baseline -- it is the n=4 LIFT ERROR:
                       vs FOURIER (bp)        vs LIFTED-MC (bp)   RMSE_F  RMSE_M
  V0-baseline (4c)   puts-171 atm-97 calls -3   +13 +31 +68        119     46
  TS-baseline (fix)  puts-170 atm-95 calls-14   +14 +33 +58        117     40
  LIFT ERROR (MC-Fourier): puts-184 atm-128 calls-71  RMSE 138 bp

So: (1) the PINN already solves the lifted PDE to ~40-46 bp vs the lifted MC,
    even with the V0 baseline -- the network compensates for the bad anchor;
(2) the ~120 bp gap vs Fourier is the lifted-MC-vs-Fourier lift error (n=4,
    ~26% kernel error), which NO baseline or training change can remove;
(3) the baseline fix is principled and modestly helps (46->40 bp vs lifted-MC,
    and shrinks the required correction 756->30 bp = better conditioning), but
    at 15k iters it barely moves the trained endpoint.
LESSON: validate the PINN against the LIFTED MC (its true target), not Fourier;
the Fourier gap is dominated by the lift and only shrinks with more factors n.
"""
import json, numpy as np, torch
from rough_heston_lift import lift_weights_geometric
from rough_heston_fourier import price_european_fourier
from lifted_mc import price_european_mc, simulate_factor_stats
from surface import implied_vol
from pinn import RoughHestonPINN

cal = json.load(open("calib_real.json"))
params = dict(V0=cal["V0"], theta=cal["theta"], lam=cal["lam"], nu=cal["nu"], rho=cal["rho"])
H = cal["H"]; r = 0.0; T = 0.15; n = 4
c, x = lift_weights_geometric(H + 0.5, n)
kk = np.linspace(-0.18, 0.18, 13); K = np.exp(kk)
buckets = [("OTMputs", kk < -0.05), ("ATM", np.abs(kk) <= 0.05), ("OTMcalls", kk > 0.05)]
ITERS = 15000

def ivs(px):
    return np.array([implied_vol(max(px[i],1e-12),1.0,K[i],T,r,"C") for i in range(len(K))])

# references
ivf = ivs(np.atleast_1d(price_european_fourier(1.0,K,T,r,params,H,N=300,u_max=120,n_u=1500)))
mpx,_ = price_european_mc(1.0,K,T,r,params,(c,x),n_paths=400_000,n_steps=400,seed=7)
ivm = ivs(mpx)
mean, std, smean, sstd = simulate_factor_stats(1.0,T,params,(c,x),n_paths=4000,n_steps=300,seed=1)

def decomp(ivp, ref):
    return {name: 1e4*np.nanmean((ivp-ref)[m]) for name, m in buckets}

def run(seed, use_ts):
    torch.manual_seed(seed); np.random.seed(seed)
    p = RoughHestonPINN(params,(c,x),K=1.0,r=r,T=T,width=64,depth=4,x_halfwidth=1.2)
    p.set_factor_sampling(mean, std)
    if use_ts: p.set_baseline_term_structure(smean)
    stack=[]
    def probe(it, net):
        if it >= 0.85*ITERS:
            stack.append(ivs(K*net.price(1.0/K,tau=T)))
    p.train(iters=ITERS,n_col=2000,n_bnd=500,n_bc=400,lr=1e-3,log_every=500,on_log=probe)
    return np.nanmean(np.array(stack),axis=0)

for use_ts, tag in [(False,"V0-baseline (4c)"),(True,"TS-baseline (fix)")]:
    ivps = [run(s, use_ts) for s in (0,1,2)]
    ivp = np.nanmean(ivps, axis=0)
    dF = decomp(ivp, ivf); dM = decomp(ivp, ivm)
    rF = 1e4*np.sqrt(np.nanmean((ivp-ivf)**2)); rM = 1e4*np.sqrt(np.nanmean((ivp-ivm)**2))
    print(f"\n=== {tag} (3-seed mean) ===")
    print(f"  vs FOURIER : puts {dF['OTMputs']:+.0f}  atm {dF['ATM']:+.0f}  calls {dF['OTMcalls']:+.0f}  | RMSE {rF:.0f} bp")
    print(f"  vs liftedMC: puts {dM['OTMputs']:+.0f}  atm {dM['ATM']:+.0f}  calls {dM['OTMcalls']:+.0f}  | RMSE {rM:.0f} bp")

dLift = decomp(ivm, ivf)
print(f"\n=== LIFT ERROR (lifted-MC - Fourier, n={n}) ===")
print(f"  puts {dLift['OTMputs']:+.0f}  atm {dLift['ATM']:+.0f}  calls {dLift['OTMcalls']:+.0f}  | "
      f"RMSE {1e4*np.sqrt(np.nanmean((ivm-ivf)**2)):.0f} bp")
