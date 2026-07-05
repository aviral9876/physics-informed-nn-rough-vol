"""
Regenerate Task 3 / 4a / 4b numbers under the solve/lift decomposition.
======================================================================

Corrects the record: earlier tasks reported a single conflated "PINN vs Fourier"
IV RMSE (~114-127 bp). Split into PINN solve-error (vs lifted MC at the same n)
and lift-error (lifted MC vs Fourier), the PINN's own error is tiny and the
figures were dominated by the n=4 lift.

RESULTS (canonical BTC calib, T=0.15, n=4, 3 seeds, 12k iters):

  Lift-error vs n (lifted MC vs Fourier, model-only, PINN-independent):
      n= 4 -> 133 bp   n= 8 -> 63 bp   n=16 -> 27 bp   n=32 -> 12 bp

  config                                solve(vsMC)  lift(vsF)  total(vsF)
  Task3 BEFORE: old box (no MC factors)     61          133        194
  Task3 AFTER / Task4a: full fixes          37          133        121
  Task4b: rho=0 (level-bias case)            8          133        128

Reading: (1) Task 3's MC-factor sampling cut the SOLVE error 61->37 bp (the
part the PINN controls); the lift is fixed at 133. (2) Task 4a's "~122 bp" is
37 bp solve + 133 bp lift (partial sign cancellation -> total 121); the PINN
already meets the <50 bp solve target. (3) Task 4b's rho=0 "~127 bp" is only
8 bp solve -- essentially ALL lift. So "PINN vs Fourier" at n=4 is mostly model
approximation; cut it by raising n (12 bp lift at n=32), not by more training.
Note solve+lift do not add as RMSEs (opposite signs partly cancel).
"""
import json, numpy as np, torch
from rough_heston_lift import lift_weights_geometric
from rough_heston_fourier import price_european_fourier
from lifted_mc import price_european_mc, simulate_factor_stats
from validate_pinn import _ivs, _rmse_bp, lift_error_vs_n
from pinn import RoughHestonPINN

cal = json.load(open("calib_real.json"))
CAN = dict(V0=cal["V0"], theta=cal["theta"], lam=cal["lam"], nu=cal["nu"], rho=cal["rho"])
H = cal["H"]; r = 0.0; T = 0.15; nf = 4
c, x = lift_weights_geometric(H + 0.5, nf)
strikes = np.array([0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15])
ITERS = 12000; SEEDS = (0, 1, 2)

# lift-error as a function of n (model-only)
lift_error_vs_n(CAN, H, T=T, n_list=(4, 8, 16, 32), mc_paths=200_000, mc_steps=400)

_refcache = {}
def refs(params):
    key = round(params["rho"], 4)
    if key not in _refcache:
        fpx = np.atleast_1d(price_european_fourier(1.0, strikes, T, r, params, H, N=300, u_max=120, n_u=1500))
        mpx, _ = price_european_mc(1.0, strikes, T, r, params, (c, x), n_paths=200_000, n_steps=400, seed=7)
        _refcache[key] = (_ivs(fpx, strikes, T, r), _ivs(mpx, strikes, T, r))
    return _refcache[key]

def pinn_iv(net):
    return _ivs(strikes * net.price(1.0 / strikes, tau=T), strikes, T, r)

def run(params, full, tag):
    ivf, ivm = refs(params)
    mean, std, smean, _ = simulate_factor_stats(1.0, T, params, (c, x), n_paths=4000, n_steps=300, seed=1)
    ivps = []
    for s in SEEDS:
        torch.manual_seed(s); np.random.seed(s)
        p = RoughHestonPINN(params, (c, x), K=1.0, r=r, T=T, width=64, depth=4, x_halfwidth=1.2)
        if full:
            p.set_factor_sampling(mean, std); p.set_baseline_term_structure(smean)
        stack = []
        def probe(it, net):
            if it >= 0.85 * ITERS: stack.append(pinn_iv(net))
        p.train(iters=ITERS, n_col=2000, n_bnd=500, n_bc=400, lr=1e-3, log_every=1000, on_log=probe)
        ivps.append(np.nanmean(np.array(stack), axis=0))
    ivp = np.nanmean(ivps, axis=0)
    solve, lift, total = _rmse_bp(ivp, ivm), _rmse_bp(ivm, ivf), _rmse_bp(ivp, ivf)
    print(f"{tag:<40} solve={solve:6.1f}  lift={lift:6.1f}  total={total:6.1f}  vol bp")
    return solve, lift, total

print(f"\n=== Decomposed (canonical BTC calib, T={T}, n={nf}, 3-seed, {ITERS} iters) ===")
print(f"{'config':<40}{'solve(vsMC)':>12}{'lift(vsF)':>10}{'total(vsF)':>11}")
run(CAN, False, "Task3 BEFORE: old box (no MC factors)")
run(CAN, True,  "Task3 AFTER / Task4a: full fixes")
run(dict(**{**CAN, 'rho': 0.0}), True, "Task4b: rho=0 (level-bias case)")
