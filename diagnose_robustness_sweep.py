"""
Robustness sweep on the n-convergence headline (3 seeds) + skew re-measurement.
=============================================================================

Part 1 -- 3-seed solve-error mean+-std per n (canonical BTC calib, T=0.15):
    n    lift(px bp)   solve mean+-std (px bp)
    4      16.4         5.4 +- 1.0
    8       7.9         6.1 +- 1.4
   16       3.5         8.8 +- 1.6
   32       1.7        10.7 +- 2.0
  CORRECTION to the single-seed headline: solve-error is NOT flat -- it GROWS
  MILDLY with n (~2x over n=4->32, sub-linear vs the 8x dimension jump); the
  earlier single-seed run (4.7/6.9/5.9/4.6) drew lucky low seeds at n=16/32.
  The meshfree claim survives in the WEAK form: solve-error degrades gently, far
  slower than a grid would (50^(n+1)), not that it is constant. lift/solve
  crossover sits ~n=8-10 (lift>solve at n<=8, solve>lift at n>=16).

Part 4 -- put-wing skew decomposition vs the LIFTED MC (signed vol bp), n=4:
    solve (PINN-MC)  : puts  -5   atm +43   calls +62
    lift  (MC-Four)  : puts -170   atm -116  calls -67
    total (PINN-Four): puts -175   atm  -73  calls  -4
  The ρ=-0.79 put-wing under-pricing (total puts -175) is ALMOST ALL LIFT: the
  n=4 lifted model under-represents the model's skew (lift puts -170). The PINN's
  OWN put-wing solve-error is ~-5 bp (essentially zero). The residual solve-error
  is instead a call/ATM OVER-pricing (+62 at n=4) that grows with n (calls +114
  at n=32) -- i.e. Task 4b's "put-wing skew" was lift error, not a solve defect.

Part 3 (FD caveat, see the module note below).
"""
# FD caveat: the 50^(n+1) figure is a NAIVE full-tensor grid. Structured solvers
# exist -- sparse grids (Smolyak), ADI/operator splitting, low-rank tensor-train
# -- and push the tractable dimension higher. But their cost/accuracy still
# degrade with n (sparse-grid error and node count grow super-linearly in n; TT
# needs low-rank structure that the coupled rho-nu cross terms erode), and none
# is standard or validated for the lifted rough-vol PDE. So a competitive FD-type
# solve at n=16-32 is IMPRACTICAL, not provably impossible -- the honest claim.
import json, numpy as np, torch
from rough_heston_lift import lift_weights_geometric
from rough_heston_fourier import price_european_fourier
from lifted_mc import price_european_mc, simulate_factor_stats
from surface import implied_vol
from pinn import RoughHestonPINN

cal = json.load(open("calib_real.json"))
P = dict(V0=cal["V0"], theta=cal["theta"], lam=cal["lam"], nu=cal["nu"], rho=cal["rho"])
H = cal["H"]; r = 0.0; T = 0.15
strikes = np.array([0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15]); kk = np.log(strikes)
NLIST = [4, 8, 16, 32]; ITERS = 10000; SEEDS = (0, 1, 2)
buckets = [("OTMputs", kk < -0.05), ("ATM", np.abs(kk) <= 0.05), ("OTMcalls", kk > 0.05)]

def pbp(a, b): return 1e4 * np.sqrt(np.mean((a - b) ** 2))
def ivs(px): return np.array([implied_vol(max(px[i], 1e-12), 1.0, strikes[i], T, r, "C") for i in range(len(strikes))])
def wings(d): return {nm: 1e4 * np.nanmean(d[m]) for nm, m in buckets}

fpx = np.atleast_1d(price_european_fourier(1.0, strikes, T, r, P, H, N=300, u_max=120, n_u=1500))
iv_f = ivs(fpx)
rows = []
for n in NLIST:
    c, x = lift_weights_geometric(H + 0.5, n)
    mpx, _ = price_european_mc(1.0, strikes, T, r, P, (c, x), n_paths=200_000, n_steps=400, seed=7)
    iv_m = ivs(mpx); lift_price = pbp(mpx, fpx)
    mean, std, smean, _ = simulate_factor_stats(1.0, T, P, (c, x), 4000, 300, seed=1)
    solves = []; smiles = []
    for s in SEEDS:
        torch.manual_seed(s); np.random.seed(s)
        p = RoughHestonPINN(P, (c, x), K=1.0, r=r, T=T, width=64, depth=4, x_halfwidth=1.2)
        p.set_factor_sampling(mean, std); p.set_baseline_term_structure(smean)
        stack = []
        def probe(it, net):
            if it >= 0.8 * ITERS:
                stack.append(strikes * net.price(1.0 / strikes, tau=T))
        p.train(iters=ITERS, n_col=2000, n_bnd=500, n_bc=400, lr=1e-3, log_every=500, on_log=probe)
        ppx = np.mean(np.array(stack), axis=0)
        solves.append(pbp(ppx, mpx)); smiles.append(ppx)
    solves = np.array(solves); ppx_avg = np.mean(np.array(smiles), axis=0)
    iv_p = ivs(ppx_avg)
    sw = wings(iv_p - iv_m); lw = wings(iv_m - iv_f); tw = wings(iv_p - iv_f)
    rows.append(dict(n=n, lift=lift_price, solve_mean=solves.mean(), solve_std=solves.std(),
                     sw=sw, lw=lw, tw=tw))
    print(f"n={n:2d}: solve = {solves.mean():5.1f} +- {solves.std():4.1f} price-bp  "
          f"(seeds {np.round(solves,1)}) | lift = {lift_price:5.1f} price-bp")

print("\n==== PART 1: solve-error mean+-std vs n (price bp) ====")
print(f"{'n':>3}{'lift':>9}{'solve_mean':>12}{'solve_std':>11}")
for rw in rows:
    print(f"{rw['n']:>3}{rw['lift']:>9.1f}{rw['solve_mean']:>12.1f}{rw['solve_std']:>11.1f}")

print("\n==== PART 4: put-wing skew decomposition vs lifted MC (signed vol bp) ====")
print(f"{'n':>3}  {'solve(PINN-MC)':>26}  {'lift(MC-Four)':>26}  {'total(PINN-Four)':>26}")
for rw in rows:
    sw, lw, tw = rw['sw'], rw['lw'], rw['tw']
    f = lambda w: f"p{w['OTMputs']:+.0f}/a{w['ATM']:+.0f}/c{w['OTMcalls']:+.0f}"
    print(f"{rw['n']:>3}  {f(sw):>26}  {f(lw):>26}  {f(tw):>26}")
