"""
Task 5 Part 2: parametric-PINN solve-error across the box vs single-point.
=========================================================================

Trains 3 seeds of the parametric PINN (n=10, chosen by the crossover rule: at
T~0.15 lift-error meets solve-error near n~8-12) over the BTC parameter box, and
reports its solve-error (vs lifted MC, same n) at several points spanning the box
plus a single-point n=10 baseline. Saves the nets to outputs/ for the calibration
demo. Result (price bp, 3 seeds):

    param point    para solve(mean+-std)   lift
    BTC-centre        11.0 +- 4.7           6.1
    low-H/mild        47.7 +- 6.1           1.7    <- rough corner, worst
    high-nu           29.8 +- 3.9           3.6
    weak-skew          9.0 +- 3.3           0.8
    strong             9.8 +- 2.9           8.4
  single-point n=10 @ BTC-centre: 5.9 +- 1.2 px bp
  => coverage penalty at the centre ~ +5 px bp (~2x); much larger in the rough
     low-H / high-nu corners. Spreading one net over the 5-D box costs accuracy,
     worst where the PDE is hardest (very rough H, high vol-of-vol). A 20k-iter,
     width-80 budget was used; more training would shrink the corner penalty.
"""
import json, numpy as np, torch
from rough_heston_lift import lift_weights_geometric
from rough_heston_fourier import price_european_fourier
from lifted_mc import price_european_mc, simulate_factor_stats
from surface import implied_vol
from pinn import RoughHestonPINN
from parametric_pinn import ParametricPINN, PNAMES

cal = json.load(open("calib_real.json"))
V0 = cal["V0"]; T = 0.15; r = 0.0; n = 10
strikes = np.array([0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15])
SEEDS = (0, 1, 2); SAVE = "outputs"

# parameter points spanning the training box (H,nu,rho,lam,theta)
PTS = {
 "BTC-centre": [cal["H"], cal["nu"], cal["rho"], cal["lam"], cal["theta"]],
 "low-H/mild": [0.05, 0.20, -0.50, 1.5, 0.15],
 "high-nu":    [0.09, 0.55, -0.80, 2.5, 0.25],
 "weak-skew":  [0.15, 0.20, -0.30, 1.0, 0.12],
 "strong":     [0.06, 0.45, -0.90, 3.5, 0.32],
}

def pbp(a, b): return 1e4 * np.sqrt(np.nanmean((a - b) ** 2))
def ivs(px): return np.array([implied_vol(max(px[i], 1e-12), 1.0, strikes[i], T, r, "C") for i in range(len(strikes))])

# ---- references: Fourier + lifted MC (n=10) at each point ----
print("computing references (Fourier + lifted MC n=10)...")
ref = {}
for name, pv in PTS.items():
    Hh = pv[0]; p = dict(V0=V0, theta=pv[4], lam=pv[3], nu=pv[1], rho=pv[2])
    c, x = lift_weights_geometric(Hh + 0.5, n)
    fpx = np.atleast_1d(price_european_fourier(1.0, strikes, T, r, p, Hh, N=300, u_max=120, n_u=1500))
    mpx, _ = price_european_mc(1.0, strikes, T, r, p, (c, x), n_paths=200_000, n_steps=400, seed=7)
    ref[name] = dict(fpx=fpx, mpx=mpx, lift=pbp(mpx, fpx))
    print(f"  {name:12s} lift={ref[name]['lift']:.1f} px-bp")

# ---- train 3 parametric PINNs ----
para_solve = {name: [] for name in PTS}
for s in SEEDS:
    torch.manual_seed(s); np.random.seed(s)
    m = ParametricPINN(V0=V0, T=T, width=80, depth=4, n=n)
    print(f"\n[parametric seed {s}] training 20k...")
    m.train(iters=20000, n_col=2500, n_bc=500, lr=1e-3, log_every=5000)
    torch.save(m.net.state_dict(), f"{SAVE}/param_net_seed{s}.pt")
    for name, pv in PTS.items():
        solve = pbp(m.smile(strikes, T, pv), ref[name]["mpx"])
        para_solve[name].append(solve)

# ---- single-point n=10 baseline at BTC centre, 3 seeds ----
print("\n[single-point n=10 baseline @ BTC centre] 3 seeds x 10k...")
pv0 = PTS["BTC-centre"]; p0 = dict(V0=V0, theta=pv0[4], lam=pv0[3], nu=pv0[1], rho=pv0[2])
c0, x0 = lift_weights_geometric(pv0[0] + 0.5, n)
mean, std, smean, _ = simulate_factor_stats(1.0, T, p0, (c0, x0), 4000, 300, seed=1)
sp_solve = []
for s in SEEDS:
    torch.manual_seed(s); np.random.seed(s)
    sp = RoughHestonPINN(p0, (c0, x0), K=1.0, r=r, T=T, width=64, depth=4, x_halfwidth=1.2)
    sp.set_factor_sampling(mean, std); sp.set_baseline_term_structure(smean)
    sp.train(iters=10000, n_col=2000, n_bnd=500, n_bc=400, lr=1e-3, log_every=10000)
    sp_solve.append(pbp(strikes * sp.price(1.0 / strikes, tau=T), ref["BTC-centre"]["mpx"]))

print("\n================ TASK 5 PART 2: solve-error (price bp) ================")
print(f"{'param point':<14}{'para solve(mean+-std)':>24}{'lift':>8}")
for name in PTS:
    a = np.array(para_solve[name])
    print(f"{name:<14}{a.mean():>14.1f} +-{a.std():>6.1f}{ref[name]['lift']:>8.1f}")
sp = np.array(sp_solve)
print(f"\nsingle-point n=10 @ BTC-centre: solve = {sp.mean():.1f} +- {sp.std():.1f} px bp")
print(f"parametric      @ BTC-centre: solve = {np.mean(para_solve['BTC-centre']):.1f} "
      f"+- {np.std(para_solve['BTC-centre']):.1f} px bp")
print(f"=> parametric-coverage penalty = {np.mean(para_solve['BTC-centre'])-sp.mean():+.1f} px bp")
