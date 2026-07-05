"""
Robustness Part 2: solve/lift for n in {4,16} across T and H; crossover shift.
============================================================================

Marginal sweeps (canonical BTC params, rho=-0.79, single seed, 10k iters),
errors in price bp of spot.

T-sweep (H=0.10):
    T     solve n4  lift n4   solve n16  lift n16   crossover
   0.05     2.1      8.6        3.3       1.4        in (4,16)
   0.15     4.5     16.1        7.7       3.5        in (4,16)
   0.50     6.5     31.8        5.5       8.5        > 16  (lift still dominates)
H-sweep (T=0.15):
    H     solve n4  lift n4   solve n16  lift n16   crossover
   0.05     3.6     17.7        9.5       3.5        in (4,16)
   0.10     4.5     16.1        7.7       3.5        in (4,16)
   0.15     1.0     15.3        4.5       3.7        in (4,16)

FINDINGS:
  * Solve-error stays SMALL and bounded across the whole grid (~1-9.5 px bp) --
    no blow-up in any corner; it rises mildly with T and shows the same gentle
    n-growth (n16 > n4). The meshfree property is robust to T and H.
  * Lift-error GROWS with T (8.6 -> 31.8 px bp as T 0.05 -> 0.5 at n=4; the
    kernel approximation costs more over longer horizons) and is ~flat / slightly
    lower for larger H (less rough -> easier to lift).
  * The lift/solve CROSSOVER moves with T: at short/mid T it sits between n=4 and
    n=16 (~n=8-10, as in the headline), but at long T=0.5 the lift is large
    enough that it still exceeds the solve floor at n=16 -- so you need MORE
    factors (crossover shifts right) for longer maturities. Across H the
    crossover is stable in (4,16).
"""
import json, numpy as np, torch
from rough_heston_lift import lift_weights_geometric
from rough_heston_fourier import price_european_fourier
from lifted_mc import price_european_mc, simulate_factor_stats
from pinn import RoughHestonPINN

cal = json.load(open("calib_real.json"))
P = dict(V0=cal["V0"], theta=cal["theta"], lam=cal["lam"], nu=cal["nu"], rho=cal["rho"])
r = 0.0; strikes = np.array([0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15])
ITERS = 10000; SEED = 0
NS = [4, 16]; H_REF = 0.10; T_REF = 0.15
TS = [0.05, 0.15, 0.5]; HS = [0.05, 0.10, 0.15]

def pbp(a, b): return 1e4 * np.sqrt(np.mean((a - b) ** 2))

# unique (n, T, H) configs: T-sweep at H_REF + H-sweep at T_REF
cfgs = set((n, t, H_REF) for n in NS for t in TS) | set((n, T_REF, h) for n in NS for h in HS)
res = {}
for (n, T, H) in sorted(cfgs):
    c, x = lift_weights_geometric(H + 0.5, n)
    fpx = np.atleast_1d(price_european_fourier(1.0, strikes, T, r, P, H, N=300, u_max=120, n_u=1500))
    mpx, _ = price_european_mc(1.0, strikes, T, r, P, (c, x), n_paths=200_000, n_steps=400, seed=7)
    lift = pbp(mpx, fpx)
    mean, std, smean, _ = simulate_factor_stats(1.0, T, P, (c, x), 4000, 300, seed=1)
    torch.manual_seed(SEED); np.random.seed(SEED)
    p = RoughHestonPINN(P, (c, x), K=1.0, r=r, T=T, width=64, depth=4, x_halfwidth=1.2)
    p.set_factor_sampling(mean, std); p.set_baseline_term_structure(smean)
    stack = []
    def probe(it, net):
        if it >= 0.8 * ITERS: stack.append(strikes * net.price(1.0 / strikes, tau=T))
    p.train(iters=ITERS, n_col=2000, n_bnd=500, n_bc=400, lr=1e-3, log_every=1000, on_log=probe)
    solve = pbp(np.mean(np.array(stack), axis=0), mpx)
    res[(n, T, H)] = (solve, lift)
    print(f"n={n:2d} T={T:.2f} H={H:.2f}: solve={solve:5.1f}  lift={lift:6.1f} price-bp  "
          f"({'lift>solve' if lift>solve else 'solve>lift'})")

def crossover(n4, n16):
    # lift>solve at n=4 and lift<solve at n=16 => crossover in (4,16)
    if n4[1] > n4[0] and n16[1] < n16[0]: return "in (4,16)"
    if n4[1] <= n4[0]: return "<4 (solve dominates already)"
    return ">16 (lift still dominates)"

print("\n==== PART 2a: T-sweep (H=%.2f) ====" % H_REF)
print(f"{'T':>6}{'solve n4':>10}{'lift n4':>9}{'solve n16':>11}{'lift n16':>10}{'crossover':>18}")
for T in TS:
    a, b = res[(4, T, H_REF)], res[(16, T, H_REF)]
    print(f"{T:>6.2f}{a[0]:>10.1f}{a[1]:>9.1f}{b[0]:>11.1f}{b[1]:>10.1f}{crossover(a,b):>18}")
print("\n==== PART 2b: H-sweep (T=%.2f) ====" % T_REF)
print(f"{'H':>6}{'solve n4':>10}{'lift n4':>9}{'solve n16':>11}{'lift n16':>10}{'crossover':>18}")
for H in HS:
    a, b = res[(4, T_REF, H)], res[(16, T_REF, H)]
    print(f"{H:>6.2f}{a[0]:>10.1f}{a[1]:>9.1f}{b[0]:>11.1f}{b[1]:>10.1f}{crossover(a,b):>18}")
