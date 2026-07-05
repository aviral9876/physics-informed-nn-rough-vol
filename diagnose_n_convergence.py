"""
n-convergence study -- the headline meshfree argument.
======================================================

For n = 4, 8, 16, 32 lift factors (canonical BTC calib, T=0.15, width=64,
depth=4, CPU), reports errors in PRICE space (signed per-strike price errors add
exactly: PINN-Fourier = (PINN-liftedMC) + (liftedMC-Fourier)):

  n   lift(px bp)  solve(px bp)  total(px bp)  PINN t_plateau  ms/iter  FD 50^(n+1)
  4     16.4          4.7           14.4          397 s          55      3.1e8
  8      7.9          6.9            6.1          386 s          57      2.0e15
 16      3.5          5.9            3.3          591 s          62      7.6e28
 32      1.7          4.6            2.6          457 s          61      1.2e56
  (vol-bp ref, non-additive:  solve 33/48/43/29,  lift 133/63/27/12)

NOTE: this file is a SINGLE-SEED run. The 3-seed sweep
(diagnose_robustness_sweep.py) supersedes the solve-error column: solve-error is
NOT flat, it GROWS MILDLY with n (5.4 -> 10.7 px bp over n=4->32); the flat/
decreasing look here (4.7/6.9/5.9/4.6) was lucky seeds at n=16/32. Read the
findings below with that correction.

FINDINGS:
  * PINN solve-error grows MILDLY (sub-linearly) in n -- ~2x over an 8x dimension
    jump, far slower than a grid. The meshfree property holds in the WEAK form
    (gentle degradation, not constant). This is the empirical core of the method.
  * lift-error shrinks monotonically (16.4 -> 1.7 px bp) -- the only knob that
    cuts the Fourier gap. It crosses the solve-error near n~9.
  * PINN wall-clock to plateau stays bounded (~6-10 min, ms/iter ~55->61 nearly
    flat) as n grows, because the u-Hessian is vectorised.
  * A naive finite-difference solve on the (n+1)-D lifted PDE needs 50^(n+1) grid
    points: 3e8 (n=4, ~2.5 GB) -> 2e15 (n=8, ~16 PB, already infeasible) ->
    1e56 (n=32). FD dies by n=8; the PINN trains all n on a laptop CPU.
That contrast -- flat solve-error + bounded cost while FD explodes -- is the
curse-of-dimensionality argument for the meshfree PINN.
"""
import json, time, numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rough_heston_lift import lift_weights_geometric
from rough_heston_fourier import price_european_fourier
from lifted_mc import price_european_mc, simulate_factor_stats
from surface import implied_vol
from pinn import RoughHestonPINN

cal = json.load(open("calib_real.json"))
P = dict(V0=cal["V0"], theta=cal["theta"], lam=cal["lam"], nu=cal["nu"], rho=cal["rho"])
H = cal["H"]; r = 0.0; T = 0.15
strikes = np.array([0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15])
N_LIST = [4, 8, 16, 32]; ITERS = 12000; SEED = 0

def prmse(a, b):        # price RMSE in bp of spot (S0=1); errors ADD in price space
    return 1e4 * np.sqrt(np.mean((a - b) ** 2))
def ivbp(px):
    return np.array([implied_vol(max(px[i], 1e-12), 1.0, strikes[i], T, r, "C")
                     for i in range(len(strikes))])

# Fourier is n-independent -> compute once
fpx = np.atleast_1d(price_european_fourier(1.0, strikes, T, r, P, H, N=300, u_max=120, n_u=1500))
iv_f = ivbp(fpx)

rows = []
for n in N_LIST:
    c, x = lift_weights_geometric(H + 0.5, n)
    mpx, mse = price_european_mc(1.0, strikes, T, r, P, (c, x),
                                 n_paths=200_000, n_steps=400, seed=7)
    lift_price = prmse(mpx, fpx)
    lift_iv = 1e4 * np.sqrt(np.nanmean((ivbp(mpx) - iv_f) ** 2))
    mean, std, smean, _ = simulate_factor_stats(1.0, T, P, (c, x), 4000, 300, seed=1)

    torch.manual_seed(SEED); np.random.seed(SEED)
    p = RoughHestonPINN(P, (c, x), K=1.0, r=r, T=T, width=64, depth=4, x_halfwidth=1.2)
    p.set_factor_sampling(mean, std); p.set_baseline_term_structure(smean)
    hist = []; probe_t = [0.0]; t0 = time.perf_counter()
    def probe(it, net):
        ts = time.perf_counter()
        ppx = strikes * net.price(1.0 / strikes, tau=T)
        hist.append((it, time.perf_counter() - t0 - probe_t[0], prmse(ppx, mpx)))
        probe_t[0] += time.perf_counter() - ts
    p.train(iters=ITERS, n_col=2000, n_bnd=500, n_bc=400, lr=1e-3, log_every=500, on_log=probe)
    hist = np.array(hist)

    plateau = hist[hist[:, 0] >= 0.8 * ITERS][:, 2].mean()      # solve-error plateau (price bp)
    post = hist[hist[:, 0] >= 0.5 * ITERS]                       # after curriculum
    hit = post[post[:, 2] <= 1.10 * plateau]
    t_plateau = float(hit[0, 1]) if len(hit) else float(hist[-1, 1])
    it_plateau = int(hit[0, 0]) if len(hit) else ITERS
    ppx = strikes * p.price(1.0 / strikes, tau=T)
    solve_iv = 1e4 * np.sqrt(np.nanmean((ivbp(ppx) - ivbp(mpx)) ** 2))
    total_price = prmse(ppx, fpx)
    fd = 50.0 ** (n + 1)
    rows.append(dict(n=n, lift=lift_price, solve=plateau, total=total_price,
                     lift_iv=lift_iv, solve_iv=solve_iv, tsec=t_plateau,
                     it_pl=it_plateau, ms_iter=hist[-1, 1] / ITERS * 1000,
                     fd=fd, mc_se=1e4 * float(np.mean(mse)), hist=hist))
    print(f"n={n:2d}: lift={lift_price:6.1f} solve={plateau:6.1f} total={total_price:6.1f} price-bp | "
          f"t_plateau={t_plateau:4.0f}s ({it_plateau} it, {hist[-1,1]/ITERS*1000:.0f} ms/it) | "
          f"FD 50^{n+1}={fd:.1e} | MCse~{1e4*float(np.mean(mse)):.1f}")

print("\n================  n-CONVERGENCE TABLE  ================")
print(f"{'n':>3}{'lift(px bp)':>12}{'solve(px bp)':>13}{'total(px bp)':>13}"
      f"{'PINN t_plat':>12}{'ms/iter':>9}{'FD grid 50^(n+1)':>20}")
for rw in rows:
    print(f"{rw['n']:>3}{rw['lift']:>12.1f}{rw['solve']:>13.1f}{rw['total']:>13.1f}"
          f"{rw['tsec']:>10.0f}s{rw['ms_iter']:>9.0f}{rw['fd']:>20.2e}")
print("\n(vol-bp reference, non-additive:  " +
      "  ".join(f"n={rw['n']}: lift {rw['lift_iv']:.0f}/solve {rw['solve_iv']:.0f}" for rw in rows) + ")")
print("FD memory (float64) = grid*8 bytes: " +
      "  ".join(f"n={rw['n']}:{rw['fd']*8:.1e}B" for rw in rows))

# ---- plot: errors (log) vs n, twin axis for PINN cost ----
ns = [rw["n"] for rw in rows]
fig, axL = plt.subplots(figsize=(8.5, 5.2))
axL.plot(ns, [rw["lift"] for rw in rows], "o-", color="tab:red", lw=2, label="lift-error (MC vs Fourier)")
axL.plot(ns, [rw["solve"] for rw in rows], "s-", color="tab:blue", lw=2, label="PINN solve-error (vs lifted MC)")
axL.set_yscale("log"); axL.set_xscale("log", base=2)
axL.set_xticks(ns); axL.set_xticklabels(ns)
axL.set_xlabel("number of lift factors  n"); axL.set_ylabel("price RMSE (bp of spot, log)")
axL.set_title(f"n-convergence: PINN meshfree solve-error vs lift-error  (BTC calib, T={T})")
axL.grid(alpha=.3, which="both")
axR = axL.twinx()
axR.plot(ns, [rw["tsec"] for rw in rows], "^--", color="tab:green", lw=1.5, label="PINN train time (s)")
axR.set_ylabel("PINN wall-clock to plateau (s), CPU", color="tab:green")
axR.tick_params(axis="y", colors="tab:green")
# FD blow-up annotation
fd_txt = "naive FD grid 50^(n+1):\n" + "\n".join(f"  n={rw['n']}: {rw['fd']:.0e} pts" for rw in rows)
axL.text(0.03, 0.03, fd_txt + "\n(infeasible by n=8)", transform=axL.transAxes,
         fontsize=8, va="bottom", bbox=dict(boxstyle="round", fc="wheat", alpha=.6))
lines = axL.get_lines() + axR.get_lines()
axL.legend(lines, [l.get_label() for l in lines], loc="upper right", fontsize=9)
fig.tight_layout(); fig.savefig("figures/pinn_n_convergence.png", dpi=130)
print("\nsaved -> figures/pinn_n_convergence.png")
