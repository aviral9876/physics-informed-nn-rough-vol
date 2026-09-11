r"""
Per-iteration training cost of the single-point PINN across the lift dimension n,
measured on an otherwise idle machine. (M7; Table 2's cost column)
================================================================================

The seed sweep does not time its runs, and the only timings on record came from
the hyperparameter grid, which ran concurrently with three other jobs: its
figures (1530 s at n=4, 522 s at n=32) are contention artefacts, not costs.
This measures the one thing Table 2 should report -- milliseconds per training
iteration at the paper's architecture and collocation budget -- with nothing
else running, a short warm-up excluded, and the median over repeats.

A full 10k-iteration run costs about 10^4 times the per-iteration figure.
"""
import json, os, time
import numpy as np, torch

from rough_heston_lift import lift_weights_geometric
from lifted_mc import simulate_factor_stats
from pinn import RoughHestonPINN

NLIST = [4, 8, 16, 32]; T = 0.15; r = 0.0
WARM, TIMED, REPEATS = 50, 200, 3


if __name__ == "__main__":
    cal = json.load(open("calib_real.json"))
    P = dict(V0=cal["V0"], theta=cal["theta"], lam=cal["lam"], nu=cal["nu"], rho=cal["rho"]); H = cal["H"]
    threads = torch.get_num_threads()
    out = []
    for n in NLIST:
        lift = lift_weights_geometric(H + 0.5, n)
        mean, std, smean, _ = simulate_factor_stats(1.0, T, P, lift, 4000, 300, seed=1)
        reps = []
        for rep in range(REPEATS):
            torch.manual_seed(rep)
            p = RoughHestonPINN(P, lift, K=1.0, r=r, T=T, width=64, depth=4, x_halfwidth=1.2)
            p.set_factor_sampling(mean, std); p.set_baseline_term_structure(smean)
            p.train(iters=WARM, n_col=2000, n_bnd=500, n_bc=400, lr=1e-3, log_every=10 ** 9)
            t0 = time.perf_counter()
            p.train(iters=TIMED, n_col=2000, n_bnd=500, n_bc=400, lr=1e-3, log_every=10 ** 9)
            reps.append(1e3 * (time.perf_counter() - t0) / TIMED)
        ms = float(np.median(reps))
        out.append(dict(n=n, ms_per_iter=ms, ms_per_iter_reps=reps, est_seconds_10k=ms * 10))
        print("n=%2d  %.1f ms/iter  (repeats %s)  -> ~%.0f s per 10k-iteration run"
              % (n, ms, ", ".join("%.1f" % v for v in reps), ms * 10), flush=True)

    from resultio import dump
    dump("n_cost", dict(rows=out, threads=threads, width=64, depth=4, n_col=2000,
                        warmup_iters=WARM, timed_iters=TIMED, repeats=REPEATS,
                        train_seconds=[q["est_seconds_10k"] for q in out],
                        note="measured with no other job running"))
    print("threads: %d; saved -> results/n_cost.json" % threads)
