r"""
J4: hyperparameter sensitivity, collocation ablation, and where the error
growth with n actually lives. (Reviewer 2, item 9; Reviewer 1, item 3)
=========================================================================

Three parts, one checkpoint file, all against the 1M-path lifted MC at the same
n (J1's reference budget). Two seeds per configuration, 10k iterations.

  A. ARCHITECTURE GRID at n = 4: width in {32, 64, 128} x depth in {3, 4, 5}.
     Answers M9 directly: is the paper's width-64 / depth-4 choice a knife-edge?

  B. COLLOCATION ABLATION at n = 32: n_col in {2000, 4000, 8000}.
     The paper holds n_col at 2000 while the domain dimension goes 5 -> 33. In
     De Ryck & Mishra's error decomposition the quadrature (generalisation) term
     depends on the number of collocation points relative to the dimension, so
     if the growth with n is quadrature-driven, more points should buy it back.

  C. RESIDUAL PERSISTENCE across n in {4, 8, 16, 32}: for every trained net we
     evaluate the PDE residual on a FIXED held-out set of 20,000 collocation
     points (seeded, never trained on). This is the measurement R1.3 asks for.
     The a-priori bound is  solve-error <= C(n) * (residual + boundary terms).
     If the held-out residual is flat in n while the solve-error grows, the
     growth sits in the stability constant C(n) -- a property of the lifted
     PDE, not of the optimiser. If the residual grows with the solve-error, it
     is optimisation or quadrature and more training / points should fix it.

Every run is checkpointed by its key, so a crash resumes. ~6 h on CPU.
"""
import json, os, time
import numpy as np, torch

from rough_heston_lift import lift_weights_geometric
from rough_heston_fourier import price_european_fourier
from lifted_mc import price_european_mc, simulate_factor_stats
from surface import implied_vol
from pinn import RoughHestonPINN

T = 0.15; r = 0.0; ITERS = 10000; SEEDS = (0, 1)
strikes = np.array([0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15])
MC = dict(n_paths=1_000_000, n_steps=400, seed=7, control_variate=True)
HELDOUT_N, HELDOUT_SEED = 20000, 12345
CKPT = "results/_hparams_ckpt.json"

PART_A = [dict(n=4, width=w, depth=d, n_col=2000) for w in (32, 64, 128) for d in (3, 4, 5)]
PART_B = [dict(n=32, width=64, depth=4, n_col=c) for c in (2000, 4000, 8000)]
PART_C = [dict(n=n, width=64, depth=4, n_col=2000) for n in (4, 8, 16, 32)]


def key(cfg, s): return "n%d_w%d_d%d_col%d_s%d" % (cfg["n"], cfg["width"], cfg["depth"], cfg["n_col"], s)
def pbp(a, b): return 1e4 * np.sqrt(np.mean((a - b) ** 2))
def ivs(px): return np.array([implied_vol(max(px[i], 1e-12), 1.0, strikes[i], T, r, "C") for i in range(len(strikes))])


def heldout_residual(p):
    """RMS PDE residual on a fixed, seeded, never-trained-on collocation set."""
    g = torch.random.get_rng_state()
    torch.manual_seed(HELDOUT_SEED)
    tau, x, u = p.sample(HELDOUT_N, 0)
    torch.random.set_rng_state(g)
    sq, cnt = 0.0, 0
    for i in range(0, HELDOUT_N, 2000):
        res = p.pde_residual(tau[i:i + 2000], x[i:i + 2000], u[i:i + 2000]).detach()
        sq += float((res ** 2).sum()); cnt += res.numel()
    return float(np.sqrt(sq / cnt))


def n_params(p): return int(sum(t.numel() for t in p.net.parameters()))


if __name__ == "__main__":
    cal = json.load(open("calib_real.json"))
    P = dict(V0=cal["V0"], theta=cal["theta"], lam=cal["lam"], nu=cal["nu"], rho=cal["rho"]); H = cal["H"]
    os.makedirs("results", exist_ok=True)
    state = json.load(open(CKPT)) if os.path.exists(CKPT) else {}

    def save(): json.dump(state, open(CKPT, "w"), indent=1)

    lifts, refs, stats = {}, {}, {}
    def prepare(n):
        if n in lifts: return
        lifts[n] = lift_weights_geometric(H + 0.5, n)
        rk = "ref_n%d" % n
        if rk not in state:
            t0 = time.time()
            mpx, mse = price_european_mc(1.0, strikes, T, r, P, lifts[n], **MC)
            fpx = np.atleast_1d(price_european_fourier(1.0, strikes, T, r, P, H, N=300, u_max=120, n_u=1500))
            state[rk] = dict(mc=mpx.tolist(), mc_se=mse.tolist(), fourier=fpx.tolist(), secs=time.time() - t0)
            save()
            print("reference n=%d in %.0f s" % (n, time.time() - t0), flush=True)
        refs[n] = np.array(state[rk]["mc"])
        stats[n] = simulate_factor_stats(1.0, T, P, lifts[n], 4000, 300, seed=1)

    runs = []
    for cfg in PART_A + PART_B + PART_C:
        for s in SEEDS:
            k = key(cfg, s)
            if k not in [key(c, ss) for c, ss in runs]:
                runs.append((cfg, s))
    print("%d unique runs" % len(runs), flush=True)

    for cfg, s in runs:
        k = key(cfg, s)
        if k in state:
            continue
        n = cfg["n"]; prepare(n)
        mean, std, smean, _ = stats[n]
        torch.manual_seed(s); np.random.seed(s)
        p = RoughHestonPINN(P, lifts[n], K=1.0, r=r, T=T, width=cfg["width"], depth=cfg["depth"], x_halfwidth=1.2)
        p.set_factor_sampling(mean, std); p.set_baseline_term_structure(smean)
        stack = []
        def probe(it, net):
            if it >= 0.8 * ITERS:
                stack.append(strikes * net.price(1.0 / strikes, tau=T))
        t0 = time.time()
        p.train(iters=ITERS, n_col=cfg["n_col"], n_bnd=500, n_bc=400, lr=1e-3, log_every=500, on_log=probe)
        secs = time.time() - t0
        ppx = np.mean(np.array(stack), axis=0)
        solve = pbp(ppx, refs[n])
        resid = heldout_residual(p)
        state[k] = dict(**cfg, seed=s, solve_px_bp=float(solve), heldout_residual_rms=resid,
                        train_seconds=secs, n_params=n_params(p), pinn_price=ppx.tolist(),
                        signed_solve_iv_bp=(1e4 * (ivs(ppx) - ivs(refs[n]))).tolist())
        save()
        print("%-24s solve %5.1f px bp  residual %.3e  %5.1f min  (%d params)"
              % (k, solve, resid, secs / 60, n_params(p)), flush=True)

    def agg(cfgs):
        out = []
        for cfg in cfgs:
            rows = [state[key(cfg, s)] for s in SEEDS]
            sv = np.array([q["solve_px_bp"] for q in rows]); rs = np.array([q["heldout_residual_rms"] for q in rows])
            out.append(dict(**cfg, solve_mean=float(sv.mean()), solve_sd=float(sv.std(ddof=1)),
                            residual_mean=float(rs.mean()), residual_sd=float(rs.std(ddof=1)),
                            train_seconds=float(np.mean([q["train_seconds"] for q in rows])),
                            n_params=rows[0]["n_params"]))
        return out

    A, B, C = agg(PART_A), agg(PART_B), agg(PART_C)
    print("\n==== A. architecture grid, n=4 (solve px bp, mean +- sd over %d seeds) ====" % len(SEEDS))
    print("  width  depth  params   solve          residual        min/run")
    for q in A:
        print("  %5d  %5d  %6d   %5.1f +- %4.1f   %.2e +- %.1e   %4.1f"
              % (q["width"], q["depth"], q["n_params"], q["solve_mean"], q["solve_sd"], q["residual_mean"], q["residual_sd"], q["train_seconds"] / 60))
    print("\n==== B. collocation ablation, n=32 ====")
    print("  n_col   solve          residual        min/run")
    for q in B:
        print("  %5d   %5.1f +- %4.1f   %.2e +- %.1e   %4.1f"
              % (q["n_col"], q["solve_mean"], q["solve_sd"], q["residual_mean"], q["residual_sd"], q["train_seconds"] / 60))
    print("\n==== C. residual persistence across n (w64/d4, n_col=2000) ====")
    print("  n    solve          residual        solve/solve(4)  resid/resid(4)")
    s4, r4 = C[0]["solve_mean"], C[0]["residual_mean"]
    for q in C:
        print("  %2d   %5.1f +- %4.1f   %.2e +- %.1e   %5.2f           %5.2f"
              % (q["n"], q["solve_mean"], q["solve_sd"], q["residual_mean"], q["residual_sd"], q["solve_mean"] / s4, q["residual_mean"] / r4))
    gs, gr = C[-1]["solve_mean"] / s4, C[-1]["residual_mean"] / r4
    verdict = ("the residual is roughly flat while the solve-error grows: the growth sits in the stability constant C(n)"
               if gr < 0.5 * gs else
               "the residual grows with the solve-error: the growth is optimisation/quadrature, not the PDE's stability constant")
    print("\n  n=4 -> 32: solve x%.2f, residual x%.2f  =>  %s" % (gs, gr, verdict))

    from resultio import dump
    dump("hparams", dict(T=T, iters=ITERS, seeds=list(SEEDS), strikes=strikes.tolist(), mc=MC,
                         heldout=dict(n=HELDOUT_N, seed=HELDOUT_SEED),
                         grid=A, collocation=B, n_sweep=C, verdict=verdict,
                         runs={k: v for k, v in state.items() if not k.startswith("ref_")},
                         references={k: v for k, v in state.items() if k.startswith("ref_")}))
    print("saved -> results/hparams.json")
