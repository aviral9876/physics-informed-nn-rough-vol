r"""
The directional bias at n = 32: which mechanism? (Reviewer 1, item 7)
=====================================================================

At n = 32 the PINN over-prices ATM and call strikes against the lifted MC
(signed solve-error ~ +87 / +114 vol bp) while the put wing is near zero, and
the bias grows with n. The referee asks for the mechanism and, in particular,
whether the factor-tail coverage is asymmetric.

On the second point we correct the premise: tail coverage is symmetric by
default. pinn.py applies `tail_scale` to both signs of the factor deviation and
`tail_scale_low` defaults to None (pinn.py, set_factor_sampling and the sampler
that follows it). So asymmetric tails cannot be the cause. Two other mechanisms
are plausible and are separable by ablation:

  (a) A NOISY BASELINE ANCHOR. The BS baseline is anchored at the lifted forward
      variance w(tau) = int_0^tau E[V_s] ds, estimated by a pre-flight MC with
      4,000 paths / 300 steps. At n = 32 the factor system is stiff and the
      estimate of w(tau) is noisier; a level error in the anchor is a level
      error in the price, which is exactly what an ATM/call bias looks like.
      Ablation: re-estimate the anchor with 200,000 paths / 400 steps.
  (b) UNDER-COVERAGE OF THE FACTOR TAILS in 32 dimensions, with the default
      20% tail fraction at 2x width. Ablation: 35% at 3x, both signs.

We train 3 seeds per configuration at n = 32 and report the solve-error and the
signed ATM / call / put errors against the same 1M-path lifted MC reference that
J1 uses. The Euler time-discretisation bias of that reference at n = 32, from
diagnose_mc_quality.py, is printed alongside: if the reference itself sits low by
a few bp (finer grids price BELOW the 400-step grid), part of the apparent
over-pricing is the reference, not the network.

Each configuration is checkpointed on completion. ~2 h on CPU.
"""
import json, os, time
import numpy as np, torch

from rough_heston_lift import lift_weights_geometric
from rough_heston_fourier import price_european_fourier
from lifted_mc import price_european_mc, simulate_factor_stats
from surface import implied_vol
from pinn import RoughHestonPINN

N_LIFT = 32; T = 0.15; r = 0.0; ITERS = 10000
SEEDS = (0, 1, 2)
strikes = np.array([0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15]); kk = np.log(strikes)
buckets = [("OTMputs", kk < -0.05), ("ATM", np.abs(kk) <= 0.05), ("OTMcalls", kk > 0.05)]
CKPT = "results/_n32_bias_ckpt.json"

CONFIGS = {
    "baseline (J1 setting)":     dict(anchor=(4000, 300),    tail_frac=0.20, tail_scale=2.0),
    "precise anchor":            dict(anchor=(200_000, 400), tail_frac=0.20, tail_scale=2.0),
    "wide tails":                dict(anchor=(4000, 300),    tail_frac=0.35, tail_scale=3.0),
    "precise anchor + wide tails": dict(anchor=(200_000, 400), tail_frac=0.35, tail_scale=3.0),
}


def pbp(a, b): return 1e4 * np.sqrt(np.mean((a - b) ** 2))
def ivs(px): return np.array([implied_vol(max(px[i], 1e-12), 1.0, strikes[i], T, r, "C") for i in range(len(strikes))])
def wings(d): return {nm: float(1e4 * np.nanmean(d[m])) for nm, m in buckets}


if __name__ == "__main__":
    cal = json.load(open("calib_real.json"))
    P = dict(V0=cal["V0"], theta=cal["theta"], lam=cal["lam"], nu=cal["nu"], rho=cal["rho"]); H = cal["H"]
    c, x = lift_weights_geometric(H + 0.5, N_LIFT)
    os.makedirs("results", exist_ok=True)
    state = json.load(open(CKPT)) if os.path.exists(CKPT) else {}

    # reference: same as J1 (1M antithetic paths, 400 steps, control variate)
    if "reference" not in state:
        t0 = time.time()
        fpx = np.atleast_1d(price_european_fourier(1.0, strikes, T, r, P, H, N=300, u_max=120, n_u=1500))
        mpx, mse = price_european_mc(1.0, strikes, T, r, P, (c, x), n_paths=1_000_000,
                                     n_steps=400, seed=7, control_variate=True)
        state["reference"] = dict(fourier=fpx.tolist(), mc=mpx.tolist(), mc_se=mse.tolist(),
                                  secs=time.time() - t0)
        json.dump(state, open(CKPT, "w"), indent=1)
        print("reference MC in %.0f s; lift-error %.1f px bp" % (time.time() - t0, pbp(mpx, fpx)), flush=True)
    fpx = np.array(state["reference"]["fourier"]); mpx = np.array(state["reference"]["mc"])
    iv_m, iv_f = ivs(mpx), ivs(fpx)

    euler = None
    if os.path.exists("results/mc_quality.json"):
        q = json.load(open("results/mc_quality.json")); q = q.get("payload", q)
        for row in q.get("headline", []):          # carries euler_bias_400_px_bp per n
            if int(row.get("n", -1)) == N_LIFT:
                euler = row
                break

    for name, cfg in CONFIGS.items():
        if name in state:
            print("skip %s (checkpointed)" % name, flush=True); continue
        t0 = time.time()
        mean, std, smean, _ = simulate_factor_stats(1.0, T, P, (c, x), cfg["anchor"][0], cfg["anchor"][1], seed=1)
        solves, smiles = [], []
        for s in SEEDS:
            torch.manual_seed(s); np.random.seed(s)
            p = RoughHestonPINN(P, (c, x), K=1.0, r=r, T=T, width=64, depth=4, x_halfwidth=1.2)
            p.set_factor_sampling(mean, std, tail_frac=cfg["tail_frac"], tail_scale=cfg["tail_scale"])
            p.set_baseline_term_structure(smean)
            stack = []
            def probe(it, net):
                if it >= 0.8 * ITERS:
                    stack.append(strikes * net.price(1.0 / strikes, tau=T))
            p.train(iters=ITERS, n_col=2000, n_bnd=500, n_bc=400, lr=1e-3, log_every=2500, on_log=probe)
            ppx = np.mean(np.array(stack), axis=0)
            solves.append(pbp(ppx, mpx)); smiles.append(ppx.tolist())
        ppx_avg = np.mean(np.array(smiles), axis=0); iv_p = ivs(ppx_avg)
        sw = wings(iv_p - iv_m)
        state[name] = dict(config=cfg, anchor_EV_at_T=float(np.asarray(smean).ravel()[-1]),   # last-step E[V], a checksum on the anchor
                           solves=[float(v) for v in solves], solve_mean=float(np.mean(solves)),
                           solve_std=float(np.std(solves)), signed_solve_iv_bp=sw,
                           pinn_price_per_seed=smiles, pinn_price=ppx_avg.tolist(),
                           signed_solve_iv_bp_per_strike=(1e4 * (iv_p - iv_m)).tolist(),
                           secs=time.time() - t0)
        json.dump(state, open(CKPT, "w"), indent=1)
        print("%-28s solve %5.1f +- %4.1f px bp   signed vol bp: puts %+6.1f  ATM %+6.1f  calls %+6.1f   (%.0f min)"
              % (name, np.mean(solves), np.std(solves), sw["OTMputs"], sw["ATM"], sw["OTMcalls"], (time.time() - t0) / 60),
              flush=True)

    print("\n==== n=32 bias ablation, 3 seeds, vs 1M-path lifted MC ====")
    print("%-28s %14s %9s %9s %9s" % ("configuration", "solve px bp", "puts", "ATM", "calls"))
    for name in CONFIGS:
        d = state[name]; sw = d["signed_solve_iv_bp"]
        print("%-28s %6.1f +- %4.1f %9.1f %9.1f %9.1f" % (name, d["solve_mean"], d["solve_std"],
                                                        sw["OTMputs"], sw["ATM"], sw["OTMcalls"]))
    if euler is not None:
        print("\n  reference Euler bias at n=32 (diagnose_mc_quality.py):", {k: v for k, v in euler.items() if "bias" in k or "euler" in k.lower()})

    from resultio import dump
    dump("n32_bias", dict(n=N_LIFT, T=T, iters=ITERS, seeds=list(SEEDS), strikes=strikes.tolist(),
                          reference=state["reference"], iv_mc=iv_m.tolist(), iv_fourier=iv_f.tolist(),
                          configs={k: state[k] for k in CONFIGS}, euler_bias_row=euler,
                          note="tail coverage is symmetric by default in pinn.py (tail_scale_low=None); "
                               "the referee's asymmetry hypothesis is ruled out by construction"))
    print("saved -> results/n32_bias.json")
