r"""
The parametric operator on both boxes: solve-error across the box, and the
three-route calibration test, at the architecture the paper states.
(Reviewer 2, items 3 and 6; M-res; correction C1)
=========================================================================

Consumes the networks trained by train_parametric_boxes.py (width 96 / depth 5,
three seeds, on the original DEFAULT_BOX and on the widened box that contains
the T = 0.15-slice optimum) and repeats, for each box:

  A. solve-error against the lifted MC (n = 10) at the five probe points of
     diagnose_task5_parametric.py, plus the single-point n = 10 baseline at the
     BTC centre (three seeds, trained here once);
  B. the three-route calibration test of demo_task5_hybrid.py on the BTC
     T ~ 0.15 slice: true-forward-model loop from scratch (box-independent),
     AI-only single shot, and hybrid (AI warm start + Nelder-Mead polish).

Training both boxes turns the identifiability failure of the original box (the
AI-only fit could not reach nu ~ 0.87, theta ~ 0.39, which lay outside it) into
a controlled ablation: the same architecture, budget and seeds, with the only
difference being whether the box contains the optimum.

Also derived here, from measured quantities only:
  * the speed-up as a range (M-res): from-scratch evals / (hybrid evals +- sd);
  * the amortisation break-even (M6): parametric training seconds divided by
    the wall-clock saved per recalibration (evals saved x seconds per eval).

Writes results/param_boxes.json and results/hybrid.json (the hybrid parameter
vectors, consumed by diagnose_risk_functionals.py).
"""
import json, os, time
import numpy as np, pandas as pd, torch
from scipy.optimize import differential_evolution, minimize

from surface import build_surface, bs_price, implied_vol
from rough_heston_lift import lift_weights_geometric
from rough_heston_fourier import price_european_fourier
from lifted_mc import price_european_mc, simulate_factor_stats
from pinn import RoughHestonPINN
from parametric_pinn import ParametricPINN, DEFAULT_BOX, PNAMES
from bench_surrogate import BOX as WIDE_BOX

WIDTH, DEPTH, N_LIFT, T = 96, 5, 10, 0.15
SEEDS = (0, 1, 2); r = 0.0
BOXES = dict(default=dict(DEFAULT_BOX), wide=dict(WIDE_BOX))
CKPT = "results/_param_boxes_ckpt.json"
strikes = np.array([0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15])
VALID = [(0.02, 0.45), (0.02, 1.2), (-0.99, -0.01), (0.05, 5.0), (0.02, 0.55)]   # from-scratch region

cal = json.load(open("calib_real.json")); V0 = float(cal["V0"])
PTS = {
    "BTC-centre": [cal["H"], cal["nu"], cal["rho"], cal["lam"], cal["theta"]],
    "low-H/mild": [0.05, 0.20, -0.50, 1.5, 0.15],
    "high-nu":    [0.09, 0.55, -0.80, 2.5, 0.25],
    "weak-skew":  [0.15, 0.20, -0.30, 1.0, 0.12],
    "strong":     [0.06, 0.45, -0.90, 3.5, 0.32],
}


def pbp(a, b): return 1e4 * np.sqrt(np.nanmean((a - b) ** 2))


def load_net(box, s):
    m = ParametricPINN(box=BOXES[box], V0=V0, T=T, width=WIDTH, depth=DEPTH, n=N_LIFT)
    m.net.load_state_dict(torch.load("outputs/param_%s_w%dd%d_seed%d.pt" % (box, WIDTH, DEPTH, s)))
    return m


# ---------------- market slice for part B (as in demo_task5_hybrid.py) ----------------
surf = build_surface(pd.read_csv("data/deribit_chain.csv"), r=0.0)
Tsel = min(surf["T"].unique(), key=lambda t: abs(t - 0.15)); g = surf[surf["T"] == Tsel]
K = np.exp(g["k"].values); ivm = g["iv"].values; vega = g["weight"].values; w = vega / vega.sum()
_nf = [0]


def fourier_iv(vec):
    p = dict(V0=V0, theta=vec[4], lam=vec[3], nu=vec[1], rho=vec[2])
    px = np.atleast_1d(price_european_fourier(1.0, K, Tsel, r, p, vec[0], N=120, u_max=100, n_u=800))
    return np.array([implied_vol(px[i], 1.0, K[i], Tsel, r, "C") for i in range(len(K))])


def wrmse(iv):
    m = np.isfinite(iv); return 1e4 * np.sqrt(np.sum(w[m] * (iv[m] - ivm[m]) ** 2) / w[m].sum())


def floss(vec):
    _nf[0] += 1
    if any(v < lo or v > hi for v, (lo, hi) in zip(vec, VALID)):
        return 1e3
    iv = fourier_iv(vec); m = np.isfinite(iv)
    return 1e3 if m.sum() < 0.5 * len(K) else np.sqrt(np.sum(w[m] * (iv[m] - ivm[m]) ** 2) / w[m].sum())


def pinn_init(m, box):
    lo = torch.tensor([box[p][0] for p in PNAMES]); hi = torch.tensor([box[p][1] for p in PNAMES])
    pmkt = torch.tensor([bs_price(1.0, K[i], Tsel, ivm[i], r, "C") for i in range(len(K))])
    vg = torch.tensor(vega)
    t0 = time.perf_counter()
    raw = torch.zeros(5, requires_grad=True); opt = torch.optim.Adam([raw], lr=0.05)
    for _ in range(400):
        opt.zero_grad()
        pv = lo + (hi - lo) * torch.sigmoid(raw)
        (((m.price_smile_grad(K, T, pv) - pmkt) ** 2 / vg).sum()).backward(); opt.step()
    return (lo + (hi - lo) * torch.sigmoid(raw)).detach().numpy(), time.perf_counter() - t0


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    state = json.load(open(CKPT)) if os.path.exists(CKPT) else {}
    def save(): json.dump(state, open(CKPT, "w"), indent=1)

    # ---- references at the probe points (net-independent) ----
    if "refs" not in state:
        refs = {}
        for name, pv in PTS.items():
            Hh = pv[0]; p = dict(V0=V0, theta=pv[4], lam=pv[3], nu=pv[1], rho=pv[2])
            c, x = lift_weights_geometric(Hh + 0.5, N_LIFT)
            fpx = np.atleast_1d(price_european_fourier(1.0, strikes, T, r, p, Hh, N=300, u_max=120, n_u=1500))
            mpx, mse = price_european_mc(1.0, strikes, T, r, p, (c, x), n_paths=1_000_000, n_steps=400,
                                         seed=7, control_variate=True)
            refs[name] = dict(fpx=fpx.tolist(), mpx=mpx.tolist(), mse=mse.tolist(), lift=pbp(mpx, fpx))
            print("  ref %-12s lift %.1f px bp" % (name, refs[name]["lift"]), flush=True)
        state["refs"] = refs; save()
    refs = state["refs"]

    # ---- single-point n=10 baseline at the BTC centre (net-independent) ----
    if "single_point" not in state:
        pv0 = PTS["BTC-centre"]; p0 = dict(V0=V0, theta=pv0[4], lam=pv0[3], nu=pv0[1], rho=pv0[2])
        c0, x0 = lift_weights_geometric(pv0[0] + 0.5, N_LIFT)
        mean, std, smean, _ = simulate_factor_stats(1.0, T, p0, (c0, x0), 4000, 300, seed=1)
        sp = []
        for s in SEEDS:
            torch.manual_seed(s); np.random.seed(s)
            net = RoughHestonPINN(p0, (c0, x0), K=1.0, r=r, T=T, width=64, depth=4, x_halfwidth=1.2)
            net.set_factor_sampling(mean, std); net.set_baseline_term_structure(smean)
            t0 = time.time()
            net.train(iters=10000, n_col=2000, n_bnd=500, n_bc=400, lr=1e-3, log_every=10000)
            sp.append(dict(seed=s, solve=pbp(strikes * net.price(1.0 / strikes, tau=T), np.array(refs["BTC-centre"]["mpx"])),
                           seconds=time.time() - t0))
            print("  single-point seed %d: %.1f px bp" % (s, sp[-1]["solve"]), flush=True)
        state["single_point"] = sp; save()

    # ---- from-scratch true-model loop (box-independent) ----
    if "from_scratch" not in state:
        _nf[0] = 0; t0 = time.perf_counter()
        resF = differential_evolution(floss, VALID, maxiter=30, popsize=10, seed=1, tol=1e-4, polish=True)
        state["from_scratch"] = dict(rmse_vol_bp=float(wrmse(fourier_iv(resF.x))), seconds=time.perf_counter() - t0,
                                     evals=int(_nf[0]), params=dict(zip(PNAMES, map(float, resF.x))))
        save()
        print("  from scratch: %.1f vol bp, %d evals, %.0f s" % (state["from_scratch"]["rmse_vol_bp"],
              state["from_scratch"]["evals"], state["from_scratch"]["seconds"]), flush=True)
    F = state["from_scratch"]
    sec_per_eval = F["seconds"] / F["evals"]

    # ---- per box: A (solve-error) and B (three routes) ----
    train_log = json.load(open("results/train_parametric_boxes.json"))
    train_log = train_log.get("payload", train_log)
    train_secs = {b: float(np.mean([q["seconds"] for q in train_log["runs"] if q["box"] == b])) for b in BOXES}

    for box in BOXES:
        if box in state:
            continue
        A = {name: [] for name in PTS}; B = []
        for s in SEEDS:
            m = load_net(box, s)
            for name, pv in PTS.items():
                A[name].append(pbp(m.smile(strikes, T, pv), np.array(refs[name]["mpx"])))
            p0, tP = pinn_init(m, BOXES[box])
            only = float(wrmse(fourier_iv(p0)))
            _nf[0] = 0; t0 = time.perf_counter()
            resH = minimize(floss, p0, method="Nelder-Mead", options=dict(maxiter=120, xatol=1e-4, fatol=1e-6))
            B.append(dict(seed=s, ai_only_rmse=only, ai_only_params=dict(zip(PNAMES, map(float, p0))),
                          hybrid_rmse=float(wrmse(fourier_iv(resH.x))), hybrid_evals=int(_nf[0]),
                          hybrid_seconds=tP + time.perf_counter() - t0,
                          hybrid_params=dict(zip(PNAMES, map(float, resH.x)))))
            print("  [%s box] seed %d: centre solve %.1f px bp | AI-only %.0f | hybrid %.1f vol bp in %d evals"
                  % (box, s, A["BTC-centre"][-1], only, B[-1]["hybrid_rmse"], B[-1]["hybrid_evals"]), flush=True)
        ev = np.array([q["hybrid_evals"] for q in B]); hf = np.array([q["hybrid_rmse"] for q in B])
        saved = F["evals"] - ev.mean()
        state[box] = dict(
            solve=A, solve_summary={k: dict(mean=float(np.mean(v)), sd=float(np.std(v))) for k, v in A.items()},
            routes=B,
            speedup=dict(mean=float(F["evals"] / ev.mean()), lo=float(F["evals"] / (ev.mean() + ev.std())),
                         hi=float(F["evals"] / max(ev.mean() - ev.std(), 1.0))),
            ai_only_rmse=dict(mean=float(np.mean([q["ai_only_rmse"] for q in B])), sd=float(np.std([q["ai_only_rmse"] for q in B]))),
            hybrid_rmse=dict(mean=float(hf.mean()), sd=float(hf.std())),
            hybrid_evals=dict(mean=float(ev.mean()), sd=float(ev.std())),
            hybrid_seconds=float(np.mean([q["hybrid_seconds"] for q in B])),
            amortisation=dict(train_seconds=train_secs[box], evals_saved=float(saved),
                              seconds_saved_per_recalibration=float(saved * sec_per_eval),
                              break_even_recalibrations=float(train_secs[box] / max(saved * sec_per_eval, 1e-9))))
        save()

    sp = np.array([q["solve"] for q in state["single_point"]])
    print("\n==== parametric operator, w%d/d%d, n=%d, 3 seeds ====" % (WIDTH, DEPTH, N_LIFT))
    print("  single-point n=10 @ BTC-centre: %.1f +- %.1f px bp" % (sp.mean(), sp.std()))
    print("  %-14s %18s %18s %8s" % ("point", "default box", "wide box", "lift"))
    for name in PTS:
        d, wv = state["default"]["solve_summary"][name], state["wide"]["solve_summary"][name]
        print("  %-14s %8.1f +- %5.1f   %8.1f +- %5.1f   %6.1f" % (name, d["mean"], d["sd"], wv["mean"], wv["sd"], refs[name]["lift"]))
    print("\n  three routes on the T~0.15 slice (vol bp): from scratch %.1f in %d evals, %.0f s"
          % (F["rmse_vol_bp"], F["evals"], F["seconds"]))
    for box in BOXES:
        b = state[box]
        print("  %-8s AI-only %5.0f +- %3.0f | hybrid %5.1f +- %4.1f in %3.0f +- %2.0f evals (%.0f s) | speed-up x%.1f (%.1f-%.1f) | break-even %.1f recalibrations"
              % (box, b["ai_only_rmse"]["mean"], b["ai_only_rmse"]["sd"], b["hybrid_rmse"]["mean"], b["hybrid_rmse"]["sd"],
                 b["hybrid_evals"]["mean"], b["hybrid_evals"]["sd"], b["hybrid_seconds"], b["speedup"]["mean"],
                 b["speedup"]["lo"], b["speedup"]["hi"], b["amortisation"]["break_even_recalibrations"]))

    from resultio import dump
    dump("param_boxes", dict(width=WIDTH, depth=DEPTH, n=N_LIFT, T=T, seeds=list(SEEDS), boxes=BOXES,
                             probe_points=PTS, refs=refs, single_point=state["single_point"],
                             from_scratch=F, sec_per_eval=sec_per_eval,
                             per_box={b: state[b] for b in BOXES}))
    # hybrid vectors for the risk-functional propagation (R1.8): the wide box, which contains the optimum
    vecs = []
    for q in state["wide"]["routes"]:
        hp = q["hybrid_params"]
        vecs.append(dict(H=hp["H"], V0=V0, theta=hp["theta"], lam=hp["lam"], nu=hp["nu"], rho=hp["rho"], seed=q["seed"]))
    fs = F["params"]
    dump("hybrid", dict(vectors=vecs, from_scratch=dict(H=fs["H"], V0=V0, theta=fs["theta"], lam=fs["lam"],
                                                        nu=fs["nu"], rho=fs["rho"]), slice_T=float(Tsel)))
    print("\nsaved -> results/param_boxes.json, results/hybrid.json")
