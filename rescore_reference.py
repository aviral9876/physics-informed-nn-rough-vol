r"""
Re-score finished PINN runs against a corrected Monte Carlo reference.
====================================================================

diagnose_hparams.py and diagnose_n32_bias.py both scored their networks against
a 1M-path lifted MC at 400 time steps. Re-measuring the Euler bias AT THE
CORRECTED CALIBRATION (diagnose_mc_quality.py) showed that reference carries
3.2-3.8 px bp of discretisation bias at n = 16 and 32 -- comparable to the
quantities being measured. Both scripts store every network's prices, so the
fix does not require retraining anything: compute a 1600-step reference once per
n and recompute every solve-error from the stored prices.

The networks themselves are untouched; only the yardstick changes. Rows at
n = 4 and 8 move by well under 1 px bp (their 400-step bias is 0.6 px bp);
rows at n = 16 and 32 are the ones this corrects.

Writes results/hparams_rescored.json and results/n32_bias_rescored.json and
leaves the originals in place, so the effect of the reference is itself on
record.
"""
import json, os, time
import numpy as np

from rough_heston_lift import lift_weights_geometric
from lifted_mc import price_european_mc
from surface import implied_vol

T = 0.15; r = 0.0
STRIKES = np.array([0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15]); KK = np.log(STRIKES)
BUCKETS = [("OTMputs", KK < -0.05), ("ATM", np.abs(KK) <= 0.05), ("OTMcalls", KK > 0.05)]
REF = dict(n_paths=1_000_000, n_steps=1600, seed=7, control_variate=True)
CACHE = "results/_ref1600_cache.json"


def pbp(a, b): return 1e4 * float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))
def ivs(px): return np.array([implied_vol(max(px[i], 1e-12), 1.0, STRIKES[i], T, r, "C") for i in range(len(STRIKES))])
def wings(d): return {nm: float(1e4 * np.nanmean(d[m])) for nm, m in BUCKETS}


def reference(n, P, H, cache):
    key = "n%d" % n
    if key not in cache:
        t0 = time.time()
        mpx, mse = price_european_mc(1.0, STRIKES, T, r, P, lift_weights_geometric(H + 0.5, n), **REF)
        cache[key] = dict(mc=np.asarray(mpx).tolist(), se=np.asarray(mse).tolist(), secs=time.time() - t0)
        json.dump(cache, open(CACHE, "w"), indent=1)
        print("  reference n=%d at %d steps in %.0f s" % (n, REF["n_steps"], time.time() - t0), flush=True)
    return np.array(cache[key]["mc"])


def load(name):
    d = json.load(open("results/%s.json" % name)); return d.get("payload", d)


if __name__ == "__main__":
    cal = json.load(open("calib_real.json"))
    P = dict(V0=cal["V0"], theta=cal["theta"], lam=cal["lam"], nu=cal["nu"], rho=cal["rho"]); H = cal["H"]
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    from resultio import dump

    # ---- hparams: every run stored its seed's price ----
    if os.path.exists("results/hparams.json"):
        hp = load("hparams")
        runs = hp["runs"]
        ns = sorted({q["n"] for q in runs.values()})
        refs = {n: reference(n, P, H, cache) for n in ns}
        old_refs = {int(k[5:]): np.array(v["mc"]) for k, v in hp["references"].items()}
        for k, q in runs.items():
            q["solve_px_bp_400"] = q["solve_px_bp"]
            q["solve_px_bp"] = pbp(q["pinn_price"], refs[q["n"]])
        def agg(cfgs):
            out = []
            for cfg in cfgs:
                sel = [q for q in runs.values() if all(q[k] == cfg[k] for k in ("n", "width", "depth", "n_col"))]
                sv = np.array([q["solve_px_bp"] for q in sel]); s4 = np.array([q["solve_px_bp_400"] for q in sel])
                rs = np.array([q["heldout_residual_rms"] for q in sel])
                out.append(dict(**cfg, solve_mean=float(sv.mean()), solve_sd=float(sv.std(ddof=1)),
                                solve_mean_400=float(s4.mean()), residual_mean=float(rs.mean()),
                                residual_sd=float(rs.std(ddof=1)),
                                train_seconds=float(np.mean([q["train_seconds"] for q in sel])),
                                n_params=sel[0]["n_params"]))
            return out
        strip = lambda rows: [{k: q[k] for k in ("n", "width", "depth", "n_col")} for q in rows]
        A, B, C = agg(strip(hp["grid"])), agg(strip(hp["collocation"])), agg(strip(hp["n_sweep"]))
        from diagnose_hparams import residual_verdict
        verdict, gs, gr = residual_verdict(C)
        print("\n==== hparams re-scored against the %d-step reference ====" % REF["n_steps"])
        print("  n    solve@400   solve@1600   residual")
        for q in C:
            print("  %2d   %8.1f   %10.1f   %.2e" % (q["n"], q["solve_mean_400"], q["solve_mean"], q["residual_mean"]))
        print("  collocation at n=32:")
        for q in B:
            print("    n_col %5d   %8.1f -> %6.1f" % (q["n_col"], q["solve_mean_400"], q["solve_mean"]))
        print("  verdict: " + verdict)
        dump("hparams_rescored", dict(reference=REF, grid=A, collocation=B, n_sweep=C, verdict=verdict,
                                      runs=runs, note="networks unchanged; solve-errors re-scored "
                                      "against a %d-step reference" % REF["n_steps"]))

    # ---- n=32 bias: every config stored per-seed prices ----
    if os.path.exists("results/n32_bias.json"):
        nb = load("n32_bias")
        ref = reference(32, P, H, cache); iv_m = ivs(ref)
        print("\n==== n=32 bias ablation re-scored ====")
        for name, c in nb["configs"].items():
            seeds = [pbp(s, ref) for s in c["pinn_price_per_seed"]]
            sw = wings(ivs(np.array(c["pinn_price"])) - iv_m)
            c["solve_mean_400"], c["signed_400"] = c["solve_mean"], c["signed_solve_iv_bp"]
            c["solve_mean"], c["solve_std"] = float(np.mean(seeds)), float(np.std(seeds))
            c["signed_solve_iv_bp"] = sw
            print("  %-28s solve %5.1f -> %5.1f   ATM %+7.1f -> %+7.1f   calls %+7.1f -> %+7.1f"
                  % (name, c["solve_mean_400"], c["solve_mean"], c["signed_400"]["ATM"], sw["ATM"],
                     c["signed_400"]["OTMcalls"], sw["OTMcalls"]))
        nb["reference_1600"] = dict(mc=ref.tolist(), cfg=REF)
        dump("n32_bias_rescored", nb)
    print("\nsaved -> results/hparams_rescored.json, results/n32_bias_rescored.json")
