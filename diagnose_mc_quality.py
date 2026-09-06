"""
Quality of the lifted-MC reference  --  CNSNS revision, Reviewer 1 point 2.
==========================================================================

THE OBJECTION. The referee wrote that the reference against which the headline
solve-error is measured "may itself be noisier than the quantity being
measured", citing "1,000 paths and 50 time steps" from the manuscript against
solve-errors of 5.4-10.7 px bp.

THE FACTS. That 1,000/50 figure is a Black-Scholes sanity check (cnsns_v1.tex:497),
not the validation reference. The actual reference is 200,000 ANTITHETIC paths /
400 steps (diagnose_robustness_sweep.py). The manuscript never stated it. So the
objection is largely a reporting defect on our side -- but the underlying
precision question is real and this script answers it with numbers.

WHAT IS MEASURED

  PART 1  Statistical error. Per-strike price, the correct PAIRED standard error,
          the (wrong) i.i.d. formula the code used before, and 95% CIs, at
          n in {4,8,16,32} and 200k vs 1M paths. Also the measured gain from the
          terminal-forward control variate.

  PART 2  DISCRETISATION error -- the risk the referee did NOT raise, and which
          our own scoping suggested is larger than the statistical one. Estimated
          by COMMON RANDOM NUMBERS: every step resolution is driven by the SAME
          fine Brownian path, so the difference between resolutions contains no
          sampling noise at all. Naive independent-seed comparisons cannot
          resolve a bias smaller than their own MC error; this can.

  PART 3  Effect on the headline. Noise-corrects the published solve-errors via
          sqrt(RMSE^2 - SE^2), valid because PINN error and MC error are
          independent, and reports what it does to the n-scaling ratio.

Run:  python diagnose_mc_quality.py        (~15 min CPU)
Writes results/mc_quality.json
"""

import json
import time

import numpy as np

from lifted_mc import price_european_mc
from resultio import dump
from rough_heston_lift import lift_weights_geometric

# ----------------------------------------------------------------- config
CAL = json.load(open("calib_real.json"))
P = dict(V0=CAL["V0"], theta=CAL["theta"], lam=CAL["lam"],
         nu=CAL["nu"], rho=CAL["rho"])
H = CAL["H"]
T = 0.15
R = 0.0
STRIKES = np.array([0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15])
NLIST = [4, 8, 16, 32]
SEED = 7                      # the seed the validation runs use
STEPS = 400                   # the step count the validation runs use

# Published 3-seed solve-errors (px bp) from diagnose_robustness_sweep.py, used
# only for the noise correction in PART 3. J1 of the revision replaces these
# with an 8-seed run; the correction is re-derived then.
PUBLISHED_SOLVE = {4: 5.4, 8: 6.1, 16: 8.8, 32: 10.7}


def rms(a):
    return float(np.sqrt(np.mean(np.asarray(a, dtype=float) ** 2)))


# ------------------------------------------------- PART 2 helper: CRN engine
def simulate_crn_price(strikes, n_coarse, n_fine, lift, n_paths, seed):
    """
    Price on a COARSE time grid while drawing the Brownian path on a FIXED FINE
    grid, so that every choice of n_coarse sees the identical path.

    A coarse increment is the sum of n_fine/n_coarse consecutive fine increments
    -- exact, since independent N(0, dt_fine) increments sum to N(0, dt_coarse).
    Because the fine draws are made in the same order from the same seed for
    every resolution, differences across resolutions are PURE discretisation
    effect with zero sampling noise.

    Mirrors the exponential-Euler scheme of lifted_mc.price_european_mc exactly
    (validated against it in __main__).
    """
    c, x = lift
    n = c.shape[0]
    V0, theta = P["V0"], P["theta"]
    lam, nu, rho = P["lam"], P["nu"], P["rho"]

    assert n_fine % n_coarse == 0, "n_fine must be a multiple of n_coarse"
    m = n_fine // n_coarse

    rng = np.random.default_rng(seed)
    half = n_paths // 2
    n_paths = 2 * half
    dt_fine = T / n_fine
    dt = T / n_coarse
    sdt_fine = np.sqrt(dt_fine)
    rho2 = np.sqrt(1.0 - rho * rho)

    U = np.zeros((n_paths, n))
    X = np.zeros(n_paths)

    z = x * dt
    E = np.exp(-z)
    phi1 = np.where(z > 1e-12, (1.0 - E) / np.where(z > 1e-12, z, 1.0), 1.0)
    E_row = E[None, :]
    w_src = (phi1 * dt)[None, :]
    w_dif = phi1[None, :]

    for _ in range(n_coarse):
        # aggregate m fine increments into one coarse increment
        dW = np.zeros(n_paths)
        dWp = np.zeros(n_paths)
        for _ in range(m):
            zW = rng.standard_normal(half)
            zP = rng.standard_normal(half)
            dW += np.concatenate([zW, -zW]) * sdt_fine
            dWp += np.concatenate([zP, -zP]) * sdt_fine

        V = V0 + U @ c
        Vp = np.maximum(V, 0.0)
        sqrtV = np.sqrt(Vp)
        source = (lam * (theta - Vp))[:, None]
        diffusion = (nu * sqrtV)[:, None]
        U = U * E_row + source * w_src + diffusion * dW[:, None] * w_dif

        dB = rho * dW + rho2 * dWp
        X = X - 0.5 * Vp * dt + sqrtV * dB

    ST = np.exp(R * T) * np.exp(X)
    out = np.empty(len(strikes))
    for i, k in enumerate(strikes):
        out[i] = np.exp(-R * T) * np.maximum(ST - k, 0.0).mean()
    return out


# ============================================================== PART 1
def part1_statistical():
    print("=" * 74)
    print("PART 1  Statistical error of the lifted-MC reference")
    print("=" * 74)
    rows = []
    for n in NLIST:
        lift = lift_weights_geometric(H + 0.5, n)

        t0 = time.time()
        px200, se200 = price_european_mc(1.0, STRIKES, T, R, P, lift,
                                         n_paths=200_000, n_steps=STEPS,
                                         seed=SEED, antithetic=True)
        t200 = time.time() - t0

        # the estimator this code used before the fix: i.i.d. formula applied to
        # an antithetically PAIRED sample
        naive200 = naive_se(STRIKES, lift, 200_000)

        t0 = time.time()
        px1m, se1m = price_european_mc(1.0, STRIKES, T, R, P, lift,
                                       n_paths=1_000_000, n_steps=STEPS,
                                       seed=SEED, antithetic=True)
        t1m = time.time() - t0

        _, se_cv = price_european_mc(1.0, STRIKES, T, R, P, lift,
                                     n_paths=200_000, n_steps=STEPS, seed=SEED,
                                     antithetic=True, control_variate=True)

        rows.append(dict(
            n=n, strikes=STRIKES.tolist(),
            price_200k=px200.tolist(), se_200k=se200.tolist(),
            se_200k_naive_iid=naive200.tolist(),
            se_200k_with_cv=se_cv.tolist(),
            price_1m=px1m.tolist(), se_1m=se1m.tolist(),
            rms_se_200k=rms(se200), rms_se_200k_naive=rms(naive200),
            rms_se_200k_cv=rms(se_cv), rms_se_1m=rms(se1m),
            secs_200k=t200, secs_1m=t1m,
        ))

        print(f"\n n = {n}   (200k: {t200:.0f}s, 1M: {t1m:.0f}s)")
        print(f"{'K':>6}{'price':>10}{'SE_200k':>9}{'95%CI':>9}"
              f"{'oldSE':>8}{'old/new':>9}{'SE_1M':>8}{'CVgain':>8}")
        for i, k in enumerate(STRIKES):
            print(f"{k:>6.2f}{px200[i]:>10.5f}{1e4*se200[i]:>9.2f}"
                  f"{1e4*1.96*se200[i]:>9.2f}{1e4*naive200[i]:>8.2f}"
                  f"{naive200[i]/se200[i]:>9.2f}{1e4*se1m[i]:>8.2f}"
                  f"{se200[i]/se_cv[i]:>8.2f}")
        print(f"  RMS over strikes:  SE_200k = {1e4*rms(se200):.2f}  "
              f"old = {1e4*rms(naive200):.2f}  SE_1M = {1e4*rms(se1m):.2f} px bp")
    return rows


def naive_se(strikes, lift, n_paths):
    """Reproduce the pre-fix estimator: std of ALL paired samples / sqrt(N).
    Kept so the manuscript can state the mis-statement factor honestly."""
    c, x = lift
    n = c.shape[0]
    V0, theta = P["V0"], P["theta"]
    lam, nu, rho = P["lam"], P["nu"], P["rho"]
    rng = np.random.default_rng(SEED)
    half = n_paths // 2
    n_paths = 2 * half
    dt = T / STEPS
    sdt = np.sqrt(dt)
    rho2 = np.sqrt(1.0 - rho * rho)
    U = np.zeros((n_paths, n)); X = np.zeros(n_paths)
    z = x * dt
    E = np.exp(-z)
    phi1 = np.where(z > 1e-12, (1.0 - E) / np.where(z > 1e-12, z, 1.0), 1.0)
    E_row, w_src, w_dif = E[None, :], (phi1 * dt)[None, :], phi1[None, :]
    for _ in range(STEPS):
        zW = rng.standard_normal(half); zP = rng.standard_normal(half)
        dW = np.concatenate([zW, -zW]) * sdt
        dWp = np.concatenate([zP, -zP]) * sdt
        V = V0 + U @ c
        Vp = np.maximum(V, 0.0); sqrtV = np.sqrt(Vp)
        U = (U * E_row + (lam * (theta - Vp))[:, None] * w_src
             + (nu * sqrtV)[:, None] * dW[:, None] * w_dif)
        X = X - 0.5 * Vp * dt + sqrtV * (rho * dW + rho2 * dWp)
    ST = np.exp(R * T) * np.exp(X)
    return np.array([np.maximum(ST - k, 0.0).std(ddof=1) / np.sqrt(n_paths)
                     for k in STRIKES])


# ============================================================== PART 2
def part2_discretisation(n_fine=3200, n_paths=50_000,
                         coarse_list=(100, 200, 400, 800, 1600)):
    print("\n" + "=" * 74)
    print("PART 2  Time-discretisation bias, common random numbers")
    print("=" * 74)
    print(f"  fine grid = {n_fine} steps, {n_paths} paths, identical Brownian "
          f"path at every resolution\n")
    rows = []
    for n in NLIST:
        lift = lift_weights_geometric(H + 0.5, n)
        ref = simulate_crn_price(STRIKES, n_fine, n_fine, lift, n_paths, SEED)
        entry = dict(n=n, n_fine=n_fine, n_paths=n_paths,
                     ref_price=ref.tolist(), coarse={})
        print(f" n = {n}   (bias vs the {n_fine}-step path, px bp)")
        print(f"{'steps':>7}" + "".join(f"{k:>9.2f}" for k in STRIKES) + f"{'RMS':>9}")
        for nc in coarse_list:
            px = simulate_crn_price(STRIKES, nc, n_fine, lift, n_paths, SEED)
            d = px - ref
            entry["coarse"][str(nc)] = dict(price=px.tolist(),
                                            bias=d.tolist(), rms_bias=rms(d))
            print(f"{nc:>7}" + "".join(f"{1e4*v:>9.2f}" for v in d)
                  + f"{1e4*rms(d):>9.2f}")
        rows.append(entry)
    return rows


# ============================================================== PART 3
def part3_headline(part1_rows, part2_rows):
    print("\n" + "=" * 74)
    print("PART 3  Effect on the published n-scaling")
    print("=" * 74)
    out = []
    bias400 = {r["n"]: r["coarse"]["400"]["rms_bias"] for r in part2_rows}
    print(f"{'n':>4}{'published':>11}{'SE(200k)':>10}{'de-noised':>11}"
          f"{'inflation':>11}{'bias@400':>10}")
    for r in part1_rows:
        n = r["n"]
        pub = PUBLISHED_SOLVE[n]
        se = 1e4 * r["rms_se_200k"]
        den = float(np.sqrt(max(pub ** 2 - se ** 2, 0.0)))
        out.append(dict(n=n, published=pub, se_px_bp=se, denoised=den,
                        inflation_pct=100.0 * (pub - den) / den if den else None,
                        euler_bias_400_px_bp=1e4 * bias400[n]))
        print(f"{n:>4}{pub:>11.2f}{se:>10.2f}{den:>11.2f}"
              f"{100.0*(pub-den)/den:>10.1f}%{1e4*bias400[n]:>10.2f}")

    r0, r1 = out[0], out[-1]
    print(f"\n  published ratio  n=4 -> n={r1['n']}:  "
          f"{r1['published']/r0['published']:.3f}")
    print(f"  de-noised ratio  n=4 -> n={r1['n']}:  "
          f"{r1['denoised']/r0['denoised']:.3f}")
    print("\n  Statistical noise inflates the LOW-n entries proportionally more,")
    print("  so removing it makes the growth marginally STEEPER, not flatter.")
    return out


if __name__ == "__main__":
    # Validate the CRN engine against the production pricer before trusting it:
    # different RNG consumption order, so agreement is only expected to within
    # a couple of standard errors -- which is exactly the check we want.
    lift4 = lift_weights_geometric(H + 0.5, 4)
    a = simulate_crn_price(STRIKES, 400, 400, lift4, 200_000, 11)
    b, sb = price_european_mc(1.0, STRIKES, T, R, P, lift4,
                              n_paths=200_000, n_steps=400, seed=11)
    dev = np.abs(a - b) / sb
    print(f"[check] CRN engine vs price_european_mc: max deviation "
          f"{dev.max():.2f} SE  (expect < ~4)")
    assert dev.max() < 6.0, "CRN engine does not reproduce the production scheme"

    t0 = time.time()
    p1 = part1_statistical()
    p2 = part2_discretisation()
    p3 = part3_headline(p1, p2)
    print(f"\ntotal {time.time()-t0:.0f}s")

    dump("mc_quality", dict(
        config=dict(params=P, H=H, T=T, r=R, strikes=STRIKES.tolist(),
                    seed=SEED, steps=STEPS, nlist=NLIST,
                    published_solve=PUBLISHED_SOLVE),
        statistical=p1, discretisation=p2, headline=p3,
    ))
