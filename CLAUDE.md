# CLAUDE.md — Project context for Claude Code

This file is auto-loaded by Claude Code. It tells you (Claude) the state of this
project, what is trusted, what is broken, and what to work on. Read it fully
before touching code.

## What this project is

A research pipeline for the thesis **"Physics-Informed Neural Networks for
Option Pricing Under Rough Volatility."** It goes end to end:

    Deribit BTC options  ->  IV surface  ->  rough Heston calibration
      ->  PINN pricing on the lifted PDE  ->  validation vs Fourier/MC truth

Rough volatility = the model where volatility follows fractional Brownian motion
with Hurst H < 0.5, so it is NON-Markovian and has no finite-dimensional pricing
PDE. We handle that with the **Markovian lift** (approximate the fractional
kernel by a sum of n exponentials), which yields an (n+1)-dimensional Markovian
PDE that the PINN can solve.

## File map

    data_deribit.py        Deribit fetcher (public REST, no key). PRIMARY data source.
    data_nse.py            NSE fetcher. DEPRECATED (NSE blocks scrapers). Keep for ref.
    surface.py             Clean chain -> vega-weighted IV surface. Robust BS IV inverter.
    rough_heston_lift.py   Markovian lift: kernel ~ sum of exponentials. VALIDATED.
    rough_heston_fourier.py Ground-truth pricer A: fractional Riccati + Lewis. VALIDATED.
    lifted_mc.py           Ground-truth pricer B: lifted-SDE Monte Carlo. VALIDATED.
    calibrate.py           Fit rough Heston to the surface (Fourier in the loop).
    pinn.py                THE PINN. Solves the lifted PDE. Solve-error ~8-37 bp.
    validate_pinn.py       DECOMPOSED validation: PINN-solve vs lift error + figs.
    run_all.py             Orchestrator: data -> ... -> figures.
    data/deribit_chain.csv Real BTC data (912 rows) already pulled. USE THIS.
    calib_real.json        Calibrated params on real BTC (H=0.090, rho=-0.79).
                           Canonical full-budget fit; regenerate w/ generate_calib.py.

## Trust map — READ THIS BEFORE RELYING ON ANYTHING

VALIDATED (safe to build on):
- Markovian lift: rel L2 kernel error 26% (n=5) -> 0.4% (n=100), monotone.
- Fourier pricer: matches Black-Scholes to ~4e-7 in the flat-vol limit.
- MC pricer: matches BS within MC error; agrees with Fourier on genuine rough
  Heston (i.e. AT HIGH n) to within Euler bias. The two pricers are INDEPENDENT
  (one approximates only an ODE, the other the model+dynamics), so their
  agreement is a real cross-check. THESE ARE YOUR GROUND TRUTH. NOTE: at the
  PINN's small n the LIFTED model genuinely differs from Fourier -- the
  "lift-error" (lifted MC vs Fourier): 133 bp (n=4), 63 (n=8), 27 (n=16),
  12 (n=32) at T=0.15. So the lifted MC (same n) is the PINN's true ground
  truth; Fourier is the model's. Validation reports BOTH (see below).
- Data + surface + calibration: run end to end on real BTC. The CANONICAL fit
  (generate_calib.py: full-budget DE + polish, all 13 maturities, <=8 pts/mat
  spread across moneyness, r=0) gives H=0.090, V0=0.103, theta=0.253, lam=2.57,
  nu=0.300, rho=-0.792 at 496 vol bp over 97 points. Economically sensible
  (rough H, strong negative skew) but a COARSE fit -- crypto smiles are wide and
  the lifted rough-Heston form is stiff. The older H=0.044/rho=-0.94 (628 bp)
  was a lean 8-step run on 25 points and is SUPERSEDED.

PINN STATUS (Tasks 1-4c done; validate ALWAYS decomposed into solve vs lift):
- The price surface is now convex/monotone (boundary losses), training is
  stabilised (adaptive weights + tau-curriculum), factor collocation is
  MC-informed, and the BS baseline is anchored at the lifted forward variance.
- Report TWO errors, never a lone "PINN vs Fourier" (see pinn-lift-error memo):
    * PINN solve-error = PINN vs lifted MC at the SAME n. What the PINN controls.
      Currently ~8-37 vol bp at n=4, T=0.15 -- MEETS the <50 bp target.
    * lift-error = lifted MC vs Fourier. Model approximation; shrinks only with n
      (133 bp n=4 -> 12 bp n=32).
- Corrected record (canonical calib, T=0.15, n=4, 3 seeds, decomposed):
      config                     solve   lift   total(vs Fourier)
      Task3 old box (no MC fac)     61    133     194
      Task3/4a full fixes           37    133     121
      Task4b rho=0                   8    133     128
  The old "~114-127 bp vs Fourier" figures were DOMINATED BY LIFT, not PINN
  error; the PINN solves its PDE to well under 50 bp. RMSEs don't add (solve is
  +signed, lift is -signed, they partly cancel). The lever for the Fourier gap
  is MORE FACTORS n, not more training.
- HEADLINE n-convergence (diagnose_robustness_sweep.py 3-seed; single-seed cost/
  FD in diagnose_n_convergence.py; figure pinn_n_convergence). As n grows 4->32:
  PINN solve-error grows MILDLY -- 5.4 -> 10.7 price bp (mean+-std 1-2 bp), i.e.
  ~2x over an 8x dimension jump (sub-linear, NOT flat -- an earlier single-seed
  run looked flat by luck). Train cost stays bounded (~6-10 min CPU, ms/iter
  ~flat). lift-error falls 16->2 px bp; lift/solve crossover ~n=8-10. A naive FD
  grid 50^(n+1) explodes (3e8 -> 1e56, infeasible by n=8; structured FD like
  sparse-grid/ADI/tensor-train push higher but still degrade with n and aren't
  standard for lifted rough vol -- impractical, not impossible). Gentle solve
  growth + bounded cost vs FD blow-up = the curse-of-dimensionality argument.
- Robustness (T in {.05,.15,.5}, H in {.05,.10,.15}, n in {4,16}) and the
  put-wing skew re-measured vs lifted MC: see the two memos + the sweep scripts.
- WATCH (high-n solve bias): the growth of solve-error with n is a real
  DIRECTIONAL BIAS, not just noise -- vs the lifted MC the PINN systematically
  OVER-prices ATM/calls (signed vol bp at n=32: atm ~+87, calls ~+114; put wing
  ~0), and it grows with n (calls +62 at n=4 -> +114 at n=32); the price-space
  RMSE reaches ~11 bp at n=32. Not fixing this now. Task 5 WATCH ITEM: a
  parametric net spread over (H,nu,rho,...) may resolve each point less
  precisely, so check whether parametric training AMPLIFIES this call/ATM bias
  (compare per-(H,nu,rho) solve-error vs the single-point runs here).

## Two hard-won implementation facts (do not regress these)

1. STIFFNESS. The lifted mean-reversions x_i span ~10 orders of magnitude for
   H~0.04-0.1. Naive explicit Euler/RK4 on the factor ODEs -> NaN. Both the
   Fourier Riccati solver and the MC simulator use an EXPONENTIAL-INTEGRATOR
   (integrating-factor) scheme that handles the -x_i term exactly. If you touch
   those solvers, keep the exponential integrator.

2. KERNEL NODE RANGE. The geometric node grid must widen as ~exp(c/sqrt(n)) or
   the kernel approximation stalls at ~25% error. See lift_weights_geometric.

## PRIORITY WORK (in order) — this is why we came to Claude Code

The whole reason for moving here: the earlier environment had 3-minute run caps
and OOM-killed background jobs, so the PINN could never train properly. You do
not have those limits. Fix the PINN:

STATUS: items 1-3 and 5 DONE; item 4 partially (validated 3-seed at width<=80,
GPU/100k not needed on CPU). See the PINN STATUS block above + the memos. In
short: boundary losses (1), adaptive weights + tau-curriculum (2), MC-informed
factor sampling + lifted-forward-variance baseline (3), and the parametric PINN
(5, parametric_pinn.py) all landed; validation is now decomposed solve-vs-lift.

1. **Boundary/asymptotic losses (BIGGEST WIN).** The wing blow-up is unanchored
   boundaries. Add loss terms enforcing:
     - P -> 0                         as S -> 0   (deep OTM call)
     - P -> S - K e^{-r tau}          as S -> inf (deep ITM call)
     - optionally dP/dS -> 1 as S->inf, -> 0 as S->0
   Sample these boundary points every iteration and add to the loss.

2. **Adaptive loss weighting + tau-curriculum.** Fixed-weight residual training
   oscillates. Implement self-adaptive weights (learnable per-term) or the NTK
   scheme (Wang, Yu, Perdikaris 2022). Add a curriculum: train small tau first,
   then expand the tau range (respect causality — Wang, Sankaran, Perdikaris 2022).

3. **Factor-space sampling.** pinn.py samples factors u_i from a fixed box
   (u_scale=0.05). Instead, run a short MC (lifted_mc.py) to see where the
   factors actually live, and sample collocation from that range per maturity.

4. **Scale up.** Once stable: width>=128, depth 5-6, 40k-100k iters, GPU. Then
   validate vs Fourier PER MATURITY and target single-digit vol bp ATM,
   <50 bp across the traded strike range.

5. **Parametric PINN (thesis headline).** DONE -- parametric_pinn.py adds
   (H,nu,rho,lam,theta) as inputs (n=10, V0 & T fixed). Solve-error ~2x the
   single-point net at the box centre (11 vs 6 px bp), worse in the rough low-H /
   high-nu corners. Calibrates the BTC T~0.15 slice 11-245x FASTER than the
   Fourier loop (1-16 s vs ~180 s, zero Fourier calls) -- BUT PINN-only fit
   quality is poor (net calibrates to its own imperfect prices + a box that
   doesn't contain the T=0.15-slice optimum). RESOLUTION (demo_task5_hybrid.py):
   PINN single-shot warm-start -> local Fourier polish reaches NEAR from-scratch
   fit (29+-7 vs 24 vol bp) at ~9x FEWER Fourier evals (237 vs 2018) and ~7x
   faster -- so the parametric PINN's applied value is as a calibration WARM-START,
   not a standalone pricer. See pinn-task5-parametric memo.

## How to verify you fixed the PINN

The acceptance test is in validate_pinn.py: train, then compare PINN prices and
implied vols to price_european_fourier across strikes 0.85..1.15. Success =
IV RMSE < ~50 vol bp and a CONVEX, monotone price curve (not linear). Always
validate against the Fourier truth — never trust the PINN's self-reported loss.

## Environment

    python -m venv venv && source venv/bin/activate
    pip install -r requirements.txt      # numpy scipy pandas torch matplotlib requests
    # GPU strongly recommended for the PINN. Everything else runs on CPU.

## Conventions

- Keep the common data schema:
  ['symbol','expiry','T','type','strike','spot','ltp','bid','ask','iv_mkt'].
- r=0 for crypto (no clean carry); forward comes from the quotes.
- Don't reintroduce NSE as primary; Deribit is the source.
- When you change a solver, re-run its __main__ self-test (each file has one).
