# Rough Volatility + PINN: End-to-End Pipeline

From NSE market data to a physics-informed neural network option pricer under
rough Heston. Six stages, each a standalone module, plus an orchestrator.

```
data_nse.py            1. Acquire option chain: live NSE (cookie handshake) OR
                          realistic synthetic fallback. Common schema out.
surface.py             2. Clean the chain; build the implied-vol surface
                          (OTM legs, robust IV inversion, vega weights).
rough_heston_lift.py   -  Markovian lift: fractional kernel ~ sum of exponentials.
rough_heston_fourier.py-  Ground-truth pricer A: fractional Riccati + Lewis.
lifted_mc.py           -  Ground-truth pricer B: lifted-SDE Monte Carlo.
calibrate.py           3. Fit rough Heston (H,V0,theta,lam,nu,rho) to the surface
                          using pricer A in the loop (vega-weighted IV loss).
pinn.py                4. PINN solving the lifted (n+1)-dim pricing PDE.
validate_pinn.py       5. Compare PINN vs Fourier truth; figures.
run_all.py             6. Orchestrator chaining 1->5 + figures.
```

## Quick start

```bash
pip install numpy scipy pandas torch matplotlib requests
python run_all.py                 # offline, synthetic data, runs anywhere
python run_all.py --online        # try a live NSE pull first
python run_all.py --pinn-iters 40000 --n-factors 5   # serious training (GPU)
```

Individual stages also run on their own (each has a `__main__` demo):
`python data_nse.py`, `python surface.py`, `python calibrate.py`, etc.

---

## HONEST STATUS - read this

This is a research scaffold. Stages 1-3 and both ground-truth pricers are
**validated and reliable**. The PINN (stage 4) is **correct in structure and
trains, but does not yet reach research-grade accuracy** in the sandbox it was
built in. Details:

### What is validated and trustworthy
- **Markovian lift** (`rough_heston_lift.py`): relative L2 kernel error falls
  monotonically 26% (n=5) -> 0.4% (n=100). See the figure `data_and_lift.png`.
- **Fourier pricer** (`rough_heston_fourier.py`): matches Black-Scholes to
  ~4e-7 in the constant-variance limit; solves the true fractional Riccati.
- **Monte Carlo pricer** (`lifted_mc.py`): matches BS within MC error; agrees
  with the Fourier pricer on genuine rough Heston to within Euler
  discretization bias (which shrinks as steps increase - verified).
- **Data + surface + calibration**: run end to end; calibration fits the
  synthetic NIFTY surface to ~20-25 vol bp.
- The two pricers are **independent** (one approximates only an ODE, the other
  approximates the model + dynamics), so their agreement is a real cross-check.

### What is NOT solved
The PINN price surface, on true rough-vol parameters (nu=0.3), currently
collapses toward `BS_baseline + (near-linear correction)` and the training
loss is unstable (oscillates ~1e-3 to 2e-2). It does:
- recover Black-Scholes approximately in the nu=0 degenerate case,
- produce monotone, in-[0,1] deltas,
but it does NOT yet match the Fourier IV smile (current gap is large, not the
single-digit vol bp you would report in a thesis).

### Why (root causes, all fixable)
1. **Wing under-determination.** The PDE residual alone does not pin down the
   solution where collocation is sparse (deep ITM/OTM). Needs explicit
   boundary/asymptotic anchor losses (e.g. P->0 as S->0, P->S-Ke^{-r tau} as
   S->inf) added to the loss.
2. **Loss balancing / stability.** Fixed-weight residual training with a cosine
   schedule oscillates. Needs adaptive weighting (self-adaptive weights or the
   NTK scheme of Wang-Yu-Perdikaris) and/or curriculum in tau (train short
   maturities first, march outward).
3. **Factor-space sampling.** `u_scale` is a guess; collocation should cover
   where the lifted factors actually live along simulated paths (sample factors
   from short MC pre-runs, not a fixed box).
4. **Budget.** Built on CPU in ~3-min slices; long background runs were
   OOM-killed. Real training is 20k-100k iterations on a GPU with a wider net.

### Concrete next steps to close the gap
- Add asymptotic boundary losses (biggest single win).
- Switch to self-adaptive or NTK loss weighting; add tau-curriculum.
- Sample factor collocation from MC path statistics.
- Train 40k+ iters, width>=128, on GPU; validate vs Fourier at each maturity.
- Then extend to the **parametric PINN** (feed (H,nu,rho,...) as inputs) so one
  network prices across the calibration space -> instant recalibration, which
  is the thesis's headline contribution.

Treat the Fourier and MC pricers as the trusted ground truth throughout; the
PINN must earn its place by matching them, and right now it does not yet.

---

## Data note (Deribit)

Data source is **Deribit** BTC/ETH options: a fully public REST API (no key, no
cookie handshake), deep and liquid, with exchange mark IV included, and a real
rough-volatility literature to cite. `data_deribit.py` pulls the live chain and
maps it into the pipeline's common schema. See `DATA_SOURCE.md` for the exact
one-command pull.

The Anthropic sandbox blocks all outbound hosts, so the live pull must be run on
YOUR network; the parser is verified offline against a real-format sample
(`python data_deribit.py --offline-test`). A BTC-shaped synthetic chain is
included so surface + calibration run without data, and drops out the moment you
supply a real `deribit_chain.csv`.

NSE support (`data_nse.py`) is retained but deprecated - NSE actively blocks
scrapers and needs a residential IP + cookie handshake; Deribit is strictly
easier.

## File map of outputs

```
data/option_chain.csv       raw (or synthetic) chain
data/iv_surface.csv         cleaned surface used for calibration
data/calibrated_params.csv  fitted rough Heston params + fit RMSE
outputs/results.json        PINN vs Fourier prices/IVs + training history
figures/data_and_lift.png   market surface + validated kernel lift
figures/pipeline_results.png calibration fit + PINN comparison + loss
```
