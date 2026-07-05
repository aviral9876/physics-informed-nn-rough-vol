"""
Stage 5 - Validate the PINN, DECOMPOSING error into solve vs lift.
=================================================================

The PINN solves the LIFTED PDE with n factors. Comparing it directly to the
Fourier pricer (true rough Heston, n -> infinity) CONFLATES two independent
errors, so we always report both separately (see pinn-lift-error-vs-fourier):

  * PINN solve-error : PINN  vs  lifted MC at the SAME n.
      How well the network solved the PDE it was actually given. This is what
      training / architecture / sampling can improve. Target < 50 vol bp.
  * Lift-error       : lifted MC  vs  Fourier, as a function of n.
      The Markovian-lift model-approximation error. Independent of the PINN;
      shrinks ONLY by raising n (kernel error ~26% at n=5 -> 0.4% at n=100).

The total PINN-vs-Fourier error is, per strike, the SIGNED sum of the two; as
an RMSE the components need not add (they can partly cancel). NEVER report a
lone "PINN vs Fourier" number without this decomposition.

The lifted MC is an INDEPENDENT pricer of the same lifted model the PINN solves,
so it is the correct ground truth for the solve-error; Fourier is the ground
truth for the model. lift_error_vs_n() tabulates the lift-error term structure.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rough_heston_lift import lift_weights_geometric
from rough_heston_fourier import price_european_fourier
from lifted_mc import simulate_factor_stats, price_european_mc
from surface import implied_vol
from pinn import RoughHestonPINN


def _ivs(px, strikes, T, r):
    """Black-Scholes implied vols of a fixed-spot (S0=1) call smile."""
    return np.array([implied_vol(max(px[i], 1e-12), 1.0, strikes[i], T, r, "C")
                     for i in range(len(strikes))])


def _rmse_bp(a, b):
    return 1e4 * np.sqrt(np.nanmean((a - b) ** 2))


def evaluate(params, H, n_factors, K=1.0, r=0.0, T=1.0,
             train_iters=4000, width=64, depth=4, seed=0, mc_factors=True,
             mc_paths=200_000, mc_steps=400, mc_seed=7):
    torch_seed(seed)
    c, x = lift_weights_geometric(H+0.5, n_factors)
    pinn = RoughHestonPINN(params, (c, x), K=K, r=r, T=T,
                           width=width, depth=depth)
    # Task 3: pre-flight MC to place factor collocation where the factors live,
    # instead of an arbitrary symmetric box (fixes the too-flat smile).
    # Task 4c: the same MC gives the factor-mean term structure, used to anchor
    # the BS baseline at the model's expected integrated variance (not V0).
    if mc_factors:
        u_mean, u_std, step_mean, _ = simulate_factor_stats(K, T, params, (c, x),
                                                    n_paths=2000, n_steps=300, seed=1)
        pinn.set_factor_sampling(u_mean, u_std)
        pinn.set_baseline_term_structure(step_mean)
        print(f"MC factor collocation: mean={np.round(u_mean,4)} std={np.round(u_std,4)}")
    print(f"Training PINN (n={n_factors}, width={width}, depth={depth})...")
    val_pts = pinn.fixed_val_set()          # held-out set for a clean loss curve
    hist = pinn.train(iters=train_iters, n_col=2000, n_bnd=500, log_every=200,
                      val_pts=val_pts)
    val_hist = pinn.val_history

    strikes = np.array([0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15])
    # ground truth of the MODEL: Fourier call at spot S0=1 across strikes.
    fpx = np.atleast_1d(price_european_fourier(1.0, strikes, T, r, params, H,
                                               N=300, u_max=120, n_u=1500))
    # ground truth of the PDE THE PINN SOLVES: an independent Monte Carlo of the
    # SAME n-factor lifted model. Prices the fixed-spot smile C(S0=1, K) directly.
    mpx, mc_se = price_european_mc(1.0, strikes, T, r, params, (c, x),
                                   n_paths=mc_paths, n_steps=mc_steps, seed=mc_seed)
    # The PINN is trained with strike fixed at K=1 and prices as a function of
    # SPOT, so pinn.price(s) = C(spot=s, K=1). To read off the fixed-spot smile
    # C(S0=1, K) we use the call's degree-1 homogeneity,
    #     C(S0, K) = K * C(S0/K, 1)   =>   C(1, K) = K * C(1/K, 1),
    # which holds for rough Heston (variance dynamics independent of spot level).
    ppx = strikes * pinn.price(1.0 / strikes, tau=T)

    iv_f = _ivs(fpx, strikes, T, r)      # Fourier   (model, n -> inf)
    iv_m = _ivs(mpx, strikes, T, r)      # lifted MC (this n)
    iv_p = _ivs(ppx, strikes, T, r)      # PINN      (this n)

    solve_bp = _rmse_bp(iv_p, iv_m)      # PINN vs lifted MC  -> PINN can improve
    lift_bp = _rmse_bp(iv_m, iv_f)       # lifted MC vs Fourier -> shrinks with n
    total_bp = _rmse_bp(iv_p, iv_f)      # PINN vs Fourier (conflated)
    price_rmse = np.sqrt(np.nanmean((ppx-fpx)**2))

    print(f"\nResults (T={T}, n={n_factors}):   [dIV in vol bp, signed]")
    print(f"{'K':>6}{'Fourier':>9}{'liftMC':>9}{'PINN':>9}"
          f"{'solve':>8}{'lift':>8}{'total':>8}")
    for i, Kk in enumerate(strikes):
        print(f"{Kk:>6.2f}{fpx[i]:>9.5f}{mpx[i]:>9.5f}{ppx[i]:>9.5f}"
              f"{1e4*(iv_p[i]-iv_m[i]):>+8.1f}{1e4*(iv_m[i]-iv_f[i]):>+8.1f}"
              f"{1e4*(iv_p[i]-iv_f[i]):>+8.1f}")
    print(f"\n  PINN solve-error (vs lifted MC, n={n_factors}) = {solve_bp:6.1f} vol bp"
          f"   [target < 50; PINN-controllable]")
    print(f"  Lift-error       (lifted MC vs Fourier, n={n_factors}) = {lift_bp:6.1f} vol bp"
          f"   [model approx; shrinks only with n]")
    print(f"  Total            (PINN vs Fourier)          = {total_bp:6.1f} vol bp"
          f"   [~signed sum of the two]")
    print(f"  (MC price std err ~ {1e4*float(np.mean(mc_se)):.1f} bp*S; Price RMSE {price_rmse:.5f})")

    return dict(pinn=pinn, strikes=strikes, fpx=fpx, mpx=mpx, ppx=ppx,
                iv_f=iv_f, iv_m=iv_m, iv_p=iv_p, hist=hist, val_hist=val_hist,
                price_rmse=price_rmse, solve_bp=solve_bp, lift_bp=lift_bp,
                total_bp=total_bp, iv_rmse_bp=total_bp)


def torch_seed(s):
    import torch
    torch.manual_seed(s); np.random.seed(s)


def lift_error_vs_n(params, H, T=1.0, r=0.0, n_list=(4, 8, 16, 32),
                    strikes=None, mc_paths=200_000, mc_steps=400, mc_seed=7):
    """
    Tabulate the lift-error (lifted MC vs Fourier IV RMSE) as a function of the
    number of factors n. Model-only -- no PINN. This is the term the PINN cannot
    touch; it quantifies how much of a PINN-vs-Fourier gap is pure model
    approximation at a given n.
    """
    if strikes is None:
        strikes = np.array([0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15])
    fpx = np.atleast_1d(price_european_fourier(1.0, strikes, T, r, params, H,
                                               N=300, u_max=120, n_u=1500))
    iv_f = _ivs(fpx, strikes, T, r)
    print(f"\nLift-error vs n (lifted MC vs Fourier, T={T}):")
    out = []
    for n in n_list:
        c, x = lift_weights_geometric(H + 0.5, n)
        mpx, _ = price_european_mc(1.0, strikes, T, r, params, (c, x),
                                   n_paths=mc_paths, n_steps=mc_steps, seed=mc_seed)
        lift_bp = _rmse_bp(_ivs(mpx, strikes, T, r), iv_f)
        out.append((n, lift_bp))
        print(f"  n={n:3d}   lift-error = {lift_bp:6.1f} vol bp")
    return out


def make_figures(res, params, H, outdir="figures"):
    strikes = res["strikes"]
    # 1) price fit: Fourier (model truth), lifted MC (PINN's target), PINN
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].plot(strikes, res["fpx"], "o-", color="k", label="Fourier (model truth)")
    ax[0].plot(strikes, res["mpx"], "d-", color="tab:green", label="lifted MC (this n)")
    ax[0].plot(strikes, res["ppx"], "s--", color="tab:blue", label="PINN")
    ax[0].set_title("Call price vs strike"); ax[0].set_xlabel("K/S0")
    ax[0].set_ylabel("price"); ax[0].legend(); ax[0].grid(alpha=.3)

    # 2) IV smile: gap PINN->liftMC is solve-error, liftMC->Fourier is lift-error
    ax[1].plot(strikes, 100*res["iv_f"], "o-", color="k", label="Fourier IV")
    ax[1].plot(strikes, 100*res["iv_m"], "d-", color="tab:green", label="lifted-MC IV")
    ax[1].plot(strikes, 100*res["iv_p"], "s--", color="tab:blue", label="PINN IV")
    ax[1].set_title(f"IV smile (H={H})  solve={res['solve_bp']:.0f}bp "
                    f"lift={res['lift_bp']:.0f}bp")
    ax[1].set_xlabel("K/S0"); ax[1].set_ylabel("IV (%)")
    ax[1].legend(); ax[1].grid(alpha=.3)

    # 3) training loss: noisy resampled BATCH residual vs the clean held-out
    #    curve. The batch loss is a jagged Monte-Carlo estimate; the held-out
    #    residual on a fixed collocation set shows the true monotone-ish descent.
    it, lo = zip(*res["hist"])
    ax[2].semilogy(it, lo, "-", color="0.7", lw=.8, label="batch (resampled, noisy)")
    if res.get("val_hist"):
        vit, vlo = zip(*res["val_hist"])
        ax[2].semilogy(vit, vlo, "-", color="tab:blue", lw=2.0,
                       label="held-out (fixed set)")
    ax[2].set_title("PINN training loss"); ax[2].set_xlabel("iteration")
    ax[2].set_ylabel("PDE residual MSE"); ax[2].legend(); ax[2].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(f"{outdir}/pinn_validation.png", dpi=130)
    print(f"saved -> {outdir}/pinn_validation.png")


if __name__ == "__main__":
    params = dict(V0=0.04, theta=0.04, lam=0.3, nu=0.3, rho=-0.7)
    H = 0.1
    res = evaluate(params, H, n_factors=4, T=1.0, train_iters=4000,
                   width=64, depth=4, seed=0)
    lift_error_vs_n(params, H, T=1.0, n_list=(4, 8, 16, 32))
    make_figures(res, params, H)
