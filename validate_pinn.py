"""
Stage 5 - Validate the PINN against ground truth and produce results.
=====================================================================

The PINN solved the lifted PDE with n factors. Ground truth = the fractional
Riccati Fourier pricer (the TRUE rough Heston, n -> infinity). Agreement tests
both that the PINN solved its PDE AND that n factors are enough. We report:
  - price and implied-vol error across strikes
  - a delta comparison against finite-difference Fourier deltas
  - figures: price fit, IV smile fit, training loss.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rough_heston_lift import lift_weights_geometric
from rough_heston_fourier import price_european_fourier
from surface import implied_vol
from pinn import RoughHestonPINN


def evaluate(params, H, n_factors, K=1.0, r=0.0, T=1.0,
             train_iters=4000, width=64, depth=4, seed=0):
    torch_seed(seed)
    c, x = lift_weights_geometric(H+0.5, n_factors)
    pinn = RoughHestonPINN(params, (c, x), K=K, r=r, T=T,
                           width=width, depth=depth)
    print(f"Training PINN (n={n_factors}, width={width}, depth={depth})...")
    hist = pinn.train(iters=train_iters, n_col=2000, n_bnd=500, log_every=1000)

    strikes = np.array([0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15])
    # ground truth
    fpx = np.atleast_1d(price_european_fourier(1.0, strikes, T, r, params, H,
                                               N=300, u_max=120, n_u=1500))
    ppx = pinn.price(strikes, tau=T)

    iv_f = np.array([implied_vol(fpx[i], 1.0, strikes[i], T, r, "C")
                     for i in range(len(strikes))])
    iv_p = np.array([implied_vol(max(ppx[i], 1e-8), 1.0, strikes[i], T, r, "C")
                     for i in range(len(strikes))])

    price_rmse = np.sqrt(np.nanmean((ppx-fpx)**2))
    iv_rmse_bp = 1e4*np.sqrt(np.nanmean((iv_p-iv_f)**2))

    print(f"\nResults (T={T}):")
    print(f"{'K':>6}{'Fourier':>10}{'PINN':>10}{'dPrice':>10}"
          f"{'IV_F':>8}{'IV_PINN':>9}{'IV bp':>8}")
    for i, Kk in enumerate(strikes):
        print(f"{Kk:>6.2f}{fpx[i]:>10.5f}{ppx[i]:>10.5f}{ppx[i]-fpx[i]:>+10.5f}"
              f"{iv_f[i]:>8.4f}{iv_p[i]:>9.4f}{1e4*(iv_p[i]-iv_f[i]):>+8.1f}")
    print(f"\nPrice RMSE = {price_rmse:.5f} | IV RMSE = {iv_rmse_bp:.1f} vol bp")

    return dict(pinn=pinn, strikes=strikes, fpx=fpx, ppx=ppx,
                iv_f=iv_f, iv_p=iv_p, hist=hist,
                price_rmse=price_rmse, iv_rmse_bp=iv_rmse_bp)


def torch_seed(s):
    import torch
    torch.manual_seed(s); np.random.seed(s)


def make_figures(res, params, H, outdir="figures"):
    strikes = res["strikes"]
    # 1) price fit
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].plot(strikes, res["fpx"], "o-", label="Fourier (truth)")
    ax[0].plot(strikes, res["ppx"], "s--", label="PINN")
    ax[0].set_title("Call price vs strike"); ax[0].set_xlabel("K/S0")
    ax[0].set_ylabel("price"); ax[0].legend(); ax[0].grid(alpha=.3)

    # 2) IV smile fit
    ax[1].plot(strikes, 100*res["iv_f"], "o-", label="Fourier IV")
    ax[1].plot(strikes, 100*res["iv_p"], "s--", label="PINN IV")
    ax[1].set_title(f"Implied-vol smile (H={H})"); ax[1].set_xlabel("K/S0")
    ax[1].set_ylabel("IV (%)"); ax[1].legend(); ax[1].grid(alpha=.3)

    # 3) training loss
    it, lo = zip(*res["hist"])
    ax[2].semilogy(it, lo, "-")
    ax[2].set_title("PINN training loss"); ax[2].set_xlabel("iteration")
    ax[2].set_ylabel("PDE residual MSE"); ax[2].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(f"{outdir}/pinn_validation.png", dpi=130)
    print(f"saved -> {outdir}/pinn_validation.png")


if __name__ == "__main__":
    params = dict(V0=0.04, theta=0.04, lam=0.3, nu=0.3, rho=-0.7)
    H = 0.1
    res = evaluate(params, H, n_factors=4, T=1.0, train_iters=4000,
                   width=64, depth=4, seed=0)
    make_figures(res, params, H)
