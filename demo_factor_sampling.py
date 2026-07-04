"""
Task 3 demo - MC-informed factor collocation fixes the too-flat PINN smile.
============================================================================

The PINN samples the lifted factors u_i to place PDE collocation points. The
old sampler used a fixed symmetric box centred at 0 for every factor. A short
pre-flight Monte-Carlo (lifted_mc.simulate_factor_stats) shows the factors
actually live at a per-factor mean/std that is NOT centred at 0 whenever
theta != V0. Sampling collocation there (pinn.set_factor_sampling) lets the PINN
learn the skew it was missing.

Run:  python demo_factor_sampling.py [T]      # T defaults to a mid maturity
Uses the calibrated BTC params in calib_real.json (make with generate_calib.py).
Writes figures/pinn_factor_sampling.png (Fourier truth, old PINN, new PINN).
"""
import json
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rough_heston_lift import lift_weights_geometric
from rough_heston_fourier import price_european_fourier
from lifted_mc import simulate_factor_stats
from surface import implied_vol
from pinn import RoughHestonPINN

STRIKES = np.array([0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15])


def run(T=None, n_factors=4, seeds=(0, 1, 2), iters=4000):
    cal = json.load(open("calib_real.json"))
    params = dict(V0=cal["V0"], theta=cal["theta"], lam=cal["lam"],
                  nu=cal["nu"], rho=cal["rho"])
    H = cal["H"]; r = cal.get("r", 0.0)
    if T is None:
        T = 0.15                       # ~2-month BTC slice: clear rough-vol skew
    c, x = lift_weights_geometric(H + 0.5, n_factors)
    print(f"calib H={H:.4f} V0={params['V0']:.4f} theta={params['theta']:.4f} "
          f"lam={params['lam']:.3f} nu={params['nu']:.3f} rho={params['rho']:.3f}")
    print(f"slice T={T:.4f}")

    # pre-flight MC: where do the factors live?
    mean, std, _, _ = simulate_factor_stats(1.0, T, params, (c, x),
                                            n_paths=2000, n_steps=300, seed=1)
    print("MC factor stats (pooled over paths x steps):")
    for i in range(n_factors):
        print(f"  factor {i}: x_i={float(x[i]):>9.3g}  "
              f"mean={mean[i]:+.3e}  std={std[i]:.3e}")
    print("  (old fixed box: mean=0, half-width=0.05, same for every factor)")

    fpx = np.atleast_1d(price_european_fourier(1.0, STRIKES, T, r, params, H,
                                               N=300, u_max=120, n_u=1500))
    ivf = np.array([implied_vol(fpx[i], 1.0, STRIKES[i], T, r, "C")
                    for i in range(len(STRIKES))])

    def train_eval(seed, use_mc):
        torch.manual_seed(seed); np.random.seed(seed)
        p = RoughHestonPINN(params, (c, x), K=1.0, r=r, T=T,
                            width=64, depth=4, x_halfwidth=1.2)
        if use_mc:
            p.set_factor_sampling(mean, std)
        p.train(iters=iters, n_col=2000, n_bnd=500, n_bc=400, log_every=iters)
        hom = STRIKES * p.price(1.0 / STRIKES, tau=T)       # homogeneity smile
        ivh = np.array([implied_vol(max(hom[i], 1e-8), 1.0, STRIKES[i], T, r, "C")
                        for i in range(len(STRIKES))])
        rmse = 1e4 * np.sqrt(np.nanmean((ivh - ivf) ** 2))
        return hom, ivh, rmse

    old_r, new_r, old0, new0 = [], [], None, None
    for s in seeds:
        ho, ivo, ro = train_eval(s, False)
        hn, ivn, rn = train_eval(s, True)
        old_r.append(ro); new_r.append(rn)
        if s == seeds[0]:
            old0, new0 = (ho, ivo), (hn, ivn)
        print(f"seed {s}: OLD box RMSE={ro:6.1f}  NEW mc RMSE={rn:6.1f} bp")
    print(f"\nIV RMSE  OLD(box) {np.mean(old_r):.1f}+-{np.std(old_r):.1f}  |  "
          f"NEW(mc) {np.mean(new_r):.1f}+-{np.std(new_r):.1f} vol bp")

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    ax[0].plot(STRIKES, fpx, "o-", color="k", label="Fourier (truth)")
    ax[0].plot(STRIKES, old0[0], "s--", color="tab:red",
               label=f"old PINN (fixed box) {np.mean(old_r):.0f} bp")
    ax[0].plot(STRIKES, new0[0], "^--", color="tab:blue",
               label=f"new PINN (MC factors) {np.mean(new_r):.0f} bp")
    ax[0].set_title("Call price vs strike"); ax[0].set_xlabel("K/S0")
    ax[0].set_ylabel("price"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(STRIKES, 100 * ivf, "o-", color="k", label="Fourier IV")
    ax[1].plot(STRIKES, 100 * old0[1], "s--", color="tab:red",
               label="old PINN (fixed box)")
    ax[1].plot(STRIKES, 100 * new0[1], "^--", color="tab:blue",
               label="new PINN (MC factors)")
    ax[1].set_title(f"Implied-vol smile  (BTC calib, H={H:.3f}, T={T:.3f})")
    ax[1].set_xlabel("K/S0"); ax[1].set_ylabel("IV (%)")
    ax[1].legend(); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig("figures/pinn_factor_sampling.png", dpi=130)
    print("saved -> figures/pinn_factor_sampling.png")


if __name__ == "__main__":
    T = float(sys.argv[1]) if len(sys.argv) > 1 else None
    run(T=T)
