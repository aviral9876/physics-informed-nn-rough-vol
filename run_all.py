"""
run_all.py - End-to-end pipeline: NSE data -> results.
======================================================

Chains every stage:
  1. load market option chain (NSE live, else synthetic fallback)
  2. clean + build the implied-vol surface
  3. calibrate rough Heston to the surface (Fourier pricer in the loop)
  4. train the PINN on the lifted PDE with the calibrated params
  5. validate PINN vs Fourier ground truth; price a fresh smile
  6. save results (CSV/JSON) and figures

Run:  python run_all.py  [--symbol NIFTY] [--online] [--pinn-iters N]

By default runs OFFLINE with synthetic data so it works anywhere. Pass
--online to attempt a live NSE pull first.
"""

import argparse
import json
import numpy as np
import pandas as pd

from data_deribit import load_chain
from surface import build_surface, implied_vol
from calibrate import calibrate
from rough_heston_lift import lift_weights_geometric
from rough_heston_fourier import price_european_fourier
from pinn import RoughHestonPINN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--currency", default="BTC", choices=["BTC", "ETH"])
    ap.add_argument("--csv", default=None,
                    help="path to a saved Deribit chain CSV (offline)")
    ap.add_argument("--online", action="store_true",
                    help="pull live from Deribit (run on your own network)")
    ap.add_argument("--r", type=float, default=0.0)
    ap.add_argument("--pinn-iters", type=int, default=6000)
    ap.add_argument("--n-factors", type=int, default=3)
    ap.add_argument("--calib-maxiter", type=int, default=12)
    args = ap.parse_args()

    print("="*70)
    print("STEP 1/6  Load option chain (Deribit)")
    if args.csv:
        df, src = load_chain(args.currency, allow_network=False, csv_path=args.csv)
    elif args.online:
        df, src = load_chain(args.currency, allow_network=True)
    else:
        # default offline: use a previously-saved CSV if present
        import os
        default_csv = "data/deribit_chain.csv"
        if os.path.exists(default_csv):
            df, src = load_chain(args.currency, allow_network=False,
                                 csv_path=default_csv)
        else:
            raise SystemExit(
                "No data. Run `python data_deribit.py --online` on your "
                "network first, or pass --csv path/to/chain.csv")
    df.to_csv("data/option_chain.csv", index=False)
    print(f"   source={src}, rows={len(df)}")

    print("="*70)
    print("STEP 2/6  Build IV surface")
    surf = build_surface(df, r=args.r)
    surf.to_csv("data/iv_surface.csv", index=False)
    print(f"   surface points={len(surf)}, "
          f"maturities={sorted(surf['T'].unique().round(4).tolist())}")

    print("="*70)
    print("STEP 3/6  Calibrate rough Heston")
    params, diag = calibrate(surf, r=args.r, maxiter=args.calib_maxiter,
                             popsize=8, seed=1, verbose=True)
    H = params.pop("H")
    pd.Series({**params, "H": H, "rmse_vol_bp": diag["rmse_vol_bp"]}
              ).to_csv("data/calibrated_params.csv")

    print("="*70)
    print(f"STEP 4/6  Train PINN on lifted PDE (n={args.n_factors} factors)")
    import torch
    torch.manual_seed(0); np.random.seed(0)
    c, x = lift_weights_geometric(H+0.5, args.n_factors)
    # price a representative maturity (median of surface)
    T = float(np.median(surf["T"]))
    spot = float(surf["spot"].iloc[0])
    K_atm = spot*np.exp(args.r*T)
    pinn = RoughHestonPINN(params, (c, x), K=K_atm, r=args.r, T=T,
                           width=64, depth=4)
    hist = pinn.train(iters=args.pinn_iters, n_col=900, n_bnd=300,
                      log_every=max(1, args.pinn_iters//6))
    torch.save(pinn.net.state_dict(), "outputs/pinn_net.pt")

    print("="*70)
    print("STEP 5/6  Validate PINN vs Fourier ground truth")
    strikes = K_atm*np.exp(np.linspace(-0.15, 0.15, 7))
    fpx = np.atleast_1d(price_european_fourier(spot, strikes, T, args.r,
                                               params, H, N=200, u_max=100,
                                               n_u=1000))
    ppx = pinn.price(strikes, tau=T)
    iv_f = np.array([implied_vol(fpx[i], spot, strikes[i], T, args.r, "C")
                     for i in range(len(strikes))])
    iv_p = np.array([implied_vol(max(ppx[i], 1e-8), spot, strikes[i], T,
                                 args.r, "C") for i in range(len(strikes))])
    price_rmse = float(np.sqrt(np.nanmean((ppx-fpx)**2)))
    iv_rmse_bp = float(1e4*np.sqrt(np.nanmean((iv_p-iv_f)**2)))
    print(f"   PINN vs Fourier:  price RMSE={price_rmse:.5f}  "
          f"IV RMSE={iv_rmse_bp:.1f} vol bp")

    results = dict(source=src, params={**params, "H": H},
                   calib_rmse_vol_bp=diag["rmse_vol_bp"],
                   T=T, spot=spot, strikes=strikes.tolist(),
                   fourier=fpx.tolist(), pinn=ppx.tolist(),
                   iv_f=iv_f.tolist(), iv_p=iv_p.tolist(),
                   pinn_price_rmse=price_rmse, pinn_iv_rmse_bp=iv_rmse_bp,
                   hist=hist)
    json.dump(results, open("outputs/results.json", "w"), indent=2)

    print("="*70)
    print("STEP 6/6  Figures")
    make_all_figures(surf, diag, results, params, H)
    print("Done. See outputs/ and figures/.")


def make_all_figures(surf, diag, results, params, H):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    # (a) market vs model IV (calibration fit)
    m_model = diag["model_iv"]; m_mkt = diag["market_iv"]
    mask = np.isfinite(m_model)
    ax[0, 0].scatter(surf["k"][mask], 100*m_mkt[mask], s=28,
                     label="market", zorder=3)
    ax[0, 0].scatter(surf["k"][mask], 100*m_model[mask], s=28, marker="x",
                     label="rough Heston fit")
    ax[0, 0].set_title(f"Calibration fit  (RMSE {diag['rmse_vol_bp']:.0f} bp)")
    ax[0, 0].set_xlabel("log-moneyness k"); ax[0, 0].set_ylabel("IV (%)")
    ax[0, 0].legend(); ax[0, 0].grid(alpha=.3)

    # (b) PINN vs Fourier price
    K = np.array(results["strikes"]); spot = results["spot"]
    ax[0, 1].plot(K/spot, results["fourier"], "o-", label="Fourier (truth)")
    ax[0, 1].plot(K/spot, results["pinn"], "s--", label="PINN")
    ax[0, 1].set_title("PINN vs Fourier price")
    ax[0, 1].set_xlabel("K / S0"); ax[0, 1].set_ylabel("call price")
    ax[0, 1].legend(); ax[0, 1].grid(alpha=.3)

    # (c) PINN vs Fourier IV smile
    ax[1, 0].plot(K/spot, 100*np.array(results["iv_f"]), "o-", label="Fourier IV")
    ax[1, 0].plot(K/spot, 100*np.array(results["iv_p"]), "s--", label="PINN IV")
    ax[1, 0].set_title(f"PINN vs Fourier IV  (RMSE "
                       f"{results['pinn_iv_rmse_bp']:.0f} bp)")
    ax[1, 0].set_xlabel("K / S0"); ax[1, 0].set_ylabel("IV (%)")
    ax[1, 0].legend(); ax[1, 0].grid(alpha=.3)

    # (d) training loss
    it, lo = zip(*results["hist"])
    ax[1, 1].semilogy(it, lo, "-o", ms=3)
    ax[1, 1].set_title("PINN training loss")
    ax[1, 1].set_xlabel("iteration"); ax[1, 1].set_ylabel("PDE residual MSE")
    ax[1, 1].grid(alpha=.3)

    fig.suptitle(f"Rough Heston + PINN pipeline  (H={H:.3f}, "
                 f"source={results['source']})", fontsize=13)
    fig.tight_layout()
    fig.savefig("figures/pipeline_results.png", dpi=130)
    print("   saved -> figures/pipeline_results.png")


if __name__ == "__main__":
    main()
