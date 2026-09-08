r"""
Supervised IV surrogate: the baseline practitioners actually use. (Reviewer 1, item 4)
======================================================================================

The referee's objection is fair and we do not intend to argue with it. A reader
who wants a fast forward operator for rough Heston would not reach for a PINN
first; they would generate labels with an exact pricer and fit a network to them,
as in Bayer et al. (2019) and Horvath et al. (2021). If our operator is worth
having, it has to be worth having next to that.

So we build it, under a protocol identical to the parametric PINN's, and report
the comparison including the parts that go against us.

WHAT WE EXPECT TO FIND, stated in advance so the result cannot be presented as a
surprise victory. The surrogate is trained directly on Fourier labels, so on
accuracy-versus-Fourier it should WIN, and probably by a lot. Fourier itself costs
about 48 ms per smile, so on single-instance speed both baselines beat an
8-10 minute PINN training run by three to four orders of magnitude. The honest
claim for the PINN is therefore not speed and not accuracy against the transform
method. It is two things a label-trained surrogate structurally cannot offer:

  1. It needs no exact pricer offline. Every label here comes from the fractional
     Riccati solver; without one the surrogate cannot be built at all. The PINN
     trains on the PDE residual, so the method transfers to models where no
     transform method exists -- which is the case the lift was invented for.
  2. It yields the solve/lift error decomposition. A surrogate fitted to Fourier
     labels inherits the lift error silently and cannot separate the two, so it
     cannot tell a user whether to train longer or raise n.

DESIGN, matched to the parametric PINN so the comparison is like-for-like:
  * inputs  : the same 5 parameters (H, nu, rho, lam, theta), V0 fixed
  * outputs : implied vol on the same fixed 21-point log-moneyness grid
  * box     : the SAME widened box the parametric net is retrained on
  * split   : BY PARAMETER VECTOR, never by strike. Splitting by row would leak
              -- 21 strikes of one vector would straddle train and test, and the
              reported error would be meaningless.
"""
import json, os, time
import numpy as np

from rough_heston_fourier import price_european_fourier
from surface import implied_vol

K_GRID = np.exp(np.linspace(-0.35, 0.35, 21))
T_FIXED = 0.15
R = 0.0
LABELS = "results/surrogate_labels.npz"

# Same widened box as the parametric PINN retrain (see refit_canonical.WIDE for
# the market-calibration box; this one is the 5-parameter operator box).
BOX = dict(H=(0.04, 0.20), nu=(0.10, 1.00), rho=(-0.95, -0.05),
           lam=(0.5, 5.0), theta=(0.08, 0.45))
PARAMS = ["H", "nu", "rho", "lam", "theta"]


def sobol(n, seed=0):
    """Sobol draw over the box, falling back to LHS if scipy is too old."""
    try:
        from scipy.stats import qmc
        u = qmc.Sobol(d=len(PARAMS), scramble=True, seed=seed).random(n)
    except Exception:
        rng = np.random.default_rng(seed)
        u = (rng.permuted(np.tile(np.arange(n), (len(PARAMS), 1)), axis=1).T
             + rng.random((n, len(PARAMS)))) / n
    lo = np.array([BOX[p][0] for p in PARAMS])
    hi = np.array([BOX[p][1] for p in PARAMS])
    return lo + u * (hi - lo)


def label_one(vec, V0):
    """One parameter vector -> 21 implied vols, or None if the pricer misbehaves."""
    H, nu, rho, lam, theta = vec
    p = dict(V0=V0, theta=theta, lam=lam, nu=nu, rho=rho)
    try:
        px = np.atleast_1d(price_european_fourier(1.0, K_GRID, T_FIXED, R, p, H,
                                                  N=120, u_max=100, n_u=800))
    except Exception:
        return None
    if not np.all(np.isfinite(px)) or np.any(px <= 0):
        return None
    iv = np.array([implied_vol(max(px[i], 1e-12), 1.0, K_GRID[i], T_FIXED, R, "C")
                   for i in range(len(K_GRID))])
    if not np.all(np.isfinite(iv)) or np.any(iv < 0.01) or np.any(iv > 2.0):
        return None
    return iv


def generate(n_total=80000, workers=4, seed=0):
    """Label generation. Embarrassingly parallel; the only expensive step."""
    from multiprocessing import Pool
    cal = json.load(open("calib_real.json"))
    V0 = float(cal["V0"])
    X = sobol(n_total, seed=seed)
    t0 = time.time()
    with Pool(workers) as pool:
        Y = pool.starmap(label_one, [(x, V0) for x in X], chunksize=64)
    ok = [i for i, y in enumerate(Y) if y is not None]
    X, Y = X[ok], np.array([Y[i] for i in ok])
    print("labelled %d/%d vectors in %.1f min (%d discarded by the pricer)"
          % (len(ok), n_total, (time.time() - t0) / 60, n_total - len(ok)), flush=True)
    os.makedirs("results", exist_ok=True)
    np.savez_compressed(LABELS, X=X, Y=Y, K=K_GRID, T=T_FIXED, V0=V0,
                        box=json.dumps(BOX), seconds=time.time() - t0)
    return X, Y


def train(X, Y, seed=0, epochs=200, batch=512, hidden=30, depth=4):
    """MLP matched to Horvath et al. (2021): 5 -> 4x30 ELU -> 21 outputs."""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    np.random.seed(seed)

    # split BY PARAMETER VECTOR -- see the module docstring on why this matters
    n = len(X)
    idx = np.random.default_rng(seed).permutation(n)
    n_te = n_va = max(int(0.125 * n), 1)
    te, va, tr = idx[:n_te], idx[n_te:n_te + n_va], idx[n_te + n_va:]

    xm, xs = X[tr].mean(0), X[tr].std(0) + 1e-12
    ym, ys = Y[tr].mean(0), Y[tr].std(0) + 1e-12
    T_ = lambda a, m, s: torch.tensor((a - m) / s, dtype=torch.float32)

    layers, d = [], X.shape[1]
    for _ in range(depth):
        layers += [nn.Linear(d, hidden), nn.ELU()]
        d = hidden
    layers += [nn.Linear(d, Y.shape[1])]
    net = nn.Sequential(*layers)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    xtr, ytr = T_(X[tr], xm, xs), T_(Y[tr], ym, ys)
    xva, yva = T_(X[va], xm, xs), T_(Y[va], ym, ys)

    best, best_state, patience, since = np.inf, None, 20, 0
    t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(len(xtr))
        net.train()
        for i in range(0, len(xtr), batch):
            b = perm[i:i + batch]
            opt.zero_grad()
            nn.functional.mse_loss(net(xtr[b]), ytr[b]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = float(nn.functional.mse_loss(net(xva), yva))
        if v < best - 1e-7:
            best, since = v, 0
            best_state = {k: t.clone() for k, t in net.state_dict().items()}
        else:
            since += 1
            if since >= patience:
                break
        if ep % 20 == 0:
            print("  epoch %3d  val MSE %.3e" % (ep, v), flush=True)
    net.load_state_dict(best_state)
    secs = time.time() - t0

    with torch.no_grad():
        pred = net(T_(X[te], xm, xs)).numpy() * ys + ym
    err_bp = 1e4 * (pred - Y[te])
    rmse = float(np.sqrt(np.mean(err_bp ** 2)))

    # the corners that matter: rough (low H) and high vol-of-vol, the same region
    # where the parametric PINN degrades, so the two answers interlock
    Hte, nute = X[te][:, 0], X[te][:, 1]
    corner = (Hte <= np.quantile(Hte, 0.25)) & (nute >= np.quantile(nute, 0.75))
    rmse_corner = float(np.sqrt(np.mean(err_bp[corner] ** 2))) if corner.sum() else float("nan")

    return dict(test_rmse_vol_bp=rmse, test_rmse_corner_vol_bp=rmse_corner,
                n_corner=int(corner.sum()), n_train=len(tr), n_val=len(va),
                n_test=len(te), epochs_run=ep + 1, train_seconds=secs,
                hidden=hidden, depth=depth, seed=seed,
                per_strike_rmse_vol_bp=np.sqrt(np.mean(err_bp ** 2, axis=0)).tolist())


if __name__ == "__main__":
    import sys
    n_total = int(sys.argv[1]) if len(sys.argv) > 1 else 80000
    workers = int(os.environ.get("SURR_WORKERS", "4"))

    if os.path.exists(LABELS):
        d = np.load(LABELS, allow_pickle=True)
        X, Y = d["X"], d["Y"]
        print("reusing %d labelled vectors from %s" % (len(X), LABELS), flush=True)
        label_secs = float(d["seconds"])
    else:
        t0 = time.time()
        X, Y = generate(n_total, workers=workers)
        label_secs = time.time() - t0

    runs = [train(X, Y, seed=s) for s in (0, 1, 2)]
    rm = np.array([r["test_rmse_vol_bp"] for r in runs])
    cn = np.array([r["test_rmse_corner_vol_bp"] for r in runs])
    tt = np.array([r["train_seconds"] for r in runs])

    print("\n==== supervised surrogate, 3 seeds ====")
    print("  test IV RMSE          %.1f +- %.1f vol bp" % (rm.mean(), rm.std(ddof=1)))
    print("  rough/high-nu corner  %.1f +- %.1f vol bp" % (cn.mean(), cn.std(ddof=1)))
    print("  training              %.1f s per seed" % tt.mean())
    print("  offline label cost    %.1f min (%d vectors, %d workers)"
          % (label_secs / 60, len(X), workers))
    print("\n  Note for the manuscript: the surrogate is trained ON Fourier labels,")
    print("  so its error against Fourier is not comparable to the PINN's solve-error,")
    print("  which is measured against an INDEPENDENT lifted Monte Carlo. Reporting")
    print("  them in one column without that caveat would flatter the surrogate.")

    from resultio import dump
    dump("surrogate", dict(
        design=dict(inputs=PARAMS, box=BOX, n_strikes=len(K_GRID), T=T_FIXED,
                    architecture="MLP 5 -> %dx%d ELU -> %d" % (4, 30, len(K_GRID)),
                    split="by parameter vector, 75/12.5/12.5"),
        n_labelled=int(len(X)), label_seconds=label_secs, runs=runs,
        summary=dict(test_rmse_mean=float(rm.mean()), test_rmse_sd=float(rm.std(ddof=1)),
                     corner_rmse_mean=float(cn.mean()), corner_rmse_sd=float(cn.std(ddof=1)),
                     train_seconds_mean=float(tt.mean()))))
    print("\nsaved -> results/surrogate.json")
