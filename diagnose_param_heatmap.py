r"""
Which parameters drive the parametric operator's error? (Reviewer 2, item 4)
=============================================================================

A 7x7 grid over (H, nu) at the centre of the box in (rho, lam, theta), n = 10,
T = 0.15. At every cell the reference is an independent lifted Monte Carlo at the
same n (200k antithetic paths, 400 steps, terminal-forward control variate), so
the number in each cell is the parametric net's SOLVE-error in price bp. The MC
references do not depend on the network, so they are cached and reused when the
net is retrained.

The referee's question is whether the degradation in the rough / high-vol-of-vol
corner is driven by one parameter or by their interaction. We answer with a
regression that has an explicit interaction term,

    err ~ a + b/H + c nu + d nu/H,

and report the interaction coefficient d and its partial R^2 (the R^2 lost by
dropping it). 1/H rather than H because the lift's stiffness scales with the
smallest mean-reversion, which grows like exp(c/sqrt(n)) in 1/H. This is cleaner
than SHAP on a 49-point grid and answers the question asked.

Also reports the network's own autograd sensitivities dC/d(H,nu,rho,lam,theta),
averaged over strikes at the box centre, with the caveat that H enters through a
48-point grid so dC/dH is a grid-trained interpolation.

Usage:
    python diagnose_param_heatmap.py [--box default|wide] [--width 80] [--depth 4]
                                     [--tag param_net] [--seeds 0 1 2]
The default arguments evaluate the currently committed nets in outputs/. After
train_parametric_boxes.py has run, use --width 96 --depth 5 --tag <its tag>.
"""
import argparse, json, os, time
import numpy as np
import torch

from rough_heston_lift import lift_weights_geometric
from lifted_mc import price_european_mc
from parametric_pinn import ParametricPINN, DEFAULT_BOX, PNAMES

NGRID = 7
T = 0.15; R = 0.0; N_LIFT = 10
STRIKES = np.array([0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15])
MC = dict(n_paths=200_000, n_steps=400, seed=7, control_variate=True)


def box_by_name(name):
    if name == "default":
        return dict(DEFAULT_BOX)
    if name == "wide":
        from bench_surrogate import BOX
        return dict(BOX)
    raise ValueError(name)


def pbp(a, b):
    return 1e4 * np.sqrt(np.nanmean((a - b) ** 2))


def references(box, V0, cache):
    """49 lifted-MC smiles; cached because they are independent of the net."""
    if os.path.exists(cache):
        d = np.load(cache)
        return d["Hs"], d["nus"], d["mpx"]
    Hs = np.linspace(*box["H"], NGRID)
    nus = np.linspace(*box["nu"], NGRID)
    rho = np.mean(box["rho"]); lam = np.mean(box["lam"]); theta = np.mean(box["theta"])
    mpx = np.zeros((NGRID, NGRID, len(STRIKES)))
    t0 = time.time()
    for i, Hh in enumerate(Hs):
        c, x = lift_weights_geometric(Hh + 0.5, N_LIFT)
        for j, nu in enumerate(nus):
            p = dict(V0=V0, theta=theta, lam=lam, nu=nu, rho=rho)
            mpx[i, j], _ = price_european_mc(1.0, STRIKES, T, R, p, (c, x), **MC)
        print("  refs row %d/%d  (%.0f s)" % (i + 1, NGRID, time.time() - t0), flush=True)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez_compressed(cache, Hs=Hs, nus=nus, mpx=mpx, rho=rho, lam=lam, theta=theta)
    return Hs, nus, mpx


def regress(Hs, nus, err):
    """err ~ a + b/H + c nu + d nu/H; return coefficients, R^2, partial R^2 of d."""
    Hg, ng = np.meshgrid(Hs, nus, indexing="ij")
    y = err.ravel()
    X_full = np.column_stack([np.ones(y.size), 1 / Hg.ravel(), ng.ravel(), ng.ravel() / Hg.ravel()])
    X_add = X_full[:, :3]

    def fit(X):
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        r2 = 1 - resid @ resid / ((y - y.mean()) @ (y - y.mean()))
        return beta, float(r2)

    b_full, r2_full = fit(X_full)
    b_add, r2_add = fit(X_add)
    return dict(coef=dict(a=float(b_full[0]), b_invH=float(b_full[1]), c_nu=float(b_full[2]),
                          d_nu_over_H=float(b_full[3])),
                r2_full=r2_full, r2_additive=r2_add, partial_r2_interaction=r2_full - r2_add)


def sensitivities(m, pv):
    """Mean |dC/dp| over strikes at one parameter vector, via autograd."""
    pt = torch.tensor(np.asarray(pv, float), requires_grad=True)
    c = m.price_smile_grad(STRIKES, T, pt)
    out = {}
    for s in range(len(STRIKES)):
        g, = torch.autograd.grad(c[s], pt, retain_graph=True)
        for k, gk in zip(PNAMES, g.detach().numpy()):
            out.setdefault(k, []).append(abs(float(gk)))
    return {k: 1e4 * float(np.mean(v)) for k, v in out.items()}      # px bp per unit


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--box", default="default")
    ap.add_argument("--width", type=int, default=80)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--tag", default="param_net")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()

    box = box_by_name(a.box)
    cal = json.load(open("calib_real.json")); V0 = float(cal["V0"])
    cache = "results/param_heatmap_refs_%s.npz" % a.box
    print("references (%s box) ..." % a.box, flush=True)
    Hs, nus, mpx = references(box, V0, cache)
    rho = np.mean(box["rho"]); lam = np.mean(box["lam"]); theta = np.mean(box["theta"])

    errs = []
    sens = []
    for s in a.seeds:
        path = "outputs/%s_seed%d.pt" % (a.tag, s)
        m = ParametricPINN(box=box, V0=V0, T=T, width=a.width, depth=a.depth, n=N_LIFT)
        m.net.load_state_dict(torch.load(path))
        e = np.zeros((NGRID, NGRID))
        for i, Hh in enumerate(Hs):
            for j, nu in enumerate(nus):
                e[i, j] = pbp(m.smile(STRIKES, T, [Hh, nu, rho, lam, theta]), mpx[i, j])
        errs.append(e)
        centre = [np.mean(box["H"]), np.mean(box["nu"]), rho, lam, theta]
        sens.append(sensitivities(m, centre))
        print("seed %d: solve-error %.1f (min) .. %.1f (max) px bp, corner(lowH,highnu)=%.1f"
              % (s, e.min(), e.max(), e[0, -1]), flush=True)

    E = np.mean(errs, axis=0)
    reg = regress(Hs, nus, E)
    print("\n  mean over seeds, rows=H (%.3f..%.3f), cols=nu (%.2f..%.2f):" % (Hs[0], Hs[-1], nus[0], nus[-1]))
    for i in range(NGRID):
        print("   H=%.3f  " % Hs[i] + " ".join("%6.1f" % v for v in E[i]))
    print("\n  err ~ a + b/H + c nu + d nu/H")
    print("   d (interaction) = %.3f   R2 full = %.3f   R2 additive = %.3f   partial R2 = %.3f"
          % (reg["coef"]["d_nu_over_H"], reg["r2_full"], reg["r2_additive"], reg["partial_r2_interaction"]))
    ms = {k: float(np.mean([d[k] for d in sens])) for k in PNAMES}
    print("  mean |dC/dp| at box centre (px bp per unit): " + ", ".join("%s %.0f" % kv for kv in ms.items()))

    from resultio import dump
    dump("param_heatmap_%s" % a.box, dict(
        box=box, width=a.width, depth=a.depth, tag=a.tag, seeds=a.seeds, n=N_LIFT, T=T,
        strikes=STRIKES.tolist(), mc=MC, H_grid=Hs.tolist(), nu_grid=nus.tolist(),
        fixed=dict(rho=float(rho), lam=float(lam), theta=float(theta)),
        err_px_bp_per_seed=[e.tolist() for e in errs], err_px_bp_mean=E.tolist(),
        regression=reg, sensitivities_per_seed=sens, sensitivities_mean=ms))
    print("saved -> results/param_heatmap_%s.json" % a.box)
