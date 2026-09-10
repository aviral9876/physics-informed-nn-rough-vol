r"""
Figures added in the CNSNS revision, each drawn from results/<name>.json.
========================================================================

Companion to figures_paper.py and make_tables.py: a figure whose JSON is
missing is reported and skipped, never drawn from stale numbers. Writes
figures/fig_<name>.png; the manuscript includes each behind an \IfFileExists.

  fig_profile   profile likelihood in H and rho (M11)
  fig_regime    weekly option-implied H, both basins, pre/post ETF (R1.5)
  fig_heatmap   parametric solve-error over (H, nu), per box (M4)
  fig_signed    per-strike signed solve-error at n = 4, 8, 16, 32 (R1.7, M1)
  fig_hparams   held-out PDE residual and solve-error across n (R1.3)
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "figures"


def _load(name):
    with open(os.path.join("results", name + ".json")) as fh:
        d = json.load(fh)
    return d.get("payload", d)


def _save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "fig_%s.png" % name)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    print("  wrote " + path)


def fig_profile():
    d = _load("calib_uncertainty")
    c = d["canonical"]; best = min(p["weighted"] for p in d["profile_H"])
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for a, key, lab, cv in ((ax[0], "profile_H", r"$H$", c["params"]["H"]),
                            (ax[1], "profile_rho", r"$\rho$", c["params"]["rho"])):
        prof = d[key]
        x = [p["fixed"] for p in prof]; y = [1e4 * p["weighted"] for p in prof]
        a.plot(x, y, "o-", color="C0", label="weighted (objective)")
        a.plot(x, [p["unweighted_vol_bp"] / 3 for p in prof], "s--", color="C1", alpha=0.7, label="unweighted / 3")
        a.axhline(1e4 * best * 1.05, color="grey", ls=":", lw=1, label="5% above best")
        a.axvline(cv, color="k", lw=0.8, alpha=0.6)
        a.set_xlabel(lab + " (fixed)"); a.set_ylabel("IV RMSE after re-optimising the rest (vol bp)")
        a.set_title("profile in " + lab)
    ax[0].legend(fontsize=8, loc="upper right")
    _save(fig, "profile")


def fig_regime():
    d = _load("regime_hurst")
    rows = [r for r in d["fits"] if r["ok"]]
    import datetime as dt
    dates = [dt.date.fromisoformat(r["date"]) for r in rows]
    fig, ax = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True)
    for name, col in (("canonical", "C0"), ("wide", "C3")):
        H = [r["fits"][name]["H"] for r in rows]
        ax[0].plot(dates, H, ".-", color=col, lw=0.8, ms=4, label="start: %s" % name)
    ax[0].axvspan(dt.date(2024, 1, 1), dt.date(2024, 1, 31), color="grey", alpha=0.25, label="ETF approval window")
    ax[0].set_ylabel(r"$H$"); ax[0].legend(fontsize=8, ncol=3)
    dl = [r["fits"]["wide"]["weighted_vol_bp"] - r["fits"]["canonical"]["weighted_vol_bp"]
          for r in rows if "wide" in r["fits"]]
    ax[1].bar(dates[:len(dl)], dl, width=5, color=["C0" if v > 0 else "C3" for v in dl])
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_ylabel("loss(wide) $-$ loss(canonical)\n(vol bp; <0 prefers smoother $H$)")
    ax[1].axvspan(dt.date(2024, 1, 1), dt.date(2024, 1, 31), color="grey", alpha=0.25)
    _save(fig, "regime")


def fig_heatmap():
    boxes = [b for b in ("default", "wide") if os.path.exists("results/param_heatmap_%s.json" % b)]
    if not boxes:
        raise FileNotFoundError("param_heatmap_*.json")
    fig, ax = plt.subplots(1, len(boxes), figsize=(4.6 * len(boxes), 3.8), squeeze=False)
    for a, b in zip(ax[0], boxes):
        d = _load("param_heatmap_%s" % b)
        E = np.array(d["err_px_bp_mean"]); Hs = d["H_grid"]; nus = d["nu_grid"]
        im = a.imshow(E, origin="lower", aspect="auto", cmap="viridis",
                      extent=[nus[0], nus[-1], Hs[0], Hs[-1]])
        a.set_xlabel(r"$\nu$"); a.set_ylabel(r"$H$")
        r = d["regression"]
        a.set_title("%s box: interaction $d=%.2f$, partial $R^2=%.2f$" % (b, r["coef"]["d_nu_over_H"], r["partial_r2_interaction"]), fontsize=9)
        fig.colorbar(im, ax=a, label="solve-error (px bp)")
    _save(fig, "heatmap")


def fig_signed():
    d = _load("n_convergence")
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for r, col in zip(d["rows"], ("C0", "C1", "C2", "C3")):
        if "signed_solve_iv_bp" not in r:
            continue
        ax.plot(r["strikes"], r["signed_solve_iv_bp"], "o-", color=col, label="n = %d" % r["n"])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("strike $K/F$"); ax.set_ylabel("network $-$ lifted MC (vol bp)")
    ax.set_title("signed solve-error by strike, seed-averaged", fontsize=10)
    ax.legend(fontsize=8)
    _save(fig, "signed")


def fig_hparams():
    d = _load("hparams_rescored")   # scored against the 1600-step reference
    C = d["n_sweep"]
    n = [q["n"] for q in C]
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.errorbar(n, [q["solve_mean"] for q in C], yerr=[q["solve_sd"] for q in C], fmt="o-", color="C0", label="solve-error (px bp)")
    ax.set_xscale("log", base=2); ax.set_xlabel("lift dimension $n$"); ax.set_ylabel("solve-error (px bp)", color="C0")
    a2 = ax.twinx()
    a2.errorbar(n, [q["residual_mean"] for q in C], yerr=[q["residual_sd"] for q in C], fmt="s--", color="C3", label="held-out PDE residual")
    a2.set_ylabel("held-out residual (RMS)", color="C3")
    ax.set_title(d.get("verdict", "")[:70], fontsize=8)
    _save(fig, "hparams")


FIGS = [fig_profile, fig_regime, fig_heatmap, fig_signed, fig_hparams]

if __name__ == "__main__":
    only = sys.argv[1:]
    ok = skipped = 0
    for fn in FIGS:
        if only and not any(o in fn.__name__ for o in only):
            continue
        try:
            fn(); ok += 1
        except FileNotFoundError as e:
            print("  SKIP %-12s no results JSON yet (%s)" % (fn.__name__, e)); skipped += 1
        except KeyError as e:
            print("  SKIP %-12s JSON present but missing key %s" % (fn.__name__, e)); skipped += 1
    print("\n%d figure(s) written, %d skipped." % (ok, skipped))
