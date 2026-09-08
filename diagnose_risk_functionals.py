r"""
Risk functionals along the calibration ridge. (Reviewer 1, item 8; ties to M11)
================================================================================

The referee asks what the calibration uncertainty means for quantities a desk
would actually use. The profile-likelihood study (diagnose_calib_uncertainty.py)
found that the surface does not pin H: every re-optimised vector along the H
profile fits within a few per cent of the best. Those vectors ARE the calibration
uncertainty, in the only form that matters -- a set of parameter vectors the data
cannot tell apart -- so we push each of them through three risk functionals and
report the spread. If the spread is small, the non-identification of H is
harmless for these purposes; if it is large, it is not. Either answer is worth
having, and we report whichever we get.

All three functionals are priced with the Fourier pricer (the exact forward
model), never with the network, so what is measured is the model's parameter
uncertainty and nothing else. S0 = 1, r = 0, maturity 3 months.

  1. Variance-swap fair strike, sqrt(K_var) in annualised vol points, from the
     model-free log-contract replication over a fine strike grid.
  2. 25-delta risk reversal: IV(25d call) - IV(25d put), in vol points. Deltas
     are Black-Scholes deltas at each strike's own model implied vol.
  3. 90/110 collar: long 90% put, short 110% call, in price bp of spot.

Vectors, in the order they are reported:
  * canonical fit (calib_real.json)
  * wide-box refit (calib_real_wide.json), if present
  * the profile ridge: every H-profile vector within TOL of the best, from
    results/calib_uncertainty.json, if present
  * hybrid-calibration vectors from results/hybrid.json, if present
"""
import json, os, time
import numpy as np
from scipy.stats import norm

from rough_heston_fourier import price_european_fourier
from surface import implied_vol
from calibrate import PARAM_NAMES

T_RISK = 0.25
R = 0.0
TOL = 0.05                       # same ridge tolerance as the profile script
K_LOG = np.linspace(-1.5, 1.5, 301)          # log-strike grid for the log contract
FOURIER = dict(N=200, u_max=150.0, n_u=2000)


def smile(vec, T=T_RISK, k=K_LOG):
    """Model call prices and implied vols on a log-strike grid, S0 = 1."""
    H, V0, theta, lam, nu, rho = vec
    p = dict(V0=V0, theta=theta, lam=lam, nu=nu, rho=rho)
    K = np.exp(k)
    C = np.atleast_1d(price_european_fourier(1.0, K, T, R, p, H, **FOURIER))
    P = C - 1.0 + K                                  # parity, r = 0
    iv = np.array([implied_vol(max(c, 1e-12), 1.0, kk, T, R, "C") for c, kk in zip(C, K)])
    return K, C, P, iv


def variance_swap(K, C, P, T=T_RISK):
    """sqrt(K_var): (2/T) [ int_0^F P/K^2 dK + int_F^inf C/K^2 dK ], F = 1."""
    otm = np.where(K < 1.0, P, C)
    kvar = 2.0 / T * np.trapezoid(otm / K ** 2, K)
    return float(np.sqrt(max(kvar, 0.0)))


def risk_reversal(K, iv, T=T_RISK):
    """IV at +25 delta (call) minus IV at -25 delta (put), vol points."""
    ok = np.isfinite(iv) & (iv > 0)
    K, iv = K[ok], iv[ok]
    d1 = (np.log(1.0 / K) + 0.5 * iv ** 2 * T) / (iv * np.sqrt(T))
    dc = norm.cdf(d1)                                # call delta
    dp = dc - 1.0                                    # put delta
    # call delta falls with K, put delta rises with K: interpolate on monotone pieces
    ic = np.argsort(dc); ip = np.argsort(dp)
    iv_c = float(np.interp(0.25, dc[ic], iv[ic]))
    iv_p = float(np.interp(-0.25, dp[ip], iv[ip]))
    return iv_c - iv_p, iv_c, iv_p


def collar(vec, T=T_RISK):
    """Long 90% put, short 110% call; price bp of spot."""
    H, V0, theta, lam, nu, rho = vec
    p = dict(V0=V0, theta=theta, lam=lam, nu=nu, rho=rho)
    C = np.atleast_1d(price_european_fourier(1.0, np.array([0.9, 1.1]), T, R, p, H, **FOURIER))
    P90 = C[0] - 1.0 + 0.9
    return 1e4 * float(P90 - C[1])


def functionals(vec):
    K, C, P, iv = smile(vec)
    vs = variance_swap(K, C, P)
    rr, ivc, ivp = risk_reversal(K, iv)
    return dict(varswap_vol=vs, rr25_vol=rr, iv_25d_call=ivc, iv_25d_put=ivp,
                collar_px_bp=collar(vec))


def collect_vectors():
    out = []
    cal = json.load(open("calib_real.json"))
    out.append(("canonical", [cal[k] for k in PARAM_NAMES], "H=%.3f" % cal["H"]))
    if os.path.exists("calib_real_wide.json"):
        w = json.load(open("calib_real_wide.json"))
        out.append(("wide-box refit", [w[k] for k in PARAM_NAMES], "H=%.3f" % w["H"]))
    if os.path.exists("results/calib_uncertainty.json"):
        d = json.load(open("results/calib_uncertainty.json"))
        d = d.get("payload", d)
        prof = d["profile_H"]
        best = min(p["weighted"] for p in prof)
        for p in prof:
            if p["weighted"] <= best * (1 + TOL):
                v = [p["params"][k] for k in PARAM_NAMES]
                out.append(("ridge", v, "H=%.3f (+%.1f%%)" % (p["fixed"], 100 * (p["weighted"] / best - 1))))
    if os.path.exists("results/hybrid.json"):
        d = json.load(open("results/hybrid.json"))
        d = d.get("payload", d)
        for i, v in enumerate(d.get("vectors", [])):
            out.append(("hybrid seed %d" % i, [v[k] for k in PARAM_NAMES], "H=%.3f" % v["H"]))
    return out


if __name__ == "__main__":
    vecs = collect_vectors()
    print("%d parameter vectors\n" % len(vecs))
    print("%-16s %-16s %9s %9s %9s" % ("source", "", "varswap", "RR25", "collar"))
    print("%-16s %-16s %9s %9s %9s" % ("", "", "vol pts", "vol pts", "px bp"))
    rows = []
    t0 = time.time()
    for src, vec, tag in vecs:
        f = functionals(vec)
        rows.append(dict(source=src, tag=tag, params=dict(zip(PARAM_NAMES, map(float, vec))), **f))
        print("%-16s %-16s %9.2f %9.2f %9.1f" % (src, tag, 100 * f["varswap_vol"],
                                                  100 * f["rr25_vol"], f["collar_px_bp"]), flush=True)

    def spread(key, sel):
        a = np.array([r[key] for r in rows if sel(r)])
        return dict(n=int(len(a)), min=float(a.min()), max=float(a.max()),
                    range=float(a.max() - a.min())) if len(a) else None

    ridge = lambda r: r["source"] == "ridge"
    allv = lambda r: True
    summary = {}
    for key, scale, unit in (("varswap_vol", 100, "vol pts"), ("rr25_vol", 100, "vol pts"),
                             ("collar_px_bp", 1, "px bp")):
        s_r, s_a = spread(key, ridge), spread(key, allv)
        summary[key] = dict(unit=unit, ridge=s_r, all=s_a)
        if s_r:
            print("\n  %-13s ridge (%d vectors): %.2f to %.2f %s, range %.2f"
                  % (key, s_r["n"], scale * s_r["min"], scale * s_r["max"], unit, scale * s_r["range"]))
    print("\n%.0f s" % (time.time() - t0))

    from resultio import dump
    dump("risk_functionals", dict(T=T_RISK, tol=TOL, fourier=FOURIER, rows=rows, summary=summary))
    print("saved -> results/risk_functionals.json")
