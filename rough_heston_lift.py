"""
Markovian lift of the rough Heston model.
=========================================

Rough Heston (El Euch & Rosenbaum 2019) variance dynamics:

    V_t = V_0 + 1/Gamma(alpha) * int_0^t (t-s)^{alpha-1} * lam*(theta - V_s) ds
              + 1/Gamma(alpha) * int_0^t (t-s)^{alpha-1} * nu*sqrt(V_s) dW_s

with alpha = H + 1/2, H the Hurst exponent (rough <=> H < 1/2 <=> alpha in (1/2, 1)).

The fractional kernel  K(t) = t^{alpha-1} / Gamma(alpha)  is NON-Markovian.
The lift (Abi Jaber & El Euch 2019) approximates it by a sum of n exponentials

    K(t) ~ sum_{i=1}^n  c_i * exp(-x_i * t),

turning the model into an (n+1)-dimensional Markovian system: n auxiliary
factors U^i plus the spot. Each factor obeys an OU-type SDE and

    V_t = V_0 + sum_i c_i * U^i_t .

This module builds the (c_i, x_i) weights/mean-reversions. We provide the
geometric-grid quadrature of Abi Jaber & El Euch (2019, Section 3), which is
robust and the standard reference construction.
"""

import numpy as np
from scipy.special import gamma as gamma_fn


def lift_weights_geometric(alpha, n, r=None, T=1.0):
    """
    Geometric-grid multi-factor approximation of the fractional kernel
    K(t) = t^{alpha-1}/Gamma(alpha),  following Abi Jaber & El Euch (2019).

    The mean-reversions x_i are placed on a geometric grid of auxiliary
    nodes (eta_j), and the weights c_i and nodes x_i are obtained from

        c_i = int_{eta_{i-1}}^{eta_i} mu(dx),   x_i = (1/c_i) int_{eta_{i-1}}^{eta_i} x mu(dx)

    where mu(dx) = x^{-alpha} / (Gamma(alpha) Gamma(1-alpha)) dx is the
    spectral measure whose Laplace transform is exactly the fractional kernel:
        K(t) = int_0^infty e^{-x t} mu(dx).

    Parameters
    ----------
    alpha : float in (0.5, 1)   ( = H + 0.5 )
    n     : int, number of factors
    r     : float, geometric ratio between nodes. If None, uses the
            near-optimal rule r = 1 + 10/n * (from AJ-EE asymptotics), which
            works well for typical maturities.
    T     : float, maturity used only to scale the node placement.

    Returns
    -------
    c : (n,) weights c_i > 0
    x : (n,) mean-reversions x_i > 0
    """
    if r is None:
        # The node RANGE must widen as n grows to resolve both the slow decay
        # (small x, large t) and the singularity (large x, small t) of the
        # fractional kernel. Empirically r slightly > 1 with a total span that
        # grows like exp(sqrt(n)) works well; AJ-EE Prop 3.3 gives the scaling.
        r = np.exp(4.0 / np.sqrt(n))
    # spectral measure density prefactor
    pref = 1.0 / (gamma_fn(alpha) * gamma_fn(1.0 - alpha))

    # geometric grid of node boundaries eta_0 < eta_1 < ... < eta_n
    # centre the grid; AJ-EE use eta_i = r^{i - n/2}
    i = np.arange(n + 1)
    eta = r ** (i - n / 2.0)

    # moments of mu on each interval:  mu(dx) = pref * x^{-alpha} dx
    #   c_i   = pref * [ x^{1-alpha}/(1-alpha) ]_{eta_{i-1}}^{eta_i}
    #   m1_i  = pref * [ x^{2-alpha}/(2-alpha) ]_{eta_{i-1}}^{eta_i}
    #   x_i   = m1_i / c_i
    lo, hi = eta[:-1], eta[1:]
    c = pref * (hi ** (1.0 - alpha) - lo ** (1.0 - alpha)) / (1.0 - alpha)
    m1 = pref * (hi ** (2.0 - alpha) - lo ** (2.0 - alpha)) / (2.0 - alpha)
    x = m1 / c
    return c, x


def kernel_true(t, alpha):
    """Exact fractional kernel K(t)=t^{alpha-1}/Gamma(alpha)."""
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t)
    pos = t > 0
    out[pos] = t[pos] ** (alpha - 1.0) / gamma_fn(alpha)
    return out


def kernel_approx(t, c, x):
    """Approximate kernel sum_i c_i exp(-x_i t)."""
    t = np.asarray(t, dtype=float)
    return (c[None, :] * np.exp(-np.outer(t, x))).sum(axis=1)


if __name__ == "__main__":
    # quick self-check: L2 error of the kernel approximation vs n
    H = 0.1
    alpha = H + 0.5
    tt = np.linspace(1e-3, 1.0, 400)
    Ktrue = kernel_true(tt, alpha)
    print(f"Kernel approximation error (H={H}, alpha={alpha:.2f})")
    print(f"{'n':>4} {'rel L2 err':>12} {'x range':>22}")
    for n in [5, 10, 20, 50, 100]:
        c, x = lift_weights_geometric(alpha, n)
        Kapp = kernel_approx(tt, c, x)
        # weight error near t=0 down (kernel is integrable-singular there);
        # report relative L2 on [t0, T]
        err = np.sqrt(np.trapezoid((Kapp - Ktrue) ** 2, tt) /
                      np.trapezoid(Ktrue ** 2, tt))
        print(f"{n:>4} {err:>12.4e}   [{x.min():.2e}, {x.max():.2e}]")
