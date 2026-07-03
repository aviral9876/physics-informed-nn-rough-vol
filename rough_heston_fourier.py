"""
Fourier pricing under rough Heston via the fractional Riccati equation.
=======================================================================

This is the CANONICAL semi-analytic pricer (El Euch & Rosenbaum 2019).
It does NOT use the lift; it solves the true fractional Riccati equation for
the characteristic function. It therefore serves as an independent ground
truth to validate the lifted Monte Carlo simulator against.

Rough Heston CF of the log-price X_T = log(S_T / (S_0 e^{rT})):

    phi(u, T) = exp( theta*lam * I^1 h(u,.) (T) + V0 * I^{1-alpha} h(u,.) (T) )

where h solves the fractional Riccati equation (Caputo derivative of order alpha):

    D^alpha h(u,t) = 1/2 (-u^2 - i u) + (i u rho nu - lam) h(u,t) + (nu^2/2) h(u,t)^2,
    I^{1-alpha} h (0) = 0,

with alpha = H + 1/2. Here I^beta is the Riemann-Liouville fractional integral.
We solve it with the fractional Adams (predictor-corrector) scheme of
Diethelm-Ford-Freed, which is the standard method used by El Euch-Rosenbaum.

Pricing uses the Lewis (2001) inversion, identical to the lifted version.
"""

import numpy as np
from scipy.special import gamma as gamma_fn


def _frac_riccati_adams(u, T, params, alpha, N):
    """
    Solve the fractional Riccati equation for a vector of complex frequencies u
    on a uniform grid of N steps to time T, returning h(u, t_k) for all k, and
    the two fractional integrals needed for the log-CF at t=T.

    Uses the Adams-Bashforth-Moulton predictor-corrector for FDEs
    (Diethelm, Ford & Freed 2002).
    """
    lam = params["lam"]; nu = params["nu"]; rho = params["rho"]
    u = np.asarray(u, dtype=complex)
    M = u.shape[0]

    dt = T / N
    # RHS of the fractional ODE as a function of h (vectorised over u)
    a0 = 0.5 * (-u * u - 1j * u)            # constant term
    a1 = 1j * u * rho * nu - lam            # linear coeff
    a2 = 0.5 * nu * nu                      # quadratic coeff

    # cap |h| to avoid overflow in the highest-frequency modes, whose
    # contribution to the Fourier integral is negligible (integrand ~ 1/u^2
    # times a bounded-modulus CF). This keeps the solver finite everywhere.
    HCAP = 1e6
    def frhs(h):
        h = np.where(np.abs(h) > HCAP, HCAP * h / np.abs(h), h)
        return a0 + a1 * h + a2 * h * h

    # Adams scheme weights (Diethelm-Ford-Freed) for the fractional integral
    # of order alpha. b-weights for the predictor, a-weights for the corrector.
    ga = gamma_fn(alpha)
    ga2 = gamma_fn(alpha + 2.0)

    # storage
    h = np.zeros((N + 1, M), dtype=complex)   # h at each time node
    f = np.zeros((N + 1, M), dtype=complex)   # f = frhs(h) at each node
    f[0] = frhs(h[0])                          # h[0]=0

    # precompute b-weights: b_{k} = ((k+1)^alpha - k^alpha)
    k = np.arange(N + 1)
    bcoef = (k + 1.0) ** alpha - k ** alpha    # length N+1

    # a-weights depend on distance; we build them on the fly per step.
    for n in range(1, N + 1):
        # --- predictor: fractional Adams-Bashforth ---
        # h_P(t_n) = (dt^alpha/Gamma(alpha+1)) * sum_{j=0}^{n-1} b_{n-1-j} f_j
        j = np.arange(n)
        w_b = bcoef[n - 1 - j]                 # (n,)
        pred = (dt ** alpha / (alpha * ga)) * (w_b[:, None] * f[:n]).sum(axis=0)

        # --- corrector: fractional Adams-Moulton ---
        # a-weights (Diethelm): for j=0
        #   a_0 = (n-1)^{alpha+1} - (n-1-alpha) n^alpha
        # for 1<=j<=n-1:
        #   a_j = (n-j+1)^{alpha+1} + (n-j-1)^{alpha+1} - 2 (n-j)^{alpha+1}
        # for j=n: a_n = 1
        aw = np.zeros(n + 1)
        jj = np.arange(1, n)
        aw[0] = (n - 1.0) ** (alpha + 1.0) - (n - 1.0 - alpha) * n ** alpha
        aw[1:n] = ((n - jj + 1.0) ** (alpha + 1.0)
                   + (n - jj - 1.0) ** (alpha + 1.0)
                   - 2.0 * (n - jj) ** (alpha + 1.0))
        aw[n] = 1.0
        corr_sum = (aw[:n, None] * f[:n]).sum(axis=0)
        fp = frhs(pred)
        h[n] = (dt ** alpha / ga2) * (corr_sum + aw[n] * fp)
        f[n] = frhs(h[n])

    # ---- assemble log-CF at t=T ----
    # log phi = theta*lam * I^1 h (T)  +  V0 * I^{1-alpha} h (T)
    # I^1 h (T) = int_0^T h dt  (trapezoid)
    tgrid = np.linspace(0.0, T, N + 1)
    I1 = np.trapezoid(h, tgrid, axis=0)                      # (M,)

    # I^{1-alpha} h (T): Riemann-Liouville integral of order (1-alpha)
    #   = 1/Gamma(1-alpha) int_0^T (T-s)^{-alpha} h(s) ds
    beta = 1.0 - alpha
    s = tgrid
    weight = np.zeros(N + 1)
    # convolution-type quadrature via product-trapezoid on (T-s)^{beta-1}
    # careful with singularity at s=T (T-s)^{-alpha}; use analytic subinterval weights
    # Use fractional integral quadrature: I^beta h(T) ~ dt^beta/Gamma(beta+2) sum a_j h_j
    gb2 = gamma_fn(beta + 2.0)
    aw = np.zeros(N + 1)
    n = N
    jj = np.arange(1, n)
    aw[0] = (n - 1.0) ** (beta + 1.0) - (n - 1.0 - beta) * n ** beta
    aw[1:n] = ((n - jj + 1.0) ** (beta + 1.0)
               + (n - jj - 1.0) ** (beta + 1.0)
               - 2.0 * (n - jj) ** (beta + 1.0))
    aw[n] = 1.0
    Ibeta = (dt ** beta / gb2) * (aw[:, None] * h).sum(axis=0)   # (M,)

    log_phi = params["theta"] * params["lam"] * I1 + params["V0"] * Ibeta
    return log_phi


def price_european_fourier(S0, K, T, r, params, H,
                           N=200, u_max=200.0, n_u=1500, option="call"):
    """
    Price European options under rough Heston via the fractional Riccati CF
    and Lewis inversion.

    params: dict(V0, theta, lam, nu, rho)
    H:      Hurst exponent (alpha = H + 0.5)
    N:      time steps for the fractional Adams solver
    """
    alpha = H + 0.5
    K = np.atleast_1d(np.asarray(K, dtype=float))
    fwd = S0 * np.exp(r * T)
    k = np.log(fwd / K)

    u = np.linspace(1e-8, u_max, n_u)
    shifted = u - 0.5j
    log_phi = _frac_riccati_adams(shifted, T, params, alpha, N)
    phi = np.exp(log_phi)

    integ = np.real(np.exp(1j * np.outer(k, u)) * phi[None, :]) / (u ** 2 + 0.25)
    integral = np.trapezoid(integ, u, axis=1)

    call = fwd * np.exp(-r * T) - (np.sqrt(fwd * K) * np.exp(-r * T) / np.pi) * integral
    call = np.maximum(call, 0.0)
    price = call if option == "call" else call - (fwd - K) * np.exp(-r * T)
    return price if price.shape[0] > 1 else float(price[0])


if __name__ == "__main__":
    from scipy.stats import norm
    def bs_call(S0, K, T, sig):
        d1 = (np.log(S0/K) + 0.5*sig**2*T)/(sig*np.sqrt(T)); d2 = d1 - sig*np.sqrt(T)
        return S0*norm.cdf(d1) - K*norm.cdf(d2)

    S0, r, T = 1.0, 0.0, 1.0
    Ks = np.array([0.8, 0.9, 1.0, 1.1, 1.2])

    # Degeneracy check 1: nu=0, lam=0 -> constant variance V0 -> Black-Scholes
    p = dict(V0=0.04, theta=0.04, lam=0.0, nu=0.0, rho=-0.7)
    price = price_european_fourier(S0, Ks, T, r, p, H=0.1, N=300)
    bs = bs_call(S0, Ks, T, np.sqrt(0.04))
    print("Check 1  nu=0,lam=0 -> BS(sig=0.2):")
    print("  Fourier:", np.round(price, 5))
    print("  BS     :", np.round(bs, 5))
    print("  max|diff|:", float(np.max(np.abs(price - bs))))
