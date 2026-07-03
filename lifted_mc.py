"""
Monte Carlo pricing of rough Heston via the Markovian LIFT.
===========================================================

The lift turns rough Heston into an (n+1)-dimensional Markovian SDE that we
can simulate with a standard Euler scheme. This is a genuinely INDEPENDENT
pricer from the fractional-Riccati Fourier method: it approximates the model
(kernel error, controlled by n) and the dynamics (time-discretisation +
statistical error), whereas Fourier approximates only the ODE.

Lifted dynamics (Abi Jaber & El Euch 2019). With weights/speeds (c_i, x_i):

    V_t = V0 + sum_i c_i U^i_t,
    dU^i_t = ( -x_i U^i_t + lam(theta - V_t) ) dt + nu sqrt(V_t^+) dW_t,   U^i_0 = 0,
    dX_t = -1/2 V_t^+ dt + sqrt(V_t^+) dB_t,   d<W,B> = rho dt,

where X = log(S/S0) under the risk-neutral measure (r handled via forward).
V_t^+ = max(V_t, 0) (full-truncation Euler, standard for Heston-type models).

Note every factor shares the SAME Brownian increment dW; they differ only in
mean-reversion x_i. This is the defining structure of the lift.

Variance reduction: antithetic variates on the Brownian increments.
"""

import numpy as np


def price_european_mc(S0, K, T, r, params, lift,
                      n_paths=200_000, n_steps=300, seed=0,
                      antithetic=True, option="call", return_stderr=True):
    """
    Monte Carlo price under the lifted rough Heston model.

    params : dict(V0, theta, lam, nu, rho)
    lift   : (c, x) from rough_heston_lift.lift_weights_geometric
    Returns (prices, stderr) if return_stderr else prices, aligned with K.
    """
    c, x = lift
    n = c.shape[0]
    V0 = params["V0"]; theta = params["theta"]; lam = params["lam"]
    nu = params["nu"]; rho = params["rho"]

    K = np.atleast_1d(np.asarray(K, dtype=float))
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sdt = np.sqrt(dt)

    if antithetic:
        half = n_paths // 2
        n_paths = 2 * half

    # correlated Brownians: B = rho W + sqrt(1-rho^2) Wperp
    rho2 = np.sqrt(1.0 - rho * rho)

    # state
    U = np.zeros((n_paths, n))          # factors
    X = np.zeros(n_paths)               # log-price
    x_row = x[None, :]

    # Exponential-Euler weights for the stiff linear term -x_i U_i.
    # Over a step, dU_i = -x_i U_i dt + (source_i) dt + (diffusion_i) dW
    # is integrated with the linear part exact:
    #   U_i(t+dt) = E_i U_i + phi1_i * dt * source_i + phi1_i * diffusion_i * dW
    # where E_i = e^{-x_i dt}, phi1_i = (1-E_i)/(x_i dt) -> 1 as x_i dt -> 0.
    z = x * dt
    E = np.exp(-z)                                    # (n,)
    phi1 = np.where(z > 1e-12, (1.0 - E) / np.where(z > 1e-12, z, 1.0), 1.0)
    E_row = E[None, :]
    w_src = (phi1 * dt)[None, :]                      # weight on the drift source
    w_dif = phi1[None, :]                             # weight on the diffusion

    for _ in range(n_steps):
        if antithetic:
            zW = rng.standard_normal(half)
            zP = rng.standard_normal(half)
            dW = np.concatenate([zW, -zW]) * sdt
            dWp = np.concatenate([zP, -zP]) * sdt
        else:
            dW = rng.standard_normal(n_paths) * sdt
            dWp = rng.standard_normal(n_paths) * sdt

        V = V0 + U @ c                   # (n_paths,)
        Vp = np.maximum(V, 0.0)
        sqrtV = np.sqrt(Vp)

        # exponential-Euler factor update; mean-reversion handled exactly
        source = (lam * (theta - Vp))[:, None]        # (n_paths,1) shared across factors
        diffusion = (nu * sqrtV)[:, None]             # (n_paths,1)
        U = U * E_row + source * w_src + diffusion * dW[:, None] * w_dif

        # log-price update with correlated Brownian
        dB = rho * dW + rho2 * dWp
        X = X - 0.5 * Vp * dt + sqrtV * dB

    fwd = S0 * np.exp(r * T)
    ST = fwd * np.exp(X)                 # forward-measure spot at T

    disc = np.exp(-r * T)
    prices = np.empty(K.shape[0])
    stderrs = np.empty(K.shape[0])
    for i, k in enumerate(K):
        if option == "call":
            payoff = np.maximum(ST - k, 0.0)
        else:
            payoff = np.maximum(k - ST, 0.0)
        disc_payoff = disc * payoff
        prices[i] = disc_payoff.mean()
        stderrs[i] = disc_payoff.std(ddof=1) / np.sqrt(n_paths)

    prices = prices if prices.shape[0] > 1 else prices
    if return_stderr:
        return prices, stderrs
    return prices


if __name__ == "__main__":
    from rough_heston_lift import lift_weights_geometric
    from scipy.stats import norm
    def bs_call(S0, K, T, sig):
        d1 = (np.log(S0/K) + 0.5*sig**2*T)/(sig*np.sqrt(T)); d2 = d1 - sig*np.sqrt(T)
        return S0*norm.cdf(d1) - K*norm.cdf(d2)

    # Degeneracy: nu=0, lam=0 -> constant variance -> BS. MC should match.
    H = 0.1; alpha = H + 0.5
    p = dict(V0=0.04, theta=0.04, lam=0.0, nu=0.0, rho=-0.7)
    c, x = lift_weights_geometric(alpha, 20)
    Ks = np.array([0.9, 1.0, 1.1])
    price, se = price_european_mc(1.0, Ks, 1.0, 0.0, p, (c, x),
                                  n_paths=100_000, n_steps=200, seed=1)
    bs = bs_call(1.0, Ks, 1.0, 0.2)
    print("MC nu=0,lam=0 vs BS:")
    for k, pr, s, b in zip(Ks, price, se, bs):
        print(f"  K={k}: MC={pr:.5f} +/- {1.96*s:.5f}   BS={b:.5f}   ok={abs(pr-b)<3*s}")
