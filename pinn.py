"""
Stage 4 - Physics-Informed Neural Network for the LIFTED rough Heston PDE.
==========================================================================

THE PDE
-------
Under the Markovian lift with factors (c_i, x_i), the state is
    (t, x=log-spot, u_1,...,u_n),   with variance  V = V0 + sum_i c_i u_i.
The discounted price P(t, x, u) solves the backward Kolmogorov PDE

  P_t + (r - 1/2 V) P_x + 1/2 V P_xx
      + sum_i [ -x_i u_i + lam(theta - V) ] P_{u_i}
      + 1/2 nu^2 V sum_{i,j} P_{u_i u_j}
      + rho nu V sum_i P_{x u_i}
      - r P = 0,

with terminal condition P(T, x, u) = payoff(e^x). All factors share the same
Brownian motion, hence the FULL double sum over (i,j) in the u-diffusion and
the cross term with x. This is the exact (n+1)-dimensional Markovian PDE whose
non-Markovian limit (n -> infinity) is rough Heston.

WHY A PINN HERE (the thesis argument)
-------------------------------------
- Finite differences on this PDE die as n grows (curse of dimensionality): a
  grid in (x,u_1..u_n) is n+1 dimensional. A PINN is meshfree, so it scales.
- The PDE is exactly known (no data needed), so this is a pure physics loss;
  market data enters only through the calibrated coefficients.
- Once trained across a payoff/parameter range, inference is a single forward
  pass -> instant Greeks (autodiff) and instant recalibration.

TRAINING CHOICES THAT MATTER (and are implemented)
--------------------------------------------------
- Solve in TIME-TO-MATURITY tau = T - t, so the terminal condition becomes an
  INITIAL condition at tau=0 -> better conditioned.
- HARD-CONSTRAIN the initial condition via an ansatz
      P = payoff_smooth(x) + tau * N_theta(tau,x,u),
  so the network never has to fight to satisfy tau=0; it only learns the
  correction. This removes the loss-balancing headache that sinks naive PINNs.
- Work in discounted, forward-moneyness coordinates and a REDUCED factor set
  (n small, e.g. 3-5) so the demo trains in seconds-minutes on CPU; the code is
  identical for larger n on a GPU.
- Smooth the kinked payoff slightly (softplus) near the strike to make the
  initial condition differentiable for the residual.
"""

import numpy as np
import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)


class MLP(nn.Module):
    """Simple tanh MLP; tanh gives smooth high-order derivatives for the PDE."""
    def __init__(self, d_in, width=64, depth=4, d_out=1):
        super().__init__()
        layers = [nn.Linear(d_in, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, d_out)]
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, z):
        return self.net(z)


class RoughHestonPINN:
    """
    PINN pricer for a European CALL under the lifted rough Heston PDE.

    Prices are expressed on the log-spot x = log(S) grid, discounted.
    We fix (K, r) and the calibrated model params; the network learns
    P(tau, x, u). Greeks come from autodiff of the trained P.
    """
    def __init__(self, params, lift, K=1.0, r=0.0, T=1.0,
                 width=64, depth=4, device="cpu", beta_smooth=40.0):
        self.p = params
        c, x = lift
        self.c = torch.tensor(c)
        self.x = torch.tensor(x)
        self.n = len(c)
        self.K = K; self.r = r; self.T = T
        self.device = device
        self.beta = beta_smooth       # payoff-smoothing sharpness

        # network input: (tau, x, u_1..u_n) -> 1
        self.net = MLP(2 + self.n, width, depth, 1).to(device)

        # coordinate ranges for collocation sampling (log-moneyness around ATM)
        self.x_lo, self.x_hi = np.log(K) - 0.6, np.log(K) + 0.6
        self.u_scale = 0.05           # factors start at 0, stay small; sample O(0.05)

    # ---- smoothed terminal payoff (call) in log-spot coords ----
    def payoff_smooth(self, x):
        S = torch.exp(x)
        # softplus approximation of max(S-K,0): (1/beta) log(1+exp(beta (S-K)))
        z = self.beta * (S - self.K)
        return torch.nn.functional.softplus(z) / self.beta

    def V_of_u(self, u):
        """Variance V = V0 + sum_i c_i u_i (clamped positive for stability)."""
        V = self.p["V0"] + (u * self.c).sum(dim=1, keepdim=True)
        return torch.clamp(V, min=1e-6)

    def _bs_baseline(self, tau, x):
        """
        Black-Scholes call with the model's initial variance V0, computed in
        torch so it is differentiable for the PDE residual. As tau -> 0 the BS
        call converges to the exact payoff max(S-K,0), so this baseline already
        satisfies the initial condition WITHOUT any blending. The network learns
        only the rough-vol correction on top (identically zero when nu=0).
        """
        S = torch.exp(x)
        sig = np.sqrt(self.p["V0"])
        sqrt_tau = torch.sqrt(torch.clamp(tau, min=1e-10))
        d1 = (torch.log(S/self.K) + (self.r + 0.5*sig*sig)*tau) / (sig*sqrt_tau)
        d2 = d1 - sig*sqrt_tau
        Phi = lambda z: 0.5*(1.0 + torch.erf(z/np.sqrt(2.0)))
        return S*Phi(d1) - self.K*torch.exp(-self.r*tau)*Phi(d2)

    def price_ansatz(self, tau, x, u):
        """
        Ansatz:  P = BS_baseline(tau,x; V0)  +  tau * N(tau,x,u).
        BS baseline solves tau=0 exactly and the flat-vol PDE exactly; the
        tau-factor forces the learned correction N to vanish at tau=0. When
        nu=0 and lam=0 the true correction is zero, so the network only has to
        learn a genuinely small, smooth stochastic-vol term.
        """
        z = torch.cat([tau, x, u], dim=1)
        N = self.net(z)
        base = self._bs_baseline(tau, x)
        return base + tau * N

    def pde_residual(self, tau, x, u):
        """Compute the backward-Kolmogorov residual in tau-form."""
        tau = tau.clone().requires_grad_(True)
        x = x.clone().requires_grad_(True)
        u = u.clone().requires_grad_(True)

        P = self.price_ansatz(tau, x, u)
        ones = torch.ones_like(P)

        # first derivatives
        P_tau = torch.autograd.grad(P, tau, ones, create_graph=True)[0]
        P_x = torch.autograd.grad(P, x, ones, create_graph=True)[0]
        grad_u = torch.autograd.grad(P, u, ones, create_graph=True)[0]  # (B,n)

        # second derivatives in x
        P_xx = torch.autograd.grad(P_x, x, torch.ones_like(P_x),
                                   create_graph=True)[0]

        V = self.V_of_u(u)                        # (B,1)
        nu = self.p["nu"]; rho = self.p["rho"]
        lam = self.p["lam"]; theta = self.p["theta"]

        # --- u-diffusion double sum: 1/2 nu^2 V sum_{i,j} P_{u_i u_j} ---
        # sum_{i,j} P_{u_i u_j} = 1^T Hess_u 1 = d/ds [ grad_u . 1 ] summed.
        # Compute J = sum_i P_{u_i} (scalar per row), then its u-gradient gives
        # sum_j d/du_j (sum_i P_{u_i}) = sum_{i,j} P_{u_i u_j}.
        sum_grad_u = grad_u.sum(dim=1, keepdim=True)       # (B,1)
        hess_uu_full = torch.autograd.grad(
            sum_grad_u, u, torch.ones_like(sum_grad_u), create_graph=True)[0]
        double_sum_uu = hess_uu_full.sum(dim=1, keepdim=True)   # sum_{i,j}

        # --- cross term: rho nu V sum_i P_{x u_i} ---
        # d/dx of (sum_i P_{u_i}) = sum_i P_{x u_i}
        cross = torch.autograd.grad(
            sum_grad_u, x, torch.ones_like(sum_grad_u), create_graph=True)[0]

        # --- factor drift term: sum_i [ -x_i u_i + lam(theta - V) ] P_{u_i} ---
        drift_u = (-self.x[None, :] * u + lam*(theta - V))    # (B,n)
        drift_term = (drift_u * grad_u).sum(dim=1, keepdim=True)

        res = (P_tau                                    # note: tau-form flips sign
               - (self.r - 0.5*V) * P_x
               - 0.5*V * P_xx
               - drift_term
               - 0.5*nu*nu*V * double_sum_uu
               - rho*nu*V * cross
               + self.r * P)
        return res

    # ---- collocation sampling ----
    def sample(self, n_col, n_bnd):
        dev = self.device
        tau = torch.rand(n_col, 1, device=dev) * self.T
        x = torch.rand(n_col, 1, device=dev)*(self.x_hi-self.x_lo)+self.x_lo
        u = (torch.rand(n_col, self.n, device=dev)-0.5)*2*self.u_scale
        # extra collocation concentrated near the strike & short tau (hard region)
        tau2 = torch.rand(n_bnd, 1, device=dev)**2 * self.T*0.2
        x2 = torch.randn(n_bnd, 1, device=dev)*0.05 + np.log(self.K)
        u2 = (torch.rand(n_bnd, self.n, device=dev)-0.5)*2*self.u_scale
        tau = torch.cat([tau, tau2]); x = torch.cat([x, x2]); u = torch.cat([u, u2])
        return tau, x, u

    def train(self, iters=3000, n_col=2000, n_bnd=500, lr=1e-3, log_every=500):
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
        history = []
        for it in range(iters):
            opt.zero_grad()
            tau, x, u = self.sample(n_col, n_bnd)
            res = self.pde_residual(tau, x, u)
            loss = (res**2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            opt.step(); sched.step()
            if it % log_every == 0 or it == iters-1:
                lval = float(loss.detach())
                history.append((it, lval))
                print(f"  iter {it:5d}  PDE loss {lval:.3e}")
        return history

    # ---- inference ----
    @torch.no_grad()
    def price(self, S, tau, u=None):
        S = np.atleast_1d(S).astype(float)
        x = torch.tensor(np.log(S)).reshape(-1, 1)
        tt = torch.full_like(x, float(tau))
        if u is None:
            u = torch.zeros(x.shape[0], self.n)     # start-of-life factors = 0
        else:
            u = torch.tensor(u).reshape(x.shape[0], self.n)
        P = self.price_ansatz(tt, x, u)
        return P.numpy().ravel()

    def delta(self, S, tau):
        """Delta via autodiff: dP/dS = (1/S) dP/dx."""
        S = np.atleast_1d(S).astype(float)
        x = torch.tensor(np.log(S)).reshape(-1, 1).requires_grad_(True)
        tt = torch.full_like(x, float(tau))
        u = torch.zeros(x.shape[0], self.n)
        P = self.price_ansatz(tt, x, u)
        dPdx = torch.autograd.grad(P, x, torch.ones_like(P))[0]
        return (dPdx.detach().numpy().ravel() / S)


if __name__ == "__main__":
    from rough_heston_lift import lift_weights_geometric
    # small demo config so it trains fast on CPU
    params = dict(V0=0.04, theta=0.04, lam=0.3, nu=0.3, rho=-0.7)
    H = 0.1; n = 4
    c, x = lift_weights_geometric(H+0.5, n)
    pinn = RoughHestonPINN(params, (c, x), K=1.0, r=0.0, T=1.0,
                           width=48, depth=4)
    print(f"Training PINN on lifted PDE (n={n} factors)...")
    pinn.train(iters=1500, n_col=1500, n_bnd=400, log_every=300)
    S = np.array([0.9, 1.0, 1.1])
    print("PINN price :", np.round(pinn.price(S, tau=1.0), 5))
    print("PINN delta :", np.round(pinn.delta(S, tau=1.0), 5))
