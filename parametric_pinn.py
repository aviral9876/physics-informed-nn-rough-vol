"""
Stage 6 - PARAMETRIC PINN: one network that prices across a parameter box.
=========================================================================

The single-point PINN (pinn.py) fixes (H, nu, rho, lam, theta) and learns
P(tau, x, u). Here we add those five model parameters as NETWORK INPUTS and
train over a whole neighbourhood of the BTC calibration, so ONE trained net
prices the lifted rough-Heston PDE for any parameter vector in the box -- and,
crucially, is DIFFERENTIABLE in the parameters, enabling gradient-descent
calibration with ZERO Fourier calls (see calibrate_via_pinn).

Design choices vs the single-point PINN:
- n is FIXED (crossover rule, see the study): at T~0.15 the lift-error meets the
  solve-error near n~8-12, so we use n=10 -- lift no longer dominates (n=4's
  16 bp) and we avoid wasted factors / the larger high-n solve bias (n=32).
- The lift (c_i, x_i) depends on H, so we PRECOMPUTE it on a fine H-grid and
  gather per collocation point (H enters the net as an exact continuous input).
- Baseline uses the closed-form mean-reversion variance
  w(tau)/tau = theta + (V0-theta)(1-e^{-lam tau})/(lam tau), which is per-point,
  cheap and differentiable; it captures the V0->theta lift that dominates the
  level, and the net learns the rest (cf. Task 4c).
- V0 is held at the calibration value (not in the parameter list).
- Same stabilisers as pinn.py: self-adaptive per-term weights + tau-curriculum.
"""

import numpy as np
import torch
import torch.nn as nn

from rough_heston_lift import lift_weights_geometric
from pinn import MLP

torch.set_default_dtype(torch.float64)

DEFAULT_BOX = dict(H=(0.04, 0.20), nu=(0.10, 0.60), rho=(-0.95, -0.20),
                   lam=(0.5, 4.0), theta=(0.08, 0.35))
PNAMES = ["H", "nu", "rho", "lam", "theta"]


class ParametricPINN:
    def __init__(self, box=None, n=10, V0=0.103, K=1.0, r=0.0, T=0.15,
                 width=96, depth=5, n_hgrid=48, device="cpu"):
        self.box = dict(box or DEFAULT_BOX)
        self.n = n; self.V0 = V0; self.K = K; self.r = r; self.T = T
        self.device = device
        self.net = MLP(2 + n + 5, width, depth, 1).to(device)
        # precompute the lift on an H-grid; gather per collocation point
        self.Hgrid = torch.tensor(np.linspace(*self.box["H"], n_hgrid), device=device)
        Cg, Xg = [], []
        for Hh in self.Hgrid.cpu().numpy():
            c, x = lift_weights_geometric(Hh + 0.5, n)
            Cg.append(c); Xg.append(x)
        self.Cgrid = torch.tensor(np.array(Cg), device=device)   # (n_hgrid, n)
        self.Xgrid = torch.tensor(np.array(Xg), device=device)
        self.x_lo, self.x_hi = np.log(K) - 1.2, np.log(K) + 1.2

    # ---- parameter + collocation sampling ----
    def _sample_params(self, B):
        dev = self.device
        hidx = torch.randint(0, self.Hgrid.numel(), (B,), device=dev)
        H = self.Hgrid[hidx][:, None]
        c_row = self.Cgrid[hidx]; x_row = self.Xgrid[hidx]        # (B,n)
        def U(name):
            lo, hi = self.box[name]
            return torch.rand(B, 1, device=dev) * (hi - lo) + lo
        nu, rho, lam, theta = U("nu"), U("rho"), U("lam"), U("theta")
        params = torch.cat([H, nu, rho, lam, theta], dim=1)       # (B,5)
        return params, c_row, x_row

    def _sample_u(self, B, c_row, theta):
        # centre factors so V spans ~[V0, theta]; param-aware mean toward theta
        sumc2 = (c_row * c_row).sum(dim=1, keepdim=True) + 1e-12
        u_mean = 0.5 * (theta - self.V0) * c_row / sumc2          # sum c_i u_mean_i = 0.5(theta-V0)
        eps = torch.randn(B, self.n, device=self.device)
        eps[:B // 5] *= 2.0                                       # 20% tail slice
        return u_mean + eps * 0.07

    def sample(self, n_col, tau_max):
        dev = self.device
        params, c_row, x_row = self._sample_params(n_col)
        theta = params[:, 4:5]
        tau = torch.rand(n_col, 1, device=dev) * tau_max
        x = torch.rand(n_col, 1, device=dev) * (self.x_hi - self.x_lo) + self.x_lo
        u = self._sample_u(n_col, c_row, theta)
        return tau, x, u, params, c_row, x_row

    # ---- baseline / ansatz / residual (per-point coefficients) ----
    def _tot_var(self, tau, theta, lam):
        lam = torch.clamp(lam, min=1e-6)
        return theta * tau + (self.V0 - theta) * (1.0 - torch.exp(-lam * tau)) / lam

    def _baseline(self, tau, x, theta, lam):
        S = torch.exp(x)
        tv = torch.clamp(self._tot_var(tau, theta, lam), min=1e-12)
        sq = torch.sqrt(tv)
        d1 = (torch.log(S / self.K) + self.r * tau + 0.5 * tv) / sq
        d2 = d1 - sq
        Phi = lambda z: 0.5 * (1.0 + torch.erf(z / np.sqrt(2.0)))
        return S * Phi(d1) - self.K * torch.exp(-self.r * tau) * Phi(d2)

    def price_ansatz(self, tau, x, u, params):
        theta, lam = params[:, 4:5], params[:, 3:4]
        N = self.net(torch.cat([tau, x, u, params], dim=1))
        return self._baseline(tau, x, theta, lam) + tau * N

    def pde_residual(self, tau, x, u, params, c_row, x_row):
        tau = tau.clone().requires_grad_(True)
        x = x.clone().requires_grad_(True)
        u = u.clone().requires_grad_(True)
        P = self.price_ansatz(tau, x, u, params)
        ones = torch.ones_like(P)
        P_tau = torch.autograd.grad(P, tau, ones, create_graph=True)[0]
        P_x = torch.autograd.grad(P, x, ones, create_graph=True)[0]
        grad_u = torch.autograd.grad(P, u, ones, create_graph=True)[0]
        P_xx = torch.autograd.grad(P_x, x, torch.ones_like(P_x), create_graph=True)[0]

        V = torch.clamp(self.V0 + (u * c_row).sum(dim=1, keepdim=True), min=1e-6)
        nu, rho = params[:, 1:2], params[:, 2:3]
        lam, theta = params[:, 3:4], params[:, 4:5]
        sum_grad_u = grad_u.sum(dim=1, keepdim=True)
        hess = torch.autograd.grad(sum_grad_u, u, torch.ones_like(sum_grad_u),
                                   create_graph=True)[0]
        double_sum_uu = hess.sum(dim=1, keepdim=True)
        cross = torch.autograd.grad(sum_grad_u, x, torch.ones_like(sum_grad_u),
                                    create_graph=True)[0]
        drift_u = (-x_row * u + lam * (theta - V))               # (B,n) broadcast
        drift_term = (drift_u * grad_u).sum(dim=1, keepdim=True)
        res = (P_tau - (self.r - 0.5 * V) * P_x - 0.5 * V * P_xx - drift_term
               - 0.5 * nu * nu * V * double_sum_uu - rho * nu * V * cross + self.r * P)
        return res

    def boundary_losses(self, n_bc, tau_max):
        params, c_row, x_row = self._sample_params(2 * n_bc)
        theta = params[:, 4:5]
        u = self._sample_u(2 * n_bc, c_row, theta)
        tau = torch.rand(2 * n_bc, 1, device=self.device) * tau_max
        xl = torch.full((2 * n_bc, 1), self.x_lo, device=self.device)
        xr = torch.full((2 * n_bc, 1), self.x_hi, device=self.device)
        Pl = self.price_ansatz(tau[:n_bc], xl[:n_bc], u[:n_bc], params[:n_bc])
        Pr = self.price_ansatz(tau[n_bc:], xr[n_bc:], u[n_bc:], params[n_bc:])
        tgt_r = torch.exp(xr[n_bc:]) - self.K * torch.exp(-self.r * tau[n_bc:])
        return (Pl ** 2).mean(), ((Pr - tgt_r) ** 2).mean()

    # ---- training (adaptive weights + tau-curriculum) ----
    def train(self, iters=30000, n_col=3000, n_bc=500, lr=1e-3, log_every=2000,
              tau0_frac=0.15, curriculum_frac=0.5, on_log=None):
        log_sigma = torch.zeros(3, dtype=torch.get_default_dtype(),
                                device=self.device, requires_grad=True)
        opt = torch.optim.Adam(list(self.net.parameters()) + [log_sigma], lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
        for it in range(iters):
            ramp = min(1.0, it / max(1.0, curriculum_frac * iters))
            tau_max = self.T * (tau0_frac + (1.0 - tau0_frac) * ramp)
            opt.zero_grad()
            tau, x, u, params, c_row, x_row = self.sample(n_col, tau_max)
            res = self.pde_residual(tau, x, u, params, c_row, x_row)
            l_pde = (res ** 2).mean()
            l_otm, l_itm = self.boundary_losses(n_bc, tau_max)
            terms = torch.stack([l_pde, l_otm, l_itm])
            loss = (torch.exp(-log_sigma) * terms + log_sigma).sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            opt.step(); sched.step()
            if it % log_every == 0 or it == iters - 1:
                print(f"  iter {it:6d}  PDE {float(l_pde.detach()):.3e}  "
                      f"BC[{float(l_otm.detach()):.1e},{float(l_itm.detach()):.1e}]  "
                      f"tau_max {tau_max:.3f}")
                if on_log is not None:
                    on_log(it, self)

    # ---- inference ----
    @torch.no_grad()
    def price(self, S, tau, param_vec):
        """Price a CALL (strike K=1) at spots S for one parameter vector."""
        S = np.atleast_1d(S).astype(float)
        x = torch.tensor(np.log(S)).reshape(-1, 1)
        tt = torch.full_like(x, float(tau))
        u = torch.zeros(x.shape[0], self.n)                      # start-of-life factors
        pv = torch.tensor(np.asarray(param_vec, dtype=float)).reshape(1, 5)
        params = pv.repeat(x.shape[0], 1)
        return self.price_ansatz(tt, x, u, params).numpy().ravel()

    @torch.no_grad()
    def smile(self, strikes, tau, param_vec):
        """Fixed-spot smile C(S0=1, K) via degree-1 homogeneity."""
        strikes = np.atleast_1d(strikes).astype(float)
        return strikes * self.price(1.0 / strikes, tau, param_vec)

    def price_smile_grad(self, strikes, tau, param_tensor):
        """Differentiable-in-parameters fixed-spot smile (for calibration)."""
        strikes = torch.tensor(np.atleast_1d(strikes).astype(float)).reshape(-1, 1)
        x = torch.log(1.0 / strikes)                             # spot = 1/K
        tt = torch.full_like(x, float(tau))
        u = torch.zeros(x.shape[0], self.n)
        params = param_tensor.reshape(1, 5).repeat(x.shape[0], 1)
        c = self.price_ansatz(tt, x, u, params).reshape(-1)      # C(1/K, 1)
        return strikes.reshape(-1) * c                           # C(1, K)


if __name__ == "__main__":
    import json
    cal = json.load(open("calib_real.json"))
    pv = [cal["H"], cal["nu"], cal["rho"], cal["lam"], cal["theta"]]
    m = ParametricPINN(V0=cal["V0"], T=0.15, width=64, depth=4, n=10)
    print("smoke train (parametric, n=10)...")
    m.train(iters=400, n_col=800, n_bc=200, log_every=200)
    K = np.array([0.9, 1.0, 1.1])
    print("smile @ BTC calib:", np.round(m.smile(K, 0.15, pv), 5))
