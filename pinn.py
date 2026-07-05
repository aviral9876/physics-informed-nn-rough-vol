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
                 width=64, depth=4, device="cpu", beta_smooth=40.0,
                 x_halfwidth=1.2):
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

        # coordinate ranges for collocation sampling (log-moneyness around ATM).
        # Widened to +/-1.2 so the domain edges sit in the ASYMPTOTIC regime
        # (S ~ 0.30 K deep OTM, S ~ 3.3 K deep ITM) where the boundary
        # conditions below are essentially exact, while the PDE residual on the
        # interior connects those anchors to the traded-strike core.
        self.x_lo, self.x_hi = np.log(K) - x_halfwidth, np.log(K) + x_halfwidth
        self.u_scale = 0.05           # legacy box scale; fallback when no MC fit

        # Per-factor collocation distribution for u. By DEFAULT this is an
        # uninformative box centred at 0 (mean 0, std u_scale). Task 3: replace
        # it with the MC-empirical mean/std of each lifted factor via
        # set_factor_sampling(), so collocation lands where the factors live
        # instead of in an arbitrary symmetric box -- the fix for the flat smile.
        self.u_mean = torch.zeros(self.n, device=device)
        self.u_std = torch.full((self.n,), self.u_scale, device=device)
        self.u_tail_frac = 0.20       # fraction of u points drawn from wider std
        self.u_tail_scale = 2.0       # tail widening on the high-variance side
        self.u_tail_scale_low = None  # if set, widen the low-variance (u<mean) side

        # BS-baseline variance term structure. By DEFAULT the baseline uses the
        # SPOT variance V0 (flat). Task 4c: because the factors mean-revert from
        # V0 toward theta, the model's expected integrated variance is larger,
        # so anchoring the baseline at V0 leaves a uniform level bias. Calling
        # set_baseline_term_structure() replaces V0 with w(tau)=int_0^tau E[V_s]ds.
        self._tau_grid = None         # (G,) time nodes s
        self._w_grid = None           # (G,) cumulative E[variance] w(s)=int_0^s E[V]

    def set_factor_sampling(self, mean, std, tail_frac=0.20, tail_scale=2.0,
                            tail_scale_low=None):
        """
        Point PINN factor-collocation at the MC-empirical factor distribution.
        `mean`, `std` are per-factor arrays (length n) from
        lifted_mc.simulate_factor_stats. A std floor keeps degenerate (instantly
        mean-reverting) factors from collapsing to a delta.

        tail_scale_low (optional): widen the tail slice ASYMMETRICALLY on the
        low-variance side (u below its mean -> lower V, since V=V0+sum c_i u_i
        with c_i>0). That is the side governing the steep OTM-put skew, so
        extending it (e.g. 3x std) probes whether the put-wing error is a
        factor-tail coverage problem. Defaults to symmetric (=tail_scale).
        """
        mean = np.atleast_1d(np.asarray(mean, dtype=float))
        std = np.atleast_1d(np.asarray(std, dtype=float))
        std = np.maximum(std, 1e-8)
        self.u_mean = torch.tensor(mean, device=self.device)
        self.u_std = torch.tensor(std, device=self.device)
        self.u_tail_frac = tail_frac
        self.u_tail_scale = tail_scale
        self.u_tail_scale_low = tail_scale_low

    def _sample_u(self, m):
        """Draw m factor vectors ~ N(u_mean, u_std); a tail slice is drawn from a
        wider std to cover the tails (optionally asymmetric via tail_scale_low)."""
        eps = torch.randn(m, self.n, device=self.device)
        n_tail = int(round(self.u_tail_frac * m))
        if n_tail > 0:
            tail = eps[:n_tail]
            if self.u_tail_scale_low is not None:
                # low-variance side (eps<0 -> u below mean) widened separately
                lo = tail < 0
                tail = torch.where(lo, tail * self.u_tail_scale_low,
                                   tail * self.u_tail_scale)
            else:
                tail = tail * self.u_tail_scale
            eps = eps.clone()
            eps[:n_tail] = tail
        return self.u_mean[None, :] + eps * self.u_std[None, :]

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

    def set_baseline_term_structure(self, step_mean):
        """
        Anchor the BS baseline at the LIFTED model's expected integrated variance
        instead of the spot variance V0 (Task 4c). `step_mean` is the per-time-step
        factor-mean array (n_steps, n) from lifted_mc.simulate_factor_stats on this
        maturity. We form E[V_s] = V0 + step_mean_s . c on the MC grid (with
        E[V_0]=V0) and cumulative-trapezoid it to the total variance
        w(s) = int_0^s E[V_r] dr, so the baseline vol at any tau is
        sigma(tau) = sqrt(w(tau)/tau). Must be the LIFTED E[V] (not the classical
        Heston closed form) because the PINN solves the lifted PDE.
        """
        step_mean = np.atleast_2d(np.asarray(step_mean, dtype=float))
        c_np = self.c.detach().cpu().numpy()
        EV = self.p["V0"] + step_mean @ c_np                 # E[V] at times dt..T
        s = np.linspace(0.0, self.T, len(step_mean) + 1)     # nodes [0, dt, .., T]
        ev = np.concatenate([[self.p["V0"]], EV])            # E[V] at nodes, E[V_0]=V0
        w = np.concatenate([[0.0],
                            np.cumsum(0.5 * (ev[1:] + ev[:-1]) * np.diff(s))])
        self._tau_grid = torch.tensor(s, device=self.device)
        self._w_grid = torch.tensor(w, device=self.device)

    def _total_variance(self, tau):
        """
        Baseline total variance to maturity tau. Uses the term structure w(tau)
        (piecewise-linear interpolation, differentiable in tau so the PDE
        residual's P_tau stays correct) when set, else the flat V0*tau.
        """
        if self._w_grid is None:
            return self.p["V0"] * tau
        shape = tau.shape
        t = tau.reshape(-1)
        tg, wg = self._tau_grid, self._w_grid
        idx = torch.searchsorted(tg, t.detach(), right=True).clamp(1, tg.numel() - 1)
        t0, t1 = tg[idx - 1], tg[idx]
        w0, w1 = wg[idx - 1], wg[idx]
        frac = (t - t0) / (t1 - t0)                           # linear in tau -> grad ok
        return (w0 + frac * (w1 - w0)).reshape(shape)

    def _bs_baseline(self, tau, x):
        """
        Black-Scholes call anchored at the model's expected TOTAL variance to tau.
        By default that is V0*tau (spot variance); after set_baseline_term_structure
        it is w(tau)=int_0^tau E[V_s] ds, which accounts for the factors mean-
        reverting from V0 toward theta. As tau -> 0 the total variance -> 0 so the
        BS call converges to the exact payoff max(S-K,0), satisfying the initial
        condition WITHOUT blending; the network learns only the correction on top.
        Written in torch so it is differentiable for the PDE residual.
        """
        S = torch.exp(x)
        tot_var = torch.clamp(self._total_variance(tau), min=1e-12)
        sqrt_tv = torch.sqrt(tot_var)
        d1 = (torch.log(S/self.K) + self.r*tau + 0.5*tot_var) / sqrt_tv
        d2 = d1 - sqrt_tv
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
    def sample(self, n_col, n_bnd, tau_max=None):
        # tau_max lets the tau-curriculum restrict collocation to short
        # maturities early in training and grow the horizon over time.
        tau_max = self.T if tau_max is None else tau_max
        dev = self.device
        tau = torch.rand(n_col, 1, device=dev) * tau_max
        x = torch.rand(n_col, 1, device=dev)*(self.x_hi-self.x_lo)+self.x_lo
        u = self._sample_u(n_col)          # factors where the MC says they live
        # extra collocation concentrated near the strike & short tau (hard region)
        tau2 = torch.rand(n_bnd, 1, device=dev)**2 * tau_max*0.2
        x2 = torch.randn(n_bnd, 1, device=dev)*0.05 + np.log(self.K)
        u2 = self._sample_u(n_bnd)
        tau = torch.cat([tau, tau2]); x = torch.cat([x, x2]); u = torch.cat([u, u2])
        return tau, x, u

    # ---- asymptotic boundary sampling ----
    def sample_boundary(self, n_bc, tau_max=None):
        """
        Draw fresh Dirichlet boundary points at the two domain edges, spanning
        the full (tau, u) range so the anchor holds for every maturity/factor
        state. Targets are the exact deep-wing asymptotics of a European call:
            deep OTM  (x = x_lo, S -> 0)   :  P -> 0
            deep ITM  (x = x_hi, S -> inf) :  P -> S - K e^{-r tau}   (fwd intrinsic)
        These are the anchors the unconstrained net was missing, which let its
        tau*N correction grow without bound and blow up in the wings.
        """
        tau_max = self.T if tau_max is None else tau_max
        dev = self.device
        tau_l = torch.rand(n_bc, 1, device=dev) * tau_max
        tau_r = torch.rand(n_bc, 1, device=dev) * tau_max
        u_l = self._sample_u(n_bc)
        u_r = self._sample_u(n_bc)
        x_l = torch.full((n_bc, 1), self.x_lo, device=dev)
        x_r = torch.full((n_bc, 1), self.x_hi, device=dev)
        tgt_l = torch.zeros(n_bc, 1, device=dev)
        S_r = torch.exp(x_r)
        tgt_r = S_r - self.K * torch.exp(-self.r * tau_r)
        return (tau_l, x_l, u_l, tgt_l), (tau_r, x_r, u_r, tgt_r)

    def boundary_losses(self, n_bc, tau_max=None):
        """
        Deep-OTM and deep-ITM boundary MSEs, returned SEPARATELY so the adaptive
        weighting can balance them (the ITM target ~ S-K is O(1), the OTM target
        is ~0, so a single lumped term would be dominated by the ITM side).
        """
        (tau_l, x_l, u_l, tgt_l), (tau_r, x_r, u_r, tgt_r) = \
            self.sample_boundary(n_bc, tau_max)
        P_l = self.price_ansatz(tau_l, x_l, u_l)
        P_r = self.price_ansatz(tau_r, x_r, u_r)
        return ((P_l - tgt_l) ** 2).mean(), ((P_r - tgt_r) ** 2).mean()

    def boundary_loss(self, n_bc, tau_max=None):
        """MSE of the net's price against the deep-OTM/ITM asymptotic targets."""
        l_otm, l_itm = self.boundary_losses(n_bc, tau_max)
        return l_otm + l_itm

    def fixed_val_set(self, n=1500, seed=12345):
        """
        A DETERMINISTIC interior collocation set for held-out residual tracking.
        Evaluating the PDE residual on the same points every log step removes the
        Monte-Carlo noise of the resampled training batch, exposing the true
        (monotone-ish) convergence -- pass this to train(val_pts=...).
        """
        g = torch.Generator(device=self.device).manual_seed(seed)
        tau = torch.rand(n, 1, generator=g, device=self.device) * self.T
        x = torch.rand(n, 1, generator=g, device=self.device) \
            * (self.x_hi - self.x_lo) + self.x_lo
        # factors from the same per-factor distribution used in training
        u = self.u_mean[None, :] \
            + torch.randn(n, self.n, generator=g, device=self.device) \
            * self.u_std[None, :]
        return (tau, x, u)

    def _tau_max(self, it, iters, curriculum, tau0_frac, curriculum_frac):
        """
        tau-curriculum (Wang, Sankaran, Perdikaris 2022): start at a short
        horizon tau0_frac*T where the PDE is easiest and the terminal data is
        nearby, then ramp linearly to the full T over the first curriculum_frac
        of training. Respecting this causal ordering keeps the residual from
        fighting long-maturity error before the short end has converged.
        """
        if not curriculum:
            return self.T
        ramp = min(1.0, it / max(1.0, curriculum_frac * iters))
        return self.T * (tau0_frac + (1.0 - tau0_frac) * ramp)

    def train(self, iters=3000, n_col=2000, n_bnd=500, n_bc=400, w_bc=0.3,
              lr=1e-3, log_every=500, adaptive=True, curriculum=True,
              tau0_frac=0.15, curriculum_frac=0.5, val_pts=None, on_log=None):
        """
        Stabilised training: self-adaptive per-term loss weights + tau-curriculum.

        Adaptive weighting (learnable homoscedastic-uncertainty scheme, Kendall
        et al. 2018): each loss term L_k gets a learnable log-variance s_k and
        the objective is  sum_k [ exp(-s_k) L_k + s_k ].  The network minimises
        it while the s_k adapt to auto-balance the PDE-residual and the two
        boundary terms -- large/noisy terms are down-weighted, small ones lifted
        -- so we no longer hand-tune w_bc. Effective weight of term k is
        exp(-s_k). Falls back to loss_pde + w_bc*(l_otm+l_itm) when adaptive=False.

        Monotonicity note: the per-iteration BATCH loss is a noisy Monte-Carlo
        estimate (collocation is resampled every step), so it is jagged by
        construction. Pass a FIXED `val_pts=(tau,x,u)` set to also record a
        held-out residual curve in self.val_history -- that curve isolates true
        convergence from sampling noise and is the honest "is it monotone?" test.
        """
        # per-term learnable log-variances: [PDE, boundary-OTM, boundary-ITM]
        log_sigma = torch.zeros(3, dtype=torch.get_default_dtype(),
                                device=self.device, requires_grad=True)
        params = list(self.net.parameters())
        if adaptive:
            params = params + [log_sigma]
        opt = torch.optim.Adam(params, lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
        history = []
        self.val_history = []
        for it in range(iters):
            tau_max = self._tau_max(it, iters, curriculum, tau0_frac, curriculum_frac)
            opt.zero_grad()
            tau, x, u = self.sample(n_col, n_bnd, tau_max)
            res = self.pde_residual(tau, x, u)
            loss_pde = (res**2).mean()
            # fresh boundary points each iter, split OTM / ITM
            l_otm, l_itm = self.boundary_losses(n_bc, tau_max)

            if adaptive:
                terms = torch.stack([loss_pde, l_otm, l_itm])
                loss = (torch.exp(-log_sigma) * terms + log_sigma).sum()
            else:
                loss = loss_pde + w_bc * (l_otm + l_itm)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            opt.step(); sched.step()
            if it % log_every == 0 or it == iters-1:
                # record the UNWEIGHTED PDE residual MSE -- the physical quantity
                # whose monotone decrease is the real convergence signal.
                pde_val = float(loss_pde.detach())
                history.append((it, pde_val))
                vmsg = ""
                if val_pts is not None:
                    vres = self.pde_residual(*val_pts)
                    vval = float((vres**2).mean().detach())
                    self.val_history.append((it, vval))
                    vmsg = f"  val {vval:.3e}"
                w = torch.exp(-log_sigma).detach()
                print(f"  iter {it:5d}  PDE {pde_val:.3e}{vmsg}  "
                      f"BC[otm {float(l_otm.detach()):.2e} itm {float(l_itm.detach()):.2e}]  "
                      f"tau_max {tau_max:.2f}  "
                      f"w[{w[0]:.2f},{w[1]:.2f},{w[2]:.2f}]")
                if on_log is not None:      # optional external probe (e.g. IV RMSE)
                    on_log(it, self)
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
