# ============================================================
# Two-Moons (2D) — RealNVP + Martingale Posterior (Section A)
# ============================================================

import json, math, time, random
from dataclasses import dataclass
from typing import Callable, Dict, Tuple, List

import numpy as np
import torch
import normflows as nf
from tqdm.auto import tqdm

# --------------------
# Global config & utils
# --------------------
def set_seed(seed=0):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

set_seed(3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------
# Ground-truth Two Moons
# --------------------
target = nf.distributions.TwoMoons()

# --------------------
# RealNVP (small but capable for 2D)
# --------------------
def build_realnvp_2d(num_layers=16, hidden=64):
    base = nf.distributions.base.DiagGaussian(2)
    flows = []
    for _ in range(num_layers):
        param_net = nf.nets.MLP([1, hidden, hidden, 2], init_zeros=True)
        flows.append(nf.flows.AffineCouplingBlock(param_net))
        flows.append(nf.flows.Permute(2, mode='swap'))
    model = nf.NormalizingFlow(base, flows).to(device)
    return model

def train_flow(model, target, steps=3000, batch=1024, lr=6e-4, wd=1e-5):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    model.train()
    for _ in range(steps):
        x = target.sample(batch).to(device)
        opt.zero_grad(set_to_none=True)
        loss = model.forward_kld(x)
        if torch.isfinite(loss):
            loss.backward()
            opt.step()

# --------------------
# Functionals
# --------------------
@torch.no_grad()
def nll_per_sample(model, x):
    return (-model.log_prob(x)).view(-1)  # nats

def make_theta1_indicator(tau: float) -> Callable[[torch.Tensor], torch.Tensor]:
    return lambda X: (X[:, 1] > tau).float()

def make_theta2_nll(model) -> Callable[[torch.Tensor], torch.Tensor]:
    return lambda X: nll_per_sample(model, X)

def theta_empirical(X: torch.Tensor, h: Callable[[torch.Tensor], torch.Tensor]) -> float:
    return float(h(X).mean().detach().cpu())

@torch.no_grad()
def theta_model(model, h: Callable[[torch.Tensor], torch.Tensor], K=50000, bs=5000) -> float:
    vals = []
    for i in range(0, K, bs):
        kk = min(bs, K - i)
        x = model.sample(kk)[0].to(device)
        vals.append(h(x))
    vals = torch.cat(vals)
    return float(vals.mean().detach().cpu())

# --------------------
# Beta-oracle by Monte Carlo (for θ1)
# --------------------
@torch.no_grad()
def theta1_beta_oracle_quantiles(
    X0: torch.Tensor,
    model,
    tau: float,
    alpha: float,
    qlo: float = 0.05,
    qhi: float = 0.95,
    K_model_q: int = 100_000,
    nsamp_beta: int = 200_000,
) -> Tuple[float, float]:
    device = X0.device
    n0 = X0.size(0)
    k = int((X0[:, 1] > tau).sum().item())

    # model tail prob
    done = 0; count = 0; bs = 10_000
    while done < K_model_q:
        kk = min(bs, K_model_q - done)
        xm, _ = model.sample(kk)
        count += int((xm[:, 1] > tau).sum().item())
        done += kk
    p_model = count / float(K_model_q)

    a = max(k + alpha * p_model, 1e-6)
    b = max((n0 - k) + alpha * (1.0 - p_model), 1e-6)

    beta_dist = torch.distributions.Beta(
        torch.tensor([a], device=device),
        torch.tensor([b], device=device),
    )
    draws = beta_dist.sample((nsamp_beta,)).squeeze(-1)
    q = torch.quantile(draws, torch.tensor([qlo, qhi], device=device))
    return float(q[0].cpu()), float(q[1].cpu())

# --------------------
# Predictive θ(P∞) estimation
# --------------------
@torch.no_grad()
def prequential_theta_limit_bounded(
    X0: torch.Tensor,
    model,
    h: Callable[[torch.Tensor], torch.Tensor],
    alpha: float,
    H: float,                
    delta: float = 0.10,
    eps_target: float = 5e-3,
    M_min: int = 2000,
    M_max: int = 200_000,
) -> Tuple[float, int, float]:
    n0 = X0.size(0)
    logterm = math.log(2.0 / delta)
    bound = 4*H*math.sqrt(2.0*logterm/max(n0,1)) + (4*H/(3*(n0+1))) * logterm

    M = M_min
    if bound > eps_target:
        tol = max(eps_target - bound, 1e-6)
        M_need = int((0.5/tol)**2 - n0)
        M = max(M_min, min(M_max, M_need))

    H0 = h(X0)                # (n0,)
    gen_h: List[torch.Tensor] = []
    s = H0.sum(); c = n0

    for t in range(M):
        lam_t = alpha / (n0 + t + alpha)
        if torch.rand((), device=X0.device) < lam_t:
            x, _ = model.sample(1); z = h(x.to(X0.device))[0]
        else:
            pool_size = n0 + t
            j = torch.randint(0, pool_size, (1,), device=X0.device).item()
            z = H0[j] if j < n0 else gen_h[j - n0]
        gen_h.append(z); s += z; c += 1

    return float((s/c).cpu()), M, float(bound)

@torch.no_grad()
def prequential_theta_limit_unbounded_via_stability(
    X0: torch.Tensor,
    model,
    h: Callable[[torch.Tensor], torch.Tensor],
    alpha: float,
    M0: int = 1500,
    M_min: int = 600,
    M_max: int = 20000,
    eps_abs: float = 1e-3,
    mbatch: int = 2048,
) -> Tuple[float, int]:
    device = X0.device
    n0 = X0.size(0)
    H0 = h(X0)

    def run_once(M: int) -> float:
        gen_list: List[torch.Tensor] = []
        s = H0.sum(); c = n0
        buf_h = torch.empty(0, device=device); ptr = 0
        for t in range(M):
            lam_t = alpha / (n0 + t + alpha)
            if torch.rand((), device=device) < lam_t:
                if ptr >= buf_h.numel():
                    xs, _ = model.sample(mbatch)
                    buf_h = h(xs.to(device)).flatten()
                    ptr = 0
                z = buf_h[ptr]; ptr += 1
            else:
                pool_size = n0 + t
                j = torch.randint(0, pool_size, (1,), device=device).item()
                z = H0[j] if j < n0 else gen_list[j - n0]
            gen_list.append(z); s += z; c += 1
        return float((s/c).cpu())

    M = max(M0, M_min)
    while True:
        m1 = run_once(max(M//2, 1))
        m2 = run_once(M)
        if abs(m2 - m1) <= eps_abs or M >= M_max:
            return m2, M
        M = min(int(M*1.7) + 1, M_max)

# --------------------
# MP (single functional, separate α)
# --------------------
@torch.no_grad()
def mp_resampling_linear_single(
    X0: torch.Tensor,
    model,
    h: Callable[[torch.Tensor], torch.Tensor],
    M: int = 1500,
    B: int = 512,
    alpha: float = 200.0,
    sample_batch: int = 2048,
    schedule: str = "dirichlet",   # "dirichlet" or "fixed"
    lambda_fixed: float = 0.2,
) -> np.ndarray:
    device = X0.device
    n0 = X0.size(0)

    H0 = h(X0)  # (n0,)
    s = H0.sum().repeat(B).to(device)      # (B,)
    c = torch.full((B,), n0, device=device, dtype=torch.long)
    gen_H = torch.empty(B, M, device=device)

    def sample_and_score(k: int) -> torch.Tensor:
        xs, _ = model.sample(k)
        xs = xs.to(device)
        return h(xs)  # (k,)

    for t in range(M):
        lam_t = (alpha / (n0 + t + alpha)) if schedule == "dirichlet" else float(lambda_fixed)
        use_model = (torch.rand(B, device=device) < lam_t)
        p_base = n0 / float(n0 + t) if t > 0 else 1.0
        use_base = (torch.rand(B, device=device) < p_base)

        H_new = torch.empty(B, device=device)

        # empirical-from-base
        mask_emp_base = (~use_model) & use_base
        if mask_emp_base.any():
            k = int(mask_emp_base.sum().item())
            idx = torch.randint(0, n0, (k,), device=device)
            H_new[mask_emp_base] = H0[idx]

        # empirical-from-generated
        mask_emp_gen = (~use_model) & (~use_base)
        if mask_emp_gen.any():
            rows = torch.nonzero(mask_emp_gen, as_tuple=False).squeeze(1)
            if t == 0:
                idx = torch.randint(0, n0, (rows.numel(),), device=device)
                H_new[rows] = H0[idx]
            else:
                ig = torch.randint(0, t, (rows.numel(),), device=device)
                H_new[rows] = gen_H[rows, ig]

        # model branch
        if use_model.any():
            k = int(use_model.sum().item())
            if k <= sample_batch:
                Hs = sample_and_score(k)
            else:
                rem = k; cols = []
                while rem > 0:
                    kk = min(sample_batch, rem)
                    cols.append(sample_and_score(kk))
                    rem -= kk
                Hs = torch.cat(cols, dim=0)
            H_new[use_model] = Hs

        gen_H[:, t] = H_new
        s += H_new
        c += 1

    thetas = (s / c.float())   # (B,)
    return thetas.detach().cpu().numpy()

# --------------------
# Simple M auto-selection (per functional)
# --------------------
@torch.no_grad()
def auto_select_M_for_mp_single(
    X0: torch.Tensor,
    model,
    h: Callable[[torch.Tensor], torch.Tensor],
    alpha: float,
    B_pilot: int = 256,
    M0: int = 1500,
    M_min: int = 600,
    M_max: int = 20000,
    eps_abs: float = 5e-3,      # absolute tol on 5th/95th quantiles
    eps_rel: float = 0.25,      # fraction of width at M
    schedule: str = "dirichlet",
) -> Tuple[int, str]:
    def quantiles_at(M: int) -> Tuple[float, float]:
        draws = mp_resampling_linear_single(
            X0, model, h, M=M, B=B_pilot, alpha=alpha,
            schedule=schedule
        )
        lo, hi = np.quantile(draws, [0.05, 0.95])
        return float(lo), float(hi)

    M = max(M0, M_min)
    while True:
        lo1, hi1 = quantiles_at(max(M//2, 1))
        lo2, hi2 = quantiles_at(M)
        width2 = hi2 - lo2
        drift = max(abs(lo2 - lo1), abs(hi2 - hi1))
        tol = max(eps_abs, eps_rel * max(width2, 1e-12))
        details = f"width@M={width2:.4g}, drift={drift:.4g}, tol={tol:.4g}"
        if drift <= tol or M >= M_max:
            return M, details
        M = min(int(M * 1.7) + 1, M_max)

# --------------------
# Baselines
# --------------------
def bootstrap_np(X: torch.Tensor, h: Callable, B=1000) -> np.ndarray:
    n = X.size(0)
    vals = np.empty(B, dtype=np.float64)
    with torch.no_grad():
        for b in range(B):
            idx = torch.randint(0, n, (n,), device=X.device)
            vals[b] = float(h(X[idx]).mean().detach().cpu())
    return vals

def bootstrap_parametric(model, h: Callable, n: int, B=1000) -> np.ndarray:
    vals = np.empty(B, dtype=np.float64)
    with torch.no_grad():
        for b in range(B):
            Xb = model.sample(n)[0].to(device)
            vals[b] = float(h(Xb).mean().detach().cpu())
    return vals

def binomial_wilson_interval(k: int, n: int, alpha: float = 0.10) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    z = 1.6448536269514722  # z_{1-alpha/2} for alpha=0.10
    denom = 1.0 + (z**2)/n
    center = p + (z**2)/(2*n)
    radius = z*math.sqrt(max(p*(1-p)/n + (z**2)/(4*n**2), 0.0))
    lo = (center - radius) / denom
    hi = (center + radius) / denom
    return max(0.0, lo), min(1.0, hi)

# --------------------
# Minimax pseudo-count estimator α̂ = σ̂² / Δ̂²
# --------------------
@torch.no_grad()
def estimate_alpha_minimax(
    X_cal: torch.Tensor,
    model,
    h: Callable[[torch.Tensor], torch.Tensor],
    t_coef: float = 1.64,      # ~90% one-sided
    K_model: int = 20_000,     # MC for θ(Qφ)
    clip_range: Tuple[float,float] = (5.0, 200.0)
) -> float:
    n = X_cal.size(0)
    vals = h(X_cal)
    theta_emp = float(vals.mean().cpu())
    sigma2 = float(vals.var(unbiased=True).cpu())
    theta_mod = theta_model(model, h, K=K_model)
    delta_hat = abs(theta_mod - theta_emp)
    t_n = t_coef * math.sqrt(max(sigma2, 1e-12)) / math.sqrt(max(n, 1))
    Delta_hat = max(delta_hat + t_n, 1e-6)
    alpha_hat = sigma2 / (Delta_hat ** 2)
    lo, hi = clip_range
    return float(np.clip(alpha_hat, lo, hi))

# --------------------
# Experiment runner
# --------------------
@dataclass
class ExpConfig:
    # training
    train_steps: int = 2000
    train_batch: int = 1024
    train_lr: float = 6e-4
    train_wd: float = 1e-5
    # MP defaults
    M: int = 1500
    B_mp: int = 512
    # predictive target approximation
    delta_target: float = 0.10
    eps_target_theta1: float = 5e-3
    eps_target_theta2: float = 1e-3
    M_min_target: int = 2000
    M_max_target: int = 200_000
    # bootstrap
    B_boot: int = 1000
    # trials
    R: int = 100
    n_list: Tuple[int, ...] = (50, 3000)

def run_sectionA(config: ExpConfig):
    model = build_realnvp_2d(num_layers=24, hidden=128)
    train_flow(
        model, target,
        steps=config.train_steps, batch=config.train_batch,
        lr=config.train_lr, wd=config.train_wd
    )
    model.eval()

    # τ and truths
    with torch.no_grad():
        BIG = 200_000
        y_big = target.sample(BIG).to(device)
        tau = float(torch.quantile(y_big[:,1], 0.95).cpu().numpy())
        theta1_true = float((y_big[:,1] > tau).float().mean().cpu().numpy())
        theta2_true = float(nll_per_sample(model, y_big).mean().cpu().numpy())

    # Functionals
    h1 = make_theta1_indicator(tau)
    h2 = make_theta2_nll(model)

    # Separate α via minimax
    X_cal = target.sample(200).to(device)
    alpha1 = estimate_alpha_minimax(X_cal, model, h1)
    alpha2 = estimate_alpha_minimax(X_cal, model, h2)

    # Cache θ(Qφ)
    t1_mod = theta_model(model, h1, K=100_000)
    t2_mod = theta_model(model, h2, K=100_000)

    results_pred: Dict[str, Dict[int, Dict[str, float]]] = {}
    results_freq: Dict[str, Dict[int, Dict[str, float]]] = {}

    def record(tbl: Dict, method: str, n0: int, cov, width, rmse, runtime):
        tbl.setdefault(method, {}).setdefault(n0, {})
        tbl[method][n0] = dict(coverage=cov, width=width, rmse=rmse, time=runtime)

    total_iters = sum(config.R for _ in config.n_list)
    pbar = tqdm(total=total_iters, desc="Experiments", dynamic_ncols=True, leave=True)

    for n0 in config.n_list:
        # Auto-size M separately for θ1 and θ2
        X_pilot = target.sample(n0).to(device)
        M1, _ = auto_select_M_for_mp_single(
            X_pilot, model, h1, alpha=alpha1,
            B_pilot=min(256, config.B_mp),
            M0=config.M, M_min=max(400, int(0.5*config.M)), M_max=min(config.M_max_target, 30_000),
            eps_abs=config.eps_target_theta1, eps_rel=0.25,
            schedule="dirichlet"
        )
        M2, _ = auto_select_M_for_mp_single(
            X_pilot, model, h2, alpha=alpha2,
            B_pilot=min(256, config.B_mp),
            M0=config.M, M_min=max(400, int(0.5*config.M)), M_max=min(config.M_max_target, 30_000),
            eps_abs=config.eps_target_theta2, eps_rel=0.25,
            schedule="dirichlet"
        )

        mp_in_pred_1, mp_in_pred_2 = [], []
        npb_in_pred_1, npb_in_pred_2 = [], []
        pb_in_pred_1,  pb_in_pred_2  = [], []
        mp_in_freq_1,  mp_in_freq_2  = [], []
        npb_in_freq_1, npb_in_freq_2 = [], []
        pb_in_freq_1,  pb_in_freq_2  = [], []
        wil_in_freq_1                = []

        mp_w1, mp_w2 = [], []
        npb_w1, npb_w2 = [], []
        pb_w1,  pb_w2  = [], []
        wil_w1 = []

        mp_err1, mp_err2 = [], []
        npb_err1, npb_err2 = [], []
        pb_err1, pb_err2 = [], []
        wil_err1 = []

        mp_t, npb_t, pb_t, wil_t = [], [], [], []

        for r in range(config.R):
            set_seed(10_000 + r + n0)
            X0 = target.sample(n0).to(device)

            t1_emp = theta_empirical(X0, h1); lam1 = alpha1/(n0 + alpha1)
            t2_emp = theta_empirical(X0, h2); lam2 = alpha2/(n0 + alpha2)
            t1_shrink = (1 - lam1) * t1_emp + lam1 * t1_mod
            t2_shrink = (1 - lam2) * t2_emp + lam2 * t2_mod

            # MP draws
            B_for_mp = 1024 if n0 >= 3000 else config.B_mp
            t0 = time.time()
            t1_draws = mp_resampling_linear_single(
                X0, model, h1, M=M1, B=B_for_mp, alpha=alpha1,
                sample_batch=2048, schedule="dirichlet"
            )
            mp_time1 = time.time() - t0

            t0 = time.time()
            t2_draws = mp_resampling_linear_single(
                X0, model, h2, M=M2, B=B_for_mp, alpha=alpha2,
                sample_batch=2048, schedule="dirichlet"
            )
            mp_time2 = time.time() - t0
            mp_t.append((mp_time1 + mp_time2)/2.0)

            lo1, hi1 = np.quantile(t1_draws, [0.05, 0.95])
            lo2, hi2 = np.quantile(t2_draws, [0.05, 0.95])
            mp_w1.append(hi1 - lo1); mp_w2.append(hi2 - lo2)
            mp_err1.append((t1_shrink - theta1_true)**2); mp_err2.append((t2_shrink - theta2_true)**2)

            # Predictive "truths" θ(P∞)
            t1_pinf, _, _ = prequential_theta_limit_bounded(
                X0, model, h1, alpha=alpha1,
                H=1.0, delta=config.delta_target, eps_target=config.eps_target_theta1,
                M_min=config.M_min_target, M_max=config.M_max_target
            )
            t2_pinf, _ = prequential_theta_limit_unbounded_via_stability(
                X0, model, h2, alpha=alpha2,
                M0=config.M, M_min=max(400, config.M//2), M_max=config.M_max_target,
                eps_abs=config.eps_target_theta2, mbatch=2048
            )

            mp_in_pred_1.append(lo1 <= t1_pinf <= hi1)
            mp_in_pred_2.append(lo2 <= t2_pinf <= hi2)
            mp_in_freq_1.append(lo1 <= theta1_true <= hi1)
            mp_in_freq_2.append(lo2 <= theta2_true <= hi2)

            # NPB
            t0 = time.time()
            npb1 = bootstrap_np(X0, h1, B=config.B_boot)
            npb2 = bootstrap_np(X0, h2, B=config.B_boot)
            npb_time = time.time() - t0
            npb_t.append(npb_time)
            lo1b, hi1b = np.quantile(npb1, [0.05, 0.95])
            lo2b, hi2b = np.quantile(npb2, [0.05, 0.95])
            npb_w1.append(hi1b - lo1b); npb_w2.append(hi2b - lo2b)
            npb_in_pred_1.append(lo1b <= t1_pinf <= hi1b); npb_in_pred_2.append(lo2b <= t2_pinf <= hi2b)
            npb_in_freq_1.append(lo1b <= theta1_true <= hi1b); npb_in_freq_2.append(lo2b <= theta2_true <= hi2b)
            npb_err1.append((t1_emp - theta1_true)**2); npb_err2.append((t2_emp - theta2_true)**2)

            # PB
            t0 = time.time()
            pb1 = bootstrap_parametric(model, h1, n=n0, B=config.B_boot)
            pb2 = bootstrap_parametric(model, h2, n=n0, B=config.B_boot)
            pb_time = time.time() - t0
            pb_t.append(pb_time)
            lo1p, hi1p = np.quantile(pb1, [0.05, 0.95])
            lo2p, hi2p = np.quantile(pb2, [0.05, 0.95])
            pb_w1.append(hi1p - lo1p); pb_w2.append(hi2p - lo2p)
            pb_in_pred_1.append(lo1p <= t1_pinf <= hi1p); pb_in_pred_2.append(lo2p <= t2_pinf <= hi2p)
            pb_in_freq_1.append(lo1p <= theta1_true <= hi1p); pb_in_freq_2.append(lo2p <= theta2_true <= hi2p)
            pb_err1.append((t1_emp - theta1_true)**2); pb_err2.append((t2_emp - theta2_true)**2)

            # Wilson (θ1 only)
            k1 = int((h1(X0).sum().item()))
            loW, hiW = binomial_wilson_interval(k1, n0, alpha=0.10)
            wil_in_freq_1.append(loW <= theta1_true <= hiW)
            wil_w1.append(hiW - loW)
            wil_err1.append((t1_emp - theta1_true)**2)
            wil_t.append(0.0)

            pbar.update(1)

        # Aggregate & record
        def mean_or_nan(x): return float(np.mean(x)) if len(x) else float('nan')

        # MP
        record(results_pred, "MP θ1", n0, mean_or_nan(mp_in_pred_1), mean_or_nan(mp_w1), math.sqrt(mean_or_nan(mp_err1)), mean_or_nan(mp_t))
        record(results_freq, "MP θ1", n0, mean_or_nan(mp_in_freq_1), mean_or_nan(mp_w1), math.sqrt(mean_or_nan(mp_err1)), mean_or_nan(mp_t))
        record(results_pred, "MP θ2", n0, mean_or_nan(mp_in_pred_2), mean_or_nan(mp_w2), math.sqrt(mean_or_nan(mp_err2)), mean_or_nan(mp_t))
        record(results_freq, "MP θ2", n0, mean_or_nan(mp_in_freq_2), mean_or_nan(mp_w2), math.sqrt(mean_or_nan(mp_err2)), mean_or_nan(mp_t))

        # NPB
        record(results_pred, "NPB θ1", n0, mean_or_nan(npb_in_pred_1), mean_or_nan(npb_w1), math.sqrt(mean_or_nan(npb_err1)), mean_or_nan(npb_t))
        record(results_freq, "NPB θ1", n0, mean_or_nan(npb_in_freq_1), mean_or_nan(npb_w1), math.sqrt(mean_or_nan(npb_err1)), mean_or_nan(npb_t))
        record(results_pred, "NPB θ2", n0, mean_or_nan(npb_in_pred_2), mean_or_nan(npb_w2), math.sqrt(mean_or_nan(npb_err2)), mean_or_nan(npb_t))
        record(results_freq, "NPB θ2", n0, mean_or_nan(npb_in_freq_2), mean_or_nan(npb_w2), math.sqrt(mean_or_nan(npb_err2)), mean_or_nan(npb_t))

        # PB
        record(results_pred, "PB θ1", n0, mean_or_nan(pb_in_pred_1), mean_or_nan(pb_w1), math.sqrt(mean_or_nan(pb_err1)), mean_or_nan(pb_t))
        record(results_freq, "PB θ1", n0, mean_or_nan(pb_in_freq_1), mean_or_nan(pb_w1), math.sqrt(mean_or_nan(pb_err1)), mean_or_nan(pb_t))
        record(results_pred, "PB θ2", n0, mean_or_nan(pb_in_pred_2), mean_or_nan(pb_w2), math.sqrt(mean_or_nan(pb_err2)), mean_or_nan(pb_t))
        record(results_freq, "PB θ2", n0, mean_or_nan(pb_in_freq_2), mean_or_nan(pb_w2), math.sqrt(mean_or_nan(pb_err2)), mean_or_nan(pb_t))

        # Wilson (θ1)
        record(results_freq, "Wilson(θ1)", n0, mean_or_nan(wil_in_freq_1), mean_or_nan(wil_w1), math.sqrt(mean_or_nan(wil_err1)), mean_or_nan(wil_t))

    pbar.close()

    output = {
        "config": {
            "R": config.R,
            "n_list": list(config.n_list),
            "B_boot": config.B_boot,
            "M_default": config.M,
            "B_mp": config.B_mp
        },
        "predictive": results_pred,
        "frequentist": results_freq
    }

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = f"two_moons_sectionA_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"[saved] {out_path}")

# --------------------
# Run
# --------------------
if __name__ == "__main__":
    cfg = ExpConfig(
        train_steps=2000,
        R=10,
        n_list=(5, 10, 20, 50, 100, 1000),
    )
    run_sectionA(cfg)