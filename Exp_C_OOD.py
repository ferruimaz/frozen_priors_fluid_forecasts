# ============================================================
# Experiment C_OOD — SVHN OOD Experiments
# CIFAR-10 DDPM (frozen) + Martingale Posterior (MP)
# Functionals: θ_CLIP(mean)
# Baseline: Nonparametric Bootstrap (NPB)
# OOD: SVHN data with CIFAR-10 thresholds (frozen policy)
# ============================================================

import time, random, json
from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np
import torch
import torchvision
from torchvision import transforms
from tqdm.auto import tqdm
from diffusers import DDPMPipeline
from transformers import CLIPModel, CLIPTokenizerFast

# --------------------
# Repro/Device
# --------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = device.type == "cuda"
if torch.backends.cudnn.is_available():
    torch.backends.cudnn.benchmark = True
if hasattr(torch, "set_float32_matmul_precision"):
    try:
        torch.set_float32_matmul_precision("high")  # speed-ups on Ampere/Hopper
    except Exception:
        pass
print(f"Device: {device} | AMP: {USE_AMP}")

# --------------------
# Models & data
# --------------------
def load_ddpm_cifar10():
    print("Loading DDPM (CIFAR-10)...")
    pipe = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32").to(device)
    pipe.unet.eval()
    for p in pipe.unet.parameters():
        p.requires_grad = False
    pipe.set_progress_bar_config(disable=True)
    return pipe

def load_clip_fast_head():
    """
    Fast CLIP head:
      - caches text features for CIFAR-10 prompts
      - uses get_image_features with a tiny preprocessor
    Returns: clip_uncertainty_fn (images[-1,1] -> tensor), class_names
    """
    print("Loading CLIP ViT-B/32 (fast head)...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    tok   = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")

    classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
    prompts = [f"a photo of a {c}" for c in classes]
    with torch.no_grad():
        txt = tok(prompts, padding=True, return_tensors="pt").to(device)
        tfeat = model.get_text_features(**txt)                # (10,d)
        tfeat = tfeat / tfeat.norm(dim=-1, keepdim=True)
        logit_scale = model.logit_scale.exp().item()

    # CLIP normalization tensors (OpenAI)
    im_mean = torch.tensor([0.48145466,0.4578275,0.40821073], device=device).view(1,3,1,1)
    im_std  = torch.tensor([0.26862954,0.26130258,0.27577711], device=device).view(1,3,1,1)

    def preprocess_clip_images(x: torch.Tensor) -> torch.Tensor:
        # x in [-1,1], resize→224, map to [0,1], then CLIP norm
        x = torch.nn.functional.interpolate(x, size=(224,224), mode="bilinear", align_corners=False)
        x = torch.clamp((x+1)/2, 0, 1)
        x = (x - im_mean) / im_std
        return x

    @torch.no_grad()
    def clip_uncertainty(images: torch.Tensor, bs: int = 64) -> torch.Tensor:
        vals = []
        for i in range(0, len(images), bs):
            xb = images[i:i+bs].to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", enabled=USE_AMP):
                xb = preprocess_clip_images(xb)
                img_feat = model.get_image_features(pixel_values=xb)   # (B,d)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                logits = logit_scale * (img_feat @ tfeat.T)            # (B,10)
                probs = logits.softmax(dim=-1)
                mp = probs.max(dim=-1).values
                mp = torch.clamp(mp, min=torch.exp(torch.tensor(-10.0, device=mp.device)))
                vals.append((-torch.log(mp)).float().detach().cpu())
        return torch.cat(vals, 0)

    return clip_uncertainty, classes

def load_datasets():
    tfm = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])  # [-1,1]
    train = torchvision.datasets.CIFAR10(root="./data", train=True,  download=True, transform=tfm)
    test  = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=tfm)
    svhn  = torchvision.datasets.SVHN(root="./data", split="test", download=True, transform=tfm)
    return train, test, svhn

# --------------------
# Generation (persistent RNG; tensor output)
# --------------------
_gen = torch.Generator(device=device.type).manual_seed(123456789)
_generation_count = 0

@torch.no_grad()
def generate_images(ddpm, n: int) -> torch.Tensor:
    """Generate n images from DDPM and return a (n,3,32,32) torch.FloatTensor on `device` in [-1,1]."""
    global _generation_count
    bs = min(64, n)
    outs = []

    for start in range(0, n, bs):
        cur = min(bs, n - start)
        with torch.amp.autocast(device_type="cuda", enabled=USE_AMP):
            x_np = None
            try:
                # Preferred path (newer diffusers): direct numpy in [0,1], (B,H,W,C)
                res = ddpm(batch_size=cur, generator=_gen, output_type="np")
                x_np = res.images
            except TypeError:
                # Older signature without output_type
                res = ddpm(batch_size=cur, generator=_gen)
                imgs = res.images
                if isinstance(imgs, torch.Tensor):
                    x = imgs  # (B,3,32,32) in [0,1]
                elif isinstance(imgs, list):
                    x_np = np.stack([np.asarray(im, dtype=np.float32) / 255.0 for im in imgs], axis=0)
                else:
                    x_np = imgs

            if x_np is not None:
                if x_np.dtype != np.float32:
                    x_np = x_np.astype(np.float32, copy=False)
                x = torch.from_numpy(x_np).permute(0, 3, 1, 2).contiguous()  # (B,3,32,32)

            x = x.to(device, non_blocking=True)
            x = x.mul_(2.0).sub_(1.0)  # [0,1] -> [-1,1]
            outs.append(x)

    _generation_count += n
    return torch.cat(outs, dim=0)

# --------------------
# Functionals (CLIP only)
# --------------------
def make_theta_clip(clip_unc_fn: Callable[[torch.Tensor], torch.Tensor]):
    return lambda images: clip_unc_fn(images)


# --------------------
# Helpers
# --------------------
@torch.no_grad()
def apply_h(images: torch.Tensor, h: Callable[[torch.Tensor], torch.Tensor], bs: int = 64) -> torch.Tensor:
    vals = []
    for i in range(0, len(images), bs):
        vals.append(h(images[i:i+bs]))
    return torch.cat(vals, 0)

@torch.no_grad()
def theta_emp(images: torch.Tensor, h: Callable, bs: int = 64) -> float:
    v = apply_h(images, h, bs)
    return float(v.mean()) if v.numel() else float("nan")

# --------------------
# Adaptive α̂ (minimax-style; per functional)
# --------------------
@torch.no_grad()
def estimate_alpha_minimax(X0: torch.Tensor, ddpm, h: Callable, c_margin: float = 1.0, model_draws: int = 192,
                           clip_bounds: Tuple[float,float]=(5.0,200.0)) -> float:
    n = len(X0)
    vals_emp = apply_h(X0, h)  # (n,)
    mu_emp  = float(vals_emp.mean())
    var_emp = float(vals_emp.var(unbiased=True)) if n > 1 else 0.0
    Xm = generate_images(ddpm, model_draws)
    mu_mod = theta_emp(Xm, h)
    delta_hat = abs(mu_mod - mu_emp) + c_margin * (np.sqrt(max(var_emp,0.0) / max(n,1)))
    if delta_hat <= 1e-8:
        alpha = clip_bounds[1]
    else:
        alpha = (var_emp) / (delta_hat**2)
        alpha = float(np.clip(alpha, clip_bounds[0], clip_bounds[1]))
    return alpha

# --------------------
# Prequential θ(P∞) and MP
# --------------------
@torch.no_grad()
def prequential_theta_limit_parallel(X0: torch.Tensor, ddpm, h_list: List[Callable], alpha: float, M: int) -> List[float]:
    n0, J = len(X0), len(h_list)
    H0 = torch.stack([apply_h(X0, h).to(device) for h in h_list], 0)  # (J,n0)
    s = H0.sum(1).clone(); c = n0
    gen_vals: List[torch.Tensor] = []
    for t in tqdm(range(M), desc="Prequential θ(P∞)", leave=False):
        lam = alpha / (n0 + t + alpha)
        if torch.rand((), device=device) < lam:
            X = generate_images(ddpm, 1)
            z = torch.stack([apply_h(X, h).to(device) for h in h_list], 0).squeeze()  # (J,)
        else:
            j = torch.randint(0, n0 + t, (1,), device=device).item()
            z = H0[:, j] if j < n0 else gen_vals[j-n0]
        gen_vals.append(z)
        s += z; c += 1
    return (s/c).detach().cpu().numpy().tolist()

@torch.no_grad()
def mp_resampling(X0: torch.Tensor, ddpm, h_list: List[Callable], M: int, B: int, alpha: float,
                  sample_batch: int = 64) -> np.ndarray:
    n0, J = len(X0), len(h_list)
    H0 = torch.stack([apply_h(X0, h).to(device) for h in h_list], 0)  # (J,n0)
    s = H0.sum(1, keepdim=True).repeat(1,B)
    c = torch.full((B,), n0, device=device, dtype=torch.long)
    gen_H = torch.empty(J, B, M, device=device, dtype=torch.float32)

    for t in tqdm(range(M), desc="MP draws", leave=False):
        lam = alpha / (n0 + t + alpha)
        use_model = (torch.rand(B, device=device) < lam)
        H_new = torch.empty(J, B, device=device, dtype=torch.float32)

        # model columns
        cols = torch.nonzero(use_model, as_tuple=False).squeeze(1)
        if cols.numel() > 0:
            k = int(cols.numel())

            def sample_and_score(bs):
                X = generate_images(ddpm, bs)
                outs = [apply_h(X, h).to(device) for h in h_list]
                return torch.stack(outs, 0)  # (J,bs)

            if k <= sample_batch:
                vals = sample_and_score(k)
            else:
                chunks, rem = [], k
                while rem > 0:
                    chunk = min(sample_batch, rem)
                    chunks.append(sample_and_score(chunk)); rem -= chunk
                vals = torch.cat(chunks, 1)
            H_new[:, cols] = vals

        # empirical columns
        ecol = torch.nonzero(~use_model, as_tuple=False).squeeze(1)
        if ecol.numel() > 0:
            if t == 0:
                idx = torch.randint(0, n0, (ecol.numel(),), device=device)
                H_new[:, ecol] = H0[:, idx]
            else:
                p_base = n0 / float(n0 + t)
                use_base = (torch.rand(ecol.numel(), device=device) < p_base)
                if use_base.any():
                    k = int(use_base.sum().item())
                    idx = torch.randint(0, n0, (k,), device=device)
                    H_new[:, ecol[use_base]] = H0[:, idx]
                if (~use_base).any():
                    rows = ecol[~use_base]
                    step_idx = torch.randint(0, t, (rows.numel(),), device=device)
                    H_new[:, rows] = gen_H[:, rows, step_idx]

        gen_H[:, :, t] = H_new
        s += H_new; c += 1

    return (s / c.float()).detach().cpu().numpy()  # (J,B)

# --------------------
# Bootstrap baseline (NPB)
# --------------------
def npb_parallel(X: torch.Tensor, h_list: List[Callable], B: int) -> List[np.ndarray]:
    n, J = len(X), len(h_list)
    outs = [[] for _ in range(J)]
    for _ in tqdm(range(B), desc="NPB", leave=False):
        idx = np.random.choice(n, size=n, replace=True)
        Xb = X[idx]
        for j, h in enumerate(h_list):
            outs[j].append(theta_emp(Xb, h))
    return [np.array(x, dtype=np.float64) for x in outs]

# --------------------
# Config
# --------------------
@dataclass
class Config:
    # Trials & sizes
    R: int = 10
    n_list: Tuple[int,...] = (5, 10, 20, 50, 100)
    # MP/Prequential/NPB effort (lean but credible)
    B_mp: int = 40
    M_prequential: int = 100
    B_boot: int = 40

    # Levels & pools
    q: float = 0.90
    calib_size: int = 500
    truth_pool: int = 1000

    # Eval batch size for all h() passes
    bs_eval: int = 64

cfg = Config()

# --------------------
# Main OOD Experiment
# --------------------
def run_ood_experiment():
    """
    Run OOD experiments: CIFAR-10 DDPM trained model, but evaluated on SVHN data.
    Uses frozen CIFAR-10 threshold policy.
    """
    print(f"Setting up SVHN OOD Experiments...")
    ddpm = load_ddpm_cifar10()
    clip_uncertainty, class_names = load_clip_fast_head()
    train, test, svhn = load_datasets()

    # Pools - calibration from CIFAR-10, but experiments on SVHN
    # Truth pool from SVHN for frequentist target
    svhn_truth_size = min(cfg.truth_pool, len(svhn))
    truth = torch.stack([svhn[i][0] for i in range(svhn_truth_size)])

    # Functionals
    h_clip = make_theta_clip(clip_uncertainty)

    h_list = [h_clip]
    names  = ["θ_CLIP(mean)"]

    # Frequentist targets θ(F*) on held-out SVHN test (OOD distribution)
    print("Frequentist targets θ(F*) on SVHN test (OOD)...")
    theta_F = [theta_emp(truth, h, bs=cfg.bs_eval) for h in h_list]
    print({k: round(v, 6) for k,v in zip(names, theta_F)})

    results = {}
    start_all = time.time()

    for n0 in cfg.n_list:
        print(f"\n===== [OOD] n0={n0} =====")
        per_name = {nm: {
            "mp_int": [], "npb_int": [],
            "pred_cov": {"mp": [], "npb": []},
            "freq_cov": {"mp": [], "npb": []},
            "emp_vals": [], "pred_targets": [],
            "t_mp": [], "t_npb": [], "t_preq": [], "alpha": []
        } for nm in names}

        for r in tqdm(range(cfg.R), desc=f"Trials n0={n0} [OOD]", leave=True):
            # draw X0 from SVHN (OOD data)
            X0 = torch.stack([svhn[i][0] for i in np.random.choice(len(svhn), size=n0, replace=False)])

            # α̂ for single functional
            alpha = estimate_alpha_minimax(X0, ddpm, h_list[0], c_margin=1.0, model_draws=192)
            for nm in names:
                per_name[nm]["alpha"].append(alpha)

            # θ(P∞) via prequential simulation (parallel)
            t0 = time.time()
            theta_P = prequential_theta_limit_parallel(X0, ddpm, h_list, alpha=alpha, M=cfg.M_prequential)
            t_preq = time.time() - t0

            # MP draws
            t0 = time.time()
            mp_draws = mp_resampling(X0, ddpm, h_list, M=cfg.M_prequential, B=cfg.B_mp, alpha=alpha)
            t_mp = time.time() - t0
            mp_int = []
            for j in range(len(h_list)):
                lo, hi = np.quantile(mp_draws[j], [(1-cfg.q)/2, 1-(1-cfg.q)/2])
                mp_int.append((float(lo), float(hi)))

            # NPB draws (purely on SVHN data)
            t0 = time.time()
            npb_draws = npb_parallel(X0, h_list, B=cfg.B_boot)
            t_npb = time.time() - t0
            npb_int = []
            for j in range(len(h_list)):
                lo, hi = np.quantile(npb_draws[j], [(1-cfg.q)/2, 1-(1-cfg.q)/2])
                npb_int.append((float(lo), float(hi)))

            # Empirical values on SVHN X0
            emp_vals = [theta_emp(X0, h, bs=cfg.bs_eval) for h in h_list]

            # Store + coverage
            for j, nm in enumerate(names):
                per_name[nm]["mp_int"].append(mp_int[j])
                per_name[nm]["npb_int"].append(npb_int[j])
                per_name[nm]["emp_vals"].append(emp_vals[j])
                per_name[nm]["pred_targets"].append(theta_P[j])

                # predictive coverage: contains θ(P∞)?
                lo, hi = mp_int[j]; per_name[nm]["pred_cov"]["mp"].append(float(lo <= theta_P[j] <= hi))
                lo, hi = npb_int[j]; per_name[nm]["pred_cov"]["npb"].append(float(lo <= theta_P[j] <= hi))
                # frequentist coverage: contains θ(F*)? (context)
                lo, hi = mp_int[j]; per_name[nm]["freq_cov"]["mp"].append(float(lo <= theta_F[j] <= hi))
                lo, hi = npb_int[j]; per_name[nm]["freq_cov"]["npb"].append(float(lo <= theta_F[j] <= hi))

                per_name[nm]["t_mp"].append(t_mp)
                per_name[nm]["t_npb"].append(t_npb)
                per_name[nm]["t_preq"].append(t_preq)

        results[n0] = per_name

    total = time.time() - start_all
    print(f"\nDone. [OOD] Total time: {int(total//60)}m {int(total%60)}s  | Generations: {_generation_count}")

    # -------- Summary --------
    print("\n=== SUMMARY (q=%.2f) — OOD (SVHN with CIFAR-10 policy) ===" % cfg.q)
    for nm in names:
        print(f"\n{nm}")
        print("n0 | MP_width | NPB_width | MP_cov(θP) | NPB_cov(θP) | MP_cov(θF) | NPB_cov(θF) | MP_t[s] | NPB_t[s] | Preq_t[s] | α̂_med")
        print("-"*120)
        for n0 in cfg.n_list:
            st = results[n0][nm]
            mp_w  = float(np.mean([hi-lo for (lo,hi) in st["mp_int"]]))
            npb_w = float(np.mean([hi-lo for (lo,hi) in st["npb_int"]]))
            mp_covP  = float(np.mean(st["pred_cov"]["mp"]))
            npb_covP = float(np.mean(st["pred_cov"]["npb"]))
            mp_covF  = float(np.mean(st["freq_cov"]["mp"]))
            npb_covF = float(np.mean(st["freq_cov"]["npb"]))
            t_mp   = float(np.mean(st["t_mp"]))
            t_npb  = float(np.mean(st["t_npb"]))
            t_preq = float(np.mean(st["t_preq"]))
            a_med  = float(np.median(st["alpha"]))
            print(f"{n0:3d} | {mp_w:8.3f} | {npb_w:9.3f} |    {mp_covP:7.3f} |     {npb_covP:7.3f} |    {mp_covF:7.3f} |     {npb_covF:7.3f} |"
                  f"  {t_mp:6.1f} |  {t_npb:6.1f} |   {t_preq:7.1f} | {a_med:6.1f}")

    # Save JSON results
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = f"expC_OOD_clean_{ts}.json"
    with open(out_path, "w") as f:
        json.dump({
            "config": cfg.__dict__,
            "names": names,
            "theta_F": dict(zip(names, theta_F)),
            "results": results,
            "device": str(device),
            "amp": USE_AMP,
            "generations": _generation_count,
            "timestamp": ts,
            "split": "OOD"
        }, f, indent=2)
    print(f"\nSaved results to {out_path}")

    return results, cfg, names, theta_F

# --------------------
# Main execution
# --------------------
if __name__ == "__main__":
    run_ood_experiment()