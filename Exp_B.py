# ============================================================
# Experiment B — GPT-2 Text Generation + Martingale Posterior
# ============================================================

import math
import time
import random
import json
from dataclasses import dataclass
from typing import Callable, Tuple, List

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
from tqdm.auto import tqdm

# --------------------
# Setup
# --------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)
torch.use_deterministic_algorithms(False)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Device: CUDA GPU - {torch.cuda.get_device_name(0)}")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.empty_cache()
else:
    device = torch.device("cpu")
    print("Device: CPU")

USE_AMP = (device.type == "cuda")
print(f"[versions] torch={torch.__version__} | AMP on CUDA: {USE_AMP}")

# --------------------
# Load GPT-2 and Wikitext-2
# --------------------
def load_gpt2_and_data():
    """Load GPT-2 model and Wikitext-2 dataset"""
    print("Loading GPT-2 model...")
    model_name = "gpt2"  # 117M parameters
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    tokenizer.pad_token = tokenizer.eos_token

    print("Loading Wikitext-2 dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    # Filter out empty/short texts
    def filter_texts(split):
        return [t.strip() for t in dataset[split]['text'] if isinstance(t, str) and len(t.strip()) > 50]

    test_texts = filter_texts('test')
    validation_texts = filter_texts('validation')

    n_params = sum(p.numel() for p in model.parameters())
    print(f"GPT-2 loaded: {n_params:,} parameters")
    print(f"Dataset loaded: {len(test_texts)} test texts, {len(validation_texts)} validation texts")

    # Conservative max length to reduce truncation bias while keeping memory moderate
    MAX_LEN = min(256, getattr(tokenizer, "model_max_length", 1024))

    return model, tokenizer, test_texts, validation_texts, MAX_LEN

# --------------------
# Tokenization
# --------------------
def tokenize_texts(texts: List[str], tokenizer, max_length=128):
    """Tokenize texts for GPT-2 (CPU tensors)"""
    return tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    )

# --------------------
# Functionals (linear)
# --------------------
@torch.inference_mode()
def nll_per_token_per_sample(model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean NLL/token per sample (linear functional)."""
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    # Autocast to fp16 on CUDA to cut memory; on CPU it stays fp32.
    with torch.cuda.amp.autocast(enabled=USE_AMP):
        outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
        # logits: [B, T, V]
        shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    shift_attention = attention_mask[..., 1:].contiguous()

    # AMP-safe: compute CE in float32 to avoid rare overflow/NaN
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    losses = loss_fct(shift_logits.float().view(-1, shift_logits.size(-1)),
                      shift_labels.view(-1))
    losses = losses.view(shift_labels.size())
    losses = losses * shift_attention
    sample_nll = losses.sum(dim=1) / shift_attention.sum(dim=1).clamp(min=1)
    return sample_nll

def make_theta1_nll(model, tokenizer, max_length=128):
    """θ₁: Mean NLL per token functional (linear)"""
    @torch.inference_mode()
    def theta1(texts: List[str]) -> torch.Tensor:
        enc = tokenize_texts(texts, tokenizer, max_length)
        return nll_per_token_per_sample(model, enc['input_ids'], enc['attention_mask']).cpu()
    return theta1


# --------------------
# Generation utilities
# --------------------
_generation_count = 0
_total_tokens_generated = 0

@torch.inference_mode()
def generate_texts(model, tokenizer, n_samples: int, prompt="The", max_length=64) -> List[str]:
    """Generate text samples from GPT-2"""
    global _generation_count, _total_tokens_generated
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    input_ids = input_ids.repeat(n_samples, 1)
    with torch.cuda.amp.autocast(enabled=USE_AMP):
        generated = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=1,
            do_sample=True,
            temperature=1.0,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )
    _generation_count += n_samples
    _total_tokens_generated += int(generated.numel())
    texts = [tokenizer.decode(seq, skip_special_tokens=True) for seq in generated]
    return texts

# --------------------
# Mini-batched evaluation helpers (avoid OOM)
# --------------------
@torch.inference_mode()
def apply_h_batched(texts: List[str], h_func, batch_size: int) -> torch.Tensor:
    outs = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i+batch_size]
        outs.append(h_func(chunk).cpu())
    return torch.cat(outs, dim=0) if len(outs) else torch.empty(0)

@torch.inference_mode()
def theta_empirical_text(texts: List[str], h_func, batch_size: int = 16) -> float:
    vals = apply_h_batched(texts, h_func, batch_size)
    return float(vals.mean().numpy()) if vals.numel() > 0 else float("nan")

# --------------------
# Data-driven α̂ (minimax-style; per functional → shared median)
# --------------------
@torch.inference_mode()
def estimate_alpha_minimax_text(
    calib_texts: List[str],
    h: Callable[[List[str]], torch.Tensor],
    model,
    tokenizer,
    c_margin: float = 1.0,
    model_draws: int = 200
) -> float:
    vals = h(calib_texts).float()
    mu_emp = float(vals.mean())
    var_emp = float(vals.var(unbiased=True)) if vals.numel() > 1 else 0.0
    # crude model mean via small generation batch
    mu_mod = float(torch.tensor(h(generate_texts(model, tokenizer, model_draws, max_length=64))).mean())
    delta_hat = abs(mu_mod - mu_emp) + c_margin * math.sqrt(max(var_emp, 0.0) / max(len(calib_texts), 1))
    if delta_hat <= 1e-8:
        alpha = 200.0
    else:
        alpha = var_emp / (delta_hat ** 2)
        alpha = float(np.clip(alpha, 5.0, 200.0))
    return alpha

# --------------------
# Prequential θ(P∞) for text functionals
# --------------------
@torch.inference_mode()
def prequential_theta_limit_text(
    texts_initial: List[str],
    model, tokenizer,
    h_func: Callable[[List[str]], torch.Tensor],
    alpha: float,
    M: int = 200,
    max_gen_len: int = 64,
    desc: str = "Prequential text"
) -> Tuple[float, int]:
    """Estimate θ(P∞) by simulating the coherent predictive rule."""
    n0 = len(texts_initial)
    H0 = h_func(texts_initial)  # (n0,) CPU
    s, c = H0.sum(), n0
    gen_vals: List[torch.Tensor] = []

    for t in tqdm(range(M), desc=desc, leave=False):
        lam_t = alpha / (n0 + t + alpha)
        if torch.rand((), device=device) < lam_t:
            # model draw
            z = h_func(generate_texts(model, tokenizer, 1, max_length=max_gen_len))[0]
        else:
            # empirical uniform over n0 + t (values known)
            j = torch.randint(0, n0 + t, (1,), device=device).item()
            z = H0[j] if j < n0 else gen_vals[j - n0]
        gen_vals.append(z)
        s += z
        c += 1
    return float((s/c).detach().cpu()), M

# --------------------
# MP resampling for text (returns J x B draws of θ)
# --------------------
@torch.inference_mode()
def mp_resampling_text(
    texts_initial: List[str],
    model, tokenizer,
    h_funcs: List[Callable[[List[str]], torch.Tensor]],
    M=200, B=64, alpha=50.0,
    max_gen_len: int = 64,
    desc: str = "MP Text"
) -> np.ndarray:
    n0, J = len(texts_initial), len(h_funcs)
    H0 = torch.stack([h(texts_initial) for h in h_funcs], dim=0).to(device)  # (J, n0)
    s = H0.sum(dim=1, keepdim=True).repeat(1, B).clone()   # (J, B)
    c = torch.full((B,), n0, device=device, dtype=torch.long)
    gen_H = torch.empty(J, B, M, device=device)

    for t in tqdm(range(M), desc=desc, leave=False):
        lam_t = alpha / (n0 + t + alpha)
        use_model = (torch.rand(B, device=device) < lam_t)
        H_new = torch.empty(J, B, device=device)

        # model branch
        model_cols = torch.nonzero(use_model, as_tuple=False).squeeze(1)
        if model_cols.numel() > 0:
            k = int(model_cols.numel())
            new_texts = generate_texts(model, tokenizer, k, max_length=max_gen_len)
            vals = torch.stack([h(new_texts) for h in h_funcs], dim=0).to(device)  # (J, k)
            H_new[:, model_cols] = vals

        # empirical branch (uniform over n0 + t)
        emp_cols = torch.nonzero(~use_model, as_tuple=False).squeeze(1)
        if emp_cols.numel() > 0:
            if t == 0:
                j = torch.randint(0, n0, (emp_cols.numel(),), device=device)
                H_new[:, emp_cols] = H0[:, j]
            else:
                j = torch.randint(0, n0 + t, (emp_cols.numel(),), device=device)
                mask_base = (j < n0)
                # base
                if mask_base.any():
                    idx = j[mask_base]
                    H_new[:, emp_cols[mask_base]] = H0[:, idx]
                # generated (pick from previous steps 0..t-1)
                if (~mask_base).any():
                    ig = j[~mask_base] - n0
                    sel = gen_H[:, emp_cols[~mask_base], :t]  # (J, k, t)
                    ig_exp = ig.view(1, -1, 1).expand(J, ig.numel(), 1)
                    H_new[:, emp_cols[~mask_base]] = sel.gather(2, ig_exp).squeeze(2)

        gen_H[:, :, t] = H_new
        s += H_new
        c += 1

    thetas = (s / c.float()).detach().cpu().numpy()  # (J, B)
    return thetas

# --------------------
# Bootstrap baseline (mini-batched inside)
# --------------------
def bootstrap_text(
    texts: List[str],
    h_func: Callable[[List[str]], torch.Tensor],
    B: int = 200,
    eval_batch_size: int = 16
) -> np.ndarray:
    n = len(texts)
    outs = []
    for _ in range(B):
        idx = np.random.choice(n, size=n, replace=True)
        resampled = [texts[i] for i in idx]
        outs.append(theta_empirical_text(resampled, h_func, eval_batch_size))
    return np.array(outs, dtype=np.float64)

# --------------------
# Configuration
# --------------------
@dataclass
class TextExpConfig:
    B_mp: int = 64
    alpha: float = 50.0           # will be replaced by data-driven α̂
    M_prequential: int = 120
    B_boot: int = 160
    R: int = 10
    n_list: Tuple[int, ...] = (5, 10, 20, 50, 100)
    max_text_length: int = 64     # will be set from tokenizer (MAX_LEN)
    eval_batch_size: int = 16
    progress: bool = True
    q_level: float = 0.90
    truth_pool_size: int = 1200

# --------------------
# Main Experiment
# --------------------
def run_experiment_B():
    print("\n" + "="*60)
    print("EXPERIMENT B: GPT-2 Text Generation + Martingale Posterior")
    print("="*60)

    # Load model and data
    model, tokenizer, test_texts, validation_texts, MAX_LEN = load_gpt2_and_data()

    print("Frozen check:",
          f"training={model.training}, any_trainable={any(p.requires_grad for p in model.parameters())}")

    # Config
    config = TextExpConfig()
    config.max_text_length = MAX_LEN

    # Create functional
    theta1_nll = make_theta1_nll(model, tokenizer, max_length=config.max_text_length)
    h_funcs = [theta1_nll]
    func_names = ["θ₁(NLL/token)"]

    # Pools
    trial_pool = validation_texts[:600]
    truth_pool = validation_texts[600:600+config.truth_pool_size] or validation_texts[:config.truth_pool_size]

    print(f"\nOverview: n_list={list(config.n_list)}, R={config.R}, "
          f"B_mp={config.B_mp}, M={config.M_prequential}, q={config.q_level:.2f}")
    print(f"eval_batch_size={config.eval_batch_size}, max_len={config.max_text_length}")

    # Data-driven α̂ on a separate calibration subset
    calib_for_alpha = test_texts[200:350]  # disjoint from trial_pool/truth_pool
    config.alpha = estimate_alpha_minimax_text(calib_for_alpha, h_funcs[0], model, tokenizer)
    print(f"α̂ for NLL functional: {config.alpha:.1f}")

    # Frequentist target θ(F*) once per functional (mini-batched)
    theta_freq = [theta_empirical_text(truth_pool, h, config.eval_batch_size) for h in h_funcs]
    print(f"θ(F*) approx: {dict(zip(func_names, [round(v,6) for v in theta_freq]))}")

    results = {}  # nested dict
    start_all = time.time()

    for n0 in config.n_list:
        print(f"\n{'='*50}\nSample size n0={n0}\n{'='*50}")
        trial_store = {
            name: {
                "mp_intervals": [],
                "boot_intervals": [],
                "emp_values": [],
                "pred_targets": [],     # θ(P∞)
                "mp_cover_pred": [],
                "boot_cover_pred": [],
                "mp_cover_freq": [],
                "boot_cover_freq": [],
                "mp_times": [],
                "boot_times": [],
                "pred_times": [],
            } for name in func_names
        }

        for r in tqdm(range(config.R), desc=f"Trials n0={n0}", leave=False):
            set_seed(1000 + r + n0)
            idx = np.random.choice(len(trial_pool), size=n0, replace=False)
            X0 = [trial_pool[i] for i in idx]

            # Empirical values (mini-batched)
            emp_vals = [theta_empirical_text(X0, h, config.eval_batch_size) for h in h_funcs]

            # Predictive target θ(P∞) via prequential simulation (per functional)
            pred_vals = []
            pred_t0 = time.time()
            for j, h in enumerate(h_funcs):
                v_pred, _ = prequential_theta_limit_text(
                    X0, model, tokenizer, h, alpha=config.alpha,
                    M=config.M_prequential, max_gen_len=config.max_text_length,
                    desc=f"Preq {func_names[j]}"
                )
                pred_vals.append(v_pred)
            pred_time = time.time() - pred_t0

            # MP draws -> intervals
            mp_t0 = time.time()
            mp_draws = mp_resampling_text(
                X0, model, tokenizer, h_funcs,
                M=config.M_prequential, B=config.B_mp, alpha=config.alpha,
                max_gen_len=config.max_text_length, desc="MP"
            )  # shape (J, B)
            mp_time = time.time() - mp_t0
            mp_intervals = []
            for j in range(len(h_funcs)):
                lo, hi = np.quantile(mp_draws[j], [(1-config.q_level)/2, 1-(1-config.q_level)/2])
                mp_intervals.append((float(lo), float(hi)))

            # Bootstrap -> intervals
            boot_t0 = time.time()
            boot_intervals = []
            for j, h in enumerate(h_funcs):
                bsmps = bootstrap_text(X0, h, B=config.B_boot, eval_batch_size=config.eval_batch_size)
                lo, hi = np.quantile(bsmps, [(1-config.q_level)/2, 1-(1-config.q_level)/2])
                boot_intervals.append((float(lo), float(hi)))
            boot_time = time.time() - boot_t0

            # Store + coverage flags
            for j, name in enumerate(func_names):
                lo_mp, hi_mp = mp_intervals[j]
                lo_bt, hi_bt = boot_intervals[j]
                thetaP = pred_vals[j]
                thetaF = theta_freq[j]

                trial_store[name]["mp_intervals"].append((lo_mp, hi_mp))
                trial_store[name]["boot_intervals"].append((lo_bt, hi_bt))
                trial_store[name]["emp_values"].append(emp_vals[j])
                trial_store[name]["pred_targets"].append(thetaP)

                trial_store[name]["mp_cover_pred"].append(float(lo_mp <= thetaP <= hi_mp))
                trial_store[name]["boot_cover_pred"].append(float(lo_bt <= thetaP <= hi_bt))
                trial_store[name]["mp_cover_freq"].append(float(lo_mp <= thetaF <= hi_mp))
                trial_store[name]["boot_cover_freq"].append(float(lo_bt <= thetaF <= hi_bt))

                trial_store[name]["mp_times"].append(mp_time)
                trial_store[name]["boot_times"].append(boot_time)
                trial_store[name]["pred_times"].append(pred_time)

        results[n0] = trial_store

    total_time = time.time() - start_all

    # ---------- Summaries ----------
    print(f"\n{'='*60}\nRESULTS SUMMARY\n{'='*60}")
    for name in func_names:
        print(f"\n{name}: (q={config.q_level:.2f})")
        print("n0 | Emp(M) | MP_Width | BT_Width | MP_cov(θP) | BT_cov(θP) | MP_cov(θF) | BT_cov(θF) | MP_t[s] | BT_t[s] | Preq_t[s]")
        print("-"*120)
        for n0 in config.n_list:
            st = results[n0][name]
            emp_mean = float(np.mean(st["emp_values"]))
            mp_w = float(np.mean([hi-lo for (lo,hi) in st["mp_intervals"]]))
            bt_w = float(np.mean([hi-lo for (lo,hi) in st["boot_intervals"]]))
            mp_cov_pred = float(np.mean(st["mp_cover_pred"]))
            bt_cov_pred = float(np.mean(st["boot_cover_pred"]))
            mp_cov_freq = float(np.mean(st["mp_cover_freq"]))
            bt_cov_freq = float(np.mean(st["boot_cover_freq"]))
            mp_t = float(np.mean(st["mp_times"]))
            bt_t = float(np.mean(st["boot_times"]))
            pq_t = float(np.mean(st["pred_times"]))
            print(f"{n0:2d} | {emp_mean:7.3f} | {mp_w:8.3f} | {bt_w:8.3f} |"
                  f"   {mp_cov_pred:7.3f} |   {bt_cov_pred:7.3f} |   {mp_cov_freq:7.3f} |   {bt_cov_freq:7.3f} |"
                  f"  {mp_t:6.2f} |  {bt_t:6.2f} |   {pq_t:6.2f}")

    print(f"\nTotal runtime: {int(total_time//3600):02d}h {int((total_time%3600)//60):02d}m {int(total_time%60):02d}s")
    print(f"Total model generations: {_generation_count}")
    print(f"Total tokens generated:  {_total_tokens_generated}")

    return {
        "results": results,
        "config": config,
        "func_names": func_names,
        "theta_freq": theta_freq
    }

# --------------------
# Main
# --------------------
if __name__ == "__main__":
    start_time = time.time()

    print(f"=== Device: {device} ===")

    print("\n" + "="*60)
    print("Starting Experiment B")
    print("="*60)

    out = run_experiment_B()
    results, config, func_names, theta_freq = out["results"], out["config"], out["func_names"], out["theta_freq"]

    # Save JSON (intervals + coverage flags)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = f"experiment_B_gpt2_results_{timestamp}.json"

    def to_basic(obj):
        if isinstance(obj, (tuple, list)):
            return [to_basic(x) for x in obj]
        if isinstance(obj, dict):
            return {str(k): to_basic(v) for k,v in obj.items()}
        if isinstance(obj, (float, int, str)):
            return obj
        return str(obj)

    json_results = {}
    for n0, store in results.items():
        json_results[str(n0)] = {}
        for name, d in store.items():
            json_results[str(n0)][name] = {
                "mp_intervals": [list(x) for x in d["mp_intervals"]],
                "boot_intervals": [list(x) for x in d["boot_intervals"]],
                "emp_values": d["emp_values"],
                "pred_targets": d["pred_targets"],
                "mp_cover_pred": d["mp_cover_pred"],
                "boot_cover_pred": d["boot_cover_pred"],
                "mp_cover_freq": d["mp_cover_freq"],
                "boot_cover_freq": d["boot_cover_freq"],
                "mp_times": d["mp_times"],
                "boot_times": d["boot_times"],
                "pred_times": d["pred_times"],
            }

    with open(results_file, 'w') as f:
        json.dump({
            "config": {
                "B_mp": config.B_mp,
                "alpha": config.alpha,
                "M_prequential": config.M_prequential,
                "B_boot": config.B_boot,
                "R": config.R,
                "n_list": list(config.n_list),
                "max_text_length": config.max_text_length,
                "eval_batch_size": config.eval_batch_size,
                "q_level": config.q_level,
                "truth_pool_size": config.truth_pool_size
            },
            "theta_freq": dict(zip(func_names, theta_freq)),
            "results": json_results,
            "device": str(device),
            "amp": USE_AMP,
            "generations": _generation_count,
            "tokens_generated": _total_tokens_generated,
            "timestamp": timestamp
        }, f, indent=2)

    total_time = time.time() - start_time
    print(f"\nEXPERIMENT B COMPLETED!")
    print(f"Total runtime: {int(total_time//3600):02d}h {int((total_time%3600)//60):02d}m {int(total_time%60):02d}s")
    print(f"Results saved to: {results_file}")

    # Convenience: NLL→Perplexity for θ₁ (empirical means)
    print("\nNLL→Perplexity (empirical means):")
    name = "θ₁(NLL/token)"
    for n0 in config.n_list:
        emp_mean_nll = float(np.mean(results[n0][name]["emp_values"]))
        print(f"  n0={n0}: mean NLL={emp_mean_nll:.6f} → ppl≈{math.exp(emp_mean_nll):.2f}")