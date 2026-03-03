# Frozen Priors, Fluid Forecasts

Code for the paper [“Frozen Priors, Fluid Forecasts”](https://openreview.net/forum?id=3FCHmUPmhe&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2026%2FConference%2FAuthors%23your-submissions), accepted at ICLR 2026.

## Setup

```bash
conda env create -f environment.yml
conda activate predictive_flows
```

## Experiments

| Script | Description |
|--------|-------------|
| `Exp_A.py` | Two-Moons (2D) — RealNVP + Martingale Posterior |
| `Exp_B.py` | GPT-2 text generation — Martingale Posterior vs. Bootstrap |
| `Exp_B_Baselines.py` | GPT-2 text generation — Bayesian Bootstrap, DWS, Jackknife |
| `Exp_C_ID.py` | CIFAR-10 (in-distribution) — DDPM + Martingale Posterior vs. Bootstrap |
| `Exp_C_ID_Baselines.py` | CIFAR-10 (in-distribution) — Bayesian Bootstrap, DWS, Jackknife |
| `Exp_C_OOD.py` | SVHN (out-of-distribution) — DDPM + Martingale Posterior vs. Bootstrap |
| `Exp_C_OOD_Baselines.py` | SVHN (out-of-distribution) — Bayesian Bootstrap, DWS, Jackknife |

Each script saves results to a timestamped JSON file.

## Requirements

Python 3.10, PyTorch ≥ 2.6. Experiments B and C require a CUDA GPU; Experiment A runs on CPU.
