# ∫ P2-ETF-SIGNATURE-KERNEL-REGRESSION

**Signature Kernel Regression Engine — Salvi et al. (2021) Goursat PDE Kernel**

Part of the **P2Quant Engine Suite** · [P2SAMAPA](https://github.com/P2SAMAPA)

---

## What This Engine Does

This engine applies **Signature Kernel Regression** (Salvi et al. 2021) to ETF
return paths. Unlike the existing SIGNATURE-ENGINE (which uses *truncated*
signature features), this engine computes the *full, untruncated* signature
kernel via the solution to the Goursat PDE — capturing all iterated integrals
of the path without information loss.

### Key Distinction from SIGNATURE-ENGINE

| | SIGNATURE-ENGINE | **SKR (this engine)** |
|---|---|---|
| Method | Truncated signature features at level N | Full signature kernel (no truncation) |
| Information | Partial — loses path structure above level N | Complete — all iterated integrals |
| Computation | Direct tensor products | Goursat PDE finite difference solver |
| Model | Linear / shallow | Kernel Ridge Regression |
| Theory | Chen (1954), Lyons (1998) | Salvi et al. (2021) |

---

## Theory

### The Signature

The **signature** of a path X: [0,T] → ℝᵈ is:

```
Sig(X) = (1,  ∫dXⁱ,  ∬dXⁱ⊗dXʲ,  ∭dXⁱ⊗dXʲ⊗dXᵏ,  ...)
```

This is a complete invariant of the path — it encodes *everything* about the
path's shape, including trend, curvature, area, and all higher-order interactions.

### The Signature Kernel

The **signature kernel** between paths X and Y is:

```
k_Sig(X, Y) = ⟨Sig(X), Sig(Y)⟩_{T((ℝᵈ))}
            = Σₙ₌₀^∞  ⟨Sig_n(X), Sig_n(Y)⟩
```

This inner product in the tensor algebra implicitly uses *all* levels of the
signature simultaneously.

### Goursat PDE (Salvi et al. 2021)

The signature kernel satisfies a system of PDEs:

```
∂²K/∂s∂t (s,t) = ⟨dX/ds, dY/dt⟩ · K(s,t)

with boundary conditions:
K(s, 0) = K(0, t) = 1  for all s, t
```

Solved by finite differences on the grid [0..S] × [0..T].
Computational complexity: **O(S·T·d)** per kernel evaluation.

### Kernel Ridge Regression

Given training paths {X₁,...,Xₙ} and targets {y₁,...,yₙ}:

```
α = (K_train + λI)⁻¹ y_train
ŷ_test = Σᵢ αᵢ · k_Sig(X_test, Xᵢ)
```

Prediction = KRR forward-return forecast = raw score (cross-sectionally z-scored).

---

## Path Construction

Each ETF path is a **4-channel time series**:

| Channel | Formula | Purpose |
|---------|---------|---------|
| time | t/T ∈ [0,1] | Translation invariance in time |
| log_return | log(pₜ/pₜ₋₁) | Primary return signal |
| log_price | Σlog_ret (normalised) | Trend / cumulative path |
| vol_proxy | \|log_return\| | Realised vol proxy |

**Augmentations:**
- **Basepoint**: prepend zero row → paths all start from same origin (removes price level effect)
- **Time channel**: standard augmentation for financial paths (Kidger et al. 2019)

---

## Hyperparameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `SKR_SIGMA` | 1.0 | Kernel lengthscale (increment scaling) |
| `SKR_LAMBDA` | 1e-3 | KRR regularisation |
| `PRED_HORIZON` | 21d | Forward return prediction target |
| `AUGMENT_TIME` | True | Time channel augmentation |
| `AUGMENT_BASEPOINT` | True | Basepoint augmentation |

---

## Universes & Windows

| Universe | Tickers |
|---|---|
| FI_COMMODITIES | TLT, VCIT, LQD, HYG, VNQ, GLD, SLV |
| EQUITY_SECTORS | SPY, QQQ, XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, GDX, XME, IWF, XSD, XBI, IWM, IWD, IWO, XLB, XLRE |
| COMBINED | All of the above |

**Windows:** `63d · 126d · 252d · 504d`

---

## Repository Structure

```
P2-ETF-SIGNATURE-KERNEL-REGRESSION/
├── config.py          # Universes, kernel hyperparameters
├── data_manager.py    # HuggingFace loader → price DataFrames
├── skr_engine.py      # Core: path builder → Goursat PDE kernel → KRR
├── trainer.py         # Orchestrator: load → score → JSON → upload
├── push_results.py    # HfApi.upload_file wrapper
├── streamlit_app.py   # Two-tab Streamlit dashboard
├── us_calendar.py     # US trading calendar helper
├── requirements.txt
└── .github/
    └── workflows/
        └── daily.yml  # Scheduled run 23:30 UTC Mon–Fri
```

---

## Setup

```bash
git clone https://github.com/P2SAMAPA/P2-ETF-SIGNATURE-KERNEL-REGRESSION
cd P2-ETF-SIGNATURE-KERNEL-REGRESSION
pip install -r requirements.txt

export HF_TOKEN=hf_...
python trainer.py
streamlit run streamlit_app.py
```

**Required GitHub secret:** `HF_TOKEN`

**Required HuggingFace dataset repo:** `P2SAMAPA/p2-etf-signature-kernel-results`

---

## References

- Salvi, C., Cass, T., Foster, J., Lyons, T. & Yang, W. (2021). The Signature Kernel
  is the solution of a Goursat PDE. *SIAM Journal on Mathematics of Data Science*, 3(3), 873–899.
- Kiraly, F.J. & Oberhauser, H. (2019). Kernels for sequentially ordered data.
  *Journal of Machine Learning Research*, 20(31), 1–45.
- Lyons, T. (1998). Differential equations driven by rough signals.
  *Revista Matemática Iberoamericana*, 14(2), 215–310.
- Chevyrev, I. & Kormilitzin, A. (2016). A primer on the signature method in machine
  learning. *arXiv:1603.03788*.
- Kidger, P., Bonnier, P., Perez Arribas, I., Salvi, C. & Lyons, T. (2019).
  Deep Signature Transforms. *NeurIPS 2019*.
