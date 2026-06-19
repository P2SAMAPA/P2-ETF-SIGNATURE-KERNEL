import os

HF_TOKEN    = os.environ.get("HF_TOKEN", "")
DATA_REPO   = "P2SAMAPA/fi-etf-macro-signal-master-data"
OUTPUT_REPO = "P2SAMAPA/p2-etf-signature-kernel-results"

UNIVERSES = {
    "FI_COMMODITIES": ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"],
    "EQUITY_SECTORS": [
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI",
        "IWM", "IWD", "IWO", "XLB", "XLRE",
    ],
    "COMBINED": [
        "TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV",
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI",
        "IWM", "IWD", "IWO", "XLB", "XLRE",
    ],
}

# ── Rolling windows (trading days) ────────────────────────────────────────────
WINDOWS = [63, 126, 252, 504]

# ── Signature kernel hyperparameters (Salvi et al. 2021) ─────────────────────

# Truncation level for signature features used in the static kernel baseline.
# The full signature kernel is computed without truncation (via the PDE solver),
# but we also provide truncated signature features for cross-validation.
SIG_TRUNCATION_LEVEL = 4

# Path augmentation: prepend time channel [0, 1, ..., T] to the path
# This makes the kernel translation-invariant in time (essential for finance)
AUGMENT_TIME = True

# Basepoint augmentation: prepend a zero row so paths start from the same origin
# Removes the effect of absolute price level
AUGMENT_BASEPOINT = True

# Path channels fed into the signature kernel:
#   "log_return"  : log(price_t / price_{t-1})
#   "log_price"   : log(price_t)  (normalised to start at 0)
#   "vol_proxy"   : |log_return|  (absolute return as vol channel)
PATH_CHANNELS = ["log_return", "log_price", "vol_proxy"]

# Static signature kernel variance (RBF lengthscale equivalent)
# Controls how quickly kernel similarity decays with path distance
SKR_SIGMA = 1.0

# Kernel Ridge Regression regularisation parameter
SKR_LAMBDA = 1e-3

# Number of lags for the target variable (forward return for regression)
# Predict the mean log return over the next PRED_HORIZON bars
PRED_HORIZON = 21   # ~1 month forward return

# Minimum path length to compute a valid signature
MIN_PATH_LEN = 10

TOP_N = 3
