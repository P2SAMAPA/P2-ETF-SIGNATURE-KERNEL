"""
skr_engine.py — Signature Kernel Regression Engine
====================================================

Theory
------
The **signature** of a path X: [0,T] → ℝᵈ is the collection of all iterated
integrals of the path:

    Sig(X) = (1, ∫dX^i, ∫∫dX^i⊗dX^j, ∫∫∫dX^i⊗dX^j⊗dX^k, ...)

The signature is a complete invariant of the path (up to tree-like equivalence)
and lives in the tensor algebra T((ℝᵈ)).

The **signature kernel** between two paths X and Y is the inner product of
their signatures in the tensor algebra:

    k_Sig(X, Y) = ⟨Sig(X), Sig(Y)⟩ = Σₙ ⟨Sig_n(X), Sig_n(Y)⟩

Salvi et al. (2021) showed this kernel can be computed *without truncation*
via the solution to a system of PDEs (the signature kernel PDE), making it
computationally tractable and avoiding the information loss of truncated
signature features.

Key distinction from SIGNATURE-ENGINE
--------------------------------------
The existing SIGNATURE-ENGINE uses *truncated signature features* as inputs
to a linear or shallow model — it requires choosing a truncation level and
loses information above that level.

This engine uses the **signature kernel** directly (Salvi et al. 2021):
- No truncation needed — the full infinite-dimensional signature is implicitly
  used via the kernel
- Kernel Ridge Regression on top gives a non-parametric predictor
- The kernel PDE computation is O(T²) in path length, scalable for daily ETF data

Path Construction
-----------------
For each ETF over a rolling window, we construct a multi-channel path:

    X(t) = [t/T,  log_return(t),  log_price_norm(t),  |log_return(t)|]

Channels:
  - time     : [0,1] augmentation for translation invariance
  - log_ret  : daily log returns (main signal)
  - log_price: cumulative log price normalised to start at 0 (trend channel)
  - vol_proxy: |log_return| (realised vol proxy channel)

Prediction target:
  y = mean log return over next PRED_HORIZON bars (forward return)

Score = KRR prediction cross-sectionally z-scored per universe/window.

References
----------
- Salvi, C., Cass, T., Foster, J., Lyons, T. & Yang, W. (2021).
  The Signature Kernel is the solution of a Goursat PDE.
  SIAM Journal on Mathematics of Data Science, 3(3), 873–899.
- Kiraly, F.J. & Oberhauser, H. (2019). Kernels for sequentially ordered data.
  Journal of Machine Learning Research, 20(31), 1–45.
- Lyons, T. (1998). Differential equations driven by rough signals.
  Revista Matemática Iberoamericana, 14(2), 215–310.
- Chevyrev, I. & Kormilitzin, A. (2016). A primer on the signature method
  in machine learning. arXiv:1603.03788.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple

import config


# ── Path construction ─────────────────────────────────────────────────────────

def _build_path(
    log_returns: np.ndarray,
    augment_time: bool = True,
    augment_basepoint: bool = True,
) -> np.ndarray:
    """
    Build a multi-channel path array from log returns.

    Parameters
    ----------
    log_returns : 1-D array of log returns, length T

    Returns
    -------
    path : np.ndarray shape (T+1, d) if basepoint else (T, d)
           Channels: [time, log_return, log_price_norm, vol_proxy]
    """
    T = len(log_returns)
    if T < 2:
        return np.empty((0, 0))

    # Channel 1: log returns
    ret_channel = log_returns.copy()

    # Channel 2: cumulative log price normalised to start at 0
    price_channel = np.cumsum(log_returns)
    price_channel = price_channel - price_channel[0]

    # Channel 3: vol proxy = |log_return|
    vol_channel = np.abs(log_returns)

    # Stack channels: shape (T, 3)
    path = np.column_stack([ret_channel, price_channel, vol_channel])

    # Time augmentation: prepend time channel [0, 1/T, 2/T, ..., 1]
    if augment_time:
        time_channel = np.linspace(0, 1, T).reshape(-1, 1)
        path = np.hstack([time_channel, path])

    # Basepoint augmentation: prepend a zero row
    if augment_basepoint:
        bp = np.zeros((1, path.shape[1]))
        path = np.vstack([bp, path])

    return path.astype(np.float64)


# ── Signature kernel (PDE method, Salvi et al. 2021) ─────────────────────────

def _sig_kernel_pde(X: np.ndarray, Y: np.ndarray, sigma: float = 1.0) -> float:
    """
    Compute the signature kernel k_Sig(X, Y) via the Goursat PDE:

        ∂²K/∂s∂t = ⟨dX/ds, dY/dt⟩ · K(s, t)
        K(s, 0) = K(0, t) = 1

    Solved by finite differences on the grid [0..S] × [0..T].

    This is the *untruncated* signature kernel — no level truncation needed.

    Parameters
    ----------
    X : np.ndarray shape (S, d)
    Y : np.ndarray shape (T, d)
    sigma : kernel lengthscale (RBF-style scaling of increments)

    Returns
    -------
    float : k_Sig(X, Y)
    """
    S = len(X) - 1   # number of increments in X
    T = len(Y) - 1   # number of increments in Y

    if S <= 0 or T <= 0:
        return 0.0

    # Increment matrices
    dX = np.diff(X, axis=0)   # shape (S, d)
    dY = np.diff(Y, axis=0)   # shape (T, d)

    # Scale increments by sigma
    dX = dX / (sigma + 1e-8)
    dY = dY / (sigma + 1e-8)

    # Inner product matrix: M[i,j] = ⟨dX[i], dY[j]⟩, shape (S, T)
    M = dX @ dY.T

    # PDE grid: K[i,j] = k_Sig(X[0:i+1], Y[0:j+1])
    K = np.ones((S + 1, T + 1), dtype=np.float64)

    for i in range(S):
        for j in range(T):
            K[i+1, j+1] = K[i+1, j] + K[i, j+1] - K[i, j] + M[i, j] * K[i, j]

    return float(K[S, T])


def _sig_kernel_matrix(
    paths: List[np.ndarray],
    sigma: float = 1.0,
    diag_only: bool = False,
) -> np.ndarray:
    """
    Compute the Gram matrix K[i,j] = k_Sig(paths[i], paths[j]).

    Parameters
    ----------
    paths : list of path arrays, each shape (T_i, d)
    sigma : kernel lengthscale
    diag_only : if True, return only diagonal (self-similarity) — O(N·T²)

    Returns
    -------
    K : np.ndarray shape (N, N) if not diag_only else (N,)
    """
    N = len(paths)
    if diag_only:
        return np.array([_sig_kernel_pde(p, p, sigma) for p in paths])

    K = np.zeros((N, N))
    for i in range(N):
        K[i, i] = _sig_kernel_pde(paths[i], paths[i], sigma)
        for j in range(i + 1, N):
            kij = _sig_kernel_pde(paths[i], paths[j], sigma)
            K[i, j] = kij
            K[j, i] = kij
    return K


# ── Kernel Ridge Regression ───────────────────────────────────────────────────

def _krr_predict(
    K_train: np.ndarray,
    y_train: np.ndarray,
    K_test_train: np.ndarray,
    lam: float,
) -> np.ndarray:
    """
    Kernel Ridge Regression prediction.

    α = (K_train + λI)⁻¹ y_train
    ŷ_test = K_test_train @ α

    Parameters
    ----------
    K_train      : (N, N) Gram matrix on training paths
    y_train      : (N,) target values
    K_test_train : (M, N) kernel between test and training paths
    lam          : regularisation parameter

    Returns
    -------
    ŷ_test : (M,) predictions
    """
    N = K_train.shape[0]
    A = K_train + lam * np.eye(N)
    try:
        alpha = np.linalg.solve(A, y_train)
    except np.linalg.LinAlgError:
        alpha = np.linalg.lstsq(A, y_train, rcond=None)[0]
    return K_test_train @ alpha


# ── Main scoring function ─────────────────────────────────────────────────────

def compute_skr_scores(
    prices:  pd.DataFrame,
    tickers: List[str],
    window:  int,
) -> pd.Series:
    """
    Compute Signature Kernel Regression scores for all ETFs in the universe.

    Method:
      For each ETF, we use a walk-forward KRR setup over the rolling window:
        - Training paths: overlapping sub-windows of length window//2
        - Target:         forward log return over PRED_HORIZON bars
        - Test path:      the most recent window bars
      The KRR prediction on the test path is the raw score.
      Final score is cross-sectionally z-scored.

    Parameters
    ----------
    prices  : DataFrame of closing prices, DatetimeIndex
    tickers : list of ETF tickers in this universe
    window  : lookback window in trading days

    Returns
    -------
    pd.Series indexed by ticker, values = composite SKR z-score
    """
    avail = [t for t in tickers if t in prices.columns]
    if not avail:
        return pd.Series(dtype=float)

    min_rows = window + config.PRED_HORIZON + 20
    if len(prices) < min_rows:
        return pd.Series(dtype=float)

    raw_scores = {}

    for ticker in avail:
        price_series = prices[ticker].dropna()
        if len(price_series) < min_rows:
            continue

        log_ret = np.log(price_series / price_series.shift(1)).dropna().values

        if len(log_ret) < window + config.PRED_HORIZON:
            continue

        # ── Build training samples ────────────────────────────────────────────
        # Each training sample = path of length (window//2) ending at time t
        # Target = mean log return over next PRED_HORIZON bars from t
        sub_win    = max(window // 2, config.MIN_PATH_LEN)
        step       = max(sub_win // 4, 1)   # stride between samples
        train_end  = len(log_ret) - config.PRED_HORIZON - window
        train_paths = []
        train_targets = []

        t = sub_win
        while t <= train_end:
            path_ret = log_ret[t - sub_win:t]
            path     = _build_path(
                path_ret,
                augment_time      = config.AUGMENT_TIME,
                augment_basepoint = config.AUGMENT_BASEPOINT,
            )
            if len(path) < config.MIN_PATH_LEN:
                t += step
                continue

            # Target: mean forward return
            fwd_ret = log_ret[t:t + config.PRED_HORIZON].mean()
            if np.isnan(fwd_ret):
                t += step
                continue

            train_paths.append(path)
            train_targets.append(fwd_ret)
            t += step

        if len(train_paths) < 3:
            continue

        # ── Build test path (most recent window bars) ─────────────────────────
        test_ret  = log_ret[-window:]
        test_path = _build_path(
            test_ret,
            augment_time      = config.AUGMENT_TIME,
            augment_basepoint = config.AUGMENT_BASEPOINT,
        )
        if len(test_path) < config.MIN_PATH_LEN:
            continue

        # ── Compute signature kernels ─────────────────────────────────────────
        try:
            sigma   = config.SKR_SIGMA
            n_train = len(train_paths)

            K_train = _sig_kernel_matrix(train_paths, sigma=sigma)

            # k(test, train_i) for all i
            k_test_train = np.array([
                _sig_kernel_pde(test_path, train_paths[i], sigma=sigma)
                for i in range(n_train)
            ]).reshape(1, -1)

            y_train = np.array(train_targets)

            # KRR prediction
            pred = _krr_predict(
                K_train      = K_train,
                y_train      = y_train,
                K_test_train = k_test_train,
                lam          = config.SKR_LAMBDA,
            )
            raw_scores[ticker] = float(pred[0])

        except Exception:
            continue

    if not raw_scores:
        return pd.Series(dtype=float)

    scores = pd.Series(raw_scores)

    # Cross-sectional z-score
    mu  = scores.mean()
    std = scores.std()
    if std < 1e-10:
        return pd.Series(0.0, index=scores.index)

    return (scores - mu) / std
