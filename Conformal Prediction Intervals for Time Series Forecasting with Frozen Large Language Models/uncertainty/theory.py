"""
Statistical Theory Module for LLM Time Series Uncertainty Quantification

Implements rigorous statistical tools grounded in the following theoretical
frameworks:

  § 1  Coverage Hypothesis Testing
       - Kupiec (1995) Proportion of Failures (POF) test
       - Christoffersen (1998) Conditional Coverage test
       - Berkowitz (2001) Density Forecast Evaluation

  § 2  Forecast Comparison
       - Diebold-Mariano (1995) test with Newey-West HAC estimator

  § 3  Three-Way Uncertainty Decomposition
       - Aleatoric / Epistemic / Representational decomposition
       - Novel "representational uncertainty" from serialization perturbation

  § 4  Optimal Forecast Combination (BLUE)
       - Bates-Granger (1969) optimal weights
       - Empirical Bayes shrinkage estimator

  § 5  Coverage Guarantees under Temporal Dependence
       - β-mixing coefficient estimation from ACF
       - Finite-sample coverage correction bound

  § 6  Cost-Constrained Optimal Allocation
       - Lagrangian solution for API call budget allocation

  § 7  Asymptotic Inference for Scoring Rules
       - CRPS difference confidence intervals via block bootstrap
"""

import numpy as np
from scipy import stats as sp_stats
from scipy.special import ndtri  # Φ^{-1}


# ======================================================================
# § 1  Coverage Hypothesis Testing
# ======================================================================

def kupiec_pof_test(violations, n_total, alpha):
    """
    Kupiec (1995) Proportion of Failures likelihood ratio test.

    Tests H₀: the true violation rate equals the nominal level α,
    against H₁: the violation rate ≠ α.

        LR_uc = -2 ln[ (1-α)^{n₁} α^{n₀} / (1-α̂)^{n₁} α̂^{n₀} ]

    where n₀ = violations, n₁ = n_total - violations, α̂ = n₀/n_total.

    Args:
        violations: int, number of times y fell outside the interval
        n_total: int, total number of predictions
        alpha: float, nominal miscoverage level (e.g., 0.05)

    Returns:
        dict with 'LR_statistic', 'p_value', 'reject_H0' (at 5% level),
             'observed_alpha', 'nominal_alpha'
    """
    n0 = violations
    n1 = n_total - violations
    alpha_hat = n0 / n_total if n_total > 0 else 0

    eps = 1e-15
    alpha_c = np.clip(alpha, eps, 1 - eps)
    alpha_hat_c = np.clip(alpha_hat, eps, 1 - eps)

    log_L0 = n1 * np.log(1 - alpha_c) + n0 * np.log(alpha_c)
    log_L1 = n1 * np.log(1 - alpha_hat_c) + n0 * np.log(alpha_hat_c)
    LR = -2 * (log_L0 - log_L1)

    p_value = 1 - sp_stats.chi2.cdf(LR, df=1)

    return {
        'LR_statistic': float(LR),
        'p_value': float(p_value),
        'reject_H0': p_value < 0.05,
        'observed_alpha': float(alpha_hat),
        'nominal_alpha': float(alpha),
    }


def christoffersen_cc_test(hit_sequence, alpha):
    """
    Christoffersen (1998) Conditional Coverage test.

    Jointly tests:
      (a) Unconditional coverage: violation rate = α   (Kupiec)
      (b) Independence: violations are not serially clustered

    Models {I_t} as a first-order Markov chain and computes:
        LR_cc = LR_uc + LR_ind,   LR_cc ~ χ²(2) under H₀

    Args:
        hit_sequence: array-like of 0/1, where 1 = violation (y outside interval)
        alpha: float, nominal miscoverage level

    Returns:
        dict with 'LR_uc', 'LR_ind', 'LR_cc', 'p_value_uc', 'p_value_ind',
             'p_value_cc', transition matrix estimates
    """
    hits = np.asarray(hit_sequence, dtype=int)
    n = len(hits)

    # Transition counts
    n00 = n01 = n10 = n11 = 0
    for t in range(1, n):
        i, j = hits[t - 1], hits[t]
        if i == 0 and j == 0:
            n00 += 1
        elif i == 0 and j == 1:
            n01 += 1
        elif i == 1 and j == 0:
            n10 += 1
        else:
            n11 += 1

    eps = 1e-15

    # Transition probabilities
    pi01 = n01 / max(n00 + n01, 1)
    pi11 = n11 / max(n10 + n11, 1)
    pi01 = np.clip(pi01, eps, 1 - eps)
    pi11 = np.clip(pi11, eps, 1 - eps)

    # Unconditional probability
    n0_total = int(hits.sum())
    n1_total = n - n0_total
    pi2 = np.clip(n0_total / n, eps, 1 - eps)

    # LR_ind: independence test
    log_L_ind = (n00 * np.log(1 - pi01) + n01 * np.log(pi01) +
                 n10 * np.log(1 - pi11) + n11 * np.log(pi11))
    log_L_dep = ((n00 + n10) * np.log(1 - pi2) +
                 (n01 + n11) * np.log(pi2))
    LR_ind = -2 * (log_L_dep - log_L_ind)

    # LR_uc: Kupiec
    alpha_c = np.clip(alpha, eps, 1 - eps)
    log_L0 = n1_total * np.log(1 - alpha_c) + n0_total * np.log(alpha_c)
    log_L1 = n1_total * np.log(1 - pi2) + n0_total * np.log(pi2)
    LR_uc = -2 * (log_L0 - log_L1)

    # Joint
    LR_cc = LR_uc + max(LR_ind, 0)

    return {
        'LR_uc': float(LR_uc),
        'LR_ind': float(max(LR_ind, 0)),
        'LR_cc': float(LR_cc),
        'p_value_uc': float(1 - sp_stats.chi2.cdf(LR_uc, df=1)),
        'p_value_ind': float(1 - sp_stats.chi2.cdf(max(LR_ind, 0), df=1)),
        'p_value_cc': float(1 - sp_stats.chi2.cdf(LR_cc, df=2)),
        'transition_matrix': {
            'pi_01': float(pi01), 'pi_11': float(pi11),
            'n00': n00, 'n01': n01, 'n10': n10, 'n11': n11,
        },
    }


def berkowitz_density_test(pit_values, max_lag=1):
    """
    Berkowitz (2001) density forecast evaluation via PIT inversion.

    If the predictive distribution is correctly specified, then
    PIT values are i.i.d. U(0,1), so z_t = Φ⁻¹(PIT_t) ~ i.i.d. N(0,1).

    We fit AR(1): z_t = μ + ρ(z_{t-1} - μ) + σε_t and test
    H₀: μ=0, ρ=0, σ=1 via LR ~ χ²(3).

    This simultaneously tests calibration (μ, σ) and independence (ρ).

    Args:
        pit_values: array of PIT values in (0, 1)
        max_lag: int, AR order (default 1)

    Returns:
        dict with 'LR_statistic', 'p_value', estimated parameters
    """
    pit = np.asarray(pit_values, dtype=float)
    pit = np.clip(pit, 1e-6, 1 - 1e-6)
    z = ndtri(pit)

    n = len(z)
    if n < 10:
        return {'LR_statistic': np.nan, 'p_value': np.nan,
                'warning': 'Too few observations'}

    # --- Restricted model: H₀ (μ=0, ρ=0, σ=1) ---
    log_L0 = np.sum(sp_stats.norm.logpdf(z, loc=0, scale=1))

    # --- Unrestricted model: AR(1) MLE ---
    z_lag = z[:-1]
    z_cur = z[1:]

    mu_hat = np.mean(z)
    if n > 2:
        rho_hat = np.corrcoef(z_lag - mu_hat, z_cur - mu_hat)[0, 1]
        rho_hat = np.clip(rho_hat, -0.99, 0.99)
    else:
        rho_hat = 0.0
    residuals = z_cur - mu_hat - rho_hat * (z_lag - mu_hat)
    sigma_hat = max(np.std(residuals, ddof=0), 1e-6)

    log_L1 = sp_stats.norm.logpdf(z[0], loc=mu_hat,
                                   scale=sigma_hat / max(np.sqrt(1 - rho_hat**2), 1e-6))
    log_L1 += np.sum(sp_stats.norm.logpdf(residuals, loc=0, scale=sigma_hat))

    LR = -2 * (log_L0 - log_L1)
    LR = max(LR, 0)
    p_value = 1 - sp_stats.chi2.cdf(LR, df=3)

    return {
        'LR_statistic': float(LR),
        'p_value': float(p_value),
        'reject_H0': p_value < 0.05,
        'mu_hat': float(mu_hat),
        'rho_hat': float(rho_hat),
        'sigma_hat': float(sigma_hat),
        'interpretation': {
            'mu_nonzero': 'bias in location (miscalibrated center)',
            'sigma_neq_1': 'bias in spread (over/under-dispersed)',
            'rho_nonzero': 'serial dependence in violations',
        },
    }


# ======================================================================
# § 2  Forecast Comparison — Diebold-Mariano Test
# ======================================================================

def diebold_mariano_test(loss_1, loss_2, h=1, loss_fn='squared'):
    """
    Diebold-Mariano (1995) test for equal predictive accuracy.

    Tests H₀: E[d_t] = 0 where d_t = L(e_{1,t}) - L(e_{2,t}).

    Uses Newey-West HAC variance estimator for serial correlation.

    Args:
        loss_1: array, losses from method 1 (or raw errors if loss_fn given)
        loss_2: array, losses from method 2
        h: int, forecast horizon (for HAC truncation lag = h-1)
        loss_fn: str, 'squared' or 'absolute' or 'precomputed'

    Returns:
        dict with 'DM_statistic', 'p_value', 'method_1_better'
    """
    e1 = np.asarray(loss_1, dtype=float)
    e2 = np.asarray(loss_2, dtype=float)

    if loss_fn == 'squared':
        d = e1**2 - e2**2
    elif loss_fn == 'absolute':
        d = np.abs(e1) - np.abs(e2)
    elif loss_fn == 'precomputed':
        d = e1 - e2
    else:
        d = e1**2 - e2**2

    n = len(d)
    d_bar = np.mean(d)

    # Newey-West HAC estimator with bandwidth h-1
    gamma_0 = np.var(d, ddof=1)
    truncation = max(h - 1, 0)
    V_hat = gamma_0
    for k in range(1, truncation + 1):
        gamma_k = np.mean((d[k:] - d_bar) * (d[:-k] - d_bar))
        V_hat += 2 * (1 - k / (truncation + 1)) * gamma_k

    V_hat = max(V_hat / n, 1e-15)
    DM = d_bar / np.sqrt(V_hat)
    p_value = 2 * (1 - sp_stats.norm.cdf(abs(DM)))

    return {
        'DM_statistic': float(DM),
        'p_value': float(p_value),
        'mean_loss_diff': float(d_bar),
        'method_1_better': d_bar < 0,
        'significant_at_5pct': p_value < 0.05,
    }


# ======================================================================
# § 3  Three-Way Uncertainty Decomposition
# ======================================================================

def decompose_uncertainty(signal_results):
    """
    Decompose total predictive variance into three orthogonal components:

      Var_total ≈ Var_aleatoric + Var_epistemic + Var_representational

    Definitions:
      - Aleatoric: irreducible data noise, estimated by within-config
        sampling variance E_config[Var(Ŷ | config)]
      - Epistemic: model/temperature uncertainty, estimated by
        between-config variance Var_config[E(Ŷ | config)]
      - Representational: serialization-induced variance from Signal D,
        a novel component unique to the LLMTime paradigm

    Theoretical basis:
      By the law of total variance applied to two-level hierarchy
      (temperature × serialization):
        Var(Ŷ) = E_s[E_τ[Var(Ŷ|τ,s)]] + E_s[Var_τ(E[Ŷ|τ,s])] + Var_s(E_τ[E[Ŷ|τ,s]])

      The first term ≈ aleatoric, second ≈ epistemic, third ≈ representational.

    Args:
        signal_results: dict from extract_all_signals with keys 'A', 'B', 'D'

    Returns:
        dict with per-step arrays for each component and proportions
    """
    result = {}

    # --- Aleatoric: within-sample variance at fixed configuration ---
    if 'A' in signal_results:
        var_aleatoric = signal_results['A']['std'] ** 2
    elif 'B' in signal_results:
        intra_vars = []
        for tau, sm in signal_results['B']['temp_predictions'].items():
            intra_vars.append(np.var(sm, axis=0))
        var_aleatoric = np.mean(intra_vars, axis=0)
    else:
        var_aleatoric = None

    # --- Epistemic: between-temperature variance ---
    if 'B' in signal_results:
        var_epistemic = signal_results['B']['inter_temp_var']
    else:
        var_epistemic = None

    # --- Representational: serialization perturbation variance ---
    if 'D' in signal_results:
        var_representational = signal_results['D']['serialization_var']
    else:
        var_representational = None

    # Collect available components
    components = {}
    if var_aleatoric is not None:
        components['aleatoric'] = var_aleatoric
    if var_epistemic is not None:
        components['epistemic'] = var_epistemic
    if var_representational is not None:
        components['representational'] = var_representational

    if not components:
        raise ValueError("Need at least Signal A or B for decomposition")

    var_total = sum(components.values())
    var_total_safe = np.maximum(var_total, 1e-15)

    proportions = {k: np.mean(v / var_total_safe) for k, v in components.items()}

    return {
        'components': components,
        'var_total': var_total,
        'proportions': proportions,
        'interpretation': {
            'aleatoric': 'Irreducible data noise (within-config sampling variance)',
            'epistemic': 'Model uncertainty (between-temperature variance)',
            'representational': 'Serialization-induced variance (unique to LLMTime)',
        },
    }


# ======================================================================
# § 4  Optimal Forecast Combination (BLUE)
# ======================================================================

def blue_optimal_weights(residual_matrix):
    """
    Best Linear Unbiased Estimator (BLUE) optimal combination weights.

    Based on Bates & Granger (1969): given K forecast sources with
    error covariance matrix Σ, the minimum-variance linear combination is:

        w* = Σ⁻¹ 1 / (1ᵀ Σ⁻¹ 1)

    We estimate Σ from calibration residuals.

    Args:
        residual_matrix: np.ndarray of shape (K, n_cal_points),
            where K = number of signal sources,
            residual_matrix[k, i] = median_prediction_k[i] - y_true[i]

    Returns:
        dict with 'weights', 'covariance_matrix', 'combined_variance'
    """
    R = np.asarray(residual_matrix, dtype=float)
    K, n = R.shape

    if n < K + 1:
        # Insufficient data for full covariance; fall back to inverse-MSE
        mse = np.mean(R**2, axis=1)
        mse = np.maximum(mse, 1e-10)
        w = (1 / mse) / np.sum(1 / mse)
        return {
            'weights': w,
            'covariance_matrix': np.diag(mse),
            'combined_variance': float(1 / np.sum(1 / mse)),
            'method': 'inverse_mse_fallback',
        }

    Sigma = np.cov(R)
    if K == 1:
        return {
            'weights': np.array([1.0]),
            'covariance_matrix': Sigma.reshape(1, 1),
            'combined_variance': float(Sigma),
            'method': 'single_source',
        }

    # Regularize for numerical stability
    Sigma += np.eye(K) * 1e-8 * np.trace(Sigma)

    try:
        Sigma_inv = np.linalg.inv(Sigma)
    except np.linalg.LinAlgError:
        Sigma_inv = np.linalg.pinv(Sigma)

    ones = np.ones(K)
    w = Sigma_inv @ ones / (ones @ Sigma_inv @ ones)

    # Ensure non-negative (project onto simplex if needed)
    if np.any(w < 0):
        w = _project_simplex(w)

    combined_var = float(w @ Sigma @ w)

    return {
        'weights': w,
        'covariance_matrix': Sigma,
        'combined_variance': combined_var,
        'method': 'BLUE',
    }


def empirical_bayes_weights(mse_per_signal, prior_precision=1.0):
    """
    Empirical Bayes shrinkage estimator for combination weights.

    Shrinks inverse-MSE weights toward uniform, reducing overfitting
    when calibration data is limited.

    The posterior weights are:
        w_k ∝ (1/MSE_k + λ)  where λ = prior_precision

    As λ → ∞, weights → uniform.  As λ → 0, weights → pure inverse-MSE.

    Args:
        mse_per_signal: array of MSE for each signal
        prior_precision: float, shrinkage strength toward uniform

    Returns:
        dict with 'weights', 'effective_shrinkage'
    """
    mse = np.asarray(mse_per_signal, dtype=float)
    mse = np.maximum(mse, 1e-10)
    K = len(mse)

    precision = 1.0 / mse + prior_precision
    w = precision / precision.sum()

    uniform = np.ones(K) / K
    shrinkage = 1 - np.sum((w - uniform)**2) / max(np.sum((1/mse / np.sum(1/mse) - uniform)**2), 1e-15)

    return {
        'weights': w,
        'effective_shrinkage': float(np.clip(shrinkage, 0, 1)),
        'method': 'empirical_bayes',
    }


# ======================================================================
# § 5  Coverage Guarantees under Temporal Dependence
# ======================================================================

def estimate_mixing_coefficient(residuals, max_lag=50):
    """
    Estimate the β-mixing decay rate from the autocorrelation function.

    Under β-mixing with β(k) ≤ C·exp(-λk), the ACF also decays
    exponentially. We fit: |ρ(k)| ≈ C·exp(-λk) and extract λ.

    This is used to compute the finite-sample coverage correction.

    Args:
        residuals: array of prediction residuals
        max_lag: int, maximum lag to consider

    Returns:
        dict with 'decay_rate', 'C_constant', 'effective_sample_size',
             'acf_values'
    """
    r = np.asarray(residuals, dtype=float)
    r = r - np.mean(r)
    n = len(r)
    max_lag = min(max_lag, n // 3)

    acf = np.correlate(r, r, mode='full')[n-1:n-1+max_lag+1]
    acf = acf / (acf[0] + 1e-15)

    # Fit exponential decay to |ACF| for lags 1..max_lag
    lags = np.arange(1, max_lag + 1)
    abs_acf = np.abs(acf[1:max_lag + 1])
    positive_mask = abs_acf > 1e-6

    if positive_mask.sum() >= 3:
        log_acf = np.log(abs_acf[positive_mask])
        lags_pos = lags[positive_mask]
        # OLS fit: log|ρ(k)| = log(C) - λ*k
        A = np.column_stack([np.ones(len(lags_pos)), -lags_pos])
        try:
            params = np.linalg.lstsq(A, log_acf, rcond=None)[0]
            C_hat = np.exp(params[0])
            lambda_hat = params[1]
        except np.linalg.LinAlgError:
            C_hat, lambda_hat = 1.0, 0.1
    else:
        C_hat, lambda_hat = 1.0, 0.5

    lambda_hat = max(lambda_hat, 0.01)

    # Effective sample size under dependence
    # n_eff ≈ n / (1 + 2 Σ_k ρ(k)) ≈ n / (1 + 2C/(1-exp(-λ)))
    sum_acf = C_hat / (1 - np.exp(-lambda_hat) + 1e-10)
    n_eff = n / (1 + 2 * sum_acf)
    n_eff = max(n_eff, 1)

    return {
        'decay_rate': float(lambda_hat),
        'C_constant': float(C_hat),
        'effective_sample_size': float(n_eff),
        'nominal_sample_size': n,
        'efficiency_ratio': float(n_eff / n),
        'acf_values': acf[:min(20, max_lag + 1)].tolist(),
    }


def compute_coverage_bound(n_cal, alpha, mixing_info, confidence=0.95):
    """
    Finite-sample coverage bound under β-mixing dependence.

    For exchangeable data, split conformal gives:
        P(Y ∈ Ĉ) ≥ 1 - α - 1/(n+1)

    Under β-mixing with rate β(k) ≤ C·exp(-λk), the coverage satisfies:
        P(Y ∈ Ĉ) ≥ 1 - α - Δ_n

    where Δ_n = O(n_eff^{-1/2}) is the dependence correction.

    Based on Barber et al. (2023) and Chernozhukov et al. (2018).

    Args:
        n_cal: int, number of calibration residuals
        alpha: float, nominal miscoverage
        mixing_info: dict from estimate_mixing_coefficient
        confidence: float, confidence level for the bound

    Returns:
        dict with 'coverage_lower_bound', 'correction_term',
             'exchangeable_bound'
    """
    n_eff = mixing_info['effective_sample_size']
    z = sp_stats.norm.ppf((1 + confidence) / 2)

    # Exchangeable bound (best case)
    exchangeable_bound = 1 - alpha - 1 / (n_cal + 1)

    # Dependence correction: Δ_n ≈ z * sqrt(α(1-α)/n_eff)
    delta_n = z * np.sqrt(alpha * (1 - alpha) / max(n_eff, 1))

    coverage_lower = 1 - alpha - 1 / (n_cal + 1) - delta_n

    return {
        'coverage_lower_bound': float(max(coverage_lower, 0)),
        'correction_term': float(delta_n),
        'exchangeable_bound': float(exchangeable_bound),
        'effective_n': float(n_eff),
        'nominal_n': n_cal,
    }


# ======================================================================
# § 6  Cost-Constrained Optimal Allocation
# ======================================================================

def optimal_budget_allocation(budget, signal_variances, signal_costs=None):
    """
    Optimal allocation of API calls across signals under budget constraint.

    Model: for signal k with n_k calls, the estimation variance is
        Var_k(n_k) = σ²_k / n_k

    The combined variance under weights w_k is:
        V(n) = Σ_k w²_k σ²_k / n_k

    Minimize V(n) subject to Σ_k c_k n_k ≤ B.

    Lagrangian solution (Neyman allocation):
        n*_k ∝ w_k σ_k / √c_k

    Args:
        budget: int, total API call budget
        signal_variances: dict mapping signal label → estimated variance (scalar)
        signal_costs: dict mapping signal label → cost per call (default all 1)

    Returns:
        dict with 'allocation', 'expected_combined_variance'
    """
    labels = sorted(signal_variances.keys())
    K = len(labels)

    if signal_costs is None:
        signal_costs = {k: 1.0 for k in labels}

    sigma = np.array([np.sqrt(max(signal_variances[k], 1e-10)) for k in labels])
    costs = np.array([signal_costs.get(k, 1.0) for k in labels])

    # Neyman allocation: n_k ∝ σ_k / √c_k
    raw = sigma / np.sqrt(costs)
    total_raw = raw.sum()
    if total_raw < 1e-10:
        n_star = np.ones(K) * budget / (costs.sum())
    else:
        n_star = budget * raw / (total_raw * costs)

    # Round to integers, ensure at least 2 per signal
    n_int = np.maximum(np.round(n_star).astype(int), 2)

    # Adjust to fit budget
    while np.sum(n_int * costs) > budget and np.max(n_int) > 2:
        excess_idx = np.argmax(n_int)
        n_int[excess_idx] -= 1

    allocation = {labels[k]: int(n_int[k]) for k in range(K)}

    combined_var = sum(
        (sigma[k]**2 / max(n_int[k], 1)) for k in range(K)
    )

    return {
        'allocation': allocation,
        'total_calls': int(np.sum(n_int * costs)),
        'budget': budget,
        'expected_combined_variance': float(combined_var),
    }


# ======================================================================
# § 7  Asymptotic Inference for Scoring Rules
# ======================================================================

def crps_confidence_interval(y_true, samples_1, samples_2=None,
                             block_size=None, n_bootstrap=1000,
                             confidence=0.95):
    """
    Block bootstrap confidence interval for CRPS and CRPS differences.

    For a single method: CI for E[CRPS].
    For two methods: CI for E[CRPS_1 - CRPS_2] (tests equal CRPS).

    Block bootstrap accounts for temporal dependence.

    Args:
        y_true: array (H,)
        samples_1: array (N, H), samples from method 1
        samples_2: array (N, H) or None, samples from method 2
        block_size: int or None (auto = ceil(H^{1/3}))
        n_bootstrap: int
        confidence: float

    Returns:
        dict with 'crps_1', 'ci_1', and optionally 'crps_diff', 'ci_diff'
    """
    y = np.asarray(y_true).ravel()
    S1 = np.asarray(samples_1)
    H = min(len(y), S1.shape[1])
    y = y[:H]
    S1 = S1[:, :H]

    if block_size is None:
        block_size = max(int(np.ceil(H ** (1/3))), 1)

    def stepwise_crps(y_arr, S):
        N = S.shape[0]
        term1 = np.mean(np.abs(S - y_arr[None, :]), axis=0)
        if N > 1:
            diffs = np.abs(S[:, None, :] - S[None, :, :])
            term2 = diffs.sum(axis=(0, 1)) / (N * (N - 1))
        else:
            term2 = 0
        return term1 - 0.5 * term2

    crps_1_steps = stepwise_crps(y, S1)
    crps_1 = float(np.mean(crps_1_steps))

    # Block bootstrap
    n_blocks = max(H // block_size, 1)
    boot_crps_1 = []
    boot_crps_diff = []

    if samples_2 is not None:
        S2 = np.asarray(samples_2)[:, :H]
        crps_2_steps = stepwise_crps(y, S2)
        diff_steps = crps_1_steps - crps_2_steps
    else:
        diff_steps = None

    rng = np.random.default_rng(42)
    for _ in range(n_bootstrap):
        block_starts = rng.integers(0, H - block_size + 1, size=n_blocks)
        indices = np.concatenate([np.arange(s, s + block_size) for s in block_starts])
        indices = indices[:H]

        boot_crps_1.append(np.mean(crps_1_steps[indices]))
        if diff_steps is not None:
            boot_crps_diff.append(np.mean(diff_steps[indices]))

    boot_crps_1 = np.array(boot_crps_1)
    q_lo = (1 - confidence) / 2
    q_hi = 1 - q_lo

    result = {
        'crps_1': crps_1,
        'ci_1': (float(np.quantile(boot_crps_1, q_lo)),
                 float(np.quantile(boot_crps_1, q_hi))),
        'bootstrap_se_1': float(np.std(boot_crps_1)),
    }

    if diff_steps is not None:
        boot_diff = np.array(boot_crps_diff)
        result['crps_2'] = float(np.mean(crps_2_steps))
        result['crps_diff'] = float(np.mean(diff_steps))
        result['ci_diff'] = (float(np.quantile(boot_diff, q_lo)),
                             float(np.quantile(boot_diff, q_hi)))
        result['diff_significant'] = not (
            np.quantile(boot_diff, q_lo) <= 0 <= np.quantile(boot_diff, q_hi)
        )

    return result


# ======================================================================
# Utility
# ======================================================================

def _project_simplex(v):
    """Project vector onto the probability simplex (Duchi et al., 2008)."""
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1
    rho = np.nonzero(u > cssv / (np.arange(1, n + 1)))[0][-1]
    theta = cssv[rho] / (rho + 1.0)
    return np.maximum(v - theta, 0)


def full_theoretical_analysis(y_true, prediction, signal_results,
                              alpha=0.05, samples_baseline=None):
    """
    Run the complete suite of theoretical tests and diagnostics.

    Args:
        y_true: array, true values
        prediction: dict from CPLLM.predict()
        signal_results: dict from extract_all_signals()
        alpha: float, nominal miscoverage
        samples_baseline: optional array from a baseline method for DM test

    Returns:
        dict with all theoretical analysis results
    """
    y = np.asarray(y_true).ravel()
    lower_key = 'conformal_lower' if 'conformal_lower' in prediction else 'lower_raw'
    upper_key = 'conformal_upper' if 'conformal_upper' in prediction else 'upper_raw'
    lower = np.asarray(prediction[lower_key])
    upper = np.asarray(prediction[upper_key])
    median = np.asarray(prediction['median'])
    H = min(len(y), len(lower))
    y, lower, upper, median = y[:H], lower[:H], upper[:H], median[:H]

    hits = ((y < lower) | (y > upper)).astype(int)
    violations = int(hits.sum())

    analysis = {}

    # § 1: Coverage tests
    analysis['kupiec'] = kupiec_pof_test(violations, H, alpha)
    analysis['christoffersen'] = christoffersen_cc_test(hits, alpha)

    samples = prediction.get('all_samples', None)
    if samples is not None and samples.shape[0] >= 2:
        pit = np.array([np.mean(samples[:, h] <= y[h]) for h in range(H)])
        analysis['berkowitz'] = berkowitz_density_test(pit)

    # § 3: Uncertainty decomposition
    try:
        analysis['decomposition'] = decompose_uncertainty(signal_results)
    except ValueError:
        pass

    # § 5: Mixing and coverage bound
    residuals = y - median
    mixing = estimate_mixing_coefficient(residuals)
    analysis['mixing'] = mixing
    analysis['coverage_bound'] = compute_coverage_bound(H, alpha, mixing)

    # § 7: CRPS inference
    if samples is not None and samples.shape[0] >= 2:
        analysis['crps_inference'] = crps_confidence_interval(
            y, samples, samples_baseline
        )

    return analysis
