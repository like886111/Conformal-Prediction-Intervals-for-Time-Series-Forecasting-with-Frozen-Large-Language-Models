"""
CP-LLM: Conformal Prediction Calibration for LLM Time Series Forecasters (Stage 3)

Provides finite-sample coverage guarantees by calibrating raw prediction
intervals from Stage 2 using nonconformity scores computed on a held-out
calibration set.

Supports three modes:
  - 'split'  : Split Conformal — symmetric intervals around the median
  - 'cqr'    : Conformalized Quantile Regression — adaptive-width intervals
  - 'aci'    : Adaptive Conformal Inference — online alpha adjustment

Theoretical guarantees (see theory.py § 5):
  Under exchangeability: P(Y ∈ Ĉ) ≥ 1 - α - 1/(n+1)
  Under β-mixing:        P(Y ∈ Ĉ) ≥ 1 - α - Δ_n  where Δ_n → 0
  ACI regret:            (1/T)Σ(I{Y_t ∉ Ĉ_t} - α) → 0  a.s.
"""

import numpy as np


class CPLLM:
    """
    CP-LLM conformal calibrator.

    Typical workflow:
        1. cpllm = CPLLM(fusion_model, method='cqr')
        2. cpllm.calibrate(cal_signals_list, cal_tests, alpha=0.05)
        3. result = cpllm.predict(signal_results, alpha=0.05)
        4. (online) cpllm.update_online(y_true, result)
    """

    def __init__(self, fusion_model, method='cqr', aci_gamma=0.005):
        """
        Args:
            fusion_model: MultiSourceUncertaintyFusion instance
            method: str, one of 'split', 'cqr', 'aci'
            aci_gamma: float, ACI learning rate for alpha adjustment
        """
        assert method in ('split', 'cqr', 'aci'), \
            f"method must be 'split', 'cqr', or 'aci', got '{method}'"
        self.fusion = fusion_model
        self.method = method
        self.aci_gamma = aci_gamma
        self.calibrated = False
        self.scores = []
        self.alpha = None
        self.alpha_t = None  # adaptive alpha for ACI
        self.q_hat = None    # split conformal quantile
        self.Q_hat = None    # CQR quantile

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(self, cal_signal_results_list, cal_tests, alpha=0.05):
        """
        Compute nonconformity scores on the calibration set.

        Args:
            cal_signal_results_list: list of signal_results dicts
                (one per calibration window)
            cal_tests: list of np.ndarray, true future values per window
            alpha: float, nominal miscoverage level

        Returns:
            self
        """
        self.alpha = alpha
        self.scores = []

        for signals, y_true in zip(cal_signal_results_list, cal_tests):
            pred = self.fusion.build_predictive_distribution(signals, alpha)
            y = np.asarray(y_true).ravel()
            horizon = min(len(y), len(pred['median']))
            y = y[:horizon]

            if self.method == 'split':
                scores = np.abs(y - pred['median'][:horizon])
            elif self.method in ('cqr', 'aci'):
                # ACI uses same CQR-style quantile-based scores as CQR (paper consistency)
                lower_q = pred['lower_raw'][:horizon]
                upper_q = pred['upper_raw'][:horizon]
                scores = np.maximum(lower_q - y, y - upper_q)

            self.scores.extend(scores.tolist())

        n = len(self.scores)
        if n == 0:
            raise ValueError("No calibration scores computed — check calibration data")

        scores_arr = np.array(self.scores)
        q_level = min(np.ceil((1 - alpha) * (n + 1)) / n, 1.0)

        if self.method == 'split':
            self.q_hat = np.quantile(scores_arr, q_level)
        elif self.method in ('cqr', 'aci'):
            self.Q_hat = np.quantile(scores_arr, q_level)
            if self.method == 'aci':
                self.alpha_t = alpha

        self.calibrated = True
        self.n_cal = n

        # Compute theoretical coverage bound under temporal dependence
        self.coverage_guarantee = self._compute_coverage_guarantee(alpha)

        q_val = self.q_hat if self.q_hat is not None else self.Q_hat
        print(f"  [CP-LLM] Calibrated ({self.method}) with {n} scores, "
              f"quantile = {q_val:.4f}")
        if self.coverage_guarantee:
            print(f"  [CP-LLM] Coverage lower bound = "
                  f"{self.coverage_guarantee['coverage_lower_bound']:.4f} "
                  f"(n_eff={self.coverage_guarantee['effective_n']:.1f})")
        return self

    def _compute_coverage_guarantee(self, alpha):
        """
        Estimate the finite-sample coverage lower bound.

        Under β-mixing with exponential decay:
            P(Y ∈ Ĉ) ≥ 1 - α - 1/(n+1) - z_{0.975}·√(α(1-α)/n_eff)
        """
        try:
            from uncertainty.theory import (
                estimate_mixing_coefficient, compute_coverage_bound
            )
            scores_arr = np.array(self.scores)
            mixing = estimate_mixing_coefficient(scores_arr)
            bound = compute_coverage_bound(len(scores_arr), alpha, mixing)
            bound['mixing_decay_rate'] = mixing['decay_rate']
            bound['efficiency_ratio'] = mixing['efficiency_ratio']
            return bound
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, signal_results, alpha=None):
        """
        Produce conformally calibrated prediction intervals.

        Args:
            signal_results: dict from extract_all_signals
            alpha: float, override nominal level (uses calibrated alpha if None)

        Returns:
            dict with all fields from fusion + conformal_lower, conformal_upper,
                 calibration_method, and optionally current_alpha (ACI)
        """
        if not self.calibrated:
            raise RuntimeError("Must call calibrate() before predict()")

        alpha = alpha or self.alpha
        pred = self.fusion.build_predictive_distribution(signal_results, alpha)
        result = {**pred}

        if self.method == 'split':
            result['conformal_lower'] = pred['median'] - self.q_hat
            result['conformal_upper'] = pred['median'] + self.q_hat
            result['calibration_method'] = 'Split Conformal'

        elif self.method == 'cqr':
            result['conformal_lower'] = pred['lower_raw'] - self.Q_hat
            result['conformal_upper'] = pred['upper_raw'] + self.Q_hat
            result['calibration_method'] = 'CQR'

        elif self.method == 'aci':
            # ACI uses same quantile-based construction as CQR; alpha_t adapts the correction
            alpha_use = self.alpha_t if self.alpha_t is not None else alpha
            n = len(self.scores)
            q_level = min(np.ceil((1 - alpha_use) * (n + 1)) / n, 1.0)
            Q_hat_current = np.quantile(np.array(self.scores), q_level)
            result['conformal_lower'] = pred['lower_raw'] - Q_hat_current
            result['conformal_upper'] = pred['upper_raw'] + Q_hat_current
            result['calibration_method'] = 'ACI'
            result['current_alpha'] = alpha_use

        if self.coverage_guarantee:
            result['coverage_guarantee'] = self.coverage_guarantee

        return result

    # ------------------------------------------------------------------
    # Online update (ACI only)
    # ------------------------------------------------------------------

    def update_online(self, y_true, prediction):
        """
        Adaptive Conformal Inference: update alpha_t based on observed
        coverage and append new nonconformity scores.

        Args:
            y_true: np.ndarray, observed true values
            prediction: dict returned by predict()
        """
        if self.method != 'aci':
            return

        y_true = np.asarray(y_true).ravel()
        lower = prediction['conformal_lower'][:len(y_true)]
        upper = prediction['conformal_upper'][:len(y_true)]
        covered = (y_true >= lower) & (y_true <= upper)
        err_rate = 1.0 - np.mean(covered)

        self.alpha_t += self.aci_gamma * (self.alpha - err_rate)
        self.alpha_t = np.clip(self.alpha_t, 0.001, 0.5)

        # CQR-style scores (same as calibration) for consistency with paper
        lower_raw = np.asarray(prediction['lower_raw'][:len(y_true)])
        upper_raw = np.asarray(prediction['upper_raw'][:len(y_true)])
        new_scores = np.maximum(lower_raw - y_true, y_true - upper_raw)
        self.scores.extend(new_scores.tolist())

        max_window = 500
        if len(self.scores) > max_window:
            self.scores = self.scores[-max_window:]


class NaiveConformal:
    """
    Baseline: apply conformal prediction directly on Signal-A samples
    without multi-source fusion, for ablation comparison (M2 / M5).
    """

    def __init__(self, method='cqr', aci_gamma=0.005):
        self.method = method
        self.aci_gamma = aci_gamma
        self.calibrated = False
        self.scores = []
        self.alpha = None
        self.alpha_t = None
        self.q_hat = None
        self.Q_hat = None

    def calibrate(self, cal_samples_list, cal_tests, alpha=0.05):
        """
        Args:
            cal_samples_list: list of np.ndarray (num_samples, horizon)
            cal_tests: list of np.ndarray (horizon,)
        """
        self.alpha = alpha
        self.scores = []

        for samples, y_true in zip(cal_samples_list, cal_tests):
            y = np.asarray(y_true).ravel()
            horizon = min(samples.shape[1], len(y))
            y = y[:horizon]
            median = np.median(samples, axis=0)[:horizon]

            if self.method == 'split':
                scores = np.abs(y - median)
            elif self.method in ('cqr', 'aci'):
                # ACI uses same CQR-style quantile-based scores as CQR (paper consistency)
                lower_q = np.percentile(samples, 100 * alpha / 2, axis=0)[:horizon]
                upper_q = np.percentile(samples, 100 * (1 - alpha / 2), axis=0)[:horizon]
                scores = np.maximum(lower_q - y, y - upper_q)

            self.scores.extend(scores.tolist())

        n = len(self.scores)
        scores_arr = np.array(self.scores)
        q_level = min(np.ceil((1 - alpha) * (n + 1)) / n, 1.0)

        if self.method == 'split':
            self.q_hat = np.quantile(scores_arr, q_level)
        elif self.method in ('cqr', 'aci'):
            self.Q_hat = np.quantile(scores_arr, q_level)
            if self.method == 'aci':
                self.alpha_t = alpha

        self.calibrated = True
        return self

    def predict(self, samples, alpha=None):
        alpha = alpha or self.alpha
        median = np.median(samples, axis=0)
        lower_raw = np.percentile(samples, 100 * alpha / 2, axis=0)
        upper_raw = np.percentile(samples, 100 * (1 - alpha / 2), axis=0)

        result = {
            'median': median,
            'mean': np.mean(samples, axis=0),
            'std': np.std(samples, axis=0),
            'lower_raw': lower_raw,
            'upper_raw': upper_raw,
            'all_samples': samples,
        }

        if self.method == 'split':
            result['conformal_lower'] = median - self.q_hat
            result['conformal_upper'] = median + self.q_hat
        elif self.method == 'cqr':
            result['conformal_lower'] = lower_raw - self.Q_hat
            result['conformal_upper'] = upper_raw + self.Q_hat
        elif self.method == 'aci':
            # ACI uses same quantile-based construction as CQR; alpha_t adapts the correction
            alpha_use = self.alpha_t if self.alpha_t is not None else alpha
            n = len(self.scores)
            q_level = min(np.ceil((1 - alpha_use) * (n + 1)) / n, 1.0)
            Q_hat_current = np.quantile(np.array(self.scores), q_level)
            result['conformal_lower'] = lower_raw - Q_hat_current
            result['conformal_upper'] = upper_raw + Q_hat_current
            result['current_alpha'] = alpha_use

        return result
