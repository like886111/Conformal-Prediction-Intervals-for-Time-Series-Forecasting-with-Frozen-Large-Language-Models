"""
Multi-Source Uncertainty Signal Fusion (Stage 2)

Fuses the variance contributions from multiple uncertainty signals into
a unified predictive distribution.

Supports three weight-learning strategies:
  1. Winkler-optimal (Nelder-Mead) — heuristic baseline
  2. BLUE (Bates & Granger, 1969) — minimum-variance linear unbiased combination
  3. Empirical Bayes — shrinkage toward equal weights for limited cal data

Includes the three-way uncertainty decomposition:
  Var_total = Var_aleatoric + Var_epistemic + Var_representational
"""

import numpy as np
from scipy.optimize import minimize
from scipy import stats as sp_stats


class MultiSourceUncertaintyFusion:
    """
    Fuses samples and variance estimates from Signals A/B/C/D into
    a single predictive distribution with calibrated uncertainty bands.
    """

    WEIGHT_METHODS = ('winkler', 'blue', 'empirical_bayes')

    def __init__(self, signals_to_use=('A', 'B', 'D'),
                 weight_method='blue'):
        """
        Args:
            signals_to_use: tuple, subset of ('A', 'B', 'C', 'D')
            weight_method: str, one of 'winkler', 'blue', 'empirical_bayes'
        """
        assert weight_method in self.WEIGHT_METHODS, \
            f"weight_method must be one of {self.WEIGHT_METHODS}"
        self.signals = signals_to_use
        self.weight_method = weight_method
        self.weights = None
        self.weight_info = None
        self._signal_keys = {
            'A': 'sampling',
            'B': 'temperature',
            'C': 'cross_model',
            'D': 'serialization',
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_all_samples(self, signal_results):
        """Pool all individual sample trajectories from every signal source."""
        all_samples = []

        if 'A' in self.signals and 'A' in signal_results:
            all_samples.append(signal_results['A']['samples'])

        if 'B' in self.signals and 'B' in signal_results:
            for tau, sm in signal_results['B']['temp_predictions'].items():
                all_samples.append(sm)

        if 'C' in self.signals and 'C' in signal_results:
            for model_sm in signal_results['C']['all_model_samples']:
                all_samples.append(model_sm)

        if 'D' in self.signals and 'D' in signal_results:
            for pert_sm in signal_results['D']['perturbation_samples']:
                all_samples.append(pert_sm)

        if not all_samples:
            return None
        return np.vstack(all_samples)

    def _compute_signal_variances(self, signal_results, horizon):
        """Return an ordered dict of per-step variance from each signal."""
        var_components = {}

        if 'A' in self.signals and 'A' in signal_results:
            var_components['sampling'] = signal_results['A']['std'] ** 2

        if 'B' in self.signals and 'B' in signal_results:
            var_components['temperature'] = signal_results['B']['inter_temp_var']

        if 'C' in self.signals and 'C' in signal_results:
            var_components['cross_model'] = signal_results['C']['disagreement_var']

        if 'D' in self.signals and 'D' in signal_results:
            var_components['serialization'] = signal_results['D']['serialization_var']

        return var_components

    # ------------------------------------------------------------------
    # Weight calibration on validation data
    # ------------------------------------------------------------------

    def calibrate_weights(self, val_signal_results_list, val_tests, alpha=0.05):
        """
        Learn optimal fusion weights. Dispatches to the chosen method.

        Args:
            val_signal_results_list: list of signal_results dicts
            val_tests: list of np.ndarray, true values per window
            alpha: float, nominal miscoverage level

        Returns:
            np.ndarray, fusion weights (sums to 1)
        """
        if self.weight_method == 'blue':
            return self._calibrate_blue(val_signal_results_list, val_tests)
        elif self.weight_method == 'empirical_bayes':
            return self._calibrate_empirical_bayes(val_signal_results_list, val_tests)
        else:
            return self._calibrate_winkler(val_signal_results_list, val_tests, alpha)

    def _calibrate_blue(self, val_signal_results_list, val_tests):
        """
        BLUE optimal combination (Bates & Granger, 1969).

        Computes the residual covariance matrix across signals on calibration
        data, then applies w* = Σ⁻¹1 / (1ᵀΣ⁻¹1).
        """
        from uncertainty.theory import blue_optimal_weights

        signal_labels = [s for s in self.signals
                         if any(s in sr for sr in val_signal_results_list)]

        # Build residual matrix: (K_signals, n_cal_points_total)
        residual_lists = {s: [] for s in signal_labels}
        for signals, y_true in zip(val_signal_results_list, val_tests):
            y = np.asarray(y_true).ravel()
            for s in signal_labels:
                if s not in signals:
                    continue
                if s == 'A':
                    median_s = signals['A']['median']
                elif s == 'B':
                    median_s = np.median(signals['B']['temp_medians'], axis=0)
                elif s == 'C':
                    median_s = signals['C']['ensemble_median']
                elif s == 'D':
                    median_s = np.median(signals['D']['perturbation_medians'], axis=0)
                else:
                    continue
                H = min(len(y), len(median_s))
                residual_lists[s].extend((median_s[:H] - y[:H]).tolist())

        # Align lengths
        min_len = min(len(v) for v in residual_lists.values())
        R = np.array([residual_lists[s][:min_len] for s in signal_labels])

        result = blue_optimal_weights(R)
        self.weights = result['weights']
        self.weight_info = result
        print(f"  [Fusion-BLUE] Weights: {dict(zip(signal_labels, self.weights))}, "
              f"method={result['method']}")
        return self.weights

    def _calibrate_empirical_bayes(self, val_signal_results_list, val_tests):
        """
        Empirical Bayes shrinkage toward uniform weights.
        Robust when calibration data is limited.
        """
        from uncertainty.theory import empirical_bayes_weights

        signal_labels = list(self.signals)
        mse_per_signal = []

        for s in signal_labels:
            residuals = []
            for signals, y_true in zip(val_signal_results_list, val_tests):
                if s not in signals:
                    continue
                y = np.asarray(y_true).ravel()
                if s == 'A':
                    pred = signals['A']['median']
                elif s == 'B':
                    pred = np.median(signals['B']['temp_medians'], axis=0)
                elif s == 'C':
                    pred = signals['C']['ensemble_median']
                elif s == 'D':
                    pred = np.median(signals['D']['perturbation_medians'], axis=0)
                else:
                    continue
                H = min(len(y), len(pred))
                residuals.extend((pred[:H] - y[:H]).tolist())
            mse = np.mean(np.array(residuals)**2) if residuals else 1e10
            mse_per_signal.append(mse)

        result = empirical_bayes_weights(np.array(mse_per_signal))
        self.weights = result['weights']
        self.weight_info = result
        print(f"  [Fusion-EB] Weights: {dict(zip(signal_labels, self.weights))}, "
              f"shrinkage={result['effective_shrinkage']:.3f}")
        return self.weights

    def _calibrate_winkler(self, val_signal_results_list, val_tests, alpha):
        """Winkler-score minimization (Nelder-Mead heuristic baseline)."""
        n_signals = len(self.signals)
        z = sp_stats.norm.ppf(1 - alpha / 2)

        def objective(log_weights):
            weights = np.exp(log_weights) / np.sum(np.exp(log_weights))
            total_winkler = 0.0
            for signals, y_true in zip(val_signal_results_list, val_tests):
                var_components = self._compute_signal_variances(signals, len(y_true))
                all_samples = self._collect_all_samples(signals)
                if all_samples is None:
                    continue
                center = np.median(all_samples, axis=0)[:len(y_true)]
                total_var = np.zeros(len(y_true))
                for i, (key, var) in enumerate(var_components.items()):
                    if i < len(weights):
                        v = var[:len(y_true)] if len(var) >= len(y_true) else var
                        total_var += weights[i] * v
                half_width = z * np.sqrt(np.maximum(total_var, 1e-10))
                lower = center - half_width
                upper = center + half_width
                delta = upper - lower
                pen_l = (2.0 / alpha) * np.maximum(lower - y_true, 0)
                pen_u = (2.0 / alpha) * np.maximum(y_true - upper, 0)
                total_winkler += np.mean(delta + pen_l + pen_u)
            return total_winkler / max(len(val_tests), 1)

        init = np.zeros(n_signals)
        result = minimize(objective, init, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-4})
        log_w = result.x
        self.weights = np.exp(log_w) / np.sum(np.exp(log_w))
        self.weight_info = {'method': 'winkler', 'opt_result': result.fun}
        print(f"  [Fusion-Winkler] Weights: {dict(zip(self.signals, self.weights))}")
        return self.weights

    # ------------------------------------------------------------------
    # Build predictive distribution
    # ------------------------------------------------------------------

    def build_predictive_distribution(self, signal_results, alpha=0.05):
        """
        Construct a fused predictive distribution from multi-source signals.

        Returns:
            dict with keys: median, mean, std, lower_raw, upper_raw,
                            all_samples, var_components, fusion_weights, alpha
        """
        all_samples = self._collect_all_samples(signal_results)
        if all_samples is None or all_samples.shape[0] < 2:
            raise ValueError("Insufficient samples for distribution construction")

        horizon = all_samples.shape[1]
        median = np.median(all_samples, axis=0)
        mean = np.mean(all_samples, axis=0)

        lower_raw = np.percentile(all_samples, 100 * alpha / 2, axis=0)
        upper_raw = np.percentile(all_samples, 100 * (1 - alpha / 2), axis=0)

        var_components = self._compute_signal_variances(signal_results, horizon)

        if self.weights is not None and len(var_components) > 0:
            weighted_var = np.zeros(horizon)
            for i, (key, var) in enumerate(var_components.items()):
                if i < len(self.weights):
                    weighted_var += self.weights[i] * var
            weighted_std = np.sqrt(np.maximum(weighted_var, 1e-10))
        else:
            weighted_std = np.std(all_samples, axis=0)

        # Three-way uncertainty decomposition
        decomposition = None
        try:
            from uncertainty.theory import decompose_uncertainty
            decomposition = decompose_uncertainty(signal_results)
        except (ValueError, ImportError):
            pass

        return {
            'median': median,
            'mean': mean,
            'std': weighted_std,
            'lower_raw': lower_raw,
            'upper_raw': upper_raw,
            'all_samples': all_samples,
            'var_components': var_components,
            'fusion_weights': self.weights,
            'weight_method': self.weight_method,
            'weight_info': self.weight_info,
            'decomposition': decomposition,
            'alpha': alpha,
        }
