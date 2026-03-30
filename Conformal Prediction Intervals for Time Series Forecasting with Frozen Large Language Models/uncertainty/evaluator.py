"""
Comprehensive Uncertainty Evaluation Metrics

Covers:
  - Coverage: ECP, Coverage Gap, Conditional Coverage
  - Sharpness: AIW, NAIW
  - Scoring rules: Winkler, CRPS, NLL
  - Calibration: ECE, PIT uniformity
  - Point accuracy: NMSE, NMAE
  - Cost efficiency: API calls per unit CRPS
"""

import numpy as np
from scipy import stats as sp_stats


class UncertaintyEvaluator:
    """Stateless evaluator — all methods are static or take data as arguments."""

    # ------------------------------------------------------------------
    # Coverage metrics
    # ------------------------------------------------------------------

    @staticmethod
    def empirical_coverage(y_true, lower, upper):
        """Fraction of true values falling inside [lower, upper]."""
        y, lo, hi = _align(y_true, lower, upper)
        return float(np.mean((y >= lo) & (y <= hi)))

    @staticmethod
    def coverage_gap(y_true, lower, upper, alpha):
        """ECP - (1-alpha).  Positive = conservative, negative = overconfident."""
        ecp = UncertaintyEvaluator.empirical_coverage(y_true, lower, upper)
        return ecp - (1 - alpha)

    @staticmethod
    def conditional_coverage_variance(y_true, lower, upper, n_bins=5):
        """
        Variance of coverage across bins of the prediction range,
        measuring conditional coverage stability.
        """
        y, lo, hi = _align(y_true, lower, upper)
        midpoints = (lo + hi) / 2
        bin_edges = np.percentile(midpoints, np.linspace(0, 100, n_bins + 1))
        bin_coverages = []
        for b in range(n_bins):
            mask = (midpoints >= bin_edges[b]) & (midpoints < bin_edges[b + 1])
            if mask.sum() > 0:
                bin_coverages.append(np.mean((y[mask] >= lo[mask]) & (y[mask] <= hi[mask])))
        return float(np.var(bin_coverages)) if bin_coverages else 0.0

    # ------------------------------------------------------------------
    # Sharpness (interval width)
    # ------------------------------------------------------------------

    @staticmethod
    def average_width(lower, upper):
        lo, hi = np.asarray(lower), np.asarray(upper)
        return float(np.mean(hi - lo))

    @staticmethod
    def normalized_average_width(lower, upper, y_true):
        """Width normalized by the standard deviation of the true values."""
        lo, hi, y = np.asarray(lower), np.asarray(upper), np.asarray(y_true)
        return float(np.mean(hi - lo) / (np.std(y) + 1e-10))

    # ------------------------------------------------------------------
    # Scoring rules
    # ------------------------------------------------------------------

    @staticmethod
    def winkler_score(y_true, lower, upper, alpha=0.05):
        y, lo, hi = _align(y_true, lower, upper)
        delta = hi - lo
        pen_l = (2.0 / alpha) * np.maximum(lo - y, 0)
        pen_u = (2.0 / alpha) * np.maximum(y - hi, 0)
        return float(np.mean(delta + pen_l + pen_u))

    @staticmethod
    def crps_empirical(y_true, samples):
        """
        Sample-based CRPS:
          CRPS = E|X - y| - 0.5 * E|X - X'|
        """
        y = np.asarray(y_true).ravel()
        S = np.asarray(samples)
        N, H = S.shape
        H = min(H, len(y))
        y = y[:H]
        S = S[:, :H]

        term1 = np.mean(np.abs(S - y[None, :]), axis=0)
        if N > 1:
            diffs = np.abs(S[:, None, :] - S[None, :, :])
            term2 = diffs.sum(axis=(0, 1)) / (N * (N - 1))
        else:
            term2 = 0.0
        return float(np.mean(term1 - 0.5 * term2))

    @staticmethod
    def nll_gaussian(y_true, pred_mean, pred_var):
        """Negative log-likelihood under Gaussian assumption."""
        y = np.asarray(y_true).ravel()
        mu = np.asarray(pred_mean).ravel()
        var = np.maximum(np.asarray(pred_var).ravel(), 1e-10)
        n = min(len(y), len(mu))
        y, mu, var = y[:n], mu[:n], var[:n]
        return float(np.mean(0.5 * np.log(2 * np.pi * var) +
                             0.5 * (y - mu) ** 2 / var))

    # ------------------------------------------------------------------
    # Calibration diagnostics
    # ------------------------------------------------------------------

    @staticmethod
    def pit_values(y_true, samples):
        """Probability Integral Transform: P(X <= y) for each step."""
        y = np.asarray(y_true).ravel()
        S = np.asarray(samples)
        H = min(S.shape[1], len(y))
        return np.array([np.mean(S[:, h] <= y[h]) for h in range(H)])

    @staticmethod
    def pit_uniformity_test(pit_vals):
        """KS test: PIT values should be ~ Uniform(0,1) if well calibrated."""
        stat, pvalue = sp_stats.kstest(pit_vals, 'uniform')
        return {'ks_statistic': float(stat), 'p_value': float(pvalue)}

    @staticmethod
    def calibration_error(y_true, samples, n_levels=20):
        """
        Expected Calibration Error (ECE) across quantile levels.
        Measures how well predicted quantiles match observed frequencies.
        """
        y = np.asarray(y_true).ravel()
        S = np.asarray(samples)
        H = min(S.shape[1], len(y))
        levels = np.linspace(0.05, 0.95, n_levels)
        observed = []
        for q in levels:
            q_pred = np.percentile(S[:, :H], 100 * q, axis=0)
            observed.append(np.mean(y[:H] <= q_pred))
        return float(np.mean(np.abs(np.array(observed) - levels)))

    @staticmethod
    def coverage_calibration_curve(y_true, samples, nominal_levels=None):
        """
        For the 'Uncertainty Bias Fingerprint': compute actual coverage
        at each nominal level.

        Returns:
            dict with 'nominal' and 'actual' arrays
        """
        if nominal_levels is None:
            nominal_levels = [0.50, 0.70, 0.80, 0.90, 0.95, 0.99]
        y = np.asarray(y_true).ravel()
        S = np.asarray(samples)
        H = min(S.shape[1], len(y))
        y = y[:H]
        S = S[:, :H]

        actual_coverages = []
        for level in nominal_levels:
            alpha = 1 - level
            lo = np.percentile(S, 100 * alpha / 2, axis=0)
            hi = np.percentile(S, 100 * (1 - alpha / 2), axis=0)
            actual_coverages.append(float(np.mean((y >= lo) & (y <= hi))))
        return {'nominal': nominal_levels, 'actual': actual_coverages}

    # ------------------------------------------------------------------
    # Point accuracy
    # ------------------------------------------------------------------

    @staticmethod
    def nmse(y_true, y_pred):
        y, yp = np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()
        n = min(len(y), len(yp))
        return float(np.mean((y[:n] - yp[:n]) ** 2) / (np.var(y[:n]) + 1e-10))

    @staticmethod
    def nmae(y_true, y_pred):
        y, yp = np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()
        n = min(len(y), len(yp))
        return float(np.mean(np.abs(y[:n] - yp[:n])) /
                     (np.mean(np.abs(y[:n])) + 1e-10))

    # ------------------------------------------------------------------
    # Full report
    # ------------------------------------------------------------------

    def full_report(self, y_true, prediction, alpha=0.05,
                    signal_results=None, include_theory=True):
        """
        Generate a comprehensive evaluation report with optional
        formal statistical tests.

        Args:
            y_true: np.ndarray
            prediction: dict from CPLLM.predict() or fusion.build_predictive_distribution()
            alpha: float
            signal_results: dict from extract_all_signals (for decomposition)
            include_theory: bool, whether to run formal hypothesis tests

        Returns:
            dict of metric_name → value
        """
        lower_key = 'conformal_lower' if 'conformal_lower' in prediction else 'lower_raw'
        upper_key = 'conformal_upper' if 'conformal_upper' in prediction else 'upper_raw'
        lower = prediction[lower_key]
        upper = prediction[upper_key]
        median = prediction['median']
        samples = prediction.get('all_samples', None)
        y = np.asarray(y_true).ravel()

        report = {
            'ECP': self.empirical_coverage(y, lower, upper),
            'Coverage_Gap': self.coverage_gap(y, lower, upper, alpha),
            'AIW': self.average_width(lower, upper),
            'NAIW': self.normalized_average_width(lower, upper, y),
            'Winkler': self.winkler_score(y, lower, upper, alpha),
            'NMSE': self.nmse(y, median),
            'NMAE': self.nmae(y, median),
        }

        if samples is not None and samples.shape[0] >= 2:
            report['CRPS'] = self.crps_empirical(y, samples)
            report['ECE'] = self.calibration_error(y, samples)
            pit = self.pit_values(y, samples)
            pit_test = self.pit_uniformity_test(pit)
            report['PIT_KS_stat'] = pit_test['ks_statistic']
            report['PIT_pvalue'] = pit_test['p_value']
            cc = self.coverage_calibration_curve(y, samples)
            report['coverage_curve'] = cc

        std = prediction.get('std', None)
        if std is not None:
            mean_pred = prediction.get('mean', median)
            report['NLL'] = self.nll_gaussian(y, mean_pred, std ** 2)

        if 'calibration_method' in prediction:
            report['calibration_method'] = prediction['calibration_method']

        # Include three-way decomposition if available
        decomp = prediction.get('decomposition', None)
        if decomp is not None:
            report['uncertainty_decomposition'] = decomp['proportions']

        # ----------------------------------------------------------
        # Formal statistical tests (§ theory.py)
        # ----------------------------------------------------------
        if include_theory:
            report['hypothesis_tests'] = self._run_hypothesis_tests(
                y, lower, upper, median, samples, alpha, signal_results
            )

        return report

    def _run_hypothesis_tests(self, y, lower, upper, median,
                              samples, alpha, signal_results):
        """Formal coverage / calibration / dependence tests."""
        from uncertainty.theory import (
            kupiec_pof_test, christoffersen_cc_test,
            berkowitz_density_test, estimate_mixing_coefficient,
            compute_coverage_bound, crps_confidence_interval,
        )

        H = min(len(y), len(lower))
        y, lower, upper, median = y[:H], lower[:H], upper[:H], median[:H]
        hits = ((y < lower) | (y > upper)).astype(int)
        violations = int(hits.sum())

        tests = {}

        # Kupiec POF test for unconditional coverage
        tests['kupiec'] = kupiec_pof_test(violations, H, alpha)

        # Christoffersen conditional coverage test
        tests['christoffersen'] = christoffersen_cc_test(hits, alpha)

        # Berkowitz density forecast test (requires samples for PIT)
        if samples is not None and samples.shape[0] >= 2:
            pit = np.array([np.mean(samples[:, h] <= y[h])
                            for h in range(min(H, samples.shape[1]))])
            tests['berkowitz'] = berkowitz_density_test(pit)

        # Mixing coefficient and coverage bound
        residuals = y - median
        mixing = estimate_mixing_coefficient(residuals)
        tests['mixing_analysis'] = mixing
        tests['coverage_bound'] = compute_coverage_bound(H, alpha, mixing)

        # CRPS bootstrap CI
        if samples is not None and samples.shape[0] >= 2:
            tests['crps_inference'] = crps_confidence_interval(y, samples)

        # Three-way decomposition
        if signal_results is not None:
            try:
                from uncertainty.theory import decompose_uncertainty
                tests['decomposition'] = decompose_uncertainty(signal_results)
            except (ValueError, ImportError):
                pass

        return tests


# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------

def _align(*arrays):
    """Ensure arrays are flat np.ndarrays of matching length."""
    arrs = [np.asarray(a).ravel() for a in arrays]
    min_len = min(len(a) for a in arrs)
    return tuple(a[:min_len] for a in arrs)
