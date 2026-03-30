"""
UncertaintyPipeline — End-to-end orchestration of the three-stage pipeline:

    Stage 1  →  Multi-source signal extraction   (signals.py)
    Stage 2  →  Signal fusion                    (fusion.py)
    Stage 3  →  Conformal calibration            (conformal.py)
              + Evaluation                       (evaluator.py)

Data split strategy:
  |<-- context (train) -->|<-- calibration -->|<-- test -->|

  Calibration windows are created by rolling over the calibration period:
    for each offset t in [0, cal_len - horizon, step]:
        context_t  = train + cal[:t]
        target_t   = cal[t : t+horizon]
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict

from uncertainty.signals import extract_all_signals
from uncertainty.fusion import MultiSourceUncertaintyFusion
from uncertainty.conformal import CPLLM
from uncertainty.evaluator import UncertaintyEvaluator


class UncertaintyPipeline:
    """
    Full pipeline for one (model, dataset) pair.

    Usage:
        pipe = UncertaintyPipeline(model='deepseek-v3', hypers=deepseek_hypers)
        results = pipe.run(train, test, cal_ratio=0.3)
    """

    def __init__(self, model, hypers,
                 signals_to_use=('A', 'B', 'D'),
                 cp_method='cqr',
                 weight_method='blue',
                 alpha=0.05,
                 aci_gamma=0.005,
                 signal_A_kwargs=None,
                 signal_B_kwargs=None,
                 signal_C_kwargs=None,
                 signal_D_kwargs=None):
        """
        Args:
            model: str, LLM name matching models/llms.py keys
            hypers: dict, must contain 'settings' (SerializerSettings)
            signals_to_use: tuple, subset of ('A','B','C','D')
            cp_method: str, 'split' / 'cqr' / 'aci'
            weight_method: str, 'blue' / 'empirical_bayes' / 'winkler'
            alpha: float, nominal miscoverage
            aci_gamma: float, ACI learning rate (C1 ablation)
            signal_*_kwargs: dict, extra kwargs for each signal extractor
        """
        self.model = model
        self.hypers = hypers
        self.signals_to_use = signals_to_use
        self.cp_method = cp_method
        self.weight_method = weight_method
        self.alpha = alpha

        self.signal_A_kwargs = signal_A_kwargs or {}
        self.signal_B_kwargs = signal_B_kwargs or {}
        self.signal_C_kwargs = signal_C_kwargs or {}
        self.signal_D_kwargs = signal_D_kwargs or {}

        self.fusion = MultiSourceUncertaintyFusion(signals_to_use,
                                                    weight_method=weight_method)
        self.cpllm = CPLLM(self.fusion, method=cp_method, aci_gamma=aci_gamma)
        self.evaluator = UncertaintyEvaluator()

    # ------------------------------------------------------------------
    # Data split helpers
    # ------------------------------------------------------------------

    @staticmethod
    def split_train_cal_test(series, cal_ratio=0.2, test_ratio=0.2):
        """
        Split a full time series into train / calibration / test.

        Returns:
            (train, cal, test) as pd.Series
        """
        n = len(series)
        test_len = int(n * test_ratio)
        cal_len = int(n * cal_ratio)
        train_len = n - cal_len - test_len

        train = series.iloc[:train_len]
        cal = series.iloc[train_len:train_len + cal_len]
        test = series.iloc[train_len + cal_len:]
        return train, cal, test

    @staticmethod
    def make_calibration_windows(train, cal, horizon, step=None):
        """
        Create rolling calibration windows from the calibration period.

        Each window uses (train + cal[:offset]) as context and
        cal[offset:offset+horizon] as the calibration target.

        Returns:
            list of (context_series, target_array) tuples
        """
        if step is None:
            step = max(1, horizon // 2)

        cal_values = cal.values if isinstance(cal, pd.Series) else np.asarray(cal)
        cal_len = len(cal_values)
        windows = []

        for offset in range(0, cal_len - horizon + 1, step):
            context = pd.concat([train, cal.iloc[:offset]]) if offset > 0 else train
            target = cal_values[offset:offset + horizon]
            windows.append((context, target))

        if len(windows) == 0 and cal_len >= horizon:
            windows.append((train, cal_values[:horizon]))

        return windows

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

    def _extract_signals_for_window(self, context, horizon):
        """Run signal extraction for one context window."""
        return extract_all_signals(
            context, horizon, self.model, self.hypers,
            signals_to_use=self.signals_to_use,
            signal_A_kwargs=self.signal_A_kwargs,
            signal_B_kwargs=self.signal_B_kwargs,
            signal_C_kwargs=self.signal_C_kwargs,
            signal_D_kwargs=self.signal_D_kwargs,
        )

    def run(self, train, test, cal_ratio=0.3, cal_step=None):
        """
        Execute the full three-stage pipeline for one dataset.

        If train and test are already split, the calibration set is carved
        from the end of train (last cal_ratio fraction).

        Args:
            train: pd.Series, training history
            test: pd.Series or np.ndarray, true future values
            cal_ratio: float, fraction of train to use for calibration
            cal_step: int or None, step size for calibration windows

        Returns:
            dict with keys:
              - prediction: full prediction dict with conformal intervals
              - report: evaluation metrics dict
              - raw_signals: signal extraction results for the test context
              - calibration_info: details about calibration
        """
        test_values = test.values if isinstance(test, pd.Series) else np.asarray(test)
        horizon = len(test_values)

        # --- Carve calibration set from the end of train ---
        cal_len = max(int(len(train) * cal_ratio), horizon)
        if cal_len > len(train) - horizon:
            cal_len = len(train) // 3
            print(f"  [Pipeline] Reduced cal_len to {cal_len} due to short train")

        train_context = train.iloc[:len(train) - cal_len]
        cal_period = train.iloc[len(train) - cal_len:]

        print(f"  [Pipeline] train_context={len(train_context)}, "
              f"cal_period={len(cal_period)}, test={horizon}")

        # --- Stage 1+2: Calibration windows ---
        cal_windows = self.make_calibration_windows(
            train_context, cal_period, horizon, step=cal_step
        )
        print(f"  [Pipeline] {len(cal_windows)} calibration windows")

        cal_signal_results = []
        cal_targets = []
        for i, (ctx, target) in enumerate(cal_windows):
            print(f"  [Pipeline] Calibration window {i+1}/{len(cal_windows)}")
            try:
                signals = self._extract_signals_for_window(ctx, horizon)
                cal_signal_results.append(signals)
                cal_targets.append(target)
            except Exception as e:
                print(f"    Calibration window {i+1} failed: {e}")

        if len(cal_signal_results) == 0:
            raise RuntimeError("All calibration windows failed")

        # --- Stage 2: Learn fusion weights ---
        print(f"  [Pipeline] Learning fusion weights...")
        self.fusion.calibrate_weights(cal_signal_results, cal_targets, self.alpha)

        # --- Stage 3: Conformal calibration ---
        print(f"  [Pipeline] Conformal calibration ({self.cp_method})...")
        self.cpllm.calibrate(cal_signal_results, cal_targets, self.alpha)

        # --- Test prediction ---
        print(f"  [Pipeline] Test prediction...")
        test_signals = self._extract_signals_for_window(train, horizon)
        prediction = self.cpllm.predict(test_signals, self.alpha)

        # --- Evaluation with formal statistical tests ---
        report = self.evaluator.full_report(
            test_values, prediction, self.alpha,
            signal_results=test_signals, include_theory=True
        )

        # --- Coverage guarantee from conformal calibration ---
        coverage_guarantee = getattr(self.cpllm, 'coverage_guarantee', None)

        return {
            'prediction': prediction,
            'report': report,
            'raw_signals': test_signals,
            'test_values': test_values.tolist(),
            'calibration_info': {
                'n_cal_windows': len(cal_signal_results),
                'fusion_weights': self.fusion.weights.tolist() if self.fusion.weights is not None else None,
                'weight_method': self.fusion.weight_method,
                'weight_info': self.fusion.weight_info,
                'cp_method': self.cp_method,
                'alpha': self.alpha,
                'coverage_guarantee': coverage_guarantee,
            },
        }

    # ------------------------------------------------------------------
    # Ablation helpers
    # ------------------------------------------------------------------

    def run_ablation(self, train, test, method_configs, cal_ratio=0.3):
        """
        Run multiple method configurations for ablation study.

        Also runs Diebold-Mariano tests between methods when possible.

        Args:
            train, test: as in run()
            method_configs: dict mapping method_name → dict with optional
                'signals_to_use', 'cp_method', 'weight_method' overrides

        Returns:
            dict mapping method_name → run() result, plus 'dm_tests'
        """
        results = {}
        residuals = {}
        for name, config in method_configs.items():
            print(f"\n{'='*60}")
            print(f"  Ablation: {name}")
            print(f"{'='*60}")
            self.signals_to_use = config.get('signals_to_use', ('A',))
            self.cp_method = config.get('cp_method', 'cqr')
            wm = config.get('weight_method', 'blue')
            self.fusion = MultiSourceUncertaintyFusion(self.signals_to_use,
                                                        weight_method=wm)
            self.cpllm = CPLLM(self.fusion, method=self.cp_method)
            try:
                results[name] = self.run(train, test, cal_ratio)
                pred = results[name]['prediction']
                test_values = test.values if isinstance(test, pd.Series) else np.asarray(test)
                residuals[name] = test_values[:len(pred['median'])] - pred['median']
            except Exception as e:
                print(f"  Ablation {name} failed: {e}")
                results[name] = {'error': str(e)}

        # Pairwise Diebold-Mariano tests
        if len(residuals) >= 2:
            from uncertainty.theory import diebold_mariano_test
            dm_tests = {}
            method_names = list(residuals.keys())
            for i in range(len(method_names)):
                for j in range(i + 1, len(method_names)):
                    m1, m2 = method_names[i], method_names[j]
                    dm = diebold_mariano_test(
                        residuals[m1], residuals[m2],
                        h=1, loss_fn='squared'
                    )
                    dm_tests[f"{m1}_vs_{m2}"] = dm
            results['dm_tests'] = dm_tests

        return results


# ============================================================================
# Convenience functions for experiments
# ============================================================================

def run_bias_diagnosis(train, test, model, hypers, num_samples=100,
                       nominal_levels=None):
    """
    Experiment 1: Diagnose LLM uncertainty bias (overconfidence).

    Uses only Signal A (pure sampling) to assess the LLM's native
    calibration without any correction.

    Returns:
        dict with 'coverage_curve', 'bias_fingerprint', 'samples', 'metrics'
    """
    from uncertainty.signals import extract_sampling_dispersion

    if nominal_levels is None:
        nominal_levels = [0.50, 0.70, 0.80, 0.90, 0.95, 0.99]

    test_values = test.values if isinstance(test, pd.Series) else np.asarray(test)
    horizon = len(test_values)

    sig_a = extract_sampling_dispersion(
        train, horizon, model, hypers, num_samples=num_samples
    )
    samples = sig_a['samples']
    evaluator = UncertaintyEvaluator()

    cc = evaluator.coverage_calibration_curve(test_values, samples, nominal_levels)

    bias_fingerprint = np.array(cc['actual']) - np.array(cc['nominal'])

    metrics = {
        'NMSE': evaluator.nmse(test_values, sig_a['median']),
        'NMAE': evaluator.nmae(test_values, sig_a['median']),
        'CRPS': evaluator.crps_empirical(test_values, samples),
        'ECE': evaluator.calibration_error(test_values, samples),
    }

    pit = evaluator.pit_values(test_values, samples)
    pit_test = evaluator.pit_uniformity_test(pit)
    metrics['PIT_KS_stat'] = pit_test['ks_statistic']
    metrics['PIT_pvalue'] = pit_test['p_value']

    # Formal hypothesis tests for overconfidence diagnosis
    hypothesis_tests = {}
    try:
        from uncertainty.theory import (
            kupiec_pof_test, christoffersen_cc_test,
            berkowitz_density_test, crps_confidence_interval,
        )

        # Test at each nominal level
        for level in nominal_levels:
            al = 1 - level
            lo = np.percentile(samples, 100 * al / 2, axis=0)[:len(test_values)]
            hi = np.percentile(samples, 100 * (1 - al / 2), axis=0)[:len(test_values)]
            hits = ((test_values < lo) | (test_values > hi)).astype(int)
            violations = int(hits.sum())

            hypothesis_tests[f'kupiec_{level}'] = kupiec_pof_test(
                violations, len(test_values), al)
            hypothesis_tests[f'christoffersen_{level}'] = christoffersen_cc_test(
                hits, al)

        hypothesis_tests['berkowitz'] = berkowitz_density_test(pit)
        hypothesis_tests['crps_ci'] = crps_confidence_interval(test_values, samples)
    except ImportError:
        pass

    return {
        'coverage_curve': cc,
        'bias_fingerprint': bias_fingerprint.tolist(),
        'samples': samples,
        'metrics': metrics,
        'hypothesis_tests': hypothesis_tests,
        'model': model,
        # store ground-truth test values to enable interval visualizations
        'test_values': test_values.tolist(),
    }


def save_results(results, output_dir, dataset_name, model_name, experiment_tag=''):
    """Persist experiment results to disk."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    tag = f"_{experiment_tag}" if experiment_tag else ""
    base_name = f"{dataset_name}_{model_name}{tag}_{timestamp}"

    report = results.get('report', results.get('metrics', {}))
    report_clean = {}
    for k, v in report.items():
        if isinstance(v, (int, float, str, bool)):
            report_clean[k] = v
        elif isinstance(v, dict):
            report_clean[k] = {
                sk: sv for sk, sv in v.items()
                if isinstance(sv, (int, float, str, bool, list))
            }
        elif isinstance(v, (list, np.ndarray)):
            report_clean[k] = list(np.asarray(v).tolist()) if isinstance(v, np.ndarray) else v

    json_path = os.path.join(output_dir, f"{base_name}_report.json")
    with open(json_path, 'w') as f:
        json.dump(report_clean, f, indent=2, default=str)

    prediction = results.get('prediction', {})
    if 'all_samples' in prediction:
        # Samples from full uncertainty pipeline (CPLLM / fusion)
        samples_df = pd.DataFrame(prediction['all_samples'])
        csv_path = os.path.join(output_dir, f"{base_name}_samples.csv")
        samples_df.to_csv(csv_path, index=False)
    elif 'samples' in results:
        # Raw sampling-based bias diagnosis (Signal A only)
        samples_df = pd.DataFrame(np.asarray(results['samples']))
        csv_path = os.path.join(output_dir, f"{base_name}_samples.csv")
        samples_df.to_csv(csv_path, index=False)

    # Save ground-truth targets when available (for interval plots)
    if 'test_values' in results:
        y = np.asarray(results['test_values'])
        y_path = os.path.join(output_dir, f"{base_name}_y_true.csv")
        pd.DataFrame({'y_true': y}).to_csv(y_path, index=False)

    print(f"  [Save] Results saved to {output_dir}/{base_name}_*")
    return json_path
