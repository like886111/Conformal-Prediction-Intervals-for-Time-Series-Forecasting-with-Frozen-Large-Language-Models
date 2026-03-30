"""
Multi-Source Uncertainty Signal Extraction (Stage 1)

Four complementary uncertainty signals for LLM time series forecasting:
  - Signal A: Sampling Dispersion (采样分散度)
  - Signal B: Temperature Sensitivity (温度敏感性)
  - Signal C: Cross-Model Disagreement (跨模型不一致性) [optional]
  - Signal D: Serialization Perturbation Sensitivity (序列化扰动敏感性)
"""

import numpy as np
import pandas as pd
from copy import deepcopy
from dataclasses import replace
from models.llmtime import get_llmtime_predictions_data
from data.serialize import SerializerSettings


def _make_dummy_test(train, horizon):
    """Create a dummy test series of the correct length for LLM prediction."""
    start_idx = len(train)
    return pd.Series(np.zeros(horizon), index=pd.RangeIndex(start_idx, start_idx + horizon))


def _extract_sample_matrix(pred_dict):
    """Extract (num_samples, horizon) numpy matrix from prediction dict."""
    samples = pred_dict['samples']
    if isinstance(samples, pd.DataFrame):
        return samples.values
    elif isinstance(samples, list):
        return np.array([s.values if isinstance(s, pd.Series) else s for s in samples])
    else:
        return np.array(samples)


def _safe_predict(train, test, model, settings, num_samples, temp,
                  alpha, beta, basic, max_retries=3):
    """Call get_llmtime_predictions_data with retry logic for robustness."""
    for attempt in range(max_retries):
        try:
            pred_dict = get_llmtime_predictions_data(
                train, test, model=model, settings=settings,
                num_samples=num_samples, temp=temp,
                alpha=alpha, beta=beta, basic=basic, parallel=False
            )
            sm = _extract_sample_matrix(pred_dict)
            valid_mask = ~np.isnan(sm).any(axis=1)
            sm = sm[valid_mask]
            if sm.shape[0] >= 1:
                return sm, pred_dict
        except Exception as e:
            print(f"  [Signal] Attempt {attempt+1}/{max_retries} failed: {e}")
    return None, None


# ============================================================================
# Signal A: Sampling Dispersion
# ============================================================================

def extract_sampling_dispersion(train, horizon, model, hypers,
                                num_samples=20, temperature=1.0):
    """
    Signal A: At a fixed temperature, draw multiple samples to capture
    the LLM's inherent sampling variability.

    Args:
        train: pd.Series, training history
        horizon: int, prediction steps
        model: str, LLM model name
        hypers: dict, model hyperparameters (must include 'settings')
        num_samples: int, number of samples to draw
        temperature: float, sampling temperature

    Returns:
        dict with keys: samples, median, mean, std, iqr, quantiles, num_valid
    """
    settings = hypers['settings']
    alpha = hypers.get('alpha', 0.95)
    beta = hypers.get('beta', 0.3)
    basic = hypers.get('basic', False)

    dummy_test = _make_dummy_test(train, horizon)

    sm, _ = _safe_predict(
        train, dummy_test, model, settings,
        num_samples=num_samples, temp=temperature,
        alpha=alpha, beta=beta, basic=basic
    )

    if sm is None or sm.shape[0] < 2:
        raise RuntimeError(f"Signal A: Failed to get valid samples for model={model}")

    quantile_levels = [0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975]
    quantiles = {q: np.percentile(sm, 100 * q, axis=0) for q in quantile_levels}

    return {
        'samples': sm,
        'median': np.median(sm, axis=0),
        'mean': np.mean(sm, axis=0),
        'std': np.std(sm, axis=0),
        'iqr': np.percentile(sm, 75, axis=0) - np.percentile(sm, 25, axis=0),
        'quantiles': quantiles,
        'num_valid': sm.shape[0],
    }


# ============================================================================
# Signal B: Temperature Sensitivity
# ============================================================================

def extract_temperature_sensitivity(train, horizon, model, hypers,
                                    temperatures=None,
                                    samples_per_temp=5):
    """
    Signal B: Probe LLM uncertainty by varying temperature. The inter-temperature
    variance relative to intra-temperature variance reveals model confidence.

    High sensitivity_ratio → model is temperature-sensitive → high uncertainty.

    Args:
        train: pd.Series
        horizon: int
        model: str
        hypers: dict
        temperatures: list of float
        samples_per_temp: int

    Returns:
        dict with keys: temp_predictions, temp_medians, inter_temp_var,
                        intra_temp_var, sensitivity_ratio, temperatures
    """
    if temperatures is None:
        temperatures = [0.3, 0.5, 0.7, 1.0, 1.3]

    settings = hypers['settings']
    alpha = hypers.get('alpha', 0.95)
    beta = hypers.get('beta', 0.3)
    basic = hypers.get('basic', False)

    dummy_test = _make_dummy_test(train, horizon)

    temp_predictions = {}
    temp_medians = []
    temp_stds = []

    for tau in temperatures:
        sm, _ = _safe_predict(
            train, dummy_test, model, settings,
            num_samples=samples_per_temp, temp=tau,
            alpha=alpha, beta=beta, basic=basic
        )
        if sm is None or sm.shape[0] < 1:
            print(f"  [Signal B] Skipping temp={tau}, no valid samples")
            continue
        temp_predictions[tau] = sm
        temp_medians.append(np.median(sm, axis=0))
        temp_stds.append(np.std(sm, axis=0))

    if len(temp_medians) < 2:
        raise RuntimeError("Signal B: Need at least 2 valid temperature points")

    temp_medians = np.array(temp_medians)   # (n_temps, horizon)
    temp_stds = np.array(temp_stds)

    inter_temp_var = np.var(temp_medians, axis=0)
    intra_temp_var = np.mean(temp_stds ** 2, axis=0)
    sensitivity_ratio = inter_temp_var / (intra_temp_var + 1e-10)

    return {
        'temp_predictions': temp_predictions,
        'temp_medians': temp_medians,
        'inter_temp_var': inter_temp_var,
        'intra_temp_var': intra_temp_var,
        'sensitivity_ratio': sensitivity_ratio,
        'temperatures': list(temp_predictions.keys()),
    }


# ============================================================================
# Signal C: Cross-Model Disagreement (optional, multi-model)
# ============================================================================

def extract_cross_model_disagreement(train, horizon, models_with_hypers,
                                     samples_per_model=5):
    """
    Signal C: Measure disagreement across multiple LLMs as an epistemic
    uncertainty proxy. Inspired by MUSE (EMNLP 2025).

    Args:
        train: pd.Series
        horizon: int
        models_with_hypers: list of (model_name, hypers_dict) tuples
        samples_per_model: int

    Returns:
        dict with keys: model_medians, disagreement_var, ensemble_median,
                        all_model_samples
    """
    dummy_test = _make_dummy_test(train, horizon)

    model_medians = []
    all_model_samples = []

    for model_name, hypers in models_with_hypers:
        settings = hypers['settings']
        alpha = hypers.get('alpha', 0.95)
        beta = hypers.get('beta', 0.3)
        basic = hypers.get('basic', False)
        temp = hypers.get('temp', 1.0)

        sm, _ = _safe_predict(
            train, dummy_test, model_name, settings,
            num_samples=samples_per_model, temp=temp,
            alpha=alpha, beta=beta, basic=basic
        )
        if sm is None or sm.shape[0] < 1:
            print(f"  [Signal C] Skipping model={model_name}, no valid samples")
            continue

        model_medians.append(np.median(sm, axis=0))
        all_model_samples.append(sm)

    if len(model_medians) < 2:
        raise RuntimeError("Signal C: Need at least 2 valid models")

    model_medians = np.array(model_medians)  # (n_models, horizon)
    disagreement_var = np.var(model_medians, axis=0)
    ensemble_median = np.median(model_medians, axis=0)

    return {
        'model_medians': model_medians,
        'disagreement_var': disagreement_var,
        'ensemble_median': ensemble_median,
        'all_model_samples': all_model_samples,
    }


# ============================================================================
# Signal D: Serialization Perturbation Sensitivity
# ============================================================================

def extract_serialization_sensitivity(train, horizon, model, hypers,
                                      perturbations=None,
                                      samples_per_pert=3):
    """
    Signal D: Exploit LLMTime's unique serialization step — small changes
    in precision, separator, or sign representation should not change a
    confident prediction, but reveal uncertainty in ambiguous regions.

    This is the key novel contribution distinguishing from NBA-LLM (which
    perturbs input data, not the representation).

    Args:
        train: pd.Series
        horizon: int
        model: str
        hypers: dict
        perturbations: list of dicts, each overriding SerializerSettings fields
        samples_per_pert: int

    Returns:
        dict with keys: perturbation_medians, serialization_var,
                        perturbation_samples, perturbation_configs
    """
    base_settings = hypers['settings']
    alpha = hypers.get('alpha', 0.95)
    beta = hypers.get('beta', 0.3)
    basic = hypers.get('basic', False)
    temp = hypers.get('temp', 1.0)

    if perturbations is None:
        perturbations = [
            {},                                     # P0: baseline (no change)
            {'prec': 2},                            # P1: lower precision
            {'prec': 4},                            # P2: higher precision
            {'time_sep': '; '},                     # P3: different separator
            {'time_sep': ' | '},                    # P4: another separator
        ]

    dummy_test = _make_dummy_test(train, horizon)

    perturbation_medians = []
    perturbation_samples = []
    perturbation_configs = []

    for pert in perturbations:
        pert_settings = replace(base_settings, **pert)

        sm, _ = _safe_predict(
            train, dummy_test, model, pert_settings,
            num_samples=samples_per_pert, temp=temp,
            alpha=alpha, beta=beta, basic=basic
        )
        if sm is None or sm.shape[0] < 1:
            print(f"  [Signal D] Skipping perturbation {pert}, no valid samples")
            continue

        perturbation_medians.append(np.median(sm, axis=0))
        perturbation_samples.append(sm)
        perturbation_configs.append(pert)

    if len(perturbation_medians) < 2:
        raise RuntimeError("Signal D: Need at least 2 valid perturbations")

    perturbation_medians = np.array(perturbation_medians)  # (n_perts, horizon)
    serialization_var = np.var(perturbation_medians, axis=0)

    return {
        'perturbation_medians': perturbation_medians,
        'serialization_var': serialization_var,
        'perturbation_samples': perturbation_samples,
        'perturbation_configs': perturbation_configs,
    }


# ============================================================================
# Unified signal extraction
# ============================================================================

def extract_all_signals(train, horizon, model, hypers,
                        signals_to_use=('A', 'B', 'D'),
                        signal_A_kwargs=None,
                        signal_B_kwargs=None,
                        signal_C_kwargs=None,
                        signal_D_kwargs=None):
    """
    Extract specified uncertainty signals for a single (train, horizon) pair.

    Args:
        train: pd.Series
        horizon: int
        model: str, primary LLM model name
        hypers: dict, primary model hyperparameters
        signals_to_use: tuple of str, subset of ('A', 'B', 'C', 'D')
        signal_*_kwargs: dict, additional kwargs for each signal extractor

    Returns:
        dict mapping signal label ('A', 'B', 'C', 'D') to signal result dict
    """
    results = {}

    if 'A' in signals_to_use:
        kwargs = signal_A_kwargs or {}
        print(f"  Extracting Signal A (Sampling Dispersion)...")
        results['A'] = extract_sampling_dispersion(
            train, horizon, model, hypers, **kwargs
        )
        print(f"    → {results['A']['num_valid']} valid samples")

    if 'B' in signals_to_use:
        kwargs = signal_B_kwargs or {}
        print(f"  Extracting Signal B (Temperature Sensitivity)...")
        results['B'] = extract_temperature_sensitivity(
            train, horizon, model, hypers, **kwargs
        )
        print(f"    → {len(results['B']['temperatures'])} temperature points")

    if 'C' in signals_to_use:
        kwargs = signal_C_kwargs or {}
        if 'models_with_hypers' not in kwargs:
            raise ValueError("Signal C requires 'models_with_hypers' in signal_C_kwargs")
        print(f"  Extracting Signal C (Cross-Model Disagreement)...")
        results['C'] = extract_cross_model_disagreement(
            train, horizon, **kwargs
        )
        print(f"    → {len(results['C']['model_medians'])} models compared")

    if 'D' in signals_to_use:
        kwargs = signal_D_kwargs or {}
        print(f"  Extracting Signal D (Serialization Sensitivity)...")
        results['D'] = extract_serialization_sensitivity(
            train, horizon, model, hypers, **kwargs
        )
        print(f"    → {len(results['D']['perturbation_configs'])} perturbation variants")

    return results
