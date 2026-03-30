"""
Main experiment entry point for:
  "Multi-Source Uncertainty Quantification and Conformal Calibration
   for LLM-Based Time Series Forecasting"

Experiments:
  1. bias_diagnosis   — LLM uncertainty bias characterization (Contribution C1)
  2. method_compare   — Full pipeline vs baselines (Contribution C2+C3)
  3. ablation         — Signal and component ablation study (Contribution C3)
  4. cost_efficiency  — Pareto front of UQ quality vs API cost (Contribution C4)

Usage:
  python run_experiment.py --experiment bias_diagnosis --model deepseek-v3 --dataset darts
  python run_experiment.py --experiment method_compare --model deepseek-v3 --dataset memorization
  python run_experiment.py --experiment ablation --model deepseek-v3 --dataset memorization
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict

from data.serialize import SerializerSettings
from data.small_context import (
    get_datasets, get_memorization_datasets,
    get_ETTh1_datasets, get_ETTh2_datasets,
    get_ETTm1_datasets, get_ETTm2_datasets,
    get_exchange_rate_datasets, get_national_illness_datasets,
)
from uncertainty.signals import extract_all_signals, extract_sampling_dispersion
from uncertainty.fusion import MultiSourceUncertaintyFusion
from uncertainty.conformal import CPLLM, NaiveConformal
from uncertainty.evaluator import UncertaintyEvaluator
from uncertainty.pipeline import (
    UncertaintyPipeline, run_bias_diagnosis, save_results,
)


# ======================================================================
# Model hyperparameter registry
# ======================================================================

DEFAULT_SETTINGS = SerializerSettings(
    base=10, prec=3, signed=True,
    time_sep=', ', bit_sep='', minus_sign='-',
    half_bin_correction=False,
)

def _make_hypers(alpha=0.95, basic=True, temp=1.0, top_p=0.8, settings=None):
    return dict(
        alpha=alpha, basic=basic, temp=temp, top_p=top_p,
        settings=settings or DEFAULT_SETTINGS,
    )

MODEL_HYPERS = {
    'deepseek-v3':              _make_hypers(),
    'deepseek-r1':              _make_hypers(),
    'gemini-2.0-flash-lite':    _make_hypers(),
    'gemini-2.0-flash':         _make_hypers(),
    'qwen-plus':                _make_hypers(),
    'qwen-turbo':               _make_hypers(),
    'qwen3-8b':                 _make_hypers(),
    'Qwen2.5-32B-Instruct':    _make_hypers(),
    'claude-3-5-haiku-20241022': _make_hypers(),
    'claude-3-5-sonnet-20240620': _make_hypers(),
    'glm-4-long':               _make_hypers(),
    'grok-2-1212':              _make_hypers(),
    'gpt-3.5-turbo':            _make_hypers(alpha=0.95, basic=False, temp=0.7),
    'gpt-4o-mini':              _make_hypers(alpha=0.95, basic=False, temp=0.7),
    'gpt-5.4':                  _make_hypers(alpha=0.95, basic=False, temp=0.7),
    'LLMTime GPT-4':             _make_hypers(alpha=0.3, basic=True, temp=1.0),
}


# ======================================================================
# Dataset loaders
# ======================================================================

# 论文实验统一预测长度，确保与 memorization 一致、LLM 输出长度达标
PAPER_HORIZON = 30

def load_datasets(dataset_group, **kwargs):
    """Load and return a dict of {name: (train, test)}.
    For darts, pass predict_steps=PAPER_HORIZON so test length is fixed (e.g. 30).
    """
    loaders = {
        'memorization': get_memorization_datasets,
        'darts':        get_datasets,
        'ETTh1':        get_ETTh1_datasets,
        'ETTh2':        get_ETTh2_datasets,
        'ETTm1':        get_ETTm1_datasets,
        'ETTm2':        get_ETTm2_datasets,
        'exchange':     get_exchange_rate_datasets,
        'illness':      get_national_illness_datasets,
    }
    if dataset_group not in loaders:
        raise ValueError(f"Unknown dataset group '{dataset_group}'. "
                         f"Choose from: {list(loaders.keys())}")
    fn = loaders[dataset_group]
    if dataset_group == 'darts' and 'predict_steps' not in kwargs:
        kwargs = {**kwargs, 'predict_steps': PAPER_HORIZON}
    return fn(**kwargs)


# ======================================================================
# Experiment 1: Bias Diagnosis
# ======================================================================

def experiment_bias_diagnosis(models, dataset_group, output_dir, num_samples=50,
                               datasets_subset=None):
    """
    For each (model, dataset), compute native coverage calibration curves
    using only Signal A. Produces the 'Uncertainty Bias Fingerprint'.
    datasets_subset: optional list of dataset names to run (e.g. ['IstanbulTraffic','TSMCStock']).
    """
    datasets = load_datasets(dataset_group)
    if datasets_subset:
        datasets = {k: v for k, v in datasets.items() if k in datasets_subset}
        if not datasets:
            raise ValueError(f"datasets_subset {datasets_subset} matched no keys in {list(load_datasets(dataset_group).keys())}")
    all_results = defaultdict(dict)

    for model_name in models:
        hypers = MODEL_HYPERS.get(model_name)
        if hypers is None:
            print(f"Skipping {model_name}: no hypers defined")
            continue

        for ds_name, (train, test) in datasets.items():
            print(f"\n{'='*60}")
            print(f"  Bias Diagnosis: {model_name} × {ds_name}")
            print(f"{'='*60}")

            try:
                result = run_bias_diagnosis(
                    train, test, model_name, hypers,
                    num_samples=num_samples,
                )
                all_results[model_name][ds_name] = {
                    'coverage_curve': result['coverage_curve'],
                    'bias_fingerprint': result['bias_fingerprint'],
                    'metrics': result['metrics'],
                }
                save_results(result, output_dir, ds_name, model_name,
                             experiment_tag='bias')
            except Exception as e:
                print(f"  FAILED: {e}")
                all_results[model_name][ds_name] = {'error': str(e)}

    summary_path = os.path.join(output_dir, f'bias_diagnosis_summary_{dataset_group}.json')
    with open(summary_path, 'w') as f:
        json.dump(dict(all_results), f, indent=2, default=str)
    print(f"\nSummary saved to {summary_path}")
    return all_results


# ======================================================================
# Experiment 2: Method Comparison
# ======================================================================

ABLATION_CONFIGS = {
    'M0_naive_sampling': {
        'signals_to_use': ('A',),
        'cp_method': None,
    },
    'M2_signalA_CQR': {
        'signals_to_use': ('A',),
        'cp_method': 'cqr',
    },
    'M3_signalAB': {
        'signals_to_use': ('A', 'B'),
        'cp_method': None,
    },
    'M4_signalABD': {
        'signals_to_use': ('A', 'B', 'D'),
        'cp_method': None,
    },
    'M5_signalA_ACI': {
        'signals_to_use': ('A',),
        'cp_method': 'aci',
    },
    'M6_full_CQR': {
        'signals_to_use': ('A', 'B', 'D'),
        'cp_method': 'cqr',
    },
    'M7_full_ACI': {
        'signals_to_use': ('A', 'B', 'D'),
        'cp_method': 'aci',
    },
}


def experiment_method_compare(models, dataset_group, output_dir,
                              methods=None, alpha=0.05, datasets_subset=None):
    """
    Compare multiple method variants across models and datasets.
    datasets_subset: optional list of dataset names to run.
    """
    if methods is None:
        methods = ['M0_naive_sampling', 'M2_signalA_CQR',
                   'M6_full_CQR', 'M7_full_ACI']

    datasets = load_datasets(dataset_group)
    if datasets_subset:
        datasets = {k: v for k, v in datasets.items() if k in datasets_subset}
        if not datasets:
            raise ValueError(f"datasets_subset {datasets_subset} matched no keys in {list(load_datasets(dataset_group).keys())}")
    all_results = defaultdict(lambda: defaultdict(dict))

    for model_name in models:
        hypers = MODEL_HYPERS.get(model_name)
        if hypers is None:
            continue

        for ds_name, (train, test) in datasets.items():
            test_values = test.values if isinstance(test, pd.Series) else np.asarray(test)
            horizon = len(test_values)

            for method_name in methods:
                config = ABLATION_CONFIGS[method_name]
                print(f"\n{'='*60}")
                print(f"  {method_name}: {model_name} × {ds_name}")
                print(f"{'='*60}")

                try:
                    if config['cp_method'] is None:
                        # No conformal: just signal extraction + fusion
                        signals = extract_all_signals(
                            train, horizon, model_name, hypers,
                            signals_to_use=config['signals_to_use'],
                        )
                        fusion = MultiSourceUncertaintyFusion(config['signals_to_use'])
                        pred = fusion.build_predictive_distribution(signals, alpha)
                        evaluator = UncertaintyEvaluator()
                        report = evaluator.full_report(test_values, pred, alpha)
                        result = {'prediction': pred, 'report': report}
                    else:
                        pipe = UncertaintyPipeline(
                            model=model_name, hypers=hypers,
                            signals_to_use=config['signals_to_use'],
                            cp_method=config['cp_method'],
                            alpha=alpha,
                        )
                        result = pipe.run(train, test, cal_ratio=0.3)

                    all_results[model_name][ds_name][method_name] = result.get('report', {})
                    save_results(result, output_dir, ds_name, model_name,
                                 experiment_tag=method_name)
                except Exception as e:
                    print(f"  FAILED: {e}")
                    all_results[model_name][ds_name][method_name] = {'error': str(e)}

    summary_path = os.path.join(output_dir, f'method_compare_summary_{dataset_group}.json')
    with open(summary_path, 'w') as f:
        json.dump(_nested_to_dict(all_results), f, indent=2, default=str)
    print(f"\nSummary saved to {summary_path}")
    return all_results


# ======================================================================
# Experiment 3: Ablation Study
# ======================================================================

def experiment_ablation(model_name, dataset_group, output_dir, alpha=0.05):
    """
    Full ablation across all 7 method variants for a single model.
    """
    return experiment_method_compare(
        [model_name], dataset_group, output_dir,
        methods=list(ABLATION_CONFIGS.keys()),
        alpha=alpha,
    )


# ======================================================================
# Experiment 4: Cost Efficiency Analysis
# ======================================================================

def experiment_cost_efficiency(model_name, dataset_group, output_dir,
                               budgets=None, alpha=0.05):
    """
    Fix a model, vary the total API call budget, and measure UQ quality
    at each budget level to trace the Pareto front.
    """
    if budgets is None:
        budgets = [
            {'label': 'budget_5',  'A_samples': 5,  'B_temps': 0, 'D_perts': 0},
            {'label': 'budget_10', 'A_samples': 10, 'B_temps': 0, 'D_perts': 0},
            {'label': 'budget_20', 'A_samples': 10, 'B_temps': 2, 'D_perts': 0},
            {'label': 'budget_35', 'A_samples': 10, 'B_temps': 3, 'D_perts': 2},
            {'label': 'budget_50', 'A_samples': 15, 'B_temps': 4, 'D_perts': 3},
            {'label': 'budget_80', 'A_samples': 20, 'B_temps': 5, 'D_perts': 5},
        ]

    hypers = MODEL_HYPERS.get(model_name)
    if hypers is None:
        raise ValueError(f"No hypers for {model_name}")

    datasets = load_datasets(dataset_group)
    all_results = defaultdict(dict)

    for ds_name, (train, test) in datasets.items():
        test_values = test.values if isinstance(test, pd.Series) else np.asarray(test)

        for budget in budgets:
            label = budget['label']
            print(f"\n  Cost: {model_name} × {ds_name} × {label}")

            signals_to_use = ['A']
            sig_A_kw = {'num_samples': budget['A_samples']}
            sig_B_kw = {}
            sig_D_kw = {}

            if budget['B_temps'] >= 2:
                signals_to_use.append('B')
                temps = np.linspace(0.3, 1.3, budget['B_temps']).tolist()
                sig_B_kw = {'temperatures': temps, 'samples_per_temp': 3}

            if budget['D_perts'] >= 2:
                signals_to_use.append('D')
                perts = [{}]
                if budget['D_perts'] >= 2:
                    perts.append({'prec': 2})
                if budget['D_perts'] >= 3:
                    perts.append({'prec': 4})
                if budget['D_perts'] >= 4:
                    perts.append({'time_sep': '; '})
                if budget['D_perts'] >= 5:
                    perts.append({'time_sep': ' | '})
                sig_D_kw = {'perturbations': perts, 'samples_per_pert': 2}

            try:
                pipe = UncertaintyPipeline(
                    model=model_name, hypers=hypers,
                    signals_to_use=tuple(signals_to_use),
                    cp_method='cqr', alpha=alpha,
                    signal_A_kwargs=sig_A_kw,
                    signal_B_kwargs=sig_B_kw,
                    signal_D_kwargs=sig_D_kw,
                )
                result = pipe.run(train, test, cal_ratio=0.3)
                total_calls = (budget['A_samples'] +
                               budget['B_temps'] * 3 +
                               budget['D_perts'] * 2)
                report = result.get('report', {})
                report['total_api_calls'] = total_calls
                all_results[ds_name][label] = report
            except Exception as e:
                print(f"    FAILED: {e}")
                all_results[ds_name][label] = {'error': str(e)}

    summary_path = os.path.join(
        output_dir, f'cost_efficiency_{model_name}_{dataset_group}.json')
    with open(summary_path, 'w') as f:
        json.dump(dict(all_results), f, indent=2, default=str)
    print(f"\nSummary saved to {summary_path}")
    return all_results


# ======================================================================
# CLI
# ======================================================================

def _nested_to_dict(d):
    if isinstance(d, defaultdict):
        return {k: _nested_to_dict(v) for k, v in d.items()}
    return d


def main():
    parser = argparse.ArgumentParser(
        description='LLM Time Series Uncertainty Quantification Experiments')
    parser.add_argument('--experiment', type=str, required=True,
                        choices=['bias_diagnosis', 'method_compare',
                                 'ablation', 'cost_efficiency'],
                        help='Which experiment to run')
    parser.add_argument('--model', type=str, nargs='+', default=['deepseek-v3'],
                        help='LLM model name(s)')
    parser.add_argument('--dataset', type=str, default='memorization',
                        choices=['memorization', 'darts', 'ETTh1', 'ETTh2',
                                 'ETTm1', 'ETTm2', 'exchange', 'illness'],
                        help='Dataset group')
    parser.add_argument('--datasets', type=str, nargs='*', default=None,
                        help='Optional subset of dataset names (e.g. IstanbulTraffic TSMCStock for memorization; AirPassengersDataset WineDataset for darts)')
    parser.add_argument('--output', type=str, default='results/',
                        help='Output directory')
    parser.add_argument('--alpha', type=float, default=0.05,
                        help='Nominal miscoverage level')
    parser.add_argument('--num_samples', type=int, default=50,
                        help='Number of samples for bias diagnosis')
    parser.add_argument('--methods', type=str, nargs='*', default=None,
                        help='For method_compare only: subset of methods to run (e.g. M2_signalA_CQR M6_full_CQR M7_full_ACI to skip Raw/M0 and reuse E1 results)')
    args = parser.parse_args()

    output_dir = os.path.join(args.output, args.experiment, args.dataset)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    if args.experiment == 'bias_diagnosis':
        experiment_bias_diagnosis(
            args.model, args.dataset, output_dir,
            num_samples=args.num_samples,
            datasets_subset=args.datasets)

    elif args.experiment == 'method_compare':
        experiment_method_compare(
            args.model, args.dataset, output_dir,
            alpha=args.alpha,
            datasets_subset=args.datasets,
            methods=args.methods)

    elif args.experiment == 'ablation':
        experiment_ablation(
            args.model[0], args.dataset, output_dir,
            alpha=args.alpha)

    elif args.experiment == 'cost_efficiency':
        experiment_cost_efficiency(
            args.model[0], args.dataset, output_dir,
            alpha=args.alpha)


if __name__ == '__main__':
    main()
