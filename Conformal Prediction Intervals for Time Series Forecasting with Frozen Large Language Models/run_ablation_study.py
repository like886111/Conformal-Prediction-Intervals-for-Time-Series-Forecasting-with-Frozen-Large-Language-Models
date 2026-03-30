"""
消融实验：A4 联合 + B1 温度 + B2 步长 + C1 ACI γ + C2 信号组合。
用于回答 RQ3（采样策略/校准设定对校准后区间质量的影响）。

顶刊版设计（消融实验完整设计方案_顶刊版.md）：
  - A4: 12 组联合配置，2 模型 × 5 数据集 × 3 种子
  - B1: temperature 0.5/0.7/1.0，2 模型 × 3 数据集 × 3
  - B2: 预测步长 H 10/20/30，2 模型 × 3 数据集 × 3
  - C1: ACI γ 0.001/0.01/0.05，1 模型 × 3 数据集 × 3
  - C2: 信号组合 A / A+B / A+B+D，1 模型 × 3 数据集 × 3

Usage:
  python run_ablation_study.py --experiment A4
  python run_ablation_study.py --experiment B1
  python run_ablation_study.py --experiment B2
  python run_ablation_study.py --experiment C1
  python run_ablation_study.py --experiment C2
  python run_ablation_study.py --paper   # A4 完整
  python run_ablation_study.py --model deepseek-v3 --grid extended \\
    --dataset memorization --datasets TurkeyPower --resume   # 仅补跑失败/缺失格
"""

import argparse
import json
import os
from collections import defaultdict
from copy import deepcopy

from param import set_seed
from run_experiment import MODEL_HYPERS, load_datasets
from uncertainty.pipeline import UncertaintyPipeline, save_results


# ========== A4 联合消融 ==========
# Phase 1 轻量：4 组
ABLATION_GRID_LIGHT = [
    {"num_samples": 15, "cal_ratio": 0.30, "cp_method": "cqr"},
    {"num_samples": 15, "cal_ratio": 0.30, "cp_method": "aci"},
    {"num_samples": 30, "cal_ratio": 0.30, "cp_method": "cqr"},
    {"num_samples": 30, "cal_ratio": 0.30, "cp_method": "aci"},
]

# Phase 1 完整：8 组
ABLATION_GRID_FULL = [
    {"num_samples": 15, "cal_ratio": 0.20, "cp_method": "cqr"},
    {"num_samples": 15, "cal_ratio": 0.20, "cp_method": "aci"},
    {"num_samples": 15, "cal_ratio": 0.30, "cp_method": "cqr"},
    {"num_samples": 15, "cal_ratio": 0.30, "cp_method": "aci"},
    {"num_samples": 30, "cal_ratio": 0.20, "cp_method": "cqr"},
    {"num_samples": 30, "cal_ratio": 0.20, "cp_method": "aci"},
    {"num_samples": 30, "cal_ratio": 0.30, "cp_method": "cqr"},
    {"num_samples": 30, "cal_ratio": 0.30, "cp_method": "aci"},
]

# 顶刊版扩展：12 组（n15/n30/n50, cal20/25/30, CQR/ACI）
ABLATION_GRID_EXTENDED = [
    {"num_samples": 15, "cal_ratio": 0.20, "cp_method": "cqr"},
    {"num_samples": 15, "cal_ratio": 0.20, "cp_method": "aci"},
    {"num_samples": 15, "cal_ratio": 0.30, "cp_method": "cqr"},
    {"num_samples": 15, "cal_ratio": 0.30, "cp_method": "aci"},
    {"num_samples": 30, "cal_ratio": 0.20, "cp_method": "cqr"},
    {"num_samples": 30, "cal_ratio": 0.20, "cp_method": "aci"},
    {"num_samples": 30, "cal_ratio": 0.30, "cp_method": "cqr"},
    {"num_samples": 30, "cal_ratio": 0.30, "cp_method": "aci"},
    {"num_samples": 50, "cal_ratio": 0.25, "cp_method": "cqr"},
    {"num_samples": 50, "cal_ratio": 0.25, "cp_method": "aci"},
    {"num_samples": 30, "cal_ratio": 0.25, "cp_method": "cqr"},
    {"num_samples": 30, "cal_ratio": 0.25, "cp_method": "aci"},
]

# 顶刊版默认数据集（与 RQ2 主表一致）
PAPER_DATASETS = {
    "memorization": ["TSMCStock", "TurkeyPower"],
    "darts": ["WineDataset", "AusBeerDataset", "MonthlyMilkDataset"],
}

# ========== B1 温度消融 ==========
# 固定 n=30, cal_ratio=0.25, CQR
B1_GRID = [
    {"temperature": 0.5, "num_samples": 30, "cal_ratio": 0.25, "cp_method": "cqr"},
    {"temperature": 0.7, "num_samples": 30, "cal_ratio": 0.25, "cp_method": "cqr"},
    {"temperature": 1.0, "num_samples": 30, "cal_ratio": 0.25, "cp_method": "cqr"},
]
B1_DATASETS = [
    ("memorization", ["TSMCStock", "TurkeyPower"]),
    ("darts", ["WineDataset"]),
]

# ========== B2 步长消融 ==========
# 固定 n=30, cal_ratio=0.25, CQR
B2_GRID = [
    {"predict_steps": 10, "num_samples": 30, "cal_ratio": 0.25, "cp_method": "cqr"},
    {"predict_steps": 20, "num_samples": 30, "cal_ratio": 0.25, "cp_method": "cqr"},
    {"predict_steps": 30, "num_samples": 30, "cal_ratio": 0.25, "cp_method": "cqr"},
]
B2_DATASETS = [
    ("memorization", ["TSMCStock", "TurkeyPower"]),
    ("darts", ["WineDataset"]),
]

# ========== C1 ACI γ 消融 ==========
# 固定 n=30, cal_ratio=0.25, ACI
C1_GRID = [
    {"aci_gamma": 0.001, "num_samples": 30, "cal_ratio": 0.25, "cp_method": "aci"},
    {"aci_gamma": 0.01, "num_samples": 30, "cal_ratio": 0.25, "cp_method": "aci"},
    {"aci_gamma": 0.05, "num_samples": 30, "cal_ratio": 0.25, "cp_method": "aci"},
]
C1_DATASETS = [
    ("memorization", ["TSMCStock", "TurkeyPower"]),
    ("darts", ["WineDataset"]),
]

# ========== C2 信号组合消融 ==========
# 固定 n=30, cal_ratio=0.25, CQR
C2_GRID = [
    {"signals_to_use": ("A",), "num_samples": 30, "cal_ratio": 0.25, "cp_method": "cqr"},
    {"signals_to_use": ("A", "B"), "num_samples": 30, "cal_ratio": 0.25, "cp_method": "cqr"},
    {"signals_to_use": ("A", "B", "D"), "num_samples": 30, "cal_ratio": 0.25, "cp_method": "cqr"},
]
C2_DATASETS = [
    ("memorization", ["TSMCStock", "TurkeyPower"]),
    ("darts", ["WineDataset"]),
]


def config_tag(c):
    """A4 配置标签"""
    return f"n{c['num_samples']}_cal{int(c['cal_ratio']*100)}_{c['cp_method']}"


def _ablation_cell_succeeded(cell) -> bool:
    """summary 中一格是否视为已成功（跳过 API 重跑）。"""
    if not cell or not isinstance(cell, dict):
        return False
    if cell.get("error"):
        return False
    return "ECP" in cell


def experiment_config_tag(c, experiment_type):
    """各实验类型的配置标签"""
    if experiment_type == "B1":
        return f"temp{c['temperature']}"
    if experiment_type == "B2":
        return f"h{c['predict_steps']}"
    if experiment_type == "C1":
        return f"aci_gamma{c['aci_gamma']}"
    if experiment_type == "C2":
        return "signals_" + "_".join(c["signals_to_use"])
    return config_tag(c)


def _get_grid(grid_mode: str, light: bool) -> list:
    """根据 grid_mode 和 light 返回消融配置列表。"""
    if light:
        return ABLATION_GRID_LIGHT
    if grid_mode == "extended":
        return ABLATION_GRID_EXTENDED
    return ABLATION_GRID_FULL


def _build_pipeline(model, hypers, cfg, alpha, experiment_type):
    """根据配置构建 UncertaintyPipeline。"""
    hypers_use = deepcopy(hypers)
    if "temperature" in cfg:
        hypers_use["temp"] = cfg["temperature"]
    signals = cfg.get("signals_to_use", ("A", "B", "D"))
    aci_gamma = cfg.get("aci_gamma", 0.005)
    return UncertaintyPipeline(
        model=model,
        hypers=hypers_use,
        signals_to_use=signals,
        cp_method=cfg["cp_method"],
        weight_method="blue",
        alpha=alpha,
        aci_gamma=aci_gamma,
        signal_A_kwargs={"num_samples": cfg["num_samples"]},
    )


def _run_one_config(pipe, train, test, cfg, cal_ratio, model, ds_name, tag, out_dir,
                    experiment_tag_suffix, seed, summary, experiment_type):
    """运行单配置并保存结果。"""
    try:
        result = pipe.run(train, test, cal_ratio=cal_ratio)
        report = result.get("report", {})
        report_clean = {}
        for k, v in report.items():
            if isinstance(v, (int, float, str, bool)):
                report_clean[k] = v
            elif isinstance(v, dict) and k != "hypothesis_tests":
                report_clean[k] = {
                    sk: sv for sk, sv in v.items()
                    if isinstance(sv, (int, float, str, bool, list))
                }
            elif isinstance(v, (list,)) and not (len(v) > 0 and isinstance(v[0], (dict,))):
                try:
                    report_clean[k] = list(v)
                except Exception:
                    pass
        report_clean["_config"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()}
        if seed is not None:
            report_clean["_seed"] = seed
        summary[ds_name][tag] = report_clean
        exp_tag = f"ablation_{experiment_type}_{tag}" if experiment_type else f"ablation_{tag}"
        save_results(
            result,
            output_dir=out_dir,
            dataset_name=ds_name,
            model_name=model,
            experiment_tag=f"{exp_tag}{experiment_tag_suffix}",
        )
    except Exception as e:
        import traceback
        print(f"  FAILED: {e}")
        traceback.print_exc()
        summary[ds_name][tag] = {"error": str(e), "_config": dict(cfg)}
        if seed is not None:
            summary[ds_name][tag]["_seed"] = seed


def run_ablation_study(
    model: str,
    dataset_group: str,
    output_dir: str,
    datasets_subset: list = None,
    light: bool = False,
    grid_mode: str = "full",
    alpha: float = 0.05,
    seed: int = None,
    experiment_tag_suffix: str = "",
    resume: bool = False,
) -> dict:
    """
    运行 A4 消融：固定模型，在指定数据集上变化 num_samples、cal_ratio、cp_method。
    返回 summary[dataset_name][config_tag] = report (含 ECP, NAIW, Winkler 等)。

    resume=True 时：若存在同名 ablation_summary_*.json，则先读入；对本次遍历到的
    (dataset, config) 若已有成功格（含 ECP 且无 error）则跳过，否则重跑；未在本次
    --datasets 中的数据集条目仍保留在最终 JSON 中。
    """
    hypers = MODEL_HYPERS.get(model)
    if hypers is None:
        raise ValueError(f"No hypers for model '{model}' in MODEL_HYPERS.")

    datasets = load_datasets(dataset_group)
    if datasets_subset:
        datasets = {k: v for k, v in datasets.items() if k in datasets_subset}
    if not datasets:
        raise ValueError(f"No datasets left after subset {datasets_subset}.")

    grid = _get_grid(grid_mode, light)
    out_dir = os.path.join(output_dir, "ablation", dataset_group)
    os.makedirs(out_dir, exist_ok=True)

    suffix = f"_seed{seed}" if seed is not None else ""
    suffix += experiment_tag_suffix
    summary_path = os.path.join(out_dir, f"ablation_summary_{model}_{dataset_group}{suffix}.json")

    summary = defaultdict(dict)
    if resume and os.path.isfile(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        for ds_name, configs in loaded.items():
            summary[ds_name] = dict(configs)
        print(f"[Ablation] Resume: loaded existing summary ({len(loaded)} datasets) from\n  {summary_path}")

    if seed is not None:
        set_seed(seed)

    for ds_name, (train, test) in datasets.items():
        for cfg in grid:
            tag = config_tag(cfg)
            if resume and _ablation_cell_succeeded(summary.get(ds_name, {}).get(tag)):
                print(f"\n[SKIP] A4 {model} × {ds_name} × {tag} (already successful in summary)")
                continue
            print("\n" + "=" * 60)
            print(f"  Ablation A4: {model} × {ds_name} × {tag}" + (f" (seed={seed})" if seed is not None else ""))
            print("=" * 60)
            pipe = _build_pipeline(model, hypers, cfg, alpha, "A4")
            _run_one_config(
                pipe, train, test, cfg, cfg["cal_ratio"], model, ds_name, tag,
                out_dir, experiment_tag_suffix, seed, summary, "",
            )
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(dict(summary), f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[Ablation] Summary saved to {summary_path}")
    return dict(summary)


def run_ablation_b1(model, output_dir, alpha=0.05, seed=None, experiment_tag_suffix=""):
    """B1: temperature 消融，2 模型 × 3 数据集 × 3 温度"""
    return _run_ablation_extra(
        model, B1_GRID, B1_DATASETS, "B1", output_dir, alpha, seed, experiment_tag_suffix,
    )


def run_ablation_b2(model, output_dir, alpha=0.05, seed=None, experiment_tag_suffix=""):
    """B2: 预测步长消融，2 模型 × 3 数据集 × 3 步长"""
    return _run_ablation_extra(
        model, B2_GRID, B2_DATASETS, "B2", output_dir, alpha, seed, experiment_tag_suffix,
        load_per_config=True,
    )


def run_ablation_c1(model, output_dir, alpha=0.05, seed=None, experiment_tag_suffix=""):
    """C1: ACI γ 消融，1 模型 × 3 数据集 × 3 γ"""
    return _run_ablation_extra(
        model, C1_GRID, C1_DATASETS, "C1", output_dir, alpha, seed, experiment_tag_suffix,
    )


def run_ablation_c2(model, output_dir, alpha=0.05, seed=None, experiment_tag_suffix=""):
    """C2: 信号组合消融，1 模型 × 3 数据集 × 3 组合"""
    return _run_ablation_extra(
        model, C2_GRID, C2_DATASETS, "C2", output_dir, alpha, seed, experiment_tag_suffix,
    )


def _run_ablation_extra(model, grid, dataset_specs, experiment_type, output_dir,
                        alpha=0.05, seed=None, experiment_tag_suffix="", load_per_config=False):
    """
    运行 B1/B2/C1/C2 消融。
    dataset_specs: [(dataset_group, [ds_names]), ...]
    load_per_config: B2 需按 predict_steps 重新加载数据
    """
    hypers = MODEL_HYPERS.get(model)
    if hypers is None:
        raise ValueError(f"No hypers for model '{model}' in MODEL_HYPERS.")

    if seed is not None:
        set_seed(seed)

    all_summary = defaultdict(dict)
    for dg, ds_list in dataset_specs:
        out_dir = os.path.join(output_dir, "ablation", experiment_type.lower(), dg)
        os.makedirs(out_dir, exist_ok=True)

        for cfg in grid:
            if load_per_config and "predict_steps" in cfg:
                datasets = load_datasets(dg, predict_steps=cfg["predict_steps"])
            else:
                datasets = load_datasets(dg)
            datasets = {k: v for k, v in datasets.items() if k in ds_list}
            if not datasets:
                continue

            for ds_name, (train, test) in datasets.items():
                tag = experiment_config_tag(cfg, experiment_type)
                print("\n" + "=" * 60)
                print(f"  Ablation {experiment_type}: {model} × {ds_name} × {tag}" + (f" (seed={seed})" if seed else ""))
                print("=" * 60)
                pipe = _build_pipeline(model, hypers, cfg, alpha, experiment_type)
                _run_one_config(
                    pipe, train, test, cfg, cfg["cal_ratio"], model, ds_name, tag,
                    out_dir, experiment_tag_suffix, seed, all_summary, experiment_type,
                )

        suffix = f"_seed{seed}" if seed else ""
        suffix += experiment_tag_suffix
        summary_path = os.path.join(out_dir, f"ablation_{experiment_type.lower()}_{model}_{dg}{suffix}.json")
        subset = {ds: configs for ds, configs in all_summary.items() if ds in ds_list}
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(dict(subset), f, indent=2, ensure_ascii=False, default=str)
        print(f"\n[Ablation {experiment_type}] Summary saved to {summary_path}")

    return dict(all_summary)


def main():
    parser = argparse.ArgumentParser(description="消融实验：A4/B1/B2/C1/C2")
    parser.add_argument("--model", type=str, default="deepseek-v3", help="Model name (single)")
    parser.add_argument(
        "--models",
        type=str,
        nargs="*",
        default=None,
        help="Multiple models (e.g. deepseek-v3 qwen-plus). Overrides --model when set.",
    )
    parser.add_argument("--dataset", type=str, default="memorization", help="Dataset group (A4)")
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="*",
        default=None,
        help="Subset of dataset names (e.g. TSMCStock TurkeyPower). Default: all in group.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/",
        help="Base output directory",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        choices=["A4", "B1", "B2", "C1", "C2"],
        default="A4",
        help="A4=联合消融, B1=温度, B2=步长, C1=ACIγ, C2=信号组合",
    )
    parser.add_argument(
        "--grid",
        type=str,
        choices=["light", "full", "extended"],
        default="full",
        help="A4 only: light=4, full=8, extended=12 configs",
    )
    parser.add_argument(
        "--light",
        action="store_true",
        help="A4 only: Use smaller grid (4 configs). Overrides --grid.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of random seeds (3 for paper: 42, 123, 456)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed; when repeat>1, uses seed, seed+81, seed+162 (42,123,456 if seed=42)",
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        help="顶刊版 A4 完整消融: 2 models × 5 datasets × 12 configs × 3 seeds",
    )
    parser.add_argument("--alpha", type=float, default=0.05, help="Nominal miscoverage")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="A4 / --paper: 读取已有 ablation_summary_*.json，跳过已有 ECP 且无 error 的格，"
        "仅重跑失败或缺失；保留汇总中未出现在本次 --datasets 里的数据集。",
    )
    args = parser.parse_args()

    PAPER_SEEDS = [42, 123, 456]

    if args.paper:
        models = ["deepseek-v3", "qwen-plus"]
        for dg, ds_list in PAPER_DATASETS.items():
            for model in models:
                for seed in (PAPER_SEEDS if args.repeat >= 3 else [None]):
                    print(f"\n{'#' * 60}\n# Paper A4: {model} × {dg} × {ds_list} (seed={seed})\n{'#' * 60}")
                    run_ablation_study(
                        model=model,
                        dataset_group=dg,
                        output_dir=args.output,
                        datasets_subset=ds_list,
                        light=False,
                        grid_mode="extended",
                        alpha=args.alpha,
                        seed=seed,
                        experiment_tag_suffix=f"_seed{seed}" if seed is not None else "",
                        resume=args.resume,
                    )
        return

    # B1/B2: 2 models; C1/C2: 1 model
    if args.experiment in ("B1", "B2"):
        models = args.models if args.models else [args.model, "qwen-plus"]
        if len(models) == 1:
            models = [models[0], "qwen-plus"]
    else:
        models = args.models if args.models else [args.model]

    seeds = [None]
    if args.repeat > 1:
        seeds = PAPER_SEEDS if (args.repeat == 3 and args.seed == 42) else [args.seed + i * 81 for i in range(args.repeat)]
    elif args.seed is not None:
        seeds = [args.seed]

    for model in models:
        for seed in seeds:
            suf = f"_seed{seed}" if seed is not None else ""
            if args.experiment == "A4":
                run_ablation_study(
                    model=model,
                    dataset_group=args.dataset,
                    output_dir=args.output,
                    datasets_subset=args.datasets,
                    light=args.light,
                    grid_mode="light" if args.light else args.grid,
                    alpha=args.alpha,
                    seed=seed,
                    experiment_tag_suffix=suf,
                    resume=args.resume,
                )
            elif args.experiment == "B1":
                run_ablation_b1(model, args.output, args.alpha, seed, suf)
            elif args.experiment == "B2":
                run_ablation_b2(model, args.output, args.alpha, seed, suf)
            elif args.experiment == "C1":
                run_ablation_c1(model, args.output, args.alpha, seed, suf)
            elif args.experiment == "C2":
                run_ablation_c2(model, args.output, args.alpha, seed, suf)


if __name__ == "__main__":
    main()
