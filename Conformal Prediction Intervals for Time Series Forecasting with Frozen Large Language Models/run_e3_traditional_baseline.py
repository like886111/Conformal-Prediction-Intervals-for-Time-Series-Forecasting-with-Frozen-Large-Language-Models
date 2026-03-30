"""
E3 传统基线对比：同一划分下 校准后 LLM vs 传统方法原生区间

支持方法：ARIMA, ETS, NAIVE, SEASONAL_NAIVE, PROPHET, LIGHTGBM, N-BEATS。
  统计/ML（无 torch）：ARIMA/ETS/NAIVE/SEASONAL_NAIVE 需 statsmodels；PROPHET 需 prophet；LIGHTGBM 需 lightgbm；N-BEATS 需 darts/torch。

用法:
  python run_e3_traditional_baseline.py --dataset-group darts --output-dir results/e3_traditional_baseline
  python run_e3_traditional_baseline.py --dataset-group memorization --methods ARIMA,ETS,NAIVE,SEASONAL_NAIVE,PROPHET,LIGHTGBM
  python run_e3_traditional_baseline.py --dataset-group darts --methods ARIMA,ETS,PROPHET,LIGHTGBM

与 E2 对齐：使用相同 train/test 划分（predict_steps=30），
在测试段直接使用各传统方法的原生置信/预测区间（不加 CQR），计算 ECP、Coverage_Gap、NAIW、Winkler，
便于与 方法对比实验结果汇总表 中校准后的 LLM 并列比较。
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd

# 与 run_experiment 一致的数据加载
from run_experiment import load_datasets, PAPER_HORIZON
from uncertainty.evaluator import UncertaintyEvaluator

try:
    from statsmodels.tsa.arima.model import ARIMA
except ImportError:
    ARIMA = None
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
except ImportError:
    ExponentialSmoothing = None
try:
    from prophet import Prophet
except ImportError:
    Prophet = None
# LightGBM 在 lightgbm_forecast_with_interval 内按需 import，避免未安装时影响其他方法

# darts/N-BEATS 不在此处 import，避免触发 torch/torchvision 版本冲突；
# 仅在选择 NBEATS 时在 nbeats_forecast_with_interval() 内懒加载


def _ensure_statsmodels():
    if ARIMA is None:
        raise ImportError("E3 需要 statsmodels。请安装: pip install statsmodels")


def arima_forecast_with_interval(series, steps, alpha=0.05, order=(1, 0, 1)):
    """
    在 series 上拟合 ARIMA，预测 steps 步，返回 (point, lower, upper)。
    """
    _ensure_statsmodels()
    y = np.asarray(series).astype(float)
    if np.any(np.isnan(y)):
        y = np.nan_to_num(y, nan=np.nanmean(y))
    try:
        model = ARIMA(y, order=order)
        fitted = model.fit()
    except Exception:
        # 降级为更简单模型
        for ord in [(1, 0, 0), (0, 1, 0), (1, 1, 0)]:
            try:
                model = ARIMA(y, order=ord)
                fitted = model.fit()
                break
            except Exception:
                continue
        else:
            raise RuntimeError("ARIMA 拟合失败")
    f = fitted.get_forecast(steps=steps)
    point = np.asarray(f.predicted_mean).ravel()
    ci = f.conf_int(alpha=alpha)
    # conf_int 可能返回 DataFrame 或 ndarray
    ci_arr = np.asarray(ci)
    if ci_arr.ndim == 1:
        ci_arr = np.reshape(ci_arr, (-1, 2))
    lower = ci_arr[:, 0].ravel()
    upper = ci_arr[:, 1].ravel()
    return point, lower, upper


def ets_forecast_with_interval(series, steps, alpha=0.05, repetitions=500, random_state=42):
    """
    在 series 上拟合 ETS（指数平滑），预测 steps 步，用 simulate 得到原生区间，返回 (point, lower, upper)。
    """
    if ExponentialSmoothing is None:
        raise ImportError("ETS 需要 statsmodels。请安装: pip install statsmodels")
    y = np.asarray(series).astype(float)
    if np.any(np.isnan(y)):
        y = np.nan_to_num(y, nan=np.nanmean(y))
    # 尽量用加法趋势；序列太短或拟合失败时用简单指数平滑
    try:
        model = ExponentialSmoothing(y, trend="add", seasonal=None)
        fitted = model.fit()
    except Exception:
        try:
            model = ExponentialSmoothing(y, trend=None, seasonal=None)
            fitted = model.fit()
        except Exception:
            raise RuntimeError("ETS 拟合失败")
    point = fitted.forecast(steps=steps)
    sim = fitted.simulate(
        nsimulations=steps, anchor="end", repetitions=repetitions, random_state=random_state
    )
    sim_arr = np.asarray(sim)
    if sim_arr.ndim == 1:
        sim_arr = sim_arr.reshape(-1, 1)
    q_lo = float(alpha / 2) * 100
    q_hi = (1 - float(alpha) / 2) * 100
    lower = np.percentile(sim_arr, q_lo, axis=1)
    upper = np.percentile(sim_arr, q_hi, axis=1)
    return np.asarray(point), lower, upper


def naive_forecast_with_interval(series, steps, alpha=0.05):
    """
    Naive（持久化）：点预测 = 最后一个观测值重复；区间由训练段一步前向残差的分位数得到（原生、无 CQR）。
    """
    y = np.asarray(series).astype(float).ravel()
    if np.any(np.isnan(y)):
        y = np.nan_to_num(y, nan=np.nanmean(y))
    point = np.full(steps, float(y[-1]))
    res = y[1:] - y[:-1]
    if len(res) < 2:
        res = np.array([0.0])
    q_lo = np.percentile(res, float(alpha / 2) * 100)
    q_hi = np.percentile(res, (1 - float(alpha) / 2) * 100)
    lower = point + q_lo
    upper = point + q_hi
    return point, lower, upper


def prophet_forecast_with_interval(series, steps, alpha=0.05):
    """
    在 series 上拟合 Prophet，预测 steps 步，使用模型原生不确定性区间，返回 (point, lower, upper)。
    需要 pip install prophet。
    """
    if Prophet is None:
        raise ImportError("Prophet 需要 prophet。请安装: pip install prophet")
    y = np.asarray(series).astype(float).ravel()
    if np.any(np.isnan(y)):
        y = np.nan_to_num(y, nan=np.nanmean(y))
    n = len(y)
    if n < 10:
        raise ValueError("Prophet 需要至少 10 个观测")
    # Prophet 要求 ds 为日期，用日频虚构时间索引
    ds = pd.date_range(start="2000-01-01", periods=n, freq="D")
    df = pd.DataFrame({"ds": ds, "y": y})
    interval_width = 1.0 - float(alpha)
    model = Prophet(interval_width=interval_width, yearly_seasonality=True, daily_seasonality=False)
    model.fit(df)
    future = model.make_future_dataframe(periods=steps, freq="D")
    forecast = model.predict(future)
    # 取最后 steps 行
    point = np.asarray(forecast["yhat"].values[-steps:]).ravel()
    lower = np.asarray(forecast["yhat_lower"].values[-steps:]).ravel()
    upper = np.asarray(forecast["yhat_upper"].values[-steps:]).ravel()
    return point, lower, upper


def lightgbm_forecast_with_interval(series, steps, alpha=0.05, n_lags=24, random_state=42):
    """
    用滞后特征 + LightGBM 分位数回归得到点预测与区间，返回 (point, lower, upper)。
    需要 pip install lightgbm。
    """
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError("LightGBM 需要 lightgbm。请安装: pip install lightgbm")
    y = np.asarray(series).astype(float).ravel()
    if np.any(np.isnan(y)):
        y = np.nan_to_num(y, nan=np.nanmean(y))
    n = len(y)
    n_lags = min(n_lags, max(2, n // 2 - 1))
    if n < n_lags + 5:
        raise ValueError("序列过短，无法拟合 LightGBM 滞后特征")
    # 构建滞后特征矩阵，使用带列名的 DataFrame 避免 sklearn 的 feature names 警告
    col_names = [f"lag_{i}" for i in range(n_lags)]
    X_list, y_med_list, y_lo_list, y_hi_list = [], [], [], []
    for t in range(n_lags, n):
        X_list.append(y[t - n_lags : t])
        y_med_list.append(y[t])
        y_lo_list.append(y[t])
        y_hi_list.append(y[t])
    X = pd.DataFrame(np.array(X_list), columns=col_names)
    y_med = np.array(y_med_list)
    y_lo = np.array(y_lo_list)
    y_hi = np.array(y_hi_list)
    q_lo = float(alpha / 2)
    q_hi = 1.0 - float(alpha) / 2
    base_params = {"verbosity": -1, "random_state": random_state, "n_estimators": 100}
    # 中位数模型：普通回归
    model_med = lgb.LGBMRegressor(objective="regression", **base_params).fit(X, y_med)
    # 下、上分位数模型：quantile 目标，分别用 q_lo / q_hi
    model_lo = lgb.LGBMRegressor(objective="quantile", alpha=q_lo, **base_params).fit(X, y_lo)
    model_hi = lgb.LGBMRegressor(objective="quantile", alpha=q_hi, **base_params).fit(X, y_hi)
    point = np.empty(steps)
    lower = np.empty(steps)
    upper = np.empty(steps)
    hist = list(y[-n_lags:])
    for s in range(steps):
        x = pd.DataFrame(np.array(hist[-n_lags:]).reshape(1, -1), columns=col_names)
        point[s] = model_med.predict(x)[0]
        lower[s] = model_lo.predict(x)[0]
        upper[s] = model_hi.predict(x)[0]
        hist.append(float(point[s]))
    return point, lower, upper


def seasonal_naive_forecast_with_interval(series, steps, alpha=0.05, period=12):
    """
    Seasonal Naive：点预测 = 上一周期同位置；区间由训练段季节残差的分位数得到（原生、无 CQR）。
    """
    y = np.asarray(series).astype(float).ravel()
    if np.any(np.isnan(y)):
        y = np.nan_to_num(y, nan=np.nanmean(y))
    n = len(y)
    period = min(period, n // 2) if n >= 2 else 1
    point = np.array([y[-(period - (i % period))] for i in range(steps)], dtype=float)
    res = y[period:] - y[:-period] if n > period else np.array([0.0])
    if len(res) < 2:
        res = np.array([0.0])
    q_lo = np.percentile(res, float(alpha / 2) * 100)
    q_hi = np.percentile(res, (1 - float(alpha) / 2) * 100)
    lower = point + q_lo
    upper = point + q_hi
    return point, lower, upper


def nbeats_forecast_with_interval(
    series, steps, alpha=0.05, num_samples=200,
    input_chunk_length=24, output_chunk_length=16, n_epochs=40, random_state=42,
):
    """
    在 series 上拟合 N-BEATS，用 MC dropout 得到采样，取分位数作为原生区间，返回 (point, lower, upper)。
    darts/torch 仅在调用时导入，避免与仅跑 ARIMA/ETS 时的环境冲突。
    """
    try:
        from darts import TimeSeries
        from darts.models import NBEATSModel
    except ImportError as e:
        raise ImportError(
            "N-BEATS 需要 darts/torch。请安装: pip install darts torch"
        ) from e
    except AttributeError as e:
        if "register_fake" in str(e) or "torch" in str(e).lower():
            raise RuntimeError(
                "N-BEATS 依赖的 torch/torchvision 版本与当前环境不兼容（如 torch.library.register_fake）。"
                "可仅跑传统统计基线: --methods ARIMA,ETS"
            ) from e
        raise
    y = np.asarray(series).astype(float)
    if np.any(np.isnan(y)):
        y = np.nan_to_num(y, nan=np.nanmean(y))
    train_ts = TimeSeries.from_values(y)
    in_len = min(input_chunk_length, len(y) // 2)
    out_len = min(output_chunk_length, steps)
    if in_len < 2 or out_len < 1:
        raise ValueError("序列过短，无法拟合 N-BEATS")
    model = NBEATSModel(
        input_chunk_length=in_len,
        output_chunk_length=out_len,
        n_epochs=n_epochs,
        dropout=0.2,
        random_state=random_state,
        pl_trainer_kwargs={"enable_progress_bar": False},
    )
    model.fit(train_ts)
    pred = model.predict(n=steps, num_samples=num_samples, mc_dropout=True)
    arr = pred.values()
    if arr.ndim == 3:
        # (time, component, samples)
        point = np.mean(arr, axis=2).flatten()
        lower = np.percentile(arr, float(alpha / 2) * 100, axis=2).flatten()
        upper = np.percentile(arr, (1 - float(alpha) / 2) * 100, axis=2).flatten()
    else:
        point = arr.flatten()
        lower = point - 1.96 * np.std(arr, axis=-1).flatten() if arr.ndim > 1 else point - 0
        upper = point + 1.96 * np.std(arr, axis=-1).flatten() if arr.ndim > 1 else point + 0
    return point[:steps], lower[:steps], upper[:steps]


def run_arima_one_dataset(train, test, horizon, alpha=0.05):
    """
    train/test 与 run_experiment 一致（test 长度 = horizon）。
    在完整 train 上拟合 ARIMA，直接对 test 段做多步预测并使用其原生置信区间。
    """
    # 统一转为 1d ndarray，兼容 Series / DataFrame / ndarray
    train_vals = np.asarray(train).ravel()
    test_vals = np.asarray(test).ravel()

    # 在完整 train 上预测 test 段，使用 statsmodels 给出的原生区间
    _, lower_test, upper_test = arima_forecast_with_interval(
        train_vals, steps=horizon, alpha=alpha
    )

    evaluator = UncertaintyEvaluator()
    ecp = evaluator.empirical_coverage(test_vals, lower_test, upper_test)
    gap = evaluator.coverage_gap(test_vals, lower_test, upper_test, alpha)
    naiw = evaluator.normalized_average_width(lower_test, upper_test, test_vals)
    winkler = evaluator.winkler_score(test_vals, lower_test, upper_test, alpha=alpha)

    return {
        "ECP": ecp,
        "Coverage_Gap": gap,
        "NAIW": naiw,
        "Winkler": winkler,
    }


def run_ets_one_dataset(train, test, horizon, alpha=0.05):
    """ETS 原生区间，与 run_arima_one_dataset 接口一致。"""
    train_vals = np.asarray(train).ravel()
    test_vals = np.asarray(test).ravel()
    _, lower_test, upper_test = ets_forecast_with_interval(
        train_vals, steps=horizon, alpha=alpha
    )
    evaluator = UncertaintyEvaluator()
    ecp = evaluator.empirical_coverage(test_vals, lower_test, upper_test)
    gap = evaluator.coverage_gap(test_vals, lower_test, upper_test, alpha)
    naiw = evaluator.normalized_average_width(lower_test, upper_test, test_vals)
    winkler = evaluator.winkler_score(test_vals, lower_test, upper_test, alpha=alpha)
    return {"ECP": ecp, "Coverage_Gap": gap, "NAIW": naiw, "Winkler": winkler}


def run_naive_one_dataset(train, test, horizon, alpha=0.05):
    """Naive（持久化）原生区间。"""
    train_vals = np.asarray(train).ravel()
    test_vals = np.asarray(test).ravel()
    _, lower_test, upper_test = naive_forecast_with_interval(
        train_vals, steps=horizon, alpha=alpha
    )
    evaluator = UncertaintyEvaluator()
    ecp = evaluator.empirical_coverage(test_vals, lower_test, upper_test)
    gap = evaluator.coverage_gap(test_vals, lower_test, upper_test, alpha)
    naiw = evaluator.normalized_average_width(lower_test, upper_test, test_vals)
    winkler = evaluator.winkler_score(test_vals, lower_test, upper_test, alpha=alpha)
    return {"ECP": ecp, "Coverage_Gap": gap, "NAIW": naiw, "Winkler": winkler}


def run_seasonal_naive_one_dataset(train, test, horizon, alpha=0.05, period=12):
    """Seasonal Naive 原生区间（默认 period=12 适用于月度）。"""
    train_vals = np.asarray(train).ravel()
    test_vals = np.asarray(test).ravel()
    _, lower_test, upper_test = seasonal_naive_forecast_with_interval(
        train_vals, steps=horizon, alpha=alpha, period=period
    )
    evaluator = UncertaintyEvaluator()
    ecp = evaluator.empirical_coverage(test_vals, lower_test, upper_test)
    gap = evaluator.coverage_gap(test_vals, lower_test, upper_test, alpha)
    naiw = evaluator.normalized_average_width(lower_test, upper_test, test_vals)
    winkler = evaluator.winkler_score(test_vals, lower_test, upper_test, alpha=alpha)
    return {"ECP": ecp, "Coverage_Gap": gap, "NAIW": naiw, "Winkler": winkler}


def run_nbeats_one_dataset(train, test, horizon, alpha=0.05):
    """N-BEATS 原生区间（MC dropout 采样），与 run_arima_one_dataset 接口一致。"""
    train_vals = np.asarray(train).ravel()
    test_vals = np.asarray(test).ravel()
    _, lower_test, upper_test = nbeats_forecast_with_interval(
        train_vals, steps=horizon, alpha=alpha
    )
    evaluator = UncertaintyEvaluator()
    ecp = evaluator.empirical_coverage(test_vals, lower_test, upper_test)
    gap = evaluator.coverage_gap(test_vals, lower_test, upper_test, alpha)
    naiw = evaluator.normalized_average_width(lower_test, upper_test, test_vals)
    winkler = evaluator.winkler_score(test_vals, lower_test, upper_test, alpha=alpha)
    return {"ECP": ecp, "Coverage_Gap": gap, "NAIW": naiw, "Winkler": winkler}


def run_prophet_one_dataset(train, test, horizon, alpha=0.05):
    """Prophet 原生区间，与 run_arima_one_dataset 接口一致。"""
    train_vals = np.asarray(train).ravel()
    test_vals = np.asarray(test).ravel()
    _, lower_test, upper_test = prophet_forecast_with_interval(
        train_vals, steps=horizon, alpha=alpha
    )
    evaluator = UncertaintyEvaluator()
    ecp = evaluator.empirical_coverage(test_vals, lower_test, upper_test)
    gap = evaluator.coverage_gap(test_vals, lower_test, upper_test, alpha)
    naiw = evaluator.normalized_average_width(lower_test, upper_test, test_vals)
    winkler = evaluator.winkler_score(test_vals, lower_test, upper_test, alpha=alpha)
    return {"ECP": ecp, "Coverage_Gap": gap, "NAIW": naiw, "Winkler": winkler}


def run_lightgbm_one_dataset(train, test, horizon, alpha=0.05):
    """LightGBM 分位数回归原生区间，与 run_arima_one_dataset 接口一致。"""
    train_vals = np.asarray(train).ravel()
    test_vals = np.asarray(test).ravel()
    _, lower_test, upper_test = lightgbm_forecast_with_interval(
        train_vals, steps=horizon, alpha=alpha
    )
    evaluator = UncertaintyEvaluator()
    ecp = evaluator.empirical_coverage(test_vals, lower_test, upper_test)
    gap = evaluator.coverage_gap(test_vals, lower_test, upper_test, alpha)
    naiw = evaluator.normalized_average_width(lower_test, upper_test, test_vals)
    winkler = evaluator.winkler_score(test_vals, lower_test, upper_test, alpha=alpha)
    return {"ECP": ecp, "Coverage_Gap": gap, "NAIW": naiw, "Winkler": winkler}


def run_one_method(method_name, train, test, horizon, alpha):
    """统一入口：按 method 调用 ARIMA / ETS / NAIVE / SEASONAL_NAIVE / PROPHET / LIGHTGBM / NBEATS。"""
    if method_name == "ARIMA":
        return run_arima_one_dataset(train, test, horizon, alpha)
    if method_name == "ETS":
        return run_ets_one_dataset(train, test, horizon, alpha)
    if method_name == "NAIVE":
        return run_naive_one_dataset(train, test, horizon, alpha)
    if method_name == "SEASONAL_NAIVE":
        return run_seasonal_naive_one_dataset(train, test, horizon, alpha)
    if method_name == "PROPHET":
        return run_prophet_one_dataset(train, test, horizon, alpha)
    if method_name == "LIGHTGBM":
        return run_lightgbm_one_dataset(train, test, horizon, alpha)
    if method_name == "NBEATS":
        return run_nbeats_one_dataset(train, test, horizon, alpha)
    raise ValueError(
        f"未知方法: {method_name}，支持 ARIMA, ETS, NAIVE, SEASONAL_NAIVE, PROPHET, LIGHTGBM, NBEATS"
    )


def main():
    parser = argparse.ArgumentParser(
        description="E3 传统基线对比：ARIMA / ETS / Naive / Seasonal Naive / Prophet / LightGBM / N-BEATS 原生区间"
    )
    parser.add_argument(
        "--dataset-group",
        type=str,
        default="darts",
        choices=["darts", "memorization"],
        help="与 E2 一致：darts 或 memorization",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/e3_traditional_baseline",
        help="结果目录",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="名义误覆盖水平（95%% 区间）",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="*",
        default=None,
        help="可选：只跑指定数据集",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="ARIMA,ETS,NAIVE,SEASONAL_NAIVE,PROPHET,LIGHTGBM",
        help="逗号分隔的传统方法：ARIMA, ETS, NAIVE, SEASONAL_NAIVE, PROPHET, LIGHTGBM, NBEATS（默认不含 NBEATS 以免 torch 依赖）",
    )
    args = parser.parse_args()

    ALL_METHODS = ("ARIMA", "ETS", "NAIVE", "SEASONAL_NAIVE", "PROPHET", "LIGHTGBM", "NBEATS")
    methods = [m.strip().upper() for m in args.methods.split(",") if m.strip()]
    for m in methods:
        if m not in ALL_METHODS:
            raise ValueError(f"不支持的方法: {m}，仅支持 {', '.join(ALL_METHODS)}")
    if "ARIMA" in methods or "ETS" in methods:
        _ensure_statsmodels()
    if "PROPHET" in methods and Prophet is None:
        raise ImportError("已选择 PROPHET 但未安装 prophet。请安装: pip install prophet")
    # N-BEATS 在首次运行时再导入 darts，避免 torch/torchvision 版本冲突

    kwargs = {}
    if args.dataset_group == "darts":
        kwargs["predict_steps"] = PAPER_HORIZON
    elif args.dataset_group == "memorization":
        kwargs["predict_steps"] = PAPER_HORIZON

    datasets = load_datasets(args.dataset_group, **kwargs)
    if args.datasets:
        datasets = {k: v for k, v in datasets.items() if k in args.datasets}

    os.makedirs(args.output_dir, exist_ok=True)
    horizon = PAPER_HORIZON
    alpha = args.alpha

    rows = []
    for ds_name, (train, test) in datasets.items():
        test_values = np.asarray(test).ravel()
        h = len(test_values)
        for method_name in methods:
            print(f"  E3 {method_name}: {ds_name} (horizon={h})")
            try:
                out = run_one_method(method_name, train, test, horizon=h, alpha=alpha)
                row = {
                    "dataset": ds_name,
                    "method": method_name,
                    "ECP(95%)": round(out["ECP"], 3),
                    "Coverage_Gap": round(out["Coverage_Gap"], 3),
                    "NAIW": round(out["NAIW"], 3),
                    "Winkler": round(out["Winkler"], 3),
                }
                rows.append(row)
                with open(
                    os.path.join(
                        args.output_dir,
                        f"e3_{args.dataset_group}_{ds_name}_{method_name}.json",
                    ),
                    "w",
                ) as f:
                    json.dump(
                        {**out, "dataset": ds_name, "method": method_name},
                        f,
                        indent=2,
                    )
            except Exception as e:
                print(f"    FAILED: {e}")
                rows.append({
                    "dataset": ds_name,
                    "method": method_name,
                    "ECP(95%)": None,
                    "Coverage_Gap": None,
                    "NAIW": None,
                    "Winkler": None,
                    "error": str(e),
                })

    df = pd.DataFrame(rows)
    out_csv = os.path.join(args.output_dir, f"e3_traditional_{args.dataset_group}.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nE3 汇总已保存: {out_csv}")

    out_md = os.path.join(args.output_dir, f"e3_traditional_{args.dataset_group}.md")
    with open(out_md, "w") as f:
        f.write(
            f"# E3 传统基线对比：ARIMA / ETS / Naive / Seasonal Naive / Prophet / LightGBM / N-BEATS 原生区间（{args.dataset_group}）\n\n"
        )
        f.write(
            "同一划分、同一 horizon=30，使用各方法原生置信/预测区间（不加 CQR）；可与 E2 方法对比表中的校准后 LLM 并列。\n\n"
        )
        f.write("| 数据集 | 方法 | ECP(95%) | Coverage_Gap | NAIW | Winkler |\n")
        f.write("|--------|------|----------|--------------|------|--------|\n")
        for _, r in df.iterrows():
            ecp = r.get("ECP(95%)")
            gap = r.get("Coverage_Gap")
            naiw = r.get("NAIW")
            wink = r.get("Winkler")
            ecp_s = f"{ecp:.3f}" if ecp is not None and not pd.isna(ecp) else "—"
            gap_s = f"{gap:.3f}" if gap is not None and not pd.isna(gap) else "—"
            naiw_s = f"{naiw:.3f}" if naiw is not None and not pd.isna(naiw) else "—"
            wink_s = f"{wink:.3f}" if wink is not None and not pd.isna(wink) else "—"
            f.write(
                f"| {r['dataset']} | {r['method']} | {ecp_s} | {gap_s} | {naiw_s} | {wink_s} |\n"
            )
    print(f"E3 Markdown 表: {out_md}")
    return df


if __name__ == "__main__":
    main()
