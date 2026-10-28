import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from scipy.signal import lombscargle

META_IN  = "/mnt/LightCurve_Classify/LEAVES_CWT_PNG_9CLS_GAUSS/meta_sever.csv"
META_OUT = "/mnt/LightCurve_Classify/meta_with_periodv2.csv"

# 周期搜索范围（天）
MIN_PERIOD = 0.02     # 约 30 分钟；DSCT 很短
MAX_PERIOD = 200.0    # 覆盖 M/SR 更长周期；

N_FREQ = 6000         # 频率网格数
TOPK_LS = 8           # LS 取前K个候选周期
PDM_BINS = 10         # PDM分箱数

MIN_POINTS = 100      # 清洗阈值一致
MIN_CYCLES = 2.0      # 至少覆盖2个周期
THETA_MAX = 0.85      # PDM阈值
POWER_RATIO_MIN = 1.2 # LS 第一峰/第二峰
CLASS_POWER_RATIO_MIN = {
    "M": 1.0,
    "SR": 1.0,
}

SEED = 42
# ==========================================

def read_leaves_csv(path: str):
    df = pd.read_csv(path, usecols=["time", "mag", "mag_err"])
    t = pd.to_numeric(df["time"], errors="coerce").to_numpy(np.float64)
    y = pd.to_numeric(df["mag"], errors="coerce").to_numpy(np.float64)

    m = np.isfinite(t) & np.isfinite(y)
    t, y = t[m], y[m]
    if len(t) == 0:
        return None, None

    # sort + t从0开始
    idx = np.argsort(t)
    t, y = t[idx], y[idx]
    t = t - t.min()
    return t, y

def pdm_theta(t, y, period, nbins=10):
    # phase fold
    phase = (t / period) % 1.0
    order = np.argsort(phase)
    phase = phase[order]
    y = y[order]

    var_all = np.var(y, ddof=1) if len(y) > 1 else 0.0
    if var_all <= 0:
        return 1.0

    # binning
    bins = np.floor(phase * nbins).astype(int)
    bins[bins == nbins] = nbins - 1

    num = 0.0
    den = (len(y) - 1) * var_all

    for b in range(nbins):
        idx = np.where(bins == b)[0]
        if len(idx) >= 2:
            vb = np.var(y[idx], ddof=1)
            num += (len(idx) - 1) * vb

    theta = num / den if den > 0 else 1.0
    # theta 越小越好
    return float(theta)

def estimate_period_ls_pdm(t, y, min_period, max_period):
    if len(t) < MIN_POINTS:
        return np.nan, np.nan, np.nan, np.nan, 0

    span = t.max() - t.min()
    if span <= 0:
        return np.nan, np.nan, np.nan, np.nan, 0

    # 频率范围（cycle/day）
    fmin = 1.0 / max_period
    fmax = 1.0 / min_period
    freqs = np.linspace(fmin, fmax, N_FREQ)
    ang = 2 * np.pi * freqs  # rad/day

    # 预处理：去均值、标准化（避免数值问题）
    y0 = y - np.mean(y)
    s = np.std(y0)
    if s > 0:
        y0 = y0 / s

    # LS power（scipy.signal.lombscargle）
    pwr = lombscargle(t, y0, ang, precenter=False, normalize=True)

    # 取 TOPK 候选频率（去掉重复）
    idx_top = np.argpartition(pwr, -TOPK_LS)[-TOPK_LS:]
    idx_top = idx_top[np.argsort(pwr[idx_top])[::-1]]  # 降序

    cand_periods = []
    cand_powers = []
    for idx in idx_top:
        per = 1.0 / freqs[idx]
        cand_periods.append(per)
        cand_powers.append(pwr[idx])

    # 计算 power ratio（第一峰/第二峰）
    if len(cand_powers) >= 2 and cand_powers[1] > 0:
        power_ratio = float(cand_powers[0] / cand_powers[1])
    else:
        power_ratio = np.inf

    # 用 PDM 在 LS 候选及其谐波里选最小 theta
    best = None
    for per, pw in zip(cand_periods, cand_powers):
        # 谐波候选：p, 2p, p/2
        for per2 in [per, 2*per, 0.5*per]:
            if per2 < MIN_PERIOD or per2 > MAX_PERIOD:
                continue
            th = pdm_theta(t, y, per2, nbins=PDM_BINS)
            if best is None or th < best["theta"]:
                best = {"period": per2, "theta": th, "ls_power": float(pw)}

    if best is None:
        return np.nan, np.nan, np.nan, np.nan, 0

    best_period = best["period"]
    best_theta = best["theta"]
    best_ls_power = best["ls_power"]

    cycles = span / best_period

    valid = int((cycles >= MIN_CYCLES) and (best_theta <= THETA_MAX) and (power_ratio >= POWER_RATIO_MIN))

    return float(best_period), float(best_ls_power), float(best_theta), float(power_ratio), valid

def main():
    print("META_IN =", META_IN)
    meta = pd.read_csv(META_IN)

    # 确保有 lc_path
    if "lc_path" not in meta.columns:
        raise RuntimeError("meta.csv 里找不到 lc_path 列名。请确认 meta.csv 表头。")

    periods = []
    ls_powers = []
    pdm_thetas = []
    power_ratios = []
    valids = []

    for p in tqdm(meta["lc_path"].astype(str), desc="Estimating periods (LS+PDM)"):
        if not os.path.isfile(p):
            periods.append(np.nan); ls_powers.append(np.nan); pdm_thetas.append(np.nan); power_ratios.append(np.nan); valids.append(0)
            continue

        t, y = read_leaves_csv(p)
        if t is None:
            periods.append(np.nan); ls_powers.append(np.nan); pdm_thetas.append(np.nan); power_ratios.append(np.nan); valids.append(0)
            continue

        # 所有样本统一周期搜索范围（无标签泄露）
        minP = MIN_PERIOD
        maxP = MAX_PERIOD

        # 保险：maxP 不能超过观测跨度，否则会浪费搜索
        span = float(t.max() - t.min())
        if span > 0:
            maxP = min(maxP, span)

        per, pw, th, pr, ok = estimate_period_ls_pdm(t, y, minP, maxP)
        periods.append(per); ls_powers.append(pw); pdm_thetas.append(th); power_ratios.append(pr); valids.append(ok)

    meta["period_best"] = periods
    meta["ls_power"] = ls_powers
    meta["pdm_theta"] = pdm_thetas
    meta["ls_power_ratio12"] = power_ratios
    meta["period_valid"] = valids

    meta.to_csv(META_OUT, index=False, encoding="utf-8-sig")
    print("\nSaved:", META_OUT)
    print("Valid periods:", int(meta["period_valid"].sum()), "/", len(meta))

if __name__ == "__main__":
    main()