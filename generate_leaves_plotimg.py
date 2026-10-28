import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

META_IN = "/mnt/LightCurve_Classify/meta_with_period.csv"
OUT_ROOT = "/mnt/LightCurve_Classify/LEAVES_PLOTIMG"
META_OUT = "/mnt/LightCurve_Classify/meta_with_plotimg.csv"

# 新增列名
NEW_COL = "plotimg_path"

# 若想让图像分支复用原来的 png_path 列，可改成 True
REPLACE_PNG_PATH = False

IMG_SIZE = (320, 320)   # 输出图像大小
DPI = 100               # 320/100 ≈ 3.20 inch
LINE_WIDTH = 0.8
MARKER_SIZE = 1.2
MIN_POINTS = 20         # 少于这个点数就跳过
USE_YERR = False        # 不用误差条


def resolve_lc_path(row, lc_root=None):
    """支持 meta 里是绝对路径或相对路径"""
    p = str(row["lc_path"] if "lc_path" in row else row["raw_path"]).replace("\\", "/")
    if os.path.isabs(p):
        return p
    if lc_root is None:
        return p
    return os.path.join(lc_root, p)


def read_light_curve(csv_path):
    df = pd.read_csv(csv_path)

    cols = {c.lower(): c for c in df.columns}
    tcol = cols.get("time")
    mcol = cols.get("mag")
    ecol = cols.get("mag_err") or cols.get("mag err") or cols.get("magerr")

    if tcol is None or mcol is None:
        return None, None, None

    t = pd.to_numeric(df[tcol], errors="coerce").to_numpy(np.float64)
    y = pd.to_numeric(df[mcol], errors="coerce").to_numpy(np.float64)

    if ecol is not None:
        err = pd.to_numeric(df[ecol], errors="coerce").to_numpy(np.float64)
    else:
        err = None

    mask = np.isfinite(t) & np.isfinite(y)
    if err is not None:
        mask = mask & np.isfinite(err)

    t = t[mask]
    y = y[mask]
    if err is not None:
        err = err[mask]

    if len(t) < MIN_POINTS:
        return None, None, None

    order = np.argsort(t)
    t = t[order]
    y = y[order]
    if err is not None:
        err = err[order]

    return t, y, err


def minmax_norm(x):
    x = np.asarray(x, dtype=np.float64)
    xmin = np.min(x)
    xmax = np.max(x)
    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmax <= xmin:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - xmin) / (xmax - xmin)).astype(np.float32)


def save_plot_image(t, y, err, out_png):
    # 时间和幅值都做 per-curve min-max
    tn = minmax_norm(t)
    yn = minmax_norm(y)

    fig_w = IMG_SIZE[0] / DPI
    fig_h = IMG_SIZE[1] / DPI

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=DPI)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    if USE_YERR and err is not None:
        errn = minmax_norm(err)
        ax.errorbar(
            tn, yn, yerr=errn * 0.02,
            fmt='o-', color='black',
            markersize=MARKER_SIZE, linewidth=LINE_WIDTH,
            elinewidth=0.3, capsize=0
        )
    else:
        ax.plot(
            tn, yn,
            color="black",
            linewidth=LINE_WIDTH,
            marker="o",
            markersize=MARKER_SIZE
        )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)

    # 移除坐标轴/刻度/文字
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(out_png, dpi=DPI, facecolor="white", edgecolor="none", bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def main():
    out_root = Path(OUT_ROOT)
    out_root.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(META_IN)

    if "label" not in meta.columns:
        raise ValueError("索引表里没有 label 列")
    if ("lc_path" not in meta.columns) and ("raw_path" not in meta.columns):
        raise ValueError("索引表里既没有 lc_path 也没有 raw_path")

    new_paths = []
    ok_flags = []

    for _, row in meta.iterrows():
        label = str(row["label"]).upper()
        lc_path = resolve_lc_path(row)

        if not os.path.isfile(lc_path):
            new_paths.append("")
            ok_flags.append(False)
            continue

        t, y, err = read_light_curve(lc_path)
        if t is None:
            new_paths.append("")
            ok_flags.append(False)
            continue

        cls_dir = out_root / label
        cls_dir.mkdir(parents=True, exist_ok=True)

        stem = Path(lc_path).stem
        out_png = cls_dir / f"{stem}.png"

        try:
            save_plot_image(t, y, err, str(out_png))
            new_paths.append(str(out_png))
            ok_flags.append(True)
        except Exception:
            new_paths.append("")
            ok_flags.append(False)

    meta[NEW_COL] = new_paths
    meta["plotimg_exists"] = ok_flags

    if REPLACE_PNG_PATH:
        meta["png_path"] = meta[NEW_COL]

    meta.to_csv(META_OUT, index=False, encoding="utf-8-sig")

    print("Done.")
    print("META_OUT =", META_OUT)
    print("OUT_ROOT =", OUT_ROOT)
    print("成功生成图像数 =", int(np.sum(ok_flags)))
    print("失败样本数 =", int(len(ok_flags) - np.sum(ok_flags)))
    print("\\n前5条新路径：")
    print(meta[[ "label", NEW_COL ]].head().to_string(index=False))


if __name__ == "__main__":
    main()
