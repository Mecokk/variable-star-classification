import os
import re
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
import torchvision.models as models

CSV_PATH = r"/mnt/LightCurve_Classify/meta_with_plotimg_320.csv"
PNG_ROOT = r"/mnt/LightCurve_Classify/LEAVES_PLOTIMG_320"

SEED = 2025
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-4

USE_SAMPLER = True
NUM_WORKERS = 4
IMG_SIZE = 320

CLASS_NAMES = ["EW","EA","SR","M","ROT","RRAB","RRC","DSCT","CEP"]

USE_IMAGENET_PRETRAIN = True  # 如果服务器不能下载权重，改 False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "SwinV2-T"
SAVE_BEST_PATH = fr"/mnt/LightCurve_Classify/checkpoints/plotimg320_swinv2_t_best_seed{SEED}.pth"

# ======================================
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _is_win_abs(p: str) -> bool:
    return re.match(r"^[A-Za-z]:[\\/]", str(p)) is not None

def _resolve_png(p: str) -> str:
    s = str(p).strip().replace("\\", "/")
    if os.path.isabs(s) or _is_win_abs(s):
        return s
    return os.path.join(PNG_ROOT, s)

def load_paths_from_meta(csv_path: str, split: str):
    df = pd.read_csv(csv_path)

    # split
    df["split"] = df["split"].astype(str).str.lower()
    df = df[df["split"] == str(split).lower()].copy()

    # 只保留 v 波段
    if "band" in df.columns:
        df["band"] = df["band"].astype(str).str.lower()
        df = df[df["band"] == "v"].copy()

    # png 列名：优先 png_path，其次 cwt_png（兼容旧meta）
    # 图像列：优先 plotimg_path，其次 png_path，再其次 cwt_png
    if "plotimg_path" in df.columns:
        png_col = "plotimg_path"
    elif "png_path" in df.columns:
        png_col = "png_path"
    elif "cwt_png" in df.columns:
        png_col = "cwt_png"
    else:
        raise ValueError(f"meta 缺少图像路径列（plotimg_path / png_path / cwt_png），当前列={list(df.columns)}")
    print(f"[debug] split={split} use image column: {png_col}")

    # 强制用 label 字符串映射，避免 label_id 顺序不一致
    if "label" not in df.columns:
        raise ValueError(f"meta 缺少 label 列，当前列={list(df.columns)}")

    label2id = {name: i for i, name in enumerate(CLASS_NAMES)}
    df["label"] = df["label"].astype(str).str.upper()
    df = df[df["label"].isin(label2id.keys())].copy()
    df["label_id"] = df["label"].map(label2id).astype(int)

    paths, labels = [], []
    for _, r in df.iterrows():
        rawp = r[png_col]
        s = str(rawp).strip().lower()
        if s in ("", "0", "nan", "none"):
            continue
        ap = _resolve_png(rawp)
        if not os.path.exists(ap):
            continue
        paths.append(ap)
        labels.append(int(r["label_id"]))

    return paths, labels

class CWTPngDataset(Dataset):
    def __init__(self, paths: List[str], labels: List[int], transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]
        y = int(self.labels[i])
        img = Image.open(p).convert("L")  # 灰度
        if self.transform is not None:
            img = self.transform(img)  # (1,H,W)
        return img, torch.tensor(y, dtype=torch.long)


def make_resnet18_gray(num_classes: int = 3):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # 把第一层 conv 从 3 通道改成 1 通道
    old_conv = model.conv1
    new_conv = nn.Conv2d(
        in_channels=1,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False
    )
    # 用原来权重的均值初始化（3->1）
    with torch.no_grad():
        new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
    model.conv1 = new_conv

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def make_swinv2_t(num_classes: int = 3):
    weights = models.Swin_V2_T_Weights.DEFAULT if USE_IMAGENET_PRETRAIN else None
    model = models.swin_v2_t(weights=weights)
    model.head = nn.Linear(model.head.in_features, num_classes)
    return model

@torch.no_grad()
def evaluate(model, loader, num_classes=3):
    model.eval()
    all_true, all_pred = [], []
    total_loss, total_n = 0.0, 0

    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        total_loss += loss.item() * x.size(0)
        total_n += x.size(0)

        pred = torch.argmax(logits, dim=1)
        all_true.append(y.cpu())
        all_pred.append(pred.cpu())

    all_true = torch.cat(all_true)
    all_pred = torch.cat(all_pred)

    # confusion matrix
    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for t, p in zip(all_true, all_pred):
        cm[int(t), int(p)] += 1

    # per-class metrics
    prec_list, rec_list, f1_list = [], [], []
    for c in range(num_classes):
        tp = int(cm[c, c])
        fp = int(cm[:, c].sum() - cm[c, c])
        fn = int(cm[c, :].sum() - cm[c, c])
        prec = tp / (tp + fp + 1e-12)
        rec = tp / (tp + fn + 1e-12)
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        prec_list.append(prec)
        rec_list.append(rec)
        f1_list.append(f1)

    acc = float((all_true == all_pred).float().mean().item())
    macro_f1 = float(np.mean(f1_list))
    avg_loss = total_loss / max(total_n, 1)

    return avg_loss, acc, macro_f1, cm, prec_list, rec_list, f1_list


def main():
    set_seed(SEED)
    os.makedirs(os.path.dirname(SAVE_BEST_PATH), exist_ok=True)
    print("save best path:", SAVE_BEST_PATH)
    print("model:", MODEL_NAME)
    print("device:", DEVICE)
    print("cuda available:", torch.cuda.is_available())

    # transforms: 灰度 -> Tensor(1,H,W) -> 归一化
    train_tf = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),  # (1,H,W)
        T.Lambda(lambda x: x.repeat(3, 1, 1)),  # -> (3,H,W)
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_tf = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Lambda(lambda x: x.repeat(3, 1, 1)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    tr_p, tr_y = load_paths_from_meta(CSV_PATH, "train")
    va_p, va_y = load_paths_from_meta(CSV_PATH, "val")
    te_p, te_y = load_paths_from_meta(CSV_PATH, "test")

    print("split sizes:", len(tr_p), len(va_p), len(te_p))

    # 总量与各类数量（基于“统一 split + 过滤缺失png 后”的数据）
    all_y = np.array(tr_y + va_y + te_y, dtype=int)
    print("total samples:", len(all_y))
    counts = np.bincount(all_y, minlength=len(CLASS_NAMES))
    print("class counts:", {CLASS_NAMES[i]: int(counts[i]) for i in range(len(CLASS_NAMES))})

    train_ds = CWTPngDataset(tr_p, tr_y, transform=train_tf)
    val_ds = CWTPngDataset(va_p, va_y, transform=test_tf)
    test_ds = CWTPngDataset(te_p, te_y, transform=test_tf)

    # sampler（可选）
    if USE_SAMPLER:
        tr_y_np = np.array(tr_y, dtype=int)
        cls_cnt = np.bincount(tr_y_np, minlength=len(CLASS_NAMES))
        cls_w = 1.0 / (cls_cnt + 1e-12)
        sample_w = cls_w[tr_y_np]
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_w, dtype=torch.double),
            num_samples=len(sample_w),
            replacement=True
        )
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                                  num_workers=NUM_WORKERS, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=NUM_WORKERS, pin_memory=True)

    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)

    model = make_swinv2_t(num_classes=len(CLASS_NAMES)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    best_val_f1 = -1.0
    best_state = None

    # 训练
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        for it, (x, y) in enumerate(train_loader, 1):
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += loss.item()
            if it == 1:
                print(f"[debug] epoch={epoch} x={tuple(x.shape)} device={x.device}")
            if it % 50 == 0 or it == len(train_loader):
                print(f"Epoch {epoch:02d} | iter {it}/{len(train_loader)} | loss {running/it:.4f}")

        v_loss, v_acc, v_f1, _, _, _, _ = evaluate(model, val_loader, num_classes=len(CLASS_NAMES))
        print(f"[VAL]  loss {v_loss:.4f} | acc {v_acc:.4f} | macroF1 {v_f1:.4f}")
        if v_f1 > best_val_f1:
            best_val_f1 = v_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, SAVE_BEST_PATH)
            print(f"[Info] Update best @ epoch={epoch} | best_macroF1={best_val_f1:.4f} | saved to {SAVE_BEST_PATH}")

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
        model.to(DEVICE)
        print(f"[Info] Loaded best val model (best_macroF1={best_val_f1:.4f})")

    # 测试
    t_loss, t_acc, t_f1, cm, prec, rec, f1 = evaluate(model, test_loader, num_classes=len(CLASS_NAMES))
    print("\n[TEST] loss {:.4f} | acc {:.4f} | macroF1 {:.4f}".format(t_loss, t_acc, t_f1))
    print("[Confusion Matrix] rows=true, cols=pred")
    header = "      " + "  ".join([f"{n:>6s}" for n in CLASS_NAMES])
    print(header)
    for i, name in enumerate(CLASS_NAMES):
        row = "  ".join([f"{int(x):6d}" for x in cm[i].tolist()])
        print(f"{name:>6s}  {row}")

    for i, name in enumerate(CLASS_NAMES):
        print(f"[{name}] prec={prec[i]:.4f} rec={rec[i]:.4f} f1={f1[i]:.4f}")

if __name__ == "__main__":
    main()
