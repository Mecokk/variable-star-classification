import os
import re
import time
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import torchvision.transforms as T
import torchvision.models as models

CSV_PATH = r"/mnt/LightCurve_Classify/meta_with_plotimg_320.csv"
LC_ROOT = r"/mnt/LightCurve_Classify/LEAVES"
IMG_ROOT = r"/mnt/LightCurve_Classify/LEAVES_PLOTIMG_320"

SPLIT_VAL = "val"
SPLIT_TEST = "test"

CLASS_NAMES = ["EW","EA","SR","M","ROT","RRAB","RRC","DSCT","CEP"]
NUM_CLASSES = len(CLASS_NAMES)

MAX_LEN = 512
IMG_SIZE = 320
BATCH_SIZE = 32
NUM_WORKERS = 4

ALPHAS = [0.10, 0.20, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 2025

TIME_CKPT = fr"/mnt/LightCurve_Classify/checkpoints/conv_bilstm_period_best_seed{SEED}.pth"
IMG_CKPT  = fr"/mnt/LightCurve_Classify/checkpoints/plotimg320_swinv2_t_best_seed{SEED}.pth"

TIME_HIDDEN_DIM = 128
TIME_CONV_CHANNELS = 64
PERIOD_FEAT_DIM = 4
USE_IMAGENET_PRETRAIN = False


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


def _resolve_path(root: str, p: str) -> str:
    s = str(p).strip().replace("\\", "/")
    if s == "" or s.lower() in ("0", "nan", "none"):
        return ""
    if os.path.isabs(s) or _is_win_abs(s):
        return s
    return os.path.join(root, s)


class LateFusionDataset(Dataset):
    def __init__(self, csv_path, split, max_len=512):
        df = pd.read_csv(csv_path)
        df["split"] = df["split"].astype(str).str.lower()
        df = df[df["split"] == str(split).lower()].reset_index(drop=True)

        if "band" in df.columns:
            df["band"] = df["band"].astype(str).str.lower()
            df = df[df["band"] == "v"].reset_index(drop=True)

        label2id = {name: i for i, name in enumerate(CLASS_NAMES)}
        if "label" not in df.columns:
            raise ValueError(f"meta 缺少 label 列，当前列={list(df.columns)}")
        df["label"] = df["label"].astype(str).str.upper()
        df = df[df["label"].isin(label2id.keys())].reset_index(drop=True)
        df["label_id"] = df["label"].map(label2id).astype(int)

        if "lc_path" not in df.columns and "raw_path" not in df.columns:
            raise ValueError("meta 里没有 lc_path / raw_path")
        if "plotimg_path" not in df.columns:
            raise ValueError("meta 里没有 plotimg_path")

        lc_col = "lc_path" if "lc_path" in df.columns else "raw_path"
        img_col = "plotimg_path"

        lc_abs, img_abs, keep = [], [], []
        for i in range(len(df)):
            lp = _resolve_path(LC_ROOT, df.loc[i, lc_col])
            ip = _resolve_path(IMG_ROOT, df.loc[i, img_col])
            ok = (lp != "" and os.path.exists(lp) and ip != "" and os.path.exists(ip))
            keep.append(ok)
            lc_abs.append(lp)
            img_abs.append(ip)

        df = df[np.array(keep)].reset_index(drop=True)
        df["lc_abs"] = [p for p, k in zip(lc_abs, keep) if k]
        df["img_abs"] = [p for p, k in zip(img_abs, keep) if k]

        self.df = df
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def _load_lc_csv(self, path):
        df = pd.read_csv(path)
        cols = {c.lower(): c for c in df.columns}
        tcol = cols.get("time")
        mcol = cols.get("mag")
        ecol = cols.get("mag_err") or cols.get("mag err") or cols.get("magerr") or cols.get("err")
        if tcol is None or mcol is None or ecol is None:
            raise ValueError(f"{path} 缺少 time/mag/mag_err 列，实际列={list(df.columns)}")

        t = df[tcol].to_numpy(np.float32)
        mag = df[mcol].to_numpy(np.float32)
        err = df[ecol].to_numpy(np.float32)

        mask = np.isfinite(t) & np.isfinite(mag) & np.isfinite(err) & (err > 0)
        t, mag, err = t[mask], mag[mask], err[mask]

        order = np.argsort(t)
        t, mag, err = t[order], mag[order], err[order]

        if len(mag) < 2:
            return np.zeros((1, 3), dtype=np.float32)

        if len(mag) > self.max_len:
            idxs = np.linspace(0, len(mag) - 1, self.max_len).astype(np.int64)
            t, mag, err = t[idxs], mag[idxs], err[idxs]

        dt = np.zeros_like(t)
        dt[1:] = t[1:] - t[:-1]
        dt = np.log1p(np.clip(dt, 0, None))

        mag = (mag - mag.mean()) / (mag.std() + 1e-6)
        err = (err - err.mean()) / (err.std() + 1e-6)
        dt = (dt - dt.mean()) / (dt.std() + 1e-6)
        return np.stack([mag, err, dt], axis=1)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x_time = self._load_lc_csv(row["lc_abs"])
        img = Image.open(row["img_abs"]).convert("RGB")
        y = int(row["label_id"])

        period_best = float(row["period_best"]) if "period_best" in row and pd.notna(row["period_best"]) else np.nan
        pdm_theta = float(row["pdm_theta"]) if "pdm_theta" in row and pd.notna(row["pdm_theta"]) else np.nan
        ls_ratio = float(row["ls_power_ratio12"]) if "ls_power_ratio12" in row and pd.notna(row["ls_power_ratio12"]) else np.nan
        period_valid = float(row["period_valid"]) if "period_valid" in row and pd.notna(row["period_valid"]) else 0.0

        if np.isfinite(period_best) and period_best > 0:
            log_period = np.log10(period_best + 1e-6)
        else:
            log_period = 0.0
        if not np.isfinite(pdm_theta):
            pdm_theta = 0.0
        if not np.isfinite(ls_ratio):
            ls_ratio = 0.0

        period_feat = np.array([log_period, pdm_theta, ls_ratio, period_valid], dtype=np.float32)

        return {
            "x_time": torch.tensor(x_time, dtype=torch.float32),
            "period_feat": torch.tensor(period_feat, dtype=torch.float32),
            "length": len(x_time),
            "img": img,
            "y": torch.tensor(y, dtype=torch.long),
        }


def build_img_tf():
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


def collate_fn(batch, img_tf):
    xs = [b["x_time"] for b in batch]
    period_feats = torch.stack([b["period_feat"] for b in batch], dim=0)
    lengths = torch.tensor([b["length"] for b in batch], dtype=torch.long)
    ys = torch.stack([b["y"] for b in batch], dim=0)
    x_pad = pad_sequence(xs, batch_first=True)
    imgs = torch.stack([img_tf(b["img"]) for b in batch], dim=0)
    return x_pad, period_feats, lengths, imgs, ys


class ConvBiLSTMPeriodClassifier(nn.Module):
    def __init__(self, input_dim=3, conv_channels=64, hidden_dim=128, period_feat_dim=4, num_classes=3):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, conv_channels, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(conv_channels)
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.period_mlp = nn.Sequential(
            nn.Linear(period_feat_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(hidden_dim * 2 + 32, num_classes)

    def forward(self, x, period_feat, lengths):
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = x.transpose(1, 2)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        h = torch.cat([h_n[0], h_n[1]], dim=1)
        p = self.period_mlp(period_feat)
        feat = torch.cat([h, p], dim=1)
        return self.fc(feat)


def make_swinv2_t(num_classes):
    weights = models.Swin_V2_T_Weights.DEFAULT if USE_IMAGENET_PRETRAIN else None
    model = models.swin_v2_t(weights=weights)
    model.head = nn.Linear(model.head.in_features, num_classes)
    return model


def load_ckpt(model, ckpt_path):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"找不到权重文件: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=True)
    return model


@torch.no_grad()
def eval_fusion(time_model, img_model, loader, alpha):
    time_model.eval()
    img_model.eval()

    all_true, all_pred = [], []
    total_loss, total_n = 0.0, 0

    for x_time, period_feat, lengths, x_img, y in loader:
        x_time = x_time.to(DEVICE, non_blocking=True)
        period_feat = period_feat.to(DEVICE, non_blocking=True)
        lengths = lengths.to(DEVICE, non_blocking=True)
        x_img = x_img.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        logits_t = time_model(x_time, period_feat, lengths)
        logits_i = img_model(x_img)
        logits = logits_t + alpha * logits_i

        loss = F.cross_entropy(logits, y, reduction="sum")
        pred = torch.argmax(logits, dim=1)

        all_true.append(y.cpu())
        all_pred.append(pred.cpu())
        total_loss += loss.item()
        total_n += y.size(0)

    all_true = torch.cat(all_true)
    all_pred = torch.cat(all_pred)

    cm = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64)
    for t, p in zip(all_true, all_pred):
        cm[int(t), int(p)] += 1

    prec_list, rec_list, f1_list = [], [], []
    for c in range(NUM_CLASSES):
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

    supports = []
    for c in range(NUM_CLASSES):
        supports.append(int(cm[c, :].sum()))
    weighted_f1 = float(np.average(f1_list, weights=supports))

    avg_loss = total_loss / max(total_n, 1)
    return avg_loss, acc, macro_f1, weighted_f1, cm, prec_list, rec_list, f1_list


def print_metrics(tag, loss, acc, macro_f1, weighted_f1, cm, prec, rec, f1):
    print(f"\n[{tag}] loss {loss:.4f} | acc {acc:.4f} | macroF1 {macro_f1:.4f} | weightedF1 {weighted_f1:.4f}")
    print("[Confusion Matrix] rows=true, cols=pred")
    header = "      " + "  ".join([f"{n:>6s}" for n in CLASS_NAMES])
    print(header)
    for i, name in enumerate(CLASS_NAMES):
        row = "  ".join([f"{int(x):6d}" for x in cm[i].tolist()])
        print(f"{name:>6s}  {row}")
    for i, name in enumerate(CLASS_NAMES):
        support = int(cm[i, :].sum())
        print(f"[{name}] prec={prec[i]:.4f} rec={rec[i]:.4f} f1={f1[i]:.4f} support={support}")
    print(f"[MacroF1-from-CM] {sum(f1)/len(f1):.4f}")


def main():
    set_seed(SEED)
    print("model: Late Fusion (logits)")
    print("device:", DEVICE)
    print("cuda available:", torch.cuda.is_available())
    print("TIME_CKPT:", TIME_CKPT)
    print("IMG_CKPT :", IMG_CKPT)

    img_tf = build_img_tf()
    val_ds = LateFusionDataset(CSV_PATH, SPLIT_VAL, max_len=MAX_LEN)
    test_ds = LateFusionDataset(CSV_PATH, SPLIT_TEST, max_len=MAX_LEN)

    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True,
                            collate_fn=lambda b: collate_fn(b, img_tf))
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True,
                             collate_fn=lambda b: collate_fn(b, img_tf))

    print("val size :", len(val_ds))
    print("test size:", len(test_ds))

    time_model = ConvBiLSTMPeriodClassifier(
        input_dim=3,
        conv_channels=TIME_CONV_CHANNELS,
        hidden_dim=TIME_HIDDEN_DIM,
        period_feat_dim=PERIOD_FEAT_DIM,
        num_classes=NUM_CLASSES
    )
    img_model = make_swinv2_t(num_classes=NUM_CLASSES)

    time_model = load_ckpt(time_model, TIME_CKPT).to(DEVICE)
    img_model = load_ckpt(img_model, IMG_CKPT).to(DEVICE)

    best_alpha = None
    best_val_f1 = -1.0
    best_val_pack = None

    print("\n===== Search alpha on VAL =====")
    for alpha in ALPHAS:
        t0 = time.time()
        result = eval_fusion(time_model, img_model, val_loader, alpha)
        v_loss, v_acc, v_f1, v_wf1, v_cm, v_prec, v_rec, v_f1_list = result
        print(f"alpha={alpha:.2f} | val_loss={v_loss:.4f} | val_acc={v_acc:.4f} | val_macroF1={v_f1:.4f} | val_weightedF1={v_wf1:.4f} | elapsed={time.time()-t0:.1f}s")
        if v_f1 > best_val_f1:
            best_val_f1 = v_f1
            best_alpha = alpha
            best_val_pack = result

    print(f"\n[Best alpha on VAL] alpha={best_alpha:.2f} | macroF1={best_val_f1:.4f}")
    print_metrics("VAL-BEST", *best_val_pack)

    print("\n===== Evaluate best alpha on TEST =====")
    test_pack = eval_fusion(time_model, img_model, test_loader, best_alpha)
    print_metrics(f"TEST alpha={best_alpha:.2f}", *test_pack)


if __name__ == "__main__":
    main()
