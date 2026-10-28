import os
import re
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence


# ===== LEAVES 配置 =====
CSV_PATH = r"/mnt/LightCurve_Classify/meta_with_plotimg_320.csv"   # 索引文件
LC_ROOT  = r"/mnt/LightCurve_Classify/LEAVES"                  # 光变csv根目录

SPLIT_TRAIN = "train"
SPLIT_VAL   = "val"
SPLIT_TEST  = "test"

CLASS_NAMES = ["EW","EA","SR","M","ROT","RRAB","RRC","DSCT","CEP"]
NUM_CLASSES = len(CLASS_NAMES)

BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-3
HIDDEN_DIM = 128
CONV_CHANNELS = 64
KERNEL_SIZE = 5
SEED = 2025
PERIOD_FEAT_DIM = 4
SAVE_BEST_PATH = fr"/mnt/LightCurve_Classify/checkpoints/conv_bilstm_period_best_seed{SEED}.pth"


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================
# 1) Dataset
# =========================
class LeavesStrongRNNDataset(Dataset):
    """
    读取 meta_LEAVES*.csv 中 split 对应的样本
    每条光变曲线 csv 读取列：time, mag, mag_err（忽略 flux/flux_err）
    输出：
    1) 序列特征：(mag, mag_err, dt)，dt=log1p(delta_time)
    2) 周期特征：(log_period, pdm_theta, ls_power_ratio12, period_valid)
    """

    def __init__(self, csv_path, split, max_len=512):
        self.df = pd.read_csv(csv_path)

        # 1) split
        self.df["split"] = self.df["split"].astype(str).str.lower()
        self.df = self.df[self.df["split"] == str(split).lower()].reset_index(drop=True)

        # 2) 只保留 v 波段（如果 meta 有 band 列）
        if "band" in self.df.columns:
            self.df["band"] = self.df["band"].astype(str).str.lower()
            self.df = self.df[self.df["band"] == "v"].reset_index(drop=True)

        # 3) label 映射：若 meta 没有 label_id，则用 label 字符串生成
        self.label2id = {name: i for i, name in enumerate(CLASS_NAMES)}
        if "label_id" not in self.df.columns:
            if "label" not in self.df.columns:
                raise ValueError("meta 里既没有 label_id 也没有 label 列，无法生成标签")
            self.df["label"] = self.df["label"].astype(str).str.upper()
            self.df = self.df[self.df["label"].isin(self.label2id.keys())].reset_index(drop=True)
            self.df["label_id"] = self.df["label"].map(self.label2id).astype(int)

        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def _resolve_path(self, p: str) -> str:
        """支持 meta 里写绝对路径或相对路径（相对 LC_ROOT）"""
        p = str(p).replace("\\", "/")
        is_win_abs = bool(re.match(r"^[A-Za-z]:/", p))
        if os.path.isabs(p) or is_win_abs:
            return p
        return os.path.join(LC_ROOT, p)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        if "lc_path" in row:
            path = self._resolve_path(row["lc_path"])
        else:
            path = self._resolve_path(row["raw_path"])

        df_lc = pd.read_csv(path)

        # 兼容列名大小写/下划线：time, mag, mag_err
        cols = {c.lower(): c for c in df_lc.columns}
        tcol = cols.get("time")
        mcol = cols.get("mag")
        ecol = cols.get("mag_err") or cols.get("mag err") or cols.get("magerr")

        if tcol is None or mcol is None or ecol is None:
            raise ValueError(f"[{os.path.basename(path)}] 缺少 time/mag/mag_err 列，实际列={list(df_lc.columns)}")

        t   = df_lc[tcol].to_numpy(np.float32)
        mag = df_lc[mcol].to_numpy(np.float32)
        err = df_lc[ecol].to_numpy(np.float32)

        # 清洗
        mask = np.isfinite(t) & np.isfinite(mag) & np.isfinite(err) & (err > 0)
        t, mag, err = t[mask], mag[mask], err[mask]

        # 防御性：按时间排序
        order = np.argsort(t)
        t, mag, err = t[order], mag[order], err[order]

        # 长度裁剪
        if len(mag) > self.max_len:
            idxs = np.linspace(0, len(mag) - 1, self.max_len).astype(np.int64)
            t, mag, err = t[idxs], mag[idxs], err[idxs]

        # dt：不均匀采样信息（time 单位：天）
        dt = np.zeros_like(t)
        dt[1:] = t[1:] - t[:-1]
        dt = np.log1p(np.clip(dt, 0, None))

        # 标准化（per-curve）
        mag = (mag - mag.mean()) / (mag.std() + 1e-6)
        err = (err - err.mean()) / (err.std() + 1e-6)
        dt  = (dt  - dt.mean())  / (dt.std()  + 1e-6)

        seq = np.stack([mag, err, dt], axis=1)  # (T,3)

        # ===== 周期特征 =====
        period_best = float(row["period_best"]) if "period_best" in row and pd.notna(row["period_best"]) else np.nan
        pdm_theta = float(row["pdm_theta"]) if "pdm_theta" in row and pd.notna(row["pdm_theta"]) else np.nan
        ls_ratio = float(row["ls_power_ratio12"]) if "ls_power_ratio12" in row and pd.notna(row["ls_power_ratio12"]) else np.nan
        period_valid = float(row["period_valid"]) if "period_valid" in row and pd.notna(row["period_valid"]) else 0.0

        # log_period：缺失时先置 0
        if np.isfinite(period_best) and period_best > 0:
            log_period = np.log10(period_best + 1e-6)
        else:
            log_period = 0.0

        # 其它缺失值也先安全置 0
        if not np.isfinite(pdm_theta):
            pdm_theta = 0.0
        if not np.isfinite(ls_ratio):
            ls_ratio = 0.0

        period_feat = np.array(
            [log_period, pdm_theta, ls_ratio, period_valid],
            dtype=np.float32
        )

        return {
            "x": torch.tensor(seq, dtype=torch.float32),
            "period_feat": torch.tensor(period_feat, dtype=torch.float32),
            "length": len(seq),
            "label": torch.tensor(int(row["label_id"]), dtype=torch.long),
        }

def collate_fn(batch):
    """
    batch: list of {"x": (T,3), "period_feat": (4,), "length": int, "label": long}
    return:
        x_pad: (B,Tmax,3)
        period_feat: (B,4)
        lengths: (B,)
        labels: (B,)
    """
    xs = [b["x"] for b in batch]
    period_feats = torch.stack([b["period_feat"] for b in batch], dim=0)
    lengths = torch.tensor([b["length"] for b in batch], dtype=torch.long)
    labels = torch.stack([b["label"] for b in batch], dim=0)

    x_pad = pad_sequence(xs, batch_first=True)  # (B, T_max, 3)
    return x_pad, period_feats, lengths, labels

# =========================
# 2) 模型：CNN 1D+biLSTM+period
# =========================
class ConvBiLSTMPeriodClassifier(nn.Module):
    def __init__(self, input_dim=3, conv_channels=64, hidden_dim=128, period_feat_dim=4, num_classes=3):
        super().__init__()

        # 先做一层 1D 卷积，提局部模式
        self.conv1 = nn.Conv1d(
            in_channels=input_dim,
            out_channels=conv_channels,
            kernel_size=5,
            padding=2
        )
        self.bn1 = nn.BatchNorm1d(conv_channels)

        # 双向 LSTM
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # 周期特征支路
        self.period_mlp = nn.Sequential(
            nn.Linear(period_feat_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
        )

        # 双向 LSTM 输出是 2 * hidden_dim
        self.fc = nn.Linear(hidden_dim * 2 + 32, num_classes)

    def forward(self, x, period_feat, lengths):
        """
        x: (B, T, 3)
        period_feat: (B, 4)
        lengths: (B,)
        """

        # Conv1D 要求输入 (B, C, T)
        x = x.transpose(1, 2)                 # (B, 3, T)
        x = self.conv1(x)                     # (B, conv_channels, T)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)

        # 再转回 LSTM 需要的 (B, T, C)
        x = x.transpose(1, 2)                 # (B, T, conv_channels)

        packed = pack_padded_sequence(
            x, lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        _, (h_n, _) = self.lstm(packed)
        # 双向 LSTM:
        # h_n[0] = forward 最后时刻
        # h_n[1] = backward 最后时刻
        h = torch.cat([h_n[0], h_n[1]], dim=1)   # (B, 2*hidden_dim)

        p = self.period_mlp(period_feat)         # (B, 32)
        feat = torch.cat([h, p], dim=1)          # (B, 2H + 32)

        logits = self.fc(feat)
        return logits

# =========================
# 3) 指标：accuracy + macro-F1
# =========================
@torch.no_grad()
def eval_model(model, loader, verbose=True):
    model.eval()
    all_pred = []
    all_true = []
    total_loss = 0.0
    total_n = 0

    for x, period_feat, lengths, y in loader:
        x = x.to(DEVICE)
        period_feat = period_feat.to(DEVICE)
        lengths = lengths.to(DEVICE)
        y = y.to(DEVICE)
        logits = model(x, period_feat, lengths)
        loss = F.cross_entropy(logits, y, reduction="sum")

        pred = torch.argmax(logits, dim=1)

        all_pred.append(pred.cpu())
        all_true.append(y.cpu())
        total_loss += loss.item()
        total_n += y.size(0)

    all_pred = torch.cat(all_pred)
    all_true = torch.cat(all_true)
    
    if verbose:
    # ===== confusion matrix + per-class metrics =====
     cm = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64)
     for t, p in zip(all_true, all_pred):
        cm[int(t), int(p)] += 1

     print("\n[Confusion Matrix] rows=true, cols=pred")
     header = "      " + "  ".join([f"{n:>6s}" for n in CLASS_NAMES])
     print(header)
     for i, name in enumerate(CLASS_NAMES):
        row = "  ".join([f"{int(x):6d}" for x in cm[i].tolist()])
        print(f"{name:>6s}  {row}")

     for c, name in enumerate(CLASS_NAMES):
        tp = int(cm[c, c])
        fp = int(cm[:, c].sum() - cm[c, c])
        fn = int(cm[c, :].sum() - cm[c, c])
        support = int(cm[c, :].sum())
        prec = tp / (tp + fp + 1e-12)
        rec = tp / (tp + fn + 1e-12)
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        print(f"[{name}] prec={prec:.4f} rec={rec:.4f} f1={f1:.4f} support={support}")

     macro_f1_cm = float(np.mean([
        (2 * (int(cm[c, c]) / (int(cm[:, c].sum()) + 1e-12)) * (int(cm[c, c]) / (int(cm[c, :].sum()) + 1e-12))
         / ((int(cm[c, c]) / (int(cm[:, c].sum()) + 1e-12)) + (int(cm[c, c]) / (int(cm[c, :].sum()) + 1e-12)) + 1e-12))
        for c in range(NUM_CLASSES)
     ]))
     print(f"[MacroF1-from-CM] {macro_f1_cm:.4f}\n")

    acc = (all_pred == all_true).float().mean().item()

    # macro-F1
    f1s = []
    for c in range(NUM_CLASSES):
        tp = ((all_pred == c) & (all_true == c)).sum().item()
        fp = ((all_pred == c) & (all_true != c)).sum().item()
        fn = ((all_pred != c) & (all_true == c)).  sum().item()
        prec = tp / (tp + fp + 1e-12)
        rec = tp / (tp + fn + 1e-12)
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        f1s.append(f1)

    macro_f1 = float(np.mean(f1s))
    avg_loss = total_loss / max(total_n, 1)

    return avg_loss, acc, macro_f1


# =========================
# 4) 训练
# =========================
def main():
    "测试是否为CPU计算"
    print("model: Conv1D + BiLSTM + period")
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("device:", DEVICE)
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))

    set_seed(SEED)
    os.makedirs(os.path.dirname(SAVE_BEST_PATH), exist_ok=True)
    print("save best path:", SAVE_BEST_PATH)

    train_ds = LeavesStrongRNNDataset(CSV_PATH, split=SPLIT_TRAIN)
    test_ds  = LeavesStrongRNNDataset(CSV_PATH, split=SPLIT_TEST)

    # ===== 类别平衡：WeightedRandomSampler =====
    train_labels = train_ds.df["label_id"].astype(int).to_numpy()
    class_count = np.bincount(train_labels, minlength=NUM_CLASSES)
    print("train class_count:", class_count)

    class_weight = 1.0 / (class_count + 1e-12)
    sample_weight = class_weight[train_labels]
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weight, dtype=torch.double),
        num_samples=len(sample_weight),
        replacement=True
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    has_val = True
    try:
        val_ds = LeavesStrongRNNDataset(CSV_PATH, split=SPLIT_VAL)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    except Exception as e:
        has_val = False
        val_loader = None
        print(f"[Info] 未启用 val（原因：{e}）")

    model = ConvBiLSTMPeriodClassifier(
        input_dim=3,
        conv_channels=CONV_CHANNELS,
        hidden_dim=HIDDEN_DIM,
        period_feat_dim=PERIOD_FEAT_DIM,
        num_classes=NUM_CLASSES
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val = -1.0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        running = 0.0

        for it, (x, period_feat, lengths, y) in enumerate(train_loader, 1):
            x = x.to(DEVICE, non_blocking=True)
            period_feat = period_feat.to(DEVICE, non_blocking=True)
            lengths = lengths.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            logits = model(x, period_feat, lengths)
            loss = F.cross_entropy(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += loss.item()

            # 第一个 batch 打印一次输入信息，确认数据尺寸/最长长度/设备
            if it == 1:
                print(f"[debug] epoch={epoch} x={tuple(x.shape)} period_feat={tuple(period_feat.shape)} max_len={int(lengths.max())} device={x.device}")

            # 每 10 个 batch 打印一次训练进度
            if it % 10 == 0 or it == len(train_loader):
                elapsed = time.time() - t0
                avg_loss = running / it
                print(
                    f"Epoch {epoch:02d} | iter {it}/{len(train_loader)} | loss {avg_loss:.4f} | elapsed {elapsed:.1f}s")
                # ===== epoch 结束：做 val 并保存 best =====
        if val_loader is not None:
            v_loss, v_acc, v_f1 = eval_model(model, val_loader, verbose=False)
            print(f"[VAL] epoch {epoch:02d} | loss {v_loss:.4f} | acc {v_acc:.4f} | macroF1 {v_f1:.4f}")

            if v_f1 > best_val:
                best_val = v_f1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                torch.save(best_state, SAVE_BEST_PATH)
                print(f"[Info] Update best @ epoch={epoch} | best_macroF1={best_val:.4f} | saved to {SAVE_BEST_PATH}")

            model.train()  

    # 用最好模型
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[Info] Loaded best val model (best_macroF1={best_val:.4f})")

    test_loss, test_acc, test_f1 = eval_model(model, test_loader, verbose=True)
    print(f"[TEST] loss {test_loss:.4f} | acc {test_acc:.4f} | macroF1 {test_f1:.4f}")


if __name__ == "__main__":
    main()
