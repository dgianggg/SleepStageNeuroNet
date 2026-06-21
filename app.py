# -*- coding: utf-8 -*-
import os
import sys
import re
import uuid
import shutil
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mne

from flask import Flask, render_template, request, send_file, url_for, redirect
from werkzeug.utils import secure_filename
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ============================================================
# PATH CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

NEURONET_DIR = BASE_DIR / "NeuroNet"
sys.path.extend([
    str(NEURONET_DIR),
    str(NEURONET_DIR / "downstream"),
])

from downstream.fine_tuning import Trainer, device


UPLOAD_DIR = BASE_DIR / "uploads"
RESULT_DIR = BASE_DIR / "static" / "results"
CKPT_DIR = BASE_DIR / "checkpoints"
import gdown

CHECKPOINT_FOLDER_URL = "https://drive.google.com/drive/folders/1rFfmAdrdbtQs91X2M59HFzPK17d0ZnrV?usp=sharing"

def ensure_checkpoints():
    CKPT_DIR.mkdir(exist_ok=True)

    required_files = [
        "best_model_single.pth",
        "best_model_multiple.pth",
        "best_model_presingle.pth",
        "best_model_premultiple.pth",
    ]

    missing = []
    for filename in required_files:
        path = CKPT_DIR / filename
        if not path.exists() or path.stat().st_size < 1024 * 1024:
            missing.append(filename)

    if missing:
        print("Downloading checkpoints from Google Drive...")
        gdown.download_folder(
            CHECKPOINT_FOLDER_URL,
            output=str(CKPT_DIR),
            quiet=False,
            use_cookies=False,
        )

UPLOAD_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)
ensure_checkpoints()

CHECKPOINTS = {
    "single": {
        "pretrain": CKPT_DIR / "best_model_presingle.pth",
        "finetune": CKPT_DIR / "best_model_single.pth",
    },
    "multi": {
        "pretrain": CKPT_DIR / "best_model_premultiple.pth",
        "finetune": CKPT_DIR / "best_model_multiple.pth",
    },
}

STAGE_ID_TO_NAME = {
    0: "Wake",
    1: "N1",
    2: "N2",
    3: "N3",
    4: "REM",
}

STAGE_NAMES = ["Wake", "N1", "N2", "N3", "REM"]

TARGET_FS = 100
EPOCH_SEC = 30
LOW_CUT = 0.5
HIGH_CUT = 40.0

EEG_CANDIDATES = [
    "C4-A1", "C4_A1", "C4A1",
    "C4-M1", "C4_M1", "C4M1",
    "C3-A2", "C3_A2", "C3A2",
    "C3-M2", "C3_M2", "C3M2",
    "FPZ-CZ", "Fpz-Cz", "EEG Fpz-Cz",
]

EOG_CANDIDATES = [
    "ROC-A1", "ROC_A1", "ROCA1",
    "LOC-A2", "LOC_A2", "LOCA2",
    "E1-M2", "E1_M2", "E1M2",
    "E2-M1", "E2_M1", "E2M1",
    "EOG horizontal", "EOG",
]

MODEL_CACHE = {}

app = Flask(__name__)


# ============================================================
# MODEL UTILS
# ============================================================

def load_torch(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def prepare_pretrain_root(pretrain_pth: Path, mode: str):
    """
    Trainer trong fine_tuning.py cần pretrain checkpoint theo cấu trúc:
    ckpt_root/0/model/best_model.pth
    """
    temp_root = BASE_DIR / f"_tmp_pretrain_{mode}"
    model_dir = temp_root / "0" / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    dst = model_dir / "best_model.pth"

    if not dst.exists():
        shutil.copy2(pretrain_pth, dst)

    return temp_root


def build_model(mode: str):
    if mode in MODEL_CACHE:
        return MODEL_CACHE[mode]

    if mode not in ["single", "multi"]:
        raise ValueError("mode phải là single hoặc multi.")

    pretrain_pth = CHECKPOINTS[mode]["pretrain"]
    finetune_pth = CHECKPOINTS[mode]["finetune"]

    if not pretrain_pth.exists():
        raise FileNotFoundError(f"Không thấy checkpoint pretrain: {pretrain_pth}")

    if not finetune_pth.exists():
        raise FileNotFoundError(f"Không thấy checkpoint fine-tune: {finetune_pth}")

    fine_ckpt = load_torch(finetune_pth)
    hp = fine_ckpt.get("hyperparameter", {})

    args = SimpleNamespace()
    args.n_fold = 0
    args.ckpt_path = str(prepare_pretrain_root(pretrain_pth, mode))
    args.temporal_context_modules = hp.get("temporal_context_modules", "lstm")
    args.temporal_context_length = int(hp.get("temporal_context_length", 20))
    args.window_size = int(hp.get("window_size", 10))
    args.epochs = 1
    args.batch_size = int(hp.get("batch_size", 16))
    args.lr = float(hp.get("lr", 5e-4))
    args.embed_dim = int(hp.get("embed_dim", 256))
    args.save_name = "web_predict_temp"

    trainer = Trainer(args)
    trainer.model.load_state_dict(fine_ckpt["model_state"], strict=True)
    trainer.model.to(device)
    trainer.model.eval()

    MODEL_CACHE[mode] = (trainer, args)

    return trainer, args


# ============================================================
# EDF PREPROCESSING
# ============================================================

def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def find_channel(raw, candidates):
    channel_map = {normalize_name(ch): ch for ch in raw.ch_names}

    for cand in candidates:
        key = normalize_name(cand)
        if key in channel_map:
            return channel_map[key]

    for cand in candidates:
        key = normalize_name(cand)
        for norm_ch, original_ch in channel_map.items():
            if key in norm_ch or norm_ch in key:
                return original_ch

    raise ValueError(
        "Không tìm thấy kênh phù hợp.\n"
        f"Candidates: {candidates}\n"
        f"Available channels: {raw.ch_names}"
    )


def zscore_per_epoch_channel(x, eps=1e-8):
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    return (x - mean) / (std + eps)


def preprocess_edf_to_npz(edf_path: Path, mode: str, output_npz_path: Path, label_path: Path = None):
    """
    Tiền xử lý file EDF:
    - đọc EDF
    - chọn EEG hoặc EEG+EOG
    - lọc 0.5-40 Hz
    - resample 100 Hz
    - chia epoch 30 giây
    - z-score từng epoch/kênh
    - lưu .npz
    """
    if mode not in ["single", "multi"]:
        raise ValueError("Với file EDF thô, vui lòng chọn Single-channel hoặc Multi-channel.")

    raw = mne.io.read_raw_edf(
        str(edf_path),
        preload=True,
        verbose="ERROR",
    )

    eeg_ch = find_channel(raw, EEG_CANDIDATES)

    if mode == "single":
        picked = [eeg_ch]
        ch_names = ["EEG"]
    else:
        eog_ch = find_channel(raw, EOG_CANDIDATES)
        picked = [eeg_ch, eog_ch]
        ch_names = ["EEG", "EOG"]

    raw.pick(picked)

    sfreq = float(raw.info["sfreq"])
    high = min(HIGH_CUT, sfreq / 2.0 - 0.5)

    raw.filter(
        l_freq=LOW_CUT,
        h_freq=high,
        method="fir",
        verbose="ERROR",
    )

    if int(round(float(raw.info["sfreq"]))) != TARGET_FS:
        raw.resample(TARGET_FS, verbose="ERROR")

    data = raw.get_data().astype(np.float32)  # (C, T)
    raw.close()

    epoch_len = TARGET_FS * EPOCH_SEC
    n_epochs = data.shape[1] // epoch_len

    if n_epochs == 0:
        raise ValueError("File EDF quá ngắn, không tạo được epoch 30 giây.")

    data = data[:, :n_epochs * epoch_len]

    x = data.reshape(data.shape[0], n_epochs, epoch_len)
    x = np.transpose(x, (1, 0, 2))  # (N, C, 3000)
    x = zscore_per_epoch_channel(x)

    payload = {
        "x": x.astype(np.float32),
        "fs": np.array(TARGET_FS, dtype=np.int32),
        "ch_names": np.array(ch_names),
        "source_file": np.array(str(edf_path)),
    }

    if label_path is not None and label_path.exists():
        y = read_label_file(label_path)
        n = min(len(y), len(x))
        payload["x"] = payload["x"][:n]
        payload["y"] = y[:n].astype(np.int64)

    np.savez_compressed(output_npz_path, **payload)

    return output_npz_path


# ============================================================
# LABEL READING
# ============================================================

def parse_label_value(v):
    if pd.isna(v):
        return None

    s = str(v).strip().upper()

    if s in ["", "?", "M", "MOVEMENT", "UNKNOWN", "MT"]:
        return None

    text_map = {
        "W": 0,
        "WAKE": 0,
        "N1": 1,
        "S1": 1,
        "N2": 2,
        "S2": 2,
        "N3": 3,
        "S3": 3,
        "N4": 3,
        "S4": 3,
        "R": 4,
        "REM": 4,
    }

    if s in text_map:
        return text_map[s]

    try:
        return int(float(s))
    except ValueError:
        return None


def standardize_labels(labels):
    labels = np.asarray(labels, dtype=np.int64).flatten()

    if len(labels) == 0:
        return labels

    unique = set(np.unique(labels).tolist())
    mapped = []

    for v in labels:
        # Trường hợp chuẩn: 0=W, 1=N1, 2=N2, 3=N3, 4=REM
        if unique.issubset({0, 1, 2, 3, 4}):
            mapped.append(v if v in [0, 1, 2, 3, 4] else -1)

        # Một số file dùng 5 cho REM: 0,1,2,3,5
        elif unique.issubset({0, 1, 2, 3, 5}):
            if v in [0, 1, 2, 3]:
                mapped.append(v)
            elif v == 5:
                mapped.append(4)
            else:
                mapped.append(-1)

        # Một số file dùng 1..5: 1=W, 2=N1, 3=N2, 4=N3, 5=REM
        elif unique.issubset({1, 2, 3, 4, 5}):
            if v in [1, 2, 3, 4, 5]:
                mapped.append(v - 1)
            else:
                mapped.append(-1)

        # Trường hợp lẫn N4/REM
        else:
            if v in [0, 1, 2, 3]:
                mapped.append(v)
            elif v == 4:
                mapped.append(4)
            elif v == 5:
                mapped.append(4)
            else:
                mapped.append(-1)

    mapped = np.asarray(mapped, dtype=np.int64)
    mapped = mapped[mapped >= 0]

    return mapped


def read_label_file(label_path: Path):
    suffix = label_path.suffix.lower()
    labels = []

    if suffix == ".txt":
        text = label_path.read_text(errors="ignore", encoding="utf-8")

        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue

            tokens = re.split(r"[\s,;]+", line)

            found = None
            for token in reversed(tokens):
                found = parse_label_value(token)
                if found is not None:
                    break

            if found is not None:
                labels.append(found)

    elif suffix == ".csv":
        df = pd.read_csv(label_path, header=None)

        for _, row in df.iterrows():
            found = None
            for v in reversed(row.tolist()):
                found = parse_label_value(v)
                if found is not None:
                    break

            if found is not None:
                labels.append(found)

    elif suffix in [".xls", ".xlsx"]:
        df = pd.read_excel(label_path, header=None)

        for _, row in df.iterrows():
            found = None
            for v in reversed(row.tolist()):
                found = parse_label_value(v)
                if found is not None:
                    break

            if found is not None:
                labels.append(found)

    else:
        raise ValueError("File label chỉ hỗ trợ .txt, .csv, .xls, .xlsx")

    labels = standardize_labels(labels)

    if len(labels) == 0:
        raise ValueError("Không đọc được nhãn từ file label.")

    return labels


def attach_label_to_npz(npz_path: Path, label_path: Path, output_npz_path: Path):
    data = np.load(npz_path, allow_pickle=True)
    payload = {k: data[k] for k in data.files}

    y = read_label_file(label_path)
    n = min(len(y), len(payload["x"]))

    payload["x"] = payload["x"][:n]
    payload["y"] = y[:n].astype(np.int64)

    np.savez_compressed(output_npz_path, **payload)

    return output_npz_path


# ============================================================
# NPZ MODE
# ============================================================

def infer_mode_from_npz(npz_path: Path):
    data = np.load(npz_path, allow_pickle=True)
    x = data["x"]

    if x.ndim == 2:
        return "single"

    if x.ndim == 3:
        if x.shape[1] == 1:
            return "single"
        if x.shape[1] == 2:
            return "multi"

    raise ValueError(f"Không xác định được mode single/multi từ shape dữ liệu: {x.shape}")


# ============================================================
# DATASET FOR PREDICTION
# ============================================================

class OneSubjectDataset(Dataset):
    def __init__(self, npz_path, context_length=20):
        self.npz_path = Path(npz_path)
        self.context_length = int(context_length)

        data = np.load(self.npz_path, allow_pickle=True)
        self.x = data["x"].astype(np.float32)

        if self.x.ndim == 2:
            self.x = self.x[:, None, :]
        elif self.x.ndim == 3:
            pass
        else:
            raise ValueError(f"Shape dữ liệu không hỗ trợ: {self.x.shape}")

        self.has_y = "y" in data.files

        if self.has_y:
            y = data["y"].astype(np.int64).flatten()
            n = min(len(self.x), len(y))
            self.x = self.x[:n]
            self.y = y[:n]
        else:
            self.y = np.full((len(self.x),), -1, dtype=np.int64)

        self.n_epochs = len(self.x)
        self.index = []

        for start in range(0, self.n_epochs, self.context_length):
            end = min(start + self.context_length, self.n_epochs)
            self.index.append((start, end))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        start, end = self.index[idx]

        x_seg = self.x[start:end]
        y_seg = self.y[start:end]

        valid_len = end - start
        L = self.context_length

        mask = np.zeros(L, dtype=np.bool_)
        mask[:valid_len] = True

        epoch_indices = np.arange(start, end)

        if valid_len < L:
            pad_len = L - valid_len

            x_pad = np.repeat(x_seg[-1:], pad_len, axis=0)
            y_pad = np.repeat(y_seg[-1:], pad_len, axis=0)
            idx_pad = np.repeat(epoch_indices[-1:], pad_len, axis=0)

            x_seg = np.concatenate([x_seg, x_pad], axis=0)
            y_seg = np.concatenate([y_seg, y_pad], axis=0)
            epoch_indices = np.concatenate([epoch_indices, idx_pad], axis=0)

        return (
            torch.tensor(x_seg, dtype=torch.float32),
            torch.tensor(y_seg, dtype=torch.long),
            torch.tensor(mask, dtype=torch.bool),
            torch.tensor(epoch_indices, dtype=torch.long),
        )


# ============================================================
# METRICS
# ============================================================

def compute_metrics(y_true, y_pred):
    label_ids = [0, 1, 2, 3, 4]
    label_names = ["Wake", "N1", "N2", "N3", "REM"]

    report = classification_report(
        y_true,
        y_pred,
        labels=label_ids,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
    }

    cm = confusion_matrix(y_true, y_pred, labels=label_ids)

    cm_df = pd.DataFrame(
        cm,
        index=["True_Wake", "True_N1", "True_N2", "True_N3", "True_REM"],
        columns=["Pred_Wake", "Pred_N1", "Pred_N2", "Pred_N3", "Pred_REM"],
    )

    return metrics, cm_df


def make_auto_conclusion(preds, probs, mode, metrics=None, y_true=None, y_pred=None):
    """
    Tạo kết luận tự động cho kết quả dự đoán.
    - Nếu có nhãn thật: kết luận thêm mức độ so sánh hypnogram, số epoch sai, Accuracy/F1.
    - Nếu không có nhãn thật: chỉ kết luận theo phân bố dự đoán và confidence.
    """
    preds = np.asarray(preds).astype(int)
    probs = np.asarray(probs)

    n_epochs = len(preds)
    counts = np.bincount(preds, minlength=5)
    percentages = counts / max(n_epochs, 1) * 100

    main_stage_id = int(np.argmax(counts))
    main_stage = STAGE_ID_TO_NAME[main_stage_id]
    main_percent = percentages[main_stage_id]

    confidence = np.max(probs, axis=1)
    mean_conf = float(np.mean(confidence))
    high_conf_percent = float(np.mean(confidence >= 0.8) * 100)
    low_conf_percent = float(np.mean(confidence < 0.5) * 100)

    transition_count = int(np.sum(preds[1:] != preds[:-1])) if n_epochs > 1 else 0

    if mean_conf >= 0.8:
        conf_level = "cao"
    elif mean_conf >= 0.6:
        conf_level = "khá"
    else:
        conf_level = "chưa cao"

    conclusion = (
        f"Mô hình {mode} đã phân loại {n_epochs} epoch tín hiệu, "
        f"mỗi epoch tương ứng 30 giây. "
        f"Giai đoạn ngủ xuất hiện nhiều nhất trong kết quả dự đoán là {main_stage}, "
        f"chiếm khoảng {main_percent:.1f}% tổng số epoch. "
        f"Confidence trung bình của mô hình là {mean_conf:.3f}, "
        f"thể hiện mức độ tin cậy {conf_level}. "
        f"Có {high_conf_percent:.1f}% epoch có confidence ≥ 0.8 và "
        f"{low_conf_percent:.1f}% epoch có confidence < 0.5. "
        f"Hypnogram dự đoán ghi nhận {transition_count} lần chuyển trạng thái "
        f"giữa các giai đoạn ngủ. "
    )

    if metrics is not None and y_true is not None and y_pred is not None:
        acc = metrics["accuracy"] * 100
        macro_f1 = metrics["macro_f1"] * 100
        weighted_f1 = metrics["weighted_f1"] * 100

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        wrong_count = int(np.sum(y_true != y_pred))
        wrong_percent = wrong_count / max(len(y_true), 1) * 100

        if acc >= 85 and macro_f1 >= 80:
            level = "rất tốt"
        elif acc >= 75 and macro_f1 >= 70:
            level = "tốt"
        elif acc >= 65 and macro_f1 >= 60:
            level = "khá"
        else:
            level = "cần cải thiện"

        conclusion += (
            f"Khi so sánh với nhãn thật, mô hình đạt Accuracy = {acc:.2f}%, "
            f"Macro-F1 = {macro_f1:.2f}% và Weighted-F1 = {weighted_f1:.2f}%. "
            f"Biểu đồ so sánh hypnogram cho thấy đường dự đoán có mức độ tương đồng {level} "
            f"so với đường nhãn thật. "
            f"Số epoch dự đoán sai là {wrong_count}, chiếm khoảng {wrong_percent:.1f}% "
            f"tổng số epoch có nhãn. "
            f"Các sai lệch chủ yếu thường xuất hiện tại vùng chuyển tiếp giữa các giai đoạn ngủ "
            f"như Wake–N1, N1–N2, N2–N3 hoặc REM. "
            f"Nhìn chung, kết quả cho thấy mô hình có khả năng nhận diện cấu trúc giấc ngủ "
            f"và có thể hỗ trợ quá trình chấm điểm giai đoạn ngủ tự động."
        )
    else:
        conclusion += (
            "Do dữ liệu đầu vào không có nhãn thật, hệ thống chỉ hiển thị kết quả dự đoán "
            "và chưa thể đánh giá Accuracy, F1-score hay mức độ sai lệch so với chuyên gia."
        )

    return conclusion

# ============================================================
# PLOTS
# ============================================================

def plot_eeg_preview(npz_path: Path, save_path: Path):
    data = np.load(npz_path, allow_pickle=True)
    x = data["x"].astype(np.float32)

    if "fs" in data.files:
        fs = int(np.asarray(data["fs"]).item())
    else:
        fs = TARGET_FS

    # Chuẩn hóa shape về (N, C, T)
    if x.ndim == 2:
        # (N, T) -> single channel
        x = x[:, None, :]
    elif x.ndim == 3:
        # (N, C, T)
        pass
    else:
        raise ValueError(f"Shape dữ liệu không hỗ trợ để vẽ preview: {x.shape}")

    first_epoch = x[0]  # (C, T)
    n_channels = first_epoch.shape[0]
    t = np.arange(first_epoch.shape[1]) / fs

    # Tên kênh nếu có trong file npz
    if "ch_names" in data.files:
        try:
            ch_names = [str(c) for c in data["ch_names"]]
        except Exception:
            ch_names = []
    else:
        ch_names = []

    if n_channels == 1:
        plt.figure(figsize=(14, 4))
        plt.plot(t, first_epoch[0])
        plt.title("Tín hiệu EEG đầu vào")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

    else:
        plt.figure(figsize=(14, 6))

        for ch_idx in range(n_channels):
            if ch_idx < len(ch_names):
                label = ch_names[ch_idx]
            else:
                label = f"Channel {ch_idx + 1}"

            # Dịch tín hiệu lên/xuống để các kênh không đè lên nhau
            offset = ch_idx * 8.0
            plt.plot(t, first_epoch[ch_idx] + offset, linewidth=1.0, label=label)

        plt.title("Tín hiệu đầu vào đa kênh EEG/EOG")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude + offset")
        plt.grid(alpha=0.3)
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()


def plot_hypnogram(pred_df: pd.DataFrame, save_path: Path):
    y = pred_df["pred_id"].values
    x = pred_df["epoch_index"].values

    plt.figure(figsize=(14, 4))
    plt.step(x, y, where="post", label="Predicted labels")
    plt.yticks([0, 1, 2, 3, 4], ["Wake", "N1", "N2", "N3", "REM"])
    plt.gca().invert_yaxis()
    plt.xlabel("Epoch")
    plt.ylabel("Sleep stage")
    plt.title("Biểu đồ hypnogram dự đoán")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_true_hypnogram(pred_df: pd.DataFrame, save_path: Path):
    if "true_id" not in pred_df.columns:
        return None

    df = pred_df.dropna(subset=["true_id"]).copy()
    if df.empty:
        return None

    x = df["epoch_index"].values
    y = df["true_id"].astype(int).values

    plt.figure(figsize=(14, 4))
    plt.step(x, y, where="post", label="Ground truth labels")
    plt.yticks([0, 1, 2, 3, 4], ["Wake", "N1", "N2", "N3", "REM"])
    plt.gca().invert_yaxis()
    plt.xlabel("Epoch")
    plt.ylabel("Sleep stage")
    plt.title("Biểu đồ hypnogram nhãn thật")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    return save_path


def plot_compare_hypnogram(pred_df: pd.DataFrame, save_path: Path):
    if "true_id" not in pred_df.columns:
        return None

    df = pred_df.dropna(subset=["true_id"]).copy()
    if df.empty:
        return None

    x = df["epoch_index"].values
    y_true = df["true_id"].astype(int).values
    y_pred = df["pred_id"].astype(int).values

    plt.figure(figsize=(14, 5))
    plt.step(x, y_true, where="post", linewidth=1.4, label="Nhãn thật")
    plt.step(x, y_pred + 0.08, where="post", linewidth=1.1, alpha=0.85, label="Dự đoán")

    wrong = y_true != y_pred
    if np.any(wrong):
        plt.scatter(x[wrong], y_pred[wrong] + 0.08, s=14, marker="x", label="Epoch dự đoán sai")

    plt.yticks([0, 1, 2, 3, 4], ["Wake", "N1", "N2", "N3", "REM"])
    plt.gca().invert_yaxis()
    plt.xlabel("Epoch")
    plt.ylabel("Sleep stage")
    plt.title("So sánh hypnogram nhãn thật và dự đoán")
    plt.grid(alpha=0.3)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    return save_path



def plot_stage_distribution(pred_df: pd.DataFrame, save_path: Path):
    counts = pred_df["pred_stage"].value_counts().reindex(
        ["Wake", "N1", "N2", "N3", "REM"],
        fill_value=0,
    )

    plt.figure(figsize=(9, 5))
    plt.bar(counts.index, counts.values)
    plt.title("Phân bố giai đoạn ngủ dự đoán")
    plt.xlabel("Giai đoạn ngủ")
    plt.ylabel("Số epoch")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_confidence_curve(pred_df: pd.DataFrame, save_path: Path):
    x = pred_df["epoch_index"].values
    confidence = pred_df["confidence"].values

    plt.figure(figsize=(14, 4))
    plt.plot(x, confidence, linewidth=1.2, label="Confidence")
    plt.axhline(0.8, linestyle="--", linewidth=1, label="Ngưỡng tin cậy cao 0.8")
    plt.axhline(0.5, linestyle="--", linewidth=1, label="Ngưỡng tin cậy thấp 0.5")

    plt.title("Độ tin cậy dự đoán theo epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Confidence")
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_confusion_matrix_image(y_true, y_pred, save_path: Path):
    label_ids = [0, 1, 2, 3, 4]
    label_names = ["Wake", "N1", "N2", "N3", "REM"]

    cm = confusion_matrix(y_true, y_pred, labels=label_ids)

    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Ma trận nhầm lẫn")
    plt.xlabel("Nhãn dự đoán")
    plt.ylabel("Nhãn thật")

    plt.xticks(np.arange(len(label_names)), label_names)
    plt.yticks(np.arange(len(label_names)), label_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=10,
            )

    plt.colorbar(label="Số epoch")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# ============================================================
# PREDICT
# ============================================================

def predict_npz(npz_path: Path, mode: str, output_id: str, data_type: str):
    if mode == "auto":
        mode = infer_mode_from_npz(npz_path)

    trainer, args = build_model(mode)

    dataset = OneSubjectDataset(
        npz_path=npz_path,
        context_length=args.temporal_context_length,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    rows = []
    y_true = []
    y_pred = []
    pred_list = []
    prob_list = []

    with torch.no_grad():
        for x, y, mask, epoch_idx in loader:
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)
            epoch_idx = epoch_idx.to(device)

            out = trainer.model(x)
            prob = torch.softmax(out, dim=-1)
            pred = torch.argmax(prob, dim=-1)

            valid_epoch_idx = epoch_idx[mask].detach().cpu().numpy()
            valid_true = y[mask].detach().cpu().numpy()
            valid_pred = pred[mask].detach().cpu().numpy()
            valid_prob = prob[mask].detach().cpu().numpy()

            for ep, yt, yp, pr in zip(valid_epoch_idx, valid_true, valid_pred, valid_prob):
                conf = float(np.max(pr))

                row = {
                    "epoch_index": int(ep),
                    "pred_id": int(yp),
                    "pred_stage": STAGE_ID_TO_NAME[int(yp)],
                    "prob_Wake": float(pr[0]),
                    "prob_N1": float(pr[1]),
                    "prob_N2": float(pr[2]),
                    "prob_N3": float(pr[3]),
                    "prob_REM": float(pr[4]),
                    "confidence": conf,
                }

                pred_list.append(int(yp))
                prob_list.append(pr)

                if int(yt) >= 0:
                    row["true_id"] = int(yt)
                    row["true_stage"] = STAGE_ID_TO_NAME[int(yt)]
                    y_true.append(int(yt))
                    y_pred.append(int(yp))

                rows.append(row)

    pred_df = pd.DataFrame(rows)

    if pred_df.empty:
        raise ValueError("Không có epoch hợp lệ để dự đoán.")

    csv_path = RESULT_DIR / f"{output_id}_predictions.csv"
    eeg_img_path = RESULT_DIR / f"{output_id}_eeg_preview.png"
    hyp_img_path = RESULT_DIR / f"{output_id}_hypnogram.png"
    dist_img_path = RESULT_DIR / f"{output_id}_stage_distribution.png"
    confidence_img_path = RESULT_DIR / f"{output_id}_confidence.png"
    true_hyp_img_path = RESULT_DIR / f"{output_id}_true_hypnogram.png"
    compare_hyp_img_path = RESULT_DIR / f"{output_id}_compare_hypnogram.png"

    pred_df.to_csv(csv_path, index=False)

    plot_eeg_preview(npz_path, eeg_img_path)
    plot_hypnogram(pred_df, hyp_img_path)
    plot_stage_distribution(pred_df, dist_img_path)
    plot_confidence_curve(pred_df, confidence_img_path)

    metrics = None
    confusion_url = None
    true_hyp_url = None
    compare_hyp_url = None

    if len(y_true) > 0:
        plot_true_hypnogram(pred_df, true_hyp_img_path)
        plot_compare_hypnogram(pred_df, compare_hyp_img_path)
        true_hyp_url = url_for("static", filename=f"results/{output_id}_true_hypnogram.png")
        compare_hyp_url = url_for("static", filename=f"results/{output_id}_compare_hypnogram.png")

        metrics, cm_df = compute_metrics(y_true, y_pred)

        metrics_path = RESULT_DIR / f"{output_id}_metrics.csv"
        cm_csv_path = RESULT_DIR / f"{output_id}_confusion_matrix.csv"
        cm_img_path = RESULT_DIR / f"{output_id}_confusion_matrix.png"

        pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
        cm_df.to_csv(cm_csv_path)
        plot_confusion_matrix_image(y_true, y_pred, cm_img_path)

        confusion_url = url_for("static", filename=f"results/{output_id}_confusion_matrix.png")

    conclusion = make_auto_conclusion(
        preds=np.asarray(pred_list),
        probs=np.asarray(prob_list),
        mode=mode,
        metrics=metrics,
        y_true=y_true if len(y_true) > 0 else None,
        y_pred=y_pred if len(y_pred) > 0 else None,
    )

    result = {
        "data_type": data_type,
        "mode": mode,
        "metrics": metrics,
        "preview": pred_df.head(20).to_dict(orient="records"),

        "csv_path": csv_path,
        "eeg_img_path": eeg_img_path,
        "hyp_img_path": hyp_img_path,
        "dist_img_path": dist_img_path,
        "confidence_img_path": confidence_img_path,
        "true_hyp_img_path": true_hyp_img_path,
        "compare_hyp_img_path": compare_hyp_img_path,

        "eeg_url": url_for("static", filename=f"results/{output_id}_eeg_preview.png"),
        "hyp_url": url_for("static", filename=f"results/{output_id}_hypnogram.png"),
        "dist_url": url_for("static", filename=f"results/{output_id}_stage_distribution.png"),
        "confidence_url": url_for("static", filename=f"results/{output_id}_confidence.png"),
        "true_hyp_url": true_hyp_url,
        "compare_hyp_url": compare_hyp_url,
        "confusion_url": confusion_url,
        "csv_url": url_for("download_result", filename=f"{output_id}_predictions.csv"),
        "conclusion": conclusion,
    }

    return result


# ============================================================
# ROUTES
# ============================================================

def save_result_metadata(output_id: str, result: dict):
    """Lưu metadata kết quả để trang /result/<output_id> có thể đọc lại."""
    safe_result = {}
    for key, value in result.items():
        # Các trường *_path là Path object, không cần đưa ra template kết quả.
        if key.endswith("_path"):
            continue
        safe_result[key] = value

    metadata_path = RESULT_DIR / f"{output_id}_result.json"
    metadata_path.write_text(
        json.dumps(safe_result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata_path


def load_result_metadata(output_id: str):
    metadata_path = RESULT_DIR / f"{output_id}_result.json"
    if not metadata_path.exists():
        raise FileNotFoundError("Không tìm thấy kết quả. Vui lòng tải file và dự đoán lại.")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/upload", methods=["GET", "POST"])
def upload():
    error = None

    if request.method == "POST":
        try:
            data_type = request.form.get("data_type", "processed")

            output_id = str(uuid.uuid4())[:8]
            job_dir = UPLOAD_DIR / output_id
            job_dir.mkdir(parents=True, exist_ok=True)

            if data_type == "processed":
                uploaded_file = request.files.get("processed_file")

                if uploaded_file is None or uploaded_file.filename == "":
                    raise ValueError("Bạn chưa chọn file .npz đã tiền xử lý.")

                filename = secure_filename(uploaded_file.filename)
                suffix = Path(filename).suffix.lower()

                if suffix != ".npz":
                    raise ValueError("File tiền xử lý chỉ hỗ trợ định dạng .npz.")

                upload_path = job_dir / filename
                uploaded_file.save(upload_path)

                npz_path = upload_path
                mode = infer_mode_from_npz(npz_path)
                data_type_label = "File tiền xử lý"

            elif data_type == "raw":
                mode = request.form.get("model_mode", "single")

                if mode not in ["single", "multi"]:
                    raise ValueError("Với file thô, vui lòng chọn Single-channel hoặc Multi-channel.")

                raw_file = request.files.get("raw_data_file")
                label_file = request.files.get("raw_label_file")

                if raw_file is None or raw_file.filename == "":
                    raise ValueError("Bạn chưa chọn file dữ liệu thô .edf.")

                if label_file is None or label_file.filename == "":
                    raise ValueError("Với file thô, vui lòng chọn thêm file hypnogram/label.")

                raw_name = secure_filename(raw_file.filename)
                raw_suffix = Path(raw_name).suffix.lower()

                if raw_suffix != ".edf":
                    raise ValueError("File dữ liệu thô chỉ hỗ trợ định dạng .edf.")

                label_name = secure_filename(label_file.filename)
                label_suffix = Path(label_name).suffix.lower()

                if label_suffix not in [".txt", ".csv", ".xls", ".xlsx"]:
                    raise ValueError("File label chỉ hỗ trợ .txt, .csv, .xls, .xlsx.")

                raw_path = job_dir / raw_name
                label_path = job_dir / label_name

                raw_file.save(raw_path)
                label_file.save(label_path)

                npz_path = job_dir / f"{Path(raw_name).stem}_processed_{mode}.npz"

                preprocess_edf_to_npz(
                    edf_path=raw_path,
                    mode=mode,
                    output_npz_path=npz_path,
                    label_path=label_path,
                )

                data_type_label = "File thô"

            else:
                raise ValueError("Loại dữ liệu không hợp lệ.")

            result = predict_npz(
                npz_path=npz_path,
                mode=mode,
                output_id=output_id,
                data_type=data_type_label,
            )
            save_result_metadata(output_id, result)
            return redirect(url_for("result_page", output_id=output_id))

        except Exception as e:
            error = str(e)

    return render_template("upload.html", error=error)


@app.route("/result/<output_id>", methods=["GET"])
def result_page(output_id):
    try:
        result = load_result_metadata(output_id)
        return render_template("result.html", result=result)
    except Exception as e:
        return render_template("upload.html", error=str(e))


@app.route("/download/<filename>")
def download_result(filename):
    file_path = RESULT_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(f"Không thấy file: {file_path}")

    return send_file(file_path, as_attachment=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
