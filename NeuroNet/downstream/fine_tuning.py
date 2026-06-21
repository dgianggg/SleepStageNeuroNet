# -*- coding:utf-8 -*-
import os
import sys
sys.path.extend([os.path.abspath("."), os.path.abspath("..")])

import argparse
import random
import warnings
import copy
import shutil
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as opt
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

try:
    from mamba_ssm import Mamba
except Exception:
    Mamba = None

from models.neuronet.model import NeuroNet, NeuroNetEncoderWrapper

warnings.filterwarnings("ignore")

SEED = 777
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--n_fold", default=0, type=int)
    parser.add_argument("--ckpt_path", required=True, type=str)

    parser.add_argument("--temporal_context_length", default=20, type=int)
    parser.add_argument("--window_size", default=10, type=int)

    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--lr", default=5e-4, type=float)
    parser.add_argument("--embed_dim", default=256, type=int)

    parser.add_argument(
        "--temporal_context_modules",
        choices=["lstm", "mha", "lstm_mha", "mamba"],
        default="lstm",
    )

    parser.add_argument("--save_name", default="fine_tuning", type=str)

    return parser.parse_args()


class SequenceDataset(Dataset):
    def __init__(self, paths, temporal_context_length=20, window_size=10):
        self.paths = paths
        self.temporal_context_length = temporal_context_length
        self.window_size = window_size

        self.x, self.y = self.load_all_subjects()

        self.x = torch.tensor(self.x, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long)

        print("Loaded sequence data:", self.x.shape, self.y.shape)

    def load_all_subjects(self):
        total_x = []
        total_y = []

        for path in self.paths:
            data = np.load(path, allow_pickle=True)

            x = data["x"]
            y = data["y"]

            # Hỗ trợ:
            # single-channel: (N, 1, 3000)
            # multi-channel:  (N, 2, 3000)
            # fallback:       (N, 3000) -> (N, 1, 3000)
            if x.ndim == 2:
                x = x[:, None, :]
            elif x.ndim == 3:
                pass
            else:
                raise ValueError(f"Shape không hỗ trợ ở file {path}: {x.shape}")

            x = x.astype(np.float32)
            y = y.astype(np.int64)

            seq_x = self.make_sequences(x)
            seq_y = self.make_sequences(y)

            if len(seq_x) == 0:
                continue

            total_x.append(seq_x)
            total_y.append(seq_y)

        if len(total_x) == 0:
            raise RuntimeError("Không tạo được sequence nào. Kiểm tra paths hoặc temporal_context_length.")

        total_x = np.concatenate(total_x, axis=0)
        total_y = np.concatenate(total_y, axis=0)

        return total_x, total_y

    def make_sequences(self, arr):
        n = len(arr)
        L = self.temporal_context_length
        step = self.window_size

        if n < L:
            return np.array([])

        seqs = []

        for i in range(0, n - L + 1, step):
            seqs.append(arr[i:i + L])

        # Thêm đoạn cuối để không bỏ phần cuối subject
        last = arr[n - L:n]
        if len(seqs) == 0 or not np.array_equal(seqs[-1], last):
            seqs.append(last)

        return np.asarray(seqs)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class CLSNeuroNetEncoderWrapper(NeuroNetEncoderWrapper):
    """
    Bản wrapper dùng CLS token thay vì mean token.

    NeuroNet paper đưa class token vào TCM:
    Cls Token = Encoder(EEG Epoch)
    Sleep Stage = TCM({Cls Token})

    Vì vậy fine-tune nên lấy x[:, 0, :] thay vì mean(x[:, 1:, :]).
    """
    def forward(self, x, semantic_token=True):
        # frame backbone
        x = self.make_frame(x)
        x = self.frame_backbone(x)

        # embed patches
        x = self.patch_embed(x)

        # add pos embed without cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for block in self.encoder_block:
            x = block(x)

        x = self.encoder_norm(x)

        # Lấy CLS token
        x = x[:, 0, :]
        return x


class TemporalContextModule(nn.Module):
    def __init__(self, backbone, backbone_final_length, embed_dim):
        super().__init__()

        self.backbone = self.freeze_backbone(backbone)
        self.backbone_final_length = backbone_final_length
        self.embed_dim = embed_dim

        self.embed_layer = nn.Sequential(
            nn.Linear(backbone_final_length, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    @staticmethod
    def freeze_backbone(backbone):
        # Freeze toàn bộ backbone trước
        for param in backbone.parameters():
            param.requires_grad = False

        # Fine-tune nhẹ: mở khóa transformer block cuối và encoder_norm nếu có
        if hasattr(backbone, "encoder_block"):
            try:
                for param in backbone.encoder_block[-1].parameters():
                    param.requires_grad = True
            except Exception:
                pass

        if hasattr(backbone, "encoder_norm"):
            for param in backbone.encoder_norm.parameters():
                param.requires_grad = True

        return backbone

    def apply_backbone(self, x):
        """
        Input:
        - single: (B, L, 1, 3000)
        - multi:  (B, L, 2, 3000)

        Output:
        - (B, L, embed_dim)
        """
        outs = []
        L = x.shape[1]

        for t in range(L):
            x_t = x[:, t]  # (B, C, T)

            feat = self.backbone(x_t)  # (B, backbone_dim)
            feat = self.embed_layer(feat)
            outs.append(feat)

        outs = torch.stack(outs, dim=1)
        return outs


class LSTM_TCM(TemporalContextModule):
    def __init__(self, backbone, backbone_final_length, embed_dim):
        super().__init__(backbone, backbone_final_length, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=embed_dim,
            num_layers=2,
            batch_first=True,
        )
        self.fc = nn.Linear(embed_dim, 5)

    def forward(self, x):
        x = self.apply_backbone(x)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x


class MHA_TCM(TemporalContextModule):
    def __init__(self, backbone, backbone_final_length, embed_dim):
        super().__init__(backbone, backbone_final_length, embed_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.fc = nn.Linear(embed_dim, 5)

    def forward(self, x):
        x = self.apply_backbone(x)
        x = self.transformer(x)
        x = self.fc(x)
        return x


class LSTM_MHA_TCM(TemporalContextModule):
    def __init__(self, backbone, backbone_final_length, embed_dim):
        super().__init__(backbone, backbone_final_length, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=embed_dim,
            num_layers=1,
            batch_first=True,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.fc = nn.Linear(embed_dim, 5)

    def forward(self, x):
        x = self.apply_backbone(x)
        x, _ = self.lstm(x)
        x = self.transformer(x)
        x = self.fc(x)
        return x


class MAMBA_TCM(TemporalContextModule):
    def __init__(self, backbone, backbone_final_length, embed_dim):
        super().__init__(backbone, backbone_final_length, embed_dim)

        if Mamba is None:
            raise ImportError(
                "Chưa cài được mamba_ssm. Hãy dùng --temporal_context_modules lstm trước."
            )

        self.mamba = nn.Sequential(
            Mamba(
                d_model=embed_dim,
                d_state=16,
                d_conv=4,
                expand=2,
            )
        )
        self.fc = nn.Linear(embed_dim, 5)

    def forward(self, x):
        x = self.apply_backbone(x)
        x = self.mamba(x)
        x = self.fc(x)
        return x


class Trainer:
    def __init__(self, args):
        self.args = args

        self.ckpt_file = os.path.join(
            self.args.ckpt_path,
            str(self.args.n_fold),
            "model",
            "best_model.pth",
        )

        print("Checkpoint File Path:", self.ckpt_file)

        try:
            self.ckpt = torch.load(self.ckpt_file, map_location="cpu", weights_only=False)
        except TypeError:
            self.ckpt = torch.load(self.ckpt_file, map_location="cpu")

        ckpt_paths = self.ckpt["paths"]

        # FIX 1:
        # train_paths dùng để train fine-tune
        # ft_paths dùng làm validation để chọn best epoch
        # eval_paths chỉ dùng test cuối cùng
        self.train_paths = ckpt_paths.get("train_paths", None)
        self.val_paths = ckpt_paths.get("ft_paths", None)
        self.test_paths = ckpt_paths.get("eval_paths", None)

        if self.train_paths is None:
            raise KeyError("Checkpoint không có paths['train_paths']. Kiểm tra lại file checkpoint pretrain.")
        if self.val_paths is None:
            raise KeyError("Checkpoint không có paths['ft_paths']. Kiểm tra lại file checkpoint pretrain.")
        if self.test_paths is None:
            raise KeyError("Checkpoint không có paths['eval_paths']. Kiểm tra lại file checkpoint pretrain.")

        self.model = self.build_model().to(device)

        self.optimizer = opt.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.args.lr,
        )
        self.scheduler = opt.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.args.epochs,
        )

        # Giữ nguyên logic: CrossEntropyLoss thường.
        # Chưa thêm class weight / focal loss để baseline sạch.
        self.criterion = nn.CrossEntropyLoss()

        self.output_dir = os.path.join(
            self.args.ckpt_path,
            str(self.args.n_fold),
            self.args.save_name,
        )
        self.metrics_dir = os.path.join(self.output_dir, "metrics")

        # FIX 2:
        # Xóa metrics cũ để tránh append kết quả nhiều lần chạy vào cùng file.
        if os.path.exists(self.metrics_dir):
            shutil.rmtree(self.metrics_dir)

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)

    def build_model(self):
        model_parameter = self.ckpt["model_parameter"]

        pretrained_model = NeuroNet(**model_parameter)
        pretrained_model.load_state_dict(self.ckpt["model_state"])

        backbone = CLSNeuroNetEncoderWrapper(
            fs=model_parameter["fs"],
            second=model_parameter["second"],
            time_window=model_parameter["time_window"],
            time_step=model_parameter["time_step"],
            frame_backbone=pretrained_model.frame_backbone,
            patch_embed=pretrained_model.autoencoder.patch_embed,
            encoder_block=pretrained_model.autoencoder.encoder_block,
            encoder_norm=pretrained_model.autoencoder.encoder_norm,
            cls_token=pretrained_model.autoencoder.cls_token,
            pos_embed=pretrained_model.autoencoder.pos_embed,
            final_length=pretrained_model.autoencoder.embed_dim,
        )

        tcm_cls = self.get_tcm_class()

        model = tcm_cls(
            backbone=backbone,
            backbone_final_length=pretrained_model.autoencoder.embed_dim,
            embed_dim=self.args.embed_dim,
        )

        return model

    def get_tcm_class(self):
        if self.args.temporal_context_modules == "lstm":
            return LSTM_TCM
        if self.args.temporal_context_modules == "mha":
            return MHA_TCM
        if self.args.temporal_context_modules == "lstm_mha":
            return LSTM_MHA_TCM
        if self.args.temporal_context_modules == "mamba":
            return MAMBA_TCM

        raise ValueError(self.args.temporal_context_modules)

    def get_loss(self, pred, real):
        # pred: (B, L, 5)
        # real: (B, L)
        pred = pred.reshape(-1, pred.size(-1))
        real = real.reshape(-1)

        loss = self.criterion(pred, real)

        return loss, pred, real

    def evaluate(self, loader):
        self.model.eval()
        losses = []
        epoch_real = []
        epoch_pred = []

        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                y = y.to(device)

                out = self.model(x)
                loss, pred, real = self.get_loss(out, y)

                pred_label = torch.argmax(pred, dim=-1)

                losses.append(loss.detach().cpu().item())
                epoch_real.extend(real.detach().cpu().numpy().tolist())
                epoch_pred.extend(pred_label.detach().cpu().numpy().tolist())

        avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0
        acc = accuracy_score(epoch_real, epoch_pred)
        mf1 = f1_score(epoch_real, epoch_pred, average="macro", zero_division=0)

        return avg_loss, acc, mf1, epoch_real, epoch_pred

    def train(self):
        print("Fine-tuning train subjects:", len(self.train_paths))
        print("Validation subjects:", len(self.val_paths))
        print("Final test subjects:", len(self.test_paths))
        print("TCM:", self.args.temporal_context_modules)

        train_dataset = SequenceDataset(
            paths=self.train_paths,
            temporal_context_length=self.args.temporal_context_length,
            window_size=self.args.window_size,
        )

        val_dataset = SequenceDataset(
            paths=self.val_paths,
            temporal_context_length=self.args.temporal_context_length,
            window_size=self.args.temporal_context_length,
        )

        test_dataset = SequenceDataset(
            paths=self.test_paths,
            temporal_context_length=self.args.temporal_context_length,
            window_size=self.args.temporal_context_length,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            drop_last=False,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            drop_last=False,
        )

        best_mf1 = -1.0
        best_state = None
        best_real = None
        best_pred = None
        best_epoch = -1
        best_train_loss = None
        best_val_loss = None

        for epoch in range(self.args.epochs):
            self.model.train()
            train_losses = []

            for x, y in train_loader:
                x = x.to(device)
                y = y.to(device)

                self.optimizer.zero_grad()

                out = self.model(x)
                loss, _, _ = self.get_loss(out, y)

                loss.backward()
                self.optimizer.step()

                train_losses.append(loss.detach().cpu().item())

            train_loss = float(np.mean(train_losses)) if len(train_losses) > 0 else 0.0

            val_loss, val_acc, val_mf1, val_real, val_pred = self.evaluate(val_loader)

            self.save_epoch_metrics(
                epoch=epoch,
                train_loss=train_loss,
                test_loss=val_loss,
                y_true=val_real,
                y_pred=val_pred,
            )

            # FIX 3:
            # Chọn best model theo validation Macro-F1, không dùng test set.
            if val_mf1 > best_mf1:
                best_mf1 = val_mf1

                # FIX 4:
                # deepcopy để best_state không bị thay đổi ở các epoch sau.
                best_state = copy.deepcopy(self.model.state_dict())
                best_real = val_real
                best_pred = val_pred
                best_epoch = epoch
                best_train_loss = train_loss
                best_val_loss = val_loss

                self.save_best_metrics(
                    epoch=epoch,
                    train_loss=train_loss,
                    test_loss=val_loss,
                    y_true=val_real,
                    y_pred=val_pred,
                )

            print(
                "[Epoch] {0:03d} | Train Loss {1:.4f} | Val Loss {2:.4f} | Val ACC {3:.4f} | Val Macro-F1 {4:.4f}".format(
                    epoch + 1,
                    train_loss,
                    val_loss,
                    val_acc,
                    val_mf1,
                )
            )

            self.scheduler.step()

        if best_state is None:
            raise RuntimeError("Không có best_state. Kiểm tra train/validation loader.")

        # Load best validation model rồi test cuối cùng trên eval_paths.
        self.model.load_state_dict(best_state)

        test_loss, test_acc, test_mf1, test_real, test_pred = self.evaluate(test_loader)

        print(
            "[Final Test] Best Epoch {0:03d} | Test Loss {1:.4f} | Test ACC {2:.4f} | Test Macro-F1 {3:.4f}".format(
                best_epoch + 1,
                test_loss,
                test_acc,
                test_mf1,
            )
        )

        self.save_final_test_metrics(
            best_epoch,
            test_loss,
            test_real,
            test_pred,
        )

        self.save_ckpt(
            model_state=best_state,
            val_real=best_real,
            val_pred=best_pred,
            test_real=test_real,
            test_pred=test_pred,
            best_epoch=best_epoch,
            best_train_loss=best_train_loss,
            best_val_loss=best_val_loss,
            final_test_loss=test_loss,
        )

    def build_metric_tables(self, epoch, train_loss, test_loss, y_true, y_pred):
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

        acc = accuracy_score(y_true, y_pred)

        rows = []
        for label_name in label_names:
            rows.append(
                {
                    "epoch": epoch + 1,
                    "class": label_name,
                    "precision": report[label_name]["precision"],
                    "recall": report[label_name]["recall"],
                    "f1_score": report[label_name]["f1-score"],
                    "support": report[label_name]["support"],
                    "accuracy": acc,
                    "macro_f1": report["macro avg"]["f1-score"],
                    "weighted_f1": report["weighted avg"]["f1-score"],
                }
            )

        summary = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "accuracy": acc,
            "macro_precision": report["macro avg"]["precision"],
            "macro_recall": report["macro avg"]["recall"],
            "macro_f1": report["macro avg"]["f1-score"],
            "weighted_precision": report["weighted avg"]["precision"],
            "weighted_recall": report["weighted avg"]["recall"],
            "weighted_f1": report["weighted avg"]["f1-score"],
        }

        cm = confusion_matrix(y_true, y_pred, labels=label_ids)

        return pd.DataFrame(rows), pd.DataFrame([summary]), cm

    def save_epoch_metrics(self, epoch, train_loss, test_loss, y_true, y_pred):
        class_df, summary_df, cm = self.build_metric_tables(
            epoch,
            train_loss,
            test_loss,
            y_true,
            y_pred,
        )

        class_path = os.path.join(self.metrics_dir, "per_class_metrics.csv")
        summary_path = os.path.join(self.metrics_dir, "summary_metrics.csv")

        class_df.to_csv(
            class_path,
            mode="a",
            index=False,
            header=not os.path.exists(class_path),
        )

        summary_df.to_csv(
            summary_path,
            mode="a",
            index=False,
            header=not os.path.exists(summary_path),
        )

        cm_df = pd.DataFrame(
            cm,
            index=["True_Wake", "True_N1", "True_N2", "True_N3", "True_REM"],
            columns=["Pred_Wake", "Pred_N1", "Pred_N2", "Pred_N3", "Pred_REM"],
        )

        cm_df.to_csv(
            os.path.join(
                self.metrics_dir,
                f"confusion_matrix_epoch_{epoch + 1:03d}.csv",
            )
        )

    def save_best_metrics(self, epoch, train_loss, test_loss, y_true, y_pred):
        class_df, summary_df, cm = self.build_metric_tables(
            epoch,
            train_loss,
            test_loss,
            y_true,
            y_pred,
        )

        class_df.to_csv(
            os.path.join(self.metrics_dir, "best_per_class_metrics.csv"),
            index=False,
        )

        summary_df.to_csv(
            os.path.join(self.metrics_dir, "best_summary_metrics.csv"),
            index=False,
        )

        cm_df = pd.DataFrame(
            cm,
            index=["True_Wake", "True_N1", "True_N2", "True_N3", "True_REM"],
            columns=["Pred_Wake", "Pred_N1", "Pred_N2", "Pred_N3", "Pred_REM"],
        )

        cm_df.to_csv(os.path.join(self.metrics_dir, "best_confusion_matrix.csv"))

        np.save(os.path.join(self.metrics_dir, "best_y_true.npy"), np.array(y_true))
        np.save(os.path.join(self.metrics_dir, "best_y_pred.npy"), np.array(y_pred))

    def save_final_test_metrics(self, epoch, test_loss, y_true, y_pred):
        class_df, summary_df, cm = self.build_metric_tables(
            epoch=epoch,
            train_loss=np.nan,
            test_loss=test_loss,
            y_true=y_true,
            y_pred=y_pred,
        )

        class_df.to_csv(
            os.path.join(self.metrics_dir, "final_test_per_class_metrics.csv"),
            index=False,
        )

        summary_df.to_csv(
            os.path.join(self.metrics_dir, "final_test_summary_metrics.csv"),
            index=False,
        )

        cm_df = pd.DataFrame(
            cm,
            index=["True_Wake", "True_N1", "True_N2", "True_N3", "True_REM"],
            columns=["Pred_Wake", "Pred_N1", "Pred_N2", "Pred_N3", "Pred_REM"],
        )

        cm_df.to_csv(os.path.join(self.metrics_dir, "final_test_confusion_matrix.csv"))

        np.save(os.path.join(self.metrics_dir, "final_test_y_true.npy"), np.array(y_true))
        np.save(os.path.join(self.metrics_dir, "final_test_y_pred.npy"), np.array(y_pred))

    def save_ckpt(
        self,
        model_state,
        val_real,
        val_pred,
        test_real,
        test_pred,
        best_epoch,
        best_train_loss,
        best_val_loss,
        final_test_loss,
    ):
        save_path = os.path.join(self.output_dir, "best_model.pth")

        torch.save(
            {
                "backbone_name": "NeuroNet_FineTuning",
                "model_state": model_state,
                "hyperparameter": self.args.__dict__,
                "best_epoch": best_epoch + 1,
                "loss": {
                    "best_train_loss": best_train_loss,
                    "best_val_loss": best_val_loss,
                    "final_test_loss": final_test_loss,
                },
                "result": {
                    "val_real": val_real,
                    "val_pred": val_pred,
                    "test_real": test_real,
                    "test_pred": test_pred,
                },
                "paths": {
                    "train_paths": self.train_paths,
                    "val_paths": self.val_paths,
                    "test_paths": self.test_paths,
                },
            },
            save_path,
        )

        print("Saved fine-tuned model:", save_path)


if __name__ == "__main__":
    args = get_args()
    args.n_fold = 0
    trainer = Trainer(args)
    trainer.train()
