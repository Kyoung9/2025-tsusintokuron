# logger.py

import os, json, csv
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
import matplotlib.pyplot as plt
import seaborn as sns

class TrainingLogger:
    """学習ログ + 主要評価指標（Precision / Recall / F1）を一括管理"""

    def __init__(self, experiment_name: str, log_dir: str = "logs", enable_tb: bool = True):
        self.exp = experiment_name
        # self.dir = os.path.join(log_dir, experiment_name)
        self.dir = log_dir
        os.makedirs(self.dir, exist_ok=True)

        self.history = []

        # CSV ヘッダーを書いておく
        self.csv_file = os.path.join(self.dir, f"{self.exp}_metrics.csv")
        with open(self.csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "train_loss", "train_acc",
                "val_loss", "val_acc",
                "prec_macro", "rec_macro", "f1_macro",
                "prec_weighted", "rec_weighted", "f1_weighted"
            ])

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        val_loss: float | None = None,
        val_acc: float | None = None,
    ) -> None:
        """エポック単位でログをまとめて記録"""

        # numpy へ変換
        y_true_np = y_true.cpu().numpy()
        y_pred_np = y_pred.cpu().numpy()

        # macro & weighted 指標
        prec_macro   = precision_score(y_true_np, y_pred_np, average="macro", zero_division=0)
        rec_macro    = recall_score   (y_true_np, y_pred_np, average="macro", zero_division=0)
        f1_macro     = f1_score       (y_true_np, y_pred_np, average="macro", zero_division=0)

        prec_weight  = precision_score(y_true_np, y_pred_np, average="weighted", zero_division=0)
        rec_weight   = recall_score   (y_true_np, y_pred_np, average="weighted", zero_division=0)
        f1_weight    = f1_score       (y_true_np, y_pred_np, average="weighted", zero_division=0)

        # コンソールに簡易レポート
        print(
            f"[Epoch {epoch}] "
            f"Test L={val_loss:.4f} A={val_acc:.4f} | "
            f"Train L={train_loss:.4f} A={train_acc:.4f} | "
            f"MacroF1={f1_macro:.4f}  WeightedF1={f1_weight:.4f}"
        )

        # CSV
        with open(self.csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, train_loss, train_acc,
                val_loss if val_loss is not None else "",
                val_acc if val_acc is not None else "",
                prec_macro, rec_macro, f1_macro,
                prec_weight, rec_weight, f1_weight
            ])

        # 内部履歴（必要なら DataFrame 化）
        self.history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "prec_macro": prec_macro,
            "rec_macro": rec_macro,
            "f1_macro": f1_macro,
            "prec_weighted": prec_weight,
            "rec_weighted": rec_weight,
            "f1_weighted": f1_weight,
        })

    # DataFrame 取得
    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)

    def save_json(self):
        path = os.path.join(self.dir, f"{self.exp}_metrics.json")
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)


    def plot_metrics(self, logscale=False, save=False):
        df = pd.DataFrame(self.history)
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        axs[0].plot(df["epoch"], df["train_loss"], label="Train Loss")
        if "val_loss" in df.columns:
            axs[0].plot(df["epoch"], df["val_loss"], label="Val Loss")
        axs[0].set_title("Loss")
        if logscale:
            axs[0].set_yscale("log")
        axs[0].legend()

        axs[1].plot(df["epoch"], df["train_acc"], label="Train Acc")
        if "val_acc" in df.columns:
            axs[1].plot(df["epoch"], df["val_acc"], label="Val Acc")
        axs[1].set_title("Accuracy")
        axs[1].legend()

        plt.tight_layout()
        if save:
            fig.savefig(os.path.join(self.log_dir, f"{self.experiment_name}_metrics.png"))
        plt.show()


def plot_confusion_matrix(model, dataloader, log_dir="logs", class_names=None, device="cpu", normalize=True, save=False):
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y.cpu().numpy())

        # クラス数を自動取得
        num_classes = len(np.unique(all_targets + all_preds))

        # クラス名が指定されなかった場合のデフォルト
        if class_names is None:
            class_names = [str(i) for i in range(num_classes)]

        cm = confusion_matrix(all_targets, all_preds, normalize="true" if normalize else None)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(log_dir, f"{self.experiment_name}_confmat.png"))
        plt.show()
