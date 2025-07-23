# mlp_trainer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from logger import TrainingLogger, plot_confusion_matrix

def train_model(
        model: nn.Module,
        train_loader: DataLoader,
        label_mode: str,
        val_loader: DataLoader = None,
        epochs: int = 10,
        lr: float = 1e-3,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        experiment_name: str = "exp1",
        log_dir: str = "logs"
):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    logger = TrainingLogger(experiment_name = experiment_name,
                            log_dir = log_dir)
    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        all_preds, all_targets = [], []

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            x, y = x.to(device), y.to(device).long()
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

            all_preds.append(preds.detach().cpu())
            all_targets.append(y.detach().cpu())

        train_loss = total_loss / total
        train_acc = correct / total
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)

        val_loss, val_acc = None, None
        if val_loader:
            val_loss, val_acc = evaluate(model, val_loader, device)

        logger.log_epoch(
            epoch,
            train_loss,
            train_acc,
            y_true=all_targets,
            y_pred=all_preds,
            val_loss=val_loss,
            val_acc=val_acc,
        )


        print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
    logger.plot_metrics()
    logger.save_json()
    plot_confusion_matrix(model = model,
                          dataloader = val_loader,
                          log_dir = log_dir)
    return model



def evaluate(model, dataloader, device="cpu"):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            total_loss += loss.item() * x.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    print(f"Validation Accuracy: {accuracy:.4f}")
    return avg_loss, accuracy
