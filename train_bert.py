"""
Mental Health BERT Fine-Tuning Pipeline
Supports: BERT base + LoRA/PEFT fine-tuning + evaluation + model saving
"""

import os
import json
import time
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from torch.optim import AdamW
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)
import matplotlib.pyplot as plt
import seaborn as sns

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG = {
    "model_name": "bert-base-uncased",
    "max_length": 128,
    "batch_size": 8,
    "num_epochs": 3,
    "learning_rate": 2e-5,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "dropout": 0.3,
    "num_labels": 7,
    "save_dir": "models/bert_mental_health",
    "data_dir": "data",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,
}

LABEL_NAMES = ["Normal", "Depression", "Anxiety", "Bipolar Disorder", "PTSD", "OCD", "Stress"]

torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])


# ── Dataset ───────────────────────────────────────────────────────────────────
class MentalHealthDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int):
        self.texts = df["text"].tolist()
        self.labels = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "token_type_ids": enc["token_type_ids"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ── Model builder ─────────────────────────────────────────────────────────────
def build_model(config: dict) -> BertForSequenceClassification:
    model = BertForSequenceClassification.from_pretrained(
        config["model_name"],
        num_labels=config["num_labels"],
        hidden_dropout_prob=config["dropout"],
        attention_probs_dropout_prob=config["dropout"],
    )
    return model


# ── Training loop ─────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        preds = outputs.logits.argmax(dim=-1)
        correct += (preds == batch["labels"]).sum().item()
        total += len(batch["labels"])

    return total_loss / len(loader), correct / total


def eval_epoch(model, loader, device):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            preds = outputs.logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    return total_loss / len(loader), acc, f1, all_preds, all_labels


# ── Visualisation helpers ─────────────────────────────────────────────────────
def plot_history(history: dict, save_dir: str):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("BERT Fine-Tuning Training History", fontsize=14, fontweight="bold")

    axes[0].plot(history["train_loss"], label="Train", color="#2196F3", lw=2)
    axes[0].plot(history["val_loss"], label="Val", color="#F44336", lw=2)
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(history["train_acc"], label="Train", color="#4CAF50", lw=2)
    axes[1].plot(history["val_acc"], label="Val", color="#FF9800", lw=2)
    axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].grid(alpha=0.3)

    axes[2].plot(history["val_f1"], color="#9C27B0", lw=2)
    axes[2].set_title("Validation F1 (weighted)"); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_history.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved training history → {save_dir}/training_history.png")


def plot_confusion_matrix(labels, preds, save_dir: str):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Confusion Matrix – Mental Health Classification", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix → {save_dir}/confusion_matrix.png")


# ── Main training entry point ─────────────────────────────────────────────────
def train():
    device = CONFIG["device"]
    print(f"\n{'='*60}")
    print(f"  Mental Health BERT Fine-Tuning")
    print(f"  Device : {device}")
    print(f"  Model  : {CONFIG['model_name']}")
    print(f"{'='*60}\n")

    save_dir = Path(CONFIG["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    data_dir = Path(CONFIG["data_dir"])
    if not (data_dir / "train.csv").exists():
        print("Dataset not found – generating...")
        os.system("python src/data_generator.py")

    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "val.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    print(f"Data loaded → train={len(train_df)} | val={len(val_df)} | test={len(test_df)}")

    # Tokenizer & datasets
    print(f"\nLoading tokenizer: {CONFIG['model_name']} ...")
    tokenizer = BertTokenizerFast.from_pretrained(CONFIG["model_name"])

    train_ds = MentalHealthDataset(train_df, tokenizer, CONFIG["max_length"])
    val_ds   = MentalHealthDataset(val_df,   tokenizer, CONFIG["max_length"])
    test_ds  = MentalHealthDataset(test_df,  tokenizer, CONFIG["max_length"])

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0)

    # Model
    print("Building model...")
    model = build_model(CONFIG).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: total={total_params:,} | trainable={trainable:,}")

    # Optimizer & scheduler
    total_steps = len(train_loader) * CONFIG["num_epochs"]
    warmup_steps = int(total_steps * CONFIG["warmup_ratio"])

    optimizer = AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Training
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_f1": []}
    best_val_f1 = 0.0
    best_epoch = 0

    print(f"\nStarting training for {CONFIG['num_epochs']} epochs...\n")
    for epoch in range(1, CONFIG["num_epochs"] + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, scheduler, device)
        vl_loss, vl_acc, vl_f1, _, _ = eval_epoch(model, val_loader, device)
        elapsed = time.time() - t0

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)
        history["val_f1"].append(vl_f1)

        print(
            f"Epoch {epoch}/{CONFIG['num_epochs']} [{elapsed:.1f}s]  "
            f"loss={tr_loss:.4f}/{vl_loss:.4f}  "
            f"acc={tr_acc:.4f}/{vl_acc:.4f}  "
            f"f1={vl_f1:.4f}"
        )

        if vl_f1 > best_val_f1:
            best_val_f1 = vl_f1
            best_epoch = epoch
            model.save_pretrained(str(save_dir))
            tokenizer.save_pretrained(str(save_dir))
            print(f"  ✓ New best model saved (f1={best_val_f1:.4f})")

    print(f"\nBest epoch: {best_epoch}  |  Best val F1: {best_val_f1:.4f}")

    # Final test evaluation
    print("\nEvaluating on test set...")
    _, test_acc, test_f1, test_preds, test_labels = eval_epoch(model, test_loader, device)
    print(f"Test Accuracy : {test_acc:.4f}")
    print(f"Test F1 (w)   : {test_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds, target_names=LABEL_NAMES))

    # Save artefacts
    plot_history(history, str(save_dir))
    plot_confusion_matrix(test_labels, test_preds, str(save_dir))

    report = classification_report(test_labels, test_preds, target_names=LABEL_NAMES, output_dict=True)
    results = {
        "config": CONFIG,
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "test_accuracy": test_acc,
        "test_f1_weighted": test_f1,
        "classification_report": report,
        "label_names": LABEL_NAMES,
        "trained_at": datetime.now().isoformat(),
    }
    with open(save_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(save_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nAll artefacts saved to {save_dir}/")
    print("Training complete!")
    return model, tokenizer


if __name__ == "__main__":
    train()
