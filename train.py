# train.py  –  Training loop with early stopping, LR scheduling & plots

import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim

import config
from dataset import build_dataloaders
from model import build_model, save_checkpoint


# ----------------------------------------------------------
# Helpers
# ----------------------------------------------------------
def run_epoch(model, loader, criterion, optimizer, device, training: bool):
    model.train() if training else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for imgs, labels in tqdm(loader, leave=False,
                                 desc="Train" if training else "Val  "):
            imgs, labels = imgs.to(device), labels.to(device)
            if training:
                optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)

    return total_loss / total, 100.0 * correct / total


def plot_history(history, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"],   label="Val")
    axes[0].set_title("Loss"); axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="Train")
    axes[1].plot(epochs, history["val_acc"],   label="Val")
    axes[1].set_title("Accuracy (%)"); axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_curves.png", dpi=150)
    plt.close()
    print(f"📊 Training curves saved → {save_dir}/training_curves.png")


def plot_confusion_matrix(cm, class_names, save_dir):
    fig, ax = plt.subplots(figsize=(max(8, len(class_names)), max(6, len(class_names) - 2)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/confusion_matrix.png", dpi=150)
    plt.close()
    print(f"📊 Confusion matrix saved → {save_dir}/confusion_matrix.png")


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
def main():
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    print("\n" + "=" * 55)
    print("  MobileNetV2 Fresh/Rotten Fruit Classifier")
    print("=" * 55)

    # Data
    train_loader, test_loader, class_names = build_dataloaders(config)
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}\n")

    # Save class names for inference
    with open(f"{config.CHECKPOINT_DIR}/class_names.json", "w") as f:
        json.dump(class_names, f)

    # Model
    model = build_model(num_classes, config.UNFREEZE_LAYERS)
    model = model.to(config.DEVICE)

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.NUM_EPOCHS
    )

    # Training loop
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    patience_counter = 0

    print(f"\n🚀 Starting training on {config.DEVICE} for {config.NUM_EPOCHS} epochs...\n")
    t0 = time.time()

    for epoch in range(1, config.NUM_EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion,
                                    optimizer, config.DEVICE, training=True)
        va_loss, va_acc = run_epoch(model, test_loader, criterion,
                                    None, config.DEVICE, training=False)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        marker = ""
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            patience_counter = 0
            save_checkpoint(model, class_names, epoch, va_acc,
                            config.BEST_MODEL_PATH)
            marker = "  ← best ✨"
        else:
            patience_counter += 1

        print(f"Epoch {epoch:3d}/{config.NUM_EPOCHS} | "
              f"Loss {tr_loss:.4f}/{va_loss:.4f} | "
              f"Acc {tr_acc:.1f}%/{va_acc:.1f}%"
              f"{marker}")

        if patience_counter >= config.PATIENCE:
            print(f"\n⏹️  Early stopping after {epoch} epochs "
                  f"(no improvement for {config.PATIENCE} epochs)")
            break

    elapsed = time.time() - t0
    print(f"\n⏱️  Training time: {elapsed/60:.1f} min")
    print(f"🏆 Best validation accuracy: {best_val_acc:.2f}%")

    # Plots
    plot_history(history, config.RESULTS_DIR)

    # Final evaluation on test set with best model checkpoint
    print("\n📋 Final evaluation on test set...")
    ckpt = torch.load(config.BEST_MODEL_PATH, map_location=config.DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Evaluating"):
            imgs = imgs.to(config.DEVICE)
            preds = model(imgs).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    print("\n" + classification_report(
        all_labels, all_preds, target_names=class_names
    ))

    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, class_names, config.RESULTS_DIR)
    print("\n✅ Training complete! Run explain.py to visualize predictions with Grad-CAM.")


if __name__ == "__main__":
    main()
