# model.py  –  MobileNetV2 classifier with selective fine-tuning

import torch
import torch.nn as nn
from torchvision import models


def build_model(num_classes: int, unfreeze_layers: int = 5) -> nn.Module:
    """
    Loads a pretrained MobileNetV2 and replaces the classifier head.

    unfreeze_layers:
        0  → freeze entire backbone, train head only
        N  → unfreeze last N features blocks
        -1 → unfreeze entire network
    """
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

    # --- Freeze backbone ---
    for param in model.parameters():
        param.requires_grad = False

    # --- Selectively unfreeze ---
    if unfreeze_layers == -1:
        for param in model.parameters():
            param.requires_grad = True
    elif unfreeze_layers > 0:
        # MobileNetV2.features is a Sequential of 19 blocks (0-18)
        feature_blocks = list(model.features.children())
        for block in feature_blocks[-unfreeze_layers:]:
            for param in block.parameters():
                param.requires_grad = True

    # --- Replace classifier head ---
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(512, num_classes),
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Model: MobileNetV2 | Classes: {num_classes}")
    print(f"Trainable params: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.1f}%)")

    return model


def save_checkpoint(model, class_names, epoch, val_acc, path):
    torch.save({
        "epoch": epoch,
        "val_acc": val_acc,
        "class_names": class_names,
        "model_state_dict": model.state_dict(),
    }, path)


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)
    class_names = ckpt["class_names"]
    model = build_model(num_classes=len(class_names), unfreeze_layers=0)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"✅ Loaded checkpoint (epoch {ckpt['epoch']}, "
          f"val_acc={ckpt['val_acc']:.2f}%)")
    return model, class_names
