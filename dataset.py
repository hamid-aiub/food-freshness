# dataset.py  –  Reads Dataset/Fresh/<FruitName>/ and Dataset/Rotten/<FruitName>/
#               Auto-splits into train/val with stratification (no manual folders needed)

import random
from pathlib import Path
from PIL import Image
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import numpy as np


# ----------------------------------------------------------
# Transforms
# ----------------------------------------------------------
def get_transforms(image_size: int, augment: bool):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if augment:
        return transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                   saturation=0.3, hue=0.1),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


# ----------------------------------------------------------
# Discover classes from Dataset/Fresh/ and Dataset/Rotten/
# ----------------------------------------------------------
def discover_classes(root: str, selected_fruits=None):
    """
    Scans Dataset/Fresh/<FruitFolder>/ and Dataset/Rotten/<FruitFolder>/
    Fuzzy-matches across both sides so spelling variants like
    'Capciscum' (Fresh) vs 'Capsicum' (Rotten) still pair up correctly.

    Returns:
        class_names  : sorted list, e.g. ['FreshApple', 'RottenApple', ...]
        class_to_dir : {class_name -> Path}
    """
    root        = Path(root)
    fresh_root  = root / "Fresh"
    rotten_root = root / "Rotten"

    for d in (fresh_root, rotten_root):
        if not d.exists():
            raise FileNotFoundError(
                f"Expected sub-folder not found: {d}\n"
                f"Your root folder must contain  Fresh/  and  Rotten/  directories."
            )

    def norm(name):
        """Normalise for fuzzy matching: lowercase, strip spaces/underscores."""
        return name.lower().replace(" ", "").replace("_", "")

    def core(norm_name):
        """Strip leading 'fresh'/'rotten' to get the bare fruit key."""
        for prefix in ("fresh", "rotten"):
            if norm_name.startswith(prefix):
                return norm_name[len(prefix):]
        return norm_name

    fresh_by_core  = {core(norm(d.name)): d
                      for d in fresh_root.iterdir()  if d.is_dir()}
    rotten_by_core = {core(norm(d.name)): d
                      for d in rotten_root.iterdir() if d.is_dir()}

    paired_cores = set(fresh_by_core) & set(rotten_by_core)

    if selected_fruits:
        sel = {norm(f).lstrip("fresh").lstrip("rotten") for f in selected_fruits}
        # also handle bare names like "Apple" → core "apple"
        sel2 = {core(norm(f)) for f in selected_fruits}
        sel  = sel | sel2
        paired_cores = {c for c in paired_cores if c in sel}

    if not paired_cores:
        raise ValueError(
            "No matching Fresh/Rotten pairs found for the selected fruits.\n"
            "Use bare fruit names in SELECTED_FRUITS, e.g. 'Apple', 'Banana'."
        )

    class_to_dir = {}
    for fruit_core in sorted(paired_cores):
        cap = fruit_core.capitalize()
        class_to_dir[f"Fresh{cap}"]  = fresh_by_core[fruit_core]
        class_to_dir[f"Rotten{cap}"] = rotten_by_core[fruit_core]

    return sorted(class_to_dir.keys()), class_to_dir


# ----------------------------------------------------------
# Stratified train / test split (no file moving needed)
# ----------------------------------------------------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def stratified_split(class_to_dir, class_names, test_ratio=0.2, seed=42):
    """
    Splits each class folder into train/test in memory.
    Returns two lists of (path_str, class_idx) tuples.
    """
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    train_samples, test_samples = [], []
    rng = random.Random(seed)

    for cls in class_names:
        folder = class_to_dir[cls]
        imgs   = [str(p) for p in folder.iterdir()
                  if p.suffix.lower() in IMG_EXTS]
        rng.shuffle(imgs)
        split_at = max(1, int(len(imgs) * (1 - test_ratio)))
        idx = class_to_idx[cls]
        train_samples.extend((p, idx) for p in imgs[:split_at])
        test_samples.extend( (p, idx) for p in imgs[split_at:])

    return train_samples, test_samples


# ----------------------------------------------------------
# Dataset class
# ----------------------------------------------------------
class FruitDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ----------------------------------------------------------
# Public API  –  called by train.py and explain.py
# ----------------------------------------------------------
def build_dataloaders(config):
    print(f"\n📂 Scanning dataset at: {config.DATASET_ROOT}")

    class_names, class_to_dir = discover_classes(
        config.DATASET_ROOT,
        selected_fruits=config.SELECTED_FRUITS,
    )

    test_ratio = getattr(config, "TEST_SPLIT", 0.2)
    train_samples, test_samples = stratified_split(
        class_to_dir, class_names, test_ratio=test_ratio, seed=42
    )

    # Pretty summary table
    tr_counts = Counter(lbl for _, lbl in train_samples)
    te_counts = Counter(lbl for _, lbl in test_samples)
    print(f"\n  {'Class':<22} {'Total':>7} {'Train':>7} {'Test':>7}")
    print("  " + "-" * 43)
    for i, cls in enumerate(class_names):
        tr, te = tr_counts[i], te_counts[i]
        print(f"  {cls:<22} {tr+te:>7,} {tr:>7,} {te:>7,}")
    print("  " + "-" * 43)
    total = len(train_samples) + len(test_samples)
    print(f"  {'TOTAL':<22} {total:>7,} {len(train_samples):>7,} {len(test_samples):>7,}")
    print(f"\n  80/20 stratified split | seed=42\n")

    train_tf = get_transforms(config.IMAGE_SIZE, augment=True)
    val_tf   = get_transforms(config.IMAGE_SIZE, augment=False)

    train_ds = FruitDataset(train_samples, transform=train_tf)
    test_ds  = FruitDataset(test_samples,  transform=val_tf)

    # Class-balanced weighted sampler for training
    labels       = [s[1] for s in train_samples]
    class_counts = np.bincount(labels, minlength=len(class_names))
    weights      = 1.0 / class_counts[labels]
    sampler      = WeightedRandomSampler(
        weights=torch.DoubleTensor(weights),
        num_samples=len(weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds, batch_size=config.BATCH_SIZE,
        sampler=sampler, num_workers=4, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.BATCH_SIZE,
        shuffle=False, num_workers=4, pin_memory=True,
    )

    return train_loader, test_loader, class_names


# ----------------------------------------------------------
# Helper used by explain.py to sample test images
# ----------------------------------------------------------
def get_test_samples(config, n=12, seed=99):
    """Returns n random (path, class_name) tuples from the test split."""
    class_names, class_to_dir = discover_classes(
        config.DATASET_ROOT,
        selected_fruits=config.SELECTED_FRUITS,
    )
    _, test_samples = stratified_split(
        class_to_dir, class_names,
        test_ratio=getattr(config, "TEST_SPLIT", 0.2), seed=42
    )
    rng = random.Random(seed)
    chosen = rng.sample(test_samples, min(n, len(test_samples)))
    return [(path, class_names[lbl]) for path, lbl in chosen], class_names
