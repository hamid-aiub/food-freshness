# 🍎 Fresh/Rotten Fruit Classifier

**MobileNetV2 + Grad-CAM XAI | Optimized for Apple M4**

---

## Quick Start

### 1. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure `config.py`

```python
DATASET_ROOT = "/Users/you/Downloads/FruitDataset"  # ← your path

SELECTED_FRUITS = [      # ← pick any subset
    "Apple",
    "Banana",
    "Tomato",
]
```

### 3. Train

```bash
python train.py
```

### 4. Explain with Grad-CAM

```bash
# Single image

# Fresh
python explain.py --image "Dataset/Fresh/FreshMango/freshMango (1).jpg"
python explain.py --image "Dataset/Fresh/FreshMango/freshMango (100).jpg"
python explain.py --image "Dataset/Fresh/FreshMango/freshMango (300).jpg"

# Rotten
python explain.py --image "Dataset/Rotten/RottenMango/RottenMango (1).jpg"
python explain.py --image "Dataset/Rotten/RottenMango/RottenMango (100).jpg"
python explain.py --image "Dataset/Rotten/RottenMango/RottenMango (300).jpg"

# Grid from test set (12 random images)
python explain.py --test_set --n 12

# Folder of images
python explain.py --folder /path/to/my_images --n 8 --method gradcam++
```

### 5. Predict new images

```bash
python predict.py --image /path/to/fruit.jpg
python predict.py --folder /path/to/images/
```

---

## Dataset Structure Expected

```
your_dataset/
├── train/
│   ├── FreshApple/    (jpg/png images)
│   ├── RottenApple/
│   ├── FreshBanana/
│   ├── RottenBanana/
│   └── ...
└── test/
    ├── FreshApple/
    ├── RottenApple/
    └── ...
```

---

## All Available Fruits (from Dataset 4)

| Fruit       | Fresh | Rotten |
| ----------- | ----- | ------ |
| Apple       | 3431  | 4437   |
| Banana      | 3473  | 4038   |
| Bellpepper  | 611   | 591    |
| Bittergroud | 327   | 357    |
| Capciscum   | 990   | 901    |
| Carrot      | 9899  | 2419   |
| Cucumber    | 1104  | 1014   |
| Mango       | 605   | 593    |
| Okara       | 635   | 338    |
| Orange      | 10835 | 3292   |
| Potato      | 1151  | 1387   |
| Strawberry  | 603   | 596    |
| Tomato      | 13681 | 4014   |

---

## Key Config Options

| Option            | Default | Description                                  |
| ----------------- | ------- | -------------------------------------------- |
| `SELECTED_FRUITS` | `[...]` | List of fruit names; `None` = use all        |
| `BATCH_SIZE`      | 32      | Reduce to 16 if memory issues                |
| `NUM_EPOCHS`      | 15      | More = better (with early stopping)          |
| `UNFREEZE_LAYERS` | 5       | 0=head only, 5=recommended, -1=full finetune |
| `PATIENCE`        | 5       | Early stopping epochs                        |

---

## XAI Methods

| Method     | Flag        | Description                     |
| ---------- | ----------- | ------------------------------- |
| Grad-CAM++ | `gradcam++` | Best for localization (default) |
| Grad-CAM   | `gradcam`   | Classic, slightly coarser       |
| EigenCAM   | `eigencam`  | No backprop needed, fastest     |

---

## Output Files

```
checkpoints/
  best_model.pth       ← saved when val_acc improves
  class_names.json     ← class list for inference

results/
  training_curves.png  ← loss & accuracy plots
  confusion_matrix.png ← full class confusion matrix
  gradcam_grid.png     ← Grad-CAM visualization grid
  explanation_single.png
```

---

## Tips for M4 MacBook

- MPS backend is auto-detected — no extra setup needed
- Start with 3-4 fruits to validate the pipeline (< 5 min/epoch)
- `BATCH_SIZE=32` works well; bump to 64 if you have headroom
- `UNFREEZE_LAYERS=5` is the sweet spot for accuracy vs. speed
- Training 4 fruits × 15 epochs ≈ 10-15 minutes on M4
