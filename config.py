# ============================================================
#  config.py  –  Edit this file to customize your training
# ============================================================

import torch

# ----------------------------------------------------------
# STEP 1: Set your dataset root path
#
#   Your folder must look like this:
#
#   Dataset/
#     Fresh/
#       FreshApple/    FreshBanana/    FreshTomato/  ...
#     Rotten/
#       RottenApple/   RottenBanana/   RottenTomato/ ...
#
#   Point DATASET_ROOT to the "Dataset" folder itself.
# ----------------------------------------------------------
DATASET_ROOT = "Dataset"   # ← CHANGE THIS

# ----------------------------------------------------------
# STEP 2: Choose which fruits to include
#   Use the bare fruit name (no Fresh/Rotten prefix).
#   Set to None to use ALL available fruits.
# ----------------------------------------------------------
SELECTED_FRUITS = [
    # "Apple",
    # "Banana",
    # "Tomato",
    # "Orange",
    # "Carrot",
    "Mango",
    # "Strawberry",
    # "Potato",
    # "Cucumber",
    # "Bellpepper",
    # "Capciscum",
    # "Bittergroud",
    # "Okara",
]
# Set SELECTED_FRUITS = None  to train on every fruit found

# ----------------------------------------------------------
# STEP 3: Train / test split
#   The split is done in memory — no files are moved.
#   0.2 = 80% train, 20% test  (recommended)
# ----------------------------------------------------------
TEST_SPLIT = 0.2

# ----------------------------------------------------------
# STEP 4: Training hyperparameters
# ----------------------------------------------------------
IMAGE_SIZE    = 224
BATCH_SIZE    = 32           # Reduce to 16 if you hit memory issues
NUM_EPOCHS    = 15
LEARNING_RATE = 1e-4
WEIGHT_DECAY  = 1e-4
PATIENCE      = 5            # Early stopping: stop if no improvement for N epochs

# Fine-tuning: unfreeze last N layers of MobileNetV2 backbone
#   0  = train classifier head only  (fastest)
#   5  = unfreeze last 5 blocks      (recommended sweet spot)
#  -1  = unfreeze entire network     (slowest, highest accuracy)
UNFREEZE_LAYERS = 5

# ----------------------------------------------------------
# STEP 5: Output paths
# ----------------------------------------------------------
CHECKPOINT_DIR  = "checkpoints"
RESULTS_DIR     = "results"
BEST_MODEL_PATH = "checkpoints/best_model.pth"

# ----------------------------------------------------------
# Auto-detect device  (MPS for Apple Silicon, CUDA, or CPU)
# ----------------------------------------------------------
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("✅ Device: Apple Silicon MPS (M4 GPU)")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("✅ Device: CUDA GPU")
else:
    DEVICE = torch.device("cpu")
    print("⚠️  Device: CPU — training will be slow")
