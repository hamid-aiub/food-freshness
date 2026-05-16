# explain.py  –  Grad-CAM XAI visualization
#
# Usage:
#   python explain.py --image path/to/fruit.jpg          # single image
#   python explain.py --test_set --n 12                  # grid from test split
#   python explain.py --folder path/to/folder --n 8      # folder of images

import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from PIL import Image

import torch
from torchvision import transforms
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import config
from model import load_checkpoint
from dataset import get_test_samples   # ← uses the same 80/20 split


# ----------------------------------------------------------
# Preprocessing (must match training)
# ----------------------------------------------------------
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD  = [0.229, 0.224, 0.225]

preprocess = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
])


def denormalize(tensor):
    t = tensor.clone().squeeze(0)
    for ch, m, s in zip(t, NORM_MEAN, NORM_STD):
        ch.mul_(s).add_(m)
    return t.permute(1, 2, 0).clamp(0, 1).numpy()


# ----------------------------------------------------------
# Core: predict + Grad-CAM for one image
# ----------------------------------------------------------
def explain_image(model, class_names, image_path, device,
                  cam_method="gradcam", true_label=None):
    pil_img      = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(pil_img).unsqueeze(0).to(device)
    rgb_img      = denormalize(input_tensor.cpu())

    # ── Step 1: fast inference on MPS/GPU ──────────────────
    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)
        probs  = torch.softmax(logits, dim=1).squeeze()

    pred_idx   = probs.argmax().item()
    confidence = probs[pred_idx].item() * 100
    pred_class = class_names[pred_idx]
    top5_idx   = probs.topk(min(5, len(class_names))).indices.tolist()
    top5       = [(class_names[i], round(probs[i].item() * 100, 1)) for i in top5_idx]

    # ── Step 2: Grad-CAM on CPU ────────────────────────────
    # MPS does not support the grad ops needed by Grad-CAM,
    # so we temporarily move model + tensor to CPU for this step only.
    cam_device   = torch.device("cpu")
    model_cpu    = model.to(cam_device)
    tensor_cpu   = input_tensor.to(cam_device)

    target_layer = [model_cpu.features[-1]]

    CAM_MAP = {
        "gradcam":   GradCAM,
        "gradcam++": GradCAMPlusPlus,
        "eigencam":  EigenCAM,
    }
    CAMClass = CAM_MAP.get(cam_method.lower(), GradCAM)

    with CAMClass(model=model_cpu, target_layers=target_layer) as cam:
        grayscale_cam = cam(
            input_tensor=tensor_cpu,
            targets=[ClassifierOutputTarget(pred_idx)]
        )[0]

    # Move model back to original device for next call
    model.to(device)

    cam_overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    return {
        "rgb_img":     rgb_img,
        "cam_overlay": cam_overlay,
        "pred_class":  pred_class,
        "confidence":  confidence,
        "top5":        top5,
        "true_label":  true_label,
        "path":        str(image_path),
    }


# ----------------------------------------------------------
# Plotting
# ----------------------------------------------------------
def plot_single(result, save_path=None):
    fig = plt.figure(figsize=(14, 5))
    gs  = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1.2])

    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(result["rgb_img"])
    ax1.set_title("Original Image", fontsize=12)
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[1])
    ax2.imshow(result["cam_overlay"])
    correct    = (result["true_label"] is None or
                  result["true_label"] == result["pred_class"])
    border_col = "#27ae60" if correct else "#e74c3c"
    for spine in ax2.spines.values():
        spine.set_edgecolor(border_col); spine.set_linewidth(3)
    ax2.set_title(
        f"Grad-CAM++ Heatmap\n"
        f"Predicted: {result['pred_class']} ({result['confidence']:.1f}%)",
        fontsize=12, color=border_col,
    )
    ax2.axis("off")

    ax3 = fig.add_subplot(gs[2])
    labels = [t[0] for t in result["top5"]][::-1]
    values = [t[1] for t in result["top5"]][::-1]
    colors = ["#2ecc71" if "Fresh" in l else "#e74c3c" for l in labels]
    bars   = ax3.barh(labels, values, color=colors, edgecolor="none", height=0.5)
    ax3.set_xlim(0, 100)
    ax3.set_xlabel("Confidence (%)")
    ax3.set_title("Top Predictions", fontsize=12)
    for bar, val in zip(bars, values):
        ax3.text(val + 1, bar.get_y() + bar.get_height() / 2,
                 f"{val:.1f}%", va="center", fontsize=9)
    ax3.tick_params(axis="y", labelsize=9)
    ax3.spines[["top", "right"]].set_visible(False)

    if result["true_label"]:
        fig.suptitle(f"True label: {result['true_label']}",
                     fontsize=11, color="gray", y=1.02)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"💾 Saved → {save_path}")
    plt.show()
    plt.close()


def plot_grid(results, save_path=None, cols=4):
    n    = len(results)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.8))
    axes = np.array(axes).flatten()

    for i, res in enumerate(results):
        ax = axes[i]
        ax.imshow(res["cam_overlay"])
        correct = (res["true_label"] is None or
                   res["true_label"] == res["pred_class"])
        color   = "#27ae60" if correct else "#e74c3c"
        title   = f"{res['pred_class']}\n{res['confidence']:.1f}%"
        if res["true_label"] and not correct:
            title += f"\n(true: {res['true_label']})"
        ax.set_title(title, fontsize=7.5, color=color)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_edgecolor(color); spine.set_linewidth(2)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Grad-CAM++ Explanations  |  🟢 correct  🔴 wrong",
                 fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"💾 Grid saved → {save_path}")
    plt.show()
    plt.close()


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Grad-CAM XAI for Fresh/Rotten Classifier"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",    type=str,
                       help="Path to a single image file")
    group.add_argument("--folder",   type=str,
                       help="Folder of images (no ground truth)")
    group.add_argument("--test_set", action="store_true",
                       help="Random sample from the held-out 20%% test split")

    parser.add_argument("--n",       type=int, default=12,
                        help="Number of images for grid mode (default 12)")
    parser.add_argument("--method",  type=str, default="gradcam++",
                        choices=["gradcam", "gradcam++", "eigencam"])
    parser.add_argument("--save_dir", type=str,
                        default=config.RESULTS_DIR)
    args = parser.parse_args()

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    model, class_names = load_checkpoint(config.BEST_MODEL_PATH, config.DEVICE)

    # ── Single image ──────────────────────────────────────
    if args.image:
        print(f"\n🔍 Explaining: {args.image}")
        result = explain_image(model, class_names, args.image,
                               config.DEVICE, cam_method=args.method)
        plot_single(result,
                    save_path=f"{args.save_dir}/explanation_single.png")
        return

    # ── Build items list ──────────────────────────────────
    if args.test_set:
        items, class_names = get_test_samples(config, n=args.n)
        print(f"\n🔍 Explaining {len(items)} images from the test split...")
    else:
        IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        all_imgs = [p for p in Path(args.folder).iterdir()
                    if p.suffix.lower() in IMG_EXTS]
        chosen   = random.sample(all_imgs, min(args.n, len(all_imgs)))
        items    = [(str(p), None) for p in chosen]
        print(f"\n🔍 Explaining {len(items)} images from {args.folder}...")

    # ── Run explanations ──────────────────────────────────
    results = []
    for path, true_label in items:
        r = explain_image(model, class_names, path, config.DEVICE,
                          cam_method=args.method, true_label=true_label)
        results.append(r)
        ok = "✅" if (true_label is None or
                      true_label == r["pred_class"]) else "❌"
        print(f"  {ok}  {Path(path).name:<40}  →  "
              f"{r['pred_class']} ({r['confidence']:.1f}%)")

    plot_grid(results, save_path=f"{args.save_dir}/gradcam_grid.png")

    labeled = [r for r in results if r["true_label"]]
    if labeled:
        correct = sum(1 for r in labeled if r["true_label"] == r["pred_class"])
        print(f"\n🎯 Accuracy on shown samples: "
              f"{correct}/{len(labeled)} ({100*correct/len(labeled):.1f}%)")


if __name__ == "__main__":
    main()