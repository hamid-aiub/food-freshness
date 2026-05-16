# predict.py  –  Quick inference on new images (no training needed)
#
# Usage:
#   python predict.py --image /path/to/fruit.jpg
#   python predict.py --folder /path/to/images/

import argparse
from pathlib import Path
from PIL import Image

import torch
from torchvision import transforms

import config
from model import load_checkpoint

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD  = [0.229, 0.224, 0.225]

preprocess = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
])


def predict(model, class_names, image_path, device):
    img    = Image.open(image_path).convert("RGB")
    tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).squeeze()
    idx  = probs.argmax().item()
    return class_names[idx], round(probs[idx].item() * 100, 2)


def main():
    parser = argparse.ArgumentParser()
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",  type=str)
    group.add_argument("--folder", type=str)
    args = parser.parse_args()

    model, class_names = load_checkpoint(config.BEST_MODEL_PATH, config.DEVICE)

    paths = (
        [Path(args.image)]
        if args.image
        else list(Path(args.folder).glob("*.[jJpP][pPnN][gG]"))
    )

    print(f"\n{'Image':<45}  {'Prediction':<25}  Confidence")
    print("-" * 80)
    for p in sorted(paths):
        pred, conf = predict(model, class_names, p, config.DEVICE)
        icon = "🟢" if "Fresh" in pred else "🔴"
        print(f"{p.name:<45}  {icon} {pred:<23}  {conf:.1f}%")


if __name__ == "__main__":
    main()
