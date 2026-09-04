"""Fine-tune an ImageNet-pretrained EfficientNet-B0 on SIPaKMeD cropped cells.

The pipeline never writes to or changes the raw dataset.  Images are decoded and
resized only in memory for the pretrained model's input, then normalized with the
ImageNet statistics used to pretrain EfficientNet-B0.

Example:
    python train.py --data-dir data/raw --output-dir outputs/efficientnet_b0
"""
from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm


CLASS_NAMES = [
    "Dyskeratotic",
    "Koilocytotic",
    "Metaplastic",
    "Parabasal",
    "Superficial-Intermediate",
]
IMAGE_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def label_from_path(path: Path) -> str | None:
    for part in path.parts:
        normalized = part.lower().replace("im_", "").replace("_", "-")
        for label in CLASS_NAMES:
            if normalized == label.lower():
                return label
    return None


def discover_cropped_images(data_dir: Path) -> pd.DataFrame:
    """Use only SIPaKMeD's supplied CROPPED cell images, never the source slides."""
    rows: list[dict[str, str]] = []
    for cropped_dir in sorted(path for path in data_dir.rglob("CROPPED") if path.is_dir()):
        for path in sorted(cropped_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            label = label_from_path(path)
            if label is None:
                continue
            try:
                with Image.open(path) as image:
                    image.verify()
            except (OSError, ValueError):
                print(f"Skipping unreadable image: {path}")
                continue
            rows.append({"path": str(path.resolve()), "label": label})

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError(f"No readable images under CROPPED folders in {data_dir}")
    counts = frame["label"].value_counts()
    missing = [label for label in CLASS_NAMES if label not in counts]
    if missing:
        raise RuntimeError(f"Missing classes: {', '.join(missing)}")
    return frame


def stratified_splits(frame: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_val, test = train_test_split(frame, test_size=0.15, stratify=frame["label"], random_state=seed)
    train, validation = train_test_split(
        train_val, test_size=0.15 / 0.85, stratify=train_val["label"], random_state=seed
    )
    return train.reset_index(drop=True), validation.reset_index(drop=True), test.reset_index(drop=True)


class CellDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform: transforms.Compose) -> None:
        self.frame = frame.reset_index(drop=True)
        self.transform = transform
        self.label_to_index = {label: index for index, label in enumerate(CLASS_NAMES)}

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        with Image.open(row.path) as image:
            image = image.convert("RGB")
        return self.transform(image), self.label_to_index[row.label]


class CellPreprocessor:
    """Optional image cleanup performed in memory, leaving raw files untouched."""

    def __init__(self, color_normalization: bool, remove_background: bool, background_threshold: int) -> None:
        self.color_normalization = color_normalization
        self.remove_background = remove_background
        self.background_threshold = background_threshold / 255.0

    def __call__(self, image: Image.Image) -> Image.Image:
        pixels = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        foreground = pixels.mean(axis=2) < self.background_threshold
        if self.color_normalization and foreground.any():
            channel_mean = pixels[foreground].mean(axis=0)
            pixels = np.clip(pixels * (channel_mean.mean() / np.maximum(channel_mean, 1e-6)), 0.0, 1.0)
        if self.remove_background:
            pixels[~foreground] = 0.0
        return Image.fromarray((pixels * 255).round().astype(np.uint8), mode="RGB")


def make_loaders(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, image_size: int, batch_size: int, workers: int, color_normalization: bool, remove_background: bool, background_threshold: int):
    # These transforms are in-memory model input preparation, not preprocessing of raw files.
    cleanup = CellPreprocessor(color_normalization, remove_background, background_threshold)
    evaluation_transform = transforms.Compose([
        cleanup,
        transforms.Resize((image_size, image_size), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    training_transform = transforms.Compose([
        cleanup,
        transforms.Resize((image_size, image_size), antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    pin_memory = torch.cuda.is_available()
    options = {"batch_size": batch_size, "num_workers": workers, "pin_memory": pin_memory}
    return (
        DataLoader(CellDataset(train, training_transform), shuffle=True, **options),
        DataLoader(CellDataset(val, evaluation_transform), shuffle=False, **options),
        DataLoader(CellDataset(test, evaluation_transform), shuffle=False, **options),
    )


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    predictions, targets = [], []
    for images, labels in loader:
        logits = model(images.to(device, non_blocking=True))
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
        targets.extend(labels.tolist())
    return {
        "accuracy": accuracy_score(targets, predictions),
        "macro_f1": f1_score(targets, predictions, average="macro"),
        "predictions": predictions,
        "targets": targets,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/efficientnet_b0"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, choices=(256, 512), default=256)
    parser.add_argument("--color-normalization", action="store_true", help="Apply in-memory gray-world colour normalization.")
    parser.add_argument("--remove-background", action="store_true", help="Mask near-white background pixels in memory.")
    parser.add_argument("--background-threshold", type=int, default=245, help="Pixels lighter than this (0-255) are background.")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0, help="Use 0 on Windows unless worker processes are needed.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 <= args.background_threshold <= 255:
        parser.error("--background-threshold must be between 0 and 255")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    preprocessing = {
        "resize": [args.image_size, args.image_size],
        "pixel_scaling": "ToTensor scales RGB values to [0, 1] before ImageNet normalization",
        "color_normalization": args.color_normalization,
        "background_removal": args.remove_background,
        "background_threshold": args.background_threshold,
        "training_augmentation": "horizontal/vertical flips, rotation up to 20 degrees, brightness and contrast jitter",
    }
    with (args.output_dir / "preprocessing.json").open("w", encoding="utf-8") as file:
        json.dump(preprocessing, file, indent=2)

    frame = discover_cropped_images(args.data_dir)
    train, val, test = stratified_splits(frame, args.seed)
    for name, split in (("train", train), ("validation", val), ("test", test)):
        split.to_csv(args.output_dir / f"{name}_manifest.csv", index=False)
    print(f"Using {len(frame)} validated cropped images: {dict(sorted(Counter(frame.label).items()))}")
    print(f"Split sizes — train: {len(train)}, validation: {len(val)}, test: {len(test)}")
    print("Note: these are image-level splits, not patient-level clinical validation.")

    train_loader, val_loader, test_loader = make_loaders(train, val, test, args.image_size, args.batch_size, args.workers, args.color_normalization, args.remove_background, args.background_threshold)
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model = models.efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASS_NAMES))
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_state, best_f1 = None, float("-inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            optimizer.zero_grad(set_to_none=True)
            logits = model(images.to(device, non_blocking=True))
            loss = criterion(logits, labels.to(device, non_blocking=True))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        validation = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_accuracy": validation["accuracy"], "validation_macro_f1": validation["macro_f1"]}
        history.append(row)
        print(json.dumps(row))
        if validation["macro_f1"] > best_f1:
            best_f1, best_state = validation["macro_f1"], copy.deepcopy(model.state_dict())

    torch.save({"model_state_dict": best_state, "class_names": CLASS_NAMES, "image_size": args.image_size}, args.output_dir / "best_model.pt")
    pd.DataFrame(history).to_csv(args.output_dir / "history.csv", index=False)
    model.load_state_dict(best_state)
    result = evaluate(model, test_loader, device)
    metrics = {
        "accuracy": result["accuracy"],
        "macro_f1": result["macro_f1"],
        "classification_report": classification_report(result["targets"], result["predictions"], target_names=CLASS_NAMES, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(result["targets"], result["predictions"], labels=range(len(CLASS_NAMES))).tolist(),
    }
    with (args.output_dir / "test_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    print(f"Saved best model and held-out metrics to {args.output_dir}")


if __name__ == "__main__":
    main()
