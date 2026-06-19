"""Evaluate a trained MNIST model and print a single accuracy percentage."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from load_mnist import build_mnist_dataloaders
from model import CNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained MNIST model")
    parser.add_argument(
        "--data-folder",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
        help="Root folder containing the official MNIST dataset files",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to a saved CNN model weight file.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size used for evaluation",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to evaluate on",
    )
    parser.add_argument("--verbose", action="store_true", help="Print extra debugging information")
    parser.add_argument(
        "--tta-passes",
        type=int,
        default=1,
        help="Number of test-time augmentation passes to average (1 = disabled)",
    )
    return parser.parse_args()


def get_device(device_choice: str) -> torch.device:
    if device_choice == "cpu":
        return torch.device("cpu")
    if device_choice == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise SystemExit("CUDA is not available on this machine.")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model_weights(model: torch.nn.Module, path: Path, device: torch.device) -> torch.nn.Module:
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    return model


def build_tta_transform() -> transforms.Compose:
    """Light augmentation used for each TTA pass (affine only, no elastic/erasing)."""
    return transforms.Compose([
        transforms.RandomAffine(
            degrees=10,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05),
            fill=0,
        ),
    ])


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    tta_passes: int = 1,
) -> float:
    model.eval()
    correct = 0
    total = 0
    tta_transform = build_tta_transform() if tta_passes > 1 else None

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            if tta_transform is None or tta_passes <= 1:
                probs = F.softmax(model(images), dim=1)
            else:
                # Average softmax probabilities over multiple augmented passes
                probs = F.softmax(model(images), dim=1)
                for _ in range(tta_passes - 1):
                    aug = torch.stack(
                        [tta_transform(img) for img in images.cpu()]
                    ).to(device)
                    probs = probs + F.softmax(model(aug), dim=1)
                probs = probs / tta_passes

            predictions = probs.argmax(dim=1)
            correct += int((predictions == labels).sum().item())
            total += labels.size(0)

    return float(correct) / float(total) * 100.0 if total else 0.0


def main() -> int:
    args = parse_args()
    device = get_device(args.device)

    if args.verbose:
        print(f"Evaluation device: {device}")
        print(f"Model path: {args.model_path}")
        print(f"Data folder: {args.data_folder}")
        print(f"Batch size: {args.batch_size}")
        print(f"TTA passes: {args.tta_passes}")

    _, test_loader = build_mnist_dataloaders(
        root=args.data_folder,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        download=False,
    )

    if len(test_loader.dataset) != 10000:
        raise SystemExit(
            f"Evaluation requires the official MNIST test set of 10000 images, "
            f"but the dataset at {args.data_folder} contains {len(test_loader.dataset)} samples."
        )

    model = CNN()
    model = load_model_weights(model, args.model_path, device)
    model.to(device)

    accuracy = evaluate_model(model, test_loader, device, tta_passes=args.tta_passes)
    print(f"Accuracy: {accuracy:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
