"""Load MNIST data into PyTorch DataLoaders using torchvision.

This loader applies strong training augmentation to handwritten digits,
including deskewing, elastic deformation, random rotation, zooming, and shifting.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class Deskew:
    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.convert("L")
        arr = np.asarray(image, dtype=np.float32)
        arr = 255.0 - arr
        total = arr.sum()
        if total <= 0:
            return image

        coords = np.indices(arr.shape)
        y = coords[0].astype(np.float32)
        x = coords[1].astype(np.float32)
        x_mean = (arr * x).sum() / total
        y_mean = (arr * y).sum() / total

        x_cent = x - x_mean
        y_cent = y - y_mean
        mu11 = (arr * x_cent * y_cent).sum() / total
        mu20 = (arr * x_cent * x_cent).sum() / total
        mu02 = (arr * y_cent * y_cent).sum() / total

        if np.isclose(mu20, mu02):
            return image

        angle = 0.5 * np.arctan2(2.0 * mu11, mu20 - mu02)
        degrees = np.degrees(angle)
        return image.rotate(-degrees, fillcolor=255, resample=Image.BILINEAR)


class ElasticTransform:
    def __init__(self, alpha: float = 28.0, sigma: float = 4.0, prob: float = 0.5) -> None:
        self.alpha = alpha
        self.sigma = sigma
        self.prob = prob

    def _gaussian_kernel(self, kernel_size: int, sigma: float, device: torch.device) -> torch.Tensor:
        ax = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, device=device, dtype=torch.float32)
        xx, yy = torch.meshgrid(ax, ax, indexing="xy")
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        return kernel / kernel.sum()

    def _smooth(self, displacement: torch.Tensor) -> torch.Tensor:
        kernel_size = max(3, int(self.sigma * 4) | 1)
        kernel = self._gaussian_kernel(kernel_size, self.sigma, displacement.device)
        kernel = kernel.view(1, 1, kernel_size, kernel_size)
        return F.conv2d(displacement, kernel, padding=kernel_size // 2)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.prob:
            return tensor

        if tensor.ndim != 3:
            raise ValueError("ElasticTransform expects a 3D tensor [C, H, W]")

        _, height, width = tensor.shape
        displacement_x = self._smooth(torch.randn(1, 1, height, width, device=tensor.device)) * self.alpha
        displacement_y = self._smooth(torch.randn(1, 1, height, width, device=tensor.device)) * self.alpha

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=tensor.device),
            torch.linspace(-1.0, 1.0, width, device=tensor.device),
            indexing="xy",
        )
        grid_x = grid_x.unsqueeze(0).unsqueeze(3)
        grid_y = grid_y.unsqueeze(0).unsqueeze(3)

        dx = (displacement_x * 2.0 / max(width - 1, 1)).permute(0, 2, 3, 1)
        dy = (displacement_y * 2.0 / max(height - 1, 1)).permute(0, 2, 3, 1)

        grid = torch.cat((grid_x + dx, grid_y + dy), dim=3)
        warped = F.grid_sample(
            tensor.unsqueeze(0),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return warped.squeeze(0)


def build_mnist_dataloaders(
    root: Path,
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = False,
    download: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    train_transform = transforms.Compose([
        Deskew(),
        transforms.RandomAffine(
            degrees=15,
            translate=(0.1, 0.1),
            scale=(0.9, 1.15),
            shear=8,
            fill=255,
        ),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.4),
        transforms.ToTensor(),
        ElasticTransform(alpha=20.0, sigma=4.0, prob=0.5),
        transforms.RandomErasing(p=0.40, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=1.0),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    test_transform = transforms.Compose([
        Deskew(),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_dataset = datasets.MNIST(
        root=str(root),
        train=True,
        download=download,
        transform=train_transform,
    )
    test_dataset = datasets.MNIST(
        root=str(root),
        train=False,
        download=download,
        transform=test_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, test_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load MNIST into PyTorch DataLoaders")
    parser.add_argument(
        "--data-folder",
        default=Path(__file__).resolve().parents[1] / "data",
        type=Path,
        help="Root folder containing the MNIST dataset",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    train_loader, test_loader = build_mnist_dataloaders(
        root=args.data_folder,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        download=True,
    )

    print(f"Loaded train batches: {len(train_loader)}")
    print(f"Loaded test batches: {len(test_loader)}")

    sample_images, sample_labels = next(iter(train_loader))
    print("Sample batch shapes:")
    print(f"  images: {sample_images.shape}")
    print(f"  labels: {sample_labels.shape}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
