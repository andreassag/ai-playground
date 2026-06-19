"""Residual CNN with Squeeze-and-Excitation attention for MNIST digit recognition.

Architecture (~2.5 M params):
  Stem  : Conv(1→64, 3×3) → BN → ReLU                           [B, 64, 28, 28]
  Stage1: ResBlock(64→64)  × 2 → SE(64)  → MaxPool              [B,  64, 14, 14]
  Stage2: ResBlock(64→128) × 2 → SE(128) → MaxPool              [B, 128,  7,  7]
  Stage3: ResBlock(128→256)× 2 → SE(256) → GlobalAvgPool        [B, 256]
  Head  : Dropout(0.5) → Linear(256→10)
"""

import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        mid = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        scale = self.pool(x).view(b, c)
        scale = self.fc(scale).view(b, c, 1, 1)
        return x * scale


class ResBlock(nn.Module):
    """Pre-activation residual block with optional projection shortcut."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut: nn.Module
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return self.relu(out)


class CNN(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Stage 1: 28×28 → 14×14
        self.stage1 = nn.Sequential(
            ResBlock(64, 64),
            ResBlock(64, 64),
            SEBlock(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Stage 2: 14×14 → 7×7
        self.stage2 = nn.Sequential(
            ResBlock(64, 128),
            ResBlock(128, 128),
            SEBlock(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Stage 3: 7×7 → 1×1 (global avg pool)
        self.stage3 = nn.Sequential(
            ResBlock(128, 256),
            ResBlock(256, 256),
            SEBlock(256),
            nn.AdaptiveAvgPool2d(1),
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
