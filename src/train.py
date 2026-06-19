"""Training pipeline for MNIST digit recognition with a single CNN."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from load_mnist import build_mnist_dataloaders
from model import CNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a single MNIST CNN")
    parser.add_argument(
        "--data-folder",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
        help="Root folder containing the MNIST dataset",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-lr", type=float, default=1e-3, help="Peak LR for OneCycleLR")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to use for training",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print debugging and runtime information"
    )
    parser.add_argument("--log-dir", type=Path, default=Path("runs"))
    parser.add_argument(
        "--save-model",
        type=Path,
        default=Path("models") / "cnn.zip",
        help="Path to save the trained model file.",
    )
    return parser.parse_args()


def get_device(device_choice: str = "auto") -> torch.device:
    if device_choice == "cpu":
        return torch.device("cpu")

    if device_choice == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise SystemExit("CUDA is not available on this machine.")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item() * images.size(0)
        _, predictions = output.max(1)
        correct += int(predictions.eq(labels).sum().item())
        total += labels.size(0)

    return running_loss / total, correct / total


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)
            loss = criterion(output, labels)

            running_loss += loss.item() * images.size(0)
            _, predictions = output.max(1)
            correct += int(predictions.eq(labels).sum().item())
            total += labels.size(0)

    return running_loss / total, correct / total


def save_model_state(model: torch.nn.Module, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)


def train_model(
    name: str,
    model: torch.nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    writer: SummaryWriter,
) -> torch.nn.Module:
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.max_lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        div_factor=25.0,
        final_div_factor=1e4,
    )

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train_epoch(
            model, train_loader, criterion, optimizer, device, scheduler
        )
        eval_loss, eval_accuracy = evaluate(model, test_loader, criterion, device)

        writer.add_scalar(f"{name}/Train/Loss", train_loss, epoch)
        writer.add_scalar(f"{name}/Train/Accuracy", train_accuracy, epoch)
        writer.add_scalar(f"{name}/Eval/Loss", eval_loss, epoch)
        writer.add_scalar(f"{name}/Eval/Accuracy", eval_accuracy, epoch)

        if args.verbose:
            print(
                f"[{name}] Epoch {epoch}/{args.epochs}: "
                f"train_loss={train_loss:.4f}, train_acc={train_accuracy:.4f}, "
                f"eval_loss={eval_loss:.4f}, eval_acc={eval_accuracy:.4f}"
            )

    save_path = args.save_model
    save_model_state(model, save_path)
    if args.verbose:
        print(f"Saved model weights to {save_path}")

    return model


def main() -> int:
    args = parse_args()
    device = get_device(args.device)

    if args.verbose:
        print("Training configuration:")
        print(f"  device: {device}")
        print(f"  batch size: {args.batch_size}")
        print(f"  epochs: {args.epochs}")
        print(f"  lr: {args.lr}")
        print(f"  max lr: {args.max_lr}")
        print(f"  weight decay: {args.weight_decay}")
        print(f"  num workers: {args.num_workers}")
        print(f"  pin memory: {args.pin_memory}")
        print(f"  log dir: {args.log_dir}")
        print(f"  save model: {args.save_model}")

    train_loader, test_loader = build_mnist_dataloaders(
        root=args.data_folder,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    args.log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(args.log_dir))
    args.save_model.parent.mkdir(parents=True, exist_ok=True)

    model = CNN()
    train_model("cnn", model, train_loader, test_loader, device, args, writer)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
