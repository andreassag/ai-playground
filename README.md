# PyTorch MNIST Pipeline

A small PyTorch project for training a convolutional neural network on the MNIST handwritten digit dataset.

## What it does

- Builds and trains a CNN model with residual blocks and squeeze-and-excitation attention.
- Loads MNIST using a custom data pipeline with deskewing, affine augmentation, elastic deformation, and normalization.
- Logs metrics to TensorBoard and saves trained model weights.

## Usage

Train the model from the repository root:

```bash
python src/train.py
```

Common options:

- `--data-folder`: root folder for MNIST data (default `data/`)
- `--batch-size`: training batch size
- `--epochs`: number of training epochs
- `--lr`: initial learning rate
- `--max-lr`: peak learning rate for OneCycleLR
- `--log-dir`: TensorBoard log directory
- `--save-model`: output path for model weights

## Project layout

- `src/train.py`: training loop, evaluation, and model saving
- `src/model.py`: CNN architecture with ResBlocks and SE attention
- `src/load_mnist.py`: MNIST dataset loading and augmentation pipeline
- `data/MNIST/raw`: expected raw MNIST files
- `models/`: saved model weights
- `runs/`: TensorBoard logs
- `notebook/`: example notebook

## Notes

This project is designed as a simple end-to-end example of a PyTorch image classification pipeline with a strong data augmentation setup for MNIST.
