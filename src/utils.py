from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def save_images(
    images: np.ndarray,
    path: str,
    n_rows: int = 8,
    n_cols: int = 8,
    fig_size: tuple = (10, 10),
) -> None:
    """
    Save a batch of images to a grid.

    Args:
        images: Tensor or ndarray of images (B, H, W, C) in range [-1, 1]
        path: Path to save the figure to
        n_rows, n_cols: Number of rows and columns in the grid
        fig_size: Size of the figure
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    # Denormalize images from [-1, 1] to [0, 1]
    images = (images + 1) / 2.0

    plt.figure(figsize=fig_size)

    print(f"Full images tensor shape: {images.shape}")
    for i in range(min(n_rows * n_cols, images.shape[0])):
        plt.subplot(n_rows, n_cols, i + 1)

        img = images[i, :, :, 0]

        plt.imshow(img, cmap="gray")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Images saved to {path}")


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: str,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        path,
    )
    print(f"Checkpoint saved to {path}")


def load_checkpoint(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer, path: str
) -> tuple[int, float]:
    """
    Load model checkpoint.
    Args:
        model: The model to load weights into
        optimizer: The optimizer to load state into
        path: Path to load the checkpoint from
    Returns:
        epoch, loss: The epoch number and loss value of the loaded checkpoint
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    print(f"Checkpoint loaded from {path}")
    return epoch, loss
