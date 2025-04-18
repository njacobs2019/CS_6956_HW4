from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class TrainingConfig:
    device: torch.device
    image_size: int = 32
    train_batch_size: int = 256
    eval_batch_size: int = 256
    num_workers: int = 16
    num_epochs: int = 50
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 500
    save_image_epochs: int = 5  # save images every x epochs
    save_num_imgs: int = 16
    seed: int = 0
    ds_root: Path = Path("./datasets")
    save_folder: Path = Path("./figures")

    def __post_init__(self) -> None:
        self.ds_root.mkdir(parents=True, exist_ok=True)
