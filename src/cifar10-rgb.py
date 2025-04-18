import os
from dataclasses import asdict

import comet_ml
import torch
from diffusers import DDPMScheduler, UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .config import TrainingConfig
from .training import train_loop

config = TrainingConfig(device=torch.device("cuda:1"))


transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

print("Getting dataset...............", end="")
train_ds = datasets.CIFAR10(
    root=config.ds_root, train=True, transform=transform, download=True
)
test_ds = datasets.CIFAR10(
    root=config.ds_root, train=False, transform=transform, download=True
)
print("Done")

train_loader = DataLoader(
    train_ds,
    batch_size=config.train_batch_size,
    shuffle=True,
    num_workers=config.num_workers,
)
test_loader = DataLoader(
    test_ds,
    batch_size=config.eval_batch_size,
    shuffle=False,
    num_workers=config.num_workers,
)

model = UNet2DModel(
    sample_size=config.image_size,  # the target image resolution
    in_channels=3,  # the number of input channels, 3 for RGB images
    out_channels=3,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    block_out_channels=(
        64,
        64,
        128,
        128,
        256,
        # 128,
        # 128,
        # 256,
        # 256,
        # 512,
        # 512,
    ),  # the number of output channels for each UNet block
    down_block_types=(
        "DownBlock2D",  # a regular ResNet downsampling block
        # "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",  # a regular ResNet upsampling block
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",
        # "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
)


noise_scheduler = DDPMScheduler(num_train_timesteps=1000)


optimizer = torch.optim.AdamW(
    model.parameters(), lr=config.learning_rate, foreach=False
)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_loader) * config.num_epochs),
)

experiment = comet_ml.start(
    api_key=os.getenv("COMET_API_KEY"),
    project_name="pdl-hw4",
    mode="create",
    online=True,
    experiment_config=comet_ml.ExperimentConfig(
        auto_metric_logging=False,
        disabled=False,  # Set True for debugging runs
        name="cifar10_rgb",
    ),
)
experiment.log_parameters(asdict(config))

train_loop(
    config,
    model,
    noise_scheduler,
    optimizer,
    train_loader,
    test_loader,
    lr_scheduler,
    experiment,
)
