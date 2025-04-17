import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from comet_ml import Experiment
from diffusers import DDPMPipeline, DDPMScheduler
from dotenv import load_dotenv
from tqdm import tqdm

from datasets import get_mnist_dataloaders
from models import create_model
from utils import load_checkpoint, save_checkpoint, save_images

load_dotenv()
COMET_API_KEY = os.getenv("COMET_API_KEY")
COMET_WORKSPACE = os.getenv("COMET_WORKSPACE")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a diffusion model on MNIST")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size training")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs to train for")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--image_size", type=int, default=32, help="Image size")
    # image_size: 32: power of 2 is necessary
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=5,
        help="Interval to evaluate and save samples",
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--log_every", type=int, default=100, help="Log batch metrics every N steps"
    )

    return parser.parse_args()


def train() -> None:
    args = parse_args()

    experiment = Experiment(
        api_key=COMET_API_KEY,
        project_name="mnist-diffusion",
        workspace=COMET_WORKSPACE,
        auto_param_logging=False,
        auto_metric_logging=False,
    )
    experiment.set_name(f"LR-{args.lr}-epochs-{args.epochs}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    experiment.log_parameter("device", str(device))

    config = {
        "image_size": args.image_size,
        "in_channels": 1,
        "out_channels": 1,
        "layers_per_block": 2,
        "block_out_channels": (64, 128, 256, 512),
        "down_block_types": (
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
        ),
        "up_block_types": ("UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
    }

    # Log hyperparameters
    experiment.log_parameters(vars(args))
    experiment.log_parameters(config, prefix="model_config")

    model = create_model(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    experiment.log_parameter("total_parameters", total_params)
    print(f"Total parameters: {total_params}")

    # Create noise scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
    )
    experiment.log_parameters(noise_scheduler.config, prefix="scheduler_config")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    train_loader, val_loader = get_mnist_dataloaders(
        batch_size=args.batch_size, image_size=args.image_size
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        start_epoch, previous_loss = load_checkpoint(model, optimizer, args.resume)
        experiment.log_parameter("resumed_from", args.resume)
        experiment.log_parameter("resumed_epoch", start_epoch)
        experiment.log_metric("resumed_loss", previous_loss, step=start_epoch * len(train_loader))

    Path("./figures").mkdir(parents=True, exist_ok=True)
    Path("./checkpoints").mkdir(parents=True, exist_ok=True)

    # Training loop
    global_step = start_epoch * len(train_loader)  # Initialize global step correctly if resuming
    start_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_start_time = time.time()

        progress_bar = tqdm(train_loader)
        progress_bar.set_description(f"Epoch {epoch + 1}/{args.epochs}")

        for _, (images, _) in enumerate(progress_bar):
            images = images.to(device)
            batch_size = images.shape[0]

            # Sample noise to add to the images
            noise = torch.randn_like(images).to(device)

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (batch_size,),
                device=device,
            ).long()

            # Add noise to the images according to the noise magnitude at each timestep
            noisy_images = noise_scheduler.add_noise(images, noise, timesteps)

            # Predict the noise
            noise_pred = model(noisy_images, timesteps).sample

            loss = nn.MSELoss()(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            epoch_loss += batch_loss
            progress_bar.set_postfix(loss=batch_loss)

            # Log batch loss periodically
            if global_step % args.log_every == 0:
                experiment.log_metric("batch_loss", batch_loss, step=global_step)
                experiment.log_metric(
                    "learning_rate", optimizer.param_groups[0]["lr"], step=global_step
                )

            global_step += 1

        # average loss for the epoch
        avg_loss = epoch_loss / len(train_loader)
        epoch_duration = time.time() - epoch_start_time
        print(
            f"Epoch {epoch + 1}/{args.epochs} - Avg Loss: {avg_loss:.6f} - Duration: {epoch_duration:.2f}s"
        )

        # Log epoch metrics
        experiment.log_metric("epoch_loss", avg_loss, epoch=epoch + 1, step=global_step)
        experiment.log_metric("epoch_duration", epoch_duration, epoch=epoch + 1, step=global_step)

        if (epoch + 1) % args.eval_interval == 0 or epoch == args.epochs - 1:
            # Create pipeline for inference
            pipeline = DDPMPipeline(unet=model, scheduler=noise_scheduler)

            model.eval()

            # Generate images
            with torch.no_grad():
                # 64 images in one batch
                images = pipeline(
                    batch_size=64,
                    generator=torch.manual_seed(epoch),
                    output_type="tensor",
                ).images

            sample_path = Path("figures") / (
                f"LR-{args.lr}-epochs-{args.epochs}/samples_epoch_{epoch + 1}.png"
            )
            save_images(images, sample_path)

            checkpoint_path = Path("checkpoints") / (
                f"LR-{args.lr}-epochs-{args.epochs}-model_epoch_{epoch + 1}.pth"
            )
            save_checkpoint(model, optimizer, epoch + 1, avg_loss, checkpoint_path)

    total_training_time = time.time() - start_time
    print(f"Training completed in {total_training_time:.2f} seconds!")
    experiment.log_metric("total_training_time", total_training_time)

    experiment.end()


if __name__ == "__main__":
    train()
