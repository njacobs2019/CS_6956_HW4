import torch
import torch.nn.functional as F
from diffusers import DDPMPipeline
from diffusers.utils import make_image_grid
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import TrainingConfig


def train_loop(
    config: TrainingConfig,
    model,
    noise_scheduler,
    optimizer: torch.optim.Optimizer,
    train_dataloader: DataLoader,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None,
) -> None:
    global_step = 0

    model.eval()
    model.to(config.device)

    pipeline = DDPMPipeline(unet=model, scheduler=noise_scheduler)

    # Now you train the model
    for epoch in range(config.num_epochs):
        for batch in tqdm(train_dataloader, leave=False, total=len(train_dataloader)):
            with torch.no_grad():
                clean_images, _labels = batch
                clean_images = clean_images.to(config.device)

                # Sample noise to add to the images
                noise = torch.randn(clean_images.shape, device=clean_images.device)
                batch_size = clean_images.shape[0]

                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (batch_size,),
                    dtype=torch.int64,
                ).to(config.device)

                # Forward diffusion (add noise)
                noisy_images = noise_scheduler.add_noise(
                    clean_images, noise, timesteps
                ).to(config.device)

            # Predict the noise residual
            noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
            loss = F.mse_loss(noise_pred, noise)
            loss.backward()

            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
            optimizer.zero_grad()

            global_step += 1

        if epoch % 3 == 0:
            evaluate(config, epoch, pipeline)
        print(f"Finished epoch {epoch}")


def evaluate(config: TrainingConfig, epoch: int, pipeline) -> None:
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(
        batch_size=config.eval_batch_size,
        generator=torch.Generator(device="cpu").manual_seed(config.seed),
    ).images

    # Make a grid out of the images
    image_grid = make_image_grid(images, rows=4, cols=4)
