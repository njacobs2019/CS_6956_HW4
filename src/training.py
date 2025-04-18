import comet_ml
import comet_ml.integration
import comet_ml.integration.pytorch
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
    train_loader: DataLoader,
    val_loader: DataLoader,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    comet_experiment: comet_ml.Experiment | None,
) -> float:
    model.to(config.device)

    pipeline = DDPMPipeline(unet=model, scheduler=noise_scheduler)

    # Now you train the model
    for epoch in tqdm(range(config.num_epochs), unit="Epoch"):
        # Training
        model.train()
        train_loss = 0.0
        total_samples = 0

        for batch in tqdm(
            train_loader,
            leave=False,
            total=len(train_loader),
            unit="Batch",
            desc="Training",
        ):
            with torch.no_grad():
                clean_images, _labels = batch
                clean_images = clean_images.to(config.device)

                batch_size = clean_images.shape[0]
                total_samples += batch_size

                # Sample noise to add to the images
                noise = torch.randn(clean_images.shape, device=clean_images.device)

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

            # Predict the noise
            noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
            loss = F.mse_loss(noise_pred, noise)
            loss.backward()
            train_loss += loss.item() * batch_size

            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
            optimizer.zero_grad()

        train_loss = train_loss / total_samples

        # Validation Loop
        model.eval()
        val_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in tqdm(
                val_loader,
                leave=False,
                total=len(val_loader),
                unit="Batch",
                desc="Validation",
            ):
                # Prepare batch
                clean_images, _labels = batch
                clean_images = clean_images.to(config.device)

                batch_size = clean_images.shape[0]
                total_samples += batch_size

                # Deterministically sample noise and timesteps
                noise = torch.randn(
                    clean_images.shape,
                    device=clean_images.device,
                    generator=torch.Generator(device=config.device).manual_seed(
                        config.seed
                    ),
                )
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (batch_size,),
                    dtype=torch.int64,
                    generator=torch.Generator(device=config.device).manual_seed(
                        config.seed
                    ),
                    device=config.device,
                )

                # Forward diffusion (add noise)
                noisy_images = noise_scheduler.add_noise(
                    clean_images, noise, timesteps
                ).to(config.device)

                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                val_loss += loss.item() * batch_size

        val_loss = val_loss / total_samples

        # Log the loss
        if comet_experiment is not None:
            comet_experiment.log_metrics(
                {"train_loss": train_loss, "val_loss": val_loss}, epoch=epoch + 1
            )

        # Sample some images
        comet_condition = comet_experiment is not None
        epoch_condition = epoch % config.save_image_epochs == 0
        final_epoch = epoch == config.num_epochs - 1
        if comet_condition and (epoch_condition or final_epoch):
            evaluate(config, pipeline, epoch, comet_experiment)

    return val_loss


def evaluate(
    config: TrainingConfig, pipeline, epoch: int, experiment: comet_ml.Experiment
) -> None:
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(
        batch_size=config.save_num_imgs,
        generator=torch.Generator(device="cpu").manual_seed(config.seed),
    ).images

    # Make a grid out of the images
    image_grid = make_image_grid(images, rows=4, cols=4)

    # Setup folder and get file path
    output_folder = config.save_folder / str(experiment.get_key())
    output_folder.mkdir(parents=True, exist_ok=True)
    fname = output_folder / f"epoch_{epoch}.png"

    image_grid.save(fname)
