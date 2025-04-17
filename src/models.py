from typing import Any

from diffusers import UNet2DModel


def create_model(config: dict[str, Any]) -> UNet2DModel:
    model = UNet2DModel(
        sample_size=config["image_size"],  # Size of the input images
        in_channels=config["in_channels"],  # Number of input channels (1 for grayscale)
        out_channels=config["out_channels"],  # Number of output channels
        layers_per_block=config["layers_per_block"],  # Number of layers per block
        block_out_channels=config["block_out_channels"],  # Channels in each block
        down_block_types=config["down_block_types"],  # Types of down blocks
        up_block_types=config["up_block_types"],  # Types of up blocks
    )
    return model
