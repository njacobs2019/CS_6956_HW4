from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_mnist_dataloaders(
    batch_size: int, image_size: int = 32, data_dir: str = "./dataset"
) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x * 2) - 1),
        ]
    )

    # Load the MNIST dataset
    train_dataset = datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )

    val_dataset = datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform
    )

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
