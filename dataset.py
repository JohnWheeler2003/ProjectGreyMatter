import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import config

def get_dataloaders():
    # Training transforms with Data Augmentation & Grayscale
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(config.MEAN, config.STD)
    ])

    # Validation/Testing transforms (No Augmentation, but Grayscale)
    eval_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(config.MEAN, config.STD)
    ])

    # Load datasets
    train_data = datasets.ImageFolder(root=config.TRAIN_DIR, transform=train_transform)
    val_data = datasets.ImageFolder(root=config.VAL_DIR, transform=eval_transform)  
    test_data = datasets.ImageFolder(root=config.TEST_DIR, transform=eval_transform)

    # Create Dataloaders
    train_loader = DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=config.BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader, test_data