import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import config

def get_dataloaders(model_name):
    target_size = 224 if model_name == "vit" else config.IMAGE_SIZE

    # Training transforms with Data Augmentation & Grayscale
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((target_size, target_size)),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(config.MEAN, config.STD)
    ])

    # Validation/Testing transforms (No Augmentation, but Grayscale)
    eval_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((target_size, target_size)),
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