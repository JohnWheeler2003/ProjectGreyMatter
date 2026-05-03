import torch
import numpy as np
import cv2
from PIL import Image, ImageOps
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import config

class SafeAutocropSquare(object):
    """
    Finds the tight bounding box of non-zero pixels (the brain), crops it, 
    and pads the shorter side with black pixels to create a perfect square, 
    preserving the natural aspect ratio before resizing.
    """
    def __init__(self, threshold=5):
        # Using a slight threshold (e.g., 5) instead of 0 to aggressively 
        # ignore faint MRI background noise and faint text artifacts.
        self.threshold = threshold 

    def __call__(self, img):
        # Convert PIL to numpy
        img_np = np.array(img)
        
        # Identify rows and columns that contain brain tissue
        rows = np.any(img_np > self.threshold, axis=1)
        cols = np.any(img_np > self.threshold, axis=0)

        # Fallback in case of a completely black/empty image
        if not rows.any() or not cols.any():
            return img

        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]

        # Crop to the tight bounding box
        cropped_img = img.crop((xmin, ymin, xmax, ymax))

        # Determine padding needed to make it a perfect square
        w, h = cropped_img.size
        max_side = max(w, h)
        pad_w = max_side - w
        pad_h = max_side - h

        # Pad equally on both sides (left, top, right, bottom)
        padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
        squared_img = ImageOps.expand(cropped_img, padding, fill=0)

        return squared_img

class ApplyCLAHE(object):
    """
    Applies Contrast Limited Adaptive Histogram Equalization to enhance 
    soft-tissue contrast and downplay the bright skull intensities.
    """
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img):
        # Convert PIL to numpy array
        img_np = np.array(img)
        
        # Ensure uint8 format for OpenCV
        if img_np.dtype != np.uint8:
            img_np = img_np.astype(np.uint8)

        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        clahe_img = clahe.apply(img_np)

        # Return as PIL Image to continue the torchvision pipeline
        return Image.fromarray(clahe_img)


def get_dataloaders(model_name):
    target_size = 224 if model_name == "vit" else config.IMAGE_SIZE

    # TRAINING TRANSFORMS
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        SafeAutocropSquare(threshold=5),       # 1. Isolate and square the brain
        ApplyCLAHE(clip_limit=2.0),            # 2. Enhance contrast
        transforms.Resize((target_size, target_size)), # 3. Safely resize
        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.15)),
        transforms.Normalize(config.MEAN, config.STD)
    ])

    # VALIDATION/TESTING TRANSFORMS 
    eval_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        SafeAutocropSquare(threshold=5),       # Must apply exact same preprocessing
        ApplyCLAHE(clip_limit=2.0),
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
        transforms.Normalize(config.MEAN, config.STD)
    ])

    # LOAD DATASETS
    train_data = datasets.ImageFolder(root=config.TRAIN_DIR, transform=train_transform)
    val_data = datasets.ImageFolder(root=config.VAL_DIR, transform=eval_transform)  
    test_data = datasets.ImageFolder(root=config.TEST_DIR, transform=eval_transform)

    # CREATE DATALOADERS
    train_loader = DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=config.BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader, test_data