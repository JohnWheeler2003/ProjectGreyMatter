import torch
import numpy as np
import cv2
from PIL import Image, ImageOps
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import config

class SafeAutocropSquare(object):
    """
    Finds the tight bounding box of the largest contiguous bright shape (the brain),
    uses morphological opening to detach touching artifacts (like scales),
    crops out peripheral noise/text, and pads to a perfect square.
    """
    def __init__(self, buffer=5):
        self.buffer = buffer

    def __call__(self, img):
        image_np = np.array(img)
        
        if len(image_np.shape) == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_np

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 45, 255, cv2.THRESH_BINARY)

        # Morphological Opening
        # This breaks thin connections between the skull and touching artifacts.
        # A 5x5 or 7x7 kernel is usually enough to sever thin scale lines.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        thresh_opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # Find contours on the *opened* mask, not the raw threshold
        contours, _ = cv2.findContours(thresh_opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return img 

        largest_contour = max(contours, key=cv2.contourArea)

        x, y, w, h = cv2.boundingRect(largest_contour)

        x_min = max(0, x - self.buffer)
        y_min = max(0, y - self.buffer)
        x_max = min(image_np.shape[1], x + w + self.buffer)
        y_max = min(image_np.shape[0], y + h + self.buffer)

        cropped = image_np[y_min:y_max, x_min:x_max]

        h_c, w_c = cropped.shape[:2]
        max_side = max(h_c, w_c)
        
        top = (max_side - h_c) // 2
        bottom = max_side - h_c - top
        left = (max_side - w_c) // 2
        right = max_side - w_c - left

        if len(image_np.shape) == 3:
            squared_image = cv2.copyMakeBorder(cropped, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        else:
            squared_image = cv2.copyMakeBorder(cropped, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

        return Image.fromarray(squared_image)

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
    """
    Initializes and returns PyTorch DataLoaders for training, validation, and testing with MRI-specific preprocessing.
    The pipeline applies adaptive contrast enhancement and automated cropping to standardize brain scans across the dataset.
    """
    target_size = 224 if model_name == "vit" else config.IMAGE_SIZE

    # TRAINING TRANSFORMS
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        SafeAutocropSquare(buffer=10),
        ApplyCLAHE(clip_limit=2.0),
        transforms.Resize((target_size, target_size)),
        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(config.MEAN, config.STD)
    ])

    # VALIDATION/TESTING TRANSFORMS 
    eval_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        SafeAutocropSquare(buffer=10),
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