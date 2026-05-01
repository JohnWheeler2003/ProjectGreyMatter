import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM, GradCAMElementWise
from pytorch_grad_cam.utils.image import show_cam_on_image

import config
from dataset import get_dataloaders
from model import BrainTumorCNN, PretrainedResNet, PretrainedViT
from utils import unnormalize_tensor, get_normalize_params

# ViT RESHAPE TRANSFORM
# ViT outputs a sequence of tokens [Batch, 197, 768] (1 CLS token + 196 patch tokens for a 224x224 image).
# Strip the CLS token and reshape the 196 patches back into a 14x14 spatial grid.
def vit_reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    # Bring the channels to the first dimension: [Batch, Channels, Height, Width]
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def find_evaluation_cases(model, test_loader, class_names):
    """
    Hunts through the test_loader to find one successful and one failed 
    prediction for each class.
    """
    # Dictionaries to store: class_name -> (image_tensor, predicted_class_name)
    successes = {name: None for name in class_names}
    failures = {name: None for name in class_names}
    
    model.eval()
    
    with torch.no_grad(): # No gradients needed just to find the images
        for images, labels in test_loader:
            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            for i in range(len(labels)):
                true_idx = labels[i].item()
                pred_idx = preds[i].item()
                true_class = class_names[true_idx]
                pred_class = class_names[pred_idx]
                
                # Check for success
                if true_idx == pred_idx and successes[true_class] is None:
                    # Move tensor back to CPU to save VRAM while we hunt
                    successes[true_class] = (images[i].cpu(), pred_class)
                
                # Check for failure
                elif true_idx != pred_idx and failures[true_class] is None:
                    failures[true_class] = (images[i].cpu(), pred_class)
            
            # Early stopping check: if all slots are filled, stop hunting
            all_successes_found = all(v is not None for v in successes.values())
            all_failures_found = all(v is not None for v in failures.values())
            
            if all_successes_found and all_failures_found:
                break
                
    return successes, failures

def plot_gradcam_grid(cases_dict, category_name, model_name, cam, mean, std):
    """
    Plots the Grad-CAM grid for either successes or failures.
    """
    # Filter out classes that didn't have a case (e.g., if there were no failures)
    valid_cases = {k: v for k, v in cases_dict.items() if v is not None}
    missing_cases = [k for k, v in cases_dict.items() if v is None]
    
    num_images = len(class_names := list(cases_dict.keys()))
    
    fig, axes = plt.subplots(num_images, 3, figsize=(12, 4 * num_images))
    fig.suptitle(f"{model_name.upper()} - Grad-CAM {category_name}", fontsize=16)
    
    # Handle the 1D array edge case if there is only 1 class (unlikely, but safe)
    if num_images == 1: axes = np.expand_dims(axes, axis=0)

    for i, true_class in enumerate(cases_dict.keys()):
        case = cases_dict[true_class]
        
        # If no image was found for this category (e.g., a perfect model has no failures)
        if case is None:
            for j in range(3):
                axes[i, j].axis('off')
            axes[i, 1].text(0.5, 0.5, f"No {category_name.lower()} found\nfor true class: {true_class}", 
                            ha='center', va='center', fontsize=12, color='gray')
            continue
            
        img_tensor, pred_class = case
        
        # Move back to device for Grad-CAM processing
        img_tensor = img_tensor.to(config.DEVICE)
        
        # Generate CAM
        grayscale_cam = cam(input_tensor=img_tensor.unsqueeze(0), targets=None)[0, :]
        
        # Prepare image for visualization
        img_np = unnormalize_tensor(img_tensor, mean, std) 
        img_rgb = np.repeat(img_np, 3, axis=2) 
        cam_overlay = show_cam_on_image(img_rgb, grayscale_cam, use_rgb=True)

        # Plot Original
        axes[i, 0].imshow(img_rgb)
        axes[i, 0].set_title(f"True: {true_class}\nPred: {pred_class}")
        axes[i, 0].axis('off')

        # Plot Heatmap
        axes[i, 1].imshow(grayscale_cam, cmap='jet')
        axes[i, 1].set_title("Heatmap")
        axes[i, 1].axis('off')

        # Plot Overlay
        axes[i, 2].imshow(cam_overlay)
        axes[i, 2].set_title("Overlay")
        axes[i, 2].axis('off')

    plt.tight_layout()
    save_path = f"{model_name}_gradcam_{category_name.lower()}.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Saved {category_name} visualizations to {save_path}")
    
    if missing_cases:
        print(f"  -> Note: No {category_name.lower()} found for classes: {', '.join(missing_cases)}")

def visualize_gradcam(model_name):
    print(f"Running Grad-CAM for: {model_name}")

    # 1. LOAD DATA (Pull from the test loader)
    _, _, test_loader, test_data = get_dataloaders(model_name)
    class_names = test_data.classes

    # Get normalization params for plotting
    mean, std = get_normalize_params(test_data.transform)
    if mean is None or std is None:
        mean, std = config.MEAN, config.STD

    # 2. INITIALIZE MODEL & TARGET LAYERS
    if model_name == "custom_cnn":
        model = BrainTumorCNN().to(config.DEVICE)
        target_layers = [model.conv5]
        cam_class = GradCAM
        use_reshape = False

    elif model_name == "resnet":
        model = PretrainedResNet(grayscale=True).to(config.DEVICE)
        # Standard target layer for ResNet50 is the last bottleneck layer
        target_layers = [model.resnet.layer4[-1]] 
        cam_class = GradCAM
        use_reshape = False

    elif model_name == "vit":
        model = PretrainedViT(grayscale=True).to(config.DEVICE)
        # Target the LayerNorm of the final encoder block
        target_layers = [model.vit.encoder.layers[-1].ln_1]
        cam_class = GradCAMElementWise
        use_reshape = True
    else:
        raise ValueError("Invalid model name selected.")

    # 3. LOAD CHECKPOINT
    checkpoint_path = f"{model_name}_best_model.pth"
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=config.DEVICE, weights_only=True)
        model.load_state_dict(ckpt["model_state"])
        print(f"Loaded {model_name} model from epoch {ckpt['epoch']}")
    else:
        print(f"No checkpoint found at '{checkpoint_path}'. Train the model first.")
        return

    model.eval()

    # 4. INITIALIZE GRAD-CAM
    cam = cam_class(
        model=model, 
        target_layers=target_layers, 
        reshape_transform=vit_reshape_transform if use_reshape else None
    )

    # 5. HUNT AND PLOT
    print("Hunting for evaluation cases in the test set...")
    successes, failures = find_evaluation_cases(model, test_loader, class_names)
    
    print("Generating Success Grids...")
    plot_gradcam_grid(successes, "Successes", model_name, cam, mean, std)
    
    print("Generating Failure Grids...")
    plot_gradcam_grid(failures, "Failures", model_name, cam, mean, std)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Grad-CAM on Trained Models")
    parser.add_argument("--model", type=str, default="custom_cnn", 
                        choices=["custom_cnn", "resnet", "vit"], 
                        help="Name of the model to evaluate")
    args = parser.parse_args()

    visualize_gradcam(args.model)