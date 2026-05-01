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

def visualize_gradcam(model_name, num_images=4):
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

    # 5. FETCH BATCH AND GENERATE CAM
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    
    images = images[:num_images].to(config.DEVICE)
    labels = labels[:num_images]

    # Generate CAM for the predicted class
    # (Passing targets=None automatically uses the highest scoring class)
    grayscale_cams = cam(input_tensor=images, targets=None)

    # 6. PLOT AND SAVE RESULTS
    fig, axes = plt.subplots(num_images, 3, figsize=(12, 4 * num_images))
    fig.suptitle(f"Grad-CAM Visualizations: {model_name.upper()}", fontsize=16)

    for i in range(num_images):
        img_tensor = images[i]
        true_label = class_names[labels[i].item()]
        
        # Get Model Prediction
        with torch.no_grad():
            output = model(img_tensor.unsqueeze(0))
            pred_idx = torch.argmax(output, dim=1).item()
            pred_label = class_names[pred_idx]

        # Prepare image for visualization (needs to be RGB [0,1] for overlay)
        img_np = unnormalize_tensor(img_tensor, mean, std) # Returns [H, W, 1]
        img_rgb = np.repeat(img_np, 3, axis=2) # Convert 1-channel to 3-channel RGB
        
        cam_map = grayscale_cams[i, :]
        cam_overlay = show_cam_on_image(img_rgb, cam_map, use_rgb=True)

        # Plot Original
        ax = axes[i, 0] if num_images > 1 else axes[0]
        ax.imshow(img_rgb)
        ax.set_title(f"True: {true_label}\nPred: {pred_label}")
        ax.axis('off')

        # Plot Grad-CAM Heatmap Only
        ax = axes[i, 1] if num_images > 1 else axes[1]
        ax.imshow(cam_map, cmap='jet')
        ax.set_title("Heatmap")
        ax.axis('off')

        # Plot Overlay
        ax = axes[i, 2] if num_images > 1 else axes[2]
        ax.imshow(cam_overlay)
        ax.set_title("Grad-CAM Overlay")
        ax.axis('off')

    plt.tight_layout()
    save_path = f"{model_name}_gradcam_results.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Saved Grad-CAM visualizations to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Grad-CAM on Trained Models")
    parser.add_argument("--model", type=str, default="custom_cnn", 
                        choices=["custom_cnn", "resnet", "vit"], 
                        help="Name of the model to evaluate")
    parser.add_argument("--num_images", type=int, default=4,
                        help="Number of test images to visualize")
    args = parser.parse_args()
    
    visualize_gradcam(args.model, args.num_images)