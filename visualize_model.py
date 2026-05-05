import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

import config
from dataset import get_dataloaders
from model import BrainTumorCNN, PretrainedResNet, PretrainedViT
from utils import unnormalize_tensor, get_normalize_params

# Import custom logic
from cam_utils import AttentionRollout, vit_reshape_transform

def find_evaluation_cases(model, test_loader, class_names):
    successes = {name: None for name in class_names}
    failures = {name: None for name in class_names}
    
    model.eval()
    with torch.no_grad(): 
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
                
                if true_idx == pred_idx and successes[true_class] is None:
                    successes[true_class] = (images[i].cpu(), pred_class)
                elif true_idx != pred_idx and failures[true_class] is None:
                    failures[true_class] = (images[i].cpu(), pred_class)
            
            if all(v is not None for v in successes.values()) and all(v is not None for v in failures.values()):
                break
                
    return successes, failures

def plot_heatmap_grid(cases_dict, category_name, model_name, vis, mean, std):
    valid_cases = {k: v for k, v in cases_dict.items() if v is not None}
    missing_cases = [k for k, v in cases_dict.items() if v is None]
    
    num_images = len(class_names := list(cases_dict.keys()))
    
    fig, axes = plt.subplots(num_images, 3, figsize=(12, 4 * num_images))
    fig.suptitle(f"{model_name.upper()} - Visualized {category_name}", fontsize=16)
    
    if num_images == 1: axes = np.expand_dims(axes, axis=0)

    for i, true_class in enumerate(cases_dict.keys()):
        case = cases_dict[true_class]
        
        if case is None:
            for j in range(3): axes[i, j].axis('off')
            axes[i, 1].text(0.5, 0.5, f"No {category_name.lower()} found\nfor true class: {true_class}", 
                            ha='center', va='center', fontsize=12, color='gray')
            continue
            
        img_tensor, pred_class = case
        img_tensor = img_tensor.to(config.DEVICE)
        
        grayscale_heatmap = vis(input_tensor=img_tensor.unsqueeze(0), targets=None)[0, :]
        
        img_np = unnormalize_tensor(img_tensor, mean, std) 
        img_rgb = np.repeat(img_np, 3, axis=2) 
        vis_overlay = show_cam_on_image(img_rgb, grayscale_heatmap, use_rgb=True)

        axes[i, 0].imshow(img_rgb)
        axes[i, 0].set_title(f"True: {true_class}\nPred: {pred_class}")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(grayscale_heatmap, cmap='jet')
        axes[i, 1].set_title("Heatmap")
        axes[i, 1].axis('off')

        axes[i, 2].imshow(vis_overlay)
        axes[i, 2].set_title("Overlay")
        axes[i, 2].axis('off')

    plt.tight_layout()
    save_path = f"{model_name}_explanation_{category_name.lower()}.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Saved {category_name} visualizations to {save_path}")
    
    if missing_cases:
        print(f"  -> Note: No {category_name.lower()} found for classes: {', '.join(missing_cases)}")

def visualize_model(model_name):
    print(f"Running Explanation Pipeline for: {model_name}")

    _, _, test_loader, test_data = get_dataloaders(model_name)
    class_names = test_data.classes

    mean, std = get_normalize_params(test_data.transform)
    if mean is None or std is None:
        mean, std = config.MEAN, config.STD

    # Routing Logic
    if model_name == "custom_cnn":
        model = BrainTumorCNN().to(config.DEVICE)
        target_layers = [model.conv5]
        vis_class = GradCAM
        use_reshape = False

    elif model_name == "resnet":
        model = PretrainedResNet(grayscale=True).to(config.DEVICE)
        target_layers = [model.resnet.layer4[-1]] 
        vis_class = GradCAM
        use_reshape = False

    elif model_name == "vit":
        model = PretrainedViT(grayscale=True).to(config.DEVICE)
        target_layers = None 
        vis_class = AttentionRollout
        use_reshape = False
    else:
        raise ValueError("Invalid model name selected.")

    checkpoint_path = f"{model_name}_best_model.pth"
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=config.DEVICE, weights_only=True)
        model.load_state_dict(ckpt["model_state"])
        print(f"Loaded {model_name} model from epoch {ckpt['epoch']}")
    else:
        print(f"No checkpoint found at '{checkpoint_path}'. Train the model first.")
        return

    model.eval()

    vis = vis_class(
        model=model, 
        target_layers=target_layers, 
        reshape_transform=vit_reshape_transform if use_reshape else None
    )

    print("Hunting for evaluation cases in the test set...")
    successes, failures = find_evaluation_cases(model, test_loader, class_names)
    
    print("Generating Success Grids...")
    plot_heatmap_grid(successes, "Successes", model_name, vis, mean, std)
    
    print("Generating Failure Grids...")
    plot_heatmap_grid(failures, "Failures", model_name, vis, mean, std)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Explation Visualizations on Trained Models")
    parser.add_argument("--model", type=str, default="custom_cnn", 
                        choices=["custom_cnn", "resnet", "vit"], 
                        help="Name of the model to evaluate")
    args = parser.parse_args()

    visualize_model(args.model)