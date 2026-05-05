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
import cv2
import torch.nn.functional as F

# Import cleanly separated custom logic
from cam_utils import AttentionRollout

def get_evaluation_cases(test_loader, class_names, resnet, vit):
    """
    Hunts through the test_loader to categorize images based on ResNet and ViT predictions.
    """
    # Initialize 4 dictionaries to store: class_name -> (image_tensor, resnet_pred, vit_pred)
    both_correct = {name: None for name in class_names}
    resnet_only = {name: None for name in class_names}
    vit_only = {name: None for name in class_names}
    both_wrong = {name: None for name in class_names}
    
    with torch.no_grad():
        for images, labels in test_loader:
            images_dev = images.to(config.DEVICE)
            
            # 1. Get ResNet Predictions
            resnet_outputs = resnet(images_dev)
            resnet_preds = torch.argmax(resnet_outputs, dim=1)
            
            # 2. Get ViT Predictions (requires resizing to 224x224)
            vit_input = F.interpolate(images_dev, size=(224, 224), mode='bilinear', align_corners=False)
            vit_outputs = vit(vit_input)
            vit_preds = torch.argmax(vit_outputs, dim=1)
            
            for i in range(len(labels)):
                true_idx = labels[i].item()
                r_idx = resnet_preds[i].item()
                v_idx = vit_preds[i].item()
                
                true_class = class_names[true_idx]
                r_class = class_names[r_idx]
                v_class = class_names[v_idx]
                
                r_correct = (r_idx == true_idx)
                v_correct = (v_idx == true_idx)
                
                # Bundle the data so we can display the predictions on the plot
                data_tuple = (images[i].cpu(), r_class, v_class)
                
                # Sort into categories
                if r_correct and v_correct and both_correct[true_class] is None:
                    both_correct[true_class] = data_tuple
                elif r_correct and not v_correct and resnet_only[true_class] is None:
                    resnet_only[true_class] = data_tuple
                elif not r_correct and v_correct and vit_only[true_class] is None:
                    vit_only[true_class] = data_tuple
                elif not r_correct and not v_correct and both_wrong[true_class] is None:
                    both_wrong[true_class] = data_tuple
                    
            # Early stopping check: Stop if all 4 categories have an image for every class
            if (all(v is not None for v in both_correct.values()) and 
                all(v is not None for v in resnet_only.values()) and 
                all(v is not None for v in vit_only.values()) and 
                all(v is not None for v in both_wrong.values())):
                break
                
    return both_correct, resnet_only, vit_only, both_wrong

def save_raw_augmented_images(cases_dicts, class_names, mean, std):
    """
    Finds one image per class from the collected dictionaries and saves it as a standalone PNG.
    This shows the image post-augmentation (CLAHE, rotations, etc.), unnormalized for human viewing.
    """
    print("Saving augmented image samples...")
    for cls in class_names:
        img_tensor = None
        
        # Look through our 4 dictionaries until we find an image for this class
        for d in cases_dicts:
            if d[cls] is not None:
                img_tensor = d[cls][0] # Grab the tensor from the tuple
                break
        
        if img_tensor is not None:
            # Reverses ONLY the statistical normalization (Step 4), keeping CLAHE and transforms
            img_np = unnormalize_tensor(img_tensor, mean, std) 
            
            # Replicate the grayscale channel to RGB for saving
            img_rgb = np.repeat(img_np, 3, axis=2) 
            
            # Save the numpy array directly to disk
            save_path = f"augmented_sample_{cls}.png"
            plt.imsave(save_path, img_rgb)
            
            print(f"  -> Saved {save_path} at resolution {img_rgb.shape[1]}x{img_rgb.shape[0]}")

def plot_side_by_side_comparison(cases_dict, category_name, resnet_cam, vit_attnroll, mean, std):
    """
    Generates a visual grid for a specific outcome category.
    """
    class_names = list(cases_dict.keys())
    num_images = len(class_names)
    
    fig, axes = plt.subplots(num_images, 5, figsize=(20, 4 * num_images))
    fig.suptitle(f"Visualization Comparison: {category_name}", fontsize=18)
    
    if num_images == 1: axes = np.expand_dims(axes, axis=0)

    for i, true_class in enumerate(class_names):
        case = cases_dict[true_class]
        
        # If no image was found for this specific outcome combination
        if case is None:
            for j in range(5):
                axes[i, j].axis('off')
            axes[i, 2].text(0.5, 0.5, f"No image exists where\n{category_name}\nfor true class: {true_class}", 
                            ha='center', va='center', fontsize=14, color='gray')
            continue
            
        img_tensor, r_class, v_class = case
        
        img_tensor = img_tensor.to(config.DEVICE)
        input_tensor = img_tensor.unsqueeze(0) 
        
        # Generate Visualizations
        resnet_grayscale = resnet_cam(input_tensor=input_tensor, targets=None)[0, :]
        
        vit_input = F.interpolate(input_tensor, size=(224, 224), mode='bilinear', align_corners=False)
        vit_grayscale_224 = vit_attnroll(input_tensor=vit_input, targets=None)[0, :]
        vit_grayscale = cv2.resize(vit_grayscale_224, (img_tensor.shape[2], img_tensor.shape[1]))
        
        # Prepare original image
        img_np = unnormalize_tensor(img_tensor, mean, std) 
        img_rgb = np.repeat(img_np, 3, axis=2) 
        
        resnet_overlay = show_cam_on_image(img_rgb, resnet_grayscale, use_rgb=True)
        vit_overlay = show_cam_on_image(img_rgb, vit_grayscale, use_rgb=True)

        # Plot Original
        axes[i, 0].imshow(img_rgb)
        axes[i, 0].set_title(f"True Class: {true_class}")
        axes[i, 0].axis('off')

        # Plot ResNet 
        axes[i, 1].imshow(resnet_grayscale, cmap='jet')
        axes[i, 1].set_title(f"ResNet Pred: {r_class}\nHeatmap")
        axes[i, 1].axis('off')

        axes[i, 2].imshow(resnet_overlay)
        axes[i, 2].set_title("ResNet Overlay")
        axes[i, 2].axis('off')

        # Plot ViT
        axes[i, 3].imshow(vit_grayscale, cmap='jet')
        axes[i, 3].set_title(f"ViT Pred: {v_class}\nHeatmap")
        axes[i, 3].axis('off')

        axes[i, 4].imshow(vit_overlay)
        axes[i, 4].set_title("ViT Overlay")
        axes[i, 4].axis('off')

    plt.tight_layout()
    filename = category_name.lower().replace(" ", "_").replace(",", "")
    save_path = f"comparison_{filename}.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Saved {category_name} visualization to {save_path}")

def visualize_comparison():
    print("Running Visualization Comparison: ResNet vs ViT")

    # 1. LOAD DATA 
    _, _, test_loader, test_data = get_dataloaders("resnet") 
    class_names = test_data.classes

    mean, std = get_normalize_params(test_data.transform)
    if mean is None or std is None:
        mean, std = config.MEAN, config.STD

    # 2. INITIALIZE RESNET
    print("Loading ResNet...")
    resnet = PretrainedResNet(grayscale=True).to(config.DEVICE)
    resnet_ckpt_path = "resnet_best_model.pth"
    if os.path.exists(resnet_ckpt_path):
        resnet_ckpt = torch.load(resnet_ckpt_path, map_location=config.DEVICE, weights_only=True)
        resnet.load_state_dict(resnet_ckpt["model_state"])
    else:
        print(f"Missing {resnet_ckpt_path}")
        return
    resnet.eval()
    resnet_cam = GradCAM(model=resnet, target_layers=[resnet.resnet.layer4[-1]])

    # 3. INITIALIZE ViT
    print("Loading ViT...")
    vit = PretrainedViT(grayscale=True).to(config.DEVICE)
    vit_ckpt_path = "vit_best_model.pth"
    if os.path.exists(vit_ckpt_path):
        vit_ckpt = torch.load(vit_ckpt_path, map_location=config.DEVICE, weights_only=True)
        vit.load_state_dict(vit_ckpt["model_state"])
    else:
        print(f"Missing {vit_ckpt_path}")
        return
    vit.eval()
    
    # Utilizing Attention Rollout class
    vit_attnroll = AttentionRollout(model=vit)

    # 4. HUNT FOR CASES
    print("Hunting for evaluation cases across the test set...")
    both_correct, resnet_only, vit_only, both_wrong = get_evaluation_cases(test_loader, class_names, resnet, vit)
    
    # 5. SAVE RAW AUGMENTED IMAGES
    all_dicts = [both_correct, resnet_only, vit_only, both_wrong]
    save_raw_augmented_images(all_dicts, class_names, mean, std)
    
    # 6. GENERATE THE 4 GRIDS
    print("\nGenerating Comparison Grids...")
    plot_side_by_side_comparison(both_correct, "Both Correct", resnet_cam, vit_attnroll, mean, std)
    plot_side_by_side_comparison(resnet_only, "ResNet Correct, ViT Wrong", resnet_cam, vit_attnroll, mean, std)
    plot_side_by_side_comparison(vit_only, "ViT Correct, ResNet Wrong", resnet_cam, vit_attnroll, mean, std)
    plot_side_by_side_comparison(both_wrong, "Both Wrong", resnet_cam, vit_attnroll, mean, std)

if __name__ == "__main__":
    visualize_comparison()