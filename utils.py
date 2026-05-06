import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report

def plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies, save_path):
    """
    Generates and saves line plots comparing training and validation metrics over epochs.
    The resulting figure contains two subplots displaying loss and accuracy trends respectively.
    """
    plt.figure(figsize=(10,4))
    
    # LOSS CURVES
    plt.subplot(1,2,1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.legend()
    plt.title("Loss over Epochs")

    # ACCURACY CURVES
    plt.subplot(1,2,2)
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.legend()
    plt.title("Accuracy over Epochs")
    
    plt.savefig(save_path)
    plt.close()


def plot_confusion_matrix(cm, class_names, test_acc, save_path, model_name):
    """
    Creates a heatmap visualization of the confusion matrix to evaluate classification performance across categories.
    The plot includes the total test accuracy in the title and saves the output to the specified path.
    """
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{model_name.upper()} Confusion Matrix (Test) - Accuracy: {test_acc*100:.2f}%")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_classification_report_image(all_labels, all_preds, class_names, save_path, model_name):
    """
    Creates a clean, styled table visualization of the classification report 
    with alternating row colors and a squarer aspect ratio.
    """
    # Extract report as a dictionary
    report_dict = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    
    cell_text = []
    for cls in class_names:
        prec = f"{report_dict[cls]['precision']:.4f}"
        rec = f"{report_dict[cls]['recall']:.4f}"
        f1 = f"{report_dict[cls]['f1-score']:.4f}"
        cell_text.append([cls, prec, rec, f1])

    # Setup the figure with a squarer dimension
    fig, ax = plt.subplots(figsize=(6, 6))  # Changed from (8, 4) to (6, 6)
    ax.axis('off')
    ax.axis('tight')

    # Create the table
    col_labels = ["Model", "Precision", "Recall", "F1-Score"]
    table = ax.table(cellText=cell_text,
                     colLabels=col_labels,
                     loc='center',
                     cellLoc='center')

    # General Styling
    table.scale(1, 3.5)  # Increased vertical scaling from 2.2 to 3.5 to make rows taller
    table.auto_set_font_size(False)
    table.set_fontsize(12)

    # Color Palette
    header_bg = '#08306B'     # Dark navy blue
    row_even_bg = '#E6F2FF'   # Very light baby blue
    row_odd_bg = '#FFFFFF'    # White
    edge_color = '#B0C4DE'    # Light steel blue borders

    # Apply specific styling cell by cell
    for (row, col), cell in table.get_celld().items():
        # Soften the borders
        cell.set_edgecolor(edge_color)
        
        if row == 0:
            # Header styling
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor(header_bg)
        else:
            # Alternating row colors
            bg_color = row_even_bg if row % 2 == 0 else row_odd_bg
            cell.set_facecolor(bg_color)
            cell.set_text_props(color='black')
            
            # First column (Model/Class names) styling
            if col == 0:
                cell.set_text_props(weight='bold')

    # Increased pad to 30 to give the taller table breathing room
    plt.title(f"{model_name.upper()} Classification Report", pad=0, fontsize=16, weight='bold', color='#08306B')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def get_normalize_params(transform):
    """
    Extracts the mean and standard deviation values from a the Torchvision transform pipeline.
    This is used to identify the specific normalization parameters applied during data preprocessing.
    """
    mean, std = None, None
    if hasattr(transform, "transforms"):
        for t in transform.transforms:
            if isinstance(t, transforms.Normalize):
                mean = t.mean
                std = t.std
                break
    return mean, std


def unnormalize_tensor(img_tensor, mean, std):
    """
    Reverses the normalization on a tensor to convert it back into a viewable image format.
    The function returns a clipped NumPy array scaled between 0 and 1 for proper display.
    """
    img = img_tensor.clone().cpu()
    for c in range(img.shape[0]):
        img[c] = img[c] * std[c] + mean[c]
    img = img.permute(1,2,0).numpy()
    img = np.clip(img, 0, 1)
    return img

def visualize_misclassified(all_preds, all_labels, test_data, class_names, save_path, num_to_show=8):
    """
    Identifies and plots a grid of images that the model failed to predict correctly.
    Each image is labeled with its true class and the model's incorrect prediction for diagnostic analysis.
    """
    mean, std = get_normalize_params(test_data.transform)
    if mean is None or std is None:
        mean, std = (0.5,), (0.5,)

    mis_idx = np.where(all_preds != all_labels)[0]
    num_to_show = min(len(mis_idx), num_to_show)

    if num_to_show == 0:
        print("No misclassified examples to show.")
        return

    plt.figure(figsize=(16, 4 * ((num_to_show + 3)//4)))
    for i in range(num_to_show):
        idx = mis_idx[i]
        img_tensor, true_label = test_data[idx]
        pred_label = all_preds[idx]
        img = unnormalize_tensor(img_tensor, mean, std)

        ax = plt.subplot((num_to_show + 3)//4, 4, i+1)
        
        # Handle grayscale plotting correctly
        if img.shape[2] == 1:
            ax.imshow(img.squeeze(), cmap='gray')
        else:
            ax.imshow(img)
            
        ax.set_title(f"True: {class_names[true_label]}\nPred: {class_names[int(pred_label)]}")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

class FocalLoss(nn.Module):
    """
    Implements Focal Loss to address class imbalance by down-weighting well-classified easy examples.
    It focuses training on hard-to-classify samples using a modulating factor governed by the gamma parameter.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        # alpha should be a 1D tensor of class weights
        self.alpha = alpha  
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 1. Compute standard Cross-Entropy Loss (unweighted, unreduced)
        # This calculates -log(p_t) for each sample in the batch.
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # 2. Extract p_t (the model's predicted probability for the TRUE class)
        # Since ce_loss = -log(p_t), we can get p_t via exp(-ce_loss)
        pt = torch.exp(-ce_loss)
        
        # 3. Calculate the Focal Loss modulating factor: (1 - p_t)^gamma
        # If p_t is high (easy example), this factor becomes tiny.
        # If p_t is low (hard example), this factor stays near 1.
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        # 4. Apply the dynamic class weights (alpha)
        if self.alpha is not None:
            # Gather the correct class weight for each sample in the batch
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = focal_loss * alpha_t
            
        # 5. Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss