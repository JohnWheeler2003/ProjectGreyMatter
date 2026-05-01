import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from torchvision import transforms

def plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies, save_path):
    plt.figure(figsize=(10,4))
    
    plt.subplot(1,2,1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.legend()
    plt.title("Loss over Epochs")

    plt.subplot(1,2,2)
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.legend()
    plt.title("Accuracy over Epochs")
    
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(cm, class_names, test_acc, save_path):
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix (Test) - Accuracy: {test_acc*100:.2f}%")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def get_normalize_params(transform):
    mean, std = None, None
    if hasattr(transform, "transforms"):
        for t in transform.transforms:
            if isinstance(t, transforms.Normalize):
                mean = t.mean
                std = t.std
                break
    return mean, std

def unnormalize_tensor(img_tensor, mean, std):
    """Given a tensor in C,H,W normalized by (mean,std), return H,W,C np array in [0,1]."""
    img = img_tensor.clone().cpu()
    for c in range(img.shape[0]):
        img[c] = img[c] * std[c] + mean[c]
    img = img.permute(1,2,0).numpy()
    img = np.clip(img, 0, 1)
    return img

def visualize_misclassified(all_preds, all_labels, test_data, class_names, save_path, num_to_show=8):
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