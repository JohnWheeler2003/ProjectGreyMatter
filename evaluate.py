import argparse
import os
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import config
from dataset import get_dataloaders
from model import BrainTumorCNN, PretrainedResNet, PretrainedViT
from utils import plot_confusion_matrix, visualize_misclassified

def evaluate_model(model_name):
    print(f"Evaluation Model: {model_name}")
    
    # LOAD DATA
    _, _, test_loader, test_data = get_dataloaders(model_name)
    class_names = test_data.classes

    # INITIALIZE MODEL
    if model_name == "custom_cnn":
        model = BrainTumorCNN().to(config.DEVICE)
    elif model_name == "resnet":
        model = PretrainedResNet(grayscale=True).to(config.DEVICE)
    elif model_name == "vit":
        model = PretrainedViT(grayscale=True).to(config.DEVICE)
    else:
        raise ValueError("Invalid model name selected.")

    # DYNAMICALLY MATCH THE SAVE PATH FROM TRAIN.PY
    checkpoint_path = f"{model_name}_best_model.pth"

    # LOAD CHECKPOINT
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=config.DEVICE, weights_only=True)
        model.load_state_dict(ckpt["model_state"])
        print(f"Loaded model from epoch {ckpt['epoch']}")
    else:
        print(f"No checkpoint found at '{checkpoint_path}'. Exiting.")
        return

    model.eval()
    
    all_preds, all_labels = [], []
    softmax = torch.nn.Softmax(dim=1)

    print("\nRunning Inference on Test Set...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(config.DEVICE)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # METRICS
    test_acc = accuracy_score(all_labels, all_preds)
    print(f"\nTest Accuracy: {test_acc*100:.2f}%")
    print("\nClassification Report:\n")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))


    # CONFUSION MATRIX
    cm_path = f"{model_name}_confusion_matrix.png"
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, class_names, test_acc, save_path=cm_path)
    print(f"Saved confusion matrix to {cm_path}")

    # Terminal Output of Confusion Matrix
    print("\n" + "="*50)
    print(f" {model_name.upper()} RAW CONFUSION MATRIX")
    print("="*50)
    print(f"{'':>12} | {'Pred Glioma':>11} | {'Pred Mening':>11} | {'Pred NoTumor':>12} | {'Pred Pituit':>11}")
    print("-" * 65)

    for i, true_name in enumerate(class_names):
        print(f"True {true_name[:7]:<7} | {cm[i][0]:>11} | {cm[i][1]:>11} | {cm[i][2]:>12} | {cm[i][3]:>11}")
    print("="*50 + "\n")

    # MISCLASSIFIED VISUALIZATIONS
    misclassified_path = f"{model_name}_misclassified_examples.png"
    visualize_misclassified(all_preds, all_labels, test_data, class_names, save_path=misclassified_path)
    print(f"Saved misclassified examples to {misclassified_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Brain Tumor Classification Models")
    parser.add_argument("--model", type=str, default="custom_cnn", 
                        choices=["custom_cnn", "resnet", "vit"], 
                        help="Name of the model to evaluate")
    args = parser.parse_args()
    
    evaluate_model(args.model)