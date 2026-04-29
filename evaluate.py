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
    
    # Load Data
    _, _, test_loader, test_data = get_dataloaders()
    class_names = test_data.classes

    # Initialize Model
    if model_name == "custom_cnn":
        model = BrainTumorCNN().to(config.DEVICE)
    elif model_name == "resnet":
        model = PretrainedResNet(grayscale=True).to(config.DEVICE)
    elif model_name == "vit":
        model = PretrainedViT(grayscale=True).to(config.DEVICE)
    else:
        raise ValueError("Invalid model name selected.")

    # Dynamically match the save path from train.py
    checkpoint_path = f"{model_name}_best_model.pth"

    # Load Checkpoint
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
            probs = softmax(outputs)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Metrics
    test_acc = accuracy_score(all_labels, all_preds)
    print(f"\nTest Accuracy: {test_acc*100:.2f}%")
    print("\nClassification Report:\n")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, class_names, test_acc)
    print("Saved confusion matrix to confusion_matrix.png")

    # Misclassified Visualizations
    visualize_misclassified(all_preds, all_labels, test_data, class_names)
    print("Saved misclassified examples to misclassified_examples.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Brain Tumor Classification Models")
    parser.add_argument("--model", type=str, default="custom_cnn", 
                        choices=["custom_cnn", "resnet", "vit"], 
                        help="Name of the model to evaluate")
    args = parser.parse_args()
    
    evaluate_model(args.model)