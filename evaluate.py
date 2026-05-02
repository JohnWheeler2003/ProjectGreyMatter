import argparse
import os
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import config
from dataset import get_dataloaders
from model import BrainTumorCNN, PretrainedResNet, PretrainedViT
from utils import plot_confusion_matrix, visualize_misclassified

def evaluate_model(model_name, vit_weight=0.5):
    print(f"Evaluation Model: {model_name.upper()}")
    
    softmax = torch.nn.Softmax(dim=1)


    # ENSEMBLE MODE INITIALIZATION

    if model_name == "ensemble":
        # Load BOTH dataloaders because ResNet needs 256x256 and ViT needs 224x224
        print("Loading Datasets...")
        _, _, test_loader_resnet, test_data = get_dataloaders("resnet")
        _, _, test_loader_vit, _ = get_dataloaders("vit")
        class_names = test_data.classes

        print("Initializing Ensemble Models...")
        model_resnet = PretrainedResNet(grayscale=True).to(config.DEVICE)
        model_vit = PretrainedViT(grayscale=True).to(config.DEVICE)

        # Load ResNet Checkpoint
        if os.path.exists("resnet_best_model.pth"):
            ckpt_rn = torch.load("resnet_best_model.pth", map_location=config.DEVICE, weights_only=True)
            model_resnet.load_state_dict(ckpt_rn["model_state"])
            print(f"Loaded ResNet from epoch {ckpt_rn['epoch']}")
        else:
            print("Missing resnet_best_model.pth! Exiting.")
            return

        # Load ViT Checkpoint
        if os.path.exists("vit_best_model.pth"):
            ckpt_vit = torch.load("vit_best_model.pth", map_location=config.DEVICE, weights_only=True)
            model_vit.load_state_dict(ckpt_vit["model_state"])
            print(f"Loaded ViT from epoch {ckpt_vit['epoch']}")
        else:
            print("Missing vit_best_model.pth! Exiting.")
            return

        model_resnet.eval()
        model_vit.eval()


    # SINGLE MODEL INITIALIZATION

    else:
        _, _, test_loader, test_data = get_dataloaders(model_name)
        class_names = test_data.classes

        if model_name == "custom_cnn":
            model = BrainTumorCNN().to(config.DEVICE)
        elif model_name == "resnet":
            model = PretrainedResNet(grayscale=True).to(config.DEVICE)
        elif model_name == "vit":
            model = PretrainedViT(grayscale=True).to(config.DEVICE)
        else:
            raise ValueError("Invalid model name selected.")

        checkpoint_path = f"{model_name}_best_model.pth"
        if os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=config.DEVICE, weights_only=True)
            model.load_state_dict(ckpt["model_state"])
            print(f"Loaded model from epoch {ckpt['epoch']}")
        else:
            print(f"No checkpoint found at '{checkpoint_path}'. Exiting.")
            return

        model.eval()


    # INFERENCE LOOP

    all_preds, all_labels = [], []
    print("\nRunning Inference on Test Set...")
    
    with torch.no_grad():
        if model_name == "ensemble":
            # Zip the loaders to iterate through 256x256 and 224x224 images simultaneously
            for (images_rn, labels), (images_vit, _) in zip(test_loader_resnet, test_loader_vit):
                images_rn = images_rn.to(config.DEVICE)
                images_vit = images_vit.to(config.DEVICE)
                labels = labels.to(config.DEVICE)
                
                # Get Raw Outputs and Apply Softmax
                out_resnet = model_resnet(images_rn)
                out_vit = model_vit(images_vit)
                
                prob_resnet = softmax(out_resnet)
                prob_vit = softmax(out_vit)
                
                # The Math (Weighted Average)
                resnet_weight = 1.0 - vit_weight
                ensemble_probs = (prob_resnet * resnet_weight) + (prob_vit * vit_weight)
                
                # Final Prediction
                preds = torch.argmax(ensemble_probs, dim=1)
                
                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())
        else:
            for images, labels in test_loader:
                images = images.to(config.DEVICE)
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)


    # EVALUATION PIPELINE

    test_acc = accuracy_score(all_labels, all_preds)
    
    if model_name == "ensemble":
        resnet_weight = 1.0 - vit_weight
        print(f"\nEnsemble Blend: ResNet ({resnet_weight*100:.0f}%) + ViT ({vit_weight*100:.0f}%)")
        
    print(f"\nTest Accuracy: {test_acc*100:.2f}%")
    print("\nClassification Report:\n")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

    # CONFUSION MATRIX
    cm_path = f"{model_name}_confusion_matrix.png"
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, class_names, test_acc, save_path=cm_path)
    print(f"Saved confusion matrix to {cm_path}")

    # TERMINAL OUTPUT
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
                        choices=["custom_cnn", "resnet", "vit", "ensemble"], 
                        help="Name of the model to evaluate")
    parser.add_argument("--vit_weight", type=float, default=0.5,
                        help="Weight given to ViT in the ensemble (e.g., 0.6 means ViT is 60%, ResNet is 40%). Default 0.5")
    args = parser.parse_args()
    
    evaluate_model(args.model, args.vit_weight)