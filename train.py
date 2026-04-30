import argparse
import torch
import torch.nn as nn
import config
from dataset import get_dataloaders
from model import BrainTumorCNN, PretrainedResNet, PretrainedViT
from utils import plot_training_curves

def train_model(model_name):
    print(f"\nUsing device: {config.DEVICE}")
    print(f"Training Model: {model_name}")

    # LOAD DATA
    train_loader, val_loader, _, _ = get_dataloaders(model_name)

    # INITIALIZE MODEL
    if model_name == "custom_cnn":
        model = BrainTumorCNN().to(config.DEVICE)
    elif model_name == "resnet":
        model = PretrainedResNet(grayscale=True).to(config.DEVICE)
    elif model_name == "vit":
        model = PretrainedViT(grayscale=True).to(config.DEVICE)
    else:
        raise ValueError("Invalid model name selected.")
    
    # DYNAMIC CLASS WEIGHTS SETUP
    # 1. Extract labels directly from PyTorch dataset
    train_targets = train_loader.dataset.targets
    class_names = train_loader.dataset.classes

    # 2. Count occurrences of each class index using torch.bincount
    class_counts = torch.bincount(torch.tensor(train_targets)).tolist()
    total_samples = sum(class_counts)
    num_classes = len(class_counts)

    print("\nDynamic Class Distribution Found:")
    for name, count in zip(class_names, class_counts):
        print(f" - {name}: {count}")

    # 3. Calculate weights and push to device
    class_weights = [total_samples / (num_classes * count) for count in class_counts]
    weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(config.DEVICE)
    
    print(f"Dynamic Class Weights Applied: {class_weights}")

    # Pass the weights into the CrossEntropyLoss
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)

    # DYNAMIC LEARNING RATE SETUP
    adjusted_lr = 1e-4 if model_name in ["resnet", "vit"] else config.LEARNING_RATE
    optimizer = torch.optim.AdamW(model.parameters(), lr=adjusted_lr, weight_decay=0.01)
    
    # DYNAMIC SETUP BASED ON MODEL
    if model_name == "custom_cnn":
        max_epochs = 40 
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=adjusted_lr,
            steps_per_epoch=len(train_loader),
            epochs=max_epochs,
            pct_start=0.1,
            div_factor=50.0
        )
    else:
        max_epochs = 100 
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',      
            factor=0.1,      
            patience=3       
        )

    # TRACKING METRICS
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
    best_val_acc = 0.0
    best_val_loss= float('inf')

    # EARLY STOPPING VARIABLES
    patience = 7 # How many epochs to wait before giving up
    patience_counter = 0

    for epoch in range(max_epochs):
        print(f"\nEpoch {epoch+1}/{max_epochs}")
        print("-" * 30)

        # TRAINING PHASE
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
            optimizer.step()

            # ONLY OneCycleLR steps per batch
            if model_name == "custom_cnn":
                scheduler.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = correct / total
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)

        print(f"Training Loss: {epoch_train_loss:.4f} | Accuracy: {epoch_train_acc*100:.2f}%")

        # VALIDATION PHASE
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = val_correct / val_total
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_acc)

        print(f"Validation Loss: {epoch_val_loss:.4f} | Accuracy: {epoch_val_acc*100:.2f}%")
        
        # ONLY ReduceLROnPlateau steps per epoch, using val_loss
        if model_name in ["resnet", "vit"]:
            scheduler.step(epoch_val_loss)

        # Check current learning rate (optional, helps see when it drops)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current Learning Rate: {current_lr}")


        # 1. SAVE BEST MODEL (TRACKS ACCURACY)
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            
            # Create a dynamic save path so models don't overwrite each other
            save_path = f"{model_name}_best_model.pth" 
            
            torch.save({
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_accuracy": best_val_acc,
            }, save_path)
            print(f"--> New best model saved as {save_path}! (Accuracy: {best_val_acc*100:.2f}%)")


        # 2. EARLY STOPPING (TRACKS LOSS)
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0  # Reset counter because loss went down!
        else:
            patience_counter += 1
            print(f"No improvement in validation loss. Early stopping counter: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print(f"\nEarly stopping triggered! Training halted early at epoch {epoch+1}.")
                break

    print("\nTraining complete!")
    
    # PLOT RESULTS
    plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies)
    print("Saved training curves to training_validation_curves.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Brain Tumor Classification Models")
    parser.add_argument("--model", type=str, default="custom_cnn", 
                        choices=["custom_cnn", "resnet", "vit"], 
                        help="Name of the model to train")
    args = parser.parse_args()
    
    train_model(args.model)