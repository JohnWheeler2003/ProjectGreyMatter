import torch
import torch.nn as nn
import config
from dataset import get_dataloaders
from model import BrainTumorCNN
from utils import plot_training_curves

def train_model():
    print(f"\nUsing device: {config.DEVICE}")

    # Load Data
    train_loader, val_loader, _, _ = get_dataloaders()

    # Initialize Model, Loss, Optimizer
    model = BrainTumorCNN().to(config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    
    # FIX 1: Changed Adam to AdamW
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=0.01)
    
    # FIX 2: Changed ReduceLROnPlateau to StepLR
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Tracking metrics
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
    best_val_acc = 0.0

    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
        print("-" * 30)

        # --- Training Phase ---
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = correct / total
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)

        print(f"Training Loss: {epoch_train_loss:.4f} | Accuracy: {epoch_train_acc*100:.2f}%")

        # --- Validation Phase ---
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
        
        # Step the scheduler (perfectly done here)
        scheduler.step()
        
        # Check current learning rate (optional, helps see when it drops)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current Learning Rate: {current_lr}")

        # Save Best Model
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save({
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_accuracy": best_val_acc,
            }, config.CHECKPOINT_PATH)
            print(f"--> New best model saved! (Accuracy: {best_val_acc*100:.2f}%)")

    print("\nTraining complete!")
    
    # Plot results
    plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies)
    print("Saved training curves to training_validation_curves.png")

if __name__ == "__main__":
    train_model()