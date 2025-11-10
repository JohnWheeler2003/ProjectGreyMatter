import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Check device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print ("Using device:", device)

# 1. Define paths
train_dir = "BrainTumorImages/Training"
val_dir = "BrainTumorImages/Validation"
test_dir = "BrainTumorImages/Testing"

# 2. Define image transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 3. Load datasets using ImageFolder
train_data = datasets.ImageFolder(root=train_dir, transform=transform)
val_data = datasets.ImageFolder(root=val_dir, transform=transform)  
test_data = datasets.ImageFolder(root=test_dir, transform=transform )

# 4. Create Dataloaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

'''
# 5. Visualization
def show_images(dataset):
    fig, axes = plt.subplots(1, 4, figsize=(10, 4))
    for i in range(4):
        image, label = dataset[i]
        image = image.permute(1,2,0) #Converts from CxHxW to HxWxC
        axes[i].imshow(image * 0.5 + 0.5) # Undo normalization for visualization
        axes[i].set_title(dataset.classes[label])
        axes[i].axis("off")
    plt.savefig("sample_images.png")

def show_images_one_per_class(dataset):
    seen_classes = set()
    fig, axes = plt.subplots(1, len(dataset.classes), figsize=(12,4))

    i = 0
    for img, label in dataset:
        class_name = dataset.classes[label]
        if class_name not in seen_classes:
            seen_classes.add(class_name)
            img = img.permute(1,2,0) # Convert from CxHxW to HxWxC
            axes[i].imshow(img * 0.5 + 0.5) # Undo normalization for visualization
            axes[i].set_title(class_name)
            axes[i].axis("off")
            i += 1
        if len(seen_classes) == len(dataset.classes):
            break
    plt.tight_layout()
    plt.savefig("sample_images_one_per_class.png")

show_images(train_data)
show_images_one_per_class(train_data)


# 6. Test the Dataloader and count images in folder
data_iter = iter(train_loader)
images, labels = next(data_iter)

print(f"Batch image tensor shape: {images.shape}")
print(f"Batch label tensor shape: {labels.shape}")
print(f"First 5 labels: {labels[:5].tolist()}")

def count_images_in_folders(base_path):
    print(f"\nImage counts for: {base_path}")
    total = 0
    for folder in sorted(os.listdir(base_path)):
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path):
            count = len(os.listdir(folder_path))
            total += count
            print(f"  {folder}: {count} images")
    print(f"Total in {os.path.basename(base_path)}: {total} images")

# Verify all sets
count_images_in_folders(train_dir)
count_images_in_folders(val_dir)
count_images_in_folders(test_dir)

'''

# 7. Define a simple CNN model
class BrainTumorCNN(nn.Module):
    def __init__(self):
        super(BrainTumorCNN, self).__init__()

        # Convolutional Block 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Convolutional Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional Block 4
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional Block 5
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully Connected Layers
        self.fc1 = nn.Linear(512 * 4 * 4, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 4)

    def forward(self, x):
        # Each conv block: Conv -> BN -> ReLU -> Pool
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = self.pool5(F.relu(self.bn5(self.conv5(x))))

        # Flatten before FC layers
        x= x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
model = BrainTumorCNN().to(device)
print(model)

'''
# Test one batch
data_iter = iter(train_loader)
images, labels = next(data_iter)
images = images.to(device)

outputs = model(images)
print("Input batch shape:", images.shape)
print("Output batch shape:", outputs.shape)
'''

# 8 . Define loss function and optimizer
criterion = nn.CrossEntropyLoss() # measures how wrong the model's predictions are

optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # updates the models weights to minimize loss

print("\nLoss function and optimizer defined successfully.")


# 9. Training and Validation Loop
num_epochs = 10 # Number of full passes through the dataset

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

best_val_acc = 0.0
checkpoint_path = "best_brain_tumor_cnn.pth"

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    print("-" * 30)

    # Training Phase
    model.train() # Sets model to training mode
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad() # reset gradients before each batch

        outputs = model(images) # Forward pass
        loss = criterion(outputs, labels) # Compute loss

        loss.backward() # Backpropagate to compute gradients
        optimizer.step() # Update model parameters

        # Accumulate metrics
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    epoch_train_loss = running_loss / len(train_data)
    epoch_train_acc = correct / total
    train_losses.append(epoch_train_loss)
    train_accuracies.append(epoch_train_acc)

    print(f"Training Loss: {epoch_train_loss:.4f} | Accuracy: {epoch_train_acc*100:.2f}%")

    # Validation Phase
    model.eval() # Set model to evaluation mode
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad(): # No gradient calculation in validation
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

    epoch_val_loss = val_loss / len(val_data)
    epoch_val_acc = val_correct / val_total
    val_losses.append(epoch_val_loss)
    val_accuracies.append(epoch_val_acc)

    print(f"Validation Loss: {epoch_val_loss:.4f} | Accuracy: {epoch_val_acc*100:.2f}%")

    # Save the best model
    if epoch_val_acc > best_val_acc:
        best_val_acc = epoch_val_acc
        checkpoint = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_accuracy": best_val_acc,
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"New best model saved with Validation Accuracy: {best_val_acc*100:.2f}%")

print("\nTraining complete!")

# 10. Plot training curves
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
plt.savefig("training_validation_curves.png")

# 10. Testing and Evaluation

# Configuration
checkpoint_path = "best_brain_tumor_cnn.pth" # Change if your saved checkpoint has a different name
cm_plot_path = "confusion_matrix.png"
misclassified_plot = "misclassified_examples.png"
num_misclassified_to_show = 8 # how many misclassified images to save/show

# 10.1: Load the best checkpoint if it exists
if os.path.exists(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location=device)
    # support different key names used in different saving snippets
    state = ckpt.get("model_state", ckpt.get("model_state_dict", None))
    if state is not None:
        model.load_state_dict(state)
        print(f"Loaded model state from checkpoint: {checkpoint_path} (epoch {ckpt.get('epoch', '?')})")
    else:
        print(f"Checkpoint found but model state key not recognized. Using current model weights.")
else:
    print(f"No checkpoint found at '{checkpoint_path}'. Using current model weights (no checkpoint).")

model.to(device)
model.eval() # ensure eval model (disables dropout, fixes batchnorm)

#10.2 Run inference on test set and collect predictions + probabilities
all_preds = []
all_probs = []
all_labels = []

softmax = torch.nn.Softmax(dim=1)

with torch.no_grad(): # no gradients needed for inference
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images) # logits, shape [B, num_classes]
        probs = softmax(outputs) # probabilities, shape [B, num_classes]
        preds = torch.argmax(probs, dim=1)

        all_probs.extend(probs.cpu().numpy()) # store probabilities if needed later
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)

#10.3 Overall Accuracy
test_acc = accuracy_score(all_labels, all_preds)
print(f"\nTest Accuracy: {test_acc*100:.2f}%")

#10.4 Confusion Matrix and classification report (precision/recall/F1)
class_names = test_data.classes
cm = confusion_matrix(all_labels, all_preds)
print("\nClassification Report (per-class precision / recall / F1 score):\n")
print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

#10.5 Plot and save confusion matrix
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(f"Confusion Matrix (Test) - Accuracy: {test_acc*100:.2f}%")
plt.tight_layout()
plt.savefig(cm_plot_path)
plt.close()
print(f"Saved confusion matrix to {cm_plot_path}")

#10.6 Per-class accuracy (true-class recall)
per_class_acc = cm.diagonal() / cm.sum(axis=1)
for idx, cls in enumerate(class_names):
    print(f"Class '{cls}': {per_class_acc[idx]*100:.2f}% ({int(cm[idx,idx])}/{int(cm.sum(axis=1)[idx])})")

# 10.7 Visualize misclassified examples (helps qualitative debugging)
#Helper: extract mean/std from transform (works of Normalize exists; handles single-channel too)
def get_normalize_params(transform):
    mean = None
    std = None
    if hasattr(transform, "transforms"):
        for t in transform.transforms:
            if isinstance(t, transforms.Normalize):
                mean = t.mean
                std = t.std
                break
    return mean, std

mean, std = get_normalize_params(test_data.transform)
if mean is None or std is None:
    # fallback if Normalize not present in transform, assume 0.5/0.5
    mean = (0.5,)
    std = (0.5,)
    print("Warning: Normalize() not found in test transform - falling back to mean=(0.5,) std=(0.5,)")

# Broadcast single-channel mean/std to 3 channels if needed
def broadcast_mean_std(mean, std):
    if len(mean) == 1:
        mean3 = tuple([float(mean[0])] * 3)
        std3 = tuple([float(std[0])] * 3)
        return mean3, std3
    else:
        return tuple([float(m) for m in mean]), tuple([float(s) for s in std])

mean3, std3 = broadcast_mean_std(mean, std)

def unnormalize_tensor(img_tensor, mean, std):
    """Given a tensor in C,H,W normalized by (mean,std), return H,W,C np array in [0,1]."""
    img = img_tensor.clone().cpu()
    for c in range(img.shape[0]):
        img[c] = img[c] * std[c] + mean[c]
    img = img.permute(1,2,0).numpy()
    img = np.clip(img, 0, 1)
    return img

mis_idx = np.where(all_preds != all_labels)[0]
num_to_show = min(len(mis_idx), num_misclassified_to_show)

if num_to_show == 0:
    print("No misclassified examples to show (model predicted all test samples correctly).")
else:
    plt.figure(figsize=(16, 4 * ((num_to_show + 3)//4)))
    for i in range(num_to_show):
        idx = mis_idx[i]
        # Because test loader was created with shuffle=False, indexing test_data directly corresponds
        img_tensor, true_label = test_data[idx] # returns normalized tensor
        pred_label = all_preds[idx]
        img = unnormalize_tensor(img_tensor, mean3, std3)

        ax = plt.subplot((num_to_show + 3)//4, 4, i+1)
        ax.imshow(img)
        ax.set_title(f"True: {class_names[true_label]}\nPred: {class_names[int(pred_label)]}")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(misclassified_plot)
    plt.close()
    print(f"Saved {num_to_show} misclassified examples to {misclassified_plot}")