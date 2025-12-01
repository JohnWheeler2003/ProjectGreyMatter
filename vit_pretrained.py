"""
Pretrained Vision Transformer fine-tune + evaluation script.

Usage:
    - Edit the dataset paths (TRAIN_DIR, VAL_DIR, TEST_DIR) if needed.
    - Run: python vit_pretrained.py

This script:
 - Loads ImageFolder datasets (same structure as your CNN script)
 - Uses a pretrained ViT (via timm) and adapts final head for `num_classes`
 - Fine-tunes the model (transfer learning)
 - Logs train/val loss & accuracy, saves best checkpoint
 - Evaluates on test set and produces metrics + confusion matrix + misclassified examples
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import timm  # pip install timm

# ---------------------------
# Configuration
# ---------------------------
TRAIN_DIR = "BrainTumorImages/Training"
VAL_DIR   = "BrainTumorImages/Validation"
TEST_DIR  = "BrainTumorImages/Testing"

IMG_SIZE = 224           # typical for pretrained ViT; you may experiment with 128 but 224 recommended
BATCH_SIZE = 16          # adjust to GPU memory; ViT can be memory-hungry
NUM_CLASSES = 4
NUM_EPOCHS = 10
LR = 3e-5                # small LR for fine-tuning
WEIGHT_DECAY = 1e-4
CHECKPOINT_PATH = "best_vit_pretrained.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ImageNet normalization (used by timm pretrained models)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# ---------------------------
# Data Transforms and Loaders
# ---------------------------
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

test_transform = val_transform

train_data = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
val_data   = datasets.ImageFolder(VAL_DIR, transform=val_transform)
test_data  = datasets.ImageFolder(TEST_DIR, transform=test_transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

print(f"Classes: {train_data.classes}")

# ---------------------------
# Model (pretrained ViT)
# ---------------------------
def create_pretrained_vit(model_name="vit_base_patch16_224", num_classes=NUM_CLASSES, pretrained=True):
    """
    Uses timm to create a pretrained ViT and replace the head for num_classes.
    model_name examples: 'vit_base_patch16_224', 'deit_base_distilled_patch16_224', etc.
    """
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    return model

model = create_pretrained_vit("vit_base_patch16_224", NUM_CLASSES, pretrained=True)
model = model.to(DEVICE)

# Print model size (params)
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Model: vit_base_patch16_224 (pretrained) -> trainable params: {count_params(model):,}")

# ---------------------------
# Loss, Optimizer, Scheduler
# ---------------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

# ---------------------------
# Training & Validation
# ---------------------------
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)   # logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc  = correct / total
    return epoch_loss, epoch_acc

def evaluate(model, loader, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, np.array(all_preds), np.array(all_labels)

# ---------------------------
# Training loop
# ---------------------------
best_val_acc = 0.0
train_losses, val_losses = [], []
train_accs, val_accs = [], []

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
    t0 = time.time()
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, DEVICE)
    val_loss, val_acc, _, _ = evaluate(model, val_loader, DEVICE)
    t1 = time.time()
    epoch_time = t1 - t0

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
    print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc*100:.2f}% | Time: {epoch_time:.1f}s")

    # Scheduler step (ReduceLROnPlateau expects metric)
    scheduler.step(val_acc)

    # Save best
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        checkpoint = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_accuracy": best_val_acc,
        }
        torch.save(checkpoint, CHECKPOINT_PATH)
        print(f"Saved new best checkpoint (val acc: {best_val_acc*100:.2f}%) to {CHECKPOINT_PATH}")

print("Training complete.")

# Plot training curves
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.legend(); plt.title("Loss")

plt.subplot(1,2,2)
plt.plot(train_accs, label="Train Acc")
plt.plot(val_accs, label="Val Acc")
plt.legend(); plt.title("Accuracy")
plt.savefig("vit_training_curves.png")
plt.close()
print("Saved vit_training_curves.png")

# ---------------------------
# Load best checkpoint (if exists)
# ---------------------------
if os.path.exists(CHECKPOINT_PATH):
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    state = ckpt.get("model_state", None)
    if state is not None:
        model.load_state_dict(state)
        print(f"Loaded best model from checkpoint (epoch {ckpt.get('epoch', '?')})")
else:
    print("No checkpoint found - using last model state.")

# ---------------------------
# Test set evaluation (metrics + confusion matrix)
# ---------------------------
model.eval()
all_preds = []
all_probs = []
all_labels = []

softmax = torch.nn.Softmax(dim=1)

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        probs = softmax(outputs)
        preds = torch.argmax(probs, dim=1)

        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)

test_acc = accuracy_score(all_labels, all_preds)
print(f"\nTest Accuracy: {test_acc*100:.2f}%")
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=test_data.classes, digits=4))

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=test_data.classes, yticklabels=test_data.classes)
plt.xlabel("Predicted"); plt.ylabel("True"); plt.title(f"Confusion Matrix - Test Acc: {test_acc*100:.2f}%")
plt.tight_layout()
plt.savefig("vit_confusion_matrix.png")
plt.close()
print("Saved vit_confusion_matrix.png")

# Per-class accuracy (recall)
per_class_acc = cm.diagonal() / cm.sum(axis=1)
for i, cls in enumerate(test_data.classes):
    print(f"Class '{cls}': {per_class_acc[i]*100:.2f}% ({int(cm[i,i])}/{int(cm.sum(axis=1)[i])})")

# ---------------------------
# Inference speed benchmark
# ---------------------------
def benchmark_inference(model, loader, device, num_batches=20):
    model.eval()
    torch.cuda.synchronize() if device.type == "cuda" else None
    t0 = time.time()
    iters = 0
    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            if i >= num_batches:
                break
            images = images.to(device)
            _ = model(images)
            iters += images.shape[0]
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()
    total_time = t1 - t0
    avg_time_per_image = total_time / iters
    return total_time, avg_time_per_image

total_time, avg_time_image = benchmark_inference(model, test_loader, DEVICE, num_batches=20)
print(f"\nInference benchmark: processed ~{min(20, len(test_loader))} batches in {total_time:.3f}s -> {avg_time_image*1000:.2f} ms/image")

# ---------------------------
# Misclassified examples visualization
# ---------------------------
# helper to unnormalize and show
def unnormalize(img_tensor):
    img = img_tensor.clone().cpu()
    for c in range(img.shape[0]):
        img[c] = img[c] * IMAGENET_STD[c] + IMAGENET_MEAN[c]
    img = img.permute(1,2,0).numpy()
    img = np.clip(img, 0, 1)
    return img

mis_idx = np.where(all_preds != all_labels)[0]
num_to_show = min(len(mis_idx), 12)
if num_to_show == 0:
    print("No misclassified examples.")
else:
    plt.figure(figsize=(16, 4 * ((num_to_show + 3)//4)))
    for i in range(num_to_show):
        idx = mis_idx[i]
        img_tensor, true_label = test_data[idx]
        pred_label = all_preds[idx]
        img = unnormalize(img_tensor)
        ax = plt.subplot((num_to_show + 3)//4, 4, i+1)
        ax.imshow(img)
        ax.set_title(f"True: {test_data.classes[true_label]}\nPred: {test_data.classes[pred_label]}")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("vit_misclassified_examples.png")
    plt.close()
    print(f"Saved {num_to_show} misclassified examples to vit_misclassified_examples.png")

# ---------------------------
# Final summary
# ---------------------------
print("\n--- Summary ---")
print(f"Test accuracy: {test_acc*100:.2f}%")
print(f"Trainable params: {count_params(model):,}")
print(f"Best validation accuracy saved: {best_val_acc*100:.2f}% (if checkpoint created)")

