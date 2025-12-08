import os
import random
import shutil

# Paths
root_dir = "BrainTumorImages"
train_dir = os.path.join(root_dir, "Training")
val_dir = os.path.join(root_dir, "Validation")

# Create Validation/ and subfolders if they don’t exist
os.makedirs(val_dir, exist_ok=True)

# Loop over each class folder
for class_name in os.listdir(train_dir):
    class_train_dir = os.path.join(train_dir, class_name)
    class_val_dir = os.path.join(val_dir, class_name)

    # Skip anything that's not a directory
    if not os.path.isdir(class_train_dir):
        continue

    # Create class folder in Validation/
    os.makedirs(class_val_dir, exist_ok=True)

    # Get all images in this class
    images = os.listdir(class_train_dir)
    random.shuffle(images)

    # Compute how many to move (20%)
    val_count = int(0.2 * len(images))
    val_images = images[:val_count]

    # Move them
    for img in val_images:
        src_path = os.path.join(class_train_dir, img)
        dst_path = os.path.join(class_val_dir, img)
        shutil.move(src_path, dst_path)

    print(f"Moved {val_count} images from {class_name} → Validation/{class_name}")

print("\n Validation set created successfully!")
