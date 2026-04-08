import os
import shutil
import random
import kagglehub
from pathlib import Path

# --- Configuration ---
VAL_SPLIT = 0.15  # 15% of the original Kaggle training data will go to validation
SEED = 42         # For reproducible dataset shuffling
CLASSES = ['glioma', 'meningioma', 'pituitary', 'notumor']
LOCAL_BASE_DIR = Path("BrainTumorImages")

def setup_dataset():
    print("Downloading dataset from Kaggle...")
    # Downloads to a local cache and returns the path
    kaggle_path = Path(kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset"))
    print(f"Dataset downloaded to cache: {kaggle_path}\n")

    # Define Kaggle cache paths
    kaggle_train_dir = kaggle_path / "Training"
    kaggle_test_dir = kaggle_path / "Testing"

    # Define new local project directories
    local_train_dir = LOCAL_BASE_DIR / "Training"
    local_val_dir = LOCAL_BASE_DIR / "Validation"
    local_test_dir = LOCAL_BASE_DIR / "Testing"

    # Create target directories (e.g., BrainTumorImages/Training/glioma)
    print("Creating folder structure...")
    for split_dir in [local_train_dir, local_val_dir, local_test_dir]:
        for class_name in CLASSES:
            os.makedirs(split_dir / class_name, exist_ok=True)

    print("Splitting and copying Training and Validation sets...")
    random.seed(SEED) # Ensures the 85/15 split is the exact same if run this twice

    for class_name in CLASSES:
        class_train_dir = kaggle_train_dir / class_name
        if not class_train_dir.exists():
            print(f"Warning: Could not find {class_train_dir}")
            continue

        # Get all images in the class
        images = list(class_train_dir.glob("*.jpg"))
        
        # Shuffle for a truly random split
        random.shuffle(images)

        # Calculate the index to split the list
        split_idx = int(len(images) * (1 - VAL_SPLIT))
        train_images = images[:split_idx]
        val_images = images[split_idx:]

        # Copy to local Training folder
        for img_path in train_images:
            shutil.copy(img_path, local_train_dir / class_name / img_path.name)
        
        # Copy to local Validation folder
        for img_path in val_images:
            shutil.copy(img_path, local_val_dir / class_name / img_path.name)

        print(f"  {class_name}: {len(train_images)} Train | {len(val_images)} Validation")

    print("\nCopying Testing set...")
    for class_name in CLASSES:
        class_test_dir = kaggle_test_dir / class_name
        if not class_test_dir.exists():
            continue
            
        images = list(class_test_dir.glob("*.jpg"))
        for img_path in images:
            shutil.copy(img_path, local_test_dir / class_name / img_path.name)
            
        print(f"  {class_name}: {len(images)} Test")

    print(f"\nSuccess! Dataset is ready at: {LOCAL_BASE_DIR.absolute()}")

if __name__ == "__main__":
    setup_dataset()