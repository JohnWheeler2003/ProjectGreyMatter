import os
import shutil
import hashlib
import kagglehub
import pandas as pd
from pathlib import Path
from PIL import Image
import imagehash
from sklearn.model_selection import train_test_split

# CONFIGURATION
SEED = 42 # For reproducible dataset shuffling
CLASSES = ['glioma', 'meningioma', 'pituitary', 'notumor']
LOCAL_BASE_DIR = Path("BrainTumorImages") # Target directory for clean data
SIMILARITY_THRESHOLD = 5 # pHash distance threshold for near-duplicates

def implement_split_physically(df, split_name):
    """Copies files from the cached Kaggle dir into the new split directory structure."""
    print(f" -> Moving {len(df)} distinct images to {split_name} directory...")
    target_dir = LOCAL_BASE_DIR / split_name
    
    for _, row in df.iterrows():
        src_path = row['path']
        label = row['label']
        filename = os.path.basename(src_path)
        
        # Create class directory if it doesn't exist
        class_dir = target_dir / label
        os.makedirs(class_dir, exist_ok=True)
        
        # Copy file to the new clean directory
        dest_path = class_dir / filename
        shutil.copy2(src_path, dest_path)

def process_and_deduplicate(split_dir_path):
    """
    Runs the hashing and deduplication logic strictly on a single folder 
    (e.g., just the Training folder, or just the Testing folder).
    """
    print(f"\nProcessing directory: {split_dir_path.name}...")
    
    # 1. Gather paths
    image_paths = []
    for root, _, files in os.walk(split_dir_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))
                
    print(f"Raw images found: {len(image_paths)}")
    
    # 2. Hashing
    image_data = {}
    for path in image_paths:
        try:
            label = Path(path).parent.name
            if label not in CLASSES:
                continue

            with open(path, 'rb') as f:
                md5_hash = hashlib.md5(f.read()).hexdigest()
            
            img = Image.open(path).convert("RGB")
            phash = imagehash.phash(img)
                
            image_data[path] = {
                'md5': md5_hash,
                'phash': phash,
                'label': label
            }
        except Exception:
            pass # Silently skip corrupted files

    # 3. Deduplication
    unique_hashes = set()
    md5_clean_paths = []

    # Step A: Exact Duplicates (MD5)
    for path in image_data.keys():
        h = image_data[path]['md5']
        if h not in unique_hashes:
            unique_hashes.add(h)
            md5_clean_paths.append(path)

    # Step B: Near-Duplicates (pHash)
    accepted_phashes = []
    final_clean_paths = []

    for path in md5_clean_paths:
        ph = image_data[path]['phash']
        is_duplicate = any(ph - existing_hash <= SIMILARITY_THRESHOLD for existing_hash in accepted_phashes)
        
        if not is_duplicate:
            accepted_phashes.append(ph)
            final_clean_paths.append(path)

    print(f"Clean images remaining: {len(final_clean_paths)}")
    
    # Return as a DataFrame for easy handling
    data = [[path, image_data[path]['label']] for path in final_clean_paths]
    return pd.DataFrame(data, columns=["path", "label"])

def print_directory_distribution():
    """Counts and prints the number of images in each class for every split."""
    print("\n--- FINAL FOLDER DISTRIBUTION ---")
    splits = ['Training', 'Validation', 'Testing']
    
    for split in splits:
        split_dir = LOCAL_BASE_DIR / split
        if not split_dir.exists():
            continue
            
        print(f"\n{split} Split:")
        split_total = 0
        
        # Count each class
        for label in CLASSES:
            class_dir = split_dir / label
            # Count files if the directory exists, otherwise 0
            count = len([f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]) if class_dir.exists() else 0
            print(f"  - {label}: {count}")
            split_total += count
            
        print(f"  TOTAL {split.upper()}: {split_total}")


def setup_dataset():

    # STAGE 1: KAGGLE DOWNLOAD
    print("--- STAGE 1: Kaggle Download ---")
    kaggle_path = Path(kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset"))
    print(f"Dataset downloaded to cache: {kaggle_path}")
    
    # Find Kaggle's predefined original splits
    original_train_dir = kaggle_path / "Training"
    original_test_dir = kaggle_path / "Testing"

    # STAGE 2 & 3: ISOLATED PROCESSING & DEDUPLICATION
    print("\n--- STAGE 2 & 3: Isolated Hashing & Deduplication ---")
    # We clean the testing set completely independently of the training set!
    clean_test_df = process_and_deduplicate(original_test_dir)
    clean_train_pool_df = process_and_deduplicate(original_train_dir)

    # STAGE 4: DATA SPLITTING (Validation only)
    print("\n--- STAGE 4: Data Splitting (Carving Validation from Training) ---")
    
    # We leave the Test set alone. We only split the clean Training pool to create our Val set.
    # We'll take 20% of the training set to use as validation.
    train_df, val_df = train_test_split(
        clean_train_pool_df, 
        test_size=0.2, 
        stratify=clean_train_pool_df['label'], 
        random_state=SEED
    )

    print("\nFinal Clean Split Sizes:")
    print(f"Training Data: {len(train_df)}")
    print(f"Validation Data: {len(val_df)}")
    print(f"Testing Data: {len(clean_test_df)}")

# STAGE 5: PHYSICAL IMPLEMENTATION
    print("\n--- STAGE 5: Physical Extraction ---")
    
    if LOCAL_BASE_DIR.exists():
        shutil.rmtree(LOCAL_BASE_DIR)
    
    implement_split_physically(train_df, 'Training')
    implement_split_physically(val_df, 'Validation')
    implement_split_physically(clean_test_df, 'Testing')

    print(f"\nSuccess! Leakage-free dataset is ready at: {LOCAL_BASE_DIR.absolute()}")
    
    # NEW: Print the final distributions
    print_directory_distribution()

if __name__ == "__main__":
    setup_dataset()