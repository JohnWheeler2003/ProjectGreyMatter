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
SEED = 42 
CLASSES = ['glioma', 'meningioma', 'pituitary', 'notumor']
LOCAL_BASE_DIR = Path("BrainTumorImages") 
SIMILARITY_THRESHOLD = 5 

def get_image_paths(target_dir):
    """Recursively fetches all image paths from a specific target directory."""
    image_paths = []
    if not target_dir.exists():
        print(f"Warning: Directory {target_dir} not found.")
        return image_paths
        
    for root, _, files in os.walk(target_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))
    return image_paths

def hash_and_deduplicate(image_paths, split_name):
    """Hashes a list of image paths and removes exact/near duplicates."""
    print(f"\n--- Hashing and Deduplicating: {split_name} ---")
    image_data = {}

    # 1. Hashing
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
            pass
            
    print(f"Successfully hashed {len(image_data)} valid images in {split_name}.")

    # 2. Step A: Remove Exact Duplicates (MD5)
    unique_hashes = set()
    md5_clean_paths = []

    for path in image_data.keys():
        h = image_data[path]['md5']
        if h not in unique_hashes:
            unique_hashes.add(h)
            md5_clean_paths.append(path)
        
    exact_dupes_removed = len(image_data) - len(md5_clean_paths)
    print(f" -> Removed {exact_dupes_removed} exact duplicate images (MD5).")

    # 3. Step B: Remove High Similarity Images (pHash)
    accepted_phashes = []
    final_clean_paths = []

    print(f" -> Checking for near-duplicates (pHash)...")
    for path in md5_clean_paths:
        ph = image_data[path]['phash']
        
        is_duplicate = any(ph - existing_hash <= SIMILARITY_THRESHOLD for existing_hash in accepted_phashes)
        
        if not is_duplicate:
            accepted_phashes.append(ph)
            final_clean_paths.append(path)

    similar_removed = len(md5_clean_paths) - len(final_clean_paths)
    print(f" -> Removed {similar_removed} highly similar images (pHash).")
    print(f" -> FINAL clean images for {split_name}: {len(final_clean_paths)}")
    
    return final_clean_paths, image_data

def implement_split_physically(df, split_name):
    """Copies files from the DataFrame paths into the new split directory structure."""
    target_dir = LOCAL_BASE_DIR / split_name
    
    for _, row in df.iterrows():
        src_path = row['path']
        label = row['label']
        filename = os.path.basename(src_path)
        
        class_dir = target_dir / label
        os.makedirs(class_dir, exist_ok=True)
        
        dest_path = class_dir / filename
        shutil.copy2(src_path, dest_path)

def print_final_directory_stats():
    """Prints the final count of images in each split and subfolder."""

    print("FINAL DATASET FOLDER COUNTS")

    total_all = 0
    for split in ['Training', 'Validation', 'Testing']:
        split_dir = LOCAL_BASE_DIR / split
        if not split_dir.exists():
            continue
            
        print(f"\n[{split.upper()}]")
        split_total = 0
        for cls in CLASSES:
            class_dir = split_dir / cls
            if class_dir.exists():
                count = len(os.listdir(class_dir))
                print(f"  ├── {cls}: {count} images")
                split_total += count
            else:
                print(f"  ├── {cls}: 0 images (Directory missing)")
        print(f"  └── Total {split}: {split_total} images")
        total_all += split_total
    

    print(f"\nOVERALL TOTAL IMAGES IN LOCAL DIR: {total_all}")


def setup_dataset():

    # STAGE 1: KAGGLE DOWNLOAD & PATH COLLECTION
    print("STAGE 1: Kaggle Download")
    kaggle_path = Path(kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset"))
    print(f"Dataset downloaded to cache: {kaggle_path}")

    kaggle_train_dir = kaggle_path / "Training"
    kaggle_test_dir = kaggle_path / "Testing"

    raw_train_paths = get_image_paths(kaggle_train_dir)
    raw_test_paths = get_image_paths(kaggle_test_dir)

    print(f"\nRaw Kaggle splits identified:")
    print(f" -> Original Training images: {len(raw_train_paths)}")
    print(f" -> Original Testing images:  {len(raw_test_paths)}")


    # STAGE 2: PROCESS & DEDUPLICATE SPLITS INDEPENDENTLY
    clean_train_paths, train_image_data = hash_and_deduplicate(raw_train_paths, "Training Pool")
    clean_test_paths, test_image_data = hash_and_deduplicate(raw_test_paths, "Testing Set")


    # STAGE 3: DATAFRAME BUILD & SPLITTING
    print("\nSTAGE 3: Data Splitting")
    
    # 1. Build DataFrame for the Cleaned Training Pool
    train_data_list = [[path, train_image_data[path]['label']] for path in clean_train_paths]
    clean_train_pool_df = pd.DataFrame(train_data_list, columns=["path", "label"])

    # Carve Validation out of the Cleaned Training pool
    train_df, val_df = train_test_split(
        clean_train_pool_df, 
        test_size=0.2, 
        stratify=clean_train_pool_df['label'], 
        random_state=SEED
    )
    
    print(f"Final Training subset:   {len(train_df)} images")
    print(f"Final Validation subset: {len(val_df)} images")

    # 2. Build DataFrame for the Cleaned Testing Set
    test_data_list = [[path, test_image_data[path]['label']] for path in clean_test_paths]
    test_df = pd.DataFrame(test_data_list, columns=["path", "label"])
    
    print(f"Final Testing subset:    {len(test_df)} images")


    # STAGE 4: PHYSICAL IMPLEMENTATION
    print("\n--- STAGE 4: Physical Extraction to Folders ---")
    
    if LOCAL_BASE_DIR.exists():
        print(f"Clearing existing directory at {LOCAL_BASE_DIR} to prevent data mixing...")
        shutil.rmtree(LOCAL_BASE_DIR)
    
    print("Writing files to disk...")
    implement_split_physically(train_df, 'Training')
    implement_split_physically(val_df, 'Validation')
    implement_split_physically(test_df, 'Testing')

    # STAGE 5: FINAL STATS
    print_final_directory_stats()
    print(f"\nSuccess! Dataset is ready at: {LOCAL_BASE_DIR.absolute()}")

if __name__ == "__main__":
    setup_dataset()