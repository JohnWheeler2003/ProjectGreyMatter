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

def get_image_paths(base_dir):
    """Recursively fetches all image paths from a given directory."""
    image_paths = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))
    return image_paths

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

def setup_dataset():

    # STAGE 1: KAGGLE DOWNLOAD & PATH COLLECTION
    print("--- STAGE 1: Kaggle Download ---")
    kaggle_path = Path(kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset"))
    print(f"Dataset downloaded to cache: {kaggle_path}")

    # Gather all images from the Kaggle cache to create a unified pool
    all_image_paths = get_image_paths(kaggle_path)
    print(f"Total raw images found: {len(all_image_paths)}")


    # STAGE 2: ONE-PASS DATA EXTRACTION
    print("\n--- STAGE 2: Image Processing (Hashing) ---")
    image_data = {}

    for path in all_image_paths:
        try:
            # Label determined from the parent folder of the image
            label = Path(path).parent.name
            
            # Skip folders that aren't in target CLASSES list
            if label not in CLASSES:
                continue

            # 1. MD5 Hash for exact duplicates
            with open(path, 'rb') as f:
                md5_hash = hashlib.md5(f.read()).hexdigest()
            
            # 2. Perceptual Hash for near-duplicates
            img = Image.open(path).convert("RGB")
            phash = imagehash.phash(img)
                
            image_data[path] = {
                'md5': md5_hash,
                'phash': phash,
                'label': label
            }
        except Exception:
            # Silently skip corrupted files
            pass

    print(f"Successfully processed {len(image_data)} valid class images.")


    # STAGE 3: DATASET DEDUPLICATION
    print("\n--- STAGE 3: Deduplication ---")

    # Step A: Remove Exact Duplicates (MD5)
    unique_hashes = set()
    md5_clean_paths = []

    for path in image_data.keys():
        h = image_data[path]['md5']
        if h not in unique_hashes:
            unique_hashes.add(h)
            md5_clean_paths.append(path)

    print(f"Images remaining after MD5 exact deduplication: {len(md5_clean_paths)}")

    # Step B: Remove High Similarity Images (pHash)
    accepted_phashes = []
    final_clean_paths = []

    for path in md5_clean_paths:
        ph = image_data[path]['phash']
        
        is_duplicate = any(ph - existing_hash <= SIMILARITY_THRESHOLD for existing_hash in accepted_phashes)
        
        if not is_duplicate:
            accepted_phashes.append(ph)
            final_clean_paths.append(path)

    print(f"Images remaining after pHash near-deduplication: {len(final_clean_paths)}")


    # STAGE 4: DATAFRAME BUILD & SPLITTING
    print("\n--- STAGE 4: Data Splitting ---")
    data = [[path, image_data[path]['label']] for path in final_clean_paths]
    clean_df = pd.DataFrame(data, columns=["path", "label"])

    # 1st Split: Isolate 20% for Testing
    train_temp_df, test_df = train_test_split(
        clean_df, 
        test_size=0.2, 
        stratify=clean_df['label'], 
        random_state=SEED
    )

    # 2nd Split: Split the remaining 80% into Train/Val (80/20)
    # This yields roughly: 64% Train, 16% Val, 20% Test overall
    train_df, val_df = train_test_split(
        train_temp_df, 
        test_size=0.2, 
        stratify=train_temp_df['label'], 
        random_state=SEED
    )

    print(f"Final Clean Class Balance:")
    print(clean_df['label'].value_counts().to_string())


    # STAGE 5: PHYSICAL IMPLEMENTATION
    print("\n--- STAGE 5: Physical Extraction ---")
    
    # Safely clear the target directory if it already exists to prevent data mixing
    if LOCAL_BASE_DIR.exists():
        shutil.rmtree(LOCAL_BASE_DIR)
    
    implement_split_physically(train_df, 'Training')
    implement_split_physically(val_df, 'Validation')
    implement_split_physically(test_df, 'Testing')

    print(f"\nSuccess! Clean dataset is ready at: {LOCAL_BASE_DIR.absolute()}")

if __name__ == "__main__":
    setup_dataset()