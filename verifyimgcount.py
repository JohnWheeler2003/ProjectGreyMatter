import os

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

train_dir = "BrainTumorImages/Training"
val_dir = "BrainTumorImages/Validation"
test_dir = "BrainTumorImages/Testing"

# Verify all sets
count_images_in_folders(train_dir)
count_images_in_folders(val_dir)
count_images_in_folders(test_dir)
