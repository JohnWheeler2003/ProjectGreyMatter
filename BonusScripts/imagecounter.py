import os

def count_images_in_folders(base_path):
    split_name = os.path.basename(base_path)

    print(f"\nImage counts for {split_name}:")
    split_total = 0

    for folder in sorted(os.listdir(base_path)):
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path):
            count = len(os.listdir(folder_path))
            split_total += count
            print(f"  {folder}: {count} images")

    print(f"Total in {split_name}: {split_total} images")
    return split_total


train_dir = "BrainTumorImages/Training"
val_dir   = "BrainTumorImages/Validation"
test_dir  = "BrainTumorImages/Testing"

train_total = count_images_in_folders(train_dir)
val_total   = count_images_in_folders(val_dir)
test_total  = count_images_in_folders(test_dir)

grand_total = train_total + val_total + test_total

print("\n==============================")
print(f"Grand Total Images: {grand_total}")
print("==============================")
