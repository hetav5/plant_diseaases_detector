import os
import shutil
import random

# Paths
SOURCE_DIR = r"F:\codes\plantdataset"  # Folder where ALL images are currently organized into class folders
DEST_DIR = r"F:\codes\plant_diseases_detector\data"         # This will create "data/train" and "data/val"

# Split ratio
TRAIN_SPLIT = 0.8

# Create train and val folders
for split in ['train', 'val']:
    split_path = os.path.join(DEST_DIR, split)
    os.makedirs(split_path, exist_ok=True)

# Process each class folder
for class_name in os.listdir(SOURCE_DIR):
    class_path = os.path.join(SOURCE_DIR, class_name)
    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    random.shuffle(images)
    train_size = int(len(images) * TRAIN_SPLIT)
    train_images = images[:train_size]
    val_images = images[train_size:]

    for split, split_images in [('train', train_images), ('val', val_images)]:
        split_class_path = os.path.join(DEST_DIR, split, class_name)
        os.makedirs(split_class_path, exist_ok=True)
        for img in split_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(split_class_path, img)
            shutil.copy2(src, dst)

print("✅ Dataset split complete!")
