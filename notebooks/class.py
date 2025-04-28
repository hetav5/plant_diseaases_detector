import os

train_dir = r"f:\codes\plant_diseases_detector\data\train"

for cls in os.listdir(train_dir):
    count = len(os.listdir(os.path.join(train_dir, cls)))
    print(f"{cls}: {count}")
