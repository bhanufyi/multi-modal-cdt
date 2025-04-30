import os
import shutil
from sklearn.model_selection import train_test_split

# Path to your "data" folder containing patient subfolders
data_dir = "./data"  # <-- CHANGE THIS to your data directory

# Create train/test subdirectories inside "data"
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Collect all patient folders except 'train'/'test'
patient_folders = [
    f
    for f in os.listdir(data_dir)
    if os.path.isdir(os.path.join(data_dir, f)) and f not in ["train", "test"]
]

# Split into train/test using sklearn
train_patients, test_patients = train_test_split(
    patient_folders,
    test_size=0.2,  # 20% for test, 80% for train
    random_state=42,  # For reproducible shuffling
    shuffle=True,
)

print(f"Total patient folders: {len(patient_folders)}")
print(f"Train split: {len(train_patients)}")
print(f"Test split: {len(test_patients)}")

# Move train patient folders to "train" directory
for pid in train_patients:
    src = os.path.join(data_dir, pid)
    dst = os.path.join(train_dir, pid)
    shutil.move(src, dst)

# Move test patient folders to "test" directory
for pid in test_patients:
    src = os.path.join(data_dir, pid)
    dst = os.path.join(test_dir, pid)
    shutil.move(src, dst)

print("Data split completed!")
