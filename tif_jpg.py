import os
from PIL import Image
from tqdm import tqdm

# Define the source and destination directories.
source_dir = "."
dest_dir = "dataset"

# Create the destination directory if it doesn't exist.
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

# First, collect all .tif files from the subfolders.
files_to_process = []
for folder in os.listdir(source_dir):
    folder_path = os.path.join(source_dir, folder)
    if os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(".tif"):
                file_path = os.path.join(folder_path, filename)
                files_to_process.append((folder, filename, file_path))

# Process the files with a progress bar.
for folder, filename, file_path in tqdm(
    files_to_process, desc="Converting images", unit="image"
):
    try:
        with Image.open(file_path) as img:
            # Ensure the image is in RGB mode for JPEG conversion.
            rgb_img = img.convert("RGB")

            # Create a new filename using folder and image names.
            base_name = os.path.splitext(filename)[0]
            new_filename = f"{folder}_{base_name}.jpg"
            dest_path = os.path.join(dest_dir, new_filename)

            # Save the image in JPEG format.
            rgb_img.save(dest_path, format="JPEG")
    except Exception as e:
        print(f"\nError processing {file_path}: {e}")

print("Conversion complete!")
