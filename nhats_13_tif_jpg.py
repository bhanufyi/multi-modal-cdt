from PIL import Image
import os

# Define the source and output directories
source_folder = "NHATS_R13_ClockDrawings"
output_folder = "NHATS_R13_ClockDrawings_JPG"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Iterate through all files in the source folder
for filename in os.listdir(source_folder):
    if filename.lower().endswith(".tif") or filename.lower().endswith(".tiff"):
        # Define file paths
        tif_path = os.path.join(source_folder, filename)
        jpg_path = os.path.join(output_folder, filename.rsplit(".", 1)[0] + ".jpg")

        # Open and convert the image
        with Image.open(tif_path) as img:
            img.convert("RGB").save(jpg_path, "JPEG", quality=95)

print("Conversion completed. JPG files are saved in:", output_folder)
