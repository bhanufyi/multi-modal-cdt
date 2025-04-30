import cv2
import numpy as np
import os
from pathlib import Path
import time


def segment_clean_with_bounding_boxes(image_path, output_path=None, buffer_percent=20):
    """https://claude.ai/chat/c58e8c89-a22c-4c6f-8ef8-3fb43320114f
    Detect clock and black regions, then delete black regions only if they're
    outside the clock's actual shape (not just the bounding box).

    Args:
        image_path: Path to the input image
        output_path: Path to save the output image (optional)
        buffer_percent: Percentage of buffer to add around clock (default 20%)

    Returns:
        Dictionary with processed image and metadata or None if processing failed
    """
    # Read the image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Could not read image: {image_path}")
        return None

    # Get original dimensions
    height, width = img.shape[:2]

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Create output image for visualization with bounding boxes
    output_img = img.copy()

    # Create a clean copy of the image for the final result (no boxes)
    clean_img = img.copy()

    # STAGE 1: Detect very dark regions (black boxes) using a low threshold
    _, black_thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)

    # Apply morphological operations to clean up the image
    kernel = np.ones((3, 3), np.uint8)
    black_thresh = cv2.morphologyEx(black_thresh, cv2.MORPH_CLOSE, kernel)

    # Find contours of very dark regions
    black_contours, _ = cv2.findContours(
        black_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # List to store coordinates of blue/purple boxes
    blue_boxes = []
    border_regions = []  # Store contours for exclusion mask

    # Filter and draw rectangles around black regions
    for contour in black_contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)

        # Skip very small regions
        if area < 50:
            continue

        # Create mask for this contour to measure darkness
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [contour], 0, 255, -1)
        mean_intensity = cv2.mean(gray, mask=mask)[0]

        # Check if region is near any border
        border_margin = min(width, height) // 6  # 1/6 of the smaller dimension
        is_near_border = (
            x <= border_margin
            or y <= border_margin
            or x + w >= width - border_margin
            or y + h >= height - border_margin
        )

        # Calculate rectangularity (how rectangular the contour is)
        rect_area = w * h
        rectangularity = area / rect_area if rect_area > 0 else 0

        # Accept if it's dark enough, near a border, and reasonably rectangular
        if mean_intensity < 50 and is_near_border and rectangularity > 0.5:
            # Draw blue box for black border regions in visualization
            cv2.rectangle(output_img, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Store blue box coordinates
            blue_boxes.append({"contour": contour, "x": x, "y": y, "w": w, "h": h})

            # Add to border regions for exclusion mask
            border_regions.append(contour)

    # STAGE 2: IMPROVED CLOCK DETECTION with multiple thresholds
    # Initialize a combined clock mask
    combined_clock_mask = np.zeros((height, width), dtype=np.uint8)

    # Try multiple thresholds to capture more of the clock
    thresholds = [220, 230, 240]
    for threshold in thresholds:
        _, clock_thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

        # Create a mask to exclude border regions from clock detection
        exclude_mask = np.ones_like(gray)
        cv2.drawContours(exclude_mask, border_regions, -1, 0, -1)

        # Apply the exclusion mask
        clock_thresh = cv2.bitwise_and(clock_thresh, clock_thresh, mask=exclude_mask)

        # Add to combined mask
        combined_clock_mask = cv2.bitwise_or(combined_clock_mask, clock_thresh)

    # Connected component analysis on combined mask
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        combined_clock_mask, connectivity=8
    )

    # Process each component
    components = []
    center_x, center_y = width // 2, height // 2

    for i in range(1, num_labels):  # Skip label 0 (background)
        x, y, w, h, area = stats[i]
        cx, cy = centroids[i]

        # Skip very small components
        if area < 20:
            continue

        # Calculate distance from image center
        distance_from_center = np.sqrt((cx - center_x) ** 2 + (cy - center_y) ** 2)

        # Calculate distance from border
        min_border_dist = min(cx, cy, width - cx, height - cy)

        # Score based on size, centrality, and border distance
        centrality_score = 1.0 / (1.0 + distance_from_center / 100)
        border_score = min_border_dist / 100

        # Higher score indicates more likely to be part of the clock
        score = area * (centrality_score + border_score)

        components.append(
            {
                "label": i,
                "area": area,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "cx": cx,
                "cy": cy,
                "score": score,
            }
        )

    # Sort components by score (highest first)
    components.sort(key=lambda c: c["score"], reverse=True)

    # No clock components found
    if not components:
        print(f"No clock components found in {image_path}")
        return None

    # Get the top component and filter others based on proximity
    top_component = components[0]
    selected_components = [top_component]

    for comp in components[1:]:
        # Check distance to top component
        distance_to_top = np.sqrt(
            (comp["cx"] - top_component["cx"]) ** 2
            + (comp["cy"] - top_component["cy"]) ** 2
        )

        # Include if it's close enough and has a good score
        if (
            distance_to_top < max(width, height) * 0.4
            and comp["score"] > top_component["score"] * 0.2
        ):
            selected_components.append(comp)

    # Create a mask for selected clock components
    clock_mask = np.zeros_like(labels)
    for comp in selected_components:
        clock_mask[labels == comp["label"]] = 255

    # Convert to uint8 for OpenCV operations
    clock_mask = clock_mask.astype(np.uint8)

    # Find contours of the combined clock region for visualization
    clock_contours, _ = cv2.findContours(
        clock_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not clock_contours:
        print(f"No clock contours found in {image_path}")
        return None

    # Combine all clock contours for bounding box calculation
    all_clock_points = np.vstack([c.reshape(-1, 2) for c in clock_contours])
    x_clock, y_clock, w_clock, h_clock = cv2.boundingRect(all_clock_points)

    # Store the red bounding box coordinates
    red_box = {"x": x_clock, "y": y_clock, "w": w_clock, "h": h_clock}

    # Draw red box around the clock (no buffer) in visualization
    cv2.rectangle(
        output_img,
        (x_clock, y_clock),
        (x_clock + w_clock, y_clock + h_clock),
        (0, 0, 255),
        2,
    )

    # Create variables for our clock contour approaches
    ellipse_contour = None
    hull_contour = None
    final_clock_contour = None

    # Combine all detected contours
    combined_contour = np.vstack([c for c in clock_contours])

    # Create a convex hull from the detected points
    hull_contour = cv2.convexHull(combined_contour)

    # Draw the hull-based contour in purple for visualization
    cv2.drawContours(output_img, [hull_contour], 0, (255, 0, 255), 2)

    # Try to fit an ellipse if we have enough points
    if len(combined_contour) >= 5:
        try:
            # Fit ellipse to all detected points
            ellipse = cv2.fitEllipse(combined_contour)

            # Convert ellipse to contour points
            ellipse_contour = cv2.ellipse2Poly(
                (int(ellipse[0][0]), int(ellipse[0][1])),
                (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2)),
                int(ellipse[2]),
                0,
                360,
                5,
            )

            # Draw the ellipse in yellow for visualization
            cv2.drawContours(output_img, [ellipse_contour], 0, (0, 255, 255), 2)

            # Use the ellipse as our primary clock contour
            final_clock_contour = ellipse_contour
        except Exception as e:
            print(f"Error fitting ellipse: {e}")
            # Fall back to hull if ellipse fitting fails
            final_clock_contour = hull_contour
    else:
        # Not enough points for ellipse, use hull
        final_clock_contour = hull_contour

    # If we still don't have a valid contour, create a fallback based on bounding box
    if final_clock_contour is None or len(final_clock_contour) < 5:
        # Calculate center and radius from bounding box
        center = (x_clock + w_clock // 2, y_clock + h_clock // 2)
        axes = (w_clock // 2, h_clock // 2)
        angle = 0

        # Create an elliptical contour from the bounding box
        final_clock_contour = cv2.ellipse2Poly(center, axes, angle, 0, 360, 5)

    # Create a mask from the final clock contour
    final_clock_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.drawContours(final_clock_mask, [final_clock_contour], 0, 255, -1)

    # Apply buffer to create the dilated contour (green)
    buffer_kernel_size = max(3, int(min(width, height) * buffer_percent / 300))
    buffer_kernel = np.ones((buffer_kernel_size, buffer_kernel_size), np.uint8)
    dilated_clock_mask = cv2.dilate(final_clock_mask, buffer_kernel, iterations=1)

    # Find contours of the dilated mask for visualization
    dilated_contours, _ = cv2.findContours(
        dilated_clock_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Draw the dilated contour in green for visualization
    cv2.drawContours(output_img, dilated_contours, -1, (0, 255, 0), 2)

    # Create a mask for regions to delete (make white)
    regions_to_delete_mask = np.zeros((height, width), dtype=np.uint8)

    # Check each blue box for overlap with the dilated clock mask
    for blue_box in blue_boxes:
        blue_x = blue_box["x"]
        blue_y = blue_box["y"]
        blue_w = blue_box["w"]
        blue_h = blue_box["h"]

        # Create a mask for this blue box
        blue_box_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(blue_box_mask, [blue_box["contour"]], 0, 255, -1)

        # Check if this blue box overlaps with the dilated clock mask
        overlap = np.any(np.logical_and(blue_box_mask > 0, dilated_clock_mask > 0))

        # Check if this is potentially a clock element (small and centered within the clock)
        clock_center_x = x_clock + w_clock // 2
        clock_center_y = y_clock + h_clock // 2

        # Calculate distance from center of blue box to center of clock
        blue_center_x = blue_x + blue_w // 2
        blue_center_y = blue_y + blue_h // 2

        distance_to_clock_center = np.sqrt(
            (blue_center_x - clock_center_x) ** 2
            + (blue_center_y - clock_center_y) ** 2
        )

        # Is this box likely to be a clock element?
        is_small = blue_w * blue_h < (
            w_clock * h_clock * 0.1
        )  # Less than 10% of clock area
        is_centered = distance_to_clock_center < (
            max(w_clock, h_clock) * 0.3
        )  # Within 30% of clock "radius"

        is_clock_element = is_small and is_centered

        # If no overlap with dilated clock mask AND not likely to be a clock element,
        # mark this region for deletion
        if not overlap and not is_clock_element:
            cv2.drawContours(regions_to_delete_mask, [blue_box["contour"]], 0, 255, -1)

    # Replace regions to delete with white (delete them)
    clean_img[regions_to_delete_mask > 0] = [255, 255, 255]

    # Also detect any dark lines/elements using a different threshold to catch hands
    _, dark_elements = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

    # Find contours of the dark elements
    dark_contours, _ = cv2.findContours(
        dark_elements, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Filter dark contours to only include those near or connected to the clock
    valid_dark_contours = []

    # Dilate the clock mask to detect elements connected or close to the clock
    connection_kernel = np.ones((20, 20), np.uint8)  # Larger kernel to check proximity
    extended_clock_mask = cv2.dilate(final_clock_mask, connection_kernel, iterations=1)

    for contour in dark_contours:
        # Check if contour is likely a clock element (hand, etc.)
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)

        # Skip very small contours
        if area < 10:
            continue

        # Create a mask for this contour
        contour_mask = np.zeros_like(gray)
        cv2.drawContours(contour_mask, [contour], 0, 255, -1)

        # Check if this contour overlaps with the extended clock mask
        overlap = np.any(np.logical_and(contour_mask > 0, extended_clock_mask > 0))

        if overlap:
            valid_dark_contours.append(contour)
            # Add to visualization in cyan
            cv2.drawContours(output_img, [contour], -1, (255, 255, 0), 2)

    # Return cleaned image, red box, blue boxes, and all contours for cropping
    all_element_contours = []

    # Add all detected clock contours
    for contour in clock_contours:
        all_element_contours.append(contour)

    # Add all valid dark contours
    all_element_contours.extend(valid_dark_contours)

    return {
        "cleaned_img": clean_img,
        "red_box": red_box,
        "blue_boxes": blue_boxes,
        "all_element_contours": all_element_contours,
        "dilated_contours": dilated_contours
    }


def crop_and_resize_image(result_dict, buffer_percent=20, output_size=(224, 224)):
    """
    Simple two-strategy approach:
    1. If bounding box < 224x224: crop to that bounding box
    2. If bounding box >= 224x224: resize to 224x224

    Args:
        result_dict: Dictionary containing the cleaned image and detection data
        buffer_percent: Percentage of buffer to add around min-max (default 20%)
        output_size: Target size for the final image (default 224x224)

    Returns:
        Resized image or None on failure
    """
    if result_dict is None:
        return None

    # Extract necessary data from the result dictionary
    clean_img = result_dict["cleaned_img"]

    # Get image dimensions
    height, width = clean_img.shape[:2]
    output_w, output_h = output_size

    # Find all non-white pixels (clock elements)
    non_white_mask = np.zeros((height, width), dtype=np.uint8)
    # Consider all channels - if any channel is not 250, it's not white
    non_white_mask[np.any(clean_img < 250, axis=2)] = 255

    # Find all non-zero coordinates in the mask
    y_coords, x_coords = np.where(non_white_mask > 0)

    # If no elements found, return the whole image resized
    if len(y_coords) == 0 or len(x_coords) == 0:
        print("No clock elements found in the image")
        result_img = cv2.resize(clean_img, output_size, interpolation=cv2.INTER_AREA)
        return {
            "resized_img": result_img,
            "crop_coords": (0, 0, width, height),
            "strategy": "full_resize",
        }

    # Find min-max coordinates of all clock elements
    min_x, max_x = np.min(x_coords), np.max(x_coords)
    min_y, max_y = np.min(y_coords), np.max(y_coords)

    # Calculate content dimensions
    content_width = max_x - min_x + 1
    content_height = max_y - min_y + 1

    # Add buffer based on content size
    buffer_x = int(content_width * buffer_percent / 100)
    buffer_y = int(content_height * buffer_percent / 100)

    # Calculate crop coordinates with buffer
    crop_min_x = max(0, min_x - buffer_x)
    crop_min_y = max(0, min_y - buffer_y)
    crop_max_x = min(width, max_x + buffer_x)
    crop_max_y = min(height, max_y + buffer_y)

    # Calculate crop dimensions
    crop_width = crop_max_x - crop_min_x
    crop_height = crop_max_y - crop_min_y

    # Ensure square crop by expanding the smaller dimension
    if crop_width > crop_height:
        # Width is larger, adjust height to match (centered)
        diff = crop_width - crop_height
        crop_min_y = max(0, crop_min_y - diff // 2)
        crop_max_y = min(height, crop_min_y + crop_width)
    else:
        # Height is larger, adjust width to match (centered)
        diff = crop_height - crop_width
        crop_min_x = max(0, crop_min_x - diff // 2)
        crop_max_x = min(width, crop_min_x + crop_height)

    # Final crop dimensions
    final_crop_w = crop_max_x - crop_min_x
    final_crop_h = crop_max_y - crop_min_y

    # Determine strategy based on crop size vs output size
    is_small_crop = final_crop_w < output_w and final_crop_h < output_h

    if is_small_crop:
        # STRATEGY 1: Bounding box is smaller than 224x224, just crop it
        cropped_img = clean_img[crop_min_y:crop_max_y, crop_min_x:crop_max_x]

        # Create a white canvas of the output size
        canvas = np.ones((*output_size, 3), dtype=np.uint8) * 255

        # Calculate position to place the crop in the center of the canvas
        paste_x = (output_w - final_crop_w) // 2
        paste_y = (output_h - final_crop_h) // 2

        # Place the cropped image on the canvas
        canvas[paste_y : paste_y + final_crop_h, paste_x : paste_x + final_crop_w] = (
            cropped_img
        )

        result_img = canvas
        strategy = "crop_to_bounding_box"
    else:
        # STRATEGY 2: Bounding box is larger than or equal to 224x224, resize the whole image
        result_img = cv2.resize(clean_img, output_size, interpolation=cv2.INTER_AREA)
        strategy = "resize_whole_image"

    return {
        "resized_img": result_img,
        "crop_coords": (crop_min_x, crop_min_y, final_crop_w, final_crop_h),
        "strategy": strategy,
    }


def process_subset_test_folder(subset_test_dir, output_size=(224, 224)):
    """
    Process all images in the subset_test directory with patient folders.
    Save results to 'subset_test_combined' with the same filename structure.

    Args:
        subset_test_dir: Path to the subset_test directory containing patient folders
        output_size: Target size for the resized images (default 224x224)

    Returns:
        Lists of successfully processed and failed image paths
    """
    subset_test_dir = Path(subset_test_dir)
    processed_images = []
    failed_images = []

    # Create output directory for processed images (subset_test_combined)
    output_base_dir = subset_test_dir.parent / "subset_test_combined"
    output_base_dir.mkdir(exist_ok=True)

    # Get list of patient folders
    patient_folders = [f for f in subset_test_dir.iterdir() if f.is_dir()]

    print(f"Found {len(patient_folders)} patient folders in {subset_test_dir}")

    # Process each patient folder
    start_time = time.time()
    for i, patient_dir in enumerate(patient_folders):
        patient_id = patient_dir.name
        print(f"[{i+1}/{len(patient_folders)}] Processing patient {patient_id}...")

        # Create output directory for this patient
        patient_output_dir = output_base_dir / patient_id
        patient_output_dir.mkdir(exist_ok=True)

        # Find all image files in this patient folder
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
        image_files = []

        for ext in image_extensions:
            image_files.extend(list(patient_dir.glob(f"*{ext}")))
            image_files.extend(list(patient_dir.glob(f"*{ext.upper()}")))

        if not image_files:
            print(f"  No images found for patient {patient_id}")
            continue

        print(f"  Found {len(image_files)} images for patient {patient_id}")

        # Process each image
        for img_path in image_files:
            # Create output path with same filename in the output directory
            output_path = patient_output_dir / img_path.name

            try:
                # Step 1: Segment and clean the image
                result_dict = segment_clean_with_bounding_boxes(img_path)

                if result_dict:
                    # Step 2: Crop and resize the cleaned image
                    crop_result = crop_and_resize_image(
                        result_dict, output_size=output_size
                    )

                    if crop_result:
                        # Step 3: Save the processed image
                        cv2.imwrite(str(output_path), crop_result["resized_img"])

                        processed_images.append(str(img_path))
                        print(f"  ✓ Processed: {img_path.name}")
                    else:
                        failed_images.append(str(img_path))
                        print(f"  ✗ Failed to crop: {img_path.name}")
                else:
                    failed_images.append(str(img_path))
                    print(f"  ✗ Failed to process: {img_path.name}")
            except Exception as e:
                failed_images.append(str(img_path))
                print(f"  ✗ Error processing {img_path.name}: {e}")

    # Calculate elapsed time
    elapsed_time = time.time() - start_time

    # Print summary
    print("\n" + "=" * 50)
    print("SUMMARY:")
    print(f"Total patient folders: {len(patient_folders)}")
    print(f"Total images processed: {len(processed_images)}")
    print(f"Total images failed: {len(failed_images)}")
    print(f"Processing time: {elapsed_time:.2f} seconds")
    print(f"Results saved to {output_base_dir}")
    print("=" * 50)

    return processed_images, failed_images


if __name__ == "__main__":
    # Path to the subset_test folder
    subset_test_directory = "./subset_test"  # Change this to your actual path

    # Process all images in the subset_test folder
    processed, failed = process_subset_test_folder(
        subset_test_directory, output_size=(224, 224)
    )
