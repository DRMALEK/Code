import os
import numpy as np
from PIL import Image
from tqdm import tqdm

def calculate_stats(dataset_path):
    """
    Calculate mean and std of pixel values across all images in the dataset.

    Args:
        dataset_path: Path to the main dataset folder

    Returns:
        mean and std per channel
    """
    # Lists to store all pixel values for each channel
    r_pixels = []
    g_pixels = []
    b_pixels = []

    # Count total images for progress tracking
    total_images = 0
    for split in ['train', 'test', 'val']:
        split_path = os.path.join(dataset_path, split)
        if not os.path.exists(split_path):
            continue

        for video_folder in os.listdir(split_path):
            video_path = os.path.join(split_path, video_folder)
            if os.path.isdir(video_path):
                total_images += len([f for f in os.listdir(video_path) if os.path.isfile(os.path.join(video_path, f))])

    print(f"Found {total_images} images in the dataset")

    # Process each image
    with tqdm(total=total_images, desc="Processing images") as pbar:
        for split in ['train', 'test', 'val']:
            split_path = os.path.join(dataset_path, split)
            if not os.path.exists(split_path):
                continue

            for video_folder in os.listdir(split_path):
                video_path = os.path.join(split_path, video_folder)
                if not os.path.isdir(video_path):
                    continue

                for frame_file in os.listdir(video_path):
                    frame_path = os.path.join(video_path, frame_file)
                    if not os.path.isfile(frame_path):
                        continue

                    try:
                        # Open image and convert to RGB
                        img = Image.open(frame_path).convert('RGB')
                        img_np = np.array(img) / 255.0  # Normalize to [0, 1]

                        # Sample pixels (using every 10th pixel to speed up calculation)
                        r_pixels.append(img_np[::10, ::10, 0].flatten())
                        g_pixels.append(img_np[::10, ::10, 1].flatten())
                        b_pixels.append(img_np[::10, ::10, 2].flatten())

                    except Exception as e:
                        print(f"Error processing {frame_path}: {e}")

                    pbar.update(1)

    # Concatenate all pixel values
    r_pixels = np.concatenate(r_pixels)
    g_pixels = np.concatenate(g_pixels)
    b_pixels = np.concatenate(b_pixels)

    # Calculate mean and std
    mean = [r_pixels.mean(), g_pixels.mean(), b_pixels.mean()]
    std = [r_pixels.std(), g_pixels.std(), b_pixels.std()]

    return mean, std

def main():
    # Get dataset path from user
    dataset_path = input("Enter the path to your dataset folder: ")

    if not os.path.exists(dataset_path):
        print(f"Error: Path {dataset_path} does not exist")
        return

    print("Calculating mean and standard deviation...")
    mean, std = calculate_stats(dataset_path)

    print("\nResults:")
    print(f"Mean: {mean}")
    print(f"Std: {std}")

    # Save results to a file
    with open("normalization_stats.txt", "w") as f:
        f.write(f"Mean: {mean}\n")
        f.write(f"Std: {std}\n")

    print("\nThese values can be used for normalization in your data loader:")
    print("transforms.Normalize(mean=mean, std=std)")

if __name__ == "__main__":
    main()

# Created/Modified files during execution:
# normalization_stats.txt