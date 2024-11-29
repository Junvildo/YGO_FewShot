import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import albumentations as A
import random
from tqdm import tqdm


# Define all the augmentation functions
def apply_perspective_transform(image):
    height, width = image.shape[:2]
    src_points = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    dst_points = np.float32([[np.random.randint(0, 30), np.random.randint(0, 30)],
                             [width - 1 - np.random.randint(0, 30), np.random.randint(0, 30)],
                             [width - 1 - np.random.randint(0, 30), height - 1 - np.random.randint(0, 30)],
                             [np.random.randint(0, 30), height - 1 - np.random.randint(0, 30)]])
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    return cv2.warpPerspective(image, matrix, (width, height))


def apply_lighting_changes(image):
    transform = A.Compose([
        A.RandomBrightnessContrast(p=1.0),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.8)
    ])
    augmented_image = transform(image=image)["image"]
    return augmented_image


def add_shadow(image):
    enhancer = ImageEnhance.Brightness(Image.fromarray(image))
    image = np.array(enhancer.enhance(np.random.uniform(0.7, 1.0)))
    return image




def apply_blur(image):
    transform = A.Compose([
        A.GaussianBlur(blur_limit=(3, 7), p=0.5),
        A.MotionBlur(blur_limit=7, p=0.5)
    ])
    augmented_image = transform(image=image)["image"]
    return augmented_image


def add_noise(image):
    noise = np.random.normal(loc=0, scale=25, size=image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def rotate_image(image, angle):
    if angle not in [90, 180, 270]:
        raise ValueError("Angle must be 90, 180, or 270 degrees")
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

def darken_image(image, factor=0.3):
    return np.clip(image * factor, 0, 255).astype(np.uint8)


# Function to apply a random combination of selected augmentations
def apply_random_augmentations(image, augmentations):
    augmented_image = image.copy()
    chosen_augmentations = random.sample(augmentations, random.randint(1, len(augmentations)))

    for aug in chosen_augmentations:
        if aug == 'perspective':
            augmented_image = apply_perspective_transform(augmented_image)
        elif aug == 'lighting':
            augmented_image = apply_lighting_changes(augmented_image)
        elif aug == 'shadow':
            augmented_image = add_shadow(augmented_image)
        elif aug == 'blur':
            augmented_image = apply_blur(augmented_image)
        elif aug == 'noise':
            augmented_image = add_noise(augmented_image)
        elif aug == 'darken':
            augmented_image = darken_image(augmented_image)
        elif aug == 'rotate':
            angle = random.choice([90, 180, 270])
            augmented_image = rotate_image(augmented_image, angle)

    return augmented_image


# Function to process images in a folder and create 4 augmentations per image
def process_images(input_dir, augmentations):
    all_files = []
    for subdir, _, files in os.walk(input_dir):
        all_files.extend([os.path.join(subdir, file) for file in files if file.endswith(('png', 'jpg', 'jpeg'))])

    # Initialize the progress bar for the entire image set
    with tqdm(total=len(all_files) * 4) as pbar:  # Each image will generate 4 augmentations
        for file_path in all_files:
            image = cv2.imread(file_path)

            # Create 4 different augmented images by randomly applying augmentations
            for i in range(1, 5):
                augmented_image = apply_random_augmentations(image, augmentations)
                subdir = os.path.dirname(file_path)
                file_name = os.path.basename(file_path)
                output_path = os.path.join(subdir, f"aug_{i}_{file_name}")
                cv2.imwrite(output_path, augmented_image)

                # Update the progress bar
                pbar.update(1)



# Function to prompt the user for augmentation choices
def prompt_for_augmentations():
    augmentations = {
        'perspective': input("Use perspective transformation? (1 for yes, Enter to skip): ") == '1',
        'lighting': input("Use lighting changes? (1 for yes, Enter to skip): ") == '1',
        'shadow': input("Add shadow? (1 for yes, Enter to skip): ") == '1',
        'blur': input("Apply blur? (1 for yes, Enter to skip): ") == '1',
        'noise': input("Add noise? (1 for yes, Enter to skip): ") == '1',
        'darken': input("Darken image? (1 for yes, Enter to skip): ") == '1',
        'rotate': input("Add rotation (90°, 180°, 270°)? (1 for yes, Enter to skip): ") == '1'
    }

    # Filter out the augmentations that are not selected
    selected_augmentations = [key for key, value in augmentations.items() if value]
    return selected_augmentations


# Main function to handle inputs
if __name__ == "__main__":
    input_dir = 'dataset_artworks_aug/train'

    augmentations = prompt_for_augmentations()


    # Process the images with randomly mixed augmentations
    process_images(input_dir, augmentations)
