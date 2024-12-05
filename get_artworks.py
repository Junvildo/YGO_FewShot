import asyncio
import aiosonic
import os
import logging
from PIL import Image, ImageOps
from io import BytesIO
import requests
import random

# Configurable settings
READ_FULL_JSON = True  # Set to False to limit to 40 images for testing
MAX_REQUESTS_PER_SECOND = 15
BASE_FOLDER = 'dataset_artworks'
TRAINING_FOLDER = 'dataset_artworks_training'
TRAIN_FOLDER = os.path.join(TRAINING_FOLDER, 'train')
TEST_FOLDER = os.path.join(TRAINING_FOLDER, 'test')
RETRY_LIMIT = 3
TIMEOUT = aiosonic.Timeouts(
    sock_read=10,
    sock_connect=3
)
SPLIT_RATIO = 0.8

# Setup logging
logging.basicConfig(filename='download_log_artworks.txt', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

FAILED_LOG_FILE = 'failed_cards.txt'

# Ensure the failed log file is empty at the start
with open(FAILED_LOG_FILE, 'w') as f:
    f.write("Failed downloads:\n")

# Load data
raw_data = requests.get('https://db.ygoprodeck.com/api/v7/cardinfo.php').json()
cards = [card for card in raw_data['data'] if card['type'] not in ['Token', 'Skill Card']]
data = {card['id']: [img['image_url_cropped'] for img in card['card_images']] for card in cards}

# Limit data for testing if READ_FULL_JSON is False
if not READ_FULL_JSON:
    data = {k: data[k] for k in list(data)[:40]}

# Create the base folders
os.makedirs(BASE_FOLDER, exist_ok=True)
os.makedirs(TRAINING_FOLDER, exist_ok=True)
os.makedirs(TRAIN_FOLDER, exist_ok=True)
os.makedirs(TEST_FOLDER, exist_ok=True)

successful_downloads = 0
total_images = sum(len(urls) for urls in data.values())
client = aiosonic.HTTPClient()
download_lock = asyncio.Lock()  # Lock for atomic operations on successful_downloads


# Save augmentations of an image
def save_augmentations(image: Image.Image, base_path: str, base_name: str):
    original_path = os.path.join(base_path, f"{base_name}_original.jpg")
    image.save(original_path)

    # Rotate 90 degrees to the left
    left_90 = image.rotate(90, resample=Image.BICUBIC, expand=True)
    left_90.save(os.path.join(base_path, f"{base_name}_rotate90_left.jpg"))

    # Rotate 90 degrees to the right
    right_90 = image.rotate(-90, resample=Image.BICUBIC, expand=True)
    right_90.save(os.path.join(base_path, f"{base_name}_rotate90_right.jpg"))

    # Rotate 180 degrees
    rotated_180 = image.rotate(180, resample=Image.BICUBIC, expand=True)
    rotated_180.save(os.path.join(base_path, f"{base_name}_rotate180.jpg"))

    # Vertical flip
    v_flip = ImageOps.flip(image)
    v_flip.save(os.path.join(base_path, f"{base_name}_vflip.jpg"))


# Download and save an image without resizing, logging failures
async def download_and_save_image(url: str, image_id: str):
    retries = 0
    while retries < RETRY_LIMIT:
        try:
            response = await client.get(url, timeouts=TIMEOUT)
            if response.status_code == 200:
                # Save original image in dataset_artworks folder
                image_folder = os.path.join(BASE_FOLDER, str(image_id))
                os.makedirs(image_folder, exist_ok=True)

                # Determine if this image belongs to train or test
                subfolder = TRAIN_FOLDER if random.random() < SPLIT_RATIO else TEST_FOLDER
                training_image_folder = os.path.join(subfolder, str(image_id))
                os.makedirs(training_image_folder, exist_ok=True)

                # Open image from bytes
                img = Image.open(BytesIO(await response.content())).convert("RGB")
                
                # Save original image in dataset_artworks
                img.save(os.path.join(image_folder, f"original.jpg"))

                # Save augmented images in dataset_artworks_training
                save_augmentations(img, training_image_folder, f"image_{url.split('/')[-1]}")

                logging.info(f"Downloaded and augmented image for ID {image_id}")

                # Atomically increment successful_downloads
                async with download_lock:
                    global successful_downloads
                    successful_downloads += 1
                break  # Exit the retry loop upon success
            else:
                logging.error(f"Failed to download {url} for ID {image_id}, status: {response.status_code}")
                retries += 1
        except Exception as e:
            retries += 1
            logging.error(f"Error downloading {url} for ID {image_id}, attempt {retries}: {e}")
    
    # If all retry attempts fail, log to failed_cards.txt once
    if retries == RETRY_LIMIT:
        logging.error(f"Max retries reached for ID {image_id} at URL: {url}")
        with open(FAILED_LOG_FILE, 'a') as f:
            f.write(f"ID {image_id}, URL: {url}\n")


# Download multiple images with rate limiting
async def download_images(id_url_dict):
    for image_id, url_list in id_url_dict.items():
        for i in range(0, len(url_list), MAX_REQUESTS_PER_SECOND):
            batch = url_list[i:i + MAX_REQUESTS_PER_SECOND]
            tasks = [download_and_save_image(url, image_id) for url in batch]
            await asyncio.gather(*tasks)
            if i + MAX_REQUESTS_PER_SECOND < len(url_list):
                await asyncio.sleep(1)

    # Display summary of downloaded images
    print(f"Downloaded {successful_downloads}/{total_images} images successfully.")


# Main function to initiate download
async def main():
    await download_images(data)
    logging.info(f"Successfully downloaded {successful_downloads} images out of {total_images}.")


# Run the asyncio event loop
if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
