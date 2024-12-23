import asyncio
import aiosonic
import os
import logging
from PIL import Image
from io import BytesIO
import requests
import time

# Configurable settings
MAX_REQUESTS_PER_SECOND = 15
RETRY_LIMIT = 3
TIMEOUT = aiosonic.Timeouts(
    sock_read=10,
    sock_connect=3
)

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
# data = {10000000: data[10000000]}  # Example for testing
# print(data)

# Limit data for testing if READ_FULL_JSON is False
READ_FULL_JSON = True  # Set to True to process all images
if not READ_FULL_JSON:
    data = {k: data[k] for k in list(data)[:500]}

# Parent folder structure
DATASET_FOLDER = 'dataset'
BASE_FOLDER = os.path.join(DATASET_FOLDER, 'test')
TRAINING_FOLDER = os.path.join(DATASET_FOLDER, 'train')

# Create parent and subfolders
os.makedirs(DATASET_FOLDER, exist_ok=True)
os.makedirs(BASE_FOLDER, exist_ok=True)
os.makedirs(TRAINING_FOLDER, exist_ok=True)

successful_downloads = 0
total_images = sum(len(urls) for urls in data.values())
client = aiosonic.HTTPClient()
download_lock = asyncio.Lock()  # Lock for atomic operations on successful_downloads

# Initialize a dictionary to keep track of counters for each image ID
image_counters = {}

# Save augmentations of an image with counters for duplicate names
def save_augmentations(image: Image.Image, base_path: str, img_name: str, image_id: str):
    if image_id not in image_counters:
        image_counters[image_id] = 0

    image_counters[image_id] += 1
    counter = image_counters[image_id]

    base_name = img_name
    base_name_with_counter = f"{base_name}_{counter}"

    # Save original and augmented images
    original_path = os.path.join(base_path, f"{base_name_with_counter}_original.jpg")
    image.save(original_path)

    # Rotate 90 degrees to the left
    left_90 = image.rotate(90, resample=Image.BICUBIC, expand=True)
    left_90.save(os.path.join(base_path, f"{base_name_with_counter}_rotate90_left.jpg"))

    # Rotate 90 degrees to the right
    right_90 = image.rotate(-90, resample=Image.BICUBIC, expand=True)
    right_90.save(os.path.join(base_path, f"{base_name_with_counter}_rotate90_right.jpg"))

    # Rotate 180 degrees
    rotated_180 = image.rotate(180, resample=Image.BICUBIC, expand=True)
    rotated_180.save(os.path.join(base_path, f"{base_name_with_counter}_rotate180.jpg"))

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

                # Save augmented images in dataset_artworks_training
                training_image_folder = os.path.join(TRAINING_FOLDER, str(image_id))
                os.makedirs(training_image_folder, exist_ok=True)

                # Open image from bytes
                img = Image.open(BytesIO(await response.content())).convert("RGB")

                img_name = ''.join(filter(str.isdigit, url.split('/')[-1]))
                # Save original image in dataset_artworks
                img.save(os.path.join(image_folder, f"{img_name}_original.jpg"))

                # Save augmented images in dataset_artworks_training
                save_augmentations(img, training_image_folder, img_name, str(image_id))

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
    start = time.time()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    end = time.time()
    print(f"Total time taken: {end - start} seconds")
