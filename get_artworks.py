import asyncio
import aiosonic
import os
import logging
from PIL import Image
from io import BytesIO
import requests


# Configurable settings
READ_FULL_JSON = True  # Set to False to limit to 40 images for testing
MAX_REQUESTS_PER_SECOND = 15
BASE_FOLDER = 'dataset_artworks'
RETRY_LIMIT = 3
TIMEOUT = aiosonic.Timeouts(
    sock_read=10,
    sock_connect=3
)
IMG_SIZE = (56, 56)  # Define your target image size here

# Setup logging
logging.basicConfig(filename='download_log_artworks.txt', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Additional log file for failed downloads
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

# Create the base folder if it doesn't exist
os.makedirs(BASE_FOLDER, exist_ok=True)

successful_downloads = 0
total_images = sum(len(urls) for urls in data.values())
client = aiosonic.HTTPClient()
download_lock = asyncio.Lock()  # Lock for atomic operations on successful_downloads


# Download and save an image with resizing, logging failures
async def download_and_resize_image(url: str, image_id: str):
    retries = 0
    while retries < RETRY_LIMIT:
        try:
            response = await client.get(url, timeouts=TIMEOUT)
            if response.status_code == 200:
                image_folder = os.path.join(BASE_FOLDER, str(image_id))
                os.makedirs(image_folder, exist_ok=True)
                image_path = os.path.join(image_folder, f'image_{url.split("/")[-1]}')

                # Open image from bytes, resize, and save
                img = Image.open(BytesIO(await response.content()))
                img = img.resize(IMG_SIZE)
                img.save(image_path)

                logging.info(f"Downloaded and resized image for ID {image_id}")

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
            tasks = [download_and_resize_image(url, image_id) for url in batch]
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
