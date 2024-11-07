import asyncio
import aiohttp
import os
import logging
import time
from tqdm import tqdm
from aiohttp import ClientTimeout
from PIL import Image
import io
import json
import requests

# Configurable settings
READ_FULL_JSON = False  # Set to True to read the entire JSON file
MAX_REQUESTS_PER_SECOND = 20
img_size = 28
BASE_FOLDER = f'dataset_artworks_{img_size}'

# Setup logging
log_file = f'download_log_artworks_{img_size}.txt'
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load data
raw_data = requests.get('https://db.ygoprodeck.com/api/v7/cardinfo.php').json()

cards = [card for card in raw_data['data'] if card['type'] not in ['Token', 'Skill Card']]

# Card image_url list
url_list = [img['image_url_cropped'] for card in cards for img in card['card_images']]

data = {}
for card in cards:
    if len(card['card_images']) != 1:
        data[card['id']] = [url['image_url_cropped'] for url in card['card_images']]
    else:
        data[card['id']] = [card['card_images'][0]['image_url_cropped']]

# Limit to the first 100 objects if READ_FULL_JSON is False
if not READ_FULL_JSON:
    data = {k: data[k] for k in list(data)[:40]}

# Create the base folder if it doesn't exist
os.makedirs(BASE_FOLDER, exist_ok=True)

successful_downloads = 0
RETRY_LIMIT = 3
TIMEOUT = ClientTimeout(total=30)

# Enhanced function to download and save an image
async def download_image(session, url: str, image_id: str, progress_bar):
    global successful_downloads
    retries = 0

    while retries < RETRY_LIMIT:
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    # Read image data
                    image_data = await response.read()
                    
                    # Resize image
                    try:
                        image = Image.open(io.BytesIO(image_data)).convert("RGB")
                        image = image.resize((img_size, img_size))

                        # Save the image
                        image_folder = os.path.join(BASE_FOLDER, str(image_id))
                        os.makedirs(image_folder, exist_ok=True)
                        image_filename = f'image_{url.split("/")[-1]}'
                        image_path = os.path.join(image_folder, image_filename)
                        image.save(image_path, format="JPEG")

                        logging.info(f"Downloaded and resized image {image_filename} for ID {image_id}")
                        successful_downloads += 1
                        break  # Exit retry loop on success

                    except Exception as e:
                        logging.error(f"Failed to process image for {image_id} from {url}: {e}")
                else:
                    logging.error(f"Failed to download {url} for {image_id}, status code: {response.status}")
        except Exception as e:
            retries += 1
            logging.error(f"Error downloading {url} for {image_id}, attempt {retries}: {e}")
            if retries == RETRY_LIMIT:
                logging.error(f"Giving up on {url} for {image_id} after {RETRY_LIMIT} retries.")
        finally:
            progress_bar.update(1)

# Download multiple images with rate limiting
async def download_images(id_url_dict: dict):
    async with aiohttp.ClientSession(timeout=TIMEOUT, connector=aiohttp.TCPConnector(ssl=False)) as session:
        total_images = sum(len(urls) for urls in id_url_dict.values())
        with tqdm(total=total_images, desc="Downloading images") as progress_bar:
            for image_id, url_list in id_url_dict.items():
                for i in range(0, len(url_list), MAX_REQUESTS_PER_SECOND):
                    # Batch to respect rate limit
                    batch = url_list[i:i + MAX_REQUESTS_PER_SECOND]
                    tasks = [download_image(session, url, image_id, progress_bar) for url in batch]

                    # Run tasks concurrently
                    await asyncio.gather(*tasks)

                    # Sleep to respect rate limit
                    if i + MAX_REQUESTS_PER_SECOND < len(url_list):
                        await asyncio.sleep(1)

# Main function to initiate download
async def main():
    global successful_downloads

    start_time = time.time()

    # Start downloading images
    await download_images(data)

    total_time = time.time() - start_time
    logging.info(f"Successfully downloaded {successful_downloads} out of {sum(len(urls) for urls in data.values())} images.")
    logging.info(f"Total run time: {total_time:.2f} seconds.")

# Run the asyncio event loop
if __name__ == "__main__":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
