import asyncio
import aiohttp
import os
import logging
from tqdm import tqdm
import requests
from aiohttp import ClientTimeout

# Configurable settings
READ_FULL_JSON = True  # Set to False to limit to 40 images for testing
MAX_REQUESTS_PER_SECOND = 20
BASE_FOLDER = 'dataset_artworks'
RETRY_LIMIT = 3
TIMEOUT = ClientTimeout(total=30)

# Setup logging
logging.basicConfig(filename='download_log_artworks.txt', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load data
raw_data = requests.get('https://db.ygoprodeck.com/api/v7/cardinfo.php').json()
cards = [card for card in raw_data['data'] if card['type'] not in ['Token', 'Skill Card']]

# Create dictionary of image URLs per card ID
data = {card['id']: [img['image_url_cropped'] for img in card['card_images']] for card in cards}

# Limit data for testing if READ_FULL_JSON is False
if not READ_FULL_JSON:
    data = {k: data[k] for k in list(data)[:40]}

# Create the base folder if it doesn't exist
os.makedirs(BASE_FOLDER, exist_ok=True)

successful_downloads = 0

# Download and save an image without resizing
async def download_image(session, url: str, image_id: str, progress_bar):
    global successful_downloads
    retries = 0
    while retries < RETRY_LIMIT:
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    image_folder = os.path.join(BASE_FOLDER, str(image_id))
                    os.makedirs(image_folder, exist_ok=True)
                    image_path = os.path.join(image_folder, f'image_{url.split("/")[-1]}')

                    with open(image_path, 'wb') as f:
                        f.write(await response.read())

                    logging.info(f"Downloaded image for ID {image_id}")
                    successful_downloads += 1
                    break
                else:
                    logging.error(f"Failed to download {url} for ID {image_id}, status: {response.status}")
        except Exception as e:
            retries += 1
            logging.error(f"Error downloading {url} for ID {image_id}, attempt {retries}: {e}")
        finally:
            progress_bar.update(1)

# Download multiple images with rate limiting
async def download_images(id_url_dict):
    async with aiohttp.ClientSession(timeout=TIMEOUT) as session:
        total_images = sum(len(urls) for urls in id_url_dict.values())
        with tqdm(total=total_images, desc="Downloading images") as progress_bar:
            for image_id, url_list in id_url_dict.items():
                for i in range(0, len(url_list), MAX_REQUESTS_PER_SECOND):
                    batch = url_list[i:i + MAX_REQUESTS_PER_SECOND]
                    tasks = [download_image(session, url, image_id, progress_bar) for url in batch]
                    await asyncio.gather(*tasks)
                    if i + MAX_REQUESTS_PER_SECOND < len(url_list):
                        await asyncio.sleep(1)

# Main function to initiate download
async def main():
    global successful_downloads
    await download_images(data)
    logging.info(f"Successfully downloaded {successful_downloads} images out of {sum(len(urls) for urls in data.values())}.")

# Run the asyncio event loop
if __name__ == "__main__":
    asyncio.run(main())
