import os
import requests
import json
import time
from PIL import Image
import pandas as pd
from io import BytesIO

# Optional external libraries: make these imports non-fatal so scripts can run
try:
    import flickrapi
except Exception:
    flickrapi = None

try:
    import kaggle
except Exception:
    kaggle = None

import numpy as np

class FootballDataCollector:
    def __init__(self, output_dir='data/images/', captions_file='data/captions.txt'):
        self.output_dir = output_dir
        self.captions_file = captions_file
        os.makedirs(output_dir, exist_ok=True)

        # Load API keys from environment or config
        self.flickr_api_key = os.getenv('FLICKR_API_KEY', 'your_flickr_key')
        self.unsplash_access_key = os.getenv('UNSPLASH_ACCESS_KEY', 'your_unsplash_key')

    def download_kaggle_dataset(self, dataset_name, download_path='data/kaggle/'):
        """Download dataset from Kaggle"""
        if kaggle is None:
            print("Kaggle package not installed or not available in this environment. Skipping Kaggle download.")
            return False

        try:
            os.makedirs(download_path, exist_ok=True)
            kaggle.api.dataset_download_files(dataset_name, path=download_path, unzip=True)
            print(f"Downloaded {dataset_name} to {download_path}")
            return True
        except Exception as e:
            print(f"Error downloading from Kaggle: {e}")
            return False

    def collect_from_flickr(self, keywords=['football', 'soccer', 'football match'], num_images=100):
        """Collect images from Flickr API"""
        if flickrapi is None:
            print("FlickrAPI package not installed or not available in this environment. Skipping Flickr collection.")
            return

        try:
            flickr = flickrapi.FlickrAPI(self.flickr_api_key, 'your_flickr_secret', format='parsed-json')

            captions = []
            image_count = 0

            for keyword in keywords:
                if image_count >= num_images:
                    break

                photos = flickr.photos.search(text=keyword, per_page=50, sort='relevance')

                for photo in photos['photos']['photo']:
                    if image_count >= num_images:
                        break

                    try:
                        # Get photo info
                        info = flickr.photos.getInfo(photo_id=photo['id'], secret=photo['secret'])

                        # Get image URL
                        sizes = flickr.photos.getSizes(photo_id=photo['id'])
                        url = sizes['sizes']['size'][-1]['source']  # Largest size

                        # Download image
                        response = requests.get(url)
                        img = Image.open(BytesIO(response.content))

                        # Save image
                        image_id = f"flickr_{photo['id']}"
                        img_path = os.path.join(self.output_dir, f"{image_id}.jpg")
                        img.save(img_path)

                        # Create caption
                        title = info['photo'].get('title', {}).get('_content', '')
                        description = info['photo'].get('description', {}).get('_content', '')

                        caption = f"{keyword} {title} {description}".strip()
                        if not caption:
                            caption = f"a {keyword} match scene"

                        captions.append(f"{image_id}\t{caption}")
                        image_count += 1

                        print(f"Downloaded {image_count}/{num_images}: {image_id}")

                        time.sleep(0.5)  # Rate limiting

                    except Exception as e:
                        print(f"Error processing photo {photo.get('id', '')}: {e}")
                        continue

            # Save captions
            with open(self.captions_file, 'a', encoding='utf-8') as f:
                for caption in captions:
                    f.write(caption + '\n')

            print(f"Collected {len(captions)} images from Flickr")

        except Exception as e:
            print(f"Flickr collection failed: {e}")

    def collect_from_unsplash(self, query='football', num_images=50):
        """Collect images from Unsplash API"""
        try:
            url = f"https://api.unsplash.com/search/photos?query={query}&per_page={num_images}"
            headers = {'Authorization': f'Client-ID {self.unsplash_access_key}'}

            response = requests.get(url, headers=headers)
            data = response.json()

            captions = []

            for i, photo in enumerate(data['results']):
                try:
                    # Download image
                    img_url = photo['urls']['regular']
                    response = requests.get(img_url)
                    img = Image.open(BytesIO(response.content))

                    # Save image
                    image_id = f"unsplash_{photo['id']}"
                    img_path = os.path.join(self.output_dir, f"{image_id}.jpg")
                    img.save(img_path)

                    # Create caption
                    description = photo.get('description', '') or photo.get('alt_description', '')
                    caption = f"{query} {description}".strip() or f"a {query} scene"

                    captions.append(f"{image_id}\t{caption}")
                    print(f"Downloaded Unsplash image {i+1}/{num_images}: {image_id}")

                    time.sleep(0.5)

                except Exception as e:
                    print(f"Error processing Unsplash photo: {e}")
                    continue

            # Save captions
            with open(self.captions_file, 'a', encoding='utf-8') as f:
                for caption in captions:
                    f.write(caption + '\n')

            print(f"Collected {len(captions)} images from Unsplash")

        except Exception as e:
            print(f"Unsplash collection failed: {e}")

    def create_synthetic_captions(self, num_samples=100):
        """Create synthetic captions for existing images"""
        football_terms = [
            'football player', 'soccer match', 'goal kick', 'corner kick', 'penalty kick',
            'football stadium', 'soccer ball', 'team celebration', 'player dribbling',
            'goalkeeper save', 'free kick', 'throw in', 'offside', 'yellow card',
            'red card', 'substitution', 'coach', 'referee', 'fans cheering'
        ]

        actions = [
            'running with the ball', 'shooting at goal', 'passing the ball',
            'tackling opponent', 'jumping for header', 'celebrating goal',
            'defending position', 'taking free kick', 'warming up', 'training'
        ]

        captions = []
        for i in range(num_samples):
            term = np.random.choice(football_terms)
            action = np.random.choice(actions)
            caption = f"{term} {action}"
            captions.append(f"synthetic_{i+1}\t{caption}")

        with open(self.captions_file, 'a', encoding='utf-8') as f:
            for caption in captions:
                f.write(caption + '\n')

        print(f"Created {num_samples} synthetic captions")

    def collect_all_sources(self):
        """Collect data from all available sources"""
        print("Starting data collection from multiple sources...")

        # 1. Try Kaggle datasets
        kaggle_datasets = [
            'jorgebuenoperez/football-players',
            'idoyoabraham/football-match-images'
        ]

        for dataset in kaggle_datasets:
            print(f"\nTrying Kaggle dataset: {dataset}")
            self.download_kaggle_dataset(dataset)

        # 2. Collect from Flickr
        print("\nCollecting from Flickr...")
        self.collect_from_flickr(num_images=200)

        # 3. Collect from Unsplash
        print("\nCollecting from Unsplash...")
        self.collect_from_unsplash(num_images=100)

        # 4. Create synthetic captions
        print("\nCreating synthetic captions...")
        self.create_synthetic_captions(500)

        print("\nData collection completed!")
        print(f"Check {self.captions_file} for captions and {self.output_dir} for images")

if __name__ == "__main__":
    collector = FootballDataCollector()
    collector.collect_all_sources()