import os
import re
import json
import pandas as pd
from PIL import Image
import re
from collections import Counter
import numpy as np

class DataPreprocessor:
    def __init__(self, data_dir='data/', output_dir='data/processed/'):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.images_dir = os.path.join(data_dir, 'images/')
        self.captions_file = os.path.join(data_dir, 'captions.txt')

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'images/'), exist_ok=True)

        # Lightweight stop words (small subset) to avoid external NLTK downloads
        self.stop_words = set([
            'a', 'an', 'the', 'and', 'or', 'in', 'on', 'at', 'to', 'of', 'for', 'with',
            'is', 'are', 'was', 'were', 'be', 'been', 'has', 'have', 'had', 'this', 'that',
            'it', 'as', 'by', 'from', 'but', 'not', 'which', 'will', 'who'
        ])

    def load_captions(self):
        """Load captions from file"""
        captions = {}
        try:
            with open(self.captions_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        image_id, caption = parts[0], ' '.join(parts[1:])
                        if image_id not in captions:
                            captions[image_id] = []
                        captions[image_id].append(caption)
        except FileNotFoundError:
            print(f"Captions file not found: {self.captions_file}")
            return {}

        print(f"Loaded {len(captions)} images with captions")
        return captions

    def clean_caption(self, caption):
        """Clean and normalize caption text"""
        # Convert to lowercase
        caption = caption.lower()

        # Remove special characters and extra whitespace
        caption = re.sub(r'[^\w\s]', '', caption)
        caption = re.sub(r'\s+', ' ', caption).strip()
        # Simple regex-based tokenizer and stop-word removal
        tokens = re.findall(r"\w+", caption)
        tokens = [token for token in tokens if token not in self.stop_words and len(token) > 1]

        return ' '.join(tokens)

    def validate_image(self, image_path):
        """Validate if image can be opened and processed"""
        try:
            with Image.open(image_path) as img:
                img.verify()
                # Check if image is RGB
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                return True, img.size
        except Exception as e:
            print(f"Invalid image {image_path}: {e}")
            return False, None

    def preprocess_images(self, captions_dict, target_size=(224, 224)):
        """Preprocess images and filter valid ones"""
        valid_captions = {}
        image_sizes = {}

        for image_id, captions in captions_dict.items():
            image_path = os.path.join(self.images_dir, f"{image_id}.jpg")

            if os.path.exists(image_path):
                is_valid, size = self.validate_image(image_path)
                if is_valid:
                    # Resize and save processed image
                    try:
                        with Image.open(image_path) as img:
                            if img.mode != 'RGB':
                                img = img.convert('RGB')

                            # Resize maintaining aspect ratio
                            img.thumbnail(target_size, Image.Resampling.LANCZOS)

                            # Create new image with target size and paste resized image
                            new_img = Image.new('RGB', target_size, (0, 0, 0))
                            x = (target_size[0] - img.size[0]) // 2
                            y = (target_size[1] - img.size[1]) // 2
                            new_img.paste(img, (x, y))

                            # Save processed image
                            processed_path = os.path.join(self.output_dir, 'images/', f"{image_id}.jpg")
                            new_img.save(processed_path, 'JPEG', quality=95)

                            # Clean captions
                            clean_captions = [self.clean_caption(caption) for caption in captions]
                            clean_captions = [c for c in clean_captions if len(c.split()) >= 3]  # At least 3 words

                            if clean_captions:
                                valid_captions[image_id] = clean_captions
                                image_sizes[image_id] = size

                    except Exception as e:
                        print(f"Error processing {image_id}: {e}")
                        continue

        print(f"Processed {len(valid_captions)} valid images")
        return valid_captions, image_sizes

    def build_vocabulary(self, captions_dict, min_freq=5):
        """Build vocabulary from captions"""
        all_words = []
        for captions in captions_dict.values():
            for caption in captions:
                all_words.extend(caption.split())

        word_counts = Counter(all_words)
        vocab = [word for word, count in word_counts.items() if count >= min_freq]

        # Add special tokens
        vocab = ['<pad>', '<start>', '<end>', '<unk>'] + vocab

        word_to_idx = {word: idx for idx, word in enumerate(vocab)}
        idx_to_word = {idx: word for word, idx in word_to_idx.items()}

        print(f"Built vocabulary with {len(vocab)} words (min_freq={min_freq})")

        return vocab, word_to_idx, idx_to_word

    def split_dataset(self, captions_dict, train_ratio=0.7, val_ratio=0.2):
        """Split dataset into train, validation, and test sets"""
        image_ids = list(captions_dict.keys())
        np.random.shuffle(image_ids)

        n_total = len(image_ids)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        train_ids = image_ids[:n_train]
        val_ids = image_ids[n_train:n_train + n_val]
        test_ids = image_ids[n_train + n_val:]

        splits = {
            'train': {id: captions_dict[id] for id in train_ids},
            'val': {id: captions_dict[id] for id in val_ids},
            'test': {id: captions_dict[id] for id in test_ids}
        }

        print(f"Dataset split: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")
        return splits

    def save_processed_data(self, splits, vocab_data):
        """Save processed data to files"""
        vocab, word_to_idx, idx_to_word = vocab_data

        # Save vocabulary
        vocab_file = os.path.join(self.output_dir, 'vocab.json')
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump({
                'vocab': vocab,
                'word_to_idx': word_to_idx,
                'idx_to_word': idx_to_word
            }, f, ensure_ascii=False, indent=2)

        # Save splits
        for split_name, captions_dict in splits.items():
            split_file = os.path.join(self.output_dir, f'{split_name}_captions.json')
            with open(split_file, 'w', encoding='utf-8') as f:
                json.dump(captions_dict, f, ensure_ascii=False, indent=2)

        # Save dataset info
        info = {
            'total_images': sum(len(captions) for captions in splits.values()),
            'vocab_size': len(vocab),
            'splits': {name: len(captions) for name, captions in splits.items()}
        }

        info_file = os.path.join(self.output_dir, 'dataset_info.json')
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=2)

        print(f"Saved processed data to {self.output_dir}")

    def create_caption_statistics(self, captions_dict):
        """Create statistics about captions"""
        caption_lengths = []
        word_counts = []

        for captions in captions_dict.values():
            for caption in captions:
                words = caption.split()
                caption_lengths.append(len(words))
                word_counts.extend(words)

        stats = {
            'total_captions': sum(len(captions) for captions in captions_dict.values()),
            'unique_images': len(captions_dict),
            'avg_caption_length': np.mean(caption_lengths),
            'max_caption_length': max(caption_lengths),
            'min_caption_length': min(caption_lengths),
            'total_words': len(word_counts),
            'unique_words': len(set(word_counts))
        }

        stats_file = os.path.join(self.output_dir, 'caption_stats.json')
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        print("Caption statistics:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}: {value}")

        return stats

    def preprocess_all(self):
        """Run complete preprocessing pipeline"""
        print("Starting data preprocessing...")

        # Load captions
        captions_dict = self.load_captions()
        if not captions_dict:
            print("No captions found. Please run data collection first.")
            return

        # Preprocess images and filter captions
        valid_captions, image_sizes = self.preprocess_images(captions_dict)

        # Build vocabulary
        vocab_data = self.build_vocabulary(valid_captions)

        # Split dataset
        splits = self.split_dataset(valid_captions)

        # Create statistics
        self.create_caption_statistics(valid_captions)

        # Save processed data
        self.save_processed_data(splits, vocab_data)

        print("Data preprocessing completed!")
        print(f"Processed data saved to: {self.output_dir}")

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    preprocessor.preprocess_all()