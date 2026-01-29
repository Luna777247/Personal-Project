import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pickle
from tensorflow.keras.applications.resnet50 import ResNet50

# Cached feature extractor to avoid repeated heavy instantiation
_RESNET_EXTRACTOR = None

def get_resnet50(weights='imagenet'):
    """Return a cached ResNet50 feature extractor (include_top=False, pooling='avg').

    This avoids creating multiple copies of the model during feature extraction.
    """
    global _RESNET_EXTRACTOR
    if _RESNET_EXTRACTOR is None:
        _RESNET_EXTRACTOR = ResNet50(weights=weights, include_top=False, pooling='avg')
    return _RESNET_EXTRACTOR

def load_image(image_path, target_size=(224, 224)):
    """Load and preprocess image"""
    try:
        img = Image.open(image_path)
        img = img.resize(target_size)
        img_array = np.array(img)

        # Convert to RGB if grayscale
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=-1)

        return img_array
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def display_image_with_caption(image_path, caption, figsize=(10, 8)):
    """Display image with generated caption"""
    img = load_image(image_path)

    if img is not None:
        plt.figure(figsize=figsize)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f'Generated Caption:\n{caption}', fontsize=14, pad=20)
        plt.tight_layout()
        plt.show()
    else:
        print(f"Could not load image: {image_path}")

def create_sample_captions():
    """Create sample football captions for demonstration"""
    sample_captions = [
        "football player kicking the ball towards the goal",
        "soccer match with players running on the field",
        "crowd cheering as player scores a goal",
        "football goalkeeper diving to save the ball",
        "team celebrating victory after scoring",
        "referee showing yellow card to player",
        "football stadium filled with fans",
        "player dribbling past defenders",
        "corner kick taken by midfielder",
        "penalty kick being taken by striker"
    ]
    return sample_captions

def save_captions_to_file(captions, filename):
    """Save captions to text file"""
    with open(filename, 'w', encoding='utf-8') as f:
        for i, caption in enumerate(captions):
            f.write(f"image_{i+1}\t{caption}\n")
    print(f"Captions saved to {filename}")

def create_demo_dataset(num_samples=100):
    """Create demo dataset for testing"""
    os.makedirs('data/images', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)

    # Create sample captions
    captions = create_sample_captions()
    repeated_captions = captions * (num_samples // len(captions) + 1)
    demo_captions = repeated_captions[:num_samples]

    # Save captions
    save_captions_to_file(demo_captions, 'data/captions.txt')

    # Create dummy features (for demo)
    features = {}
    for i in range(num_samples):
        # Random features simulating ResNet50 output
        features[f'image_{i+1}'] = np.random.rand(2048)

    # Save features
    with open('data/processed/features.pkl', 'wb') as f:
        pickle.dump(features, f)

    # Create dummy tokenizer
    from tensorflow.keras.preprocessing.text import Tokenizer
    tokenizer = Tokenizer(num_words=5000, oov_token='<unk>')
    tokenizer.fit_on_texts(demo_captions)

    with open('data/processed/tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)

    print(f"Demo dataset created with {num_samples} samples")

def plot_caption_statistics(captions, save_path=None):
    """Plot statistics about captions"""
    word_counts = [len(cap.split()) for cap in captions]

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(word_counts, bins=20, edgecolor='black')
    plt.title('Caption Length Distribution')
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    # Word frequency
    from collections import Counter
    all_words = ' '.join(captions).lower().split()
    word_freq = Counter(all_words).most_common(20)

    words, counts = zip(*word_freq)
    plt.barh(words, counts)
    plt.title('Top 20 Most Frequent Words')
    plt.xlabel('Frequency')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def generate_model_summary_report(model, save_path='results/model_summary.txt'):
    """Generate model summary report"""
    with open(save_path, 'w') as f:
        f.write("Football Captioning Model Summary\n")
        f.write("=" * 40 + "\n\n")

        # Model architecture
        f.write("Model Architecture:\n")
        model.summary(print_fn=lambda x: f.write(x + '\n'))

        f.write("\nModel Configuration:\n")
        f.write(f"Total Parameters: {model.count_params()}\n")

        trainable_params = sum([layer.count_params() for layer in model.layers if layer.trainable])
        f.write(f"Trainable Parameters: {trainable_params}\n")

        non_trainable_params = sum([layer.count_params() for layer in model.layers if not layer.trainable])
        f.write(f"Non-trainable Parameters: {non_trainable_params}\n")

    print(f"Model summary saved to {save_path}")

def create_project_structure():
    """Create complete project structure"""
    directories = [
        'data/images',
        'data/processed',
        'models',
        'notebooks',
        'src',
        'results'
    ]

    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")

    # Create __init__.py files
    init_files = ['src/__init__.py']
    for file in init_files:
        with open(file, 'w') as f:
            f.write("# Football Captioning Project\n")
        print(f"Created file: {file}")

if __name__ == "__main__":
    create_project_structure()
    create_demo_dataset(50)
    print("Project setup completed!")