import os
import random
from train import load_processed_data, FootballCaptionModel


def subset_captions(captions_dict, max_images):
    keys = list(captions_dict.keys())[:max_images]
    return {k: captions_dict[k] for k in keys}


def main():
    print("Starting smoke-test training (very small run)")

    vocab, word_to_idx, idx_to_word, train_captions, val_captions = load_processed_data()

    # Use a small subset to keep runtime short
    train_captions_small = subset_captions(train_captions, max_images=30)
    val_captions_small = subset_captions(val_captions, max_images=10)

    vocab_size = len(vocab)
    max_length = 25

    print(f"Vocab size: {vocab_size}")
    print(f"Train images (subset): {len(train_captions_small)}")
    print(f"Val images (subset): {len(val_captions_small)}")

    model = FootballCaptionModel(vocab_size=vocab_size, max_length=max_length)
    model.build_model()

    # Extract features (this will use ResNet50 and may download weights if needed)
    print("Extracting train features (small subset)...")
    train_features = model.extract_all_features('data/processed/images/', train_captions_small)
    print("Extracting val features (small subset)...")
    val_features = model.extract_all_features('data/processed/images/', val_captions_small)

    # Create sequences
    X1_train, X2_train, y_train = model.create_sequences(train_captions_small, train_features, word_to_idx)
    X1_val, X2_val, y_val = model.create_sequences(val_captions_small, val_features, word_to_idx)

    print(f"Training sequences: {X1_train.shape[0]}")
    print(f"Validation sequences: {X1_val.shape[0]}")

    if X1_train.shape[0] == 0:
        print("No training sequences generated. Aborting smoke test.")
        return

    # Small training run
    history = model.train(
        X1_train, X2_train, y_train,
        X1_val, X2_val, y_val,
        epochs=2,
        batch_size=8,
        model_save_path='models/smoke_football_caption_model.h5'
    )

    # Save minimal artifacts
    model.build_inference_models()
    model.save_models(save_dir='models/smoke_models')

    print("Smoke-test training finished.")


if __name__ == '__main__':
    main()
