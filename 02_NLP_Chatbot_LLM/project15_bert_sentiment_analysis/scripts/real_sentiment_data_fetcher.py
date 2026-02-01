#!/usr/bin/env python3
"""
Data Fetcher for BERT Sentiment Analysis Project
Downloads and prepares sentiment analysis datasets for training
"""

import os
import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import json
from tqdm import tqdm

def download_imdb_dataset():
    """Download IMDB dataset and convert to 3-class sentiment"""
    print("Downloading IMDB dataset...")

    # Load IMDB dataset
    dataset = load_dataset("imdb")

    # Convert to DataFrame
    train_df = pd.DataFrame({
        'text': dataset['train']['text'],
        'label': dataset['train']['label']  # 0=negative, 1=positive
    })

    test_df = pd.DataFrame({
        'text': dataset['test']['text'],
        'label': dataset['test']['label']
    })

    print(f"Train samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")

    return train_df, test_df

def create_three_class_dataset(train_df, test_df):
    """Create 3-class dataset by sampling neutral reviews"""
    print("Creating 3-class sentiment dataset...")

    # Sample some neutral reviews from positive and negative
    # We'll take reviews that are shorter and less extreme as "neutral"

    # Get neutral samples from train
    neutral_train = []
    for _, row in train_df.iterrows():
        text = row['text']
        # Consider shorter reviews as potentially neutral
        if len(text.split()) < 50 and ('good' in text.lower() or 'bad' in text.lower()) and not ('excellent' in text.lower() or 'terrible' in text.lower()):
            neutral_train.append({'text': text, 'label': 1})  # neutral
            if len(neutral_train) >= 5000:  # Sample 5k neutral
                break

    # Get neutral samples from test
    neutral_test = []
    for _, row in test_df.iterrows():
        text = row['text']
        if len(text.split()) < 50 and ('good' in text.lower() or 'bad' in text.lower()) and not ('excellent' in text.lower() or 'terrible' in text.lower()):
            neutral_test.append({'text': text, 'label': 1})  # neutral
            if len(neutral_test) >= 1250:  # Sample 1.25k neutral for test
                break

    # Convert to DataFrames
    neutral_train_df = pd.DataFrame(neutral_train)
    neutral_test_df = pd.DataFrame(neutral_test)

    # Sample from original classes (IMDB has 12500 positive and 12500 negative each)
    pos_train = train_df[train_df['label'] == 1].sample(n=10000, random_state=42)  # positive
    neg_train = train_df[train_df['label'] == 0].sample(n=10000, random_state=42)  # negative

    pos_test = test_df[test_df['label'] == 1].sample(n=2500, random_state=42)  # positive
    neg_test = test_df[test_df['label'] == 0].sample(n=2500, random_state=42)  # negative

    # Combine datasets
    combined_train = pd.concat([
        neg_train,
        neutral_train_df,
        pos_train
    ], ignore_index=True)

    combined_test = pd.concat([
        neg_test,
        neutral_test_df,
        pos_test
    ], ignore_index=True)

    # Shuffle
    combined_train = combined_train.sample(frac=1, random_state=42).reset_index(drop=True)
    combined_test = combined_test.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Final train dataset: {len(combined_train)} samples")
    print(f"Final test dataset: {len(combined_test)} samples")
    print("Label distribution (train):")
    print(combined_train['label'].value_counts())
    print("Label distribution (test):")
    print(combined_test['label'].value_counts())

    return combined_train, combined_test

def save_datasets(train_df, test_df, output_dir="data"):
    """Save datasets to CSV files"""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "raw"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "processed"), exist_ok=True)

    # Save raw data
    train_df.to_csv(os.path.join(output_dir, "raw", "train.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "raw", "test.csv"), index=False)

    # Create validation split from train
    train_split, val_split = train_test_split(
        train_df, test_size=0.2, random_state=42, stratify=train_df['label']
    )

    # Save processed data
    train_split.to_csv(os.path.join(output_dir, "processed", "train.csv"), index=False)
    val_split.to_csv(os.path.join(output_dir, "processed", "val.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "processed", "test.csv"), index=False)

    # Save dataset info
    dataset_info = {
        "dataset_name": "IMDB_3Class_Sentiment",
        "description": "IMDB movie reviews dataset converted to 3-class sentiment (negative, neutral, positive)",
        "classes": ["negative", "neutral", "positive"],
        "train_samples": len(train_split),
        "val_samples": len(val_split),
        "test_samples": len(test_df),
        "features": ["text", "label"],
        "label_mapping": {0: "negative", 1: "neutral", 2: "positive"}
    }

    with open(os.path.join(output_dir, "processed", "dataset_info.json"), 'w') as f:
        json.dump(dataset_info, f, indent=2)

    print(f"✅ Datasets saved to {output_dir}")
    print(f"   - Raw data: {len(train_df)} train, {len(test_df)} test samples")
    print(f"   - Processed: {len(train_split)} train, {len(val_split)} val, {len(test_df)} test samples")

def main():
    """Main function to download and prepare sentiment dataset"""
    print("🚀 Starting BERT Sentiment Analysis Data Fetcher")
    print("=" * 50)

    try:
        # Download IMDB dataset
        train_df, test_df = download_imdb_dataset()

        # Create 3-class dataset
        train_3class, test_3class = create_three_class_dataset(train_df, test_df)

        # Save datasets
        save_datasets(train_3class, test_3class)

        print("\n✅ Sentiment analysis dataset preparation completed!")
        print("📁 Dataset saved in data/ directory")
        print("🎯 Ready for BERT fine-tuning")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()