#!/usr/bin/env python3
"""
Real Data Fetcher for MLOps Mini Project
Downloads California Housing dataset for regression modeling
"""

import os
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import json
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_california_housing_data():
    """Fetch California Housing dataset from scikit-learn"""
    logger.info("Fetching California Housing dataset...")

    # Load the dataset
    housing = fetch_california_housing()

    # Create DataFrame
    df = pd.DataFrame(
        data=np.c_[housing['data'], housing['target']],
        columns=housing['feature_names'] + ['target']
    )

    logger.info(f"Dataset loaded with shape: {df.shape}")
    logger.info(f"Features: {housing['feature_names']}")
    logger.info("Target: Median house value (in $100,000s)")

    return df, housing['feature_names'], housing['DESCR']

def create_data_directory():
    """Create data directory if it doesn't exist"""
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    return data_dir

def save_dataset_info(data_dir, feature_names, description, stats):
    """Save dataset information"""
    info = {
        "dataset_name": "California Housing",
        "description": description,
        "features": feature_names,
        "target": "Median house value (in $100,000s)",
        "fetch_timestamp": datetime.now().isoformat(),
        "statistics": stats
    }

    info_path = os.path.join(data_dir, "dataset_info.json")
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)

    logger.info(f"Dataset info saved to {info_path}")

def main():
    """Main function to fetch and prepare data"""
    logger.info("🚀 Starting California Housing data fetch...")

    try:
        # Fetch data
        df, feature_names, description = fetch_california_housing_data()

        # Create data directory
        data_dir = create_data_directory()

        # Calculate statistics
        stats = {
            "total_samples": len(df),
            "num_features": len(feature_names),
            "target_stats": {
                "mean": float(df['target'].mean()),
                "std": float(df['target'].std()),
                "min": float(df['target'].min()),
                "max": float(df['target'].max()),
                "median": float(df['target'].median())
            },
            "feature_stats": {}
        }

        # Feature statistics
        for feature in feature_names:
            stats["feature_stats"][feature] = {
                "mean": float(df[feature].mean()),
                "std": float(df[feature].std()),
                "min": float(df[feature].min()),
                "max": float(df[feature].max())
            }

        # Split data for training
        logger.info("Splitting data into train/val/test sets...")
        train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

        logger.info(f"Train samples: {len(train_df)}")
        logger.info(f"Validation samples: {len(val_df)}")
        logger.info(f"Test samples: {len(test_df)}")

        # Save datasets
        train_path = os.path.join(data_dir, "train.csv")
        val_path = os.path.join(data_dir, "val.csv")
        test_path = os.path.join(data_dir, "test.csv")
        full_path = os.path.join(data_dir, "california_housing.csv")

        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)
        df.to_csv(full_path, index=False)

        # Save dataset info
        save_dataset_info(data_dir, feature_names, description, stats)

        logger.info("✅ California Housing data fetch completed successfully!")
        logger.info(f"📊 Dataset: {len(df)} samples, {len(feature_names)} features")
        logger.info(f"🎯 Target: House prices (median in $100,000s)")
        logger.info(f"📁 Data saved to {data_dir}/")
        logger.info(f"   - Full dataset: california_housing.csv")
        logger.info(f"   - Train set: train.csv ({len(train_df)} samples)")
        logger.info(f"   - Validation set: val.csv ({len(val_df)} samples)")
        logger.info(f"   - Test set: test.csv ({len(test_df)} samples)")

        return True

    except Exception as e:
        logger.error(f"❌ Data fetch failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)