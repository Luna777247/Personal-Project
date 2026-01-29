import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import json
from tqdm import tqdm

class TextPreprocessor:
    def __init__(self):
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')

        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')

        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

        # Custom stop words to keep for sentiment analysis
        self.keep_words = {'not', 'no', 'nor', 'neither', 'never', 'none'}
        self.stop_words = self.stop_words - self.keep_words

    def clean_text(self, text):
        """Clean and preprocess text"""
        if not isinstance(text, str):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Remove user mentions and hashtags (keep for some analysis)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+', '', text)

        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', '', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def remove_stopwords(self, text):
        """Remove stopwords from text"""
        words = text.split()
        filtered_words = [word for word in words if word not in self.stop_words]
        return ' '.join(filtered_words)

    def lemmatize_text(self, text):
        """Lemmatize words in text"""
        words = text.split()
        lemmatized_words = [self.lemmatizer.lemmatize(word) for word in words]
        return ' '.join(lemmatized_words)

    def preprocess_text(self, text, remove_stopwords=True, lemmatize=True):
        """Complete text preprocessing pipeline"""
        text = self.clean_text(text)

        if remove_stopwords:
            text = self.remove_stopwords(text)

        if lemmatize:
            text = self.lemmatize_text(text)

        return text

    def preprocess_dataframe(self, df, text_column='text', apply_preprocessing=True):
        """Preprocess text in a pandas DataFrame"""
        if apply_preprocessing:
            tqdm.pandas(desc="Preprocessing text")
            df['processed_text'] = df[text_column].progress_apply(self.preprocess_text)
        else:
            df['processed_text'] = df[text_column]

        return df

class SentimentDataLoader:
    def __init__(self):
        self.preprocessor = TextPreprocessor()

    def load_csv(self, filepath, text_column='text', label_column='label',
                encoding='utf-8', sep=','):
        """Load data from CSV file"""
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath, encoding=encoding, sep=sep)

        print(f"Loaded {len(df)} samples")
        print(f"Columns: {list(df.columns)}")

        # Check for required columns
        if text_column not in df.columns:
            raise ValueError(f"Text column '{text_column}' not found in dataset")
        if label_column not in df.columns:
            raise ValueError(f"Label column '{label_column}' not found in dataset")

        return df

    def balance_dataset(self, df, label_column='label', strategy='upsample'):
        """Balance dataset by upsampling or downsampling"""
        print("Balancing dataset...")

        # Get class counts
        class_counts = df[label_column].value_counts()
        print("Original class distribution:")
        print(class_counts)

        # Find majority and minority classes
        max_count = class_counts.max()
        min_count = class_counts.min()

        balanced_dfs = []

        for class_label in class_counts.index:
            class_df = df[df[label_column] == class_label]

            if strategy == 'upsample':
                # Upsample minority classes
                if len(class_df) < max_count:
                    class_df = resample(class_df,
                                      replace=True,
                                      n_samples=max_count,
                                      random_state=42)
            elif strategy == 'downsample':
                # Downsample majority classes
                if len(class_df) > min_count:
                    class_df = resample(class_df,
                                      replace=False,
                                      n_samples=min_count,
                                      random_state=42)

            balanced_dfs.append(class_df)

        balanced_df = pd.concat(balanced_dfs)
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

        print("Balanced class distribution:")
        print(balanced_df[label_column].value_counts())

        return balanced_df

    def split_dataset(self, df, train_size=0.7, val_size=0.15, test_size=0.15,
                     text_column='processed_text', label_column='label', random_state=42):
        """Split dataset into train/val/test sets"""
        if abs(train_size + val_size + test_size - 1.0) > 1e-6:
            raise ValueError("Train, validation, and test sizes must sum to 1.0")

        # First split: train and temp (val+test)
        train_df, temp_df = train_test_split(
            df, train_size=train_size, random_state=random_state, stratify=df[label_column]
        )

        # Second split: val and test
        val_df, test_df = train_test_split(
            temp_df,
            train_size=val_size/(val_size + test_size),
            random_state=random_state,
            stratify=temp_df[label_column]
        )

        print(f"Dataset split:")
        print(f"  Train: {len(train_df)} samples")
        print(f"  Validation: {len(val_df)} samples")
        print(f"  Test: {len(test_df)} samples")

        return train_df, val_df, test_df

    def analyze_dataset(self, df, text_column='processed_text', label_column='label',
                       save_plots=True):
        """Analyze dataset characteristics"""
        print("\n" + "="*50)
        print("DATASET ANALYSIS")
        print("="*50)

        # Basic statistics
        print(f"Total samples: {len(df)}")
        print(f"Unique labels: {df[label_column].nunique()}")

        # Label distribution
        label_counts = df[label_column].value_counts()
        print(f"\nLabel distribution:")
        for label, count in label_counts.items():
            percentage = (count / len(df)) * 100
            print(".1f")

        # Text length analysis
        df['text_length'] = df[text_column].str.len()
        print(f"\nText length statistics:")
        print(f"  Mean length: {df['text_length'].mean():.1f} characters")
        print(f"  Median length: {df['text_length'].median():.1f} characters")
        print(f"  Max length: {df['text_length'].max()} characters")
        print(f"  Min length: {df['text_length'].min()} characters")

        # Most common words
        all_words = ' '.join(df[text_column].dropna()).split()
        word_freq = Counter(all_words)
        print(f"\nMost common words (top 10):")
        for word, freq in word_freq.most_common(10):
            print(f"  {word}: {freq}")

        # Plot distributions
        if save_plots:
            self._plot_distributions(df, label_column)

        return {
            'total_samples': len(df),
            'label_distribution': label_counts.to_dict(),
            'text_stats': {
                'mean_length': df['text_length'].mean(),
                'median_length': df['text_length'].median(),
                'max_length': df['text_length'].max(),
                'min_length': df['text_length'].min()
            },
            'common_words': dict(word_freq.most_common(20))
        }

    def _plot_distributions(self, df, label_column):
        """Create distribution plots"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Label distribution
        label_counts = df[label_column].value_counts()
        sns.barplot(x=label_counts.index, y=label_counts.values, ax=ax1)
        ax1.set_title('Label Distribution')
        ax1.set_xlabel('Label')
        ax1.set_ylabel('Count')

        # Text length distribution
        sns.histplot(df['text_length'], bins=50, ax=ax2)
        ax2.set_title('Text Length Distribution')
        ax2.set_xlabel('Text Length (characters)')
        ax2.set_ylabel('Frequency')

        plt.tight_layout()
        plt.savefig('data/dataset_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    def save_processed_data(self, train_df, val_df, test_df, output_dir='data/processed'):
        """Save processed datasets"""
        import os
        os.makedirs(output_dir, exist_ok=True)

        train_df.to_csv(f'{output_dir}/train.csv', index=False)
        val_df.to_csv(f'{output_dir}/val.csv', index=False)
        test_df.to_csv(f'{output_dir}/test.csv', index=False)

        print(f"Processed datasets saved to {output_dir}")

def main():
    """Main data processing pipeline"""
    # Initialize components
    data_loader = SentimentDataLoader()

    # Load raw data (replace with your actual data file)
    try:
        # Example: df = data_loader.load_csv('data/raw/sentiment_data.csv')
        print("Please provide your dataset file path in the code")
        print("Example usage:")
        print("""
        # Load your data
        df = data_loader.load_csv('data/raw/your_dataset.csv')

        # Preprocess text
        df = data_loader.preprocessor.preprocess_dataframe(df)

        # Balance dataset if needed
        df = data_loader.balance_dataset(df)

        # Analyze dataset
        stats = data_loader.analyze_dataset(df)

        # Split dataset
        train_df, val_df, test_df = data_loader.split_dataset(df)

        # Save processed data
        data_loader.save_processed_data(train_df, val_df, test_df)
        """)

    except FileNotFoundError:
        print("No dataset file found. Please add your data file to data/raw/")
        print("Expected format: CSV with 'text' and 'label' columns")

if __name__ == "__main__":
    main()