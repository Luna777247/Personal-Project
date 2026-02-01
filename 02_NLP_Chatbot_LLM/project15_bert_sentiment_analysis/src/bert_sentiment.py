import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    get_linear_schedule_with_warmup,
    AutoTokenizer, AutoModelForSequenceClassification
)
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import json
from datetime import datetime

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class BERTSentimentClassifier:
    def __init__(self, model_name='bert-base-uncased', num_labels=3, max_length=128):
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = None
        self.model = None
        self.history = {}

    def initialize_model(self):
        """Initialize BERT tokenizer and model"""
        print(f"Loading {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels
        )
        self.model.to(self.device)
        print(f"Model loaded successfully. Using device: {self.device}")

    def prepare_data(self, texts, labels, batch_size=16, val_split=0.2):
        """Prepare train and validation datasets"""
        # Split data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=val_split, random_state=42, stratify=labels
        )

        # Create datasets
        train_dataset = SentimentDataset(train_texts, train_labels, self.tokenizer, self.max_length)
        val_dataset = SentimentDataset(val_texts, val_labels, self.tokenizer, self.max_length)

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")

        return train_loader, val_loader

    def train(self, train_loader, val_loader, epochs=3, learning_rate=2e-5,
             warmup_steps=0, save_path='models/bert_sentiment'):
        """Fine-tune BERT model"""
        optimizer = AdamW(self.model.parameters(), lr=learning_rate, correct_bias=False)
        total_steps = len(train_loader) * epochs

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        best_f1 = 0.0
        training_stats = []

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 50)

            # Training phase
            self.model.train()
            train_loss = 0
            train_preds = []
            train_labels = []

            progress_bar = tqdm(train_loader, desc="Training")
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                optimizer.zero_grad()

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                train_loss += loss.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                # Get predictions for metrics
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                train_preds.extend(preds)
                train_labels.extend(labels.cpu().numpy())

                progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

            avg_train_loss = train_loss / len(train_loader)
            train_f1 = f1_score(train_labels, train_preds, average='weighted')

            # Validation phase
            val_loss, val_f1, val_preds, val_labels = self.evaluate(val_loader)

            print(".4f")
            print(".4f")

            # Save best model
            if val_f1 > best_f1:
                best_f1 = val_f1
                self.save_model(save_path)
                print(f"New best model saved with F1: {best_f1:.4f}")

            # Store statistics
            epoch_stats = {
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'train_f1': train_f1,
                'val_loss': val_loss,
                'val_f1': val_f1
            }
            training_stats.append(epoch_stats)

        self.history = training_stats
        return training_stats

    def evaluate(self, data_loader):
        """Evaluate model on validation/test data"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                total_loss += loss.item()

                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(data_loader)
        f1 = f1_score(all_labels, all_preds, average='weighted')

        return avg_loss, f1, all_preds, all_labels

    def predict(self, texts, batch_size=16):
        """Make predictions on new texts"""
        self.model.eval()
        predictions = []

        # Create dataset and dataloader for prediction
        pred_dataset = SentimentDataset(texts, [0] * len(texts), self.tokenizer, self.max_length)
        pred_loader = DataLoader(pred_dataset, batch_size=batch_size, shuffle=False)

        with torch.no_grad():
            for batch in pred_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                predictions.extend(preds)

        return predictions

    def save_model(self, save_path):
        """Save model and tokenizer"""
        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"Model saved to {save_path}")

    def load_model(self, load_path):
        """Load saved model"""
        self.tokenizer = AutoTokenizer.from_pretrained(load_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(load_path)
        self.model.to(self.device)
        print(f"Model loaded from {load_path}")

    def plot_training_history(self, save_path='models/training_history.png'):
        """Plot training history"""
        if not self.history:
            print("No training history available")
            return

        epochs = [stat['epoch'] for stat in self.history]
        train_loss = [stat['train_loss'] for stat in self.history]
        val_loss = [stat['val_loss'] for stat in self.history]
        train_f1 = [stat['train_f1'] for stat in self.history]
        val_f1 = [stat['val_f1'] for stat in self.history]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Loss plot
        ax1.plot(epochs, train_loss, label='Training Loss')
        ax1.plot(epochs, val_loss, label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()

        # F1 plot
        ax2.plot(epochs, train_f1, label='Training F1')
        ax2.plot(epochs, val_f1, label='Validation F1')
        ax2.set_title('Training and Validation F1 Score')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('F1 Score')
        ax2.legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def get_classification_report(self, true_labels, predictions, target_names=None):
        """Generate detailed classification report"""
        if target_names is None:
            target_names = [f'Class_{i}' for i in range(self.num_labels)]

        report = classification_report(true_labels, predictions, target_names=target_names)
        return report

    def save_training_stats(self, stats, filepath='models/training_stats.json'):
        """Save training statistics to JSON"""
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Training statistics saved to {filepath}")

def load_data_from_csv(filepath, text_column='text', label_column='label'):
    """Load data from CSV file"""
    df = pd.read_csv(filepath)
    texts = df[text_column].tolist()
    labels = df[label_column].tolist()
    return texts, labels

def main():
    """Main training pipeline"""
    # Initialize classifier
    classifier = BERTSentimentClassifier(num_labels=3)  # negative, neutral, positive

    # Load model
    classifier.load_model()

    # Example data loading (replace with your actual data)
    # texts, labels = load_data_from_csv('data/train.csv')

    # For demonstration, using sample data
    sample_texts = [
        "I love this product! It's amazing!",
        "This is terrible. I hate it.",
        "It's okay, nothing special.",
        "Absolutely fantastic experience!",
        "Worst purchase I've ever made."
    ]
    sample_labels = [2, 0, 1, 2, 0]  # 0: negative, 1: neutral, 2: positive

    # Prepare data
    train_loader, val_loader = classifier.prepare_data(
        sample_texts, sample_labels, batch_size=4
    )

    # Train model
    print("Starting training...")
    stats = classifier.train(train_loader, val_loader, epochs=2)

    # Save training statistics
    classifier.save_training_stats(stats)

    # Plot training history
    classifier.plot_training_history()

    # Make predictions
    test_texts = ["This is great!", "I don't like it."]
    predictions = classifier.predict(test_texts)
    print(f"Predictions: {predictions}")

    print("BERT sentiment analysis training complete!")

if __name__ == "__main__":
    main()