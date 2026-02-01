#!/usr/bin/env python3
"""
Fine-tuning Script for Named Entity Recognition (NER)
Specialized for Disaster Information Extraction
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback
)
import numpy as np
from sklearn.metrics import classification_report, f1_score
import wandb
from omegaconf import OmegaConf


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NERDataset(Dataset):
    """Dataset for NER training"""

    def __init__(self, data: List[Dict[str, Any]], tokenizer, label2id: Dict[str, int], max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Tokenize text
        encoding = self.tokenizer(
            item["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        # Convert labels
        labels = self._convert_labels_to_ids(item["entities"], encoding)

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(labels, dtype=torch.long)
        }

    def _convert_labels_to_ids(self, entities: List[Dict[str, Any]], encoding) -> List[int]:
        """Convert entity annotations to label IDs"""
        labels = ["O"] * len(encoding["input_ids"][0])

        for entity in entities:
            start_char = entity["start"]
            end_char = entity["end"]
            label = entity["label"]

            # Find tokens that correspond to the entity
            token_start = encoding.char_to_token(start_char)
            token_end = encoding.char_to_token(end_char - 1)

            if token_start is None or token_end is None:
                continue

            # Set B- label for first token
            if token_start < len(labels):
                labels[token_start] = f"B-{label}"

            # Set I- label for subsequent tokens
            for i in range(token_start + 1, min(token_end + 1, len(labels))):
                labels[i] = f"I-{label}"

        # Convert to IDs
        label_ids = [self.label2id.get(label, self.label2id["O"]) for label in labels]
        return label_ids


def load_data(data_path: str) -> List[Dict[str, Any]]:
    """Load annotated data"""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def create_label_mappings(label_list: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Create label to ID and ID to label mappings"""
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for label, i in label2id.items()}
    return label2id, id2label


def compute_metrics(eval_pred):
    """Compute evaluation metrics"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [p for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [l for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    # Flatten
    true_predictions = [p for pred in true_predictions for p in pred]
    true_labels = [l for label in true_labels for l in label]

    # Calculate F1 score
    f1 = f1_score(true_labels, true_predictions, average='weighted')

    return {
        'f1': f1,
        'accuracy': (np.array(true_predictions) == np.array(true_labels)).mean()
    }


def train_ner_model(
    config_path: str,
    train_data_path: str,
    val_data_path: Optional[str] = None,
    output_dir: str = "output/ner_model"
):
    """Train NER model"""

    # Load configuration
    config = OmegaConf.load(config_path)
    model_config = config.models.phobert  # Use PhoBERT for Vietnamese
    training_config = config.training
    data_config = config.data

    # Load data
    logger.info("Loading training data...")
    train_data = load_data(train_data_path)

    if val_data_path:
        val_data = load_data(val_data_path)
    else:
        # Split train data for validation
        split_idx = int(len(train_data) * 0.9)
        val_data = train_data[split_idx:]
        train_data = train_data[:split_idx]

    logger.info(f"Train samples: {len(train_data)}")
    logger.info(f"Validation samples: {len(val_data)}")

    # Create label mappings
    label2id, id2label = create_label_mappings(config.tasks.ner.label_list)

    # Initialize tokenizer and model
    logger.info(f"Loading model: {model_config.name}")
    tokenizer = AutoTokenizer.from_pretrained(model_config.name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_config.name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )

    # Create datasets
    train_dataset = NERDataset(train_data, tokenizer, label2id, model_config.max_length)
    val_dataset = NERDataset(val_data, tokenizer, label2id, model_config.max_length)

    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=training_config.num_epochs,
        per_device_train_batch_size=training_config.batch_size,
        per_device_eval_batch_size=training_config.batch_size,
        learning_rate=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
        warmup_steps=training_config.warmup_steps,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        max_grad_norm=training_config.max_grad_norm,
        evaluation_strategy=training_config.evaluation_strategy,
        eval_steps=training_config.eval_steps,
        save_steps=training_config.save_steps,
        logging_steps=training_config.logging_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        dataloader_pin_memory=False,
        report_to="wandb" if config.wandb.enabled else "none"
    )

    # Initialize wandb if enabled
    if config.wandb.enabled:
        wandb.init(
            project=config.wandb.project,
            name=f"ner-{model_config.name.split('/')[-1]}",
            config=OmegaConf.to_container(config)
        )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=training_config.early_stopping_patience,
            early_stopping_threshold=training_config.early_stopping_threshold
        )]
    )

    # Train model
    logger.info("Starting training...")
    trainer.train()

    # Save model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save label mappings
    with open(os.path.join(output_dir, "label2id.json"), 'w') as f:
        json.dump(label2id, f)

    with open(os.path.join(output_dir, "id2label.json"), 'w') as f:
        json.dump(id2label, f)

    logger.info(f"Model saved to {output_dir}")

    # Evaluate on validation set
    logger.info("Evaluating model...")
    eval_results = trainer.evaluate()
    logger.info(f"Evaluation results: {eval_results}")

    # Save evaluation results
    with open(os.path.join(output_dir, "eval_results.json"), 'w') as f:
        json.dump(eval_results, f, indent=2)

    return trainer, eval_results


def predict_ner(text: str, model_path: str, device: str = "auto") -> List[Dict[str, Any]]:
    """Predict NER on new text"""

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)

    # Load label mappings
    with open(os.path.join(model_path, "label2id.json"), 'r') as f:
        label2id = json.load(f)

    id2label = {int(k): v for k, v in model["id2label"].items()}

    # Set device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    model.eval()

    # Tokenize text
    encoding = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    encoding = {k: v.to(device) for k, v in encoding.items()}

    # Predict
    with torch.no_grad():
        outputs = model(**encoding)
        predictions = torch.argmax(outputs.logits, dim=2)

    # Convert predictions to entities
    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
    labels = [id2label[pred.item()] for pred in predictions[0]]

    entities = []
    current_entity = None

    for i, (token, label) in enumerate(zip(tokens, labels)):
        if label.startswith("B-"):
            if current_entity:
                entities.append(current_entity)

            current_entity = {
                "text": tokenizer.convert_tokens_to_string([token]),
                "label": label[2:],  # Remove B- prefix
                "start": encoding.char_to_token(i) if encoding.char_to_token(i) is not None else 0,
                "end": encoding.char_to_token(i) + len(token) if encoding.char_to_token(i) is not None else len(token),
                "confidence": outputs.logits[0][i][predictions[0][i]].item()
            }
        elif label.startswith("I-") and current_entity:
            current_entity["text"] += tokenizer.convert_tokens_to_string([token])
            current_entity["end"] = encoding.char_to_token(i) + len(token) if encoding.char_to_token(i) is not None else len(token)
        elif label == "O" and current_entity:
            entities.append(current_entity)
            current_entity = None

    if current_entity:
        entities.append(current_entity)

    return entities


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Fine-tune NER Model for Disaster Extraction")
    parser.add_argument("--config", required=True, help="Configuration file path")
    parser.add_argument("--train-data", required=True, help="Training data path")
    parser.add_argument("--val-data", help="Validation data path")
    parser.add_argument("--output-dir", default="output/ner_model", help="Output directory")
    parser.add_argument("--predict", help="Predict on text file instead of training")
    parser.add_argument("--model-path", help="Model path for prediction")

    args = parser.parse_args()

    if args.predict:
        # Prediction mode
        if not args.model_path:
            logger.error("Model path required for prediction")
            return

        with open(args.predict, 'r', encoding='utf-8') as f:
            texts = f.readlines()

        for text in texts:
            text = text.strip()
            if text:
                entities = predict_ner(text, args.model_path)
                print(f"Text: {text}")
                print(f"Entities: {entities}")
                print("-" * 50)

    else:
        # Training mode
        trainer, eval_results = train_ner_model(
            args.config,
            args.train_data,
            args.val_data,
            args.output_dir
        )

        logger.info("Training completed!")
        logger.info(f"F1 Score: {eval_results.get('eval_f1', 'N/A')}")


if __name__ == "__main__":
    main()