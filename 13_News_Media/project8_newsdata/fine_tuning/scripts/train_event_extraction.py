#!/usr/bin/env python3
"""
Fine-tuning Script for Event Extraction
Specialized for Disaster Event Classification and Extraction
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
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
import numpy as np
from sklearn.metrics import classification_report, f1_score, accuracy_score
import wandb
from omegaconf import OmegaConf


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EventExtractionDataset(Dataset):
    """Dataset for Event Extraction training"""

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

        # Get label
        label = item.get("event_type", "O")
        label_id = self.label2id.get(label, self.label2id["O"])

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label_id, dtype=torch.long)
        }


def load_data(data_path: str) -> List[Dict[str, Any]]:
    """Load annotated data"""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def create_event_label_mappings() -> Tuple[Dict[str, int], Dict[int, str]]:
    """Create label mappings for disaster event types"""
    labels = [
        "O",           # No event
        "BAO",         # Storm/Typhoon
        "LUU_LUT",    # Flood
        "HAN_HAN",    # Drought
        "CHAY_RUNG",  # Forest Fire
        "DONG_DAT",   # Earthquake
        "SAT_LO",     # Landslide
        "BAO_TAP"     # Other disasters
    ]

    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}
    return label2id, id2label


def compute_metrics(eval_pred):
    """Compute evaluation metrics for sequence classification"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')

    return {
        'accuracy': accuracy,
        'f1': f1
    }


def train_event_extraction_model(
    config_path: str,
    train_data_path: str,
    val_data_path: Optional[str] = None,
    output_dir: str = "output/event_model"
):
    """Train Event Extraction model"""

    # Load configuration
    config = OmegaConf.load(config_path)
    model_config = config.models.phobert  # Use PhoBERT for Vietnamese
    training_config = config.training

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
    label2id, id2label = create_event_label_mappings()

    # Initialize tokenizer and model
    logger.info(f"Loading model: {model_config.name}")
    tokenizer = AutoTokenizer.from_pretrained(model_config.name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_config.name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )

    # Create datasets
    train_dataset = EventExtractionDataset(train_data, tokenizer, label2id, model_config.max_length)
    val_dataset = EventExtractionDataset(val_data, tokenizer, label2id, model_config.max_length)

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
            name=f"event-extraction-{model_config.name.split('/')[-1]}",
            config=OmegaConf.to_container(config)
        )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
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


def predict_event_type(text: str, model_path: str, device: str = "auto") -> Dict[str, Any]:
    """Predict event type for new text"""

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    # Load label mappings
    with open(os.path.join(model_path, "id2label.json"), 'r') as f:
        id2label = json.load(f)
        id2label = {int(k): v for k, v in id2label.items()}

    # Set device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    model.eval()

    # Tokenize text
    encoding = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    encoding = {k: v.to(device) for k, v in encoding.items()}

    # Predict
    with torch.no_grad():
        outputs = model(**encoding)
        predictions = torch.argmax(outputs.logits, dim=1)
        probabilities = torch.softmax(outputs.logits, dim=1)

    predicted_label = id2label[predictions[0].item()]
    confidence = probabilities[0][predictions[0]].item()

    return {
        "event_type": predicted_label,
        "confidence": confidence,
        "all_probabilities": {
            id2label[i]: prob.item()
            for i, prob in enumerate(probabilities[0])
        }
    }


class EventExtractor:
    """Complete Event Extraction System"""

    def __init__(self, ner_model_path: str, event_model_path: str):
        self.ner_model_path = ner_model_path
        self.event_model_path = event_model_path

        # Load models
        self.ner_tokenizer = AutoTokenizer.from_pretrained(ner_model_path)
        self.ner_model = AutoModelForTokenClassification.from_pretrained(ner_model_path)

        self.event_tokenizer = AutoTokenizer.from_pretrained(event_model_path)
        self.event_model = AutoModelForSequenceClassification.from_pretrained(event_model_path)

        # Load label mappings
        with open(os.path.join(ner_model_path, "id2label.json"), 'r') as f:
            self.ner_id2label = json.load(f)
            self.ner_id2label = {int(k): v for k, v in self.ner_id2label.items()}

        with open(os.path.join(event_model_path, "id2label.json"), 'r') as f:
            self.event_id2label = json.load(f)
            self.event_id2label = {int(k): v for k, v in self.event_id2label.items()}

    def extract_events(self, text: str) -> Dict[str, Any]:
        """Extract complete event information"""

        # Predict event type
        event_result = predict_event_type(text, self.event_model_path)

        # Extract entities
        ner_result = predict_ner(text, self.ner_model_path)

        # Structure the result
        result = {
            "text": text,
            "event_type": event_result["event_type"],
            "event_confidence": event_result["confidence"],
            "entities": ner_result,
            "structured_info": self._structure_entities(ner_result, event_result["event_type"])
        }

        return result

    def _structure_entities(self, entities: List[Dict[str, Any]], event_type: str) -> Dict[str, Any]:
        """Structure entities into disaster information"""

        structured = {
            "disaster_type": event_type,
            "location": None,
            "time": None,
            "damage": [],
            "response": [],
            "impact": [],
            "forecast": []
        }

        for entity in entities:
            label = entity["label"]
            text = entity["text"]

            if label == "LOCATION":
                structured["location"] = text
            elif label == "TIME":
                structured["time"] = text
            elif label == "DAMAGE":
                structured["damage"].append(text)
            elif label == "RESPONSE":
                structured["response"].append(text)
            elif label == "IMPACT":
                structured["impact"].append(text)
            elif label == "FORECAST":
                structured["forecast"].append(text)

        return structured


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Fine-tune Event Extraction Model")
    parser.add_argument("--config", required=True, help="Configuration file path")
    parser.add_argument("--train-data", required=True, help="Training data path")
    parser.add_argument("--val-data", help="Validation data path")
    parser.add_argument("--output-dir", default="output/event_model", help="Output directory")
    parser.add_argument("--predict", help="Predict on text file instead of training")
    parser.add_argument("--model-path", help="Model path for prediction")
    parser.add_argument("--extract-events", help="Extract events from text file (requires both NER and Event models)")
    parser.add_argument("--ner-model", help="NER model path for event extraction")

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
                result = predict_event_type(text, args.model_path)
                print(f"Text: {text}")
                print(f"Event Type: {result['event_type']} (confidence: {result['confidence']:.3f})")
                print("-" * 50)

    elif args.extract_events:
        # Complete event extraction mode
        if not args.ner_model or not args.model_path:
            logger.error("Both NER model and Event model paths required for event extraction")
            return

        extractor = EventExtractor(args.ner_model, args.model_path)

        with open(args.extract_events, 'r', encoding='utf-8') as f:
            texts = f.readlines()

        for text in texts:
            text = text.strip()
            if text:
                result = extractor.extract_events(text)
                print(f"Text: {text}")
                print(f"Event: {result['event_type']} ({result['event_confidence']:.3f})")
                print(f"Location: {result['structured_info']['location']}")
                print(f"Time: {result['structured_info']['time']}")
                print(f"Damage: {result['structured_info']['damage']}")
                print("-" * 50)

    else:
        # Training mode
        trainer, eval_results = train_event_extraction_model(
            args.config,
            args.train_data,
            args.val_data,
            args.output_dir
        )

        logger.info("Training completed!")
        logger.info(f"F1 Score: {eval_results.get('eval_f1', 'N/A')}")
        logger.info(f"Accuracy: {eval_results.get('eval_accuracy', 'N/A')}")


if __name__ == "__main__":
    main()