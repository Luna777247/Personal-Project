#!/usr/bin/env python3
"""
Fine-tuning Script for Relation Extraction (RE)
Specialized for Disaster Information Relation Extraction
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


class RelationExtractionDataset(Dataset):
    """Dataset for Relation Extraction training"""

    def __init__(self, data: List[Dict[str, Any]], tokenizer, label2id: Dict[str, int], max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Create input text with entity markers
        text = self._create_input_text(item)

        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        # Get label
        label = item.get("relation_type", "NO_RELATION")
        label_id = self.label2id.get(label, self.label2id["NO_RELATION"])

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label_id, dtype=torch.long)
        }

    def _create_input_text(self, item: Dict[str, Any]) -> str:
        """Create input text with entity markers"""
        text = item["text"]
        head = item["head"]
        tail = item["tail"]

        # Mark entities in text
        marked_text = text
        offset = 0

        # Insert tail entity marker first (higher position)
        tail_start = tail["start"] + offset
        tail_end = tail["end"] + offset
        marked_text = marked_text[:tail_start] + "[TAIL] " + marked_text[tail_start:tail_end] + " [/TAIL]" + marked_text[tail_end:]
        offset += len("[TAIL]  [/TAIL]")

        # Insert head entity marker
        head_start = head["start"] + offset
        head_end = head["end"] + offset
        marked_text = marked_text[:head_start] + "[HEAD] " + marked_text[head_start:head_end] + " [/HEAD]" + marked_text[head_end:]

        return marked_text


def load_relation_data(data_path: str) -> List[Dict[str, Any]]:
    """Load relation extraction data"""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Convert to relation instances
    relation_instances = []

    for doc in data:
        text = doc["text"]
        entities = doc.get("entities", [])
        relations = doc.get("relations", [])

        # Create entity lookup
        entity_lookup = {f"{e['start']}-{e['end']}": e for e in entities}

        # Process existing relations
        for rel in relations:
            head_key = f"{rel['head']['start']}-{rel['head']['end']}"
            tail_key = f"{rel['tail']['start']}-{rel['tail']['end']}"

            if head_key in entity_lookup and tail_key in entity_lookup:
                relation_instances.append({
                    "text": text,
                    "head": rel["head"],
                    "tail": rel["tail"],
                    "relation_type": rel["relation_type"]
                })

        # Generate negative examples (no relation)
        for i, head in enumerate(entities):
            for j, tail in enumerate(entities):
                if i != j:
                    # Check if this pair already has a relation
                    has_relation = any(
                        (r["head"]["start"] == head["start"] and r["head"]["end"] == head["end"] and
                         r["tail"]["start"] == tail["start"] and r["tail"]["end"] == tail["end"])
                        for r in relations
                    )

                    if not has_relation:
                        relation_instances.append({
                            "text": text,
                            "head": head,
                            "tail": tail,
                            "relation_type": "NO_RELATION"
                        })

    return relation_instances


def create_relation_label_mappings() -> Tuple[Dict[str, int], Dict[int, str]]:
    """Create label mappings for relation types"""
    labels = [
        "NO_RELATION",
        "LOCATION_OF",    # Location of disaster
        "TIME_OF",        # Time of disaster
        "CAUSE_OF",       # Damage caused by disaster
        "IMPACT_OF"       # Social impact of disaster
    ]

    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}
    return label2id, id2label


def compute_metrics(eval_pred):
    """Compute evaluation metrics for relation extraction"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')

    return {
        'accuracy': accuracy,
        'f1': f1
    }


def train_relation_extraction_model(
    config_path: str,
    train_data_path: str,
    val_data_path: Optional[str] = None,
    output_dir: str = "output/relation_model"
):
    """Train Relation Extraction model"""

    # Load configuration
    config = OmegaConf.load(config_path)
    model_config = config.models.phobert  # Use PhoBERT for Vietnamese
    training_config = config.training

    # Load data
    logger.info("Loading training data...")
    train_data = load_relation_data(train_data_path)

    if val_data_path:
        val_data = load_relation_data(val_data_path)
    else:
        # Split train data for validation
        split_idx = int(len(train_data) * 0.9)
        val_data = train_data[split_idx:]
        train_data = train_data[:split_idx]

    logger.info(f"Train samples: {len(train_data)}")
    logger.info(f"Validation samples: {len(val_data)}")

    # Create label mappings
    label2id, id2label = create_relation_label_mappings()

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
    train_dataset = RelationExtractionDataset(train_data, tokenizer, label2id, model_config.max_length)
    val_dataset = RelationExtractionDataset(val_data, tokenizer, label2id, model_config.max_length)

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
            name=f"relation-extraction-{model_config.name.split('/')[-1]}",
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


def predict_relation(head_entity: Dict[str, Any], tail_entity: Dict[str, Any],
                    text: str, model_path: str, device: str = "auto") -> Dict[str, Any]:
    """Predict relation between two entities"""

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

    # Create input text with entity markers
    marked_text = text
    offset = 0

    # Insert tail entity marker first (higher position)
    tail_start = tail_entity["start"] + offset
    tail_end = tail_entity["end"] + offset
    marked_text = marked_text[:tail_start] + "[TAIL] " + marked_text[tail_start:tail_end] + " [/TAIL]" + marked_text[tail_end:]
    offset += len("[TAIL]  [/TAIL]")

    # Insert head entity marker
    head_start = head_entity["start"] + offset
    head_end = head_entity["end"] + offset
    marked_text = marked_text[:head_start] + "[HEAD] " + marked_text[head_start:head_end] + " [/HEAD]" + marked_text[head_end:]

    # Tokenize
    encoding = tokenizer(marked_text, return_tensors="pt", truncation=True, max_length=512)
    encoding = {k: v.to(device) for k, v in encoding.items()}

    # Predict
    with torch.no_grad():
        outputs = model(**encoding)
        predictions = torch.argmax(outputs.logits, dim=1)
        probabilities = torch.softmax(outputs.logits, dim=1)

    predicted_label = id2label[predictions[0].item()]
    confidence = probabilities[0][predictions[0]].item()

    return {
        "relation_type": predicted_label,
        "confidence": confidence,
        "all_probabilities": {
            id2label[i]: prob.item()
            for i, prob in enumerate(probabilities[0])
        }
    }


class RelationExtractor:
    """Complete Relation Extraction System"""

    def __init__(self, ner_model_path: str, relation_model_path: str):
        self.ner_model_path = ner_model_path
        self.relation_model_path = relation_model_path

        # Import NER prediction function
        from train_ner import predict_ner
        self.predict_ner = predict_ner

    def extract_relations(self, text: str) -> List[Dict[str, Any]]:
        """Extract all relations from text"""

        # Extract entities first
        entities = self.predict_ner(text, self.ner_model_path)

        relations = []

        # Predict relations between all entity pairs
        for i, head in enumerate(entities):
            for j, tail in enumerate(entities):
                if i != j:  # Don't predict relation with itself
                    relation_result = predict_relation(
                        head, tail, text, self.relation_model_path
                    )

                    if relation_result["relation_type"] != "NO_RELATION":
                        relations.append({
                            "head": head,
                            "tail": tail,
                            "relation_type": relation_result["relation_type"],
                            "confidence": relation_result["confidence"]
                        })

        return relations


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Fine-tune Relation Extraction Model")
    parser.add_argument("--config", required=True, help="Configuration file path")
    parser.add_argument("--train-data", required=True, help="Training data path")
    parser.add_argument("--val-data", help="Validation data path")
    parser.add_argument("--output-dir", default="output/relation_model", help="Output directory")
    parser.add_argument("--predict", help="Predict relations for text file instead of training")
    parser.add_argument("--model-path", help="Model path for prediction")
    parser.add_argument("--ner-model", help="NER model path for relation extraction")
    parser.add_argument("--extract-relations", help="Extract relations from text file (requires both NER and Relation models)")

    args = parser.parse_args()

    if args.predict:
        # Prediction mode for single relation
        if not args.model_path:
            logger.error("Model path required for prediction")
            return

        # This would require entity pairs as input
        logger.error("Prediction mode requires entity pairs. Use --extract-relations instead.")
        return

    elif args.extract_relations:
        # Complete relation extraction mode
        if not args.ner_model or not args.model_path:
            logger.error("Both NER model and Relation model paths required for relation extraction")
            return

        extractor = RelationExtractor(args.ner_model, args.model_path)

        with open(args.extract_relations, 'r', encoding='utf-8') as f:
            texts = f.readlines()

        for text in texts:
            text = text.strip()
            if text:
                relations = extractor.extract_relations(text)
                print(f"Text: {text}")
                print(f"Found {len(relations)} relations:")
                for rel in relations:
                    print(f"  {rel['head']['text']} --{rel['relation_type']}--> {rel['tail']['text']} "
                          f"(confidence: {rel['confidence']:.3f})")
                print("-" * 50)

    else:
        # Training mode
        trainer, eval_results = train_relation_extraction_model(
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