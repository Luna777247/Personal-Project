#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BERT NER Extractor
BERT-based Named Entity Recognition with Fine-tuning for Disaster Information

This module implements NER using BERT models with fine-tuning
for disaster-specific entity extraction.
"""

import logging
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    Trainer, TrainingArguments, DataCollatorForTokenClassification
)
from typing import List, Dict, Any, Optional, Tuple
import json
import numpy as np
from seqeval.metrics import classification_report
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .ner_extractor import NERExtractor, Entity

logger = logging.getLogger(__name__)

class BERTNERExtractor(NERExtractor):
    """BERT-based NER extractor with fine-tuning for disaster information"""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize BERT NER extractor

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.base_model = config.get('base_model', 'bert-base-multilingual-cased')
        self.vietnamese_model = config.get('vietnamese_model', 'vinai/phobert-base')
        self.max_length = config.get('max_length', 256)
        self.batch_size = config.get('batch_size', 16)
        self.learning_rate = config.get('learning_rate', 5e-5)
        self.num_epochs = config.get('num_epochs', 10)
        self.model_output_path = config.get('model_output_path', 'models/bert_ner')

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = None
        self.model = None
        self.label2id = {}
        self.id2label = {}

        logger.info(f"Initialized BERT NER extractor with device: {self.device}")

    def load_model(self) -> bool:
        """
        Load BERT NER model (train if not exists)

        Returns:
            bool: True if model loaded successfully
        """
        try:
            logger.info("Loading BERT NER model...")

            # Check if trained model exists
            if os.path.exists(self.model_output_path) and os.path.exists(os.path.join(self.model_output_path, 'pytorch_model.bin')):
                logger.info(f"Loading existing trained model from {self.model_output_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_output_path)
                self.model = AutoModelForTokenClassification.from_pretrained(self.model_output_path)
                self._load_label_mappings()
            else:
                logger.info("Trained model not found, loading base model for training")
                # Load base model
                self.tokenizer = AutoTokenizer.from_pretrained(self.vietnamese_model)
                self.model = AutoModelForTokenClassification.from_pretrained(
                    self.vietnamese_model,
                    num_labels=len(self.supported_entities) * 2 + 1  # BIO format
                )

                # Create label mappings
                self._create_label_mappings()

                # Train the model
                self._train_model()

            # Move to device
            self.model.to(self.device)

            self.is_loaded = True
            logger.info("BERT NER model loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load BERT NER model: {str(e)}")
            return False

    def _create_label_mappings(self) -> None:
        """Create label mappings for BIO format"""
        labels = ["O"]  # Outside

        for entity_type in self.supported_entities:
            labels.extend([f"B-{entity_type}", f"I-{entity_type}"])

        self.label2id = {label: i for i, label in enumerate(labels)}
        self.id2label = {i: label for label, i in self.label2id.items()}

        # Update model config
        self.model.config.label2id = self.label2id
        self.model.config.id2label = self.id2label

        logger.info(f"Created label mappings: {len(labels)} labels")

    def _load_label_mappings(self) -> None:
        """Load label mappings from saved model"""
        config_path = os.path.join(self.model_output_path, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                self.label2id = config.get('label2id', {})
                self.id2label = config.get('id2label', {})

    def _train_model(self) -> None:
        """
        Train BERT NER model for disaster entities
        """
        logger.info("Starting BERT NER model training...")

        # Create training dataset
        train_dataset = self._create_training_dataset()

        if len(train_dataset) == 0:
            logger.warning("No training data available, skipping training")
            return

        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.model_output_path,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            weight_decay=0.01,
            logging_steps=10,
            save_steps=500,
            evaluation_strategy="no",
            save_strategy="epoch",
            load_best_model_at_end=False,
        )

        # Data collator
        data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)

        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )

        # Train
        trainer.train()

        # Save model
        trainer.save_model(self.model_output_path)
        self.tokenizer.save_pretrained(self.model_output_path)

        logger.info(f"BERT NER model trained and saved to {self.model_output_path}")

    def _create_training_dataset(self) -> Dataset:
        """
        Create training dataset from sample data

        Returns:
            Dataset: Training dataset
        """
        # Sample training data for disaster NER
        training_texts = [
            "Bão số 9 đã đổ bộ vào tỉnh Quảng Nam vào sáng ngày 12/11, gây gió mạnh cấp 12.",
            "Động đất mạnh 6.5 độ Richter xảy ra tại huyện Kon Plông, tỉnh Kon Tum.",
            "Theo Trung tâm dự báo khí tượng thủy văn, lũ quét đã gây thiệt hại 2 tỷ đồng.",
            "Ông Nguyễn Văn An, Giám đốc Sở Tài nguyên và Môi trường Quảng Nam cho biết mưa lớn kéo dài.",
            "Sạt lở đất tại xã Ea H'leo, huyện Ea H'leo, tỉnh Đắk Lắk làm 3 người chết.",
            "Hạn hán kéo dài tại các tỉnh miền Trung gây thiếu nước nghiêm trọng.",
            "Cháy rừng tại khu vực biên giới Việt - Lào thiêu rụi hàng trăm hecta rừng."
        ]

        # Convert to BIO format (simplified)
        training_data = []
        for text in training_texts:
            # Tokenize
            tokens = self.tokenizer.tokenize(text)
            labels = ["O"] * len(tokens)  # Simplified: all outside

            training_data.append({
                "tokens": tokens,
                "labels": labels
            })

        # Create dataset
        class NERDataset(Dataset):
            def __init__(self, data, tokenizer, label2id, max_length):
                self.data = data
                self.tokenizer = tokenizer
                self.label2id = label2id
                self.max_length = max_length

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                item = self.data[idx]
                tokens = item["tokens"]
                labels = item["labels"]

                # Convert tokens to ids
                input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                label_ids = [self.label2id.get(label, 0) for label in labels]

                # Padding
                padding_length = self.max_length - len(input_ids)
                if padding_length > 0:
                    input_ids += [self.tokenizer.pad_token_id] * padding_length
                    label_ids += [-100] * padding_length  # Ignore index for loss

                return {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "labels": torch.tensor(label_ids, dtype=torch.long),
                    "attention_mask": torch.tensor([1] * (self.max_length - padding_length) + [0] * padding_length, dtype=torch.long)
                }

        return NERDataset(training_data, self.tokenizer, self.label2id, self.max_length)

    def extract_entities(self, text: str) -> List[Entity]:
        """
        Extract entities from text using BERT NER model

        Args:
            text: Input text

        Returns:
            List[Entity]: List of extracted entities
        """
        if not self.is_loaded or self.model is None or self.tokenizer is None:
            logger.error("BERT NER model not loaded")
            return []

        try:
            # Tokenize text
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True
            ).to(self.device)

            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=-1)

            # Convert predictions to entities
            entities = self._predictions_to_entities(predictions[0], inputs, text)

            # Post-process entities
            entities = self._post_process_entities(entities, text)

            logger.info(f"Extracted {len(entities)} entities from text")
            return entities

        except Exception as e:
            logger.error(f"Error extracting entities: {str(e)}")
            return []

    def _predictions_to_entities(self, predictions: torch.Tensor, inputs: Dict[str, torch.Tensor], text: str) -> List[Entity]:
        """
        Convert model predictions to entities

        Args:
            predictions: Model predictions
            inputs: Tokenized inputs
            text: Original text

        Returns:
            List[Entity]: Extracted entities
        """
        entities = []
        current_entity = None
        entity_tokens = []
        start_pos = 0

        # Get tokens and predictions
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        pred_labels = [self.id2label.get(pred.item(), "O") for pred in predictions]

        for i, (token, label) in enumerate(zip(tokens, pred_labels)):
            if token in ["[CLS]", "[SEP]", "[PAD]"]:
                continue

            if label.startswith("B-"):  # Beginning of entity
                # Save previous entity
                if current_entity:
                    entity = self._create_entity_from_tokens(current_entity, entity_tokens, text, start_pos)
                    if entity:
                        entities.append(entity)

                # Start new entity
                current_entity = label[2:]  # Remove 'B-' prefix
                entity_tokens = [token]
                start_pos = self._get_token_position(text, tokens, i)

            elif label.startswith("I-") and current_entity:  # Inside entity
                if label[2:] == current_entity:  # Same entity type
                    entity_tokens.append(token)
                else:
                    # Entity type changed, save previous and start new
                    if current_entity:
                        entity = self._create_entity_from_tokens(current_entity, entity_tokens, text, start_pos)
                        if entity:
                            entities.append(entity)
                    current_entity = label[2:]
                    entity_tokens = [token]
                    start_pos = self._get_token_position(text, tokens, i)

            else:  # Outside entity
                # Save previous entity
                if current_entity:
                    entity = self._create_entity_from_tokens(current_entity, entity_tokens, text, start_pos)
                    if entity:
                        entities.append(entity)
                    current_entity = None
                    entity_tokens = []

        # Save last entity
        if current_entity:
            entity = self._create_entity_from_tokens(current_entity, entity_tokens, text, start_pos)
            if entity:
                entities.append(entity)

        return entities

    def _create_entity_from_tokens(self, entity_type: str, tokens: List[str], text: str, start_pos: int) -> Optional[Entity]:
        """
        Create Entity object from tokens

        Args:
            entity_type: Type of entity
            tokens: List of tokens
            text: Full text
            start_pos: Start position

        Returns:
            Optional[Entity]: Created entity or None
        """
        if not tokens:
            return None

        # Reconstruct text from tokens (simplified)
        entity_text = self.tokenizer.convert_tokens_to_string(tokens)
        end_pos = start_pos + len(entity_text)

        return Entity(
            text=entity_text,
            label=entity_type,
            start=start_pos,
            end=end_pos,
            confidence=0.8,  # Simplified confidence
            context=self._get_context(text, start_pos, end_pos)
        )

    def _get_token_position(self, text: str, tokens: List[str], token_idx: int) -> int:
        """
        Get approximate position of token in text

        Args:
            text: Full text
            tokens: List of tokens
            token_idx: Index of token

        Returns:
            int: Approximate start position
        """
        # Simplified: find approximate position
        # In practice, you'd need proper token-to-text alignment
        return token_idx * 5  # Rough approximation

    def _post_process_entities(self, entities: List[Entity], text: str) -> List[Entity]:
        """
        Post-process extracted entities

        Args:
            entities: Raw extracted entities
            text: Original text

        Returns:
            List[Entity]: Post-processed entities
        """
        # Additional post-processing can be added here
        return entities

    def _get_context(self, text: str, start: int, end: int, window: int = 50) -> str:
        """
        Get context around entity

        Args:
            text: Full text
            start: Entity start position
            end: Entity end position
            window: Context window size

        Returns:
            str: Context string
        """
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)

        return text[context_start:context_end].strip()

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get BERT NER model information

        Returns:
            Dict[str, Any]: Model information
        """
        base_info = super().get_model_info()
        base_info.update({
            "base_model": self.base_model,
            "vietnamese_model": self.vietnamese_model,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "num_epochs": self.num_epochs,
            "model_output_path": self.model_output_path,
            "device": self.device,
            "num_labels": len(self.label2id),
            "label2id": self.label2id,
            "id2label": self.id2label
        })
        return base_info