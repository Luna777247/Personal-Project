#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PhoBERT Relation Extraction
Sử dụng PhoBERT fine-tuned cho việc trích xuất quan hệ
"""

import logging
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AdamW
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import os
import json

from .relation_extractor import RelationExtractor, Relation

logger = logging.getLogger(__name__)

class PhoBERTREModel(nn.Module):
    """PhoBERT model for relation extraction"""

    def __init__(self, model_name: str, num_labels: int):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

class PhoBERTREExtractor(RelationExtractor):
    """PhoBERT-based relation extraction"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.model_name = "PhoBERT-RE"
        self.device = self._get_device(config.get('device', 'auto'))
        self.max_length = config.get('max_length', 256)
        self.batch_size = config.get('batch_size', 16)
        self.save_path = config.get('save_path', 'models/phobert_re')

        # Load relation classes
        self.relation_classes = config.get('relation_classes', [])
        self.num_labels = len(self.relation_classes)
        self.label2id = {label: i for i, label in enumerate(self.relation_classes)}
        self.id2label = {i: label for label, i in self.label2id.items()}

        # Initialize model and tokenizer
        self.tokenizer = None
        self.model = None
        self._load_model()

        logger.info(f"Initialized PhoBERT RE extractor with device: {self.device}")

    def _get_device(self, device_config: str) -> torch.device:
        """Get torch device"""
        if device_config == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif device_config == 'cuda':
            if not torch.cuda.is_available():
                logger.warning("CUDA not available, falling back to CPU")
                return torch.device('cpu')
            return torch.device('cuda')
        else:
            return torch.device('cpu')

    def _load_model(self):
        """Load or initialize the model"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')

            # Try to load trained model
            if os.path.exists(self.save_path):
                logger.info(f"Loading trained model from {self.save_path}")
                self.model = PhoBERTREModel('vinai/phobert-base', self.num_labels)
                self.model.load_state_dict(torch.load(
                    os.path.join(self.save_path, 'pytorch_model.bin'),
                    map_location=self.device
                ))
            else:
                logger.info("No trained model found, using base PhoBERT")
                # Initialize with base model (no fine-tuning)
                self.model = PhoBERTREModel('vinai/phobert-base', self.num_labels)

            self.model.to(self.device)
            self.model.eval()

        except Exception as e:
            logger.error(f"Failed to load PhoBERT model: {e}")
            self.model = None

    def _create_relation_input(self, text: str, head_entity: str,
                             tail_entity: str) -> str:
        """
        Create input text for relation classification

        Args:
            text: Original text
            head_entity: Head entity text
            tail_entity: Tail entity text

        Returns:
            Formatted input text
        """
        # Simple template: "[CLS] head_entity [SEP] tail_entity [SEP] context [SEP]"
        # Find context around entities
        head_start = text.find(head_entity)
        tail_start = text.find(tail_entity)

        if head_start == -1 or tail_start == -1:
            context = text[:200]  # Fallback to beginning of text
        else:
            # Get context around both entities
            start = min(head_start, tail_start)
            end = max(head_start + len(head_entity), tail_start + len(tail_entity))
            context_start = max(0, start - 100)
            context_end = min(len(text), end + 100)
            context = text[context_start:context_end]

        # Create input with special tokens
        input_text = f"{head_entity} [SEP] {tail_entity} [SEP] {context}"
        return input_text

    def _predict_relation(self, input_text: str) -> Tuple[str, float]:
        """
        Predict relation for a single entity pair

        Args:
            input_text: Formatted input text

        Returns:
            Tuple of (relation_type, confidence)
        """
        if self.model is None or self.tokenizer is None:
            return "NO_RELATION", 0.0

        try:
            # Tokenize input
            inputs = self.tokenizer(
                input_text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Forward pass
            with torch.no_grad():
                logits = self.model(**inputs)
                probs = torch.softmax(logits, dim=1)
                pred_id = torch.argmax(probs, dim=1).item()
                confidence = probs[0][pred_id].item()

            # Get relation type
            relation_type = self.id2label.get(pred_id, "NO_RELATION")

            return relation_type, confidence

        except Exception as e:
            logger.error(f"Error predicting relation: {e}")
            return "NO_RELATION", 0.0

    def extract_relations(self, text: str, entities: List[Dict[str, Any]]) -> List[Relation]:
        """
        Extract relations from text given entities

        Args:
            text: Input text
            entities: List of entities from NER

        Returns:
            List of extracted relations
        """
        if self.model is None:
            logger.warning("PhoBERT model not loaded, returning empty relations")
            return []

        relations = []

        # Create entity pairs to check for relations
        entity_list = [(e['text'], e.get('label', '')) for e in entities]

        # Check all possible pairs (this could be optimized)
        for i, (head_text, head_type) in enumerate(entity_list):
            for j, (tail_text, tail_type) in enumerate(entity_list):
                if i == j:  # Skip self-relations
                    continue

                # Create input for this pair
                input_text = self._create_relation_input(text, head_text, tail_text)

                # Predict relation
                relation_type, confidence = self._predict_relation(input_text)

                if relation_type != "NO_RELATION" and confidence > 0.5:
                    # Find the sentence containing both entities
                    sentence = self._extract_sentence(text, head_text, tail_text)

                    relation = Relation(
                        head_entity=head_text,
                        tail_entity=tail_text,
                        relation_type=relation_type,
                        confidence=confidence,
                        context=input_text,
                        head_entity_type=head_type,
                        tail_entity_type=tail_type,
                        sentence=sentence
                    )
                    relations.append(relation)

        # Filter by confidence
        min_confidence = self.config.get('min_confidence', 0.5)
        relations = self.filter_relations_by_confidence(relations, min_confidence)

        logger.info(f"Extracted {len(relations)} relations using PhoBERT")
        return relations

    def _extract_sentence(self, text: str, entity1: str, entity2: str) -> str:
        """Extract sentence containing both entities"""
        # Simple sentence extraction
        sentences = text.split('.')
        for sentence in sentences:
            if entity1 in sentence and entity2 in sentence:
                return sentence.strip()
        return text[:200]  # Fallback

    def train(self, training_data: List[Dict[str, Any]], num_epochs: int = 10):
        """
        Fine-tune the model on training data

        Args:
            training_data: List of training examples
            num_epochs: Number of training epochs
        """
        if self.model is None or self.tokenizer is None:
            logger.error("Model not loaded, cannot train")
            return

        logger.info("Starting PhoBERT RE fine-tuning...")

        # Prepare training data
        train_inputs = []
        train_labels = []

        for example in training_data:
            text = example['text']
            for relation in example.get('relations', []):
                head_entity = relation['head']
                tail_entity = relation['tail']
                relation_type = relation['relation']

                input_text = self._create_relation_input(text, head_entity, tail_entity)
                label_id = self.label2id.get(relation_type, 0)  # Default to first class

                train_inputs.append(input_text)
                train_labels.append(label_id)

        # Create dataset
        train_encodings = self.tokenizer(
            train_inputs,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        train_labels = torch.tensor(train_labels)

        # Training setup
        optimizer = AdamW(self.model.parameters(), lr=self.config.get('learning_rate', 2e-5))
        criterion = nn.CrossEntropyLoss()

        self.model.train()

        # Training loop
        for epoch in range(num_epochs):
            epoch_loss = 0
            for i in range(0, len(train_encodings['input_ids']), self.batch_size):
                batch_inputs = {k: v[i:i+self.batch_size] for k, v in train_encodings.items()}
                batch_labels = train_labels[i:i+self.batch_size]

                batch_inputs = {k: v.to(self.device) for k, v in batch_inputs.items()}
                batch_labels = batch_labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(**batch_inputs)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

        # Save model
        os.makedirs(self.save_path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(self.save_path, 'pytorch_model.bin'))

        # Save config
        config = {
            'model_name': 'vinai/phobert-base',
            'num_labels': self.num_labels,
            'label2id': self.label2id,
            'id2label': self.id2label
        }
        with open(os.path.join(self.save_path, 'config.json'), 'w') as f:
            json.dump(config, f)

        logger.info(f"Model saved to {self.save_path}")