#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PhoNER Extractor
PhoBERT-based Named Entity Recognition for Vietnamese Disaster Information

This module implements NER using PhoNER (PhoBERT-based NER) for Vietnamese text.
"""

import logging
import os
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from typing import List, Dict, Any, Optional
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .ner_extractor import NERExtractor, Entity

logger = logging.getLogger(__name__)

class PhoNERExtractor(NERExtractor):
    """PhoNER-based NER extractor for Vietnamese"""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize PhoNER extractor

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.model_path = config.get('model_path', 'vinai/phobert-base')
        self.tokenizer_path = config.get('tokenizer_path', 'vinai/phobert-base')
        self.ner_model_path = config.get('ner_model_path', 'vinai/phobert-base-v2')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.pipeline = None

        logger.info(f"Initialized PhoNER extractor with device: {self.device}")

    def load_model(self) -> bool:
        """
        Load PhoNER model and tokenizer

        Returns:
            bool: True if model loaded successfully
        """
        try:
            logger.info("Loading PhoNER model...")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_path,
                use_fast=True
            )

            # Load NER model
            self.model = AutoModelForTokenClassification.from_pretrained(
                self.ner_model_path
            )

            # Move to device
            self.model.to(self.device)

            # Create NER pipeline
            self.pipeline = pipeline(
                "ner",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == 'cuda' else -1,
                aggregation_strategy="simple"
            )

            self.is_loaded = True
            logger.info("PhoNER model loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load PhoNER model: {str(e)}")
            return False

    def extract_entities(self, text: str) -> List[Entity]:
        """
        Extract entities from Vietnamese text using PhoNER

        Args:
            text: Input Vietnamese text

        Returns:
            List[Entity]: List of extracted entities
        """
        if not self.is_loaded or self.pipeline is None:
            logger.error("PhoNER model not loaded")
            return []

        try:
            # Run NER pipeline
            ner_results = self.pipeline(text)

            entities = []
            for result in ner_results:
                # Map PhoNER labels to our entity types
                label = self._map_phoner_label(result['entity_group'])
                if label:
                    entity = Entity(
                        text=result['word'],
                        label=label,
                        start=result['start'],
                        end=result['end'],
                        confidence=result['score'],
                        context=self._get_context(text, result['start'], result['end'])
                    )
                    entities.append(entity)

            # Post-process entities
            entities = self._post_process_entities(entities, text)

            logger.info(f"Extracted {len(entities)} entities from text")
            return entities

        except Exception as e:
            logger.error(f"Error extracting entities: {str(e)}")
            return []

    def _map_phoner_label(self, phoner_label: str) -> Optional[str]:
        """
        Map PhoNER label to our entity types

        Args:
            phoner_label: PhoNER entity label

        Returns:
            Optional[str]: Mapped entity type or None
        """
        # PhoNER labels: LOCATION, PERSON, ORGANIZATION, MISC
        label_mapping = {
            'LOCATION': 'LOCATION',
            'PERSON': 'PERSON',
            'ORGANIZATION': 'ORGANIZATION',
            'MISC': None  # Will be handled by custom logic
        }

        return label_mapping.get(phoner_label)

    def _post_process_entities(self, entities: List[Entity], text: str) -> List[Entity]:
        """
        Post-process extracted entities to identify disaster-specific entities

        Args:
            entities: Raw extracted entities
            text: Original text

        Returns:
            List[Entity]: Post-processed entities
        """
        processed_entities = []

        # Disaster-related keywords for context
        disaster_keywords = [
            'bão', 'lũ', 'động đất', 'sạt lở', 'sóng thần', 'hạn hán',
            'cháy', 'thiên tai', 'thiệt hại', 'chết', 'mất tích', 'bị thương'
        ]

        for entity in entities:
            # Keep standard NER entities
            if entity.label in ['LOCATION', 'PERSON', 'ORGANIZATION']:
                processed_entities.append(entity)
                continue

            # Try to identify disaster-specific entities from context
            context_words = entity.context.lower().split()

            # Check for disaster type
            if any(keyword in context_words for keyword in ['bão', 'lũ', 'động đất', 'sạt lở']):
                entity.label = 'DISASTER_TYPE'
                processed_entities.append(entity)

            # Check for time expressions
            elif any(keyword in context_words for keyword in ['ngày', 'tháng', 'năm', 'sáng', 'chiều']):
                entity.label = 'TIME'
                processed_entities.append(entity)

            # Check for damage information
            elif any(keyword in context_words for keyword in ['chết', 'mất tích', 'thiệt hại', 'bị thương']):
                entity.label = 'DAMAGE'
                processed_entities.append(entity)

            # Check for quantities
            elif any(char.isdigit() for char in entity.text):
                if any(keyword in context_words for keyword in ['độ', 'cấp', 'mét', 'km', 'mm']):
                    entity.label = 'QUANTITY'
                    processed_entities.append(entity)

        return processed_entities

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
        Get PhoNER model information

        Returns:
            Dict[str, Any]: Model information
        """
        base_info = super().get_model_info()
        base_info.update({
            "model_path": self.model_path,
            "tokenizer_path": self.tokenizer_path,
            "ner_model_path": self.ner_model_path,
            "device": self.device,
            "pipeline_loaded": self.pipeline is not None
        })
        return base_info