#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
spaCy Custom NER Extractor
spaCy with Custom Named Entity Recognition Model for Disaster Information

This module implements NER using spaCy with a custom trained model
for disaster-specific entity extraction.
"""

import logging
import os
import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding
import random
from typing import List, Dict, Any, Optional, Tuple
import json
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .ner_extractor import NERExtractor, Entity

logger = logging.getLogger(__name__)

class SpacyCustomExtractor(NERExtractor):
    """spaCy custom NER extractor for disaster information"""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize spaCy custom extractor

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.base_model = config.get('base_model', 'vi_core_news_lg')
        self.training_data_path = config.get('training_data_path', 'data/training_data.json')
        self.model_output_path = config.get('model_output_path', 'models/spacy_custom')
        self.nlp = None
        self.is_trained = False

        logger.info(f"Initialized spaCy custom extractor with base model: {self.base_model}")

    def load_model(self) -> bool:
        """
        Load spaCy model (train if not exists)

        Returns:
            bool: True if model loaded successfully
        """
        try:
            logger.info("Loading spaCy custom NER model...")

            # Check if trained model exists
            if os.path.exists(self.model_output_path) and os.path.exists(os.path.join(self.model_output_path, 'meta.json')):
                logger.info(f"Loading existing trained model from {self.model_output_path}")
                self.nlp = spacy.load(self.model_output_path)
                self.is_trained = True
            else:
                logger.info("Trained model not found, loading base model for training")
                # Load base model
                try:
                    self.nlp = spacy.load(self.base_model)
                except OSError:
                    logger.warning(f"Base model {self.base_model} not found, using blank model")
                    self.nlp = spacy.blank("vi")

                # Train the model
                self._train_model()

            self.is_loaded = True
            logger.info("spaCy custom NER model loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load spaCy model: {str(e)}")
            return False

    def _train_model(self) -> None:
        """
        Train custom NER model for disaster entities
        """
        logger.info("Starting model training...")

        # Create training data if not exists
        if not os.path.exists(self.training_data_path):
            self._create_training_data()

        # Load training data
        with open(self.training_data_path, 'r', encoding='utf-8') as f:
            training_data = json.load(f)

        # Prepare training examples
        examples = []
        for item in training_data:
            text = item['text']
            entities = item['entities']
            example = Example.from_dict(self.nlp.make_doc(text), {"entities": entities})
            examples.append(example)

        # Get NER component
        ner = self.nlp.get_pipe("ner")

        # Add labels
        for example in examples:
            for ent in example.reference.ents:
                ner.add_label(ent.label_)

        # Training configuration
        n_iter = 30
        batch_size = 4

        # Disable other pipes during training
        other_pipes = [pipe for pipe in self.nlp.pipe_names if pipe != "ner"]
        with self.nlp.disable_pipes(*other_pipes):
            optimizer = self.nlp.begin_training()

            for itn in range(n_iter):
                random.shuffle(examples)
                losses = {}

                # Batch training
                for batch in minibatch(examples, size=batch_size):
                    self.nlp.update(batch, drop=0.5, losses=losses)

                logger.info(f"Iteration {itn + 1}/{n_iter} - Losses: {losses}")

        # Save trained model
        os.makedirs(self.model_output_path, exist_ok=True)
        self.nlp.to_disk(self.model_output_path)
        self.is_trained = True

        logger.info(f"Model trained and saved to {self.model_output_path}")

    def _create_training_data(self) -> None:
        """
        Create sample training data for disaster NER
        """
        logger.info("Creating sample training data...")

        training_data = [
            {
                "text": "Bão số 9 đã đổ bộ vào tỉnh Quảng Nam vào sáng ngày 12/11, gây gió mạnh cấp 12, sóng biển cao 5-7m.",
                "entities": [
                    [0, 7, "DISASTER_TYPE"],  # Bão số 9
                    [25, 35, "LOCATION"],     # tỉnh Quảng Nam
                    [44, 55, "TIME"],         # sáng ngày 12/11
                    [65, 74, "QUANTITY"],     # cấp 12
                    [84, 90, "QUANTITY"]      # 5-7m
                ]
            },
            {
                "text": "Động đất mạnh 6.5 độ Richter xảy ra tại huyện Kon Plông, tỉnh Kon Tum vào lúc 14:30.",
                "entities": [
                    [0, 28, "DISASTER_TYPE"],  # Động đất mạnh 6.5 độ Richter
                    [42, 57, "LOCATION"],      # huyện Kon Plông
                    [59, 71, "LOCATION"],      # tỉnh Kon Tum
                    [79, 85, "TIME"]           # 14:30
                ]
            },
            {
                "text": "Theo Trung tâm dự báo khí tượng thủy văn, lũ quét đã gây thiệt hại 2 tỷ đồng và làm 5 người chết.",
                "entities": [
                    [5, 41, "ORGANIZATION"],   # Trung tâm dự báo khí tượng thủy văn
                    [43, 51, "DISASTER_TYPE"], # lũ quét
                    [67, 76, "DAMAGE"],        # 2 tỷ đồng
                    [87, 98, "DAMAGE"]         # 5 người chết
                ]
            },
            {
                "text": "Ông Nguyễn Văn An, Giám đốc Sở Tài nguyên và Môi trường Quảng Nam cho biết mưa lớn kéo dài gây ngập úng 500 ha lúa.",
                "entities": [
                    [4, 18, "PERSON"],         # Nguyễn Văn An
                    [20, 54, "ORGANIZATION"],  # Sở Tài nguyên và Môi trường Quảng Nam
                    [73, 82, "QUANTITY"]       # 500 ha
                ]
            }
        ]

        # Save training data
        os.makedirs(os.path.dirname(self.training_data_path), exist_ok=True)
        with open(self.training_data_path, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Training data saved to {self.training_data_path}")

    def extract_entities(self, text: str) -> List[Entity]:
        """
        Extract entities from text using spaCy custom model

        Args:
            text: Input text

        Returns:
            List[Entity]: List of extracted entities
        """
        if not self.is_loaded or self.nlp is None:
            logger.error("spaCy model not loaded")
            return []

        try:
            # Process text
            doc = self.nlp(text)

            entities = []
            for ent in doc.ents:
                # Map spaCy labels to our entity types
                label = self._map_spacy_label(ent.label_)
                if label:
                    entity = Entity(
                        text=ent.text,
                        label=label,
                        start=ent.start_char,
                        end=ent.end_char,
                        confidence=0.85,  # spaCy doesn't provide confidence for NER
                        context=self._get_context(text, ent.start_char, ent.end_char)
                    )
                    entities.append(entity)

            # Post-process entities
            entities = self._post_process_entities(entities, text)

            logger.info(f"Extracted {len(entities)} entities from text")
            return entities

        except Exception as e:
            logger.error(f"Error extracting entities: {str(e)}")
            return []

    def _map_spacy_label(self, spacy_label: str) -> Optional[str]:
        """
        Map spaCy label to our entity types

        Args:
            spacy_label: spaCy entity label

        Returns:
            Optional[str]: Mapped entity type or None
        """
        # spaCy labels depend on training data
        label_mapping = {
            'DISASTER_TYPE': 'DISASTER_TYPE',
            'LOCATION': 'LOCATION',
            'TIME': 'TIME',
            'DAMAGE': 'DAMAGE',
            'ORGANIZATION': 'ORGANIZATION',
            'PERSON': 'PERSON',
            'QUANTITY': 'QUANTITY',
            'GPE': 'LOCATION',  # Geo-political entity
            'ORG': 'ORGANIZATION',
            'PER': 'PERSON'
        }

        return label_mapping.get(spacy_label)

    def _post_process_entities(self, entities: List[Entity], text: str) -> List[Entity]:
        """
        Post-process extracted entities

        Args:
            entities: Raw extracted entities
            text: Original text

        Returns:
            List[Entity]: Post-processed entities
        """
        # For spaCy custom model, we mainly keep the trained entities
        # Additional post-processing can be added here if needed
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
        Get spaCy model information

        Returns:
            Dict[str, Any]: Model information
        """
        base_info = super().get_model_info()
        base_info.update({
            "base_model": self.base_model,
            "training_data_path": self.training_data_path,
            "model_output_path": self.model_output_path,
            "is_trained": self.is_trained,
            "spacy_version": spacy.__version__ if hasattr(spacy, '__version__') else 'unknown'
        })
        return base_info