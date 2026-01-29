#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VnCoreNLP Extractor
Official Vietnamese NLP Toolkit for Named Entity Recognition

This module implements NER using VnCoreNLP for Vietnamese disaster information.
"""

import logging
import os
import subprocess
import tempfile
import time
from typing import List, Dict, Any, Optional
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .ner_extractor import NERExtractor, Entity

logger = logging.getLogger(__name__)

class VnCoreNLPExtractor(NERExtractor):
    """VnCoreNLP-based NER extractor for Vietnamese"""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize VnCoreNLP extractor

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.model_path = config.get('model_path', 'VnCoreNLP-1.1.1.jar')
        self.annotators = config.get('annotators', ['wseg', 'pos', 'ner', 'parse'])
        self.java_path = self._find_java()
        self.vncorenlp_dir = None
        self.temp_dir = None

        logger.info(f"Initialized VnCoreNLP extractor")

    def _find_java(self) -> Optional[str]:
        """Find Java executable path"""
        import shutil
        java_path = shutil.which('java')
        if java_path:
            logger.info(f"Found Java at: {java_path}")
            return java_path
        else:
            logger.warning("Java not found in PATH")
            return None

    def load_model(self) -> bool:
        """
        Load VnCoreNLP model

        Returns:
            bool: True if model loaded successfully
        """
        try:
            logger.info("Loading VnCoreNLP model...")

            if not self.java_path:
                logger.error("Java not found, cannot load VnCoreNLP")
                return False

            # Try to import vncorenlp
            try:
                import vncorenlp
            except ImportError:
                logger.error("vncorenlp package not installed")
                return False

            # Download VnCoreNLP if not exists
            self.vncorenlp_dir = vncorenlp.download_model(save_dir='models')
            logger.info(f"VnCoreNLP model downloaded to: {self.vncorenlp_dir}")

            # Initialize VnCoreNLP
            self.vncorenlp = vncorenlp.VnCoreNLP(
                self.vncorenlp_dir,
                annotators=self.annotators,
                max_heap_size='-Xmx2g'
            )

            self.is_loaded = True
            logger.info("VnCoreNLP model loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load VnCoreNLP model: {str(e)}")
            return False

    def extract_entities(self, text: str) -> List[Entity]:
        """
        Extract entities from Vietnamese text using VnCoreNLP

        Args:
            text: Input Vietnamese text

        Returns:
            List[Entity]: List of extracted entities
        """
        if not self.is_loaded or self.vncorenlp is None:
            logger.error("VnCoreNLP model not loaded")
            return []

        try:
            # Annotate text
            annotated = self.vncorenlp.annotate(text)

            entities = []

            # Extract entities from annotated result
            if 'sentences' in annotated:
                for sentence in annotated['sentences']:
                    sentence_entities = self._extract_from_sentence(sentence, text)
                    entities.extend(sentence_entities)

            # Post-process entities
            entities = self._post_process_entities(entities, text)

            logger.info(f"Extracted {len(entities)} entities from text")
            return entities

        except Exception as e:
            logger.error(f"Error extracting entities: {str(e)}")
            return []

    def _extract_from_sentence(self, sentence: Dict[str, Any], full_text: str) -> List[Entity]:
        """
        Extract entities from a single sentence

        Args:
            sentence: Annotated sentence
            full_text: Full text for context

        Returns:
            List[Entity]: Entities from sentence
        """
        entities = []

        if 'words' not in sentence:
            return entities

        current_entity = None
        entity_words = []
        start_pos = 0

        for i, word in enumerate(sentence['words']):
            ner_tag = word.get('nerTag', 'O')

            if ner_tag.startswith('B-'):  # Beginning of entity
                # Save previous entity if exists
                if current_entity:
                    entity = self._create_entity(current_entity, entity_words, start_pos, full_text)
                    if entity:
                        entities.append(entity)

                # Start new entity
                current_entity = ner_tag[2:]  # Remove 'B-' prefix
                entity_words = [word['form']]
                start_pos = word.get('beginOffset', 0)

            elif ner_tag.startswith('I-') and current_entity:  # Inside entity
                if ner_tag[2:] == current_entity:  # Same entity type
                    entity_words.append(word['form'])
                else:
                    # Entity type changed, save previous and start new
                    if current_entity:
                        entity = self._create_entity(current_entity, entity_words, start_pos, full_text)
                        if entity:
                            entities.append(entity)
                    current_entity = ner_tag[2:]
                    entity_words = [word['form']]
                    start_pos = word.get('beginOffset', 0)

            else:  # Outside entity
                # Save previous entity if exists
                if current_entity:
                    entity = self._create_entity(current_entity, entity_words, start_pos, full_text)
                    if entity:
                        entities.append(entity)
                    current_entity = None
                    entity_words = []

        # Save last entity if exists
        if current_entity:
            entity = self._create_entity(current_entity, entity_words, start_pos, full_text)
            if entity:
                entities.append(entity)

        return entities

    def _create_entity(self, entity_type: str, words: List[str], start_pos: int, full_text: str) -> Optional[Entity]:
        """
        Create Entity object from extracted information

        Args:
            entity_type: Type of entity
            words: List of words in entity
            start_pos: Start position in text
            full_text: Full text

        Returns:
            Optional[Entity]: Created entity or None
        """
        if not words:
            return None

        entity_text = ' '.join(words)
        end_pos = start_pos + len(entity_text)

        # Map VnCoreNLP labels to our entity types
        mapped_label = self._map_vncorenlp_label(entity_type)

        if not mapped_label:
            return None

        return Entity(
            text=entity_text,
            label=mapped_label,
            start=start_pos,
            end=end_pos,
            confidence=0.8,  # VnCoreNLP doesn't provide confidence scores
            context=self._get_context(full_text, start_pos, end_pos)
        )

    def _map_vncorenlp_label(self, vncorenlp_label: str) -> Optional[str]:
        """
        Map VnCoreNLP label to our entity types

        Args:
            vncorenlp_label: VnCoreNLP entity label

        Returns:
            Optional[str]: Mapped entity type or None
        """
        # VnCoreNLP labels: LOCATION, PERSON, ORGANIZATION, etc.
        label_mapping = {
            'LOCATION': 'LOCATION',
            'PERSON': 'PERSON',
            'ORGANIZATION': 'ORGANIZATION',
            'MISC': None  # Will be handled by custom logic
        }

        return label_mapping.get(vncorenlp_label)

    def _post_process_entities(self, entities: List[Entity], text: str) -> List[Entity]:
        """
        Post-process extracted entities for disaster-specific information

        Args:
            entities: Raw extracted entities
            text: Original text

        Returns:
            List[Entity]: Post-processed entities
        """
        processed_entities = []

        for entity in entities:
            # Keep standard NER entities
            if entity.label in ['LOCATION', 'PERSON', 'ORGANIZATION']:
                processed_entities.append(entity)
                continue

            # Try to identify disaster-specific entities
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
        Get VnCoreNLP model information

        Returns:
            Dict[str, Any]: Model information
        """
        base_info = super().get_model_info()
        base_info.update({
            "model_path": self.model_path,
            "annotators": self.annotators,
            "java_path": self.java_path,
            "vncorenlp_dir": self.vncorenlp_dir,
            "vncorenlp_loaded": hasattr(self, 'vncorenlp') and self.vncorenlp is not None
        })
        return base_info