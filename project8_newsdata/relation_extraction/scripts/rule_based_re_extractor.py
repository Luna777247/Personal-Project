#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rule-based Relation Extraction
Sử dụng patterns và rules để trích xuất quan hệ
"""

import logging
import re
from typing import Dict, List, Any, Optional, Tuple, Pattern
import json

from .relation_extractor import RelationExtractor, Relation

logger = logging.getLogger(__name__)

class RuleBasedREExtractor(RelationExtractor):
    """Rule-based relation extraction using patterns"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.model_name = "Rule-Based-RE"

        # Load patterns from config
        self.patterns = config.get('patterns', self._get_default_patterns())

        # Entity placeholders
        self.entity_placeholders = config.get('entity_placeholders', {})

        # Compile regex patterns
        self.compiled_patterns = self._compile_patterns()

        logger.info("Initialized rule-based RE extractor")

    def _get_default_patterns(self) -> Dict[str, List[str]]:
        """Get default relation patterns"""
        return {
            'OCCURS_AT': [
                r'({disaster}) xảy ra tại ({location})',
                r'({disaster}) tại ({location})',
                r'({location}) hứng chịu ({disaster})',
                r'({location}) xảy ra ({disaster})'
            ],
            'OCCURS_ON': [
                r'({disaster}) vào lúc ({time})',
                r'({disaster}) vào ngày ({time})',
                r'({time}) xảy ra ({disaster})',
                r'({time}) có ({disaster})'
            ],
            'CAUSES_DAMAGE': [
                r'({disaster}) gây thiệt hại ({damage})',
                r'({disaster}) làm ({damage})',
                r'thiệt hại ({damage}) do ({disaster})',
                r'({damage}) từ ({disaster})'
            ],
            'HAS_INTENSITY': [
                r'({disaster}) có cấp ([0-9]+)',
                r'({disaster}) độ richter ([0-9.]+)',
                r'cường độ ({disaster}) ([0-9.]+)',
                r'({disaster}) cấp ([0-9]+)'
            ],
            'REPORTED_BY': [
                r'({disaster}) được báo cáo bởi ({organization})',
                r'({organization}) báo cáo về ({disaster})',
                r'theo ({organization}), ({disaster})'
            ]
        }

    def _compile_patterns(self) -> Dict[str, List[Tuple[Pattern, str, str]]]:
        """Compile regex patterns with entity types"""
        compiled = {}

        for relation_type, patterns in self.patterns.items():
            compiled[relation_type] = []

            for pattern in patterns:
                # Replace entity placeholders with regex groups
                processed_pattern = pattern

                # Replace placeholders like {disaster}, {location}, etc.
                for placeholder, values in self.entity_placeholders.items():
                    if f'{{{placeholder}}}' in processed_pattern:
                        # Create regex alternation for entity values
                        alternation = '|'.join(map(re.escape, values))
                        processed_pattern = processed_pattern.replace(
                            f'{{{placeholder}}}',
                            f'({alternation})'
                        )

                # If no placeholders were replaced, treat as generic pattern
                if '{' in processed_pattern:
                    # Replace remaining placeholders with generic word patterns
                    processed_pattern = re.sub(r'\{([^}]+)\}', r'([^,.!?]+)', processed_pattern)

                try:
                    compiled_pattern = re.compile(processed_pattern, re.IGNORECASE)
                    compiled[relation_type].append((compiled_pattern, pattern, processed_pattern))
                except re.error as e:
                    logger.warning(f"Failed to compile pattern '{pattern}': {e}")

        return compiled

    def _find_entities_in_text(self, text: str, entities: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Group entities by type found in text"""
        entity_by_type = {}

        for entity in entities:
            entity_type = entity.get('label', 'UNKNOWN')
            entity_text = entity['text']

            if entity_type not in entity_by_type:
                entity_by_type[entity_type] = []

            # Check if entity appears in text
            if entity_text.lower() in text.lower():
                entity_by_type[entity_type].append(entity_text)

        return entity_by_type

    def _extract_relations_from_patterns(self, text: str,
                                       entities_by_type: Dict[str, List[str]]) -> List[Relation]:
        """Extract relations using compiled patterns"""
        relations = []

        for relation_type, pattern_list in self.compiled_patterns.items():
            for pattern, original_pattern, processed_pattern in pattern_list:
                matches = pattern.findall(text)

                for match in matches:
                    if isinstance(match, tuple):
                        # Multiple groups in pattern
                        head_entity = match[0] if len(match) > 0 else ""
                        tail_entity = match[1] if len(match) > 1 else ""
                    else:
                        # Single group (shouldn't happen with our patterns)
                        continue

                    if not head_entity or not tail_entity:
                        continue

                    # Find confidence based on entity presence
                    confidence = self._calculate_confidence(
                        head_entity, tail_entity, entities_by_type
                    )

                    if confidence > 0.3:  # Minimum threshold
                        # Extract sentence containing the match
                        sentence = self._extract_sentence(text, head_entity, tail_entity)

                        relation = Relation(
                            head_entity=head_entity,
                            tail_entity=tail_entity,
                            relation_type=relation_type,
                            confidence=confidence,
                            context=original_pattern,
                            sentence=sentence
                        )
                        relations.append(relation)

        return relations

    def _calculate_confidence(self, head_entity: str, tail_entity: str,
                            entities_by_type: Dict[str, List[str]]) -> float:
        """Calculate confidence score for extracted relation"""
        confidence = 0.5  # Base confidence

        # Boost confidence if entities are in the recognized entity list
        head_found = any(head_entity in entities
                        for entities in entities_by_type.values())
        tail_found = any(tail_entity in entities
                        for entities in entities_by_type.values())

        if head_found and tail_found:
            confidence += 0.3
        elif head_found or tail_found:
            confidence += 0.1

        # Cap at 1.0
        return min(confidence, 1.0)

    def _extract_sentence(self, text: str, entity1: str, entity2: str) -> str:
        """Extract sentence containing both entities"""
        sentences = re.split(r'[.!?]+', text)

        for sentence in sentences:
            if entity1.lower() in sentence.lower() and entity2.lower() in sentence.lower():
                return sentence.strip()

        # Fallback: return first 200 characters
        return text[:200]

    def _apply_entity_type_mapping(self, relations: List[Relation],
                                 entities: List[Dict[str, Any]]) -> List[Relation]:
        """Map entity types to relations"""
        # Create entity text to type mapping
        entity_type_map = {e['text']: e.get('label', 'UNKNOWN') for e in entities}

        for relation in relations:
            relation.head_entity_type = entity_type_map.get(relation.head_entity, 'UNKNOWN')
            relation.tail_entity_type = entity_type_map.get(relation.tail_entity, 'UNKNOWN')

        return relations

    def extract_relations(self, text: str, entities: List[Dict[str, Any]]) -> List[Relation]:
        """
        Extract relations using rule-based patterns

        Args:
            text: Input text
            entities: List of entities from NER

        Returns:
            List of extracted relations
        """
        # Group entities by type
        entities_by_type = self._find_entities_in_text(text, entities)

        # Extract relations using patterns
        relations = self._extract_relations_from_patterns(text, entities_by_type)

        # Apply entity type mapping
        relations = self._apply_entity_type_mapping(relations, entities)

        # Remove duplicates
        seen = set()
        unique_relations = []
        for relation in relations:
            key = (relation.head_entity, relation.tail_entity, relation.relation_type)
            if key not in seen:
                seen.add(key)
                unique_relations.append(relation)

        # Filter by confidence
        min_confidence = self.config.get('min_confidence', 0.3)
        unique_relations = self.filter_relations_by_confidence(unique_relations, min_confidence)

        logger.info(f"Extracted {len(unique_relations)} relations using rule-based patterns")
        return unique_relations

    def add_pattern(self, relation_type: str, pattern: str):
        """Add a new pattern for relation extraction"""
        if relation_type not in self.patterns:
            self.patterns[relation_type] = []

        self.patterns[relation_type].append(pattern)

        # Recompile patterns
        self.compiled_patterns = self._compile_patterns()

        logger.info(f"Added pattern for {relation_type}: {pattern}")

    def get_pattern_stats(self) -> Dict[str, int]:
        """Get statistics about loaded patterns"""
        return {relation_type: len(patterns) for relation_type, patterns in self.patterns.items()}