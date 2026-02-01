"""
Pattern-Based Extractor for Disaster Information

This module implements a rule-based extraction system using regex patterns
and template matching for structured disaster information extraction.
"""

import re
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Pattern
from dataclasses import dataclass, asdict
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from config.patterns import ALL_PATTERNS, PATTERN_CATEGORIES, ExtractionPattern
from config.settings import EXTRACTION_SETTINGS, ENTITY_TYPE_MAPPING, TEMPLATE_RULES


@dataclass
class ExtractedEntity:
    """Represents an extracted entity with metadata"""
    text: str
    entity_type: str
    confidence: float
    start_pos: int
    end_pos: int
    context: str = ""
    pattern_name: str = ""
    template_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExtractionResult:
    """Container for extraction results"""
    extraction_id: str
    timestamp: str
    source_text: str
    entities: List[ExtractedEntity]
    metadata: Dict[str, Any]
    processing_time: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "extraction_id": self.extraction_id,
            "timestamp": self.timestamp,
            "source_text": self.source_text,
            "entities": [entity.to_dict() for entity in self.entities],
            "metadata": self.metadata,
            "processing_time": self.processing_time
        }


class PatternBasedExtractor:
    """
    Pattern-based extractor using regex patterns and template rules
    for structured disaster information extraction.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the pattern-based extractor.

        Args:
            config: Optional configuration override
        """
        self.config = EXTRACTION_SETTINGS.copy()
        if config:
            self.config.update(config)

        self.logger = self._setup_logging()
        self.patterns = ALL_PATTERNS
        self.templates = TEMPLATE_RULES

        # Pre-compile patterns for better performance
        self._compile_patterns()

        self.logger.info(f"Initialized PatternBasedExtractor with {len(self.patterns)} patterns")

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger(__name__)
        logger.setLevel(getattr(logging, self.config["log_level"]))

        if self.config["enable_console_logging"]:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        if self.config.get("log_file"):
            file_handler = logging.FileHandler(self.config["log_file"])
            file_handler.setLevel(logging.DEBUG)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

        return logger

    def _compile_patterns(self):
        """Pre-compile all regex patterns for performance"""
        for pattern in self.patterns:
            if not hasattr(pattern, 'compiled_pattern') or pattern.compiled_pattern is None:
                flags = re.IGNORECASE | re.UNICODE
                if not self.config["case_sensitive"]:
                    flags |= re.IGNORECASE
                pattern.compiled_pattern = re.compile(pattern.pattern, flags)

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text according to configuration.

        Args:
            text: Input text to preprocess

        Returns:
            Preprocessed text
        """
        if not self.config["preprocessing"]["normalize_unicode"]:
            return text

        # Normalize unicode characters
        import unicodedata
        text = unicodedata.normalize('NFC', text)

        if self.config["preprocessing"]["remove_extra_spaces"]:
            text = re.sub(r'\s+', ' ', text).strip()

        if self.config["preprocessing"]["standardize_numbers"]:
            # Standardize number formats
            text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)  # Remove spaces in numbers

        if self.config["preprocessing"]["lowercase_text"]:
            text = text.lower()

        return text

    def extract_entities(self, text: str) -> List[ExtractedEntity]:
        """
        Extract entities from text using pattern matching.

        Args:
            text: Input text to extract from

        Returns:
            List of extracted entities
        """
        start_time = time.time()
        self.logger.debug(f"Starting entity extraction on text of length {len(text)}")

        # Preprocess text
        processed_text = self.preprocess_text(text)

        entities = []

        # Extract using individual patterns
        pattern_entities = self._extract_with_patterns(processed_text, text)
        entities.extend(pattern_entities)

        # Extract using template rules
        template_entities = self._extract_with_templates(processed_text, text)
        entities.extend(template_entities)

        # Filter and deduplicate entities
        entities = self._filter_and_deduplicate(entities)

        processing_time = time.time() - start_time
        self.logger.info(f"Extracted {len(entities)} entities in {processing_time:.3f}s")

        return entities

    def _extract_with_patterns(self, processed_text: str, original_text: str) -> List[ExtractedEntity]:
        """Extract entities using individual regex patterns"""
        entities = []

        for pattern in self.patterns:
            matches = pattern.compiled_pattern.finditer(processed_text)

            for match in matches:
                # Get the matched text
                matched_text = match.group(0).strip()

                # Calculate position in original text
                start_pos = match.start()
                end_pos = match.end()

                # Extract context
                context = self._extract_context(original_text, start_pos, end_pos)

                entity = ExtractedEntity(
                    text=matched_text,
                    entity_type=pattern.entity_type,
                    confidence=pattern.confidence,
                    start_pos=start_pos,
                    end_pos=end_pos,
                    context=context,
                    pattern_name=pattern.name
                )

                entities.append(entity)

                # Limit matches per pattern
                if len([e for e in entities if e.pattern_name == pattern.name]) >= self.config["max_matches_per_type"]:
                    break

        return entities

    def _extract_with_templates(self, processed_text: str, original_text: str) -> List[ExtractedEntity]:
        """Extract entities using template rules"""
        entities = []

        for template_name, template_config in self.templates.items():
            template_pattern = re.compile(template_config["pattern"], re.IGNORECASE | re.UNICODE | re.DOTALL)
            matches = template_pattern.finditer(processed_text)

            for match in matches:
                template_text = match.group(0)

                # Extract sub-patterns within the template
                for sub_name, sub_pattern in template_config["sub_patterns"].items():
                    sub_compiled = re.compile(sub_pattern, re.IGNORECASE | re.UNICODE)
                    sub_matches = sub_compiled.finditer(template_text)

                    for sub_match in sub_matches:
                        matched_text = sub_match.group(0).strip()

                        # Calculate position in original text
                        template_start = match.start()
                        start_pos = template_start + sub_match.start()
                        end_pos = template_start + sub_match.end()

                        # Extract context
                        context = self._extract_context(original_text, start_pos, end_pos)

                        # Determine entity type based on sub-pattern name
                        entity_type = self._map_subpattern_to_entity_type(sub_name)

                        entity = ExtractedEntity(
                            text=matched_text,
                            entity_type=entity_type,
                            confidence=0.85,  # Template matches are generally reliable
                            start_pos=start_pos,
                            end_pos=end_pos,
                            context=context,
                            template_name=template_name
                        )

                        entities.append(entity)

        return entities

    def _extract_context(self, text: str, start_pos: int, end_pos: int) -> str:
        """Extract context around a match"""
        if not self.config["enable_context_extraction"]:
            return ""

        window = self.config["context_window_size"]
        context_start = max(0, start_pos - window)
        context_end = min(len(text), end_pos + window)

        context = text[context_start:context_end]

        # Add markers for the actual match
        relative_start = start_pos - context_start
        relative_end = end_pos - context_start

        marked_context = (
            context[:relative_start] +
            f"[{context[relative_start:relative_end]}]" +
            context[relative_end:]
        )

        return marked_context

    def _map_subpattern_to_entity_type(self, subpattern_name: str) -> str:
        """Map subpattern names to entity types"""
        mapping = {
            "deaths": "CASUALTY",
            "injured": "CASUALTY",
            "missing": "CASUALTY",
            "money_damage": "DAMAGE",
            "houses_destroyed": "DAMAGE",
            "infrastructure": "DAMAGE",
            "disaster_type": "DISASTER_TYPE",
            "location": "LOCATION",
            "impact": "DAMAGE"
        }
        return mapping.get(subpattern_name, "UNKNOWN")

    def _filter_and_deduplicate(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """Filter and deduplicate extracted entities"""
        # Sort by confidence (highest first)
        entities.sort(key=lambda x: x.confidence, reverse=True)

        # Remove duplicates based on text and type
        seen = set()
        filtered_entities = []

        for entity in entities:
            key = (entity.text.lower(), entity.entity_type)
            if key not in seen:
                seen.add(key)
                filtered_entities.append(entity)

        # Apply confidence threshold
        min_confidence = self.config["min_confidence"]
        filtered_entities = [
            entity for entity in filtered_entities
            if entity.confidence >= min_confidence
        ]

        return filtered_entities

    def extract_from_texts(self, texts: List[str], batch_size: Optional[int] = None) -> List[ExtractionResult]:
        """
        Extract entities from multiple texts with batch processing.

        Args:
            texts: List of texts to process
            batch_size: Optional batch size override

        Returns:
            List of extraction results
        """
        if batch_size is None:
            batch_size = self.config["batch_size"]

        results = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            self.logger.info(f"Processing batch {i//batch_size + 1} with {len(batch_texts)} texts")

            batch_results = self._process_batch(batch_texts)
            results.extend(batch_results)

        return results

    def _process_batch(self, texts: List[str]) -> List[ExtractionResult]:
        """Process a batch of texts"""
        results = []

        for text in texts:
            start_time = time.time()

            entities = self.extract_entities(text)
            processing_time = time.time() - start_time

            result = ExtractionResult(
                extraction_id=f"extract_{int(time.time() * 1000)}_{len(results)}",
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                source_text=text,
                entities=entities,
                metadata=self._generate_metadata(entities, processing_time),
                processing_time=processing_time
            )

            results.append(result)

        return results

    def _generate_metadata(self, entities: List[ExtractedEntity], processing_time: float) -> Dict[str, Any]:
        """Generate metadata for extraction results"""
        entity_counts = {}
        confidences = []

        for entity in entities:
            entity_counts[entity.entity_type] = entity_counts.get(entity.entity_type, 0) + 1
            confidences.append(entity.confidence)

        metadata = {
            "total_entities": len(entities),
            "entity_counts": entity_counts,
            "processing_time": processing_time,
            "patterns_used": list(set(entity.pattern_name for entity in entities if entity.pattern_name)),
            "templates_used": list(set(entity.template_name for entity in entities if entity.template_name))
        }

        if confidences:
            metadata["confidence_stats"] = {
                "mean": sum(confidences) / len(confidences),
                "min": min(confidences),
                "max": max(confidences)
            }

        return metadata

    def save_results(self, results: List[ExtractionResult], output_path: str):
        """
        Save extraction results to file.

        Args:
            output_path: Path to save results
        """
        output_data = [result.to_dict() for result in results]

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        self.logger.info(f"Saved {len(results)} extraction results to {output_path}")

    def get_pattern_stats(self) -> Dict[str, Any]:
        """Get statistics about patterns and their usage"""
        stats = {
            "total_patterns": len(self.patterns),
            "pattern_categories": {},
            "entity_types": set()
        }

        for category_name, category_patterns in PATTERN_CATEGORIES.items():
            if category_name != "all":
                stats["pattern_categories"][category_name] = len(category_patterns)

        for pattern in self.patterns:
            stats["entity_types"].add(pattern.entity_type)

        stats["entity_types"] = list(stats["entity_types"])

        return stats