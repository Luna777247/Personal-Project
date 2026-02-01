#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Relation Extractor Base Class
Lớp cơ sở cho việc trích xuất quan hệ giữa các entities
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import json
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class Relation:
    """Relation data structure"""
    head_entity: str
    tail_entity: str
    relation_type: str
    confidence: float
    context: str = ""
    head_entity_type: str = ""
    tail_entity_type: str = ""
    sentence: str = ""

@dataclass
class RelationResult:
    """Result of relation extraction for one article"""
    article_title: str
    article_url: str
    article_source: str
    relations: List[Relation]
    processing_time: float
    model_used: str
    confidence_score: float

class RelationExtractor(ABC):
    """Base class for relation extraction"""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize relation extractor

        Args:
            config: Configuration dictionary for the extractor
        """
        self.config = config
        self.model_name = config.get('model_name', 'Unknown')
        self.logger = logging.getLogger(self.__class__.__name__)

        # Load relation definitions
        try:
            from config.relation_definitions import RELATION_DEFINITIONS
            self.relation_definitions = RELATION_DEFINITIONS
        except ImportError:
            self.logger.warning("Could not load relation definitions")
            self.relation_definitions = {}

    @abstractmethod
    def extract_relations(self, text: str, entities: List[Dict[str, Any]]) -> List[Relation]:
        """
        Extract relations from text given entities

        Args:
            text: Input text
            entities: List of entities from NER

        Returns:
            List of extracted relations
        """
        pass

    def process_article(self, article: Dict[str, Any], entities: List[Dict[str, Any]]) -> RelationResult:
        """
        Process a single article for relation extraction

        Args:
            article: Article dictionary with title, content, url, source
            entities: List of entities from NER

        Returns:
            RelationResult object
        """
        start_time = time.time()

        try:
            # Combine title and content for better context
            full_text = f"{article.get('title', '')}. {article.get('content', '')}"

            # Extract relations
            relations = self.extract_relations(full_text, entities)

            processing_time = time.time() - start_time

            # Calculate average confidence
            avg_confidence = sum(r.confidence for r in relations) / len(relations) if relations else 0.0

            result = RelationResult(
                article_title=article.get('title', ''),
                article_url=article.get('url', ''),
                article_source=article.get('source', ''),
                relations=relations,
                processing_time=processing_time,
                model_used=self.model_name,
                confidence_score=avg_confidence
            )

            self.logger.info(f"Processed article '{article.get('title', '')[:50]}...' "
                           f"with {len(relations)} relations in {processing_time:.3f}s")

            return result

        except Exception as e:
            self.logger.error(f"Error processing article: {e}")
            processing_time = time.time() - start_time

            return RelationResult(
                article_title=article.get('title', ''),
                article_url=article.get('url', ''),
                article_source=article.get('source', ''),
                relations=[],
                processing_time=processing_time,
                model_used=self.model_name,
                confidence_score=0.0
            )

    def process_batch(self, articles: List[Dict[str, Any]],
                     entities_batch: List[List[Dict[str, Any]]]) -> List[RelationResult]:
        """
        Process a batch of articles

        Args:
            articles: List of article dictionaries
            entities_batch: List of entity lists for each article

        Returns:
            List of RelationResult objects
        """
        if len(articles) != len(entities_batch):
            raise ValueError("Articles and entities_batch must have same length")

        results = []
        total_start_time = time.time()

        self.logger.info(f"Processing batch of {len(articles)} articles with {self.model_name}")

        for i, (article, entities) in enumerate(zip(articles, entities_batch)):
            self.logger.info(f"Processing article {i+1}/{len(articles)}")
            result = self.process_article(article, entities)
            results.append(result)

        total_time = time.time() - total_start_time
        total_relations = sum(len(r.relations) for r in results)

        self.logger.info(f"Batch processing completed in {total_time:.3f}s. "
                        f"Total relations extracted: {total_relations}")

        return results

    def save_results(self, results: List[RelationResult], output_file: str):
        """
        Save results to JSON file

        Args:
            results: List of RelationResult objects
            output_file: Output file path
        """
        # Convert to serializable format
        serializable_results = []
        for result in results:
            result_dict = {
                'article_title': result.article_title,
                'article_url': result.article_url,
                'article_source': result.article_source,
                'relations': [
                    {
                        'head_entity': r.head_entity,
                        'tail_entity': r.tail_entity,
                        'relation_type': r.relation_type,
                        'confidence': r.confidence,
                        'context': r.context,
                        'head_entity_type': r.head_entity_type,
                        'tail_entity_type': r.tail_entity_type,
                        'sentence': r.sentence
                    } for r in result.relations
                ],
                'processing_time': result.processing_time,
                'model_used': result.model_used,
                'confidence_score': result.confidence_score,
                'relation_count': len(result.relations)
            }
            serializable_results.append(result_dict)

        # Create output directory if needed
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)

        self.logger.info(f"Results saved to: {output_file}")

    def filter_relations_by_confidence(self, relations: List[Relation],
                                     min_confidence: float = 0.5) -> List[Relation]:
        """
        Filter relations by confidence threshold

        Args:
            relations: List of relations
            min_confidence: Minimum confidence threshold

        Returns:
            Filtered list of relations
        """
        return [r for r in relations if r.confidence >= min_confidence]

    def get_supported_relations(self) -> List[str]:
        """
        Get list of supported relation types

        Returns:
            List of relation type names
        """
        return list(self.relation_definitions.keys())

    def validate_relation(self, head_entity: str, tail_entity: str,
                         relation_type: str) -> bool:
        """
        Validate if a relation is possible between two entities

        Args:
            head_entity: Head entity text
            tail_entity: Tail entity text
            relation_type: Relation type

        Returns:
            True if relation is valid
        """
        if relation_type not in self.relation_definitions:
            return False

        # Get entity types from config (this would need entity type info)
        # For now, return True if relation type exists
        return True