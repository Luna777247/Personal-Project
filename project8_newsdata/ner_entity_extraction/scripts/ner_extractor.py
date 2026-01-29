#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NER Entity Extractor Base Class
Named Entity Recognition for Disaster Information Extraction

This module provides the base class for NER-based entity extraction
from disaster news articles.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import json
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class Entity:
    """Entity data structure"""
    text: str
    label: str
    start: int
    end: int
    confidence: float
    context: str = ""

@dataclass
class ExtractionResult:
    """Result of entity extraction for one article"""
    article_title: str
    article_url: str
    article_source: str
    entities: List[Entity]
    processing_time: float
    model_used: str
    confidence_score: float

class NERExtractor(ABC):
    """Base class for NER-based entity extraction"""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize NER extractor

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model_name = config.get('name', 'Unknown')
        self.language = config.get('language', 'vi')
        self.supported_entities = config.get('supported_entities', [])
        self.custom_entities = config.get('custom_entities', [])
        self.is_loaded = False

        # Initialize logger
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    def load_model(self) -> bool:
        """
        Load the NER model

        Returns:
            bool: True if model loaded successfully
        """
        pass

    @abstractmethod
    def extract_entities(self, text: str) -> List[Entity]:
        """
        Extract entities from text

        Args:
            text: Input text

        Returns:
            List[Entity]: List of extracted entities
        """
        pass

    def process_article(self, article: Dict[str, Any]) -> ExtractionResult:
        """
        Process a single article

        Args:
            article: Article dictionary with title, content, url, source

        Returns:
            ExtractionResult: Extraction results
        """
        start_time = time.time()

        try:
            # Combine title and content for better context
            full_text = f"{article.get('title', '')}\n{article.get('content', '')}"

            # Extract entities
            entities = self.extract_entities(full_text)

            # Filter entities based on confidence threshold
            min_confidence = self.config.get('min_confidence', 0.5)
            filtered_entities = [
                entity for entity in entities
                if entity.confidence >= min_confidence
            ]

            processing_time = time.time() - start_time

            # Calculate average confidence
            avg_confidence = (
                sum(e.confidence for e in filtered_entities) / len(filtered_entities)
                if filtered_entities else 0.0
            )

            result = ExtractionResult(
                article_title=article.get('title', ''),
                article_url=article.get('url', ''),
                article_source=article.get('source', ''),
                entities=filtered_entities,
                processing_time=processing_time,
                model_used=self.model_name,
                confidence_score=avg_confidence
            )

            self.logger.info(f"Processed article '{article.get('title', '')[:50]}...' "
                           f"with {len(filtered_entities)} entities in {processing_time:.2f}s")

            return result

        except Exception as e:
            self.logger.error(f"Error processing article: {str(e)}")
            processing_time = time.time() - start_time

            return ExtractionResult(
                article_title=article.get('title', ''),
                article_url=article.get('url', ''),
                article_source=article.get('source', ''),
                entities=[],
                processing_time=processing_time,
                model_used=self.model_name,
                confidence_score=0.0
            )

    def process_batch(self, articles: List[Dict[str, Any]]) -> List[ExtractionResult]:
        """
        Process a batch of articles

        Args:
            articles: List of article dictionaries

        Returns:
            List[ExtractionResult]: List of extraction results
        """
        self.logger.info(f"Processing batch of {len(articles)} articles with {self.model_name}")

        results = []
        total_start_time = time.time()

        for i, article in enumerate(articles, 1):
            self.logger.info(f"Processing article {i}/{len(articles)}")
            result = self.process_article(article)
            results.append(result)

        total_time = time.time() - total_start_time
        self.logger.info(f"Batch processing completed in {total_time:.2f}s")

        return results

    def save_results(self, results: List[ExtractionResult], output_path: str) -> None:
        """
        Save extraction results to file

        Args:
            results: List of extraction results
            output_path: Output file path
        """
        # Convert to dictionary format
        results_dict = {
            "metadata": {
                "model": self.model_name,
                "total_articles": len(results),
                "total_entities": sum(len(r.entities) for r in results),
                "average_processing_time": sum(r.processing_time for r in results) / len(results),
                "average_confidence": sum(r.confidence_score for r in results) / len(results),
                "timestamp": time.time()
            },
            "results": []
        }

        for result in results:
            result_dict = {
                "article_info": {
                    "title": result.article_title,
                    "url": result.article_url,
                    "source": result.article_source
                },
                "extraction_info": {
                    "model_used": result.model_used,
                    "processing_time": result.processing_time,
                    "confidence_score": result.confidence_score,
                    "num_entities": len(result.entities)
                },
                "entities": [
                    {
                        "text": entity.text,
                        "label": entity.label,
                        "start": entity.start,
                        "end": entity.end,
                        "confidence": entity.confidence,
                        "context": entity.context
                    }
                    for entity in result.entities
                ]
            }
            results_dict["results"].append(result_dict)

        # Save to JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, ensure_ascii=False, indent=2)

        self.logger.info(f"Results saved to {output_path}")

    def save_csv_results(self, results: List[ExtractionResult], output_path: str) -> None:
        """
        Save extraction results to CSV format

        Args:
            results: List of extraction results
            output_path: Output file path
        """
        rows = []

        for result in results:
            for entity in result.entities:
                row = {
                    "article_title": result.article_title,
                    "article_url": result.article_url,
                    "article_source": result.article_source,
                    "model_used": result.model_used,
                    "entity_text": entity.text,
                    "entity_label": entity.label,
                    "entity_start": entity.start,
                    "entity_end": entity.end,
                    "confidence": entity.confidence,
                    "context": entity.context,
                    "processing_time": result.processing_time
                }
                rows.append(row)

        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
            self.logger.info(f"CSV results saved to {output_path} with {len(rows)} rows")
        else:
            self.logger.warning("No entities found, CSV file not created")

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model

        Returns:
            Dict[str, Any]: Model information
        """
        return {
            "name": self.model_name,
            "language": self.language,
            "supported_entities": self.supported_entities,
            "custom_entities": self.custom_entities,
            "is_loaded": self.is_loaded,
            "config": self.config
        }