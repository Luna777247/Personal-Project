#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NER Demo Script - Lightweight version for testing
"""

import json
import logging
import os
import sys
from typing import Dict, List, Any, Optional

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_sample_articles() -> List[Dict[str, Any]]:
    """Load sample disaster articles for testing"""
    return [
        {
            'title': 'Bão số 12 gây thiệt hại nặng nề tại Hà Nội',
            'content': 'Bão số 12 đã đổ bộ vào khu vực Hà Nội hôm qua, gây ra lũ lụt nghiêm trọng. Hàng trăm ngôi nhà bị sạt lở.',
            'url': 'https://example.com/article1',
            'source': 'VNExpress'
        },
        {
            'title': 'Động đất mạnh tại Kon Tum',
            'content': 'Trận động đất có độ richter 5.2 xảy ra tại Kon Tum sáng nay, không có thiệt hại về người.',
            'url': 'https://example.com/article2',
            'source': 'TuoiTre'
        }
    ]

def create_extractor(model_name: str):
    """Create NER extractor based on model name"""
    try:
        # Load default config
        from config.nlp_config import MODEL_CONFIGS
        config = MODEL_CONFIGS.get(model_name, {})

        if model_name == 'phoner':
            from scripts.phoner_extractor import PhoNERExtractor
            return PhoNERExtractor(config)
        elif model_name == 'vncorenlp':
            from scripts.vncorenlp_extractor import VnCoreNLPExtractor
            return VnCoreNLPExtractor(config)
        elif model_name == 'spacy':
            from scripts.spacy_custom_extractor import SpacyCustomExtractor
            return SpacyCustomExtractor(config)
        elif model_name == 'bert':
            # Skip BERT for now due to dependency issues
            logger.warning("BERT NER skipped due to dependency issues")
            return None
        else:
            raise ValueError(f"Unknown model: {model_name}")
    except ImportError as e:
        logger.warning(f"Failed to import {model_name} extractor: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to create {model_name} extractor: {e}")
        return None

def run_model_demo(model_name: str, articles: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Run demo for a specific model"""
    logger.info(f"Running demo for model: {model_name}")

    extractor = create_extractor(model_name)
    if extractor is None:
        return None

    try:
        results = extractor.process_batch(articles)

        # Convert results to JSON-serializable format
        serializable_results = []
        for result in results:
            if hasattr(result, '__dict__'):
                # Convert dataclass to dict
                result_dict = {
                    'article_title': result.article_title,
                    'article_url': result.article_url,
                    'article_source': result.article_source,
                    'entities': [
                        {
                            'text': entity.text,
                            'label': entity.label,
                            'start': entity.start,
                            'end': entity.end,
                            'confidence': entity.confidence,
                            'context': getattr(entity, 'context', '')
                        } for entity in result.entities
                    ],
                    'processing_time': result.processing_time,
                    'model_used': result.model_used,
                    'confidence_score': result.confidence_score,
                    'entity_count': len(result.entities)
                }
                serializable_results.append(result_dict)
            else:
                serializable_results.append(result)

        # Save results
        output_file = f'data/ner_results_{model_name}.json'
        os.makedirs('data', exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)

        logger.info(f"Results saved to: {output_file}")
        return serializable_results

    except Exception as e:
        logger.error(f"Error running {model_name} demo: {e}")
        return None

def run_comparison_demo(articles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Run comparison across all available models"""
    logger.info("Running comparison demo across all models")

    models = ['phoner', 'vncorenlp', 'spacy']  # Skip BERT for now
    results = {}

    for model_name in models:
        logger.info(f"Testing model: {model_name}")
        model_results = run_model_demo(model_name, articles)
        if model_results:
            results[model_name] = model_results

    # Save comparison results
    comparison_file = 'data/ner_comparison_results.json'
    with open(comparison_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"Comparison results saved to: {comparison_file}")
    return results

def test_model_loading():
    """Test if models can be loaded"""
    logger.info("Testing model loading...")

    models = ['phoner', 'vncorenlp', 'spacy']  # Skip BERT
    loaded_models = {}

    for model_name in models:
        try:
            extractor = create_extractor(model_name)
            if extractor:
                loaded_models[model_name] = "Loaded successfully"
                logger.info(f"✅ {model_name}: Loaded successfully")
            else:
                loaded_models[model_name] = "Failed to create"
                logger.warning(f"⚠️  {model_name}: Failed to create")
        except Exception as e:
            loaded_models[model_name] = f"Error: {str(e)}"
            logger.error(f"❌ {model_name}: {e}")

    # Save loading results
    loading_file = 'data/model_loading_test.json'
    os.makedirs('data', exist_ok=True)

    with open(loading_file, 'w', encoding='utf-8') as f:
        json.dump(loaded_models, f, ensure_ascii=False, indent=2)

    logger.info(f"Loading test results saved to: {loading_file}")
    return loaded_models

if __name__ == "__main__":
    print("🚀 NER Entity Extraction Demo")
    print("=" * 50)

    # Test model loading
    print("🔍 Testing model loading...")
    loading_results = test_model_loading()

    print(f"📊 Loading Results: {len(loading_results)} models tested")

    # Load sample articles
    articles = load_sample_articles()
    print(f"📄 Loaded {len(articles)} sample articles")

    # Run comparison if any models loaded
    if loading_results:
        print("\n🏁 Running comparison demo...")
        comparison_results = run_comparison_demo(articles)
        print(f"📊 Comparison completed for {len(comparison_results)} models")
    else:
        print("❌ No models could be loaded")

    print("\n🎉 Demo completed!")