#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Relation Extraction Demo
Demo script cho việc trích xuất quan hệ
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
    data_file = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'disaster_data_multisource_20251207_165113.json')
    if os.path.exists(data_file):
        import json
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        articles = []
        for item in data[:5]:  # Load first 5 articles for demo
            article = {
                'title': item.get('title', ''),
                'content': item.get('content', ''),
                'url': item.get('url', ''),
                'source': item.get('source', '')
            }
            articles.append(article)
        return articles
    else:
        # Fallback to sample data
        return [
            {
                'title': 'Bão số 12 gây thiệt hại nặng nề tại Hà Nội',
                'content': 'Bão số 12 xảy ra tại Hà Nội vào ngày 15/10, gây thiệt hại 20 tỷ đồng. Cơn bão có cấp gió 12.',
                'url': 'https://example.com/article1',
                'source': 'VNExpress'
            },
            {
                'title': 'Động đất mạnh tại Kon Tum',
                'content': 'Động đất có độ richter 5.5 xảy ra tại Kon Tum sáng nay. Không có thiệt hại về người.',
                'url': 'https://example.com/article2',
                'source': 'TuoiTre'
            }
        ]

def load_sample_entities() -> List[List[Dict[str, Any]]]:
    """Load sample entities for each article"""
    articles = load_sample_articles()
    num_articles = len(articles)
    if num_articles == 2 and articles[0]['title'] == 'Bão số 12 gây thiệt hại nặng nề tại Hà Nội':
        # Sample data
        return [
            # Entities for article 1
            [
                {'text': 'Bão số 12', 'label': 'DISASTER_TYPE', 'start': 0, 'end': 9},
                {'text': 'Hà Nội', 'label': 'LOCATION', 'start': 35, 'end': 41},
                {'text': '15/10', 'label': 'TIME', 'start': 51, 'end': 56},
                {'text': '20 tỷ đồng', 'label': 'DAMAGE', 'start': 72, 'end': 82},
                {'text': 'cấp gió 12', 'label': 'QUANTITY', 'start': 85, 'end': 95}
            ],
            # Entities for article 2
            [
                {'text': 'Động đất', 'label': 'DISASTER_TYPE', 'start': 0, 'end': 9},
                {'text': 'Kon Tum', 'label': 'LOCATION', 'start': 32, 'end': 39},
                {'text': 'sáng nay', 'label': 'TIME', 'start': 40, 'end': 48},
                {'text': 'độ richter 5.5', 'label': 'QUANTITY', 'start': 13, 'end': 27}
            ]
        ]
    else:
        # For loaded data, return empty entities
        return [[] for _ in range(num_articles)]

def create_extractor(model_name: str):
    """Create relation extractor based on model name"""
    try:
        from config.re_config import MODEL_CONFIGS

        if model_name == 'phobert':
            from scripts.phobert_re_extractor import PhoBERTREExtractor
            config = MODEL_CONFIGS.get('phobert_re', {})
            return PhoBERTREExtractor(config)
        elif model_name == 'llm':
            from scripts.llm_re_extractor import LLMREExtractor
            config = MODEL_CONFIGS.get('llm_re', {})
            return LLMREExtractor(config)
        elif model_name == 'rule':
            from scripts.rule_based_re_extractor import RuleBasedREExtractor
            config = MODEL_CONFIGS.get('rule_based_re', {})
            return RuleBasedREExtractor(config)
        else:
            raise ValueError(f"Unknown model: {model_name}")
    except ImportError as e:
        logger.warning(f"Failed to import {model_name} extractor: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to create {model_name} extractor: {e}")
        return None

def run_model_demo(model_name: str, articles: List[Dict[str, Any]],
                  entities_batch: List[List[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    """Run demo for a specific model"""
    logger.info(f"Running demo for model: {model_name}")

    extractor = create_extractor(model_name)
    if extractor is None:
        return None

    try:
        results = extractor.process_batch(articles, entities_batch)

        # Save results
        output_file = f'data/re_results_{model_name}.json'
        os.makedirs('data', exist_ok=True)

        extractor.save_results(results, output_file)

        logger.info(f"Results saved to: {output_file}")
        return {'model': model_name, 'results': results}

    except Exception as e:
        logger.error(f"Error running {model_name} demo: {e}")
        return None

def run_comparison_demo(articles: List[Dict[str, Any]],
                       entities_batch: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Run comparison across all available models"""
    logger.info("Running comparison demo across all models")

    models = ['rule', 'phobert', 'llm']  # Rule-based first, then others
    results = {}

    for model_name in models:
        logger.info(f"Testing model: {model_name}")
        model_results = run_model_demo(model_name, articles, entities_batch)
        if model_results:
            results[model_name] = model_results

    # Save comparison summary
    comparison_file = 'data/re_comparison_summary.json'
    os.makedirs('data', exist_ok=True)

    summary = {
        'total_articles': len(articles),
        'models_tested': list(results.keys()),
        'results': {}
    }

    for model_name, model_data in results.items():
        model_results = model_data['results']
        total_relations = sum(len(r.relations) for r in model_results)
        avg_confidence = sum(r.confidence_score for r in model_results) / len(model_results) if model_results else 0

        summary['results'][model_name] = {
            'total_relations': total_relations,
            'avg_confidence': avg_confidence,
            'articles_processed': len(model_results)
        }

    with open(comparison_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info(f"Comparison summary saved to: {comparison_file}")
    return results

def test_model_loading():
    """Test if models can be loaded"""
    logger.info("Testing model loading...")

    models = ['rule', 'phobert', 'llm']
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
    loading_file = 'data/re_model_loading_test.json'
    os.makedirs('data', exist_ok=True)

    with open(loading_file, 'w', encoding='utf-8') as f:
        json.dump(loaded_models, f, ensure_ascii=False, indent=2)

    logger.info(f"Loading test results saved to: {loading_file}")
    return loaded_models

def print_sample_output():
    """Print sample of what the output looks like"""
    print("\n📄 Sample Output Format:")
    print("=" * 50)

    sample_output = {
        "article_title": "Bão số 12 gây thiệt hại nặng nề tại Hà Nội",
        "article_url": "https://example.com/article1",
        "relations": [
            {
                "head_entity": "Bão số 12",
                "tail_entity": "Hà Nội",
                "relation_type": "OCCURS_AT",
                "confidence": 0.85,
                "head_entity_type": "DISASTER_TYPE",
                "tail_entity_type": "LOCATION",
                "sentence": "Bão số 12 xảy ra tại Hà Nội vào ngày 15/10"
            },
            {
                "head_entity": "Bão số 12",
                "tail_entity": "15/10",
                "relation_type": "OCCURS_ON",
                "confidence": 0.92,
                "head_entity_type": "DISASTER_TYPE",
                "tail_entity_type": "TIME",
                "sentence": "Bão số 12 xảy ra tại Hà Nội vào ngày 15/10"
            }
        ],
        "processing_time": 0.15,
        "model_used": "Rule-Based-RE",
        "confidence_score": 0.885
    }

    print(json.dumps(sample_output, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    print("🚀 Relation Extraction Demo")
    print("=" * 50)

    # Show sample output format
    print_sample_output()

    print("\n🔍 Testing model loading...")
    loading_results = test_model_loading()

    print(f"📊 Loading Results: {len(loading_results)} models tested")

    # Load sample data
    articles = load_sample_articles()
    entities_batch = load_sample_entities()

    print(f"📄 Loaded {len(articles)} sample articles with entities")

    # Run comparison if any models loaded
    if loading_results:
        print("🏁 Running comparison demo...")
        comparison_results = run_comparison_demo(articles, entities_batch)
        print(f"📊 Comparison completed for {len(comparison_results)} models")
    else:
        print("❌ No models could be loaded")

    print("\n🎉 Demo completed!")
    print("📋 Next steps:")
    print("   1. Install dependencies: pip install -r requirements.txt")
    print("   2. Set up API keys for LLM models (.env file)")
    print("   3. Run full demo: python demo_re.py")
    print("   4. Check results in data/ directory")