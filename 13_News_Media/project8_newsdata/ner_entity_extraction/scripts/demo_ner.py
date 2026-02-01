#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NER Entity Extraction Demo
Demonstration of Named Entity Recognition for Disaster Information
Uses real data from disaster_data_multisource_20251207_165113.json

This script demonstrates the usage of different NER models for extracting
disaster-related entities from Vietnamese news articles.
"""

import logging
import json
import os
import time
from typing import List, Dict, Any
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.nlp_config import MODEL_CONFIGS, EXTRACTION_CONFIG
from .phoner_extractor import PhoNERExtractor
from .vncorenlp_extractor import VnCoreNLPExtractor
from .spacy_custom_extractor import SpacyCustomExtractor
from .bert_ner_extractor import BERTNERExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_sample_articles() -> List[Dict[str, Any]]:
    """
    Load real disaster data from JSON file
    
    Returns:
        List[Dict[str, Any]]: List of articles
    """
    try:
        # Try to load real data
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from real_data_loader import load_real_disaster_data, convert_to_articles
        
        records = load_real_disaster_data(limit=10)
        articles = convert_to_articles(records)
        logger.info(f"✅ Loaded {len(articles)} real articles from JSON")
        return articles
    except Exception as e:
        logger.warning(f"Could not load real data: {e}")
        logger.info("Using fallback sample articles...")
        return load_sample_articles_fallback()

def load_sample_articles_fallback() -> List[Dict[str, Any]]:
    """
    Load fallback sample articles
    """
    return [
        {
            "title": "Bão số 9 gây thiệt hại nặng tại các tỉnh miền Trung",
            "content": """Bão số 9 đã đổ bộ vào các tỉnh miền Trung vào sáng ngày 12/11,
            gây gió mạnh cấp 12-13, sóng biển cao 5-7m. Hàng trăm ngôi nhà bị tốc mái,
            nhiều diện tích lúa bị ngập úng. Theo báo cáo sơ bộ của Ban chỉ huy phòng
            chống thiên tai tỉnh Quảng Nam, có 3 người chết, 10 người bị thương.
            Thiệt hại ban đầu ước tính 500 tỷ đồng.""",
            "url": "https://vnexpress.net/bao-so-9-gay-thiet-hai-nang-123456",
            "source": "vnexpress"
        },
        {
            "title": "Động đất mạnh 6.5 độ Richter tại Kon Tum",
            "content": """Sáng nay 15/8, một trận động đất mạnh 6.5 độ Richter đã xảy ra
            tại huyện Kon Plông, tỉnh Kon Tum. Theo Trung tâm báo tin động đất và
            cảnh báo sóng thần, tâm chấn nằm ở phường Trường Chinh, thành phố Kon Tum,
            độ sâu khoảng 10km. Người dân địa phương cho biết cảm nhận được rung lắc
            mạnh khoảng 10-15 giây. Hiện chưa có báo cáo về thiệt hại.""",
            "url": "https://dantri.com.vn/động-đất-kon-tum-123456",
            "source": "dantri"
        },
        {
            "title": "Lũ quét tại Gia Lai làm 5 người mất tích",
            "content": """Mưa lớn kéo dài nhiều ngày qua đã gây lũ quét tại xã Ia Pal,
            huyện Chư Prông, tỉnh Gia Lai. Theo ông Nguyễn Văn Bình, Giám đốc Sở Tài
            nguyên và Môi trường Gia Lai, lũ quét đã cuốn trôi 3 ngôi nhà, làm 5 người
            mất tích. Đội cứu hộ đã được triển khai đến hiện trường. Nguyên nhân ban đầu
            được xác định là do mưa lớn kết hợp với địa hình núi rừng.""",
            "url": "https://tuoitre.vn/lu-quet-gia-lai-123456",
            "source": "tuoitre"
        }
    ]

def create_extractor(model_name: str) -> Any:
    """
    Create NER extractor instance

    Args:
        model_name: Name of the model to create

    Returns:
        NERExtractor instance or None
    """
    if model_name not in MODEL_CONFIGS:
        logger.error(f"Unknown model: {model_name}")
        return None

    config = MODEL_CONFIGS[model_name].copy()
    config.update(EXTRACTION_CONFIG)

    try:
        if model_name == "phoner":
            return PhoNERExtractor(config)
        elif model_name == "vncorenlp":
            return VnCoreNLPExtractor(config)
        elif model_name == "spacy_custom":
            return SpacyCustomExtractor(config)
        elif model_name == "bert_ner":
            return BERTNERExtractor(config)
        else:
            logger.error(f"Unsupported model: {model_name}")
            return None
    except Exception as e:
        logger.error(f"Failed to create {model_name} extractor: {str(e)}")
        return None

def run_model_demo(model_name: str, articles: List[Dict[str, Any]]) -> None:
    """
    Run demo for a specific NER model

    Args:
        model_name: Name of the model to test
        articles: List of articles to process
    """
    logger.info(f"\n🚀 Testing {model_name.upper()} Model")
    logger.info("=" * 60)

    # Create extractor
    extractor = create_extractor(model_name)
    if not extractor:
        logger.error(f"Failed to create {model_name} extractor")
        return

    # Load model
    if not extractor.load_model():
        logger.error(f"Failed to load {model_name} model")
        return

    # Process articles
    start_time = time.time()
    results = extractor.process_batch(articles)
    total_time = time.time() - start_time

    # Display results
    logger.info(f"\n📊 {model_name.upper()} RESULTS SUMMARY:")
    logger.info(f"   Total articles processed: {len(results)}")
    logger.info(f"   Total entities extracted: {sum(len(r.entities) for r in results)}")
    logger.info(f"   Total processing time: {total_time:.2f} seconds")
    logger.info(f"   Average time per article: {total_time/len(results):.2f} seconds")

    # Display detailed results
    for i, result in enumerate(results, 1):
        logger.info(f"\n📄 Article {i}: {result.article_title[:50]}...")
        logger.info(f"   Source: {result.article_source}")
        logger.info(f"   Entities found: {len(result.entities)}")
        logger.info(f"   Processing time: {result.processing_time:.2f}s")
        logger.info(f"   Confidence score: {result.confidence_score:.2f}")

        if result.entities:
            logger.info("   📋 Extracted Entities:")
            for entity in result.entities[:5]:  # Show first 5 entities
                logger.info(f"      • {entity.label}: '{entity.text}' (conf: {entity.confidence:.2f})")
            if len(result.entities) > 5:
                logger.info(f"      ... and {len(result.entities) - 5} more entities")
        else:
            logger.info("   ❌ No entities extracted")

    # Save results
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, f"ner_{model_name}_demo.json")
    csv_path = os.path.join(output_dir, f"ner_{model_name}_demo.csv")

    extractor.save_results(results, json_path)
    extractor.save_csv_results(results, csv_path)

    logger.info(f"\n💾 Results saved:")
    logger.info(f"   JSON: {json_path}")
    logger.info(f"   CSV: {csv_path}")

def run_comparison_demo(articles: List[Dict[str, Any]]) -> None:
    """
    Run comparison demo across all models

    Args:
        articles: List of articles to process
    """
    logger.info("\n🔄 COMPARISON ACROSS ALL MODELS")
    logger.info("=" * 80)

    model_names = ["phoner", "vncorenlp", "spacy_custom", "bert_ner"]
    comparison_results = {}

    for model_name in model_names:
        logger.info(f"\n🧪 Testing {model_name.upper()}...")

        extractor = create_extractor(model_name)
        if not extractor:
            logger.warning(f"Skipping {model_name} - failed to create")
            continue

        if not extractor.load_model():
            logger.warning(f"Skipping {model_name} - failed to load model")
            continue

        start_time = time.time()
        results = extractor.process_batch(articles)
        total_time = time.time() - start_time

        total_entities = sum(len(r.entities) for r in results)
        avg_confidence = sum(r.confidence_score for r in results) / len(results) if results else 0

        comparison_results[model_name] = {
            "articles_processed": len(results),
            "total_entities": total_entities,
            "total_time": total_time,
            "avg_time_per_article": total_time / len(results) if results else 0,
            "avg_confidence": avg_confidence,
            "entities_per_article": total_entities / len(results) if results else 0
        }

        logger.info(f"   ✅ {model_name.upper()}: {total_entities} entities, {total_time:.2f}s, conf: {avg_confidence:.2f}")

    # Display comparison table
    logger.info(f"\n📊 MODEL COMPARISON TABLE:")
    logger.info("-" * 100)
    logger.info("<20")
    logger.info("-" * 100)

    for model_name, stats in comparison_results.items():
        logger.info("<20")

    logger.info("-" * 100)

def main():
    """Main demo function"""
    logger.info("🚀 NER Entity Extraction Demo")
    logger.info("Named Entity Recognition for Disaster Information")
    logger.info("=" * 80)

    # Load sample articles
    articles = load_sample_articles()
    logger.info(f"📚 Loaded {len(articles)} sample disaster articles")

    # Run individual model demos
    model_names = ["phoner", "vncorenlp", "spacy_custom", "bert_ner"]

    for model_name in model_names:
        try:
            run_model_demo(model_name, articles)
        except Exception as e:
            logger.error(f"Error testing {model_name}: {str(e)}")
            continue

    # Run comparison
    try:
        run_comparison_demo(articles)
    except Exception as e:
        logger.error(f"Error in comparison demo: {str(e)}")

    logger.info("\n🎉 Demo completed!")
    logger.info("Check the 'data/' directory for output files")
    logger.info("Each model generates JSON and CSV results")

if __name__ == "__main__":
    main()