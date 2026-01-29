#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NER Entity Extraction Runner
Convenience script to run NER demos and experiments

Usage:
    python run.py                    # Run full demo with all models
    python run.py --model phoner     # Run only PhoNER demo
    python run.py --model vncorenlp  # Run only VnCoreNLP demo
    python run.py --model spacy      # Run only spaCy custom demo
    python run.py --model bert       # Run only BERT NER demo
    python run.py --compare          # Run comparison across all models
"""

import argparse
import logging
import os
import sys
from typing import List, Dict, Any

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.demo_ner import (
    load_sample_articles,
    run_model_demo,
    run_comparison_demo,
    create_extractor
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_single_model_demo(model_name: str) -> None:
    """
    Run demo for a single model

    Args:
        model_name: Name of the model to test
    """
    logger.info(f"🧪 Running {model_name.upper()} NER Demo")
    logger.info("=" * 60)

    # Load sample articles
    articles = load_sample_articles()
    logger.info(f"📚 Loaded {len(articles)} sample articles")

    # Run demo
    run_model_demo(model_name, articles)

def run_full_demo() -> None:
    """Run full demo with all models"""
    logger.info("🚀 Running Full NER Entity Extraction Demo")
    logger.info("Testing all NER models for disaster information extraction")
    logger.info("=" * 80)

    # Load sample articles
    articles = load_sample_articles()
    logger.info(f"📚 Loaded {len(articles)} sample disaster articles")

    # Run all model demos
    model_names = ["phoner", "vncorenlp", "spacy_custom", "bert_ner"]

    for model_name in model_names:
        try:
            run_model_demo(model_name, articles)
            logger.info(f"\n{'='*60}\n")
        except Exception as e:
            logger.error(f"Error testing {model_name}: {str(e)}")
            continue

    # Run comparison
    try:
        run_comparison_demo(articles)
    except Exception as e:
        logger.error(f"Error in comparison demo: {str(e)}")

def run_comparison_only() -> None:
    """Run only the comparison demo"""
    logger.info("🔄 Running NER Model Comparison")
    logger.info("=" * 60)

    # Load sample articles
    articles = load_sample_articles()
    logger.info(f"📚 Loaded {len(articles)} sample articles")

    # Run comparison
    run_comparison_demo(articles)

def test_model_loading(model_name: str) -> None:
    """
    Test model loading only

    Args:
        model_name: Name of the model to test
    """
    logger.info(f"🔍 Testing {model_name.upper()} Model Loading")
    logger.info("=" * 60)

    extractor = create_extractor(model_name)
    if not extractor:
        logger.error(f"❌ Failed to create {model_name} extractor")
        return

    if extractor.load_model():
        logger.info(f"✅ {model_name.upper()} model loaded successfully")

        # Show model info
        info = extractor.get_model_info()
        logger.info("📋 Model Information:")
        for key, value in info.items():
            if isinstance(value, dict):
                logger.info(f"   {key}: {len(value)} items")
            else:
                logger.info(f"   {key}: {value}")
    else:
        logger.error(f"❌ Failed to load {model_name} model")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="NER Entity Extraction Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                    # Run full demo with all models
  python run.py --model phoner     # Run only PhoNER demo
  python run.py --model vncorenlp  # Run only VnCoreNLP demo
  python run.py --model spacy      # Run only spaCy custom demo
  python run.py --model bert       # Run only BERT NER demo
  python run.py --compare          # Run comparison across all models
  python run.py --test phoner      # Test only PhoNER model loading
        """
    )

    parser.add_argument(
        "--model",
        choices=["phoner", "vncorenlp", "spacy", "bert"],
        help="Run demo for specific model only"
    )

    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run comparison across all models"
    )

    parser.add_argument(
        "--test",
        choices=["phoner", "vncorenlp", "spacy", "bert"],
        help="Test model loading only"
    )

    args = parser.parse_args()

    # Map short names to full names
    model_name_mapping = {
        "phoner": "phoner",
        "vncorenlp": "vncorenlp",
        "spacy": "spacy_custom",
        "bert": "bert_ner"
    }

    if args.test:
        # Test model loading
        model_name = model_name_mapping.get(args.test, args.test)
        test_model_loading(model_name)

    elif args.model:
        # Run single model demo
        model_name = model_name_mapping.get(args.model, args.model)
        run_single_model_demo(model_name)

    elif args.compare:
        # Run comparison only
        run_comparison_only()

    else:
        # Run full demo
        run_full_demo()

if __name__ == "__main__":
    main()