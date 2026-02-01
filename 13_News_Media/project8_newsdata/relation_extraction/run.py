#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Relation Extraction Runner
Convenience script to run RE demos and experiments

Usage:
    python run.py                    # Run full demo with all models
    python run.py --model rule       # Run only rule-based demo
    python run.py --model phobert    # Run only PhoBERT demo
    python run.py --model llm        # Run only LLM demo
    python run.py --compare          # Run comparison across all models
    python run.py --test-loading     # Test model loading only
"""

import argparse
import logging
import os
import sys
from typing import List, Dict, Any

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.demo_re import (
    load_sample_articles,
    load_sample_entities,
    run_model_demo,
    run_comparison_demo,
    test_model_loading
)

def main():
    parser = argparse.ArgumentParser(description='Relation Extraction Runner')
    parser.add_argument('--model', choices=['rule', 'phobert', 'llm'],
                       help='Run demo for specific model')
    parser.add_argument('--compare', action='store_true',
                       help='Run comparison across all models')
    parser.add_argument('--test-loading', action='store_true',
                       help='Test model loading only')

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("🚀 Relation Extraction Runner")
    print("=" * 50)

    if args.test_loading:
        # Test model loading only
        loading_results = test_model_loading()
        print(f"\n📊 Loading Results: {len(loading_results)} models tested")

        # Print summary
        for model, status in loading_results.items():
            status_icon = "✅" if "successfully" in status else "❌"
            print(f"{status_icon} {model}: {status}")

    elif args.model:
        # Run specific model demo
        articles = load_sample_articles()
        entities_batch = load_sample_entities()

        print(f"Running demo for model: {args.model}")
        result = run_model_demo(args.model, articles, entities_batch)

        if result:
            print(f"✅ {args.model} demo completed successfully")
        else:
            print(f"❌ {args.model} demo failed")

    elif args.compare:
        # Run comparison
        articles = load_sample_articles()
        entities_batch = load_sample_entities()

        print("Running comparison across all models...")
        results = run_comparison_demo(articles, entities_batch)

        print(f"✅ Comparison completed for {len(results)} models")

    else:
        # Run full demo (default)
        print("Running full demo...")

        # Test loading first
        loading_results = test_model_loading()

        if not loading_results:
            print("❌ No models could be loaded. Check dependencies and configuration.")
            return

        # Load sample data
        articles = load_sample_articles()
        entities_batch = load_sample_entities()

        # Run comparison
        results = run_comparison_demo(articles, entities_batch)

        print("\n🎉 Full demo completed!")
        print(f"📊 Results saved in data/ directory")
        print(f"📈 Models tested: {len(results)}")

if __name__ == "__main__":
    main()