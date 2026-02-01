#!/usr/bin/env python3
"""
Pattern Extraction Runner

Command-line interface for running pattern-based extraction demos
and processing disaster news articles.
"""

import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from scripts.demo_pattern_extraction import (
    run_single_extraction_demo,
    run_batch_extraction_demo,
    run_pattern_analysis_demo,
    run_custom_pattern_demo
)
from scripts.pattern_extractor import PatternBasedExtractor


def setup_environment():
    """Setup environment and check dependencies"""
    print("🔧 Setting up Pattern Extraction environment...")

    # Check if we're in the right directory
    if not Path("config/patterns.py").exists():
        print("❌ Error: Please run this script from the pattern_extraction directory")
        sys.exit(1)

    print("✅ Environment ready")


def run_demo(args):
    """Run demo modes"""
    print("🚀 Running Pattern Extraction Demo")
    print("=" * 50)

    if args.mode == "single":
        run_single_extraction_demo()
    elif args.mode == "batch":
        run_batch_extraction_demo()
    elif args.mode == "analysis":
        run_pattern_analysis_demo()
    elif args.mode == "custom":
        run_custom_pattern_demo()
    elif args.mode == "all":
        print("Running all demo modes...")
        run_single_extraction_demo()
        run_batch_extraction_demo()
        run_pattern_analysis_demo()
        run_custom_pattern_demo()
    else:
        print(f"❌ Unknown demo mode: {args.mode}")


def run_extraction(args):
    """Run extraction on custom input"""
    print("🔍 Running Pattern Extraction on custom input")
    print("=" * 50)

    # Initialize extractor
    extractor = PatternBasedExtractor()

    # Read input
    if args.input_file:
        print(f"Reading from file: {args.input_file}")
        with open(args.input_file, 'r', encoding='utf-8') as f:
            text = f.read()
    elif args.text:
        text = args.text
    else:
        print("❌ Error: Please provide either --input-file or --text")
        return

    print(f"Processing text of length: {len(text)} characters")

    # Extract entities
    entities = extractor.extract_entities(text)

    print(f"\n📊 Found {len(entities)} entities:")

    # Group by type
    entities_by_type = {}
    for entity in entities:
        if entity.entity_type not in entities_by_type:
            entities_by_type[entity.entity_type] = []
        entities_by_type[entity.entity_type].append(entity)

    # Display results
    for entity_type, type_entities in entities_by_type.items():
        print(f"\n{entity_type}:")
        for entity in type_entities:
            confidence_pct = int(entity.confidence * 100)
            print(f"  • '{entity.text}' (confidence: {confidence_pct}%)")
            if args.show_context and entity.context:
                print(f"    Context: {entity.context}")

    # Save results if requested
    if args.output:
        results = extractor.extract_from_texts([text])
        extractor.save_results(results, args.output)
        print(f"\n💾 Results saved to: {args.output}")


def show_patterns(args):
    """Show available patterns"""
    print("📋 Available Extraction Patterns")
    print("=" * 50)

    from config.patterns import PATTERN_CATEGORIES, ALL_PATTERNS

    print(f"Total patterns: {len(ALL_PATTERNS)}")

    for category_name, patterns in PATTERN_CATEGORIES.items():
        if category_name != "all":
            print(f"\n{category_name.upper()} ({len(patterns)} patterns):")
            for pattern in patterns:
                print(f"  • {pattern.name}: {pattern.pattern}")
                if pattern.examples:
                    print(f"    Examples: {', '.join(pattern.examples)}")


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="Pattern-Based Extraction for Disaster Information",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all demos
  python run.py demo --mode all

  # Extract from text
  python run.py extract --text "Bão số 12 khiến 15 người chết tại Quảng Nam"

  # Extract from file
  python run.py extract --input-file news.txt --output results.json

  # Show available patterns
  python run.py patterns
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run extraction demos')
    demo_parser.add_argument(
        '--mode',
        choices=['single', 'batch', 'analysis', 'custom', 'all'],
        default='all',
        help='Demo mode to run'
    )

    # Extract command
    extract_parser = subparsers.add_parser('extract', help='Extract entities from text')
    extract_parser.add_argument(
        '--text',
        type=str,
        help='Text to extract entities from'
    )
    extract_parser.add_argument(
        '--input-file',
        type=str,
        help='File containing text to extract from'
    )
    extract_parser.add_argument(
        '--output',
        type=str,
        help='Output file to save results (JSON format)'
    )
    extract_parser.add_argument(
        '--show-context',
        action='store_true',
        help='Show context around extracted entities'
    )

    # Patterns command
    patterns_parser = subparsers.add_parser('patterns', help='Show available patterns')

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Setup environment
    setup_environment()

    # Execute command
    try:
        if args.command == 'demo':
            run_demo(args)
        elif args.command == 'extract':
            run_extraction(args)
        elif args.command == 'patterns':
            show_patterns(args)
        else:
            print(f"❌ Unknown command: {args.command}")

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        if '--debug' in sys.argv:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()