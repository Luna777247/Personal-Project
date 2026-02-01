#!/usr/bin/env python3
"""
LLM-Based Disaster Information Extraction CLI

Command-line interface for extracting disaster information from Vietnamese news
using Large Language Models (OpenAI GPT, Anthropic Claude, Groq Llama).
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from scripts.llm_extractor import LLMExtractor


def load_texts_from_file(file_path: str) -> List[str]:
    """Load texts from various file formats"""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if path.suffix.lower() == '.json':
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and 'articles' in data:
                return data['articles']
            else:
                return [json.dumps(data)]  # Single article as JSON string

    elif path.suffix.lower() == '.csv':
        import csv
        texts = []
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Try common text columns
                text = row.get('text') or row.get('content') or row.get('article') or row.get('news')
                if text:
                    texts.append(text)
        return texts

    elif path.suffix.lower() == '.txt':
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
            # Split by double newlines (paragraphs)
            return [p.strip() for p in content.split('\n\n') if p.strip()]

    else:
        # Try to read as plain text
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
            return [content] if content.strip() else []


def save_results(results: List, output_path: str, format: str = 'json'):
    """Save extraction results to file"""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if format.lower() == 'json':
        # Convert results to dict for JSON serialization
        output_data = {
            'metadata': {
                'total_extractions': len(results),
                'timestamp': str(Path(__file__).parent.parent / 'scripts' / 'llm_extractor.py').replace('\\', '/').split('/')[-1],  # Simple timestamp
                'format_version': '1.0'
            },
            'results': []
        }

        for i, result in enumerate(results):
            result_dict = {
                'id': i + 1,
                'model': result.model_used,
                'processing_time': result.processing_time,
                'cost_estimate': result.cost_estimate,
                'confidence_score': result.confidence_score,
                'extracted_info': result.extracted_info,
                'text_preview': result.text_preview
            }
            output_data['results'].append(result_dict)

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

    elif format.lower() == 'csv':
        import csv
        with open(path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)

            # Write header
            writer.writerow([
                'id', 'model', 'processing_time', 'cost_estimate', 'confidence_score',
                'type', 'location', 'time', 'severity', 'damage', 'deaths', 'injured',
                'missing', 'organizations', 'forecast', 'text_preview'
            ])

            # Write data
            for i, result in enumerate(results):
                info = result.extracted_info
                writer.writerow([
                    i + 1,
                    result.model_used,
                    result.processing_time,
                    result.cost_estimate,
                    result.confidence_score,
                    info.get('type', ''),
                    info.get('location', ''),
                    info.get('time', ''),
                    info.get('severity', ''),
                    info.get('damage', ''),
                    info.get('deaths', ''),
                    info.get('injured', ''),
                    info.get('missing', ''),
                    '; '.join(info.get('organizations', [])),
                    info.get('forecast', ''),
                    result.text_preview
                ])

    print(f"💾 Results saved to: {path}")


def show_available_models(extractor: LLMExtractor):
    """Display available models"""
    print("🤖 Available Models:")
    models = extractor.available_models
    if not models:
        print("  ❌ No models available. Please set API keys.")
        return False

    for model in models:
        config = extractor.get_model_config(model)
        provider = config['provider']
        cost_per_1k = config['cost_per_1k_tokens']
        print(f"  • {model} ({provider}) - ${cost_per_1k:.4f}/1K tokens")
    return True


def extract_single_text(args):
    """Extract from single text input"""
    try:
        extractor = LLMExtractor()

        if not show_available_models(extractor):
            return

        # Get text from args or stdin
        if args.text:
            text = args.text
        elif not sys.stdin.isatty():
            text = sys.stdin.read().strip()
        else:
            print("❌ No text provided. Use --text or pipe text to stdin.")
            return

        print(f"📰 Processing text ({len(text)} characters)...")

        # Extract information
        result = extractor.extract_disaster_info(
            text,
            model=args.model,
            prompt_type=args.prompt_type
        )

        # Display result
        print("
📋 EXTRACTION RESULT:"        print("-" * 50)
        print(f"🤖 Model: {result.model_used}")
        print(".2f"        print(".2f"        print(f"💰 Cost: ${result.cost_estimate:.4f}")
        print()

        info = result.extracted_info
        if "error" not in info:
            print("📊 EXTRACTED INFORMATION:")
            for key, value in info.items():
                if value and value != "N/A":
                    print(f"  • {key}: {value}")
        else:
            print(f"❌ Error: {info.get('error', 'Unknown error')}")

        # Save if requested
        if args.output:
            save_results([result], args.output, args.format)

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)


def extract_from_file(args):
    """Extract from file input"""
    try:
        extractor = LLMExtractor()

        if not show_available_models(extractor):
            return

        # Load texts from file
        print(f"📂 Loading texts from: {args.input_file}")
        texts = load_texts_from_file(args.input_file)
        print(f"📄 Found {len(texts)} text(s) to process")

        if not texts:
            print("❌ No texts found in file")
            return

        # Extract from all texts
        print(f"🚀 Starting batch extraction with model: {args.model}")
        results = extractor.extract_from_texts(
            texts,
            model=args.model,
            prompt_type=args.prompt_type,
            batch_size=args.batch_size
        )

        # Show summary
        successful = sum(1 for r in results if "error" not in r.extracted_info)
        total_cost = sum(r.cost_estimate for r in results)

        print("
📊 BATCH EXTRACTION SUMMARY:"        print("-" * 50)
        print(f"✅ Successful: {successful}/{len(results)}")
        print(".2f"        print(".2f"
        # Show details for first few results
        print("
📋 FIRST FEW RESULTS:"        for i, result in enumerate(results[:3]):
            status = "✅" if "error" not in result.extracted_info else "❌"
            info = result.extracted_info
            disaster_type = info.get('type', 'N/A') if "error" not in info else "ERROR"
            location = info.get('location', 'N/A') if "error" not in info else ""
            print(f"  {status} Text {i+1}: {disaster_type} | {location}")

        if len(results) > 3:
            print(f"  ... and {len(results) - 3} more results")

        # Save results
        if args.output:
            save_results(results, args.output, args.format)

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)


def show_metrics(args):
    """Show extraction metrics"""
    try:
        extractor = LLMExtractor()

        metrics = extractor.get_metrics()

        print("📊 LLM EXTRACTION METRICS:")
        print("=" * 50)
        print(f"🤖 Available Models: {len(metrics['available_models'])}")
        print(f"📈 Total Requests: {metrics['total_requests']}")
        print(f"✅ Success Rate: {metrics['success_rate']:.1%}")
        print(".2f"        print(".2f"        print(".2f"        print(".1f"
        print("
🤖 MODEL DETAILS:"        for model in metrics['available_models']:
            config = extractor.get_model_config(model)
            print(f"  • {model} ({config['provider']})")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="LLM-Based Disaster Information Extraction CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract from text
  python run.py extract --text "Bão số 12 gây thiệt hại tại Quảng Nam..."

  # Extract from file
  python run.py extract --input disaster_news.json --output results.json

  # Extract from CSV file
  python run.py extract --input news.csv --output results.csv --format csv

  # Show available models and metrics
  python run.py models
  python run.py metrics

  # Use specific model and prompt type
  python run.py extract --text "..." --model gpt-4 --prompt-type detailed
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Extract command
    extract_parser = subparsers.add_parser('extract', help='Extract disaster information')
    extract_parser.add_argument('--text', help='Text to extract from')
    extract_parser.add_argument('--input', dest='input_file', help='Input file path (JSON, CSV, TXT)')
    extract_parser.add_argument('--output', help='Output file path')
    extract_parser.add_argument('--format', choices=['json', 'csv'], default='json',
                               help='Output format (default: json)')
    extract_parser.add_argument('--model', help='LLM model to use')
    extract_parser.add_argument('--prompt-type', choices=['basic', 'detailed', 'full'],
                               default='detailed', help='Prompt type (default: detailed)')
    extract_parser.add_argument('--batch-size', type=int, default=5,
                               help='Batch size for multiple texts (default: 5)')

    # Models command
    models_parser = subparsers.add_parser('models', help='Show available models')

    # Metrics command
    metrics_parser = subparsers.add_parser('metrics', help='Show extraction metrics')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Execute commands
    if args.command == 'extract':
        if args.input_file:
            extract_from_file(args)
        else:
            extract_single_text(args)
    elif args.command == 'models':
        try:
            extractor = LLMExtractor()
            show_available_models(extractor)
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            sys.exit(1)
    elif args.command == 'metrics':
        show_metrics(args)


if __name__ == "__main__":
    main()