#!/usr/bin/env python3
"""
RAG-Based Disaster Information Extraction CLI

Command-line interface for extracting disaster information using
Retrieval-Augmented Generation (RAG) with vector databases.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any

from scripts.rag_extractor import RAGDisasterExtractor, create_rag_extractor


def load_documents_from_file(file_path: str) -> List[Dict[str, Any]]:
    """Load documents from various file formats"""
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
                return [data]  # Single article

    elif path.suffix.lower() == '.csv':
        import csv
        documents = []
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                # Try common text columns
                content = row.get('content') or row.get('text') or row.get('article') or row.get('news')
                if content:
                    doc = {
                        "id": row.get('id', f'doc_{i+1}'),
                        "content": content,
                        "metadata": {
                            "source": row.get('source', 'unknown'),
                            "date": row.get('date', ''),
                            "title": row.get('title', ''),
                            "url": row.get('url', '')
                        }
                    }
                    documents.append(doc)
        return documents

    else:
        # Try to read as plain text (single document)
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
            return [{
                "id": path.stem,
                "content": content,
                "metadata": {"source": "file", "filename": path.name}
            }]


def save_results(results: List[Any], output_path: str, format: str = 'json'):
    """Save extraction results to file"""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if format.lower() == 'json':
        # Convert results to dict for JSON serialization
        output_data = {
            'metadata': {
                'total_results': len(results),
                'timestamp': str(Path(__file__).parent.parent / 'scripts' / 'rag_extractor.py').replace('\\', '/').split('/')[-1],
                'format_version': '1.0',
                'system': 'RAG-based extraction'
            },
            'results': []
        }

        for i, result in enumerate(results):
            if hasattr(result, 'extracted_info'):
                # LLMExtractionResult
                result_dict = {
                    'id': i + 1,
                    'query': getattr(result, 'query', ''),
                    'model': getattr(result, 'model_used', ''),
                    'processing_time': getattr(result, 'processing_time', 0),
                    'cost_estimate': getattr(result, 'cost_estimate', 0),
                    'confidence_score': getattr(result, 'confidence_score', 0),
                    'extracted_info': getattr(result, 'extracted_info', {}),
                    'relevant_chunks': getattr(result, 'relevant_chunks', 0)
                }
            else:
                # Search result
                result_dict = {
                    'id': i + 1,
                    'text': result.get('text', ''),
                    'metadata': result.get('metadata', {}),
                    'score': result.get('score', 0.0)
                }

            output_data['results'].append(result_dict)

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

    elif format.lower() == 'csv':
        import csv
        with open(path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)

            # Write header based on result type
            if results and hasattr(results[0], 'extracted_info'):
                # Extraction results
                writer.writerow([
                    'id', 'query', 'model', 'processing_time', 'cost_estimate', 'confidence_score',
                    'type', 'location', 'time', 'severity', 'damage', 'deaths', 'injured',
                    'missing', 'organizations', 'forecast'
                ])

                for i, result in enumerate(results):
                    info = result.extracted_info
                    writer.writerow([
                        i + 1,
                        getattr(result, 'query', ''),
                        getattr(result, 'model_used', ''),
                        getattr(result, 'processing_time', 0),
                        getattr(result, 'cost_estimate', 0),
                        getattr(result, 'confidence_score', 0),
                        info.get('type', ''),
                        info.get('location', ''),
                        info.get('time', ''),
                        info.get('severity', ''),
                        info.get('damage', ''),
                        info.get('deaths', ''),
                        info.get('injured', ''),
                        info.get('missing', ''),
                        '; '.join(info.get('organizations', [])),
                        info.get('forecast', '')
                    ])
            else:
                # Search results
                writer.writerow(['id', 'score', 'text', 'source', 'date'])
                for i, result in enumerate(results):
                    metadata = result.get('metadata', {})
                    writer.writerow([
                        i + 1,
                        result.get('score', 0.0),
                        result.get('text', '')[:500],  # Truncate for CSV
                        metadata.get('source', ''),
                        metadata.get('date', '')
                    ])

    print(f"💾 Results saved to: {path}")


def add_documents_cli(args):
    """Add documents to vector database"""
    try:
        # Create extractor
        extractor = create_rag_extractor(
            vector_db=args.vector_db,
            embedding=args.embedding
        )

        # Load documents
        print(f"📂 Loading documents from: {args.input_file}")
        documents = load_documents_from_file(args.input_file)
        print(f"📄 Found {len(documents)} document(s)")

        if not documents:
            print("❌ No documents found")
            return

        # Clear database if requested
        if args.clear:
            print("🧹 Clearing existing database...")
            extractor.clear_database()

        # Add documents
        print("📥 Adding documents to vector database...")
        success = extractor.add_documents(documents)

        if success:
            print("✅ Documents added successfully")
            print(f"📊 Total documents: {extractor.metrics['total_documents']}")
            print(f"📊 Total chunks: {extractor.metrics['total_chunks']}")
        else:
            print("❌ Failed to add documents")
            sys.exit(1)

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)


def search_documents_cli(args):
    """Search documents in vector database"""
    try:
        # Create extractor
        extractor = create_rag_extractor(
            vector_db=args.vector_db,
            embedding=args.embedding
        )

        print(f"🔍 Searching for: '{args.query}'")

        # Perform search
        results = extractor.search_documents(args.query, top_k=args.top_k)

        if results:
            print(f"✅ Found {len(results)} relevant chunks:")
            print("-" * 80)

            for i, result in enumerate(results, 1):
                score = result.get('score', 0.0)
                text = result.get('text', '')
                metadata = result.get('metadata', {})

                print(f"{i}. Score: {score:.3f}")
                print(f"   Source: {metadata.get('source', 'Unknown')}")
                print(f"   Date: {metadata.get('date', 'Unknown')}")
                print(f"   Preview: {text[:200]}...")
                print("-" * 40)

            # Save results if requested
            if args.output:
                save_results(results, args.output, args.format)
        else:
            print("❌ No relevant documents found")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)


def extract_information_cli(args):
    """Extract disaster information using RAG"""
    try:
        # Create extractor
        extractor = create_rag_extractor(
            vector_db=args.vector_db,
            embedding=args.embedding
        )

        # Get queries
        queries = []
        if args.query:
            queries = [args.query]
        elif args.query_file:
            with open(args.query_file, 'r', encoding='utf-8') as f:
                queries = [line.strip() for line in f if line.strip()]
        else:
            print("❌ No query provided. Use --query or --query-file")
            return

        print(f"🤖 Processing {len(queries)} extraction request(s)")

        all_results = []
        for i, query in enumerate(queries, 1):
            print(f"\n🔄 Query {i}/{len(queries)}: '{query}'")

            # Perform extraction
            result = extractor.extract_disaster_info(query, model=args.model)

            if result:
                print("✅ Extraction successful")
                print(".2f"                print(".4f"                print(".2f"
                # Display key information
                info = result.extracted_info
                if "error" not in info:
                    key_fields = ['type', 'location', 'deaths', 'damage']
                    for field in key_fields:
                        value = info.get(field, 'N/A')
                        if value and value != 'N/A':
                            print(f"  • {field}: {value}")
                else:
                    print(f"  ❌ Error: {info.get('error', 'Unknown')}")
            else:
                print("❌ Extraction failed")
                result = None

            all_results.append(result)

        # Save results if requested
        if args.output:
            # Filter out None results
            valid_results = [r for r in all_results if r is not None]
            if valid_results:
                save_results(valid_results, args.output, args.format)
                print(f"\n💾 Saved {len(valid_results)} results")
            else:
                print("\n⚠️  No valid results to save")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)


def show_metrics_cli(args):
    """Show system metrics"""
    try:
        # Create extractor
        extractor = create_rag_extractor(
            vector_db=args.vector_db,
            embedding=args.embedding
        )

        metrics = extractor.get_metrics()

        print("📊 RAG SYSTEM METRICS:")
        print("=" * 50)
        print(f"Vector Database: {metrics['vector_db_type']}")
        print(f"Embedding Model: {metrics['embedding_model']}")
        print(f"Chunking Strategy: {metrics['chunking_strategy']}")
        print(f"Total Documents: {metrics['total_documents']}")
        print(f"Total Chunks: {metrics['total_chunks']}")
        print(f"Total Queries: {metrics['total_queries']}")
        print(f"Cache Hits: {metrics['cache_hits']}")
        print(".2f"
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)


def clear_database_cli(args):
    """Clear vector database"""
    try:
        # Create extractor
        extractor = create_rag_extractor(
            vector_db=args.vector_db,
            embedding=args.embedding
        )

        print("🧹 Clearing vector database...")
        success = extractor.clear_database()

        if success:
            print("✅ Database cleared successfully")
        else:
            print("❌ Failed to clear database")
            sys.exit(1)

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="RAG-Based Disaster Information Extraction CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Add documents to database
  python run_rag.py add --input disaster_news.json --vector-db chroma

  # Search for relevant documents
  python run_rag.py search --query "bão tại Quảng Nam" --top-k 5

  # Extract disaster information
  python run_rag.py extract --query "thiệt hại bão số 12" --output results.json

  # Extract from multiple queries
  python run_rag.py extract --query-file queries.txt --output batch_results.json

  # Show system metrics
  python run_rag.py metrics

  # Clear database
  python run_rag.py clear

Vector DB Options: chroma, qdrant, milvus
Embedding Options: sentence-transformers, openai, bge
        """
    )

    # Global options
    parser.add_argument('--vector-db', choices=['chroma', 'qdrant', 'milvus'],
                       default='chroma', help='Vector database type (default: chroma)')
    parser.add_argument('--embedding', choices=['sentence-transformers', 'openai', 'bge'],
                       default='sentence-transformers', help='Embedding model (default: sentence-transformers)')

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Add documents command
    add_parser = subparsers.add_parser('add', help='Add documents to vector database')
    add_parser.add_argument('--input', required=True, help='Input file path (JSON, CSV)')
    add_parser.add_argument('--clear', action='store_true', help='Clear database before adding')

    # Search command
    search_parser = subparsers.add_parser('search', help='Search documents')
    search_parser.add_argument('--query', required=True, help='Search query')
    search_parser.add_argument('--top-k', type=int, default=5, help='Number of results (default: 5)')
    search_parser.add_argument('--output', help='Output file path')
    search_parser.add_argument('--format', choices=['json', 'csv'], default='json',
                              help='Output format (default: json)')

    # Extract command
    extract_parser = subparsers.add_parser('extract', help='Extract disaster information')
    extract_parser.add_argument('--query', help='Single extraction query')
    extract_parser.add_argument('--query-file', help='File with multiple queries (one per line)')
    extract_parser.add_argument('--model', help='LLM model to use for extraction')
    extract_parser.add_argument('--output', help='Output file path')
    extract_parser.add_argument('--format', choices=['json', 'csv'], default='json',
                               help='Output format (default: json)')

    # Metrics command
    metrics_parser = subparsers.add_parser('metrics', help='Show system metrics')

    # Clear command
    clear_parser = subparsers.add_parser('clear', help='Clear vector database')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Execute commands
    if args.command == 'add':
        add_documents_cli(args)
    elif args.command == 'search':
        search_documents_cli(args)
    elif args.command == 'extract':
        extract_information_cli(args)
    elif args.command == 'metrics':
        show_metrics_cli(args)
    elif args.command == 'clear':
        clear_database_cli(args)


if __name__ == "__main__":
    main()