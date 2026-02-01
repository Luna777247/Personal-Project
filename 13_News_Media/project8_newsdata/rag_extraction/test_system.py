#!/usr/bin/env python3
"""
Test script for RAG Disaster Extraction System
Comprehensive testing of all system components
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent / 'scripts'))

from rag_extractor import RAGDisasterExtractor, create_rag_extractor


def test_vector_databases():
    """Test different vector database configurations"""
    print("🧪 Testing vector databases...")

    databases = ['chroma', 'qdrant', 'milvus']
    results = {}

    for db in databases:
        try:
            print(f"  Testing {db}...")
            extractor = create_rag_extractor(vector_db=db)
            results[db] = "✅ Working"
            print(f"    ✅ {db}: Working")
        except Exception as e:
            results[db] = f"❌ Failed: {str(e)}"
            print(f"    ❌ {db}: Failed - {str(e)}")

    return results


def test_embedding_models():
    """Test different embedding models"""
    print("\n🧪 Testing embedding models...")

    models = ['sentence-transformers', 'openai', 'bge']
    results = {}

    for model in models:
        try:
            print(f"  Testing {model}...")
            extractor = create_rag_extractor(embedding=model)
            results[model] = "✅ Working"
            print(f"    ✅ {model}: Working")
        except Exception as e:
            results[model] = f"❌ Failed: {str(e)}"
            print(f"    ❌ {model}: Failed - {str(e)}")

    return results


def test_document_processing():
    """Test document ingestion and processing"""
    print("\n🧪 Testing document processing...")

    try:
        extractor = create_rag_extractor()

        # Sample documents
        documents = [
            {
                "id": "test_1",
                "content": "Bão số 12 gây thiệt hại nặng tại Quảng Nam. 3 người chết, 12 người bị thương.",
                "metadata": {"source": "test", "date": "2023-11-15"}
            },
            {
                "id": "test_2",
                "content": "Lũ lụt miền Trung làm ngập hàng nghìn hecta lúa. Chính phủ hỗ trợ cứu trợ.",
                "metadata": {"source": "test", "date": "2023-10-20"}
            }
        ]

        # Clear and add documents
        extractor.clear_database()
        success = extractor.add_documents(documents)

        if success:
            metrics = extractor.get_metrics()
            print(f"  ✅ Added {metrics['total_documents']} documents")
            print(f"  ✅ Created {metrics['total_chunks']} chunks")
            return True
        else:
            print("  ❌ Failed to add documents")
            return False

    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
        return False


def test_search_functionality():
    """Test document search functionality"""
    print("\n🧪 Testing search functionality...")

    try:
        extractor = create_rag_extractor()

        queries = [
            "bão tại Quảng Nam",
            "lũ lụt miền Trung",
            "thiệt hại thảm họa"
        ]

        for query in queries:
            print(f"  Searching: '{query}'")
            results = extractor.search_documents(query, top_k=3)

            if results:
                print(f"    ✅ Found {len(results)} results")
                # Show top result
                top_result = results[0]
                score = top_result.get('score', 0.0)
                preview = top_result.get('text', '')[:100]
                print(f"    📄 Top result (score: {score:.3f}): {preview}...")
            else:
                print("    ❌ No results found"
        return True

    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
        return False


def test_extraction_functionality():
    """Test disaster information extraction"""
    print("\n🧪 Testing extraction functionality...")

    try:
        extractor = create_rag_extractor()

        queries = [
            "Thiệt hại do bão số 12",
            "Tình hình lũ lụt miền Trung"
        ]

        for query in queries:
            print(f"  Extracting: '{query}'")
            result = extractor.extract_disaster_info(query)

            if result and result.extracted_info:
                info = result.extracted_info
                if "error" not in info:
                    print("    ✅ Extraction successful"                    print(f"    📊 Type: {info.get('type', 'N/A')}")
                    print(f"    📍 Location: {info.get('location', 'N/A')}")
                    print(".2f"                else:
                    print(f"    ❌ Extraction error: {info.get('error', 'Unknown')}")
            else:
                print("    ❌ Extraction failed"
        return True

    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
        return False


def test_batch_processing():
    """Test batch processing capabilities"""
    print("\n🧪 Testing batch processing...")

    try:
        extractor = create_rag_extractor()

        # Add more test documents
        batch_docs = []
        for i in range(10):
            batch_docs.append({
                "id": f"batch_{i+1}",
                "content": f"Thảm họa số {i+1} xảy ra tại khu vực {i+1}. Thiệt hại ước tính {i*100} triệu đồng.",
                "metadata": {"source": "batch_test", "date": f"2023-12-{i+1:02d}"}
            })

        extractor.clear_database()
        success = extractor.add_documents(batch_docs)

        if not success:
            print("  ❌ Failed to add batch documents")
            return False

        # Test batch search
        batch_queries = [f"thảm họa số {i+1}" for i in range(5)]
        start_time = time.time()

        for query in batch_queries:
            results = extractor.search_documents(query, top_k=2)

        end_time = time.time()
        processing_time = end_time - start_time

        print(f"  ✅ Batch search completed in {processing_time:.2f}s")
        print(".2f"
        return True

    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
        return False


def test_performance_metrics():
    """Test performance metrics collection"""
    print("\n🧪 Testing performance metrics...")

    try:
        extractor = create_rag_extractor()

        # Get initial metrics
        initial_metrics = extractor.get_metrics()

        # Perform some operations
        extractor.search_documents("test query")
        extractor.extract_disaster_info("test extraction")

        # Get updated metrics
        updated_metrics = extractor.get_metrics()

        print("  📊 Metrics collected successfully:")
        print(f"    Total documents: {updated_metrics['total_documents']}")
        print(f"    Total chunks: {updated_metrics['total_chunks']}")
        print(f"    Total queries: {updated_metrics['total_queries']}")
        print(f"    Cache hits: {updated_metrics['cache_hits']}")

        return True

    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
        return False


def test_error_handling():
    """Test error handling capabilities"""
    print("\n🧪 Testing error handling...")

    try:
        extractor = create_rag_extractor()

        # Test with invalid query
        result = extractor.extract_disaster_info("")
        if result and "error" in result.extracted_info:
            print("  ✅ Invalid query handled correctly")
        else:
            print("  ❌ Invalid query not handled properly")

        # Test with non-existent document search
        results = extractor.search_documents("nonexistent_query_xyz_12345")
        if not results:
            print("  ✅ Empty search results handled correctly")
        else:
            print("  ❌ Unexpected search results for non-existent query")

        return True

    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
        return False


def run_full_system_test():
    """Run complete system test"""
    print("🚀 RUNNING FULL SYSTEM TEST")
    print("="*50)

    test_results = {}

    # Test components
    test_results['vector_databases'] = test_vector_databases()
    test_results['embedding_models'] = test_embedding_models()
    test_results['document_processing'] = test_document_processing()
    test_results['search_functionality'] = test_search_functionality()
    test_results['extraction_functionality'] = test_extraction_functionality()
    test_results['batch_processing'] = test_batch_processing()
    test_results['performance_metrics'] = test_performance_metrics()
    test_results['error_handling'] = test_error_handling()

    # Summary
    print("\n" + "="*50)
    print("📊 TEST RESULTS SUMMARY")
    print("="*50)

    passed = 0
    total = len(test_results)

    for test_name, result in test_results.items():
        if isinstance(result, dict):
            # Sub-tests (like vector databases)
            sub_passed = sum(1 for r in result.values() if r.startswith("✅"))
            sub_total = len(result)
            status = "✅" if sub_passed == sub_total else "❌"
            print(f"{status} {test_name}: {sub_passed}/{sub_total} passed")
            if sub_passed == sub_total:
                passed += 1
        else:
            # Boolean results
            status = "✅" if result else "❌"
            print(f"{status} {test_name}")
            if result:
                passed += 1

    print(f"\n🎯 Overall: {passed}/{total} test suites passed")

    if passed == total:
        print("🎉 ALL TESTS PASSED! System is ready for production.")
        return True
    else:
        print("⚠️  Some tests failed. Check configuration and dependencies.")
        return False


def main():
    """Main test function"""
    if len(sys.argv) > 1:
        test_name = sys.argv[1]

        # Run specific test
        if test_name == 'vector-db':
            test_vector_databases()
        elif test_name == 'embedding':
            test_embedding_models()
        elif test_name == 'documents':
            test_document_processing()
        elif test_name == 'search':
            test_search_functionality()
        elif test_name == 'extraction':
            test_extraction_functionality()
        elif test_name == 'batch':
            test_batch_processing()
        elif test_name == 'metrics':
            test_performance_metrics()
        elif test_name == 'errors':
            test_error_handling()
        else:
            print(f"❌ Unknown test: {test_name}")
            print("Available tests: vector-db, embedding, documents, search, extraction, batch, metrics, errors")
    else:
        # Run full test suite
        success = run_full_system_test()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()