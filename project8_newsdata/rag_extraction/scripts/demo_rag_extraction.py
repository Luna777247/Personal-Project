"""
Demo Script for RAG-Based Disaster Information Extraction

This script demonstrates the Retrieval-Augmented Generation (RAG) system
for disaster information extraction using vector databases.
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any

from scripts.rag_extractor import RAGDisasterExtractor, create_rag_extractor


def load_sample_documents() -> List[Dict[str, Any]]:
    """Load sample disaster news documents for testing"""
    sample_docs = [
        {
            "id": "doc_001",
            "content": """
            Bão số 12 gây thiệt hại nặng nề tại các tỉnh miền Trung. Theo báo cáo sơ bộ từ Ban Chỉ huy
            Phòng chống thiên tai Trung ương, cơn bão đã khiến 15 người thiệt mạng, 27 người bị thương
            và 5 người mất tích. Thiệt hại về vật chất ước tính khoảng 1.200 tỷ đồng, với 150 căn nhà
            bị sập hoàn toàn và hàng trăm hecta lúa bị ngập úng.

            Tại tỉnh Quảng Nam, bão số 12 đổ bộ vào lúc 14h30 ngày 15/11/2023, gây mưa lớn liên tục
            trong 3 ngày. Đội cứu hộ đã triển khai ứng cứu khẩn cấp tại các khu vực bị ảnh hưởng
            nặng nhất. Quân đội và Hội Chữ thập đỏ đã huy động hàng trăm cán bộ, chiến sĩ tham gia
            cứu hộ, tìm kiếm người mất tích.
            """,
            "metadata": {
                "source": "Vietnam News",
                "date": "2023-11-16",
                "location": "Quảng Nam",
                "disaster_type": "Bão"
            }
        },
        {
            "id": "doc_002",
            "content": """
            Lũ quét xảy ra tại huyện Mường Khương, tỉnh Lào Cai vào sáng ngày 20/10/2023.
            Theo thông tin từ Ủy ban nhân dân huyện, trận lũ đã gây thiệt hại nghiêm trọng với
            8 người chết, 12 người mất tích và hàng chục ngôi nhà bị cuốn trôi. Thiệt hại kinh tế
            ban đầu ước tính 50 tỷ đồng.

            Nguyên nhân ban đầu được xác định là do mưa lớn kéo dài nhiều ngày, khiến đất đá từ
            các quả đồi cao bị sạt lở, tạo thành dòng lũ quét với tốc độ rất nhanh. Quân đội và
            Hội Chữ thập đỏ đã huy động lực lượng cứu hộ, tìm kiếm người mất tích tại khu vực
            xảy ra lũ. Công tác cứu hộ đang gặp nhiều khó khăn do địa hình hiểm trở.
            """,
            "metadata": {
                "source": "Lao Cai News",
                "date": "2023-10-21",
                "location": "Lào Cai",
                "disaster_type": "Lũ quét"
            }
        },
        {
            "id": "doc_003",
            "content": """
            Động đất mạnh 6.5 độ Richter xảy ra tại huyện Sìn Hồ, tỉnh Lai Châu vào lúc 22h45
            ngày 18/6/2024. Theo Trung tâm Báo tin động đất và Cảnh báo sóng thần, tâm chấn nằm
            ở độ sâu 10km với bán kính ảnh hưởng 50km. Hiện chưa có thông tin về thiệt hại về
            người và tài sản.

            Các cơ quan chức năng đang kiểm tra, đánh giá mức độ ảnh hưởng của trận động đất.
            Người dân tại khu vực tâm chấn cảm nhận được rung động mạnh, một số đồ đạc trong
            nhà bị rơi vỡ. Không có cảnh báo sóng thần vì tâm chấn nằm sâu dưới đất liền.
            """,
            "metadata": {
                "source": "Lai Chau News",
                "date": "2024-06-19",
                "location": "Lai Châu",
                "disaster_type": "Động đất"
            }
        },
        {
            "id": "doc_004",
            "content": """
            Hạn hán kéo dài tại các tỉnh Tây Nguyên khiến hàng nghìn hecta cà phê bị khô hạn.
            Theo Sở Nông nghiệp tỉnh Đắk Lắk, hạn hán năm 2024 nghiêm trọng hơn mọi năm,
            ảnh hưởng đến 50.000 hộ dân trồng cà phê. Nắng nóng kéo dài từ đầu năm khiến
            các hồ chứa nước cạn kiệt, sông suối khô hạn.

            UBND tỉnh Đắk Lắk đã chỉ đạo các địa phương triển khai các biện pháp ứng phó
            với hạn hán như khoan giếng, đào ao trữ nước, chuyển đổi cơ cấu cây trồng.
            Dự báo hạn hán sẽ còn kéo dài đến mùa mưa năm 2024.
            """,
            "metadata": {
                "source": "Dak Lak News",
                "date": "2024-04-15",
                "location": "Đắk Lắk",
                "disaster_type": "Hạn hán"
            }
        }
    ]

    return sample_docs


def demo_document_ingestion():
    """Demonstrate document ingestion into vector database"""
    print("=" * 70)
    print("RAG DOCUMENT INGESTION DEMO")
    print("=" * 70)

    try:
        # Create RAG extractor
        extractor = create_rag_extractor(vector_db="chroma", embedding="sentence-transformers")
        print("✅ RAG extractor initialized")

        # Load sample documents
        documents = load_sample_documents()
        print(f"📄 Loaded {len(documents)} sample documents")

        # Clear existing data
        print("🧹 Clearing existing database...")
        extractor.clear_database()

        # Add documents
        print("📥 Adding documents to vector database...")
        start_time = time.time()
        success = extractor.add_documents(documents)
        ingestion_time = time.time() - start_time

        if success:
            print(".2f"            print(f"📊 Total chunks created: {extractor.metrics['total_chunks']}")
        else:
            print("❌ Document ingestion failed")
            return False

        return True

    except Exception as e:
        print(f"❌ Ingestion demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def demo_similarity_search():
    """Demonstrate similarity search in vector database"""
    print("\n" + "=" * 70)
    print("VECTOR SIMILARITY SEARCH DEMO")
    print("=" * 70)

    try:
        extractor = create_rag_extractor()

        # Test queries
        test_queries = [
            "thông tin bão tại Quảng Nam",
            "lũ quét ở Lào Cai",
            "động đất Lai Châu",
            "hạn hán Tây Nguyên",
            "thiên tai miền núi"
        ]

        for query in test_queries:
            print(f"\n🔍 Query: '{query}'")
            print("-" * 40)

            # Search for relevant chunks
            results = extractor.search_documents(query, top_k=3)

            if results:
                for i, result in enumerate(results, 1):
                    score = result.get('score', 0.0)
                    preview = result.get('text', '')[:150] + '...'
                    metadata = result.get('metadata', {})

                    print(f"  {i}. Score: {score:.3f}")
                    print(f"     Preview: {preview}")
                    print(f"     Source: {metadata.get('source', 'Unknown')}")
                    print(f"     Date: {metadata.get('date', 'Unknown')}")
            else:
                print("  ❌ No results found")

        return True

    except Exception as e:
        print(f"❌ Search demo failed: {e}")
        return False


def demo_rag_extraction():
    """Demonstrate full RAG extraction pipeline"""
    print("\n" + "=" * 70)
    print("RAG-BASED EXTRACTION DEMO")
    print("=" * 70)

    try:
        extractor = create_rag_extractor()

        # Test extraction queries
        extraction_queries = [
            "Thiệt hại do bão số 12 tại Quảng Nam",
            "Thông tin lũ quét Mường Khương Lào Cai",
            "Động đất tại Sìn Hồ Lai Châu",
            "Tình hình hạn hán ở Đắk Lắk",
            "Thiên tai gây thiệt hại lớn nhất năm 2023"
        ]

        for query in extraction_queries:
            print(f"\n🤖 Extracting: '{query}'")
            print("-" * 50)

            # Perform RAG extraction
            start_time = time.time()
            result = extractor.extract_disaster_info(query)
            extraction_time = time.time() - start_time

            if result:
                print(".2f"                print(f"💰 Cost: ${result.cost_estimate:.4f}")
                print(f"🎯 Confidence: {result.confidence_score:.2f}")
                print()

                # Display extracted information
                info = result.extracted_info
                if "error" not in info:
                    print("📋 EXTRACTED INFORMATION:")
                    for key, value in info.items():
                        if value and value != "N/A":
                            print(f"  • {key}: {value}")
                else:
                    print(f"❌ Error: {info.get('error', 'Unknown error')}")
            else:
                print("❌ Extraction failed or no relevant information found")

            print("-" * 50)

        return True

    except Exception as e:
        print(f"❌ RAG extraction demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def demo_metrics_and_performance():
    """Demonstrate system metrics and performance"""
    print("\n" + "=" * 70)
    print("SYSTEM METRICS & PERFORMANCE DEMO")
    print("=" * 70)

    try:
        extractor = create_rag_extractor()

        # Get system metrics
        metrics = extractor.get_metrics()

        print("📊 SYSTEM METRICS:")
        print("-" * 40)
        print(f"Vector DB Type: {metrics['vector_db_type']}")
        print(f"Embedding Model: {metrics['embedding_model']}")
        print(f"Chunking Strategy: {metrics['chunking_strategy']}")
        print(f"Total Documents: {metrics['total_documents']}")
        print(f"Total Chunks: {metrics['total_chunks']}")
        print(f"Total Queries: {metrics['total_queries']}")
        print(f"Cache Hits: {metrics['cache_hits']}")
        print(".2f"
        # Test performance with multiple queries
        print("
⏱️  PERFORMANCE TEST:"        print("-" * 40)

        test_queries = [
            "bão lũ miền Trung",
            "thiên tai Tây Nguyên",
            "động đất miền núi"
        ]

        total_time = 0
        for query in test_queries:
            start_time = time.time()
            results = extractor.search_documents(query, top_k=5)
            query_time = time.time() - start_time
            total_time += query_time
            print(".3f"
        avg_time = total_time / len(test_queries)
        print(".3f"
        return True

    except Exception as e:
        print(f"❌ Metrics demo failed: {e}")
        return False


def demo_batch_processing():
    """Demonstrate batch document processing"""
    print("\n" + "=" * 70)
    print("BATCH PROCESSING DEMO")
    print("=" * 70)

    try:
        extractor = create_rag_extractor()

        # Create larger batch of documents
        base_docs = load_sample_documents()
        batch_docs = []

        # Create variations of documents
        for i, doc in enumerate(base_docs):
            for j in range(3):  # Create 3 variations each
                new_doc = doc.copy()
                new_doc["id"] = f"{doc['id']}_var_{j}"
                new_doc["content"] = doc["content"] + f"\n\n[Phiên bản {j+1}]"
                new_doc["metadata"] = {**doc["metadata"], "variation": j+1}
                batch_docs.append(new_doc)

        print(f"📄 Created batch of {len(batch_docs)} documents")

        # Clear database
        extractor.clear_database()

        # Batch add documents
        print("📥 Adding documents in batch...")
        start_time = time.time()
        success = extractor.add_documents(batch_docs)
        batch_time = time.time() - start_time

        if success:
            print(".2f"            print(f"📊 Documents processed: {len(batch_docs)}")
            print(f"📊 Chunks created: {extractor.metrics['total_chunks']}")
            print(".1f"
        else:
            print("❌ Batch processing failed")

        return success

    except Exception as e:
        print(f"❌ Batch demo failed: {e}")
        return False


def run_full_demo():
    """Run complete RAG system demonstration"""
    print("🚀 STARTING COMPLETE RAG SYSTEM DEMO")
    print("This demo showcases the full RAG pipeline for disaster information extraction")
    print()

    demo_results = []

    # Run all demos
    demos = [
        ("Document Ingestion", demo_document_ingestion),
        ("Similarity Search", demo_similarity_search),
        ("RAG Extraction", demo_rag_extraction),
        ("Batch Processing", demo_batch_processing),
        ("Metrics & Performance", demo_metrics_and_performance)
    ]

    for demo_name, demo_func in demos:
        print(f"\n{'='*20} {demo_name.upper()} {'='*20}")
        try:
            result = demo_func()
            demo_results.append((demo_name, result))
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"\n{status}: {demo_name}")
        except Exception as e:
            print(f"\n❌ FAILED: {demo_name} - {e}")
            demo_results.append((demo_name, False))

    # Summary
    print("\n" + "=" * 70)
    print("DEMO SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, result in demo_results if result)
    total = len(demo_results)

    print(f"Total Demos: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")

    for demo_name, result in demo_results:
        status = "✅" if result else "❌"
        print(f"  {status} {demo_name}")

    if passed == total:
        print("\n🎉 ALL DEMOS PASSED! RAG system is working correctly.")
    else:
        print(f"\n⚠️  {total - passed} demo(s) failed. Check the errors above.")

    print("=" * 70)


def main():
    """Main demo function"""
    try:
        run_full_demo()
    except KeyboardInterrupt:
        print("\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()