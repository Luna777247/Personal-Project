"""
Demo Script for Pattern-Based Extraction

This script demonstrates the pattern-based extraction system
for disaster information from Vietnamese news articles.
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any

from scripts.pattern_extractor import PatternBasedExtractor, ExtractionResult


def load_sample_articles() -> List[str]:
    """Load sample disaster news articles for testing"""
    sample_articles = [
        """
        Bão số 12 gây thiệt hại nặng nề tại các tỉnh miền Trung. Theo báo cáo sơ bộ,
        cơn bão đã khiến 15 người thiệt mạng, 27 người bị thương và 5 người mất tích.
        Thiệt hại về vật chất ước tính khoảng 1.200 tỷ đồng, với 150 căn nhà bị sập
        hoàn toàn và hàng trăm hecta lúa bị ngập úng.

        Tại tỉnh Quảng Nam, bão số 12 đổ bộ vào lúc 14h30 ngày 15/11/2023,
        gây mưa lớn liên tục trong 3 ngày. Đội cứu hộ đã triển khai ứng cứu
        khẩn cấp tại các khu vực bị ảnh hưởng nặng nhất.
        """,

        """
        Lũ quét xảy ra tại huyện Mường Khương, tỉnh Lào Cai vào sáng ngày 20/10/2023.
        Theo thông tin từ Ủy ban nhân dân huyện, trận lũ đã gây thiệt hại nghiêm trọng
        với 8 người chết, 12 người mất tích và hàng chục ngôi nhà bị cuốn trôi.

        Thiệt hại kinh tế ban đầu ước tính 50 tỷ đồng. Quân đội và Hội Chữ thập đỏ
        đã huy động lực lượng cứu hộ, tìm kiếm người mất tích tại khu vực xảy ra lũ.
        """,

        """
        Động đất mạnh 6.5 độ Richter xảy ra tại huyện Sìn Hồ, tỉnh Lai Châu
        vào lúc 22h45 ngày 18/6/2024. Theo Trung tâm Báo tin động đất và Cảnh báo
        sóng thần, tâm chấn nằm ở độ sâu 10km với bán kính ảnh hưởng 50km.

        Hiện chưa có thông tin về thiệt hại về người và tài sản. Các cơ quan chức năng
        đang kiểm tra, đánh giá mức độ ảnh hưởng của trận động đất.
        """,

        """
        Mưa lũ kéo dài tại các tỉnh Tây Nguyên gây thiệt hại nặng nề.
        Tại tỉnh Đắk Lắk, mưa lớn trong 5 ngày qua khiến 25 người chết,
        40 người bị thương và thiệt hại kinh tế lên tới 300 tỷ đồng.

        Hàng nghìn hecta cà phê và hồ tiêu bị ngập úng, nhiều tuyến đường
        giao thông bị sạt lở. Chính quyền địa phương đã chỉ đạo các lực lượng
        chức năng triển khai cứu trợ khẩn cấp cho người dân bị ảnh hưởng.
        """
    ]

    return sample_articles


def run_single_extraction_demo():
    """Demonstrate single text extraction"""
    print("=" * 60)
    print("PATTERN-BASED EXTRACTION DEMO")
    print("=" * 60)

    # Initialize extractor
    extractor = PatternBasedExtractor()

    # Load sample articles
    articles = load_sample_articles()

    print(f"\nLoaded {len(articles)} sample articles")
    print("\n" + "=" * 60)

    # Process each article
    for i, article in enumerate(articles, 1):
        print(f"\n📄 ARTICLE {i}")
        print("-" * 40)

        # Show first 200 characters of article
        preview = article.strip()[:200] + "..." if len(article.strip()) > 200 else article.strip()
        print(f"Preview: {preview}")
        print()

        # Extract entities
        start_time = time.time()
        entities = extractor.extract_entities(article)
        processing_time = time.time() - start_time

        print(f"⏱️  Processing time: {processing_time:.3f}s")
        print(f"🔍 Found {len(entities)} entities:")
        print()

        # Group entities by type
        entities_by_type = {}
        for entity in entities:
            if entity.entity_type not in entities_by_type:
                entities_by_type[entity.entity_type] = []
            entities_by_type[entity.entity_type].append(entity)

        # Display entities by type
        for entity_type, type_entities in entities_by_type.items():
            type_info = extractor.config.get("entity_type_mapping", {}).get(entity_type, {})
            display_name = type_info.get("display_name", entity_type)

            print(f"  {display_name} ({entity_type}):")
            for entity in type_entities:
                confidence_pct = int(entity.confidence * 100)
                print(f"    • '{entity.text}' (confidence: {confidence_pct}%)")
                if entity.context:
                    # Show context with match highlighted
                    context_preview = entity.context[:100] + "..." if len(entity.context) > 100 else entity.context
                    print(f"      Context: {context_preview}")
            print()

        print("-" * 40)


def run_batch_extraction_demo():
    """Demonstrate batch processing"""
    print("\n" + "=" * 60)
    print("BATCH EXTRACTION DEMO")
    print("=" * 60)

    # Initialize extractor
    extractor = PatternBasedExtractor()

    # Load sample articles
    articles = load_sample_articles()

    print(f"\nProcessing {len(articles)} articles in batch mode...")

    # Process batch
    start_time = time.time()
    results = extractor.extract_from_texts(articles, batch_size=2)
    total_time = time.time() - start_time

    print(".2f")
    print(f"📊 Total entities extracted: {sum(len(result.entities) for result in results)}")

    # Show summary statistics
    entity_type_counts = {}
    total_confidences = []

    for result in results:
        for entity in result.entities:
            entity_type_counts[entity.entity_type] = entity_type_counts.get(entity.entity_type, 0) + 1
            total_confidences.append(entity.confidence)

    print("\n📈 Entity Type Distribution:")
    for entity_type, count in sorted(entity_type_counts.items()):
        type_info = extractor.config.get("entity_type_mapping", {}).get(entity_type, {})
        display_name = type_info.get("display_name", entity_type)
        print(f"  • {display_name}: {count}")

    if total_confidences:
        avg_confidence = sum(total_confidences) / len(total_confidences)
        print(".1f")
    # Save results
    output_path = Path(__file__).parent.parent / "data" / "batch_extraction_results.json"
    extractor.save_results(results, str(output_path))
    print(f"\n💾 Results saved to: {output_path}")


def run_pattern_analysis_demo():
    """Demonstrate pattern statistics and analysis"""
    print("\n" + "=" * 60)
    print("PATTERN ANALYSIS DEMO")
    print("=" * 60)

    extractor = PatternBasedExtractor()

    # Get pattern statistics
    stats = extractor.get_pattern_stats()

    print(f"\n📊 Pattern Statistics:")
    print(f"  • Total patterns: {stats['total_patterns']}")
    print(f"  • Entity types: {len(stats['entity_types'])}")
    print(f"    {', '.join(stats['entity_types'])}")

    print("\n📂 Pattern Categories:")
    for category, count in stats['pattern_categories'].items():
        print(f"  • {category}: {count} patterns")

    # Show sample patterns
    print("\n🔍 Sample Patterns:")
    from config.patterns import PATTERN_CATEGORIES

    for category_name, patterns in PATTERN_CATEGORIES.items():
        if category_name != "all" and patterns:
            print(f"\n  {category_name.upper()}:")
            for pattern in patterns[:2]:  # Show first 2 patterns per category
                print(f"    • {pattern.name}: {pattern.pattern}")
                if pattern.examples:
                    print(f"      Examples: {', '.join(pattern.examples[:2])}")


def run_custom_pattern_demo():
    """Demonstrate custom pattern creation and testing"""
    print("\n" + "=" * 60)
    print("CUSTOM PATTERN DEMO")
    print("=" * 60)

    # Create custom extractor with additional patterns
    custom_config = {
        "min_confidence": 0.7,
        "max_matches_per_type": 3
    }

    extractor = PatternBasedExtractor(config=custom_config)

    # Test text with various disaster information
    test_text = """
    Theo báo cáo của Bộ Nông nghiệp và Phát triển nông thôn, bão số 8
    đã gây thiệt hại nặng nề tại 8 tỉnh miền Trung với 45 người chết,
    120 người bị thương và thiệt hại kinh tế lên tới 2.500 tỷ đồng.

    Tại tỉnh Quảng Bình, gió bão tốc độ 150km/h khiến 200 căn nhà bị tốc mái,
    50 cây cầu bị sập và hàng nghìn hecta lúa bị ngập. Đội cứu hộ đã
    triển khai ứng cứu tại khu vực bị ảnh hưởng từ ngày 28/9/2023.
    """

    print("Test Text:")
    print(test_text.strip())
    print("\n" + "-" * 60)

    # Extract entities
    entities = extractor.extract_entities(test_text)

    print(f"Found {len(entities)} entities with custom configuration:")

    for entity in entities:
        confidence_pct = int(entity.confidence * 100)
        print(f"  • {entity.entity_type}: '{entity.text}' ({confidence_pct}%)")


def main():
    """Main demo function"""
    print("🚀 Starting Pattern-Based Extraction Demo")
    print("This demo showcases rule-based extraction for disaster information")
    print()

    try:
        # Run different demo modes
        run_single_extraction_demo()
        run_batch_extraction_demo()
        run_pattern_analysis_demo()
        run_custom_pattern_demo()

        print("\n" + "=" * 60)
        print("✅ All demos completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Error during demo: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()