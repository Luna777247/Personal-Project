"""
Demo Script for LLM-Based Disaster Information Extraction

This script demonstrates the Large Language Model-based extraction system
for disaster information from Vietnamese news articles.
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any

from scripts.llm_extractor import LLMExtractor, LLMExtractionResult


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
        """
    ]

    return sample_articles


def run_single_extraction_demo():
    """Demonstrate single text extraction with different models"""
    print("=" * 70)
    print("LLM-BASED DISASTER EXTRACTION DEMO")
    print("=" * 70)

    # Initialize extractor
    try:
        extractor = LLMExtractor()
    except ValueError as e:
        print(f"❌ Initialization failed: {e}")
        print("Please set API keys for at least one LLM provider:")
        print("  - OPENAI_API_KEY for GPT models")
        print("  - ANTHROPIC_API_KEY for Claude models")
        print("  - GROQ_API_KEY for Llama models")
        return

    # Load sample articles
    articles = load_sample_articles()

    print(f"\nLoaded {len(articles)} sample articles")
    print(f"Available models: {', '.join(extractor.available_models)}")
    print("\n" + "=" * 70)

    # Test different models if available
    models_to_test = ["gpt-3.5-turbo", "llama3-8b", "claude-3-haiku"]
    available_test_models = [m for m in models_to_test if m in extractor.available_models]

    if not available_test_models:
        available_test_models = [extractor.available_models[0]]  # Use first available

    # Process first article with different models
    test_article = articles[0]
    print("📰 TESTING ARTICLE:")
    print("-" * 50)
    preview = test_article.strip()[:300] + "..." if len(test_article.strip()) > 300 else test_article.strip()
    print(preview)
    print("\n" + "=" * 70)

    for model in available_test_models:
        print(f"\n🤖 MODEL: {model.upper()}")
        print("-" * 30)

        try:
            # Extract information
            start_time = time.time()
            result = extractor.extract_disaster_info(test_article, model=model)
            processing_time = time.time() - start_time

            print(f"⏱️  Processing Time: {processing_time:.2f}s")
            print(f"💰 Cost: ${result.cost_estimate:.4f}")
            print(f"🎯 Confidence: {result.confidence_score:.2f}")
            print()

            # Display extracted information
            info = result.extracted_info
            if "error" not in info:
                print("📋 EXTRACTED INFORMATION:")
                print(f"  • Type: {info.get('type', 'N/A')}")
                print(f"  • Location: {info.get('location', 'N/A')}")
                print(f"  • Time: {info.get('time', 'N/A')}")
                print(f"  • Severity: {info.get('severity', 'N/A')}")
                print(f"  • Damage: {info.get('damage', 'N/A')}")
                print(f"  • Deaths: {info.get('deaths', 'N/A')}")
                print(f"  • Injured: {info.get('injured', 'N/A')}")
                print(f"  • Missing: {info.get('missing', 'N/A')}")
                print(f"  • Forecast: {info.get('forecast', 'N/A')}")

                if info.get('organizations'):
                    print(f"  • Organizations: {', '.join(info['organizations'])}")
            else:
                print(f"❌ Extraction error: {info.get('error', 'Unknown error')}")

        except Exception as e:
            print(f"❌ Model {model} failed: {str(e)}")

        print("-" * 30)


def run_batch_extraction_demo():
    """Demonstrate batch processing"""
    print("\n" + "=" * 70)
    print("BATCH EXTRACTION DEMO")
    print("=" * 70)

    try:
        extractor = LLMExtractor()
    except ValueError as e:
        print(f"❌ Initialization failed: {e}")
        return

    # Load sample articles
    articles = load_sample_articles()

    print(f"\nProcessing {len(articles)} articles in batch mode...")
    print(f"Using model: {extractor.config['default_model']}")

    # Process batch
    start_time = time.time()
    results = extractor.extract_from_texts(articles, batch_size=2)
    total_time = time.time() - start_time

    print(f"⏱️  Total Time: {total_time:.2f}s")
    print(f"📊 Total extractions: {len(results)}")

    # Calculate statistics
    successful = sum(1 for r in results if "error" not in r.extracted_info)
    total_cost = sum(r.cost_estimate for r in results)
    avg_confidence = sum(r.confidence_score for r in results) / len(results) if results else 0

    print(f"✅ Successful: {successful}/{len(results)}")
    print(f"💰 Total Cost: ${total_cost:.2f}")
    print(f"🎯 Avg Confidence: {avg_confidence:.3f}")

    # Show summary for each article
    print("\n📋 EXTRACTION SUMMARY:")
    for i, result in enumerate(results, 1):
        status = "✅" if "error" not in result.extracted_info else "❌"
        info = result.extracted_info
        disaster_type = info.get('type', 'N/A') if "error" not in info else "ERROR"
        location = info.get('location', 'N/A') if "error" not in info else ""
        deaths = info.get('deaths', 'N/A') if "error" not in info else ""

        print(f"  {status} Article {i}: {disaster_type} | {location} | Deaths: {deaths}")

    # Save results
    output_path = Path(__file__).parent.parent / "data" / "batch_llm_extraction_results.json"
    extractor.save_results(results, str(output_path))
    print(f"\n💾 Results saved to: {output_path}")


def run_model_comparison_demo():
    """Compare different models on the same text"""
    print("\n" + "=" * 70)
    print("MODEL COMPARISON DEMO")
    print("=" * 70)

    try:
        extractor = LLMExtractor()
    except ValueError as e:
        print(f"❌ Initialization failed: {e}")
        return

    # Test article
    test_article = """
    Bão Noru đổ bộ vào Phú Yên sáng 28/9, gây mưa lớn trên diện rộng.
    Theo Ban Chỉ huy Phòng chống thiên tai tỉnh Phú Yên, bão đã làm
    2 người chết, 5 người bị thương, tốc mái 50 căn nhà và làm ngập
    200 ha lúa. Tổng thiệt hại ban đầu ước tính 80 tỷ đồng.

    Ông Nguyễn Văn Bé - Phó Chủ tịch UBND tỉnh Phú Yên cho biết:
    "Bão Noru là cơn bão rất mạnh, tốc độ gió giật trên 40m/s.
    Các lực lượng chức năng đã triển khai ứng cứu kịp thời."
    """

    print("📰 TEST ARTICLE:")
    print("-" * 50)
    print(test_article.strip())
    print("\n" + "=" * 70)

    # Test available models
    models_to_compare = [m for m in extractor.available_models if m in
                        ["gpt-3.5-turbo", "gpt-4", "claude-3-haiku", "llama3-8b"]]

    if not models_to_compare:
        models_to_compare = extractor.available_models[:3]  # Use first 3 available

    results = {}

    print("🏁 COMPARING MODELS:")
    print("-" * 50)

    for model in models_to_compare:
        print(f"\n🤖 {model.upper()}:")
        try:
            result = extractor.extract_disaster_info(test_article, model=model)
            results[model] = result

            info = result.extracted_info
            if "error" not in info:
                print(f"  ⏱️  Time: {result.processing_time:.2f}s | 💰 Cost: ${result.cost_estimate:.4f}")
                print(f"  • Type: {info.get('type', 'N/A')}")
                print(f"  • Location: {info.get('location', 'N/A')}")
                print(f"  • Deaths: {info.get('deaths', 'N/A')}")
                print(f"  • Damage: {info.get('damage', 'N/A')}")
            else:
                print(f"  ❌ Error: {info.get('error', 'Unknown')}")

        except Exception as e:
            print(f"  ❌ Failed: {str(e)}")

    # Comparison summary
    print("\n📊 COMPARISON SUMMARY:")
    print("-" * 50)
    print(f"{'Model':<15} {'Cost':<10} {'Confidence':<12} {'Deaths':<8} {'Location':<15}")
    print("-" * 50)

    for model, result in results.items():
        info = result.extracted_info
        if "error" not in info:
            deaths = info.get('deaths', 'N/A')
            location = info.get('location', 'N/A')
            confidence = result.confidence_score
            cost = result.cost_estimate
            time_taken = result.processing_time

            print(f"{model:<15} ${cost:<9.4f} {confidence:<11.2f} {deaths:<7} {location:<15}")


def run_metrics_demo():
    """Show extraction metrics and performance"""
    print("\n" + "=" * 70)
    print("EXTRACTION METRICS DEMO")
    print("=" * 70)

    try:
        extractor = LLMExtractor()
    except ValueError as e:
        print(f"❌ Initialization failed: {e}")
        return

    # Get current metrics
    metrics = extractor.get_metrics()

    print("📊 CURRENT METRICS:")
    print("-" * 50)
    print(f"Total Requests: {metrics['total_requests']}")
    print(f"Success Rate: {metrics['success_rate']:.1%}")
    print(f"💰 Total Cost: ${metrics['total_cost']:.2f}")
    print(f"⏱️  Avg Processing Time: {metrics['avg_processing_time']:.2f}s")
    print(f"💾 Cache Hit Rate: {metrics['cache_hit_rate']:.1f}")
    print(f"🤖 Available Models: {metrics['available_models_count']}")

    print("\n🤖 AVAILABLE MODELS:")
    for model in metrics['available_models']:
        print(f"  • {model}")

    # Test with a few extractions to show metrics change
    print("\n🔄 RUNNING SAMPLE EXTRACTIONS...")
    articles = load_sample_articles()[:2]  # Just 2 for demo

    for i, article in enumerate(articles, 1):
        print(f"  Processing article {i}...")
        extractor.extract_disaster_info(article)

    # Show updated metrics
    updated_metrics = extractor.get_metrics()
    print("\n📈 UPDATED METRICS:")
    print("-" * 50)
    print(f"Total Requests: {updated_metrics['total_requests']}")
    print(f"Success Rate: {updated_metrics['success_rate']:.1%}")
    print(f"💰 Total Cost: ${updated_metrics['total_cost']:.2f}")
    print(f"⏱️  Avg Processing Time: {updated_metrics['avg_processing_time']:.2f}s")
    print(f"💾 Cache Hit Rate: {updated_metrics['cache_hit_rate']:.1f}")
    print(f"🤖 Available Models: {updated_metrics['available_models_count']}")


def run_custom_prompt_demo():
    """Demonstrate custom prompt usage"""
    print("\n" + "=" * 70)
    print("CUSTOM PROMPT DEMO")
    print("=" * 70)

    try:
        extractor = LLMExtractor()
    except ValueError as e:
        print(f"❌ Initialization failed: {e}")
        return

    # Test article
    test_article = """
    Hạn hán kéo dài tại các tỉnh Tây Nguyên khiến hàng nghìn hecta cà phê
    bị khô hạn. Theo Sở Nông nghiệp tỉnh Đắk Lắk, hạn hán năm 2024
    nghiêm trọng hơn mọi năm, ảnh hưởng đến 50.000 hộ dân.
    """

    print("📰 TEST ARTICLE:")
    print("-" * 50)
    print(test_article.strip())
    print("\n" + "=" * 70)

    # Test different prompt types
    prompt_types = ["basic", "detailed", "full"]

    for prompt_type in prompt_types:
        print(f"\n📝 PROMPT TYPE: {prompt_type.upper()}")
        print("-" * 30)

        try:
            result = extractor.extract_disaster_info(
                test_article,
                prompt_type=prompt_type
            )

            print(".2f"            print(f"💰 Cost: ${result.cost_estimate:.4f}")

            info = result.extracted_info
            if "error" not in info:
                # Show key fields
                key_fields = ['type', 'location', 'time', 'damage', 'deaths']
                for field in key_fields:
                    value = info.get(field, 'N/A')
                    print(f"  • {field}: {value}")
            else:
                print(f"❌ Error: {info.get('error', 'Unknown')}")

        except Exception as e:
            print(f"❌ Failed: {str(e)}")

        print("-" * 30)


def main():
    """Main demo function"""
    print("🚀 Starting LLM-Based Extraction Demo")
    print("This demo showcases Large Language Model-based extraction for disaster information")
    print()

    try:
        # Run different demo modes
        run_single_extraction_demo()
        run_batch_extraction_demo()
        run_model_comparison_demo()
        run_metrics_demo()
        run_custom_prompt_demo()

        print("\n" + "=" * 70)
        print("✅ All demos completed successfully!")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during demo: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()