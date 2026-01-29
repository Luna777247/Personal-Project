"""
Simple Demo Script for LLM-Based Disaster Information Extraction

This script provides a basic demonstration of the LLM extraction system.
"""

import json
import time
from pathlib import Path

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.llm_extractor import LLMExtractor


def main():
    """Main demo function"""
    print("🚀 LLM-Based Disaster Extraction Demo")
    print("=" * 50)

    try:
        # Initialize extractor
        extractor = LLMExtractor()
        print("✅ Extractor initialized successfully")

        # Show available models
        models = extractor.available_models
        print(f"🤖 Available models: {', '.join(models) if models else 'None'}")

        if not models:
            print("\n❌ No API keys found. Please set at least one:")
            print("  - OPENAI_API_KEY")
            print("  - ANTHROPIC_API_KEY")
            print("  - GROQ_API_KEY")
            return

        # Sample text
        sample_text = """
        Bão Noru đổ bộ vào Phú Yên sáng 28/9, gây mưa lớn trên diện rộng.
        Theo Ban Chỉ huy Phòng chống thiên tai tỉnh Phú Yên, bão đã làm
        2 người chết, 5 người bị thương, tốc mái 50 căn nhà và làm ngập
        200 ha lúa. Tổng thiệt hại ban đầu ước tính 80 tỷ đồng.
        """

        print(f"\n📰 Testing with sample text ({len(sample_text)} chars)")

        # Test extraction
        print("\n⏳ Extracting information...")
        start_time = time.time()
        result = extractor.extract_disaster_info(sample_text)
        processing_time = time.time() - start_time

        print(f"⏱️  Processing Time: {processing_time:.2f}s")
        print(f"💰 Cost: ${result.cost_estimate:.4f}")
        print(f"🎯 Confidence: {result.confidence_score:.2f}")

        # Show results
        info = result.extracted_info
        if "error" not in info:
            print("\n📋 EXTRACTED INFORMATION:")
            for key, value in info.items():
                if value and value != "N/A":
                    print(f"  • {key}: {value}")
        else:
            print(f"\n❌ Error: {info.get('error', 'Unknown error')}")

        # Show metrics
        metrics = extractor.get_metrics()
        print("\n📊 METRICS:")
        print(f"  • Total requests: {metrics['total_requests']}")
        print(f"  • Success rate: {metrics['success_rate']:.1%}")
        print(f"  • Total cost: ${metrics['total_cost']:.2f}")

        print("\n✅ Demo completed successfully!")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()