#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo Keyword-based Disaster Information Extraction
Demo đầy đủ cho hệ thống trích xuất thông tin thiên tai dựa trên từ khóa
Uses real data from disaster_data_multisource_20251207_165113.json
"""

import os
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(__file__))

from keyword_extractor import KeywordExtractor, save_results_to_csv, save_results_to_json


def load_sample_articles():
    """Load real disaster data from JSON file"""
    data_file = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'disaster_data_multisource_20251207_165113.json')
    if os.path.exists(data_file):
        import json
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        articles = []
        for item in data:  # Load all articles
            article = {
                'title': item.get('title', ''),
                'content': item.get('content', ''),
                'url': item.get('url', ''),
                'source': item.get('source', '')
            }
            articles.append(article)
        return articles
    else:
        print("⚠️ Could not load real data file")
        print("Using fallback sample data...")
        return load_sample_articles_fallback()

def load_sample_articles_fallback():
    """Tạo dữ liệu mẫu để demo (fallback)"""
    sample_articles = [
        {
            'title': 'Bão số 9 gây thiệt hại nặng tại các tỉnh miền Trung',
            'content': '''Bão số 9 đã đổ bộ vào các tỉnh miền Trung vào sáng nay. Gió mạnh cấp 12-13, sóng biển cao 5-7m. Hàng trăm ngôi nhà bị tốc mái, nhiều diện tích lúa bị ngập úng. Có 3 người chết, 10 người bị thương. Thiệt hại ban đầu ước tính hàng trăm tỷ đồng.''',
            'url': 'https://vnexpress.net/bao-so-9',
            'source': 'vnexpress'
        },
        {
            'title': 'Động đất mạnh 6.5 độ Richter tại Kon Tum',
            'content': '''Sáng nay xảy ra động đất mạnh 6.5 độ Richter tại huyện Kon Plông, tỉnh Kon Tum. Trung tâm báo tin động đất ghi nhận trận động đất xảy ra vào lúc 7 giờ 45 phút. Rung chấn kéo dài khoảng 30 giây.''',
            'url': 'https://dantri.com.vn/dong-dat-kon-tum',
            'source': 'dantri'
        }
    ]
    return sample_articles


def run_demo():
    """Chạy demo keyword extraction"""
    print("🚀 Demo Keyword-based Disaster Information Extraction")
    print("=" * 60)

    # Khởi tạo extractor
    print("📋 Khởi tạo KeywordExtractor...")
    extractor = KeywordExtractor()

    # Load sample data
    print("📚 Load dữ liệu mẫu...")
    sample_articles = load_sample_articles()
    print(f"   Tìm thấy {len(sample_articles)} bài báo mẫu")

    # Process batch
    print("\n🔍 Đang xử lý batch...")
    start_time = datetime.now()
    results = extractor.process_batch(sample_articles)
    end_time = datetime.now()

    processing_time = (end_time - start_time).total_seconds()

    # Hiển thị kết quả tổng quan
    print("\n📊 KẾT QUẢ TỔNG QUAN:")
    print(f"   Thời gian xử lý: {processing_time:.2f} giây")
    print(f"   Số bài báo xử lý: {len(results)}")

    total_sentences = sum(r.get('summary', {}).get('total_sentences_extracted', 0)
                         for r in results if 'error' not in r)
    print(f"   Tổng câu trích xuất: {total_sentences}")

    # Chi tiết từng bài
    print("\n📝 CHI TIẾT TỪNG BÀI BÁO:")
    for i, result in enumerate(results, 1):
        if 'error' in result:
            print(f"   {i}. ❌ Lỗi: {result['error']}")
            continue

        article = result['article_info']
        summary = result['summary']

        print(f"   {i}. ✅ {article['title'][:50]}...")
        print(f"      Nguồn: {article['source']}")
        print(f"      Câu trích xuất: {summary['total_sentences_extracted']}")
        print(f"      Từ khóa unique: {summary['unique_keywords']}")
        print(f"      Loại thiên tai: {summary['disaster_types_detected']}")
        print(f"      Độ tin cậy TB: {summary['avg_confidence']:.2f}")

        # Hiển thị sample sentences
        if result['extraction_results']:
            print("      📄 Sample câu trích xuất:")
            for sent in result['extraction_results'][:2]:  # Show max 2 sentences
                keywords = [kw for kw, _ in sent['keywords_found']]
                print(f"         • {sent['sentence'][:80]}...")
                print(f"           Từ khóa: {keywords}")

    # Lưu kết quả
    print("\n💾 Đang lưu kết quả...")
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, 'keyword_extraction_demo.csv')
    json_path = os.path.join(output_dir, 'keyword_extraction_demo.json')

    save_results_to_csv(results, csv_path)
    save_results_to_json(results, json_path)

    print(f"   ✅ Đã lưu CSV: {csv_path}")
    print(f"   ✅ Đã lưu JSON: {json_path}")

    print("\n🎯 DEMO HOÀN THÀNH!")
    print("   Bạn có thể xem kết quả chi tiết trong thư mục data/")
    print("   File CSV có thể mở bằng Excel để xem dễ dàng hơn.")


if __name__ == "__main__":
    run_demo()