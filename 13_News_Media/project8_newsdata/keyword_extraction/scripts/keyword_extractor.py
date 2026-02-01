#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keyword-based Disaster Information Extraction
Trích xuất thông tin thiên tai dựa trên từ khóa

Author: AI Assistant
Date: December 11, 2025
Version: 1.0
"""

import re
import json
import pandas as pd
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict
import os
import sys
from datetime import datetime
import logging

# Hardcoded keywords for demo
DISASTER_KEYWORDS = {
    "storm": [
        "bão", "áp thấp nhiệt đới", "lốc xoáy", "mưa lớn",
        "lũ", "lũ lụt", "ngập úng", "hạn hán"
    ],
    "geological": [
        "động đất", "sóng thần", "núi lửa", "sạt lở đất"
    ],
    "impact": [
        "thiệt hại", "chết", "bị thương", "mất tích"
    ]
}

DISASTER_PHRASES = [
    "cảnh báo bão", "bão mạnh", "động đất mạnh", "thiệt hại nặng"
]

EXTRACTION_CONFIG = {
    "min_sentence_length": 10,
    "max_sentence_length": 500,
    "context_window": 2,
    "case_sensitive": False,
    "remove_duplicates": True,
    "min_keyword_matches": 1,
}

DISASTER_TYPE_MAPPING = {}

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class KeywordExtractor:
    """
    Bộ trích xuất thông tin thiên tai dựa trên từ khóa
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Khởi tạo extractor

        Args:
            config: Cấu hình tùy chỉnh (mặc định dùng từ keywords.py)
        """
        self.config = config or EXTRACTION_CONFIG
        self.keywords = self._prepare_keywords()
        self.phrases = set(DISASTER_PHRASES)

        logger.info(f"Khởi tạo KeywordExtractor với {len(self.keywords)} từ khóa và {len(self.phrases)} cụm từ")

    def _prepare_keywords(self) -> Set[str]:
        """
        Chuẩn bị tập từ khóa từ cấu hình

        Returns:
            Set của tất cả từ khóa
        """
        all_keywords = set()

        for category, keywords in DISASTER_KEYWORDS.items():
            all_keywords.update(keywords)

        return all_keywords

    def split_into_sentences(self, text: str) -> List[str]:
        """
        Chia văn bản thành các câu

        Args:
            text: Văn bản đầu vào

        Returns:
            Danh sách các câu
        """
        # Pattern để split câu (đơn giản)
        sentence_pattern = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_pattern, text.strip())

        # Lọc và làm sạch câu
        clean_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) >= self.config.get('min_sentence_length', 10):
                clean_sentences.append(sentence)

        return clean_sentences

    def find_keyword_matches(self, sentence: str) -> List[Tuple[str, str]]:
        """
        Tìm các từ khóa trong câu

        Args:
            sentence: Câu cần kiểm tra

        Returns:
            List của tuples (keyword, category)
        """
        matches = []
        sentence_lower = sentence.lower() if not self.config.get('case_sensitive', False) else sentence

        # Kiểm tra từ khóa đơn
        for keyword in self.keywords:
            keyword_lower = keyword.lower() if not self.config.get('case_sensitive', False) else keyword
            if keyword_lower in sentence_lower:
                # Xác định category
                category = "unknown"
                for cat, keywords_list in DISASTER_KEYWORDS.items():
                    if keyword in keywords_list:
                        category = cat
                        break
                matches.append((keyword, category))

        # Kiểm tra cụm từ
        for phrase in self.phrases:
            phrase_lower = phrase.lower() if not self.config.get('case_sensitive', False) else phrase
            if phrase_lower in sentence_lower:
                matches.append((phrase, "phrase"))

        return matches

    def extract_sentences_with_keywords(self, text: str) -> List[Dict]:
        """
        Trích xuất các câu chứa từ khóa

        Args:
            text: Văn bản đầu vào

        Returns:
            List của dict chứa thông tin câu và từ khóa
        """
        sentences = self.split_into_sentences(text)
        extracted_data = []

        for i, sentence in enumerate(sentences):
            matches = self.find_keyword_matches(sentence)

            if len(matches) >= self.config.get('min_keyword_matches', 1):
                # Lấy context xung quanh
                context_window = self.config.get('context_window', 2)
                start_idx = max(0, i - context_window)
                end_idx = min(len(sentences), i + context_window + 1)

                context_sentences = sentences[start_idx:end_idx]
                context = ' '.join(context_sentences)

                # Tạo record
                record = {
                    'sentence': sentence,
                    'context': context,
                    'keywords_found': matches,
                    'sentence_index': i,
                    'confidence': len(matches) / len(sentence.split()),  # Đơn giản
                    'disaster_types': self._infer_disaster_types(matches),
                    'timestamp': datetime.now().isoformat()
                }

                extracted_data.append(record)

        return extracted_data

    def _infer_disaster_types(self, matches: List[Tuple[str, str]]) -> List[str]:
        """
        Suy luận loại thiên tai từ matches

        Args:
            matches: List của (keyword, category)

        Returns:
            List loại thiên tai
        """
        disaster_types = set()

        for keyword, category in matches:
            if category in ['storm', 'geological', 'biological', 'environmental']:
                disaster_types.add(category)
            elif category == 'phrase':
                # Có thể suy luận từ phrase
                if any(word in keyword for word in ['bão', 'lũ', 'động đất']):
                    disaster_types.add('storm')
                elif any(word in keyword for word in ['dịch', 'bệnh']):
                    disaster_types.add('biological')

        return list(disaster_types)

    def process_article(self, article_data: Dict) -> Dict:
        """
        Xử lý một bài báo

        Args:
            article_data: Dict chứa thông tin bài báo

        Returns:
            Dict kết quả xử lý
        """
        title = article_data.get('title', '')
        content = article_data.get('content', '')
        url = article_data.get('url', '')
        source = article_data.get('source', '')

        # Kết hợp title và content
        full_text = f"{title}. {content}"

        # Trích xuất
        extracted_sentences = self.extract_sentences_with_keywords(full_text)

        # Tổng hợp kết quả
        result = {
            'article_info': {
                'title': title,
                'url': url,
                'source': source,
                'content_length': len(content)
            },
            'extraction_results': extracted_sentences,
            'summary': {
                'total_sentences_extracted': len(extracted_sentences),
                'unique_keywords': len(set(kw for sent in extracted_sentences
                                         for kw, _ in sent['keywords_found'])),
                'disaster_types_detected': list(set(dt for sent in extracted_sentences
                                                  for dt in sent['disaster_types'])),
                'avg_confidence': sum(sent['confidence'] for sent in extracted_sentences) / len(extracted_sentences) if extracted_sentences else 0
            },
            'processing_timestamp': datetime.now().isoformat()
        }

        return result

    def process_batch(self, articles: List[Dict]) -> List[Dict]:
        """
        Xử lý batch các bài báo

        Args:
            articles: List của article dicts

        Returns:
            List kết quả xử lý
        """
        results = []

        for i, article in enumerate(articles):
            logger.info(f"Đang xử lý bài báo {i+1}/{len(articles)}: {article.get('title', 'N/A')[:50]}...")
            try:
                result = self.process_article(article)
                results.append(result)
            except Exception as e:
                logger.error(f"Lỗi xử lý bài báo {i+1}: {str(e)}")
                results.append({
                    'error': str(e),
                    'article_info': article,
                    'processing_timestamp': datetime.now().isoformat()
                })

        return results


def save_results_to_csv(results: List[Dict], output_path: str):
    """
    Lưu kết quả ra CSV

    Args:
        results: List kết quả từ process_batch
        output_path: Đường dẫn file output
    """
    # Flatten results cho CSV
    flattened_data = []

    for result in results:
        if 'error' in result:
            continue

        article_info = result['article_info']
        summary = result['summary']

        for sentence_data in result['extraction_results']:
            row = {
                'title': article_info['title'],
                'url': article_info['url'],
                'source': article_info['source'],
                'sentence': sentence_data['sentence'],
                'context': sentence_data['context'],
                'keywords_found': '; '.join([f"{kw} ({cat})" for kw, cat in sentence_data['keywords_found']]),
                'disaster_types': '; '.join(sentence_data['disaster_types']),
                'confidence': sentence_data['confidence'],
                'total_sentences_extracted': summary['total_sentences_extracted'],
                'unique_keywords': summary['unique_keywords'],
                'processing_timestamp': result['processing_timestamp']
            }
            flattened_data.append(row)

    if flattened_data:
        df = pd.DataFrame(flattened_data)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        logger.info(f"Đã lưu {len(flattened_data)} records vào {output_path}")
    else:
        logger.warning("Không có dữ liệu để lưu")


def save_results_to_json(results: List[Dict], output_path: str):
    """
    Lưu kết quả ra JSON

    Args:
        results: List kết quả từ process_batch
        output_path: Đường dẫn file output
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"Đã lưu kết quả vào {output_path}")


# Demo function
def demo_keyword_extraction():
    """
    Demo function để test keyword extraction
    """
    # Sample article
    sample_article = {
        'title': 'Bão số 9 gây thiệt hại nặng tại các tỉnh miền Trung',
        'content': '''
        Bão số 9 đã đổ bộ vào các tỉnh miền Trung gây thiệt hại nặng nề.
        Gió mạnh cấp 12, sóng biển cao 5-7m. Hàng trăm ngôi nhà bị tốc mái,
        nhiều diện tích lúa bị ngập úng. Có 3 người chết, 10 người bị thương.
        Chính phủ đã chỉ đạo ứng phó khẩn cấp với bão này.
        ''',
        'url': 'https://example.com/bao-so-9',
        'source': 'vnexpress'
    }

    # Khởi tạo extractor
    extractor = KeywordExtractor()

    # Process
    result = extractor.process_article(sample_article)

    # In kết quả
    print("=== KẾT QUẢ TRÍCH XUẤT ===")
    print(f"Tiêu đề: {result['article_info']['title']}")
    print(f"Số câu trích xuất: {result['summary']['total_sentences_extracted']}")
    print(f"Loại thiên tai: {result['summary']['disaster_types_detected']}")

    for i, sentence_data in enumerate(result['extraction_results'], 1):
        print(f"\nCâu {i}: {sentence_data['sentence']}")
        print(f"Từ khóa: {[kw for kw, _ in sentence_data['keywords_found']]}")
        print(f"Loại: {sentence_data['disaster_types']}")
        print(".2f")


if __name__ == "__main__":
    # Chạy demo
    demo_keyword_extraction()