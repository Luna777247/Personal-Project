#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo Keyword-based Disaster Information Extraction
"""

import os
import sys
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from keyword_extractor import KeywordExtractor, save_results_to_csv, save_results_to_json


def demo():
    """Simple demo function"""
    print("Demo Keyword Extraction")

    # Sample article
    article = {
        'title': 'Test Article',
        'content': 'Bão số 9 gây thiệt hại nặng nề. Gió mạnh cấp 12.',
        'url': 'test',
        'source': 'test'
    }

    extractor = KeywordExtractor()
    result = extractor.process_article(article)

    print("Result:", result['summary'])


if __name__ == "__main__":
    demo()