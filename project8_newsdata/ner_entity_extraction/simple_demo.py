#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple NER Demo - Test basic functionality without heavy dependencies
"""

import sys
import os
import json
from typing import Dict, List, Any
from dataclasses import dataclass

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

@dataclass
class Entity:
    """Entity dataclass for NER results"""
    text: str
    label: str
    start: int
    end: int
    confidence: float = 1.0

class SimpleNERExtractor:
    """Simple rule-based NER extractor for testing"""

    def __init__(self):
        self.entity_patterns = {
            'LOCATION': ['Hà Nội', 'TP.HCM', 'Đà Nẵng', 'Cần Thơ', 'Hải Phòng'],
            'DISASTER_TYPE': ['bão', 'lũ', 'động đất', 'sạt lở', 'sóng thần'],
            'TIME': ['hôm qua', 'hôm nay', 'tuần trước', 'tháng này']
        }

    def extract_entities(self, text: str) -> List[Entity]:
        """Extract entities using simple string matching"""
        entities = []
        for label, patterns in self.entity_patterns.items():
            for pattern in patterns:
                if pattern.lower() in text.lower():
                    start = text.lower().find(pattern.lower())
                    end = start + len(pattern)
                    entities.append(Entity(
                        text=pattern,
                        label=label,
                        start=start,
                        end=end,
                        confidence=0.8
                    ))
        return entities

def test_basic_functionality():
    """Test basic NER functionality"""
    print("🧪 Testing Basic NER Functionality")
    print("=" * 50)

    # Sample disaster article
    sample_article = {
        'title': 'Bão số 12 gây thiệt hại nặng nề tại Hà Nội',
        'content': 'Bão số 12 đã đổ bộ vào khu vực Hà Nội hôm qua, gây ra lũ lụt nghiêm trọng. Hàng trăm ngôi nhà bị sạt lở.',
        'url': 'https://example.com/article1',
        'source': 'VNExpress'
    }

    # Test simple extractor
    extractor = SimpleNERExtractor()
    entities = extractor.extract_entities(sample_article['content'])

    print(f"📄 Article: {sample_article['title']}")
    print(f"📝 Content: {sample_article['content']}")
    print(f"🎯 Extracted Entities: {len(entities)}")

    for entity in entities:
        print(f"  - {entity.label}: '{entity.text}' (confidence: {entity.confidence})")

    # Test JSON output
    result = {
        'article': sample_article,
        'entities': [
            {
                'text': e.text,
                'label': e.label,
                'start': e.start,
                'end': e.end,
                'confidence': e.confidence
            } for e in entities
        ],
        'metadata': {
            'extractor': 'SimpleNERExtractor',
            'timestamp': '2024-12-11T13:40:00Z',
            'entity_count': len(entities)
        }
    }

    # Save to file
    output_file = 'data/ner_demo_output.json'
    os.makedirs('data', exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"💾 Results saved to: {output_file}")
    print("✅ Basic functionality test completed successfully!")

def test_imports():
    """Test if all modules can be imported"""
    print("🔍 Testing Module Imports")
    print("=" * 50)

    try:
        from scripts.ner_extractor import NERExtractor, Entity
        print("✅ Base NER extractor imported successfully")
    except ImportError as e:
        print(f"❌ Base NER extractor import failed: {e}")
        return False

    # Test individual extractors (may fail due to missing dependencies)
    extractors = [
        ('scripts.phoner_extractor', 'PhoNERExtractor'),
        ('scripts.vncorenlp_extractor', 'VnCoreNLPExtractor'),
        ('scripts.spacy_custom_extractor', 'SpacyCustomExtractor'),
        ('scripts.bert_ner_extractor', 'BERTNERExtractor')
    ]

    for module_name, class_name in extractors:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            print(f"✅ {class_name} imported successfully")
        except ImportError as e:
            print(f"⚠️  {class_name} import failed (expected due to dependencies): {e}")
        except Exception as e:
            print(f"❌ {class_name} import error: {e}")

    return True

if __name__ == "__main__":
    print("🚀 NER Entity Extraction - Simple Demo")
    print("=" * 50)

    # Test imports first
    if not test_imports():
        print("❌ Import tests failed")
        sys.exit(1)

    print()

    # Test basic functionality
    test_basic_functionality()

    print()
    print("🎉 Demo completed!")
    print("📋 Next steps:")
    print("   1. Install full dependencies: pip install -r requirements.txt")
    print("   2. Run full demo: python run.py --demo")
    print("   3. Test individual models: python run.py --model <model_name>")