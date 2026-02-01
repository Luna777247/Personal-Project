#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Relation Extraction Configuration
Cấu hình cho các mô hình trích xuất quan hệ
"""

from typing import Dict, Any, List

# Relation types for disaster information
RELATION_TYPES = {
    'OCCURS_AT': 'Thiên tai xảy ra tại địa điểm nào',
    'OCCURS_IN': 'Thiên tai xảy ra trong khu vực nào',
    'OCCURS_ON': 'Thiên tai xảy ra vào thời gian nào',
    'CAUSES_DAMAGE': 'Thiên tai gây thiệt hại gì',
    'AFFECTS_PEOPLE': 'Thiên tai ảnh hưởng đến bao nhiêu người',
    'HAS_INTENSITY': 'Thiên tai có cường độ như thế nào',
    'REPORTED_BY': 'Thiên tai được báo cáo bởi tổ chức nào',
    'RESPONDED_BY': 'Thiên tai được ứng phó bởi tổ chức nào'
}

# Entity pair patterns for relations
RELATION_PATTERNS = {
    'DISASTER_LOCATION': {
        'head_entity': 'DISASTER_TYPE',
        'tail_entity': 'LOCATION',
        'relations': ['OCCURS_AT', 'OCCURS_IN']
    },
    'DISASTER_TIME': {
        'head_entity': 'DISASTER_TYPE',
        'tail_entity': 'TIME',
        'relations': ['OCCURS_ON']
    },
    'DISASTER_DAMAGE': {
        'head_entity': 'DISASTER_TYPE',
        'tail_entity': 'DAMAGE',
        'relations': ['CAUSES_DAMAGE']
    },
    'DISASTER_INTENSITY': {
        'head_entity': 'DISASTER_TYPE',
        'tail_entity': 'QUANTITY',
        'relations': ['HAS_INTENSITY']
    },
    'DISASTER_ORGANIZATION': {
        'head_entity': 'DISASTER_TYPE',
        'tail_entity': 'ORGANIZATION',
        'relations': ['REPORTED_BY', 'RESPONDED_BY']
    }
}

# Model configurations
MODEL_CONFIGS = {
    'phobert_re': {
        'model_name': 'vinai/phobert-base',
        'max_length': 256,
        'batch_size': 16,
        'learning_rate': 2e-5,
        'num_epochs': 10,
        'device': 'auto',  # auto, cpu, cuda
        'save_path': 'models/phobert_re',
        'relation_classes': list(RELATION_TYPES.keys())
    },

    'llm_re': {
        'provider': 'openai',  # openai, anthropic, groq, local
        'model': 'gpt-5.1-codex-max',
        'temperature': 0.1,
        'max_tokens': 500,
        'api_key_env': 'OPENAI_API_KEY',
        'prompt_template': """
        Phân tích bài báo sau và trích xuất các quan hệ giữa các thực thể thiên tai:

        Bài báo: {article_text}

        Các thực thể đã được nhận diện: {entities}

        Trích xuất các quan hệ theo định dạng JSON:
        {{
            "relations": [
                {{
                    "head_entity": "thực thể đầu",
                    "tail_entity": "thực thể cuối",
                    "relation_type": "loại quan hệ",
                    "confidence": 0.95
                }}
            ]
        }}

        Các loại quan hệ có thể: {relation_types}
        """,
        'fallback_provider': 'groq'
    },

    'rule_based_re': {
        'patterns': {
            'OCCURS_AT': [
                r'({disaster}) xảy ra tại ({location})',
                r'({disaster}) tại ({location})',
                r'({location}) hứng chịu ({disaster})'
            ],
            'OCCURS_ON': [
                r'({disaster}) vào lúc ({time})',
                r'({disaster}) vào ngày ({time})',
                r'({time}) xảy ra ({disaster})'
            ],
            'CAUSES_DAMAGE': [
                r'({disaster}) gây thiệt hại ({damage})',
                r'({disaster}) làm ({damage})',
                r'thiệt hại ({damage}) do ({disaster})'
            ]
        },
        'entity_placeholders': {
            'disaster': ['bão', 'lũ', 'động đất', 'sạt lở', 'sóng thần'],
            'location': ['Hà Nội', 'TP.HCM', 'Đà Nẵng', 'Cần Thơ'],
            'time': ['hôm qua', 'hôm nay', 'tuần trước'],
            'damage': ['tỷ đồng', 'người chết', 'người bị thương']
        }
    },

    'spacy_re': {
        'model_name': 'vi_core_news_lg',
        'relation_labels': list(RELATION_TYPES.keys()),
        'confidence_threshold': 0.5
    }
}

# Extraction configuration
EXTRACTION_CONFIG = {
    'max_entity_distance': 50,  # Maximum distance between entities for relation
    'min_confidence': 0.3,      # Minimum confidence for relation extraction
    'batch_size': 8,            # Batch size for processing
    'max_relations_per_pair': 3, # Maximum relations per entity pair
    'enable_caching': True,     # Cache LLM responses
    'cache_dir': 'cache/re_cache'
}

# Vietnamese relation templates for better understanding
RELATION_TEMPLATES = {
    'OCCURS_AT': '{head} xảy ra tại {tail}',
    'OCCURS_IN': '{head} xảy ra trong {tail}',
    'OCCURS_ON': '{head} xảy ra vào {tail}',
    'CAUSES_DAMAGE': '{head} gây thiệt hại {tail}',
    'AFFECTS_PEOPLE': '{head} ảnh hưởng đến {tail} người',
    'HAS_INTENSITY': '{head} có cường độ {tail}',
    'REPORTED_BY': '{head} được báo cáo bởi {tail}',
    'RESPONDED_BY': '{head} được ứng phó bởi {tail}'
}