#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Relation Definitions for Disaster Information
Định nghĩa các quan hệ và patterns cho thông tin thiên tai
"""

from typing import Dict, List, Any, Tuple

# Disaster relation definitions
RELATION_DEFINITIONS = {
    'OCCURS_AT': {
        'description': 'Thiên tai xảy ra tại địa điểm cụ thể',
        'head_entity_types': ['DISASTER_TYPE'],
        'tail_entity_types': ['LOCATION'],
        'examples': [
            'Bão số 12 xảy ra tại Hà Nội',
            'Động đất xảy ra tại Kon Tum',
            'Lũ quét xảy ra tại Lào Cai'
        ],
        'vietnamese_template': '{head} xảy ra tại {tail}',
        'confidence_threshold': 0.8
    },

    'OCCURS_IN': {
        'description': 'Thiên tai xảy ra trong khu vực hoặc tỉnh',
        'head_entity_types': ['DISASTER_TYPE'],
        'tail_entity_types': ['LOCATION'],
        'examples': [
            'Bão số 12 xảy ra trong khu vực miền Bắc',
            'Lũ lụt xảy ra trong tỉnh Quảng Ninh'
        ],
        'vietnamese_template': '{head} xảy ra trong {tail}',
        'confidence_threshold': 0.7
    },

    'OCCURS_ON': {
        'description': 'Thiên tai xảy ra vào thời gian cụ thể',
        'head_entity_types': ['DISASTER_TYPE'],
        'tail_entity_types': ['TIME'],
        'examples': [
            'Bão số 12 xảy ra vào ngày 15/10',
            'Động đất xảy ra vào lúc 9h sáng',
            'Lũ quét xảy ra vào tuần trước'
        ],
        'vietnamese_template': '{head} xảy ra vào {tail}',
        'confidence_threshold': 0.9
    },

    'CAUSES_DAMAGE': {
        'description': 'Thiên tai gây ra thiệt hại',
        'head_entity_types': ['DISASTER_TYPE'],
        'tail_entity_types': ['DAMAGE'],
        'examples': [
            'Bão số 12 gây thiệt hại 20 tỷ đồng',
            'Động đất gây thiệt hại nghiêm trọng',
            'Lũ lụt gây thiệt hại về người và tài sản'
        ],
        'vietnamese_template': '{head} gây thiệt hại {tail}',
        'confidence_threshold': 0.8
    },

    'AFFECTS_PEOPLE': {
        'description': 'Thiên tai ảnh hưởng đến số lượng người',
        'head_entity_types': ['DISASTER_TYPE'],
        'tail_entity_types': ['QUANTITY'],
        'examples': [
            'Bão số 12 ảnh hưởng đến 1000 người',
            'Động đất làm 5 người chết',
            'Lũ lụt khiến 20 người mất tích'
        ],
        'vietnamese_template': '{head} ảnh hưởng đến {tail} người',
        'confidence_threshold': 0.85
    },

    'HAS_INTENSITY': {
        'description': 'Thiên tai có cường độ hoặc cấp độ',
        'head_entity_types': ['DISASTER_TYPE'],
        'tail_entity_types': ['QUANTITY'],
        'examples': [
            'Bão số 12 có cấp gió 12',
            'Động đất có độ richter 5.5',
            'Lũ lụt có mức độ nghiêm trọng'
        ],
        'vietnamese_template': '{head} có cường độ {tail}',
        'confidence_threshold': 0.9
    },

    'REPORTED_BY': {
        'description': 'Thiên tai được báo cáo bởi tổ chức',
        'head_entity_types': ['DISASTER_TYPE'],
        'tail_entity_types': ['ORGANIZATION'],
        'examples': [
            'Bão số 12 được báo cáo bởi Trung tâm Dự báo Khí tượng',
            'Động đất được thông báo bởi Viện Vật lý Địa cầu'
        ],
        'vietnamese_template': '{head} được báo cáo bởi {tail}',
        'confidence_threshold': 0.7
    },

    'RESPONDED_BY': {
        'description': 'Thiên tai được ứng phó bởi tổ chức',
        'head_entity_types': ['DISASTER_TYPE'],
        'tail_entity_types': ['ORGANIZATION'],
        'examples': [
            'Bão số 12 được ứng phó bởi Ban Chỉ huy Phòng chống thiên tai',
            'Động đất được cứu hộ bởi lực lượng Quân đội'
        ],
        'vietnamese_template': '{head} được ứng phó bởi {tail}',
        'confidence_threshold': 0.75
    }
}

# Entity pair compatibility matrix
ENTITY_PAIR_COMPATIBILITY = {
    ('DISASTER_TYPE', 'LOCATION'): ['OCCURS_AT', 'OCCURS_IN'],
    ('DISASTER_TYPE', 'TIME'): ['OCCURS_ON'],
    ('DISASTER_TYPE', 'DAMAGE'): ['CAUSES_DAMAGE'],
    ('DISASTER_TYPE', 'QUANTITY'): ['AFFECTS_PEOPLE', 'HAS_INTENSITY'],
    ('DISASTER_TYPE', 'ORGANIZATION'): ['REPORTED_BY', 'RESPONDED_BY'],
    ('LOCATION', 'DISASTER_TYPE'): ['HOSTS_DISASTER'],  # Reverse relation
    ('TIME', 'DISASTER_TYPE'): ['HAS_DISASTER'],       # Reverse relation
    ('DAMAGE', 'DISASTER_TYPE'): ['CAUSED_BY'],         # Reverse relation
}

# Vietnamese text patterns for relation extraction
VIETNAMESE_PATTERNS = {
    'location_patterns': [
        r'xảy ra tại\s+([^,.!?]+)',
        r'tại\s+([^,.!?]+)\s+xảy ra',
        r'ở\s+([^,.!?]+)\s+hứng chịu',
        r'khu vực\s+([^,.!?]+)\s+bị',
        r'tỉnh\s+([^,.!?]+)\s+gặp'
    ],

    'time_patterns': [
        r'vào lúc\s+([^,.!?]+)',
        r'vào ngày\s+([^,.!?]+)',
        r'hôm\s+([^,.!?]+)\s+xảy ra',
        r'tuần\s+([^,.!?]+)\s+gặp',
        r'tháng\s+([^,.!?]+)\s+có'
    ],

    'damage_patterns': [
        r'gây thiệt hại\s+([^,.!?]+)',
        r'thiệt hại\s+([^,.!?]+)\s+do',
        r'làm\s+([^,.!?]+)\s+bị',
        r'ước tính\s+([^,.!?]+)\s+đồng',
        r'mất mát\s+([^,.!?]+)'
    ],

    'intensity_patterns': [
        r'cấp\s+(\d+)',
        r'độ richter\s+([\d.]+)',
        r'cường độ\s+([^,.!?]+)',
        r'mức độ\s+([^,.!?]+)'
    ]
}

# Confidence thresholds for different relation types
CONFIDENCE_THRESHOLDS = {
    'OCCURS_AT': 0.8,
    'OCCURS_IN': 0.7,
    'OCCURS_ON': 0.9,
    'CAUSES_DAMAGE': 0.8,
    'AFFECTS_PEOPLE': 0.85,
    'HAS_INTENSITY': 0.9,
    'REPORTED_BY': 0.7,
    'RESPONDED_BY': 0.75
}

# Training data examples for fine-tuning
TRAINING_EXAMPLES = [
    {
        'text': 'Bão số 12 xảy ra tại Hà Nội vào ngày 15/10, gây thiệt hại 20 tỷ đồng',
        'relations': [
            {'head': 'Bão số 12', 'tail': 'Hà Nội', 'relation': 'OCCURS_AT'},
            {'head': 'Bão số 12', 'tail': '15/10', 'relation': 'OCCURS_ON'},
            {'head': 'Bão số 12', 'tail': '20 tỷ đồng', 'relation': 'CAUSES_DAMAGE'}
        ]
    },
    {
        'text': 'Động đất có độ richter 5.5 xảy ra tại Kon Tum sáng nay',
        'relations': [
            {'head': 'Động đất', 'tail': 'Kon Tum', 'relation': 'OCCURS_AT'},
            {'head': 'Động đất', 'tail': 'sáng nay', 'relation': 'OCCURS_ON'},
            {'head': 'Động đất', 'tail': '5.5', 'relation': 'HAS_INTENSITY'}
        ]
    }
]

# Colors for visualization
RELATION_COLORS = {
    'OCCURS_AT': '#FF6B6B',
    'OCCURS_IN': '#4ECDC4',
    'OCCURS_ON': '#45B7D1',
    'CAUSES_DAMAGE': '#FFA07A',
    'AFFECTS_PEOPLE': '#98D8C8',
    'HAS_INTENSITY': '#F7DC6F',
    'REPORTED_BY': '#BB8FCE',
    'RESPONDED_BY': '#85C1E9'
}