"""
Pattern Configuration for Disaster Information Extraction

This module defines regex patterns and template rules for extracting
structured disaster information from Vietnamese news articles.
"""

import re
from typing import Dict, List, Pattern, Any
from dataclasses import dataclass


@dataclass
class ExtractionPattern:
    """Data class for extraction patterns"""
    name: str
    pattern: str
    compiled_pattern: Pattern = None
    entity_type: str = ""
    confidence: float = 0.8
    examples: List[str] = None

    def __post_init__(self):
        if self.compiled_pattern is None:
            self.compiled_pattern = re.compile(self.pattern, re.IGNORECASE | re.UNICODE)


# Disaster Type Patterns
DISASTER_TYPE_PATTERNS = [
    ExtractionPattern(
        name="storm_pattern",
        pattern=r'(bão|bão số\s*\d+|cơn\s+bão)\s+([^,\n]{1,50})',
        entity_type="DISASTER_TYPE",
        confidence=0.9,
        examples=["bão số 12", "cơn bão Damrey"]
    ),
    ExtractionPattern(
        name="flood_pattern",
        pattern=r'(lũ|lũ\s+lụt|lũ\s+quét)\s+([^,\n]{1,50})',
        entity_type="DISASTER_TYPE",
        confidence=0.9,
        examples=["lũ quét", "lũ lụt"]
    ),
    ExtractionPattern(
        name="earthquake_pattern",
        pattern=r'(động\s+đất|động\s+đất\s+\d+\.\d+)\s+([^,\n]{1,50})',
        entity_type="DISASTER_TYPE",
        confidence=0.9,
        examples=["động đất 6.5 độ"]
    ),
    ExtractionPattern(
        name="typhoon_pattern",
        pattern=r'(bão\s+tropical|bão\s+cường\s+lượng|bão\s+hurricane)\s+([^,\n]{1,50})',
        entity_type="DISASTER_TYPE",
        confidence=0.8,
        examples=["bão tropical"]
    )
]

# Location Patterns
LOCATION_PATTERNS = [
    ExtractionPattern(
        name="province_pattern",
        pattern=r'(tỉnh|thành\s+phố)\s+([^,\n]{1,30})',
        entity_type="LOCATION",
        confidence=0.85,
        examples=["tỉnh Lào Cai", "thành phố Hà Nội"]
    ),
    ExtractionPattern(
        name="district_pattern",
        pattern=r'(huyện|quận|thị\s+xã)\s+([^,\n]{1,30})',
        entity_type="LOCATION",
        confidence=0.8,
        examples=["huyện Mường Khương", "quận Hoàn Kiếm"]
    ),
    ExtractionPattern(
        name="region_pattern",
        pattern=r'(miền|miền\s+(bắc|nam|trung)|vùng\s+([^,\n]{1,30}))',
        entity_type="LOCATION",
        confidence=0.7,
        examples=["miền Bắc", "vùng núi"]
    )
]

# Time Patterns
TIME_PATTERNS = [
    ExtractionPattern(
        name="date_pattern",
        pattern=r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}\s+tháng\s+\d{1,2})',
        entity_type="TIME",
        confidence=0.9,
        examples=["15/10/2023", "15 tháng 10"]
    ),
    ExtractionPattern(
        name="time_pattern",
        pattern=r'(lúc\s+\d{1,2}:\d{2}|vào\s+lúc\s+\d{1,2}h|\d{1,2}\s+giờ\s+\d{1,2})',
        entity_type="TIME",
        confidence=0.8,
        examples=["lúc 14:30", "vào lúc 14h"]
    ),
    ExtractionPattern(
        name="duration_pattern",
        pattern=r'(trong\s+vòng\s+\d+\s+(ngày|giờ|tiếng)|\d+\s+(ngày|giờ|tiếng)\s+qua)',
        entity_type="TIME",
        confidence=0.7,
        examples=["trong vòng 3 ngày", "2 giờ qua"]
    )
]

# Damage/Casualty Patterns
DAMAGE_PATTERNS = [
    ExtractionPattern(
        name="death_pattern",
        pattern=r'(\d+(?:\.\d+)?)\s*(người\s+)?(?:đã\s+)?(?:thiệt\s+mạng|chết|tử\s+vong)',
        entity_type="CASUALTY",
        confidence=0.95,
        examples=["15 người chết", "27 thiệt mạng"]
    ),
    ExtractionPattern(
        name="missing_pattern",
        pattern=r'(\d+(?:\.\d+)?)\s*(người\s+)?(?:mất\s+tích|bị\s+mất\s+tích)',
        entity_type="CASUALTY",
        confidence=0.9,
        examples=["5 người mất tích", "12 bị mất tích"]
    ),
    ExtractionPattern(
        name="injured_pattern",
        pattern=r'(\d+(?:\.\d+)?)\s*(người\s+)?(?:bị\s+thương|bị\s+đơn)',
        entity_type="CASUALTY",
        confidence=0.85,
        examples=["30 người bị thương", "45 bị đơn"]
    ),
    ExtractionPattern(
        name="damage_money_pattern",
        pattern=r'(?:thiệt\s+hại|thiệt\s+hại\s+khoảng)\s+(\d+(?:\.\d+)?)\s*(tỷ|triệu|nghìn)?\s*(?:đồng|VNĐ)',
        entity_type="DAMAGE",
        confidence=0.9,
        examples=["thiệt hại khoảng 100 tỷ đồng", "thiệt hại 50 triệu VNĐ"]
    ),
    ExtractionPattern(
        name="damage_houses_pattern",
        pattern=r'(\d+(?:\.\d+)?)\s*(căn\s+)?(?:nhà\s+)?(?:bị\s+sập|bị\s+phá\s+hủy|bị\s+thiệt\s+hại)',
        entity_type="DAMAGE",
        confidence=0.85,
        examples=["150 căn nhà bị sập", "20 nhà bị phá hủy"]
    )
]

# Organization Patterns
ORGANIZATION_PATTERNS = [
    ExtractionPattern(
        name="gov_org_pattern",
        pattern=r'(?:Bộ\s+|Ủy\s+ban\s+|Sở\s+|Ban\s+|Chính\s+phủ|Chính\s+quyền)\s+([^,\n]{1,50})',
        entity_type="ORGANIZATION",
        confidence=0.8,
        examples=["Bộ Nông nghiệp", "Ủy ban nhân dân"]
    ),
    ExtractionPattern(
        name="relief_org_pattern",
        pattern=r'(?:Đội\s+cứu\s+hộ|Hội\s+Chữ\s+thập\s+đỏ|Quân\s+đội|Lực\s+lượng\s+vũ\s+trang)\s+([^,\n]{1,50})',
        entity_type="ORGANIZATION",
        confidence=0.85,
        examples=["Đội cứu hộ", "Hội Chữ thập đỏ"]
    )
]

# Combine all patterns
ALL_PATTERNS = (
    DISASTER_TYPE_PATTERNS +
    LOCATION_PATTERNS +
    TIME_PATTERNS +
    DAMAGE_PATTERNS +
    ORGANIZATION_PATTERNS
)

# Pattern categories for easy access
PATTERN_CATEGORIES = {
    "disaster_types": DISASTER_TYPE_PATTERNS,
    "locations": LOCATION_PATTERNS,
    "times": TIME_PATTERNS,
    "damages": DAMAGE_PATTERNS,
    "organizations": ORGANIZATION_PATTERNS,
    "all": ALL_PATTERNS
}

# Extraction configuration
EXTRACTION_CONFIG = {
    "max_matches_per_pattern": 10,
    "min_confidence_threshold": 0.6,
    "overlap_resolution": "keep_highest_confidence",  # or "keep_all"
    "context_window": 50,  # characters around match for context
    "enable_preprocessing": True,
    "preprocessing_steps": [
        "normalize_unicode",
        "remove_extra_spaces",
        "standardize_numbers"
    ]
}