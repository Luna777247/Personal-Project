"""
Extraction Configuration Settings

This module contains configuration settings for the pattern-based
extraction system, including preprocessing options and output formats.
"""

import os
from typing import Dict, Any, List
from pathlib import Path


# Base paths
BASE_DIR = Path(__file__).parent.parent
CONFIG_DIR = BASE_DIR / "config"
DATA_DIR = BASE_DIR / "data"
SCRIPTS_DIR = BASE_DIR / "scripts"

# Extraction settings
EXTRACTION_SETTINGS = {
    # Pattern matching settings
    "case_sensitive": False,
    "unicode_support": True,
    "overlap_handling": "longest_match",  # longest_match, all_matches, no_overlap

    # Confidence and filtering
    "min_confidence": 0.6,
    "max_matches_per_type": 5,
    "enable_context_extraction": True,
    "context_window_size": 100,  # characters

    # Preprocessing options
    "preprocessing": {
        "normalize_unicode": True,
        "remove_extra_spaces": True,
        "standardize_numbers": True,
        "lowercase_text": False,
        "remove_punctuation": False
    },

    # Output settings
    "output_format": "json",  # json, csv, both
    "include_metadata": True,
    "include_context": True,
    "include_confidence_scores": True,

    # Performance settings
    "batch_size": 10,
    "max_workers": 4,
    "timeout_seconds": 30,

    # Logging settings
    "log_level": "INFO",
    "log_file": str(DATA_DIR / "extraction.log"),
    "enable_console_logging": True
}

# Entity type mappings
ENTITY_TYPE_MAPPING = {
    "DISASTER_TYPE": {
        "display_name": "Loại thiên tai",
        "color": "red",
        "priority": 1
    },
    "LOCATION": {
        "display_name": "Địa điểm",
        "color": "blue",
        "priority": 2
    },
    "TIME": {
        "display_name": "Thời gian",
        "color": "green",
        "priority": 3
    },
    "CASUALTY": {
        "display_name": "Thương vong",
        "color": "orange",
        "priority": 4
    },
    "DAMAGE": {
        "display_name": "Thiệt hại",
        "color": "purple",
        "priority": 5
    },
    "ORGANIZATION": {
        "display_name": "Tổ chức",
        "color": "brown",
        "priority": 6
    }
}

# Template rules for structured extraction
TEMPLATE_RULES = {
    "casualty_report": {
        "pattern": r"(?:thiệt\s+hại\s+về\s+người|thương\s+vong).*?(?=\n|$)",
        "sub_patterns": {
            "deaths": r"(\d+(?:\.\d+)?)\s*(?:người\s+)?(?:chết|thiệt\s+mạng)",
            "injured": r"(\d+(?:\.\d+)?)\s*(?:người\s+)?(?:bị\s+thương|bị\s+đơn)",
            "missing": r"(\d+(?:\.\d+)?)\s*(?:người\s+)?(?:mất\s+tích|bị\s+mất\s+tích)"
        }
    },
    "damage_report": {
        "pattern": r"(?:thiệt\s+hại\s+về\s+vật\s+chất|thiệt\s+hại\s+kính\s+tế).*?(?=\n|$)",
        "sub_patterns": {
            "money_damage": r"(\d+(?:\.\d+)?)\s*(?:tỷ|triệu|nghìn)?\s*(?:đồng|VNĐ)",
            "houses_destroyed": r"(\d+(?:\.\d+)?)\s*(?:căn\s+nhà|ngôi\s+nhà)\s*(?:bị\s+sập|bị\s+phá\s+hủy)",
            "infrastructure": r"(\d+(?:\.\d+)?)\s*(?:km\s+đường|cây\s+cầu|cống\s+ngầm)"
        }
    },
    "disaster_description": {
        "pattern": r"(?:thiên\s+tai|mưa\s+bão|lũ\s+lụt).*?(?:khiến|dẫn\s+đến|gây).*?(?=\n|$)",
        "sub_patterns": {
            "disaster_type": r"(bão|lũ|động\s+đất|sạt\s+lở|hoả\s+hoạn)",
            "location": r"(?:tại|tỉnh|huyện|quận)\s+([^,\n]{1,30})",
            "impact": r"(?:khiến|gây)\s+([^,\n]{1,100})"
        }
    }
}

# Validation rules
VALIDATION_RULES = {
    "number_range": {
        "casualty_count": {"min": 0, "max": 10000},
        "damage_amount": {"min": 0, "max": 1000000},  # in billions VND
        "house_count": {"min": 0, "max": 100000}
    },
    "text_length": {
        "location_name": {"min": 2, "max": 50},
        "organization_name": {"min": 3, "max": 100}
    },
    "date_format": {
        "allowed_formats": ["DD/MM/YYYY", "DD-MM-YYYY", "DD tháng MM"],
        "future_date_allowed": False
    }
}

# Default output schema
OUTPUT_SCHEMA = {
    "extraction_id": "string",
    "timestamp": "datetime",
    "source_text": "string",
    "entities": [
        {
            "text": "string",
            "type": "string",
            "confidence": "float",
            "start_pos": "int",
            "end_pos": "int",
            "context": "string",
            "pattern_name": "string"
        }
    ],
    "metadata": {
        "total_entities": "int",
        "processing_time": "float",
        "patterns_used": ["string"],
        "confidence_stats": {
            "mean": "float",
            "min": "float",
            "max": "float"
        }
    }
}