# Entity Definitions for Disaster NER
# Named Entity Recognition for Disaster Information Extraction

# Entity Type Definitions
ENTITY_DEFINITIONS = {
    "DISASTER_TYPE": {
        "description": "Loại thiên tai",
        "examples": [
            "bão Yagi", "lũ quét", "động đất 6.2 độ richter",
            "sạt lở đất", "sóng thần", "hạn hán", "cháy rừng"
        ],
        "vietnamese_examples": [
            "bão số 9", "cơn bão nhiệt đới", "động đất mạnh",
            "lũ lụt lớn", "sạt lở đất", "sóng thần cao"
        ],
        "color": "#FF6B6B",  # Red
        "abbreviation": "DIS"
    },

    "LOCATION": {
        "description": "Địa điểm xảy ra thiên tai",
        "examples": [
            "Quảng Nam", "Philippines", "Tokyo", "Hà Nội",
            "Đà Nẵng", "Kon Tum", "Gia Lai"
        ],
        "vietnamese_examples": [
            "tỉnh Quảng Nam", "thành phố Đà Nẵng", "huyện Kon Plông",
            "xã Ea H'leo", "thôn Bon Bon"
        ],
        "color": "#4ECDC4",  # Teal
        "abbreviation": "LOC"
    },

    "TIME": {
        "description": "Thời gian xảy ra thiên tai",
        "examples": [
            "ngày 12/11", "sáng 15/8", "lúc 14:30",
            "hôm qua", "tuần trước", "tháng 10/2023"
        ],
        "vietnamese_examples": [
            "sáng ngày 12/11", "chiều tối 15/8", "lúc 14 giờ 30",
            "hôm qua", "tuần trước", "tháng 10 năm 2023"
        ],
        "color": "#45B7D1",  # Blue
        "abbreviation": "TIME"
    },

    "DAMAGE": {
        "description": "Thiệt hại từ thiên tai",
        "examples": [
            "5 người chết", "10 người mất tích", "100 ngôi nhà sập",
            "500 ha lúa ngập", "2 tỷ đồng thiệt hại"
        ],
        "vietnamese_examples": [
            "5 người chết", "10 người mất tích", "100 ngôi nhà bị sập",
            "500 ha lúa bị ngập", "thiệt hại 2 tỷ đồng"
        ],
        "color": "#FFA07A",  # Orange
        "abbreviation": "DAM"
    },

    "ORGANIZATION": {
        "description": "Tổ chức liên quan",
        "examples": [
            "Trung tâm dự báo KTTV", "FEMA", "Red Cross",
            "Ban chỉ huy phòng chống thiên tai", "UBND tỉnh"
        ],
        "vietnamese_examples": [
            "Trung tâm dự báo khí tượng thủy văn", "Ban chỉ huy PCTT",
            "Ủy ban nhân dân tỉnh", "Sở tài nguyên và môi trường"
        ],
        "color": "#98D8C8",  # Mint
        "abbreviation": "ORG"
    },

    "PERSON": {
        "description": "Người liên quan",
        "examples": [
            "ông Nguyễn Văn A", "bà Trần Thị B",
            "Chánh văn phòng UBND tỉnh", "Giám đốc Sở NN&PTNT"
        ],
        "vietnamese_examples": [
            "ông Nguyễn Văn An", "bà Trần Thị Bình",
            "Chánh văn phòng UBND tỉnh Quảng Nam"
        ],
        "color": "#F7DC6F",  # Yellow
        "abbreviation": "PER"
    },

    "QUANTITY": {
        "description": "Số lượng, kích thước",
        "examples": [
            "6.2 độ richter", "cấp 12", "cao 5-7m",
            "tốc độ 150km/h", "mưa 500mm"
        ],
        "vietnamese_examples": [
            "6.2 độ richter", "cấp gió 12", "sóng cao 5-7m",
            "tốc độ gió 150km/h", "lượng mưa 500mm"
        ],
        "color": "#BB8FCE",  # Purple
        "abbreviation": "QUA"
    }
}

# Entity Relationships
ENTITY_RELATIONSHIPS = {
    "DISASTER_TYPE": {
        "related_to": ["LOCATION", "TIME", "DAMAGE", "QUANTITY"],
        "context_keywords": ["xảy ra", "tại", "vào lúc", "gây", "mạnh"]
    },

    "LOCATION": {
        "related_to": ["DISASTER_TYPE", "TIME", "DAMAGE"],
        "context_keywords": ["tại", "ở", "tỉnh", "huyện", "xã", "thôn"]
    },

    "TIME": {
        "related_to": ["DISASTER_TYPE", "LOCATION"],
        "context_keywords": ["vào lúc", "ngày", "tháng", "năm", "sáng", "chiều"]
    },

    "DAMAGE": {
        "related_to": ["DISASTER_TYPE", "LOCATION"],
        "context_keywords": ["gây", "làm", "thiệt hại", "chết", "mất tích", "bị thương"]
    },

    "ORGANIZATION": {
        "related_to": ["LOCATION", "DISASTER_TYPE"],
        "context_keywords": ["theo", "theo thông tin từ", "ban", "trung tâm", "sở"]
    }
}

# Confidence Thresholds by Entity Type
CONFIDENCE_THRESHOLDS = {
    "DISASTER_TYPE": 0.7,
    "LOCATION": 0.8,
    "TIME": 0.6,
    "DAMAGE": 0.75,
    "ORGANIZATION": 0.7,
    "PERSON": 0.8,
    "QUANTITY": 0.7
}

# Post-processing Rules
POST_PROCESSING_RULES = {
    "merge_adjacent": {
        "enabled": True,
        "entity_types": ["LOCATION", "ORGANIZATION"],
        "max_distance": 3  # words
    },

    "filter_by_context": {
        "enabled": True,
        "disaster_keywords_required": True,
        "min_context_score": 0.3
    },

    "validate_patterns": {
        "enabled": True,
        "strict_validation": False,
        "allow_partial_matches": True
    }
}

# Training Data Statistics (for reference)
TRAINING_DATA_STATS = {
    "total_samples": 1000,
    "entity_distribution": {
        "DISASTER_TYPE": 250,
        "LOCATION": 300,
        "TIME": 150,
        "DAMAGE": 120,
        "ORGANIZATION": 100,
        "PERSON": 50,
        "QUANTITY": 30
    },
    "average_entities_per_sample": 2.5,
    "language_distribution": {
        "vietnamese": 0.85,
        "mixed": 0.15
    }
}