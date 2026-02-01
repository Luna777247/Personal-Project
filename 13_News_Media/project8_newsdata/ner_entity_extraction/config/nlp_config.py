# NER Entity Extraction Configuration
# Named Entity Recognition for Disaster Information Extraction

# Entity Types to Extract
NER_ENTITY_TYPES = {
    "DISASTER_TYPE": "Loại thiên tai (bão, lũ, động đất, v.v.)",
    "LOCATION": "Địa điểm xảy ra thiên tai",
    "TIME": "Thời gian xảy ra thiên tai",
    "DAMAGE": "Thiệt hại (số người chết, mất tích, nhà hư hỏng)",
    "ORGANIZATION": "Tổ chức liên quan (Trung tâm dự báo, FEMA, v.v.)",
    "PERSON": "Người liên quan",
    "QUANTITY": "Số lượng, kích thước (độ richter, cấp gió, v.v.)"
}

# Model Configurations
MODEL_CONFIGS = {
    "phoner": {
        "name": "PhoNER",
        "description": "PhoBERT-based NER for Vietnamese",
        "model_path": "vinai/phobert-base",
        "tokenizer_path": "vinai/phobert-base",
        "ner_model_path": "vinai/phobert-base-v2",
        "language": "vi",
        "supported_entities": ["LOCATION", "PERSON", "ORGANIZATION", "MISC"],
        "custom_entities": ["DISASTER_TYPE", "TIME", "DAMAGE", "QUANTITY"]
    },

    "vncorenlp": {
        "name": "VnCoreNLP",
        "description": "Official Vietnamese NLP toolkit",
        "model_path": "VnCoreNLP-1.1.1.jar",
        "annotators": ["wseg", "pos", "ner", "parse"],
        "language": "vi",
        "supported_entities": ["LOCATION", "PERSON", "ORGANIZATION"],
        "custom_entities": ["DISASTER_TYPE", "TIME", "DAMAGE", "QUANTITY"]
    },

    "spacy_custom": {
        "name": "spaCy Custom",
        "description": "spaCy with custom NER model",
        "base_model": "vi_core_news_lg",
        "language": "vi",
        "training_data_path": "data/training_data.json",
        "model_output_path": "models/spacy_custom",
        "supported_entities": ["DISASTER_TYPE", "LOCATION", "TIME", "DAMAGE", "ORGANIZATION", "PERSON", "QUANTITY"]
    },

    "bert_ner": {
        "name": "BERT NER",
        "description": "BERT-based NER with fine-tuning",
        "base_model": "bert-base-multilingual-cased",
        "vietnamese_model": "vinai/phobert-base",
        "language": "vi",
        "max_length": 256,
        "batch_size": 16,
        "learning_rate": 5e-5,
        "num_epochs": 10,
        "model_output_path": "models/bert_ner",
        "supported_entities": ["DISASTER_TYPE", "LOCATION", "TIME", "DAMAGE", "ORGANIZATION", "PERSON", "QUANTITY"]
    }
}

# Extraction Configuration
EXTRACTION_CONFIG = {
    "min_confidence": 0.5,  # Minimum confidence for entity extraction
    "max_entities_per_type": 10,  # Maximum entities to extract per type
    "context_window": 2,  # Sentences before/after for context
    "remove_duplicates": True,  # Remove duplicate entities
    "case_sensitive": False,  # Case sensitivity for matching
    "language": "vi",  # Primary language
    "fallback_models": ["phoner", "vncorenlp", "spacy_custom"],  # Fallback order
    "cache_models": True,  # Cache downloaded models
    "parallel_processing": False,  # Enable parallel processing
    "batch_size": 10  # Batch size for processing
}

# Disaster-specific Keywords for Enhancement
DISASTER_KEYWORDS = {
    "storm": ["bão", "cơn bão", "bão nhiệt đới", "áp thấp nhiệt đới", "lốc xoáy"],
    "flood": ["lũ", "lũ lụt", "ngập lụt", "lũ quét", "lũ ống"],
    "earthquake": ["động đất", "động đất", "trận động đất"],
    "landslide": ["sạt lở", "sạt lở đất", "lở đất"],
    "tsunami": ["sóng thần", "thần sóng"],
    "drought": ["hạn hán", "khô hạn"],
    "fire": ["cháy", "hỏa hoạn", "cháy rừng"],
    "volcano": ["núi lửa", "phun trào"],
    "typhoon": ["bão", "cơn bão", "bão số"],
    "hurricane": ["bão", "cơn bão"]
}

# Location Keywords for Vietnamese locations
VIETNAM_LOCATIONS = [
    "Hà Nội", "Hồ Chí Minh", "Đà Nẵng", "Cần Thơ", "Hải Phòng",
    "Quảng Nam", "Quảng Ngãi", "Bình Định", "Phú Yên", "Khánh Hòa",
    "Ninh Thuận", "Bình Thuận", "Kon Tum", "Gia Lai", "Đắk Lắk",
    "Đắk Nông", "Lâm Đồng", "Bình Phước", "Tây Ninh", "Bình Dương",
    "Đồng Nai", "Bà Rịa - Vũng Tàu", "Long An", "Tiền Giang",
    "Bến Tre", "Trà Vinh", "Vĩnh Long", "Đồng Tháp", "An Giang",
    "Kiên Giang", "Cà Mau", "Bạc Liêu", "Sóc Trăng", "Hậu Giang"
]

# Time Patterns for Vietnamese
TIME_PATTERNS = [
    r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',  # dd/mm/yyyy
    r'\d{1,2}[:/]\d{1,2}',  # hh:mm
    r'sáng|trưa|chiều|tối',  # time of day
    r'hôm qua|hôm nay|ngày mai',  # relative days
    r'tuần này|tháng này|năm nay',  # relative periods
    r'tháng \d{1,2}|năm \d{4}',  # specific months/years
]

# Damage Patterns
DAMAGE_PATTERNS = [
    r'\d+ người chết',
    r'\d+ người mất tích',
    r'\d+ người bị thương',
    r'\d+ ngôi nhà',
    r'\d+ ha đất',
    r'\d+ tỷ đồng',
    r'\d+ triệu đồng',
    r'\d+ căn nhà',
    r'\d+ công trình'
]

# Organization Patterns
ORGANIZATION_PATTERNS = [
    r'Trung tâm dự báo',
    r'Ban chỉ huy',
    r'Ủy ban nhân dân',
    r'Sở tài nguyên',
    r'Cục quản lý',
    r'Bộ .*',
    r'Tỉnh ủy',
    r'Huyện ủy'
]