# Relation Extraction for Disaster Information

## 📋 Tổng Quan

Thư mục `relation_extraction/` triển khai hệ thống **Relation Extraction (RE)** nâng cao để tự động trích xuất quan hệ giữa các entities trong bài báo thiên tai. Đây là bước tiếp theo sau NER, giúp trả lời các câu hỏi như:

- **Thiên tai gì?** xảy ra ở đâu?
- **Thiên tai gì?** xảy ra khi nào?
- **Thiên tai gì?** gây hậu quả như thế nào?

## 🎯 Các Loại Quan Hệ Hỗ Trợ

| Loại Quan Hệ | Mô Tả | Ví Dụ |
|-------------|--------|--------|
| `OCCURS_AT` | Thiên tai xảy ra tại địa điểm | "Bão số 12 xảy ra tại Hà Nội" |
| `OCCURS_IN` | Thiên tai xảy ra trong khu vực | "Lũ quét xảy ra trong tỉnh Lào Cai" |
| `OCCURS_ON` | Thiên tai xảy ra vào thời gian | "Động đất xảy ra vào sáng nay" |
| `CAUSES_DAMAGE` | Thiên tai gây thiệt hại | "Bão gây thiệt hại 20 tỷ đồng" |
| `AFFECTS_PEOPLE` | Ảnh hưởng đến số người | "Bão ảnh hưởng đến 1000 người" |
| `HAS_INTENSITY` | Cường độ của thiên tai | "Động đất có độ richter 5.5" |
| `REPORTED_BY` | Được báo cáo bởi tổ chức | "Bão được báo cáo bởi Trung tâm Dự báo" |
| `RESPONDED_BY` | Được ứng phó bởi tổ chức | "Bão được ứng phó bởi Ban Chỉ huy" |

## 🏗️ Kiến Trúc Hệ Thống

### Core Components

#### 1. Base RelationExtractor Class
```python
from scripts.relation_extractor import RelationExtractor, Relation
```
- Abstract base class cho tất cả RE models
- Xử lý batch processing và output formatting
- Validation và filtering relations

#### 2. Model Implementations

##### PhoBERT RE Extractor
```python
from scripts.phobert_re_extractor import PhoBERTREExtractor
```
- Sử dụng PhoBERT fine-tuned cho relation classification
- Input format: `[HEAD] [SEP] [TAIL] [SEP] [CONTEXT]`
- Training với custom disaster relation dataset

##### LLM RE Extractor
```python
from scripts.llm_re_extractor import LLMREExtractor
```
- Sử dụng prompt engineering với LLM (GPT, Claude, Groq)
- Zero-shot relation extraction
- Support caching để tối ưu cost

##### Rule-based RE Extractor
```python
from scripts.rule_based_re_extractor import RuleBasedREExtractor
```
- Pattern matching với regex
- Entity-aware relation extraction
- High precision, customizable patterns

## 🚀 Cài Đặt và Sử Dụng

### 1. Cài Đặt Dependencies
```bash
cd relation_extraction
pip install -r requirements.txt
```

### 2. Cấu Hình API Keys (cho LLM)
Tạo file `.env`:
```bash
OPENAI_API_KEY=your_openai_key
GROQ_API_KEY=your_groq_key
```

### 3. Chạy Demo
```bash
# Test model loading
python run.py --test-loading

# Chạy demo cho model cụ thể
python run.py --model rule       # Rule-based
python run.py --model phobert    # PhoBERT
python run.py --model llm        # LLM-based

# Chạy comparison tất cả models
python run.py --compare

# Chạy full demo (default)
python run.py
```

## 📊 Kết Quả Demo

### Sample Output Format
```json
{
  "article_title": "Bão số 12 gây thiệt hại nặng nề tại Hà Nội",
  "article_url": "https://example.com/article1",
  "relations": [
    {
      "head_entity": "Bão số 12",
      "tail_entity": "Hà Nội",
      "relation_type": "OCCURS_AT",
      "confidence": 0.85,
      "head_entity_type": "DISASTER_TYPE",
      "tail_entity_type": "LOCATION",
      "sentence": "Bão số 12 xảy ra tại Hà Nội vào ngày 15/10"
    }
  ],
  "processing_time": 0.15,
  "model_used": "Rule-Based-RE",
  "confidence_score": 0.85
}
```

## 🔧 Cấu Hình Chi Tiết

### Model Configurations (`config/re_config.py`)

#### PhoBERT RE Config
```python
MODEL_CONFIGS['phobert_re'] = {
    'model_name': 'vinai/phobert-base',
    'max_length': 256,
    'batch_size': 16,
    'learning_rate': 2e-5,
    'num_epochs': 10,
    'device': 'auto',
    'save_path': 'models/phobert_re',
    'relation_classes': ['OCCURS_AT', 'OCCURS_ON', 'CAUSES_DAMAGE', ...]
}
```

#### LLM RE Config
```python
MODEL_CONFIGS['llm_re'] = {
    'provider': 'openai',
    'model': 'gpt-3.5-turbo',
    'temperature': 0.1,
    'max_tokens': 500,
    'api_key_env': 'OPENAI_API_KEY',
    'prompt_template': '...',
    'fallback_provider': 'groq'
}
```

#### Rule-based RE Config
```python
MODEL_CONFIGS['rule_based_re'] = {
    'patterns': {
        'OCCURS_AT': [
            r'({disaster}) xảy ra tại ({location})',
            r'({disaster}) tại ({location})'
        ]
    },
    'entity_placeholders': {
        'disaster': ['bão', 'lũ', 'động đất'],
        'location': ['Hà Nội', 'TP.HCM', 'Đà Nẵng']
    }
}
```

## 🎯 Performance Comparison

| Model | Precision | Recall | Speed | Resource Usage |
|-------|-----------|--------|-------|----------------|
| Rule-based | High | Medium | Fast | Low |
| PhoBERT | High | High | Medium | Medium |
| LLM | Medium | High | Slow | High (API calls) |

### Use Cases
- **Rule-based**: Production với high precision requirements
- **PhoBERT**: Balanced performance cho offline processing
- **LLM**: Research, flexible relations, low development time

## 🔄 Integration với NER Pipeline

### Workflow
1. **NER** → Extract entities từ text
2. **RE** → Extract relations giữa entities
3. **Knowledge Graph** → Build graph từ entities và relations

### Example Integration
```python
# Từ NER system
entities = ner_extractor.extract_entities(article_text)

# Feed vào RE system
relations = re_extractor.extract_relations(article_text, entities)

# Kết hợp thành knowledge graph
knowledge_graph = build_graph(entities, relations)
```

## 📈 Training và Fine-tuning

### PhoBERT RE Training
```python
from scripts.phobert_re_extractor import PhoBERTREExtractor

extractor = PhoBERTREExtractor(config)
extractor.train(training_data, num_epochs=10)
```

### Custom Rule Addition
```python
from scripts.rule_based_re_extractor import RuleBasedREExtractor

extractor = RuleBasedREExtractor(config)
extractor.add_pattern('NEW_RELATION', r'pattern here')
```

## 🛠️ Development

### Adding New Relation Types
1. Thêm vào `RELATION_DEFINITIONS` trong `relation_definitions.py`
2. Update patterns trong config
3. Add training examples nếu cần

### Adding New Models
1. Extend `RelationExtractor` base class
2. Implement `extract_relations()` method
3. Add to `MODEL_CONFIGS`
4. Update demo script

## 📁 File Structure

```
relation_extraction/
├── __init__.py                    # Package initialization
├── requirements.txt              # Python dependencies
├── run.py                        # CLI runner script
├── config/
│   ├── re_config.py             # Model configurations
│   └── relation_definitions.py  # Relation type definitions
├── scripts/
│   ├── relation_extractor.py    # Base RE class
│   ├── phobert_re_extractor.py  # PhoBERT implementation
│   ├── llm_re_extractor.py      # LLM implementation
│   ├── rule_based_re_extractor.py # Rule-based implementation
│   └── demo_re.py               # Demo script
├── docs/
│   └── README.md                # This documentation
└── data/                        # Output directory
    ├── re_results_*.json        # Model results
    ├── re_comparison_summary.json # Comparison results
    └── re_model_loading_test.json # Loading tests
```

## 🤝 Contributing

1. Fork repository
2. Create feature branch
3. Add tests cho new functionality
4. Update documentation
5. Submit pull request

## 📄 License

This project is part of the Disaster Information Extraction system.

## 🙋 Support

For questions or issues:
- Check existing documentation
- Run demo scripts để troubleshoot
- Check logs trong data/ directory