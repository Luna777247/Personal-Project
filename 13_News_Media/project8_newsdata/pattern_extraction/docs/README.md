# Pattern-Based Extraction System

Hệ thống trích xuất dựa trên mẫu (Pattern / Rule-based Extraction) để trích xuất thông tin thiên tai có cấu trúc từ bài báo tiếng Việt.

## 📋 Tổng Quan

Hệ thống này sử dụng **Regular Expression (Regex)** và **Template Rules** để trích xuất thông tin có cấu trúc từ bài báo thiên tai. Phương pháp này đặc biệt hiệu quả khi bài báo có cấu trúc đồng nhất và format chuẩn.

### Ưu điểm
- ✅ **Độ chính xác cao** khi bài báo đồng nhất về format
- ✅ **Tốc độ xử lý nhanh** (500-1000 entities/giây)
- ✅ **Dễ hiểu và debug** (rules rõ ràng)
- ✅ **Không cần training data** (rule-based)
- ✅ **Chi phí thấp** (không cần GPU/điện toán mạnh)

### Nhược điểm
- ❌ **Không linh hoạt** → dễ lỗi nếu cấu trúc thay đổi
- ❌ **Cần domain knowledge** để viết patterns
- ❌ **Khó mở rộng** cho nhiều loại entities mới
- ❌ **Manual maintenance** khi format bài báo thay đổi

## 🚀 Cài Đặt

### Yêu cầu hệ thống
- Python 3.8+
- Các thư viện trong `requirements.txt`

### Cài đặt dependencies
```bash
cd pattern_extraction
pip install -r requirements.txt
```

### Cài đặt Vietnamese NLP (tùy chọn)
```bash
pip install spacy
python -m spacy download vi_core_news_lg
```

## 📁 Cấu Trúc Thư Mục

```
pattern_extraction/
├── __init__.py              # Package initialization
├── requirements.txt         # Python dependencies
├── run.py                   # CLI runner script
├── config/
│   ├── patterns.py          # Regex patterns & rules
│   └── settings.py          # Extraction settings
├── scripts/
│   ├── pattern_extractor.py # Main extractor class
│   └── demo_pattern_extraction.py  # Demo scripts
├── data/                    # Output data & logs
└── docs/                    # Documentation
```

## 🔧 Sử Dụng

### Chạy Demo
```bash
# Chạy tất cả demo
python run.py demo --mode all

# Chạy demo từng bài báo
python run.py demo --mode single

# Chạy demo batch processing
python run.py demo --mode batch

# Phân tích patterns
python run.py demo --mode analysis
```

### Trích xuất từ văn bản
```bash
# Trích xuất từ text trực tiếp
python run.py extract --text "Bão số 12 khiến 15 người chết tại Quảng Nam"

# Trích xuất từ file
python run.py extract --input-file news_article.txt --output results.json --show-context
```

### Xem patterns có sẵn
```bash
python run.py patterns
```

## 📊 Patterns & Rules

### Entity Types Hỗ Trợ

| Entity Type | Display Name | Mô tả | Ví dụ |
|-------------|-------------|--------|--------|
| DISASTER_TYPE | Loại thiên tai | Loại hình thiên tai | bão số 12, lũ quét |
| LOCATION | Địa điểm | Nơi xảy ra thiên tai | tỉnh Quảng Nam, huyện Mường Khương |
| TIME | Thời gian | Thời điểm xảy ra | 15/11/2023, lúc 14h30 |
| CASUALTY | Thương vong | Thiệt hại về người | 15 người chết, 27 bị thương |
| DAMAGE | Thiệt hại | Thiệt hại vật chất | 100 tỷ đồng, 150 nhà sập |
| ORGANIZATION | Tổ chức | Tổ chức liên quan | Bộ Nông nghiệp, Đội cứu hộ |

### Regex Patterns Ví Dụ

#### Thiệt Hại Về Người
```python
# Số người chết
r'(\d+(?:\.\d+)?)\s*(người\s+)?(?:đã\s+)?(?:thiệt\s+mạng|chết|tử\s+vong)'

# Số người mất tích
r'(\d+(?:\.\d+)?)\s*(người\s+)?(?:mất\s+tích|bị\s+mất\s+tích)'

# Số người bị thương
r'(\d+(?:\.\d+)?)\s*(người\s+)?(?:bị\s+thương|bị\s+đơn)'
```

#### Thiệt Hại Vật Chất
```python
# Thiệt hại tiền tệ
r'(?:thiệt\s+hại|thiệt\s+hại\s+khoảng)\s+(\d+(?:\.\d+)?)\s*(tỷ|triệu|nghìn)?\s*(?:đồng|VNĐ)'

# Nhà cửa bị phá hủy
r'(\d+(?:\.\d+)?)\s*(căn\s+)?(?:nhà\s+)?(?:bị\s+sập|bị\s+phá\s+hủy|bị\s+thiệt\s+hại)'
```

#### Template Rules
```python
# Báo cáo thương vong
"casualty_report": {
    "pattern": r"(?:thiệt\s+hại\s+về\s+người|thương\s+vong).*?(?=\n|$)",
    "sub_patterns": {
        "deaths": r"(\d+(?:\.\d+)?)\s*(?:người\s+)?(?:chết|thiệt\s+mạng)",
        "injured": r"(\d+(?:\.\d+)?)\s*(?:người\s+)?(?:bị\s+thương|bị\s+đơn)",
        "missing": r"(\d+(?:\.\d+)?)\s*(?:người\s+)?(?:mất\s+tích|bị\s+mất\s+tích)"
    }
}
```

## 🎯 API Usage

### Basic Usage
```python
from scripts.pattern_extractor import PatternBasedExtractor

# Khởi tạo extractor
extractor = PatternBasedExtractor()

# Trích xuất từ văn bản
text = "Bão số 12 khiến 15 người chết tại Quảng Nam"
entities = extractor.extract_entities(text)

for entity in entities:
    print(f"{entity.entity_type}: {entity.text} (confidence: {entity.confidence})")
```

### Batch Processing
```python
# Xử lý nhiều văn bản
texts = ["Bão số 12...", "Lũ quét tại Lào Cai..."]
results = extractor.extract_from_texts(texts, batch_size=5)

# Lưu kết quả
extractor.save_results(results, "output.json")
```

### Custom Configuration
```python
# Cấu hình tùy chỉnh
config = {
    "min_confidence": 0.8,
    "max_matches_per_type": 3,
    "preprocessing": {
        "normalize_unicode": True,
        "remove_extra_spaces": True
    }
}

extractor = PatternBasedExtractor(config=config)
```

## 📈 Performance & Accuracy

### Metrics (Estimated)
- **Precision**: 85-95% (đối với bài báo format chuẩn)
- **Recall**: 60-75% (phụ thuộc vào pattern coverage)
- **Speed**: 500-1000 entities/second
- **Memory**: ~50MB cho 1000 patterns

### Factors Affecting Performance
- **Pattern Quality**: Regex patterns càng cụ thể càng chính xác
- **Text Format**: Bài báo đồng nhất format cho kết quả tốt nhất
- **Domain Knowledge**: Hiểu cấu trúc bài báo để viết patterns hiệu quả

## 🔧 Customization

### Thêm Pattern Mới
```python
from config.patterns import ExtractionPattern

# Thêm pattern mới
new_pattern = ExtractionPattern(
    name="new_disaster_pattern",
    pattern=r"(?:mưa\s+lớn|lũ\s+lụt)\s+([^,\n]{1,50})",
    entity_type="DISASTER_TYPE",
    confidence=0.8,
    examples=["mưa lớn", "lũ lụt"]
)

# Thêm vào danh sách patterns
from config.patterns import ALL_PATTERNS
ALL_PATTERNS.append(new_pattern)
```

### Tùy Chỉnh Settings
```python
# Trong settings.py
EXTRACTION_SETTINGS.update({
    "min_confidence": 0.7,
    "max_matches_per_type": 5,
    "context_window_size": 150
})
```

## 🧪 Testing & Validation

### Chạy Tests
```bash
# Chạy demo để validate
python run.py demo --mode all

# Test với data thực tế
python run.py extract --input-file real_news.txt --output validation.json
```

### Validation Metrics
- **Manual Review**: Kiểm tra 100 samples đầu tiên
- **Precision Check**: TP / (TP + FP)
- **Recall Check**: TP / (TP + FN)
- **F1 Score**: 2 * Precision * Recall / (Precision + Recall)

## 📋 Examples

### Input Text
```
Bão số 12 gây thiệt hại nặng nề tại các tỉnh miền Trung. Theo báo cáo sơ bộ,
cơn bão đã khiến 15 người thiệt mạng, 27 người bị thương và 5 người mất tích.
Thiệt hại về vật chất ước tính khoảng 1.200 tỷ đồng, với 150 căn nhà bị sập
hoàn toàn và hàng trăm hecta lúa bị ngập úng.
```

### Output Entities
```
DISASTER_TYPE: 'Bão số 12' (confidence: 90%)
LOCATION: 'tỉnh miền Trung' (confidence: 85%)
CASUALTY: '15 người thiệt mạng' (confidence: 95%)
CASUALTY: '27 người bị thương' (confidence: 85%)
CASUALTY: '5 người mất tích' (confidence: 90%)
DAMAGE: '1.200 tỷ đồng' (confidence: 90%)
DAMAGE: '150 căn nhà bị sập' (confidence: 85%)
```

## 🚨 Troubleshooting

### Common Issues
1. **Low Accuracy**: Kiểm tra pattern quality và text preprocessing
2. **Missing Entities**: Thêm patterns mới cho entity types chưa được cover
3. **False Positives**: Tăng confidence threshold hoặc cải thiện patterns
4. **Performance Issues**: Giảm batch size hoặc tối ưu regex patterns

### Debug Mode
```bash
# Chạy với debug logging
python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
from scripts.pattern_extractor import PatternBasedExtractor
extractor = PatternBasedExtractor()
# ... run extraction
"
```

## 🔄 Integration

### Với NER Pipeline
```python
# Kết hợp với NER system
from ner_entity_extraction.scripts.ner_extractor import NERExtractor
from scripts.pattern_extractor import PatternBasedExtractor

# NER trước
ner_extractor = NERExtractor()
ner_entities = ner_extractor.extract_entities(text)

# Pattern extraction sau
pattern_extractor = PatternBasedExtractor()
pattern_entities = pattern_extractor.extract_entities(text)

# Merge results
all_entities = ner_entities + pattern_entities
```

### Với Relation Extraction
```python
# Feed vào relation extraction
from relation_extraction.scripts.relation_extractor import RelationExtractor

relation_extractor = RelationExtractor()
relations = relation_extractor.extract_relations(pattern_entities)
```

## 📚 References

- [Python re module](https://docs.python.org/3/library/re.html)
- [Regex patterns for Vietnamese text](https://github.com/undertheseanlp)
- [Named Entity Recognition best practices](https://spacy.io/usage/linguistic-features#named-entities)

## 🤝 Contributing

1. Fork repository
2. Tạo feature branch
3. Thêm patterns mới trong `config/patterns.py`
4. Test với demo scripts
5. Submit pull request

## 📄 License

MIT License - Xem LICENSE file để biết thêm chi tiết.