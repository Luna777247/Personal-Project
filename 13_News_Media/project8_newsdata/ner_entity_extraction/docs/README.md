# NER Entity Extraction for Disaster Information

## 📋 Tổng Quan

Hệ thống trích xuất thực thể (Named Entity Recognition - NER) nâng cao để tự động nhận diện và trích xuất thông tin từ bài báo thiên tai. Sử dụng các mô hình học máy hiện đại để nhận diện:

- **Loại thiên tai**: "bão Yagi", "lũ quét", "động đất 6.2 độ richter"
- **Địa điểm**: "Quảng Nam", "Philippines", "Tokyo"
- **Thời gian**: "ngày 12/11", "sáng 15/8"
- **Thiệt hại**: số người chết, mất tích, nhà hư hỏng
- **Tổ chức**: "Trung tâm dự báo KTTV", "FEMA"

## 🎯 Các Mô Hình NER Được Thử Nghiệm

### 1. **PhoNER** (PhoBERT-based NER)
- **Ưu điểm**: Tối ưu cho tiếng Việt, độ chính xác cao
- **Nhược điểm**: Cần GPU, thời gian xử lý lâu
- **Use case**: Production với tài nguyên đầy đủ

### 2. **VnCoreNLP** (Official Vietnamese NLP)
- **Ưu điểm**: Toolkit chính thức, ổn định
- **Nhược điểm**: Cần Java, setup phức tạp
- **Use case**: Research và academic

### 3. **spaCy Custom** (spaCy với mô hình tùy chỉnh)
- **Ưu điểm**: Dễ tùy chỉnh, nhanh trên CPU
- **Nhược điểm**: Cần training data, độ chính xác phụ thuộc data
- **Use case**: Customization cao, resource limited

### 4. **BERT NER** (BERT/Vietnamese-BERT + Fine-tuning)
- **Ưu điểm**: State-of-the-art accuracy, transfer learning
- **Nhược điểm**: Cần GPU, training time dài
- **Use case**: Best accuracy, research

## 📁 Cấu Trúc Thư Mục

```
ner_entity_extraction/
├── config/
│   ├── nlp_config.py          # Cấu hình các mô hình NER
│   └── entity_definitions.py  # Định nghĩa entity types
├── scripts/
│   ├── ner_extractor.py       # Base class cho NER
│   ├── phoner_extractor.py    # PhoNER implementation
│   ├── vncorenlp_extractor.py # VnCoreNLP implementation
│   ├── spacy_custom_extractor.py # spaCy custom implementation
│   ├── bert_ner_extractor.py  # BERT NER implementation
│   └── demo_ner.py           # Demo script
├── models/                    # Thư mục lưu models đã train
├── data/                      # Output từ demo
│   ├── ner_phoner_demo.json
│   ├── ner_vncorenlp_demo.json
│   ├── ner_spacy_custom_demo.json
│   └── ner_bert_ner_demo.json
├── docs/
│   └── README.md              # Tài liệu này
├── run.py                     # Script tiện ích
├── SUMMARY.md                 # Tóm tắt kỹ thuật
├── __init__.py               # Package marker
└── requirements.txt          # Dependencies
```

## 🚀 Cách Chạy Nhanh

### Chạy Demo Đầy Đủ (Tất Cả Mô Hình)
```bash
cd ner_entity_extraction
python run.py
```

### Chạy Demo Từng Mô Hình
```bash
# Chỉ PhoNER
python run.py --model phoner

# Chỉ VnCoreNLP
python run.py --model vncorenlp

# Chỉ spaCy Custom
python run.py --model spacy

# Chỉ BERT NER
python run.py --model bert
```

### Chạy So Sánh Các Mô Hình
```bash
python run.py --compare
```

### Test Model Loading
```bash
# Test PhoNER loading
python run.py --test phoner
```

## 📊 Kết Quả Demo Mẫu

### Input Sample
```
Bão số 9 đã đổ bộ vào tỉnh Quảng Nam vào sáng ngày 12/11,
gây gió mạnh cấp 12-13, sóng biển cao 5-7m. Theo Ban chỉ huy
PCT tỉnh Quảng Nam, có 3 người chết, 10 người bị thương.
```

### Output Ví Dụ (PhoNER)
```json
{
  "article_info": {
    "title": "Bão số 9 gây thiệt hại nặng tại các tỉnh miền Trung",
    "source": "vnexpress"
  },
  "entities": [
    {
      "text": "Bão số 9",
      "label": "DISASTER_TYPE",
      "confidence": 0.95,
      "context": "Bão số 9 đã đổ bộ vào tỉnh Quảng Nam"
    },
    {
      "text": "tỉnh Quảng Nam",
      "label": "LOCATION",
      "confidence": 0.88,
      "context": "đổ bộ vào tỉnh Quảng Nam vào sáng"
    },
    {
      "text": "sáng ngày 12/11",
      "label": "TIME",
      "confidence": 0.82,
      "context": "vào sáng ngày 12/11, gây gió"
    }
  ]
}
```

## ⚙️ Cấu Hình Entity Types

### Các Loại Entity
- **DISASTER_TYPE**: Loại thiên tai (bão, lũ, động đất,...)
- **LOCATION**: Địa điểm xảy ra
- **TIME**: Thời gian xảy ra
- **DAMAGE**: Thiệt hại (số người, tài sản)
- **ORGANIZATION**: Tổ chức liên quan
- **PERSON**: Người liên quan
- **QUANTITY**: Số lượng, kích thước

### Confidence Thresholds
```python
CONFIDENCE_THRESHOLDS = {
    "DISASTER_TYPE": 0.7,
    "LOCATION": 0.8,
    "TIME": 0.6,
    "DAMAGE": 0.75,
    "ORGANIZATION": 0.7,
    "PERSON": 0.8,
    "QUANTITY": 0.7
}
```

## 🔧 Dependencies & Setup

### Cài Đặt Cơ Bản
```bash
pip install -r requirements.txt
```

### PhoNER Setup
```bash
# PhoNER sẽ tự động download model khi chạy lần đầu
# Cần transformers, torch
```

### VnCoreNLP Setup
```bash
# Cần Java 8+
# vncorenlp sẽ tự động download model
```

### spaCy Custom Setup
```bash
# Cần spacy
python -m spacy download vi_core_news_lg
```

### BERT NER Setup
```bash
# Cần transformers, torch
# Sẽ training model từ đầu (có thể mất thời gian)
```

## 📊 So Sánh Hiệu Suất

| Mô Hình | Độ Chính Xác | Tốc Độ | Resource | Setup Complexity |
|---------|-------------|--------|----------|------------------|
| **PhoNER** | Cao | Trung bình | GPU recommended | Trung bình |
| **VnCoreNLP** | Trung bình | Nhanh | CPU | Cao |
| **spaCy Custom** | Trung bình-Khóa | Nhanh | CPU | Thấp |
| **BERT NER** | Rất cao | Chậm | GPU | Cao |

## 🎯 Use Cases Phù Hợp

### PhoNER
- Production system với accuracy cao
- Có GPU và thời gian setup
- Cần độ chính xác tối đa

### VnCoreNLP
- Academic research
- Stable, well-tested toolkit
- Khi cần Java environment

### spaCy Custom
- Quick prototyping
- Limited resources
- Easy customization
- CPU-only deployment

### BERT NER
- State-of-the-art performance
- Research và development
- Khi có large training data

## 🚀 Mở Rộng & Customization

### 1. Thêm Entity Types Mới
```python
# Trong entity_definitions.py
NEW_ENTITY_TYPES = {
    "WEATHER_CONDITION": {
        "description": "Điều kiện thời tiết",
        "examples": ["gió mạnh", "mưa lớn", "sóng cao"]
    }
}
```

### 2. Training Data Mới
- Thêm sample articles vào training data
- Label entities theo BIO format
- Fine-tune trên domain-specific data

### 3. Model Optimization
- Quantization cho deployment
- Model distillation
- Ensemble methods

### 4. Integration
- Kết hợp với keyword extraction
- Pipeline với main disaster system
- Real-time processing

## 📈 Kết Luận

Thư mục NER extraction cung cấp **state-of-the-art approach** cho entity extraction trong disaster information processing. Với 4 mô hình khác nhau, hệ thống có thể adapt cho various use cases từ research đến production.

**Khuyến nghị**: Bắt đầu với spaCy custom cho quick prototyping, sau đó scale lên PhoNER hoặc BERT cho production với high accuracy requirements.

**Next Steps**:
1. Test với real disaster news data
2. Fine-tune models với domain-specific data
3. Compare với keyword-based approach
4. Integrate vào main pipeline