# OCR Banking - Document Recognition System

## 📋 Tổng quan

Hệ thống OCR Banking là giải pháp tự động nhận diện và trích xuất thông tin từ các giấy tờ ngân hàng, được thiết kế đặc biệt cho quy trình eKYC của MB Bank.

## 🎯 Tính năng chính

### 1. Text Detection (Phát hiện văn bản)
- **CRAFT** (Character Region Awareness For Text detection)
- **DBNet** (Differentiable Binarization)
- Phát hiện chính xác vùng văn bản trong ảnh
- Hỗ trợ nhiều layout phức tạp

### 2. Text Recognition (Nhận diện văn bản)
- **VietOCR**: Tối ưu cho tiếng Việt
- **PaddleOCR**: Hỗ trợ đa ngôn ngữ
- **EasyOCR**: Dễ sử dụng, độ chính xác cao
- Nhận diện chữ viết tay và in

### 3. Information Extraction (Trích xuất thông tin)
- Trích xuất tự động các trường thông tin
- Regex patterns cho tài liệu Việt Nam
- Post-processing với fuzzy matching
- Validation và format chuẩn hóa

### 4. Document Types (Loại tài liệu)
- **CCCD**: Căn cước công dân (12 số)
- **CMND**: Chứng minh nhân dân (9 số)
- **Sao kê ngân hàng**: Lịch sử giao dịch
- **Hợp đồng vay**: Thông tin khoản vay

## 🏗️ Kiến trúc hệ thống

```
project19_ocr_banking/
├── src/
│   ├── detection/           # Text detection models
│   │   ├── craft_detector.py
│   │   └── __init__.py
│   ├── recognition/         # Text recognition models
│   │   ├── vietocr_recognizer.py
│   │   └── __init__.py
│   ├── extraction/          # Information extraction
│   │   ├── field_extractor.py
│   │   └── postprocessing.py
│   └── ocr_pipeline.py      # Complete pipeline
├── api/
│   ├── main.py             # FastAPI service
│   └── models.py           # Pydantic models
├── web/
│   └── app.py              # Streamlit interface
├── models/                 # Trained models
├── data/
│   ├── samples/           # Sample documents
│   └── uploads/           # Uploaded files
├── config/
│   └── config.yaml        # Configuration
├── tests/                 # Unit tests
├── scripts/               # Utility scripts
├── notebooks/             # Jupyter notebooks
├── requirements.txt
├── .env.example
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## 🚀 Cài đặt

### Requirements
- Python 3.8+
- CUDA (optional, for GPU acceleration)

### 1. Clone repository
```bash
git clone <repository-url>
cd project19_ocr_banking
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment
```bash
cp .env.example .env
# Edit .env with your settings
```

## 💻 Sử dụng

### 1. Command Line Interface

```python
from src.ocr_pipeline import OCRPipeline

# Initialize pipeline
pipeline = OCRPipeline(
    detector_type='craft',
    recognizer_type='vietocr',
    device='cpu'
)

# Process image
result = pipeline.process_image('path/to/document.jpg')

# Display results
print(f"Document Type: {result['document_type']}")
print(f"Confidence: {result['confidence']:.2f}")
print(f"Extracted Fields: {result['extracted_fields']}")
```

### 2. FastAPI Service

```bash
# Run API server
cd api
python main.py

# Or with uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

API Endpoints:
- `GET /`: Service information
- `GET /health`: Health check
- `POST /api/ocr`: OCR single document
- `POST /api/ocr/batch`: Batch OCR
- `GET /api/document-types`: Supported document types
- `GET /api/stats`: Service statistics

Example request:
```bash
curl -X POST "http://localhost:8000/api/ocr" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.jpg"
```

### 3. Streamlit Web Interface

```bash
# Run web app
cd web
streamlit run app.py
```

Features:
- Upload and process single document
- Batch processing (up to 10 files)
- View extracted information
- Export results (JSON, TXT)
- Interactive visualization

### 4. Docker Deployment

```bash
# Build image
docker-compose build

# Run services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## 📊 Examples

### CCCD (Căn cước công dân)

Input: Image of CCCD card

Output:
```json
{
  "document_type": "cccd",
  "confidence": 0.92,
  "extracted_fields": {
    "id_number": "001234567890",
    "full_name": "NGUYỄN VĂN A",
    "date_of_birth": "15/03/1990",
    "gender": "Nam",
    "nationality": "Việt Nam",
    "place_of_residence": "..."
  }
}
```

### Bank Statement

Input: Image of bank statement

Output:
```json
{
  "document_type": "bank_statement",
  "confidence": 0.85,
  "extracted_fields": {
    "account_number": "1234567890123456",
    "account_holder": "NGUYỄN VĂN A",
    "opening_balance": "10,000,000",
    "closing_balance": "12,500,000",
    "transactions": [...]
  }
}
```

## ⚙️ Configuration

Edit `config/config.yaml`:

```yaml
# Model settings
models:
  detection:
    type: "craft"
    text_threshold: 0.7
    link_threshold: 0.4
  
  recognition:
    type: "vietocr"
    config: "vgg_transformer"
    device: "cpu"

# Document types
document_types:
  cccd:
    fields:
      - id_number
      - full_name
      - date_of_birth
      # ...

# Extraction patterns
extraction:
  patterns:
    cccd: '\b\d{12}\b'
    date: '\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b'
    # ...
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_detection.py

# Run with coverage
pytest --cov=src tests/
```

## 📈 Performance

- **Detection**: ~1-2s per image (CPU), ~0.3-0.5s (GPU)
- **Recognition**: ~0.5-1s per text region (CPU), ~0.1-0.2s (GPU)
- **Total Pipeline**: ~3-5s per document (CPU), ~1-2s (GPU)

## 🔧 Troubleshooting

### Common Issues

1. **ImportError: craft-text-detector not found**
   ```bash
   pip install craft-text-detector
   ```

2. **CUDA out of memory**
   - Reduce batch size
   - Use CPU mode: `device='cpu'`

3. **Low accuracy**
   - Use higher resolution images
   - Ensure good lighting
   - Preprocess images (denoise, contrast)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit pull request

## 📄 License

[Add license information]

## 👥 Authors

Development Team - MB Bank eKYC Project

## 📞 Contact

For questions or support, contact: [Add contact info]

## 🙏 Acknowledgments

- VietOCR: https://github.com/pbcquoc/vietocr
- CRAFT: https://github.com/clovaai/CRAFT-pytorch
- PaddleOCR: https://github.com/PaddlePaddle/PaddleOCR
- EasyOCR: https://github.com/JaidedAI/EasyOCR
