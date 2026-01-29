# Quick Start Guide - OCR Banking System

## 🚀 Hướng dẫn khởi động nhanh

### 1. Cài đặt môi trường

```bash
# Di chuyển vào thư mục dự án
cd project19_ocr_banking

# Tạo virtual environment
python -m venv venv

# Kích hoạt virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Cài đặt dependencies
pip install -r requirements.txt
```

### 2. Cấu hình

```bash
# Copy file cấu hình môi trường
copy .env.example .env

# Chỉnh sửa .env theo nhu cầu
# DETECTOR_TYPE=craft
# RECOGNIZER_TYPE=vietocr
# DEVICE=cpu
```

### 3. Chạy ứng dụng

#### Option 1: Web Interface (Streamlit)
```bash
streamlit run web/app.py
```
Truy cập: http://localhost:8501

#### Option 2: API Service (FastAPI)
```bash
cd api
python main.py
```
Truy cập:
- API: http://localhost:8000
- Docs: http://localhost:8000/docs

#### Option 3: Docker
```bash
docker-compose up -d
```
Services:
- API: http://localhost:8000
- Web: http://localhost:8501

### 4. Sử dụng API

```bash
# Health check
curl http://localhost:8000/health

# OCR document
curl -X POST "http://localhost:8000/api/ocr" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/document.jpg"

# Document types
curl http://localhost:8000/api/document-types
```

### 5. Sử dụng Python Code

```python
from src.ocr_pipeline import OCRPipeline

# Khởi tạo pipeline
pipeline = OCRPipeline(
    detector_type='craft',
    recognizer_type='vietocr',
    device='cpu'
)

# Xử lý ảnh
result = pipeline.process_image('path/to/document.jpg')

print(f"Loại tài liệu: {result['document_type']}")
print(f"Độ tin cậy: {result['confidence']:.2%}")
print(f"Thông tin trích xuất: {result['extracted_fields']}")
```

### 6. Tiện ích hỗ trợ

#### Tiền xử lý ảnh
```bash
# Xử lý một ảnh
python scripts/preprocess_images.py input.jpg -o output.jpg

# Xử lý hàng loạt
python scripts/preprocess_images.py data/samples/ -o data/preprocessed/ -b
```

#### Đánh giá độ chính xác
```bash
python scripts/evaluate_ocr.py data/ground_truth_example.json -o results.json
```

### 7. Chạy tests

```bash
# Chạy tất cả tests
pytest tests/ -v

# Chạy test cụ thể
pytest tests/test_detection.py -v

# Chạy với coverage
pytest --cov=src tests/
```

## 📚 Các loại tài liệu hỗ trợ

1. **CCCD** (Căn cước công dân)
   - Số CCCD: 12 số
   - Họ tên, ngày sinh, giới tính
   - Quốc tịch, nơi thường trú

2. **CMND** (Chứng minh nhân dân)
   - Số CMND: 9 số
   - Thông tin cá nhân

3. **Sao kê ngân hàng**
   - Số tài khoản
   - Lịch sử giao dịch
   - Số dư

4. **Hợp đồng vay**
   - Thông tin người vay
   - Số tiền vay, lãi suất
   - Thời hạn

## 🛠️ Troubleshooting

### Lỗi thường gặp:

1. **ImportError: craft-text-detector not found**
   ```bash
   pip install craft-text-detector
   ```

2. **CUDA out of memory**
   - Chuyển sang CPU mode: `DEVICE=cpu`
   - Hoặc giảm batch size

3. **Độ chính xác thấp**
   - Sử dụng ảnh độ phân giải cao
   - Tiền xử lý ảnh trước
   - Đảm bảo ánh sáng tốt

## 📞 Liên hệ

Để được hỗ trợ, vui lòng liên hệ team phát triển.

## 🔗 Tài liệu tham khảo

- [VietOCR Documentation](https://github.com/pbcquoc/vietocr)
- [CRAFT Paper](https://arxiv.org/abs/1904.01941)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
