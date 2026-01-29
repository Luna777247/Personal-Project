# Keyword-based Disaster Information Extraction

## 📋 Tổng Quan

Hệ thống trích xuất thông tin thiên tai đơn giản nhất dựa trên **từ khóa (Keyword-based Extraction)**. Phương pháp này tìm và trích xuất các câu chứa từ khóa liên quan đến thiên tai từ nội dung bài báo.

## 🎯 Nguyên Lý Hoạt Động

1. **Danh sách từ khóa**: Định nghĩa trước các từ khóa thiên tai
2. **Tách câu**: Chia văn bản thành các câu riêng biệt
3. **Tìm kiếm**: Scan từng câu để tìm từ khóa
4. **Trích xuất**: Lấy các câu chứa từ khóa cùng context xung quanh

## 📁 Cấu Trúc Thư Mục

```
keyword_extraction/
├── config/
│   └── keywords.py          # Danh sách từ khóa thiên tai
├── scripts/
│   ├── keyword_extractor.py # Class chính xử lý extraction
│   ├── demo_simple.py       # Demo đơn giản
│   └── demo_full.py         # Demo đầy đủ với sample data
├── data/                    # Output từ demo
│   ├── keyword_extraction_demo.csv
│   └── keyword_extraction_demo.json
├── docs/
│   └── README.md            # Tài liệu này
├── run.py                   # Script tiện ích để chạy demo
├── SUMMARY.md               # Tóm tắt thư mục
├── __init__.py              # Package marker
└── requirements.txt         # Dependencies
```

## 🚀 Cách Chạy Nhanh

### Chạy Demo Đầy Đủ
```bash
cd keyword_extraction
python run.py
```

### Chạy Demo Đơn Giản
```bash
cd keyword_extraction
python run.py --demo simple
```

## 📊 Kết Quả Demo Gần Đây

**Thống kê từ lần chạy cuối:**
- ✅ **2 bài báo** được xử lý thành công
- ✅ **8 câu** chứa từ khóa được trích xuất
- ✅ **8 từ khóa unique** được phát hiện
- ✅ **2 loại thiên tai** được phân loại: `storm`, `geological`
- ✅ **2 file output**: CSV và JSON được tạo

## 🔧 Tính Năng Chính

### ✅ Điểm Mạnh
- **Đơn giản**: Chỉ cần Python + pandas
- **Nhanh**: Xử lý tức thời, không cần model
- **Dễ hiểu**: Logic rõ ràng, dễ debug
- **Tùy chỉnh**: Thêm/bớt từ khóa dễ dàng
- **Không phụ thuộc**: Không cần GPU hay internet

### ⚠️ Hạn Chế
- **Độ chính xác**: Chỉ dựa trên từ khóa
- **False positive**: Có thể match nhầm
- **Cần maintenance**: Cập nhật từ khóa thủ công
- **Không học**: Không cải thiện theo thời gian

## 🎯 So Sánh Với Hệ Thống AI

| Phương Pháp | Độ Phức Tạp | Độ Chính Xác | Tốc Độ | Dependencies |
|-------------|-------------|--------------|--------|--------------|
| **Keyword-based** (này) | Thấp | Trung bình | Cao | Chỉ pandas |
| **AI-powered** (hệ thống chính) | Cao | Cao | Trung bình | Spacy, Transformers |

## 💡 Use Cases Phù Hợp

- **Prototype nhanh**: Test concept trước khi build hệ thống phức tạp
- **Domain hạn chế**: Khi có danh sách từ khóa rõ ràng
- **Resource limited**: Khi không có GPU hoặc data lớn
- **Explainability**: Khi cần logic dễ hiểu và traceable

## 📈 Kết Luận

Thư mục này cung cấp **baseline đơn giản** để so sánh với các phương pháp advanced hơn trong dự án chính. Đây là điểm khởi đầu lý tưởng cho việc hiểu và phát triển các hệ thống trích xuất thông tin thiên tai.

**Trạng thái:** ✅ Hoạt động tốt cho use cases đơn giản
**Khuyến nghị:** Sử dụng làm baseline để so sánh với AI-powered system