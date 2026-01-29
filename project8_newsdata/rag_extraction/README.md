# RAG-Based Disaster Information Extraction System

Hệ thống trích xuất thông tin thảm họa sử dụng Retrieval-Augmented Generation (RAG) với cơ sở dữ liệu vector, được tối ưu hóa cho tiếng Việt.

## Tính năng chính

- **Trích xuất thông tin thảm họa**: Tự động trích xuất thông tin về loại thảm họa, vị trí, thời gian, mức độ nghiêm trọng, thiệt hại, số người chết/thương, tổ chức liên quan
- **Hỗ trợ đa nguồn dữ liệu**: JSON, CSV, text files
- **Cơ sở dữ liệu vector**: Chroma (mặc định), Qdrant, Milvus
- **Mô hình embedding**: SentenceTransformers (đa ngôn ngữ), OpenAI, BGE
- **Tối ưu hóa tiếng Việt**: Chunking và prompts được tối ưu hóa cho tiếng Việt
- **Xử lý batch**: Hỗ trợ xử lý nhiều truy vấn cùng lúc
- **Giao diện CLI**: Dễ sử dụng qua command line
- **Metrics và monitoring**: Theo dõi hiệu suất và chi phí

## Cài đặt

### Yêu cầu hệ thống
- Python 3.8+
- 4GB RAM (tối thiểu)
- 8GB RAM (khuyến nghị cho datasets lớn)

### Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### Cài đặt vector databases (tùy chọn)

#### Chroma (mặc định - không cần cài đặt thêm)

#### Qdrant
```bash
# Sử dụng Docker
docker run -p 6333:6333 qdrant/qdrant

# Hoặc cài đặt từ source
pip install qdrant-client
```

#### Milvus
```bash
# Sử dụng Docker
docker run -p 19530:19530 milvusdb/milvus:latest

# Hoặc cài đặt từ source
pip install pymilvus
```

## Sử dụng

### Chuẩn bị dữ liệu

Tạo file JSON với cấu trúc:
```json
[
  {
    "id": "doc_1",
    "content": "Nội dung bài báo về thảm họa...",
    "metadata": {
      "source": "vnexpress",
      "date": "2024-01-15",
      "title": "Bão số 12 gây thiệt hại nặng",
      "url": "https://example.com/article"
    }
  }
]
```

Hoặc file CSV với các cột: `content`, `source`, `date`, `title`, `url`

### Thêm dữ liệu vào hệ thống

```bash
# Thêm dữ liệu từ file JSON
python run_rag.py add --input data/disaster_news.json

# Thêm dữ liệu từ file CSV
python run_rag.py add --input data/disaster_news.csv

# Xóa dữ liệu cũ và thêm mới
python run_rag.py add --input data/disaster_news.json --clear
```

### Tìm kiếm tài liệu

```bash
# Tìm kiếm cơ bản
python run_rag.py search --query "bão tại Quảng Nam"

# Tìm kiếm với số lượng kết quả tùy chỉnh
python run_rag.py search --query "lũ lụt miền Trung" --top-k 10

# Lưu kết quả ra file
python run_rag.py search --query "thảm họa" --output search_results.json
```

### Trích xuất thông tin thảm họa

```bash
# Trích xuất từ một truy vấn
python run_rag.py extract --query "Thiệt hại do bão số 12 tại Quảng Nam"

# Trích xuất từ nhiều truy vấn trong file
python run_rag.py extract --query-file queries.txt --output extraction_results.json

# Sử dụng model cụ thể
python run_rag.py extract --query "..." --model gpt-4 --output results.json
```

### Giám sát hệ thống

```bash
# Xem metrics
python run_rag.py metrics
```

### Quản lý cơ sở dữ liệu

```bash
# Xóa toàn bộ dữ liệu
python run_rag.py clear
```

## Cấu hình nâng cao

### Thay đổi vector database

```bash
# Sử dụng Qdrant
python run_rag.py add --input data.json --vector-db qdrant

# Sử dụng Milvus
python run_rag.py extract --query "..." --vector-db milvus --embedding openai
```

### Thay đổi embedding model

```bash
# Sử dụng OpenAI embeddings
python run_rag.py search --query "..." --embedding openai

# Sử dụng BGE model
python run_rag.py extract --query "..." --embedding bge
```

## Định dạng đầu ra

### JSON Output
```json
{
  "metadata": {
    "total_results": 5,
    "timestamp": "2024-01-15T10:30:00",
    "format_version": "1.0",
    "system": "RAG-based extraction"
  },
  "results": [
    {
      "id": 1,
      "query": "thiệt hại bão số 12",
      "model": "gpt-4",
      "processing_time": 2.34,
      "cost_estimate": 0.012,
      "confidence_score": 0.85,
      "extracted_info": {
        "type": "Bão",
        "location": "Quảng Nam",
        "time": "15/11/2023",
        "severity": "Nặng",
        "damage": "Hàng trăm ngôi nhà bị tốc mái",
        "deaths": 3,
        "injured": 12,
        "organizations": ["UBND tỉnh Quảng Nam", "Bộ Quốc phòng"]
      }
    }
  ]
}
```

### CSV Output
```csv
id,query,model,processing_time,cost_estimate,confidence_score,type,location,time,severity,damage,deaths,injured,missing,organizations,forecast
1,"thiệt hại bão số 12",gpt-4,2.34,0.012,0.85,Bão,Quảng Nam,15/11/2023,Nặng,"Hàng trăm ngôi nhà bị tốc mái",3,12,0,"UBND tỉnh Quảng Nam; Bộ Quốc phòng",
```

## Demo và Testing

Chạy demo đầy đủ:
```bash
python scripts/demo_rag_extraction.py
```

Demo sẽ:
- Thêm dữ liệu mẫu vào vector database
- Thực hiện tìm kiếm
- Trích xuất thông tin thảm họa
- Hiển thị metrics và kết quả

## Tối ưu hóa hiệu suất

### Cho datasets lớn
- Sử dụng Qdrant hoặc Milvus thay vì Chroma
- Tăng chunk_size trong config để giảm số lượng chunks
- Sử dụng embedding models nhẹ hơn như `sentence-transformers`

### Cho tốc độ nhanh
- Giảm top_k trong tìm kiếm
- Sử dụng caching (tự động được bật)
- Chọn models nhỏ hơn cho extraction

### Cho độ chính xác cao
- Sử dụng GPT-4 hoặc Claude-3
- Tăng context window
- Sử dụng semantic chunking

## Troubleshooting

### Lỗi kết nối vector database
- Đảm bảo Qdrant/Milvus đang chạy
- Kiểm tra địa chỉ và port trong config

### Lỗi embedding
- Kiểm tra API key cho OpenAI
- Đảm bảo đủ RAM cho large models

### Lỗi trích xuất
- Kiểm tra API key cho LLM providers
- Xem logs để debug

### Hiệu suất chậm
- Giảm batch size
- Sử dụng models nhỏ hơn
- Tăng cache size

## API Reference

### RAGDisasterExtractor class

#### Methods
- `add_documents(documents)`: Thêm documents vào vector DB
- `search_documents(query, top_k=5)`: Tìm kiếm documents
- `extract_disaster_info(query, model=None)`: Trích xuất thông tin thảm họa
- `clear_database()`: Xóa toàn bộ dữ liệu
- `get_metrics()`: Lấy metrics hệ thống

#### Configuration
Xem `config/rag_config.py` để biết các tùy chọn cấu hình.

## Contributing

1. Fork repository
2. Tạo feature branch
3. Commit changes
4. Push và tạo Pull Request

## License

MIT License - Xem LICENSE file để biết thêm chi tiết.

## Liên hệ

- Email: your-email@example.com
- GitHub Issues: https://github.com/your-repo/issues

---

*Hệ thống được phát triển để hỗ trợ công tác phòng chống thảm họa và ứng phó khẩn cấp tại Việt Nam.*