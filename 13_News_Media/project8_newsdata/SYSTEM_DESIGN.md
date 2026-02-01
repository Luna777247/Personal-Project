# Thiết Kế Hệ Thống - Vietnamese Disaster Information Extraction

## 1. Tổng Quan Hệ Thống

### 1.1. Mục Tiêu
Hệ thống trích xuất thông tin thiên tai từ các nguồn tin tức tiếng Việt, chuyển đổi dữ liệu phi cấu trúc thành thông tin có cấu trúc phục vụ phân tích và cảnh báo.

### 1.2. Kiến Trúc Tổng Thể
```
┌─────────────────────────────────────────────────────────────────┐
│                      Data Sources Layer                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  News Sites  │  │  RSS Feeds   │  │  Social Media│          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Data Collection Layer                        │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │         Web Crawler (scripts/crawl_news_from_web.py)     │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Processing Pipeline Layer                    │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐   │
│  │    Keyword     │  │      NER       │  │    Pattern     │   │
│  │   Extraction   │  │   Extraction   │  │   Extraction   │   │
│  └────────────────┘  └────────────────┘  └────────────────┘   │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐   │
│  │      LLM       │  │   Relation     │  │      RAG       │   │
│  │   Extraction   │  │   Extraction   │  │   Extraction   │   │
│  └────────────────┘  └────────────────┘  └────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Storage & Output Layer                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  JSON Files  │  │  CSV Files   │  │   Database   │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

## 2. Chi Tiết Các Module

### 2.1. Keyword Extraction Module
**Vị trí:** `keyword_extraction/`

**Chức năng:**
- Phát hiện từ khóa liên quan đến thiên tai
- Trích xuất câu văn bản chứa thông tin thiên tai
- Xử lý toàn bộ dataset JSON

**Công nghệ:**
- Rule-based pattern matching
- Vietnamese text processing
- Frequency analysis

**Input:** `data/disaster_data_multisource_*.json`

**Output:** 
- `data/keyword_extraction_demo.csv`
- `data/keyword_extraction_demo.json`

**Cấu hình:** `keyword_extraction/config/`

### 2.2. NER Entity Extraction Module
**Vị trí:** `ner_entity_extraction/`

**Chức năng:**
- Nhận diện thực thể (Location, Time, Casualties, etc.)
- Hỗ trợ nhiều mô hình (PhoBERT, lightweight models)
- Tối ưu cho tiếng Việt

**Công nghệ:**
- Transformer-based models (PhoBERT)
- Spacy pipelines
- Custom NER models

**Modes:**
- `simple_demo.py`: Demo đơn giản
- `lightweight_demo.py`: Mô hình nhẹ
- `run.py`: Pipeline đầy đủ

**Output:** Entities với các nhãn:
- LOCATION (Địa điểm)
- TIME (Thời gian)
- CASUALTIES (Thiệt hại)
- DISASTER_TYPE (Loại thiên tai)

### 2.3. Relation Extraction Module
**Vị trí:** `relation_extraction/`

**Chức năng:**
- Trích xuất mối quan hệ giữa các thực thể
- Hỗ trợ cả rule-based và LLM-based
- So sánh hiệu suất các phương pháp

**Công nghệ:**
- Rule-based extraction
- LLM (GPT-5.1-codex-max)
- Hybrid approach

**Quan hệ hỗ trợ:**
- `occurred_at`: Thiên tai xảy ra tại địa điểm
- `occurred_on`: Thiên tai xảy ra vào thời gian
- `caused`: Thiên tai gây ra thiệt hại
- `affected`: Thiên tai ảnh hưởng đến khu vực

**Output:**
- `data/re_results_rule.json`
- `data/re_results_llm.json`
- `data/re_comparison_summary.json`
- `data/relation_extraction_results.csv`

### 2.4. LLM Extraction Module
**Vị trí:** `llm_extraction/`

**Chức năng:**
- Trích xuất thông tin bằng LLM
- Prompt engineering cho tiếng Việt
- Xử lý các trường hợp phức tạp

**Công nghệ:**
- OpenAI API (GPT-5.1-codex-max)
- Custom prompts
- Response parsing

**Cấu hình:**
- Model: `gpt-5.1-codex-max`
- API Key: `OPENAI_API_KEY` environment variable
- Config: `llm_extraction/config/llm_config.py`

### 2.5. Pattern Extraction Module
**Vị trí:** `pattern_extraction/`

**Chức năng:**
- Phát hiện patterns trong văn bản tin tức
- Trích xuất theo mẫu có cấu trúc
- Rule-based với regex patterns

**Công nghệ:**
- Regex patterns
- Vietnamese linguistic rules
- Template matching

### 2.6. RAG Extraction Module
**Vị trí:** `rag_extraction/`

**Chức năng:**
- Retrieval-Augmented Generation
- Tìm kiếm ngữ cảnh từ knowledge base
- Trích xuất với context awareness

**Công nghệ:**
- Vector database
- Embedding models
- LLM integration

**Components:**
- Docker support (`docker-compose.yml`, `Dockerfile`)
- Knowledge base management
- Query processing

**Scripts:**
- `run_rag.py`: Main pipeline
- `test_system.py`: Testing utilities

### 2.7. Fine-tuning Module
**Vị trí:** `fine_tuning/`

**Chức năng:**
- Fine-tune models cho disaster extraction
- Training pipeline cho NER, RE, Event Extraction
- Model evaluation và visualization

**Components:**
- `train_ner.py`: Train NER models
- `train_relation_extraction.py`: Train RE models
- `train_event_extraction.py`: Train event models
- `evaluate_models.py`: Đánh giá hiệu suất
- `visualize_results.py`: Visualization
- `inference.py`: Inference pipeline
- `annotate_data.py`: Data annotation tools

**Notebooks:**
- `disaster_extraction_demo.ipynb`: Interactive demo

## 3. Data Flow

### 3.1. Collection Phase
```
Web Sources → Crawler → Raw JSON/CSV
```

### 3.2. Processing Phase
```
Raw Data → Keyword Extraction → Filtered Data
         → NER Extraction → Entities
         → Relation Extraction → Relations
         → LLM Extraction → Enhanced Information
         → Pattern Extraction → Structured Data
         → RAG Extraction → Contextual Information
```

### 3.3. Output Phase
```
Processed Data → Multiple Formats
              → JSON (structured)
              → CSV (tabular)
              → Database (queryable)
```

## 4. Cấu Trúc Dữ Liệu

### 4.1. Input Format (JSON)
```json
{
  "articles": [
    {
      "id": "article_001",
      "title": "Bão số 9 gây thiệt hại nặng tại miền Trung",
      "content": "Full article content...",
      "source": "vnexpress.net",
      "published_date": "2024-12-15",
      "url": "https://..."
    }
  ]
}
```

### 4.2. Entity Output Format
```json
{
  "article_id": "article_001",
  "entities": [
    {
      "text": "Bão số 9",
      "type": "DISASTER_TYPE",
      "start": 0,
      "end": 8,
      "confidence": 0.95
    },
    {
      "text": "miền Trung",
      "type": "LOCATION",
      "start": 35,
      "end": 45,
      "confidence": 0.92
    }
  ]
}
```

### 4.3. Relation Output Format
```json
{
  "article_id": "article_001",
  "relations": [
    {
      "subject": "Bão số 9",
      "relation": "occurred_at",
      "object": "miền Trung",
      "confidence": 0.88,
      "method": "llm"
    }
  ]
}
```

### 4.4. CSV Output Format
| article_id | disaster_type | location | time | casualties | severity | source |
|------------|---------------|----------|------|------------|----------|--------|
| article_001 | Bão | miền Trung | 2024-12-15 | 50 người | High | vnexpress |

## 5. Configuration Management

### 5.1. Global Config (`config/default.yaml`)
- Database settings
- Extractor parameters
- Output formats
- Logging configuration

### 5.2. Module-specific Configs
- `keyword_extraction/config/`: Keyword patterns, thresholds
- `ner_entity_extraction/config/`: Model configs, entity types
- `relation_extraction/config/re_config.py`: RE models, rules
- `llm_extraction/config/llm_config.py`: LLM settings, API keys
- `rag_extraction/config/`: RAG parameters, vector DB settings

### 5.3. Knowledge Bases
- `config/knowledge_base.json`: Domain knowledge
- `config/vietnam_locations.json`: Vietnamese locations
- `config/extractor.yaml`: Extractor rules

## 6. API & Integration

### 6.1. External APIs
- **OpenAI API**: LLM extraction
  - Model: `gpt-5.1-codex-max`
  - Requires: `OPENAI_API_KEY`

### 6.2. Internal APIs
- Module interfaces để kết nối các components
- Shared utilities và helpers

## 7. Deployment & Operations

### 7.1. Environment Setup
```bash
# Create virtual environment
python -m venv .venv
& .\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Install module-specific deps
pip install -r keyword_extraction/requirements.txt
pip install -r ner_entity_extraction/requirements.txt
pip install -r relation_extraction/requirements.txt
pip install -r llm_extraction/requirements.txt
pip install -r rag_extraction/requirements.txt
```

### 7.2. Environment Variables
```powershell
# Required for LLM modules
$env:OPENAI_API_KEY = "your-api-key"

# Optional
$env:DATABASE_URL = "sqlite:///data/disaster_data.db"
$env:LOG_LEVEL = "INFO"
```

### 7.3. Docker Support (RAG Module)
```bash
cd rag_extraction
docker-compose up -d
```

### 7.4. Running Modules

**Keyword Extraction:**
```bash
cd keyword_extraction
python run.py
```

**NER Extraction:**
```bash
cd ner_entity_extraction
python run.py
```

**Relation Extraction:**
```bash
cd relation_extraction
python run.py
```

**LLM Extraction:**
```bash
cd llm_extraction
python run.py
```

**RAG Extraction:**
```bash
cd rag_extraction
python run_rag.py
```

**Full Pipeline:**
```bash
cd fine_tuning/scripts
python run_pipeline.py
```

## 8. Performance & Optimization

### 8.1. Processing Optimization
- **Batch Processing**: Process articles in batches (default: 100)
- **Parallel Processing**: Multi-threading with configurable workers (default: 4)
- **Caching**: Cache intermediate results in `relation_extraction/cache/`

### 8.2. Model Optimization
- **Lightweight Models**: Use `lightweight_demo.py` for quick testing
- **Model Selection**: Choose appropriate model based on accuracy/speed trade-off
- **Fine-tuning**: Train custom models in `fine_tuning/` for better performance

### 8.3. Memory Management
- Stream processing for large datasets
- Clear cache periodically
- Compress outputs when possible

## 9. Monitoring & Logging

### 9.1. Logging Configuration
```yaml
logging:
  level: "INFO"
  file: "logs/disaster_extractor.log"
  max_size: 10485760  # 10MB
  backup_count: 5
```

### 9.2. Metrics Tracking
- `data/dataset_statistics.json`: Dataset statistics
- Processing time per module
- Extraction accuracy metrics
- API usage tracking

## 10. Quality Assurance

### 10.1. Testing
- Unit tests với pytest
- Integration tests
- End-to-end pipeline tests

### 10.2. Evaluation
- `fine_tuning/scripts/evaluate_models.py`: Model evaluation
- Comparison between methods
- Performance benchmarks

### 10.3. Validation
- Data quality checks
- Entity validation
- Relation validation

## 11. Security & Privacy

### 11.1. API Key Management
- Store keys in environment variables
- Never commit keys to version control
- Use `.env` files (not tracked)

### 11.2. Data Security
- Local processing (no external data sharing except LLM APIs)
- Secure database connections
- Access control for sensitive outputs

## 12. Future Enhancements

### 12.1. Planned Features
- [ ] Real-time processing pipeline
- [ ] Web dashboard for visualization
- [ ] REST API for external integrations
- [ ] Multi-language support
- [ ] Advanced analytics and insights

### 12.2. Model Improvements
- [ ] Better Vietnamese NER models
- [ ] Enhanced relation extraction
- [ ] Event extraction with temporal reasoning
- [ ] Causal analysis

### 12.3. Scalability
- [ ] Distributed processing
- [ ] Cloud deployment (Azure/AWS)
- [ ] Kubernetes orchestration
- [ ] Message queue integration

## 13. Documentation & Resources

### 13.1. Module Documentation
- Each module has its own `README.md` and `SUMMARY.md`
- `docs/` folders contain additional documentation
- Notebooks provide interactive examples

### 13.2. Configuration Docs
- YAML/JSON schema documentation
- Example configurations
- Best practices guide

### 13.3. Troubleshooting Guide
See [README.md](README.md) for common issues and solutions

## 14. Development Workflow

### 14.1. Development Process
1. **Data Collection**: Run crawler scripts
2. **Data Exploration**: Use notebooks in `fine_tuning/notebooks/`
3. **Module Development**: Develop in respective module folders
4. **Testing**: Run tests and evaluate
5. **Integration**: Combine modules in pipeline
6. **Deployment**: Export and deploy

### 14.2. Code Quality
- **Formatting**: Black, isort
- **Linting**: Flake8
- **Type Checking**: MyPy (optional)
- **Testing**: Pytest with coverage

### 14.3. Version Control
- Git for version control
- Semantic versioning (current: 1.0.0)
- Feature branches for development
- Main branch for stable releases

## 15. Dependencies Summary

### 15.1. Core Dependencies
- Python >= 3.8 (tested with 3.13)
- pandas, numpy: Data processing
- scikit-learn: ML utilities
- requests, beautifulsoup4: Web scraping

### 15.2. NLP Dependencies
- spacy: NLP processing
- transformers: Pre-trained models
- PhoBERT: Vietnamese BERT

### 15.3. LLM Dependencies
- openai: OpenAI API
- langchain: LLM orchestration (RAG)
- chromadb: Vector database (RAG)

### 15.4. Visualization Dependencies
- matplotlib, seaborn, plotly: Visualization
- jupyter, notebook: Interactive notebooks

## 16. Architecture Patterns

### 16.1. Design Patterns
- **Pipeline Pattern**: Sequential processing stages
- **Strategy Pattern**: Multiple extraction strategies
- **Factory Pattern**: Model and extractor creation
- **Observer Pattern**: Event-driven processing

### 16.2. Data Patterns
- **ETL Pattern**: Extract-Transform-Load
- **Batch Processing**: Process data in batches
- **Stream Processing**: Real-time processing (future)

### 16.3. Integration Patterns
- **Module Independence**: Each module can run standalone
- **Shared Storage**: Common data directory
- **Configuration Injection**: Config-driven behavior

## 17. Conclusion

Hệ thống Vietnamese Disaster Information Extraction là một pipeline xử lý dữ liệu phức tạp với nhiều phương pháp trích xuất thông tin bổ trợ lẫn nhau. Kiến trúc modular cho phép dễ dàng mở rộng, bảo trì và tối ưu hóa từng component độc lập.

**Key Strengths:**
- Modular và flexible
- Multiple extraction methods
- Vietnamese-optimized
- Production-ready structure
- Comprehensive documentation

**Contact & Support:**
- Xem các file SUMMARY.md trong từng module
- Check README.md cho quick start
- Explore notebooks cho examples
