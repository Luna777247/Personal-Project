# Quick Start Guide - Fraud Detection AI

## 🚀 Hướng dẫn sử dụng nhanh

### Bước 1: Setup môi trường

```bash
# Di chuyển vào thư mục dự án
cd project18_fraud_detection_ai

# Tạo virtual environment
python -m venv venv

# Kích hoạt virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Bước 2: Generate dữ liệu và Train model

```bash
# Tạo dữ liệu synthetic
python src/data_generator.py

# Train tất cả các models
python scripts/train.py --model all

# Hoặc train từng model riêng:
python scripts/train.py --model isolation_forest
python scripts/train.py --model autoencoder
python scripts/train.py --model lstm
```

### Bước 3: Chạy API và Dashboard

#### Option 1: Chạy local

```bash
# Terminal 1: Chạy API
python api/main.py

# Terminal 2: Chạy Dashboard
streamlit run dashboard/app.py

# Terminal 3: Chạy MLflow (optional)
mlflow ui --backend-store-uri mlflow/
```

#### Option 2: Chạy với Docker

```bash
# Development mode (với auto-reload)
docker-compose -f docker-compose.dev.yml up -d

# Production mode
docker-compose up -d

# Xem logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Bước 4: Truy cập ứng dụng

- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Dashboard**: http://localhost:8501
- **MLflow**: http://localhost:5000
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000

### Bước 5: Test API

```bash
# Test với script có sẵn
python api/test_api.py

# Hoặc dùng curl
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "TXN_TEST_001",
    "customer_id": 1234,
    "merchant_id": 567,
    "amount": 1250.50,
    "timestamp": "2024-12-11T14:30:00",
    "merchant_category": "electronics",
    "card_type": "credit",
    "transaction_type": "online",
    "country_code": "US"
  }'
```

## 📊 Các lệnh thường dùng

### Training

```bash
# Train với custom config
python scripts/train.py --config config/config.yaml

# Force generate new data
python scripts/train.py --generate-data

# Train specific model
python scripts/train.py --model isolation_forest
```

### Evaluation

```bash
# Evaluate model
python scripts/evaluate.py \
    --model-path models/isolation_forest_20241211_120000.pkl \
    --data-path data/raw/transactions.csv
```

### Docker

```bash
# Build images
docker-compose build

# Start services
docker-compose up -d

# View logs
docker-compose logs -f api
docker-compose logs -f dashboard

# Restart service
docker-compose restart api

# Stop and remove
docker-compose down -v
```

### MLflow

```bash
# Start MLflow UI
mlflow ui --backend-store-uri mlflow/

# List experiments
mlflow experiments list

# Search runs
mlflow runs list --experiment-id 0
```

## 🔍 Troubleshooting

### Issue: Model không load được

```bash
# Kiểm tra model path
ls -la models/

# Re-train model
python scripts/train.py --model isolation_forest
```

### Issue: API không start

```bash
# Kiểm tra port
netstat -ano | findstr :8000  # Windows
lsof -i :8000  # Linux/Mac

# Kill process nếu cần
# Windows: taskkill /PID <PID> /F
# Linux/Mac: kill -9 <PID>
```

### Issue: Docker container error

```bash
# Xem logs
docker-compose logs api

# Rebuild container
docker-compose build --no-cache api
docker-compose up -d api
```

## 📝 Tips

1. **Performance**: Isolation Forest nhanh nhất cho real-time prediction
2. **Accuracy**: AutoEncoder cho accuracy cao hơn
3. **Time Series**: LSTM AutoEncoder cho sequential patterns
4. **Monitoring**: Check Prometheus metrics tại /metrics endpoint
5. **Debugging**: Set LOG_LEVEL=DEBUG trong .env file

## 🎯 Next Steps

1. Customize feature engineering trong `src/feature_engineering.py`
2. Tune hyperparameters trong `config/config.yaml`
3. Add custom validation logic trong `api/main.py`
4. Enhance dashboard trong `dashboard/app.py`
5. Setup CI/CD pipeline với GitHub Actions

## 📞 Support

Nếu gặp vấn đề, check:
1. README.md chi tiết
2. API documentation tại /docs
3. Logs trong thư mục logs/
4. MLflow UI để track experiments
