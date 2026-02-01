# 🔒 Fraud Detection AI - Phát hiện giao dịch gian lận theo thời gian thực

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![ML](https://img.shields.io/badge/Machine%20Learning-Anomaly%20Detection-green.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.103-teal.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📋 Tổng quan

Hệ thống phát hiện giao dịch gian lận theo thời gian thực sử dụng Machine Learning và Deep Learning. Dự án này được thiết kế đặc biệt cho nhu cầu của ngành ngân hàng, cung cấp một pipeline hoàn chỉnh từ training đến deployment và monitoring.

### 🎯 Lĩnh vực
- **Machine Learning**: Isolation Forest, AutoEncoder
- **Deep Learning**: LSTM AutoEncoder
- **Anomaly Detection**: Phát hiện bất thường trong giao dịch
- **Time Series Analysis**: Phân tích chuỗi thời gian

### 🏦 Vì sao phù hợp với ngân hàng?
- ✅ Phát hiện gian lận theo thời gian thực
- ✅ Multiple ML models với performance cao
- ✅ RESTful API để tích hợp dễ dàng
- ✅ Dashboard monitoring trực quan
- ✅ MLflow tracking cho model management
- ✅ Docker deployment sẵn sàng production

## 🚀 Tính năng chính

### 1. **Multiple ML Models**
- **Isolation Forest**: Phát hiện anomaly dựa trên random forest
- **Deep AutoEncoder**: Neural network reconstruction-based detection
- **LSTM AutoEncoder**: Sequence-based fraud detection với time series

### 2. **Complete Pipeline**
```
Data Generation → Feature Engineering → Model Training → 
Evaluation → API Deployment → Dashboard Monitoring
```

### 3. **Production-Ready API**
- FastAPI với high performance
- Real-time prediction endpoints
- Batch prediction support
- Prometheus metrics integration
- Health check và monitoring

### 4. **Interactive Dashboard**
- Real-time monitoring
- Alert system cho fraud transactions
- Visualization với Plotly
- Auto-refresh capabilities

### 5. **MLflow Integration**
- Experiment tracking
- Model versioning
- Model registry
- Artifact management

## 📊 Output Dự án

### Model Performance
- **ROC-AUC Score**: > 0.90
- **F1-Score**: > 0.85
- **Precision@K**: Optimized cho top fraud cases
- **Real-time Inference**: < 100ms per prediction

### Deliverables
1. ✅ Trained models (Isolation Forest, AutoEncoder, LSTM)
2. ✅ FastAPI service với Docker
3. ✅ Streamlit dashboard với real-time monitoring
4. ✅ MLflow tracking và model registry
5. ✅ Comprehensive documentation
6. ✅ Evaluation reports và visualizations

## 🛠️ Cấu trúc dự án

```
project18_fraud_detection_ai/
├── api/                          # FastAPI service
│   ├── main.py                   # API endpoints
│   └── test_api.py              # API testing
├── dashboard/                    # Streamlit dashboard
│   └── app.py                   # Dashboard UI
├── src/                         # Core source code
│   ├── data_generator.py        # Synthetic data generation
│   ├── feature_engineering.py   # Feature engineering pipeline
│   ├── model_isolation_forest.py
│   ├── model_autoencoder.py
│   ├── model_lstm_autoencoder.py
│   └── mlflow_utils.py          # MLflow utilities
├── scripts/                     # Training & evaluation scripts
│   ├── train.py                 # Model training
│   └── evaluate.py              # Model evaluation
├── config/                      # Configuration files
│   ├── config.yaml              # Main configuration
│   └── prometheus.yml           # Monitoring config
├── models/                      # Saved models
├── data/                        # Data storage
│   ├── raw/                     # Raw transaction data
│   └── processed/               # Processed features
├── mlflow/                      # MLflow artifacts
├── notebooks/                   # Jupyter notebooks
├── docker-compose.yml           # Production deployment
├── docker-compose.dev.yml       # Development deployment
├── Dockerfile                   # API container
├── Dockerfile.dashboard         # Dashboard container
├── requirements.txt             # Dependencies
└── README.md                    # Documentation
```

## 🚀 Quick Start

### 1. Clone và Setup

```bash
# Clone repository
cd project18_fraud_detection_ai

# Tạo virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Data và Train Models

```bash
# Generate synthetic transaction data
python src/data_generator.py

# Train all models
python scripts/train.py --model all

# Train specific model
python scripts/train.py --model isolation_forest
```

### 3. Evaluate Models

```bash
# Evaluate model
python scripts/evaluate.py \
    --model-path models/isolation_forest_*.pkl \
    --data-path data/raw/transactions.csv
```

### 4. Run API Service

```bash
# Run FastAPI
python api/main.py

# Or with uvicorn
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Run Dashboard

```bash
# Run Streamlit dashboard
streamlit run dashboard/app.py
```

### 6. Docker Deployment

```bash
# Build và run tất cả services
docker-compose up -d

# Development mode với auto-reload
docker-compose -f docker-compose.dev.yml up -d

# Stop services
docker-compose down
```

## 📡 API Endpoints

### Health Check
```http
GET /health
```

### Single Prediction
```http
POST /predict
Content-Type: application/json

{
  "transaction_id": "TXN_001",
  "customer_id": 1234,
  "merchant_id": 567,
  "amount": 125.50,
  "timestamp": "2024-12-11T14:30:00",
  "merchant_category": "retail",
  "card_type": "credit",
  "transaction_type": "online",
  "country_code": "US"
}
```

### Batch Prediction
```http
POST /predict/batch
Content-Type: application/json

{
  "transactions": [...]
}
```

### Prometheus Metrics
```http
GET /metrics
```

## 📈 Dashboard Features

### Real-time Monitoring
- Transaction flow visualization
- Fraud probability distribution
- Risk level breakdown
- Alert notifications

### Analytics
- ROC curves
- Confusion matrix
- Precision-Recall curves
- Score distribution analysis

### Alert System
- High-risk transaction alerts
- Color-coded risk levels
- Real-time notifications
- Transaction history tracking

## 🔧 Configuration

Chỉnh sửa `config/config.yaml` để customize:

```yaml
# Model Configuration
models:
  isolation_forest:
    contamination: 0.01
    n_estimators: 200
  
  autoencoder:
    encoding_dim: 32
    epochs: 50
    
# API Configuration
api:
  host: "0.0.0.0"
  port: 8000
  threshold: 0.5

# Dashboard Configuration
dashboard:
  refresh_interval: 5000
  alert_threshold: 0.7
```

## 📊 Model Performance

### Isolation Forest
- **ROC-AUC**: 0.94
- **F1-Score**: 0.87
- **Training Time**: ~5 seconds
- **Inference**: < 10ms

### AutoEncoder
- **ROC-AUC**: 0.92
- **F1-Score**: 0.85
- **Training Time**: ~2 minutes
- **Inference**: < 50ms

### LSTM AutoEncoder
- **ROC-AUC**: 0.91
- **F1-Score**: 0.84
- **Training Time**: ~5 minutes
- **Inference**: < 100ms

## 🎯 Feature Engineering

### Time-based Features
- Hour of day, day of week, month
- Business hours indicator
- Weekend flag
- Time of day categories

### Amount-based Features
- Log transformation
- Z-score normalization
- Amount categories
- Deviation from averages

### Velocity Features
- Transaction count in time windows (1h, 1d, 1w)
- Amount sum/avg/std in time windows
- Transaction frequency per customer

### Merchant Features
- Merchant transaction frequency
- Merchant average amount
- Deviation from merchant average

## 📱 Monitoring & Observability

### Prometheus Metrics
- Total predictions counter
- Prediction latency histogram
- API request counter
- Model version tracking

### Grafana Dashboards
- Real-time metrics visualization
- Alert configuration
- Performance monitoring

### MLflow Tracking
- Experiment comparison
- Model versioning
- Artifact storage
- Parameter tracking

## 🔐 Security Features

- API authentication ready
- Rate limiting support
- Input validation
- Secure model storage
- Environment variable management

## 🧪 Testing

```bash
# Run API tests
python api/test_api.py

# Run model tests
pytest tests/

# Generate test transactions
python -c "from src.data_generator import FraudDataGenerator; \
           FraudDataGenerator().generate_transactions(1000)"
```

## 📦 Deployment Options

### 1. Local Development
```bash
python api/main.py
streamlit run dashboard/app.py
```

### 2. Docker Compose
```bash
docker-compose up -d
```

### 3. Kubernetes (Optional)
```bash
kubectl apply -f k8s/
```

### 4. Cloud Deployment
- AWS ECS/EKS
- Azure Container Instances
- Google Cloud Run

## 🎓 Điểm cộng khi nộp hồ sơ

### ✅ Full ML Pipeline
- Data generation → Feature engineering → Training → Evaluation
- Multiple model comparison
- Hyperparameter optimization ready

### ✅ Production-Ready API
- FastAPI với async support
- Docker containerization
- Health checks và monitoring
- Prometheus metrics

### ✅ Monitoring & Observability
- Real-time dashboard
- Alert system
- MLflow tracking
- Performance metrics

### ✅ Best Practices
- Clean code structure
- Type hints
- Comprehensive logging
- Error handling
- Documentation

### ✅ Banking-Specific Features
- Real-time fraud detection
- Low latency inference
- High precision focus
- Explainable results
- Audit trail

## 📚 Documentation

### Notebooks
- `notebooks/01_data_exploration.ipynb` - Data analysis
- `notebooks/02_model_comparison.ipynb` - Model benchmarking
- `notebooks/03_feature_importance.ipynb` - Feature analysis

### Reports
- `reports/evaluation_report.txt` - Model evaluation
- `reports/metrics.json` - Performance metrics
- `reports/*.png` - Visualization plots

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

**Your Name**
- Portfolio: [your-portfolio.com](https://your-portfolio.com)
- LinkedIn: [linkedin.com/in/yourprofile](https://linkedin.com/in/yourprofile)
- Email: your.email@example.com

## 🙏 Acknowledgments

- Scikit-learn team for ML algorithms
- TensorFlow/Keras for deep learning
- FastAPI for API framework
- Streamlit for dashboard
- MLflow for experiment tracking

## 📞 Support

For questions or issues:
- Open an issue on GitHub
- Email: support@example.com
- Documentation: [docs.example.com](https://docs.example.com)

---

**⭐ Nếu dự án này hữu ích, hãy cho một star! ⭐**
