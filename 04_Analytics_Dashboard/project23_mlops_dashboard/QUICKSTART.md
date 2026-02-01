# Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Prerequisites

- Docker & Docker Compose installed
- Git installed
- 8GB+ RAM recommended
- GPU optional (Triton supports both CPU and GPU)

### Step 1: Clone & Setup (1 min)

```bash
# Clone repository
git clone <your-repo-url>
cd project23_mlops_dashboard

# Copy environment variables
cp .env.example .env

# Review and customize .env if needed
# Default settings work for local development
```

### Step 2: Start Services (2 min)

```bash
# Start all services
docker-compose up -d

# Wait for services to be ready (check health)
docker-compose ps

# View logs
docker-compose logs -f
```

**Services Starting:**
- PostgreSQL (port 5432) - MLflow backend
- MinIO (port 9000, 9001) - Model artifacts storage
- MLflow (port 5000) - Tracking server
- Triton (ports 8000, 8001, 8002) - Inference server
- Prometheus (port 9090) - Metrics collection
- Grafana (port 3000) - Visualization
- Dashboard (port 8501) - Streamlit UI
- API (port 8080) - Inference API

### Step 3: Train Your First Model (1 min)

```bash
# Activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Train fraud detection model
python examples/train_fraud_detector.py
```

**Expected Output:**
```
Fraud Detection Model Training Pipeline
✓ Generated 10000 samples
✓ Random Forest trained - F1: 0.9524
✓ XGBoost trained - F1: 0.9687
✓ Best Model: XGBoost
```

### Step 4: View Results (1 min)

#### MLflow UI
```bash
# Open browser
http://localhost:5000
```

**What to see:**
- Experiments list with "fraud-detection"
- 2 runs (Random Forest + XGBoost)
- Metrics comparison charts
- Model artifacts

#### Streamlit Dashboard
```bash
# Open browser
http://localhost:8501
```

**Features:**
- 🏠 Overview: System status, recent runs
- 📊 Experiments: Compare metrics, visualizations
- 📦 Models: Registry browser, version management
- 🚀 Deployment: Test inference, model status
- 📈 Monitoring: Real-time metrics

### Step 5: Deploy Model (Quick)

```bash
# Export best model
python scripts/export_best_model.py \
  --experiment-name fraud-detection \
  --metric val_f1_score \
  --output artifacts

# Export to Triton format
python scripts/export_to_triton.py \
  --model-path artifacts/best_model/model \
  --model-name fraud_detector \
  --version 1 \
  --output models
```

### Step 6: Test Inference

#### Using Python Client
```python
from src.inference.triton_client import TritonGRPCClient
import numpy as np

# Connect to Triton
client = TritonGRPCClient(url="localhost:8001")

# Check model
is_ready = client.is_model_ready("fraud_detector")
print(f"Model ready: {is_ready}")

# Test inference
test_data = np.random.rand(1, 20).astype(np.float32)
results = client.infer(
    model_name="fraud_detector",
    inputs={"float_input": test_data}
)
print(f"Prediction: {results}")
```

#### Using REST API
```bash
# Test API health
curl http://localhost:8080/health

# Get model info
curl http://localhost:8080/models/fraud_detector

# Run inference
curl -X POST http://localhost:8080/infer \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "fraud_detector",
    "model_version": "1",
    "inputs": {
      "float_input": [[0.5, 0.3, 0.8, ...]]
    }
  }'
```

### Verify Everything Works

```bash
# Check all services
docker-compose ps

# Should see all services "Up (healthy)"

# Access UIs
MLflow:    http://localhost:5000
Grafana:   http://localhost:3000  (admin/admin)
MinIO:     http://localhost:9001  (minioadmin/minioadmin)
Dashboard: http://localhost:8501
API:       http://localhost:8080
Prometheus: http://localhost:9090
```

## 🎯 What You Just Built

- ✅ Complete MLOps pipeline
- ✅ Experiment tracking with MLflow
- ✅ Model registry with versioning
- ✅ Production inference with Triton
- ✅ REST + gRPC APIs
- ✅ Metrics monitoring
- ✅ Interactive dashboard

## 📚 Next Steps

1. **Customize Training**: Modify `examples/train_fraud_detector.py`
2. **Setup CI/CD**: Configure GitHub Actions (see `.github/workflows/`)
3. **Deploy to Cloud**: Use Kubernetes configs in `k8s/`
4. **Monitor Performance**: Setup alerts in Grafana
5. **A/B Testing**: Deploy multiple model versions

## 🐛 Troubleshooting

### Services not starting?
```bash
# Check logs
docker-compose logs

# Restart specific service
docker-compose restart mlflow

# Full reset
docker-compose down -v
docker-compose up -d
```

### Port conflicts?
```bash
# Edit .env and change ports
MLFLOW_PORT=5001
TRITON_HTTP_PORT=8001
# etc.

# Restart
docker-compose down
docker-compose up -d
```

### Model not loading in Triton?
```bash
# Check model repository
ls -la models/fraud_detector/1/

# Should contain:
# - model.onnx
# - config.pbtxt

# Check Triton logs
docker-compose logs triton

# Reload model
curl -X POST http://localhost:8000/v2/repository/models/fraud_detector/load
```

## 📞 Need Help?

- **Documentation**: See `docs/` folder
- **Examples**: Check `examples/` folder
- **API Docs**: http://localhost:8080/docs (FastAPI auto-docs)
- **MLflow Docs**: https://mlflow.org
- **Triton Docs**: https://github.com/triton-inference-server

## 🎉 Success!

You now have a production-ready MLOps system running locally!

**Time to complete**: ~5 minutes  
**Services running**: 8 Docker containers  
**Model trained**: Fraud detection with 96%+ F1 score  
**APIs available**: REST (FastAPI) + gRPC (Triton)  
**Monitoring**: Prometheus + Grafana dashboards
