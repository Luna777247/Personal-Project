# Quickstart Guide - Credit Scoring System

## ⚡ 5-Minute Setup

### Bước 1: Clone và Install (2 phút)

```bash
# Clone repository
git clone <repo-url>
cd project21_credit_scoring

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Bước 2: Train Model (2 phút)

```bash
# Generate data và train model
python train_model.py
```

Output:
```
[Step 1/9] Loading data...
Generated new data: (10000, 30)
[Step 2/9] Cleaning data...
Cleaned data: (9950, 30)
[Step 3/9] Engineering features...
Engineered features: (9950, 65)
...
Best Model: XGBOOST
Test AUC: 0.8542
```

### Bước 3: Start API (1 phút)

```bash
# Start FastAPI server
uvicorn api.main:app --reload --port 8000
```

API đã sẵn sàng tại: http://localhost:8000

## 🧪 Test API

### Test 1: Predict Endpoint

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "gender": "Male",
    "marital_status": "Married",
    "dependents": 2,
    "employment_status": "Employed",
    "employment_length": 10,
    "income": 75000,
    "credit_history_length": 8,
    "num_credit_lines": 5,
    "num_open_accounts": 3,
    "total_debt": 25000,
    "loan_amount": 20000,
    "loan_term": 60,
    "loan_purpose": "home_improvement",
    "interest_rate": 7.5,
    "monthly_payment": 400,
    "num_late_payments": 1,
    "num_delinquencies": 0,
    "home_ownership": "Own",
    "education_level": "Bachelor",
    "has_cosigner": false,
    "has_guarantor": false
  }'
```

Response:
```json
{
  "score": 0.3245,
  "probability": 0.3245,
  "risk_level": "medium",
  "approval_decision": "Approved",
  "key_factors": [
    "High debt-to-income ratio (0.33)",
    "1 late payment(s)"
  ]
}
```

### Test 2: Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "1.0.0",
  "uptime_seconds": 123.45
}
```

### Test 3: API Docs

Mở browser: http://localhost:8000/docs

Swagger UI với interactive API documentation.

## 📊 View Results

Sau khi train model, check:

```bash
# Model files
ls models/
# xgboost_model.json
# lightgbm_model.txt
# scaler.pkl

# Evaluation plots
ls results/
# roc_curve.png
# pr_curve.png
# confusion_matrix.png
# threshold_analysis.png

# SHAP explanations
ls models/explanations/
# shap_summary.png
# shap_importance.png
# feature_importance.csv
```

## 🎯 Next Steps

### 1. Customize Features

Chỉnh sửa `src/features/feature_engineer.py`:

```python
# Thêm feature mới
def _create_custom_features(self, df):
    # VD: Tỷ lệ khoản vay so với tài sản
    df['loan_to_assets'] = df['loan_amount'] / (df['assets'] + 1)
    return df
```

### 2. Tune Hyperparameters

Chỉnh sửa `config/config.yaml`:

```yaml
model:
  xgboost:
    max_depth: 8        # Tăng độ phức tạp
    learning_rate: 0.05 # Giảm learning rate
    n_estimators: 200   # Thêm trees
```

### 3. Adjust Risk Thresholds

Chỉnh sửa `.env`:

```
APPROVAL_THRESHOLD=0.5  # Thấp hơn = approve nhiều hơn
```

### 4. Add Custom Business Logic

Chỉnh sửa `api/predictor.py`:

```python
def determine_approval(self, probability, data):
    # Custom logic
    if data['income'] > 100000 and probability < 0.7:
        return "Approved"
    elif data['has_cosigner'] and probability < 0.8:
        return "Review"
    else:
        return "Denied"
```

## 🐛 Troubleshooting

### Issue: Module not found

```bash
# Ensure virtual environment is activated
which python  # Should point to venv/bin/python

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: Model not loading

```bash
# Check model file exists
ls models/xgboost_model.json

# Retrain if missing
python train_model.py
```

### Issue: API error 503

```bash
# Check MODEL_PATH in .env
cat .env | grep MODEL_PATH

# Should be: MODEL_PATH=models/xgboost_model.json
```

### Issue: Low model performance

```bash
# Generate more data
# Edit train_model.py line:
# df = generate_credit_data(n_samples=50000)  # Increase from 10000

python train_model.py
```

## 💡 Tips

### Tip 1: Fast Iteration

```bash
# Train small model for testing
python train_model.py --samples 1000 --estimators 50
```

### Tip 2: Monitor API

```bash
# Check metrics
curl http://localhost:8000/metrics

# Response:
# {
#   "total_predictions": 100,
#   "avg_response_time_ms": 45.2,
#   "approval_rate": 0.65,
#   "avg_risk_score": 0.42
# }
```

### Tip 3: Batch Processing

```python
import requests

applications = [
    {...},  # Application 1
    {...},  # Application 2
    {...},  # Application 3
]

response = requests.post(
    "http://localhost:8000/batch_predict",
    json={"applications": applications}
)

print(response.json()['summary'])
# {'approved': 2, 'review': 0, 'denied': 1}
```

## 🚀 Production Deployment

### Docker

```bash
# Build
docker build -t credit-scoring .

# Run
docker run -p 8000:8000 \
  -e MODEL_PATH=/app/models/xgboost_model.json \
  credit-scoring
```

### Environment Variables

```bash
export API_PORT=8000
export MODEL_TYPE=xgboost
export MODEL_VERSION=v1.0
export APPROVAL_THRESHOLD=0.6
export LOG_LEVEL=INFO

uvicorn api.main:app --host 0.0.0.0 --port $API_PORT
```

## 📞 Support

- **Documentation**: Check README.md
- **Issues**: Open GitHub issue
- **Email**: your.email@example.com

---

**Ready to go!** 🎉

Your credit scoring system is now running at http://localhost:8000
