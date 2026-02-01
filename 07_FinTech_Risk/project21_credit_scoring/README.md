# Credit Scoring Mini System

Mô hình chấm điểm tín dụng sử dụng Machine Learning với khả năng giải thích kết quả bằng SHAP.

## 🎯 Tổng Quan

Dự án này xây dựng hệ thống chấm điểm tín dụng hoàn chỉnh, từ xử lý dữ liệu đến triển khai API, tập trung vào:

- **ML Model**: XGBoost và LightGBM cho dữ liệu tabular
- **Feature Engineering**: Các chỉ số tài chính quan trọng (DTI, income ratio, payment history)
- **Explainability**: Sử dụng SHAP để giải thích quyết định
- **Production-Ready**: API FastAPI với rate limiting và monitoring
- **Business Focus**: Tư duy business kết hợp ML thực chiến

## 🏗️ Kiến Trúc

```
project21_credit_scoring/
├── data/                      # Dữ liệu
│   ├── raw/                  # Dữ liệu thô
│   └── processed/            # Dữ liệu đã xử lý
├── models/                   # Model artifacts
│   └── explanations/         # SHAP explanations
├── src/                      # Source code
│   ├── data/                # Data processing
│   ├── features/            # Feature engineering
│   ├── models/              # Model training
│   └── explainability/      # SHAP analysis
├── api/                      # FastAPI service
├── web/                      # Streamlit UI
├── notebooks/               # Jupyter notebooks
├── tests/                   # Unit tests
└── config/                  # Configuration
```

## 🚀 Cài Đặt

### 1. Clone Repository

```bash
git clone <repository-url>
cd project21_credit_scoring
```

### 2. Tạo Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### 3. Cài Đặt Dependencies

```bash
pip install -r requirements.txt
```

### 4. Cấu Hình Environment

```bash
cp .env.example .env
# Chỉnh sửa .env với cấu hình phù hợp
```

## 📊 Sử Dụng

### Bước 1: Train Model

```bash
python train_model.py
```

Script này sẽ:
1. Generate/load dữ liệu
2. Clean và preprocess data
3. Engineer features (DTI, income ratio, payment history)
4. Train XGBoost và LightGBM
5. Evaluate models
6. Generate SHAP explanations

### Bước 2: Khởi Động API

```bash
cd project21_credit_scoring
uvicorn api.main:app --reload --port 8000
```

API sẽ chạy tại: http://localhost:8000

### Bước 3: Test API

```bash
curl -X POST "http://localhost:8000/predict" \
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

## 🔑 Features Chính

### 1. Feature Engineering

**Financial Ratios:**
- **Debt-to-Income (DTI)**: `total_debt / annual_income` - Chỉ số quan trọng nhất
- **Loan-to-Income**: `loan_amount / income` - Khả năng thanh toán
- **Payment-to-Income**: `monthly_payment / monthly_income` - Gánh nặng hàng tháng

**Payment History:**
- **Payment Consistency Score**: Điểm đánh giá tính ổn định thanh toán
- **Delinquency Rate**: Tỷ lệ vi phạm / tổng số tháng credit history
- **Late Payment Rate**: Tỷ lệ thanh toán trễ

**Credit Utilization:**
- **Credit Utilization Ratio**: Mức sử dụng credit so với limit
- **Active Account Ratio**: Tài khoản hoạt động / tổng tài khoản
- **Debt Concentration**: Nợ trung bình trên mỗi credit line

### 2. Model Performance

**Metrics:**
- AUC-ROC: Đo lường khả năng phân loại
- Precision/Recall: Tradeoff giữa false positive và false negative
- F1 Score: Harmonic mean của precision và recall
- Business Metrics: Approval rate, default rate, expected loss

**Model Selection:**
- XGBoost: Hiệu quả cao, xử lý missing values tốt
- LightGBM: Nhanh hơn, phù hợp với dữ liệu lớn
- Cross-validation 5-fold để đảm bảo robustness

### 3. SHAP Explainability

**Why SHAP?**
- Giải thích được từng quyết định
- Tuân thủ quy định (regulatory compliance)
- Xây dựng trust với khách hàng

**Visualizations:**
- Summary Plot: Feature importance toàn cục
- Force Plot: Giải thích cho từng prediction
- Dependence Plot: Mối quan hệ feature-output
- Waterfall Plot: Contribution từng feature

### 4. API Endpoints

**POST /predict**
- Input: Thông tin khách hàng
- Output: Score, risk level, approval decision, key factors

**POST /explain**
- Input: Thông tin khách hàng  
- Output: Prediction + SHAP explanations

**POST /batch_predict**
- Input: Danh sách applications
- Output: Batch predictions + summary

**GET /health**
- Health check và model status

**GET /metrics**
- API usage statistics

## 📈 Kết Quả

### Model Performance (Test Set)

```
XGBoost:
- AUC: 0.85+
- Precision: 0.80+
- Recall: 0.75+
- F1 Score: 0.77+

LightGBM:
- AUC: 0.84+
- Precision: 0.79+
- Recall: 0.74+
- F1 Score: 0.76+
```

### Top 10 Most Important Features

1. **debt_to_income_ratio** - Chỉ số DTI chuẩn
2. **payment_consistency_score** - Lịch sử thanh toán
3. **credit_history_length** - Độ dài credit history
4. **loan_to_income_ratio** - Tỷ lệ khoản vay/thu nhập
5. **num_late_payments** - Số lần thanh toán trễ
6. **income** - Thu nhập hàng năm
7. **employment_stability** - Tính ổn định công việc
8. **credit_utilization** - Tỷ lệ sử dụng credit
9. **total_negative_events** - Tổng sự kiện tiêu cực
10. **age** - Tuổi khách hàng

## 🎓 Business Insights

### Risk Thresholds

```yaml
Low Risk (0.0 - 0.3):
  - DTI < 0.35
  - No late payments
  - Good credit history
  - Action: Auto-approve

Medium Risk (0.3 - 0.6):
  - DTI 0.35 - 0.50
  - Max 2 late payments
  - Decent credit history
  - Action: Manual review

High Risk (0.6 - 0.8):
  - DTI > 0.50
  - Multiple late payments
  - Short credit history
  - Action: Deny or require collateral

Very High Risk (0.8 - 1.0):
  - DTI > 0.65
  - Recent delinquencies
  - Bankruptcy history
  - Action: Auto-deny
```

### Cost Analysis

```python
# False Positive Cost: Approve bad loan = $100 loss
# False Negative Cost: Reject good customer = $20 opportunity cost

# Optimal threshold balances these costs
threshold = 0.6  # Can be adjusted based on business strategy
```

## 🧪 Testing

```bash
# Run unit tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test
pytest tests/test_models.py
```

## 📦 Deployment

### Docker

```bash
# Build image
docker build -t credit-scoring-api .

# Run container
docker run -p 8000:8000 credit-scoring-api
```

### Docker Compose

```bash
docker-compose up -d
```

## 📝 Configuration

Chỉnh sửa `config/config.yaml` để thay đổi:
- Data processing parameters
- Feature engineering logic
- Model hyperparameters
- API settings
- Risk thresholds

## 🔐 Security

- Rate limiting: 100 requests/minute
- Input validation với Pydantic
- Environment variables cho sensitive data
- CORS configuration

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork repository
2. Create feature branch
3. Make changes
4. Add tests
5. Submit pull request

## 📄 License

MIT License - see LICENSE file

## 👥 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- Email: your.email@example.com

## 🙏 Acknowledgments

- XGBoost and LightGBM teams
- SHAP library by Scott Lundberg
- FastAPI framework
- Scikit-learn community

---

**Note**: Đây là dự án demo showcase. Trong production, cần:
- Dữ liệu thực từ credit bureau
- Model monitoring và retraining
- A/B testing
- Compliance với quy định
- Disaster recovery plan
