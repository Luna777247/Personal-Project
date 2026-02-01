# Communication Campaign Analysis Tool (Project 6)

## Tổng quan (Overview)

Công cụ phân tích chiến dịch truyền thông toàn diện cho doanh nghiệp Việt Nam, giúp đánh giá hiệu quả chiến dịch, phân tích tác động PR, và đưa ra khuyến nghị chiến lược.

A comprehensive communication campaign analysis tool for Vietnamese businesses, designed to evaluate campaign effectiveness, analyze PR impact, and provide strategic recommendations.

## Tính năng chính (Key Features)

### 📊 Phân tích hiệu suất chiến dịch (Campaign Performance Analysis)
- Đo lường reach, engagement rate, conversion rate
- Tính toán ROI và hiệu quả tổng thể
- Phân tích theo loại chiến dịch và mục tiêu

### 📢 Phân tích hiệu quả kênh (Channel Effectiveness Analysis)
- So sánh hiệu suất giữa các kênh truyền thông
- Phân tích chi phí trên mỗi tương tác
- Xác định kênh hiệu quả nhất

### 😊 Phân tích tác động cảm xúc (Sentiment Impact Analysis)
- Phân tích cảm xúc từ mạng xã hội
- Tương quan giữa cảm xúc và hiệu quả chiến dịch
- Dự đoán tác động cảm xúc lên kết quả

### 📰 Phân tích tác động PR (PR Impact Analysis)
- Đo lường lượng đề cập trên truyền thông
- Tính toán giá trị truyền thông thu được
- Đánh giá mức độ nâng cao nhận thức thương hiệu

### 🎯 Phân khúc chiến dịch (Campaign Segmentation)
- Phân loại chiến dịch theo hiệu suất
- Xác định chiến dịch champion và underperforming
- Phân tích đặc điểm của từng nhóm

### 💡 Khuyến nghị chiến lược (Strategic Recommendations)
- Đề xuất phân bổ ngân sách tối ưu
- Gợi ý kênh truyền thông hiệu quả
- Chiến lược cải thiện hiệu quả

## Cấu trúc dự án (Project Structure)

```
project6_communication_analysis/
├── scripts/
│   ├── communication_analyzer.py    # Main analysis tool
│   ├── test_communication_analysis.py  # Test suite
│   ├── data/                        # Generated data files
│   │   ├── campaign_performance_data.csv
│   │   └── analysis_summary.json
│   └── results/                     # Analysis results
│       ├── communication_campaign_report.md
│       └── communication_analysis_dashboard.png
├── docs/                            # Documentation
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## Yêu cầu hệ thống (System Requirements)

- Python 3.8+
- 4GB RAM (khuyến nghị)
- Windows/Linux/macOS

## Cài đặt (Installation)

1. **Tải về dự án (Download project):**
   ```bash
   git clone <repository-url>
   cd project6_communication_analysis
   ```

2. **Cài đặt dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Sử dụng (Usage)

### Chạy phân tích đầy đủ (Run Full Analysis)

```python
from communication_analyzer import CommunicationCampaignAnalyzer

# Khởi tạo analyzer
analyzer = CommunicationCampaignAnalyzer()

# Chạy phân tích hoàn chỉnh
analyzer.run_complete_analysis()
```

### Chạy từ command line

```bash
cd scripts
python communication_analyzer.py
```

### Chạy test

```bash
cd scripts
python test_communication_analysis.py
```

## Dependencies

```
pandas>=1.5.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
scikit-learn>=1.0.0
scipy>=1.7.0
statsmodels>=0.13.0
plotly>=5.0.0
streamlit>=1.10.0
textblob>=0.17.0
wordcloud>=1.8.0
nltk>=3.7
requests>=2.25.0
beautifulsoup4>=4.9.0
```

## Đầu ra (Outputs)

### Báo cáo chi tiết (Detailed Report)
- `results/communication_campaign_report.md`: Báo cáo phân tích toàn diện
- `results/communication_analysis_dashboard.png`: Dashboard trực quan

### Dữ liệu phân tích (Analysis Data)
- `data/campaign_performance_data.csv`: Dữ liệu hiệu suất chiến dịch
- `data/analysis_summary.json`: Tóm tắt kết quả phân tích

## Các chỉ số chính (Key Metrics)

### Hiệu suất chiến dịch (Campaign Performance)
- **Reach**: Số người tiếp cận
- **Engagement Rate**: Tỷ lệ tương tác
- **Conversion Rate**: Tỷ lệ chuyển đổi
- **ROI**: Tỷ suất lợi nhuận

### Hiệu quả kênh (Channel Effectiveness)
- **Cost per Engagement**: Chi phí trên mỗi tương tác
- **Cost per Conversion**: Chi phí trên mỗi chuyển đổi
- **ROI Contribution**: Đóng góp vào lợi nhuận tổng thể

### Tác động PR (PR Impact)
- **Media Mentions**: Số đề cập trên truyền thông
- **Earned Media Value**: Giá trị truyền thông thu được
- **Brand Awareness Lift**: Nâng cao nhận thức thương hiệu

## Phương pháp phân tích (Analysis Methodology)

### 1. Thu thập dữ liệu (Data Collection)
- Tạo dữ liệu giả lập cho các chiến dịch
- Thu thập metrics từ nhiều nguồn
- Chuẩn hóa định dạng dữ liệu

### 2. Phân tích thống kê (Statistical Analysis)
- Phân tích tương quan
- Kiểm định ý nghĩa thống kê
- Mô hình hồi quy tuyến tính

### 3. Phân tích cluster (Cluster Analysis)
- K-means clustering cho phân khúc chiến dịch
- Phân tích đặc điểm của từng nhóm
- Xác định chiến lược cho từng phân khúc

### 4. Trực quan hóa (Visualization)
- Biểu đồ hiệu suất kênh
- Ma trận tương quan
- Dashboard tổng hợp

## Kết quả mẫu (Sample Results)

```
Campaign Performance Summary:
Average Reach: 55,314
Average Engagement Rate: 3.9%
Average ROI: 99.5%

Top Performing Channels by ROI:
  Events: 45.3% ROI
  Paid Ads: 40.3% ROI
  Email: 38.2% ROI
```

## Đóng góp (Contributing)

1. Fork dự án
2. Tạo feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Tạo Pull Request

## Giấy phép (License)

Dự án này được phân phối dưới giấy phép MIT. Xem file `LICENSE` để biết thêm chi tiết.

## Liên hệ (Contact)

**Portfolio Projects**
- Email: [your-email@example.com]
- LinkedIn: [your-linkedin-profile]
- GitHub: [your-github-profile]

---

*Đây là Project 6 trong series Data Engineering & Business Intelligence Portfolio*