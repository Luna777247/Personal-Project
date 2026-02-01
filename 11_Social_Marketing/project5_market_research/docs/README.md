# Market Research Analysis Tool (Project 5)

Công cụ Phân tích Nghiên cứu Thị trường - Một công cụ toàn diện để thu thập, phân tích và báo cáo nghiên cứu thị trường.

## Tổng quan

Công cụ này cung cấp khả năng phân tích nghiên cứu thị trường toàn diện bao gồm:
- Phân tích nhân khẩu học khách hàng
- Đánh giá nhận thức thương hiệu
- Phân tích hành vi mua sắm
- Xác định điểm đau của khách hàng
- Phân khúc khách hàng tự động
- Đề xuất chiến lược truyền thông
- Báo cáo và trực quan hóa tự động

## Tính năng chính

### 📊 Phân tích nhân khẩu học
- Phân bố độ tuổi, giới tính, thu nhập
- Mức độ giáo dục
- Thống kê chi tiết theo nhóm

### 🏷️ Đánh giá thương hiệu
- Mức độ nhận thức thương hiệu
- Mức độ hài lòng khách hàng
- Chỉ số lòng trung thành
- Net Promoter Score (NPS)

### 🛒 Phân tích hành vi mua sắm
- Mức chi tiêu hàng tháng
- Tần suất mua hàng
- Danh mục sản phẩm ưa thích
- Kênh mua hàng ưu tiên

### 😞 Xác định điểm đau
- Phân tích vấn đề khách hàng gặp phải
- Ưu tiên vấn đề theo tần suất
- Đề xuất giải pháp cải thiện

### 🎯 Phân khúc khách hàng
- Phân tích cụm tự động (K-means)
- Đặc điểm từng phân khúc
- Chiến lược tiếp cận phù hợp

### 📢 Chiến lược truyền thông
- Kênh truyền thông hiệu quả nhất
- Thông điệp chính cần truyền tải
- Chiến lược theo phân khúc

## Cài đặt

### Yêu cầu hệ thống
- Python 3.8+
- pip (trình quản lý gói Python)

### Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### Chạy chương trình
```bash
cd scripts
python market_research_analyzer.py
```

## Cấu trúc thư mục

```
project5_market_research/
├── scripts/
│   └── market_research_analyzer.py  # Script chính
├── data/
│   ├── processed_survey_data.csv    # Dữ liệu khảo sát đã xử lý
│   └── customer_segments.csv        # Dữ liệu phân khúc khách hàng
├── results/
│   ├── market_research_dashboard.png # Biểu đồ tổng quan
│   └── market_research_report.md     # Báo cáo chi tiết
├── docs/
│   └── README.md                     # Tài liệu hướng dẫn
└── requirements.txt                  # Danh sách thư viện cần thiết
```

## Cách sử dụng

### 1. Chuẩn bị dữ liệu
Công cụ sử dụng dữ liệu khảo sát mẫu được tạo tự động. Để sử dụng dữ liệu thực:
- Thay thế hàm `generate_mock_survey_data()` bằng dữ liệu khảo sát thực
- Đảm bảo format dữ liệu phù hợp với cấu trúc mong đợi

### 2. Chạy phân tích
```python
from market_research_analyzer import MarketResearchAnalyzer

analyzer = MarketResearchAnalyzer()
results = analyzer.run_complete_analysis(num_respondents=200)
```

### 3. Tùy chỉnh phân tích
- Thay đổi số lượng người trả lời khảo sát
- Điều chỉnh số lượng phân khúc khách hàng
- Thêm/bớt câu hỏi khảo sát

## Đầu ra

### Báo cáo văn bản
- Báo cáo chi tiết đầy đủ trong `results/market_research_report.md`
- Bao gồm tất cả phân tích và khuyến nghị

### Trực quan hóa
- Dashboard tổng quan trong `results/market_research_dashboard.png`
- 6 biểu đồ chính về nhân khẩu học, thương hiệu, và hành vi

### Dữ liệu đã xử lý
- Dữ liệu khảo sát đã làm sạch trong `data/processed_survey_data.csv`
- Dữ liệu phân khúc khách hàng trong `data/customer_segments.csv`

## Ví dụ kết quả

### Chỉ số chính
```
Total respondents: 150
Average NPS: 6.2/10
Average monthly spending: $148.50
Top pain point: High prices
Number of segments: 4
```

### Phân tích NPS
- **NPS Score:** 6.2 (cần cải thiện)
- **Promoters:** 25% (khách hàng ủng hộ)
- **Passives:** 35% (khách hàng trung lập)
- **Detractors:** 40% (khách hàng phản đối)

### Điểm đau hàng đầu
1. Giá cao (45 mentions)
2. Chất lượng kém (38 mentions)
3. Giao hàng chậm (32 mentions)

## Mở rộng

### Thêm nguồn dữ liệu
- Kết nối với Google Forms, SurveyMonkey
- Tích hợp dữ liệu CRM
- Nhập dữ liệu từ Excel/CSV

### Phân tích nâng cao
- Phân tích cảm xúc (sentiment analysis)
- Dự đoán hành vi khách hàng
- A/B testing recommendations

### Tích hợp
- Dashboard web với Streamlit
- API REST cho tích hợp hệ thống
- Xuất báo cáo PDF tự động

## Công nghệ sử dụng

- **Python 3.8+**: Ngôn ngữ lập trình chính
- **pandas**: Xử lý và phân tích dữ liệu
- **scikit-learn**: Machine learning và clustering
- **matplotlib/seaborn**: Trực quan hóa dữ liệu
- **scipy**: Thống kê và phân tích khoa học

## Đóng góp

1. Fork dự án
2. Tạo branch tính năng mới (`git checkout -b feature/AmazingFeature`)
3. Commit thay đổi (`git commit -m 'Add some AmazingFeature'`)
4. Push lên branch (`git push origin feature/AmazingFeature`)
5. Tạo Pull Request

## Giấy phép

Dự án này được phân phối dưới giấy phép MIT. Xem file `LICENSE` để biết thêm chi tiết.

## Liên hệ

- **Tác giả:** [Tên của bạn]
- **Email:** [email của bạn]
- **GitHub:** [link GitHub]

---

*Công cụ được phát triển như một phần của portfolio dự án data engineering và business intelligence.*