# 🌊 Hệ Thống Phát Hiện Lũ Lụt / Flood Detection System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Earth Engine API](https://img.shields.io/badge/Earth%20Engine-API-green.svg)](https://earthengine.google.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Hệ thống phát hiện lũ lụt tự động sử dụng Sentinel-1 SAR với phương pháp ensemble và đánh giá tác động cấp huyện.

*Automated flood detection system using Sentinel-1 SAR with ensemble approach and district-level impact assessment.*

---

## 📋 Mục Lục / Table of Contents

- [Tính Năng / Features](#-tính-năng--features)
- [Cài Đặt / Installation](#-cài-đặt--installation)
- [Sử Dụng Nhanh / Quick Start](#-sử-dụng-nhanh--quick-start)
- [Cấu Hình / Configuration](#-cấu-hình--configuration)
- [Kết Quả / Results](#-kết-quả--results)
- [Tài Liệu / Documentation](#-tài-liệu--documentation)
- [Giấy Phép / License](#-giấy-phép--license)

---

## 🎯 Tính Năng / Features

### Tiếng Việt

- ✅ **5 Phương Pháp Phát Hiện**: EMS Conservative, K-means, Adaptive Landcover, Adaptive Mean-Std, Change Detection
- ✅ **Ensemble Method**: Kết hợp kết quả với majority voting (≥3/5 phương pháp đồng ý)
- ✅ **Validation Đa Nguồn**: So sánh với Sentinel-2 optical và JRC permanent water
- ✅ **Phân Tích Tác Động**: Đánh giá thiệt hại cấp huyện với dữ liệu dân số và nông nghiệp
- ✅ **Tối Ưu Hiệu Suất**: Adaptive scale (10m-30m) dựa trên diện tích vùng
- ✅ **Lọc Morphological**: Giảm nhiễu với kernel 1 pixel
- ✅ **Confidence Mapping**: Đánh giá độ tin cậy với phân tích percentile

### English

- ✅ **5 Detection Methods**: EMS Conservative, K-means, Adaptive Landcover, Adaptive Mean-Std, Change Detection
- ✅ **Ensemble Method**: Combined results with majority voting (≥3/5 methods agree)
- ✅ **Multi-Source Validation**: Comparison with Sentinel-2 optical and JRC permanent water
- ✅ **Impact Analysis**: District-level damage assessment with population and cropland data
- ✅ **Performance Optimization**: Adaptive scale (10m-30m) based on area size
- ✅ **Morphological Filtering**: Noise reduction with 1-pixel kernel
- ✅ **Confidence Mapping**: Reliability assessment with percentile analysis

---

## 🚀 Cài Đặt / Installation

### Yêu Cầu Hệ Thống / System Requirements

```bash
Python 3.8+
Google Earth Engine account
4GB+ RAM
```

### Bước 1: Clone Repository

```bash
git clone https://github.com/yourusername/flood-detection-gee.git
cd flood-detection-gee
```

### Bước 2: Cài Đặt Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
earthengine-api>=0.1.300
geemap>=0.20.0
pandas>=1.3.0
numpy>=1.20.0
matplotlib>=3.4.0
ipywidgets>=7.6.0
```

### Bước 3: Xác Thực Earth Engine / Authenticate Earth Engine

```bash
# Lần đầu tiên / First time
earthengine authenticate

# Khởi tạo trong code / Initialize in code
import ee
ee.Authenticate()
ee.Initialize(project='your-project-id')
```

---

## ⚡ Sử Dụng Nhanh / Quick Start

### Chạy Script Cơ Bản / Run Basic Script

```bash
python gee_khoanh_cùng_ngập_lụt_v2.py
```

### Sử Dụng Trong Python / Use in Python

```python
import ee
from gee_khoanh_cùng_ngập_lụt_v2 import Config, load_s1, calculate_water_area

# Khởi tạo / Initialize
ee.Initialize(project='your-project')
config = Config()

# Tải dữ liệu / Load data
roi = ee.Geometry.Rectangle([105.0, 16.0, 107.0, 17.0])
col, count, ids = load_s1('A', roi, '2025-09-01', '2025-11-30')
print(f"Found {count} images")

# Xử lý... / Process...
# (Xem DOCUMENTATION.md để biết workflow đầy đủ)
# (See DOCUMENTATION.md for full workflow)
```

---

## ⚙️ Cấu Hình / Configuration

### Tùy Chỉnh Tham Số / Customize Parameters

Sửa trong file `gee_khoanh_cùng_ngập_lụt_v2.py`:

```python
class Config:
    def __init__(self):
        # Vùng nghiên cứu / Study area
        self.country = "Viet Nam"
        self.provinces = ["Thua Thien - Hue", "Da Nang City"]
        
        # Thời gian / Time range
        self.start_date = "2025-09-01"
        self.end_date = "2025-11-30"
        
        # Ngưỡng phát hiện / Detection thresholds
        self.ems_threshold = -18.0  # dB
        self.adaptive_k = 1.0
        
        # Độ phân giải / Resolution
        self.scale_small_area = 10  # meters for areas < 1000 km²
        self.scale_large_area = 30  # meters for areas ≥ 1000 km²
```

### Các Tham Số Quan Trọng / Key Parameters

| Tham Số / Parameter | Mặc Định / Default | Mô Tả / Description |
|---------------------|-------------------|---------------------|
| `ems_threshold` | -18.0 dB | Ngưỡng EMS Conservative / EMS Conservative threshold |
| `adaptive_k` | 1.0 | Hệ số K cho adaptive mean-std / K-factor for adaptive mean-std |
| `baseline_days` | 60 | Số ngày tính baseline / Days for baseline calculation |
| `hand_threshold` | 20 m | Ngưỡng HAND / HAND threshold |
| `slope_threshold` | 15° | Độ dốc tối đa / Maximum slope |
| `kernel_size` | 1 pixel | Kernel morphological / Morphological kernel |
| `max_pixels` | 1e9 | Số pixel tối đa / Maximum pixels |

---

## 📊 Kết Quả / Results

### Ví Dụ Output / Example Output

```
✓ Final flood area (ensemble): 45.67 km²

Flood area by method (after cleaning):
  adaptive_landcover       :    80.23 km²
  ems                      :    55.83 km²
  kmeans                   :    92.47 km²
  adaptive_meanstd         :    93.85 km²
  ensemble                 :    93.77 km²

Overall confidence: 59.5%
Reliability: VERY HIGH

District Impact:
  Tổng số districts được phân tích: 21
  Số districts bị ngập > 0.1 ha: 15
```

### Files Được Tạo / Generated Files

```
📁 Output Files
├── validation_report.csv          # Validation metrics
├── flood_impact_<date>.csv        # District impact report
├── confidence_map.tif             # Confidence raster (if enabled)
└── flood_mask_ensemble.tif        # Final flood mask (if enabled)
```

### Định Dạng CSV / CSV Format

**flood_impact_<date>.csv:**

| Column | Type | Description |
|--------|------|-------------|
| district_name | str | Tên huyện / District name |
| province_name | str | Tên tỉnh / Province name |
| total_area_ha | float | Diện tích tổng (ha) / Total area (ha) |
| flood_area_ha | float | Diện tích ngập (ha) / Flooded area (ha) |
| flood_ratio_percent | float | Tỷ lệ ngập (%) / Flood ratio (%) |
| crop_flooded_ha | float | Đất nông nghiệp ngập (ha) / Cropland flooded (ha) |
| exposed_population | int | Dân số ảnh hưởng / Exposed population |
| lat | float | Vĩ độ / Latitude |
| lon | float | Kinh độ / Longitude |

---

## 📚 Tài Liệu / Documentation

### Chi Tiết Kỹ Thuật / Technical Details

Xem file [DOCUMENTATION.md](DOCUMENTATION.md) để biết:
- API documentation đầy đủ
- Mô tả chi tiết các phương pháp
- Ví dụ sử dụng
- Error handling

*See [DOCUMENTATION.md](DOCUMENTATION.md) for:*
- *Complete API documentation*
- *Detailed method descriptions*
- *Usage examples*
- *Error handling*

### Phương Pháp Phát Hiện / Detection Methods

#### 1. EMS Conservative
- Ngưỡng cố định: -18 dB / Fixed threshold: -18 dB
- Phù hợp cho vùng đô thị / Suitable for urban areas

#### 2. Adaptive Landcover
- Điều chỉnh theo địa hình / Terrain-adjusted
- Xem xét độ dốc và độ cao / Considers slope and elevation

#### 3. K-means Clustering
- Unsupervised classification
- 20 training samples

#### 4. Adaptive Mean-Std
- Thống kê từ baseline / Statistical from baseline
- Threshold = Mean - K×Std

#### 5. Change Detection
- So sánh event vs baseline / Event vs baseline comparison
- Otsu thresholding

### Ensemble Method

```
vote_sum = Σ(all methods)
ensemble = vote_sum ≥ 3  # Ít nhất 3/5 phương pháp đồng ý
                          # At least 3/5 methods agree
```

---

## 🐛 Xử Lý Lỗi / Troubleshooting

### Lỗi Thường Gặp / Common Errors

#### 1. "Too many concurrent aggregations"

**Nguyên nhân / Cause:** Quá nhiều reduceRegion cùng lúc

**Giải pháp / Solution:**
```python
# Sử dụng sequential processing
results = process_districts_enhanced(districts, batch_size=1)
```

#### 2. "Computation timeout"

**Nguyên nhân / Cause:** Vùng quá lớn

**Giải pháp / Solution:**
```python
# Tăng scale hoặc giảm vùng
config.scale_large_area = 50  # Tăng từ 30 lên 50
```

#### 3. "Multi-band mask error"

**Nguyên nhân / Cause:** Mask có >1 band

**Giải pháp / Solution:**
```python
# Đảm bảo single-band
mask = mask.select([0])
```

---

## 🤝 Đóng Góp / Contributing

Chúng tôi hoan nghênh mọi đóng góp! / We welcome contributions!

### Quy Trình / Process

1. Fork repository
2. Tạo branch mới / Create feature branch
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. Commit changes
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. Push to branch
   ```bash
   git push origin feature/AmazingFeature
   ```
5. Mở Pull Request / Open Pull Request

---

## 📝 Trích Dẫn / Citation

Nếu sử dụng code này trong nghiên cứu, vui lòng trích dẫn:

*If you use this code in your research, please cite:*

```bibtex
@software{flood_detection_gee_2025,
  author = {Your Name},
  title = {Flood Detection System using Google Earth Engine},
  year = {2025},
  url = {https://github.com/yourusername/flood-detection-gee}
}
```

---

## 📄 Giấy Phép / License

MIT License - Xem file [LICENSE](LICENSE) để biết chi tiết

*MIT License - See [LICENSE](LICENSE) file for details*

---

## 👥 Tác Giả / Authors

- **Your Name** - *Initial work* - [GitHub](https://github.com/yourusername)

---

## 🙏 Cảm Ơn / Acknowledgments

- Google Earth Engine team
- ESA Sentinel mission
- FAO GAUL administrative boundaries
- ESA WorldCover project
- WorldPop project

---

## 📧 Liên Hệ / Contact

Có câu hỏi? Liên hệ qua / Questions? Contact via:
- Email: your.email@example.com
- Issues: [GitHub Issues](https://github.com/yourusername/flood-detection-gee/issues)

---

## 🔄 Lịch Sử Phiên Bản / Version History

### v2.0 (2025-12-11)
- ✨ Thêm documentation đầy đủ / Added comprehensive documentation
- 🐛 Sửa lỗi multi-band mask / Fixed multi-band mask errors
- ⚡ Cải thiện xử lý districts / Improved district processing
- 🛡️ Tăng cường error handling / Enhanced error handling

### v1.0 (2025-11-09)
- 🎉 Phiên bản đầu tiên / Initial release
- 🌊 5 phương pháp ensemble / 5-method ensemble approach
- ✅ Validation cơ bản / Basic validation

---

## 📈 Hiệu Suất / Performance

### Thời Gian Xử Lý / Processing Time

| Vùng / Area | Số Huyện / Districts | Thời Gian / Time |
|-------------|---------------------|------------------|
| 5,540 km² | 21 | ~15-20 phút / min |
| 10,000 km² | 40 | ~30-40 phút / min |
| 20,000 km² | 80 | ~1-1.5 giờ / hours |

**Lưu ý:** Thời gian phụ thuộc vào số ảnh Sentinel-1 và tốc độ mạng.

*Note: Time depends on number of Sentinel-1 images and network speed.*

---

## 🔗 Tài Nguyên / Resources

- [Earth Engine Documentation](https://developers.google.com/earth-engine)
- [Sentinel-1 User Guide](https://sentinel.esa.int/web/sentinel/user-guides/sentinel-1-sar)
- [SAR Handbook](https://servirglobal.net/Global/Articles/Article/2674/sar-handbook-comprehensive-methodologies-for-forest-monitoring-and-biomass-estimation)
- [GEE Tutorial](https://developers.google.com/earth-engine/tutorials)

---

**⭐ Nếu thấy hữu ích, hãy cho repo một star! / If you find this useful, please star the repo!**
