import ee
import geemap
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv
import logging
import argparse
from typing import Dict, List, Tuple, Optional, Any
import ipywidgets as widgets
from IPython.display import display

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Cấu hình tham số (có thể cấu hình từ command line hoặc widgets)
class Config:
    def __init__(self):
        self.country = "Viet Nam"
        self.provinces = ["Thua Thien - Hue", "Da Nang City"]
        self.start_date = "2025-09-01"
        self.end_date = "2025-11-30"
        self.ems_threshold = -18.0  # dB
        self.adaptive_k = 1.0
        self.baseline_days = 60
        self.baseline_vv_threshold = -8.5
        self.hand_threshold = 20  # meters
        self.slope_threshold = 15  # degrees
        self.kernel_size = 1  # pixels for morphological filtering
        self.confidence_thresholds = [80, 60, 40, 20]
        self.ensemble_vote_threshold = 3  # out of 5 methods

config = Config()

# --- Utility Functions ---

def calculate_water_area(mask: ee.Image, geometry: ee.Geometry, scale: int = 10) -> Tuple[float, float]:
    """
    Tính toán diện tích nước từ mask.

    Args:
        mask (ee.Image): Mask nước (1 = nước, 0 = không).
        geometry (ee.Geometry): Vùng tính toán.
        scale (int): Độ phân giải pixel (mặc định 10m).

    Returns:
        Tuple[float, float]: (diện tích m², diện tích km²)
    """
    try:
        pixel_area = ee.Image.pixelArea()
        area_img = mask.selfMask().multiply(pixel_area)
        area_result = area_img.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=geometry,
            scale=scale,
            maxPixels=1e13
        )
        area_m2 = ee.Number(area_result.values().get(0)).getInfo() if area_result.values().size().getInfo() > 0 else 0.0
        area_km2 = area_m2 / 1e6
        return area_m2, area_km2
    except Exception as e:
        logger.warning(f"Lỗi tính diện tích: {e}")
        return 0.0, 0.0

def safe_reduce_region(image: ee.Image, geometry: ee.Geometry, reducer: ee.Reducer, scale: int = 10) -> Dict[str, Any]:
    """
    Thực hiện reduceRegion an toàn với xử lý lỗi.

    Args:
        image (ee.Image): Ảnh GEE.
        geometry (ee.Geometry): Vùng.
        reducer (ee.Reducer): Reducer.
        scale (int): Scale.

    Returns:
        Dict[str, Any]: Kết quả reduceRegion hoặc dict rỗng nếu lỗi.
    """
    try:
        result = image.reduceRegion(
            reducer=reducer,
            geometry=geometry,
            scale=scale,
            maxPixels=1e13
        ).getInfo()
        return result if result else {}
    except Exception as e:
        logger.warning(f"reduceRegion thất bại: {e}")
        return {}

def get_region(country: str, province_list: List[str]) -> ee.Geometry:
    """
    Lấy hình học của các tỉnh thuộc một quốc gia.

    Args:
        country (str): Tên quốc gia.
        province_list (List[str]): Danh sách tỉnh.

    Returns:
        ee.Geometry: Hình học vùng.
    """
    adm1 = ee.FeatureCollection("FAO/GAUL/2015/level1")
    fc_country = adm1.filter(ee.Filter.eq("ADM0_NAME", country))
    features = [
        fc_country.filter(ee.Filter.eq("ADM1_NAME", p)).first()
        for p in province_list
    ]
    return ee.FeatureCollection(features).union().geometry()

def adjust_threshold_by_terrain(threshold: float, slope: ee.Image, dem: ee.Image, geometry: ee.Geometry) -> float:
    """
    Điều chỉnh ngưỡng theo địa hình (slope và elevation).

    Args:
        threshold (float): Ngưỡng gốc.
        slope (ee.Image): Ảnh slope.
        dem (ee.Image): Ảnh DEM.
        geometry (ee.Geometry): Vùng.

    Returns:
        float: Ngưỡng điều chỉnh.
    """
    # Tính trung bình slope và elevation
    stats = safe_reduce_region(
        slope.addBands(dem),
        geometry,
        ee.Reducer.mean().combine(ee.Reducer.stdDev(), '', True),
        scale=30
    )
    mean_slope = stats.get('slope_mean', 0)
    std_slope = stats.get('slope_stdDev', 0)
    mean_elev = stats.get('elevation_mean', 0)

    # Điều chỉnh: Ngưỡng chặt hơn ở vùng núi cao
    adjustment = (mean_slope / 10) + (mean_elev / 1000)  # Ví dụ đơn giản
    adjusted_threshold = threshold - adjustment
    logger.info(f"Ngưỡng gốc: {threshold}, điều chỉnh: {adjusted_threshold} (slope: {mean_slope:.1f}, elev: {mean_elev:.1f})")
    return adjusted_threshold

def validate_sar_conditions(image: ee.Image, geometry: ee.Geometry) -> bool:
    """
    Kiểm tra điều kiện ảnh SAR (ví dụ: wind speed nếu có).

    Args:
        image (ee.Image): Ảnh SAR.
        geometry (ee.Geometry): Vùng.

    Returns:
        bool: True nếu hợp lệ.
    """
    # Giả sử kiểm tra wind speed (Sentinel-1 có metadata này)
    wind_speed = image.get('windSpeed')  # Nếu có
    if wind_speed:
        wind_val = ee.Number(wind_speed).getInfo()
        if wind_val > 10:  # Ví dụ ngưỡng gió cao
            logger.warning(f"Ảnh có gió mạnh: {wind_val} m/s, có thể ảnh hưởng chất lượng")
            return False
    return True

def cross_validate_with_ground_truth(predicted_mask: ee.Image, ground_truth_mask: ee.Image, geometry: ee.Geometry) -> Dict[str, float]:
    """
    So sánh predicted mask với ground truth để tính các metrics.

    Args:
        predicted_mask (ee.Image): Mask dự đoán.
        ground_truth_mask (ee.Image): Mask ground truth.
        geometry (ee.Geometry): Vùng.

    Returns:
        Dict[str, float]: IoU, precision, recall.
    """
    try:
        # Intersection
        intersection = predicted_mask.And(ground_truth_mask).selfMask()
        inter_area, _ = calculate_water_area(intersection, geometry)

        # Union
        union = predicted_mask.Or(ground_truth_mask).selfMask()
        union_area, _ = calculate_water_area(union, geometry)

        # Predicted positive
        pred_pos_area, _ = calculate_water_area(predicted_mask.selfMask(), geometry)

        # True positive (intersection)
        tp = inter_area

        # False positive
        fp = pred_pos_area - tp

        # False negative (ground truth - intersection)
        gt_area, _ = calculate_water_area(ground_truth_mask.selfMask(), geometry)
        fn = gt_area - tp

        # Metrics
        iou = tp / union_area if union_area > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        logger.info(f"Cross-validation: IoU={iou:.3f}, Precision={precision:.3f}, Recall={recall:.3f}")
        return {'iou': iou, 'precision': precision, 'recall': recall}
    except Exception as e:
        logger.warning(f"Lỗi cross-validation: {e}")
        return {'iou': 0, 'precision': 0, 'recall': 0}

def process_districts_enhanced(affected_districts: ee.FeatureCollection, max_iterations: int = 100) -> List[Dict[str, Any]]:
    """
    Xử lý danh sách districts từ server-side list, tránh lỗi concurrent aggregations.

    Args:
        affected_districts (ee.FeatureCollection): FeatureCollection đã filter.
        max_iterations (int): Số lần lặp tối đa.

    Returns:
        List[Dict[str, Any]]: Danh sách properties của các features.
    """
    ee_server_side_list = affected_districts.toList(max_iterations)
    results_list = []

    for i in range(max_iterations):
        try:
            single_ee_feature = ee.Feature(ee_server_side_list.get(i))
            single_feature_info = single_ee_feature.getInfo()

            if single_feature_info and 'properties' in single_feature_info:
                results_list.append(single_feature_info['properties'])
            else:
                continue
        except ee.EEException as e:
            logger.warning(f"EEException at index {i}: {e}")
            continue
        except Exception as e:
            logger.warning(f"Unexpected exception at index {i}: {e}")
            continue

    return results_list

# --- Khởi tạo Earth Engine ---
# Chỉ yêu cầu Authenticate nếu chưa đăng nhập trước đó trong Colab
try:
    ee.Initialize(project='driven-torus-431807-u3')
except Exception:
    ee.Authenticate()
    ee.Initialize(project='driven-torus-431807-u3')

# --- Thông tin kiểm tra ---
print("Earth Engine version:", ee.__version__)
print("Chào mừng đến với Google Earth Engine trên Colab!")
print(ee.String('Hello from Google Earth Engine!').getInfo())

# --- Widgets cho nhập tham số (Jupyter) ---
def create_input_widgets():
    """Tạo widgets để nhập tham số."""
    country_widget = widgets.Text(value=config.country, description='Quốc gia:')
    provinces_widget = widgets.Text(value=', '.join(config.provinces), description='Tỉnh (cách nhau bởi dấu phẩy):')
    start_date_widget = widgets.DatePicker(value=datetime.datetime.strptime(config.start_date, '%Y-%m-%d').date(), description='Ngày bắt đầu:')
    end_date_widget = widgets.DatePicker(value=datetime.datetime.strptime(config.end_date, '%Y-%m-%d').date(), description='Ngày kết thúc:')
    ems_threshold_widget = widgets.FloatText(value=config.ems_threshold, description='Ngưỡng EMS (dB):')
    adaptive_k_widget = widgets.FloatText(value=config.adaptive_k, description='Hệ số k cho Adaptive:')

    button = widgets.Button(description='Cập nhật cấu hình')
    output = widgets.Output()

    def update_config(b):
        with output:
            output.clear_output()
            config.country = country_widget.value
            config.provinces = [p.strip() for p in provinces_widget.value.split(',')]
            config.start_date = start_date_widget.value.strftime('%Y-%m-%d') if start_date_widget.value else config.start_date
            config.end_date = end_date_widget.value.strftime('%Y-%m-%d') if end_date_widget.value else config.end_date
            config.ems_threshold = ems_threshold_widget.value
            config.adaptive_k = adaptive_k_widget.value
            print("Cấu hình đã cập nhật!")

    button.on_click(update_config)
    display(country_widget, provinces_widget, start_date_widget, end_date_widget, ems_threshold_widget, adaptive_k_widget, button, output)

# Hiển thị widgets nếu trong Jupyter
try:
    create_input_widgets()
except:
    logger.info("Không thể hiển thị widgets (không phải Jupyter)")

# --- Thiết lập vùng và thời gian từ config ---
geometry = get_region(config.country, config.provinces)
start, end = config.start_date, config.end_date

# Thêm lên bản đồ
m = geemap.Map(center=[16.2, 107.8], zoom=8)
m.add_layer(geometry, {"color": "red"}, "Boundary")
m

"""## 2. Tải ảnh từ Sentinel 1

"""

def load_s1(platform: str, geom: ee.Geometry, start: str, end: str) -> Tuple[ee.ImageCollection, int, List[str]]:
    """
    Tải ảnh Sentinel-1 GRD theo platform.

    Args:
        platform (str): Platform (A, B, C, D).
        geom (ee.Geometry): Vùng.
        start (str): Ngày bắt đầu (YYYY-MM-DD).
        end (str): Ngày kết thúc (YYYY-MM-DD).

    Returns:
        Tuple[ee.ImageCollection, int, List[str]]: (collection, count, ids)
    """
    col = (ee.ImageCollection("COPERNICUS/S1_GRD")
           .filterBounds(geom)
           .filterDate(start, end)
           .filter(ee.Filter.eq("platform_number", platform))
           .filter(ee.Filter.eq("instrumentMode", "IW")))

    count = col.size().getInfo()
    ids = col.aggregate_array("system:index").getInfo() if count > 0 else []
    return col, count, ids

print("ĐANG TẢI ẢNH SENTINEL-1...")

for platform in ["A", "B", "C", "D"]:
    col, count, ids = load_s1(platform, geometry, start, end)
    print(f"\nSentinel-1{platform}: {count} ảnh")
    for img in ids:
        print("   ", img)

"""## 3. Tải ảnh từ Sentinel 2"""

def load_s2(geom: ee.Geometry, start: str, end: str, cloud: float = 20, bands: Optional[List[str]] = None) -> Tuple[ee.ImageCollection, int, List[str]]:
    """
    Tải ảnh Sentinel-2 SR.

    Args:
        geom (ee.Geometry): Vùng.
        start (str): Ngày bắt đầu.
        end (str): Ngày kết thúc.
        cloud (float): Ngưỡng mây (%).
        bands (Optional[List[str]]): Danh sách bands.

    Returns:
        Tuple[ee.ImageCollection, int, List[str]]: (collection, count, ids)
    """
    bands = bands or ["B2", "B3", "B4", "B8", "B8A", "B11", "B12"]
    col = (ee.ImageCollection("COPERNICUS/S2_SR")
           .filterBounds(geom)
           .filterDate(start, end)
           .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", cloud))
           .select(bands))

    count = col.size().getInfo()
    ids = col.aggregate_array("system:index").getInfo() if count > 0 else []
    return col, count, ids

s2_collection, s2_count, s2_ids = load_s2(geometry, start, end, cloud=10)
print(f"\nSentinel-2: {s2_count} ảnh")
for img in s2_ids:
    print("   ", img)

"""## 4. Tải DEM SRTM và tính slope từ DEM"""

dem = ee.Image("USGS/SRTMGL1_003").clip(geometry)
slope = ee.Terrain.slope(dem)

m = geemap.Map()
m.centerObject(geometry, 10)
m.add_layer(dem, {"min": 0, "max": 1000}, "DEM")
m.add_layer(slope, {"min": 0, "max": 30, "palette": ["green", "yellow", "red"]}, "Slope")
m

"""## 5. Tải dữ liệu Permanent Water"""

jrc = ee.Image("JRC/GSW1_3/GlobalSurfaceWater")
permanent_water = jrc.select("occurrence").gt(95).clip(geometry)

m = geemap.Map()
m.set_center(107.75, 16.25, 9)
m.add_layer(permanent_water, {"palette": ["blue"]}, "Permanent Water")
m

"""## 6. Tính diện tích nước"""

pixel_area = ee.Image.pixelArea()

total_area = geometry.area().getInfo()
water_area_img = permanent_water.multiply(pixel_area)

water_area = water_area_img.reduceRegion(
    reducer=ee.Reducer.sum(),
    geometry=geometry,
    scale=30,
    maxPixels=1e13
).get("occurrence").getInfo()

# km2
total_area_km2 = total_area / 1e6
water_area_km2 = water_area / 1e6
water_percent = (water_area / total_area) * 100

print(f"Diện tích vùng: {total_area_km2:.2f} km²")
print(f"Diện tích nước thường xuyên: {water_area_km2:.2f} km²")
print(f"Tỷ lệ nước thường xuyên: {water_percent:.2f} %")

"""# Thống kê và phân tích dữ liệu

## 1. Thống kê số ảnh theo tháng
"""

def count_images_per_month(collection):
    """Đếm số ảnh theo từng tháng (1–12)."""
    months = ee.List.sequence(1, 12)

    def per_month(m):
        m = ee.Number(m)
        count = collection.filter(
            ee.Filter.calendarRange(m, m, "month")
        ).size()
        return ee.Feature(None, {"month": m, "count": count})

    return ee.FeatureCollection(months.map(per_month))

s1A_collection, s1A_count, s1A_ids = load_s1("A", geometry, start, end)
s1C_collection, s1C_count, s1C_ids = load_s1("C", geometry, start, end)

s1A_monthly = count_images_per_month(s1A_collection)
s1C_monthly = count_images_per_month(s1C_collection)

def print_section(title, data):
    print(f"{title}".upper())
    print(data.getInfo())

print_section("SỐ ẢNH THEO THÁNG (S1A)", s1A_monthly)
print_section("SỐ ẢNH THEO THÁNG (S1C)", s1C_monthly)

"""## 2. Thống kê Orbit"""

def print_orbit_stats(collection, label):
    """In thống kê quỹ đạo Sentinel-1."""
    ids = collection.aggregate_array("system:index").getInfo()
    orbits = collection.aggregate_array("relativeOrbitNumber_start").getInfo()

    print(f"\n--- ORBIT STATS {label} ---")
    for image_id, orbit in zip(ids, orbits):
        print(f"{image_id} | Orbit = {orbit}")


# print_orbit_stats(s1A_collection, "S1A")
# print_orbit_stats(s1C_collection, "S1C")

"""## 3. Tính chất lượng ảnh S1"""

def compute_image_quality(img, geom):
    """Tính Coverage (%), mean VV, std VV và góc Incidence."""
    footprint = img.geometry()
    inter = footprint.intersection(geom, 1)

    # Coverage %
    coverage = (
        inter.area(1)
        .divide(geom.area(1))
        .multiply(100)
        .getInfo()
    )

    # Mean VV
    stats = img.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geom,
        scale=10,
        maxPixels=5e8
    )
    mean_vv = stats.get("VV")
    mean_vv = ee.Number(mean_vv).getInfo() if mean_vv else None

    # Std VV
    stats_std = img.reduceRegion(
        reducer=ee.Reducer.stdDev(),
        geometry=geom,
        scale=10,
        maxPixels=5e8
    )
    std_vv = stats_std.get("VV")
    std_vv = ee.Number(std_vv).getInfo() if std_vv else None

    # Incidence angle
    incidence = img.get("incidentAngle").getInfo()

    return coverage, mean_vv, std_vv, incidence

def export_quality_to_csv(collection, label, geom, out_csv):
    """Xuất kết quả chất lượng ảnh S1 ra file CSV."""
    num = collection.size().getInfo()
    images = collection.toList(num)
    orbits = collection.aggregate_array("relativeOrbitNumber_start").getInfo()

    rows = []

    for i in range(num):
        img = ee.Image(images.get(i))
        img_id = img.get("system:index").getInfo()

        coverage, mean_vv, std_vv, incidence = compute_image_quality(img, geom)

        rows.append({
            "label": label,
            "image_id": img_id,
            "coverage_percent": coverage,
            "mean_vv": mean_vv,
            "std_vv": std_vv,
            "incidence_angle": incidence,
            "orbit": orbits[i]
        })

    print(f"Ghi CSV: {out_csv} ({len(rows)} dòng)")

    # Ghi ra file CSV
    with open(out_csv, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("Xuất CSV xong.")

# export_quality_to_csv(s1A_collection, "S1A", geometry, "s1A_quality.csv")
# export_quality_to_csv(s1C_collection, "S1C", geometry, "s1C_quality.csv")

"""## 4. Histogram/Backscatter signature"""

def attach_histogram(img):
    """Gắn histogram 50 bins cho band VV vào metadata."""
    hist = img.reduceRegion(
        reducer=ee.Reducer.histogram(50),
        geometry=geometry,
        scale=10,
        maxPixels=1e10
    )
    return img.set("hist", hist.get("VV"))


def export_histogram_to_csv(collection, filename):
    num = collection.size().getInfo()
    imgs = collection.toList(num)

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "bin_counts", "bin_means"])

        for i in range(num):
            img = ee.Image(imgs.get(i))
            img_id = img.get("system:index").getInfo()
            hist = img.get("hist").getInfo()

            if hist:
                writer.writerow([
                    img_id,
                    hist["histogram"],
                    hist["bucketMeans"]
                ])
            else:
                writer.writerow([img_id, None, None])

s1A_with_hist = s1A_collection.map(attach_histogram)
s1C_with_hist = s1C_collection.map(attach_histogram)

# export_histogram_to_csv(s1A_with_hist, "s1A_hist.csv")
# export_histogram_to_csv(s1C_with_hist, "s1C_hist.csv")
# print("Đã lưu xong CSV!")

"""## Phân tích dữ liệu"""

# s1A_df = pd.read_csv("/content/s1A_quality.csv")
# s1C_df = pd.read_csv("/content/s1C_quality.csv")

# s1_df = pd.concat([s1A_df, s1C_df], ignore_index=True)
# s1_df.head()

"""# So sánh các phương pháp khoanh vùng lũ lụt

## Chuẩn bị dữ liệu
"""

# Gộp S1A + S1C
s1_collection = s1A_collection.merge(s1C_collection)
print(f"S1: {s1_collection.size().getInfo()} images")

# Thêm thuộc tính COVERAGE (%)
def add_quality_properties(image):
    footprint = image.geometry()
    inter = footprint.intersection(geometry, 1)
    coverage = (
        inter.area(1)
        .divide(geometry.area(1))
        .multiply(100)
    )
    return image.set({
        'COVERAGE': ee.Algorithms.If(coverage, ee.Number(coverage), 0)
    })

s1_collection_with_props = s1_collection.map(add_quality_properties)

# Mosaics theo ngày
def mosaic_by_day(date_string):
    start = ee.Date(date_string)
    end = start.advance(1, 'day')
    same_day = s1_collection_with_props.filterDate(start, end)

    mosaic_image = ee.Algorithms.If(
        same_day.size().gt(0),
        ee.Image(same_day.min()).clip(geometry),
        ee.Image().set({
            'COVERAGE': 0,
            'DATE': start.format('YYYY-MM-dd')
        })
    )

    mosaic_image = ee.Image(mosaic_image)

    # Tính coverage lại (chính xác hơn vì mosaic đã clip)
    footprint = mosaic_image.geometry()
    inter = footprint.intersection(geometry, 1)
    coverage = (
        inter.area(1)
        .divide(geometry.area(1))
        .multiply(100)
    )

    return mosaic_image.set({
        'COVERAGE': coverage,
        'DATE': start.format('YYYY-MM-dd')
    })


# Danh sách ngày duy nhất
dates = (
    s1_collection_with_props
    .aggregate_array('system:time_start')
    .map(lambda t: ee.Date(t).format('YYYY-MM-dd'))
    .distinct()
)

daily_mosaic = ee.ImageCollection.fromImages(
    dates.map(mosaic_by_day)
)

# Lọc ảnh có coverage > 50%
filtered = daily_mosaic.filter(ee.Filter.gt('COVERAGE', 50))

total_scenes = filtered.size().getInfo()
print(f"Total scenes (coverage > 50%): {total_scenes}")

# Lấy danh sách ảnh
image_list = filtered.toList(total_scenes)
num_images = image_list.size().getInfo()
print(f"Summary {num_images} images")

"""## Pre-processing"""

# Thêm trường VV_mean (server-side)
def add_vv_mean(image):
    mean_vv = image.select('VV').reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geometry,
        scale=10,
        maxPixels=1e13
    ).get('VV')
    return image.set('VV_mean', mean_vv)

filtered_with_mean = filtered.map(add_vv_mean)

# Ảnh có VV_mean thấp nhất (sự kiện lũ)
min_vv_image = filtered_with_mean.sort('VV_mean').first()

event_date = min_vv_image.get('DATE').getInfo()
print(f"Event image date (lowest VV): {event_date}")

# Load ảnh sự kiện
event_image = min_vv_image.select(['VV', 'VH']).clip(geometry)

# Kiểm tra ảnh đã ở dB chưa
stats = event_image.select('VV').reduceRegion(
    reducer=ee.Reducer.mean(),
    geometry=geometry,
    scale=10,
    maxPixels=1e13
)

mean_vv_val = ee.Number(stats.get('VV')).getInfo()

if mean_vv_val is None:
    raise ValueError("Cannot determine VV value — check geometry or mask")
elif mean_vv_val > 1:
    print("Converting to dB (10 * log10(sigma0))...")
    event_image = event_image.log10().multiply(10)
    print("Converted to dB")
else:
    print("Already in dB")

# Làm sạch tín hiệu radar
# Giới hạn dB để loại noise
event_image = event_image.clamp(-30, 0)
print("Clipped to [-30, 0] dB range (thermal noise removal)")

# Speckle filtering
event_image_filtered = event_image.focal_median(2.5, 'circle', 'meters')
print("Speckle filter applied")

"""## Tạo baseline/reference"""

# Thêm VV_mean vào từng ảnh (server-side)
def add_vv_mean(image):
    mean_vv = image.select('VV').reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geometry,
        scale=10,
        maxPixels=1e13
    ).get('VV')
    return image.set('VV_mean', mean_vv)

filtered_with_mean = filtered.map(add_vv_mean)

# Xác định ngày sự kiện
event_date_ee = ee.Date(min_vv_image.get('DATE'))

# Khoảng thời gian baseline: 60 ngày trước event
start_date_ee = event_date_ee.advance(-60, 'day')

# Lọc ảnh baseline
# Điều kiện:
#   1. Thuộc 30 ngày trước event
#   2. VV_mean < -8.5 dB (loại bỏ ảnh kém chất lượng)
baseline_collection = filtered_with_mean.filter(
    ee.Filter.And(
        ee.Filter.gte('DATE', start_date_ee.format('YYYY-MM-dd')),
        ee.Filter.lt('DATE', event_date_ee.format('YYYY-MM-dd')),
        ee.Filter.lt('VV_mean', -8.5)
    )
)

baseline_count = baseline_collection.size().getInfo()
print(f"Baseline images count: {baseline_count}")

# Tạo baseline image
# Nếu không có ảnh nào đạt điều kiện → fallback = event_image_filtered
if baseline_count > 0:
    baseline_image = (
        baseline_collection
        .select(['VV', 'VH'])
        .median()
        .clip(geometry)
        .clamp(-30, 0)
        .focal_median(2.5, 'circle', 'meters')
    )
    print(f"Baseline tạo từ {baseline_count} ảnh")
else:
    logger.warning("Không có ảnh baseline thỏa điều kiện, sử dụng event image làm reference.")
    baseline_image = event_image_filtered

"""## Chuẩn bị masks"""

# 1. Slope mask (giữ nguyên)
slope_mask = slope.lt(15)

# 2. Permanent water mask (giữ nguyên)
permanent_water_mask = permanent_water.Not()

# 3. HAND Mask (Mới)
# Sử dụng dataset MERIT Hydro (độ chính xác cao toàn cầu)
merit_hydro = ee.Image("MERIT/Hydro/v1_0_1")
hand = merit_hydro.select('hnd') # hnd = Height Above Nearest Drainage

# Chỉ giữ lại những vùng thấp hơn 15m so với dòng chảy gần nhất
# (Có thể điều chỉnh thành 20m nếu lũ cực lớn)
hand_threshold = 20
hand_mask = hand.lt(hand_threshold).clip(geometry)

print("Slope mask: loại bỏ vùng dốc > 15°")
print("Permanent water mask: loại bỏ nước thường trực")
print(f"HAND mask: loại bỏ vùng cao hơn {hand_threshold}m so với sông suối (loại nhiễu vùng núi)")

# 4. Kết hợp tất cả masks
combined_mask = slope_mask.And(permanent_water_mask).And(hand_mask)

# Tính diện tích vùng khả thi (Analysis Domain)
masked = combined_mask.selfMask()
total_pixels = geometry.area(1).divide(100).getInfo() # Tổng pixel ROI
valid_count = masked.reduceRegion(
    reducer=ee.Reducer.count(),
    geometry=geometry,
    scale=10,
    maxPixels=1e13
).get(combined_mask.bandNames().get(0)).getInfo()

coverage_pct = (valid_count / total_pixels) * 100 if valid_count else 0
print(f"Combined mask coverage: {coverage_pct:.1f}% of ROI")

"""## Water detection"""

vv_stats = event_image_filtered.updateMask(combined_mask).reduceRegion(
    reducer = ee.Reducer.minMax()
                .combine(ee.Reducer.mean(), '', True)
                .combine(ee.Reducer.stdDev(), '', True),
    geometry = geometry,
    scale = 10,
    maxPixels = 1e13
)

def safe_get(d, key):
    val = d.get(key)
    return ee.Number(val) if val else ee.Number(float('nan'))

vv_min  = safe_get(vv_stats, 'VV_min').getInfo()
vv_max  = safe_get(vv_stats, 'VV_max').getInfo()
vv_mean = safe_get(vv_stats, 'VV_mean').getInfo()

# stdDev có thể là 'VV_stdDev' hoặc 'VV_std'
vv_std_key = 'VV_stdDev' if vv_stats.get('VV_stdDev').getInfo() is not None else 'VV_std'
vv_std  = safe_get(vv_stats, vv_std_key).getInfo()

print("EVENT IMAGE STATISTICS:")
print(f"   Min: {vv_min:.2f} dB | Max: {vv_max:.2f} dB")
print(f"   Mean: {vv_mean:.2f} dB | Std: {vv_std:.2f} dB")

# Khởi tạo pixel area cho tính toán diện tích
pixel_area = ee.Image.pixelArea()

# Dictionary để lưu các water mask
water_masks = {}

"""### 1. Adaptive Fixed Thresholds (by land type)"""

worldcover = ee.Image("ESA/WorldCover/v200/2021")

# Gộp thành các loại:
# 1 = Urban
# 2 = Rural (cropland + grassland)
# 3 = Coastal (wetlands + mangroves + water edge)
# 4 = Mountain (bare + steep terrain from DEM)

urban = worldcover.eq(50)   # Built-up
rural = worldcover.eq(40).Or(worldcover.eq(30))
coastal = (
    worldcover.eq(80)   # wetlands
    .Or(worldcover.eq(90))     # mangroves
)
slope = ee.Terrain.slope(dem)
mountain = slope.gt(15)  # độ dốc > 15° coi như vùng núi
landcover_class = (
    urban.multiply(1)
    .add(rural.multiply(2))
    .add(coastal.multiply(3))
    .add(mountain.multiply(4))
)
urban_mask = landcover_class.eq(1)
rural_mask = landcover_class.eq(2)
coastal_mask = landcover_class.eq(3)
mountain_mask = landcover_class.eq(4)

urban_water    = event_image_filtered.lt(-14).And(urban_mask)
rural_water    = event_image_filtered.lt(-17).And(rural_mask)
coastal_water  = event_image_filtered.lt(-15).And(coastal_mask)
mountain_water = event_image_filtered.lt(-19).And(mountain_mask)

water_mask_adaptive = (
    urban_water
    .Or(rural_water)
    .Or(coastal_water)
    .Or(mountain_water)
).And(combined_mask)

water_masks['adaptive_landcover'] = water_mask_adaptive

water_area_adaptive = water_mask_adaptive.updateMask(water_mask_adaptive) \
    .multiply(pixel_area) \
    .reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=geometry,
        scale=10,
        maxPixels=1e13
    ) \
    .get('VV')

total_water_area_adaptive = ee.Number(water_area_adaptive).getInfo() if water_area_adaptive else 0

area_km2 = total_water_area_adaptive / 1e6
percent = (total_water_area_adaptive / geometry.area(1).getInfo()) * 100

print(f" Urban     (< -14 dB)")
print(f" Rural     (< -17 dB)")
print(f" Coastal   (< -15 dB)")
print(f" Mountain  (< -19 dB)")
print(f"Diện tích nước: {area_km2:.2f} km²")
print(f"Tỷ lệ ngập: {percent:.2f}%")

"""### 2. EMS Conservative"""

# Điều chỉnh ngưỡng theo địa hình
ems_threshold_adjusted = adjust_threshold_by_terrain(config.ems_threshold, slope, dem, geometry)
water_mask_ems = event_image_filtered.lt(ems_threshold_adjusted).And(combined_mask)
water_masks['ems'] = water_mask_ems

total_water_area_ems, area_km2_ems = calculate_water_area(water_mask_ems, geometry)
percent_ems = (total_water_area_ems / geometry.area(1).getInfo()) * 100

print(f"Ngưỡng: {ems_threshold_adjusted:.2f} dB (điều chỉnh theo địa hình)")
print(f"Diện tích nước: {area_km2_ems:.2f} km²")
print(f"Tỷ lệ ngập: {percent_ems:.2f}%")

"""### 3. K-means Clustering"""

masked_image = event_image_filtered.updateMask(combined_mask)

training = masked_image.sample(
    region=geometry,
    scale=10,
    numPixels=3000,
    seed=42,
    dropNulls=True
)

training_size = training.size().getInfo()
print(f"Training samples: {training_size}")

if training_size > 0:
    clusterer = ee.Clusterer.wekaKMeans(2).train(training)
    class_img = masked_image.cluster(clusterer)

    # Tìm cluster nào là water (VV thấp hơn)
    mean_vv_0 = event_image_filtered.updateMask(class_img.select('cluster').eq(0)).reduceRegion(
        reducer=ee.Reducer.mean(), geometry=geometry, scale=10, maxPixels=1e13
    ).get('VV')
    mean_vv_0_val = ee.Number(mean_vv_0).getInfo() if mean_vv_0 else float('inf')

    mean_vv_1 = event_image_filtered.updateMask(class_img.select('cluster').eq(1)).reduceRegion(
        reducer=ee.Reducer.mean(), geometry=geometry, scale=10, maxPixels=1e13
    ).get('VV')
    mean_vv_1_val = ee.Number(mean_vv_1).getInfo() if mean_vv_1 else float('inf')

    water_label = 1 if mean_vv_1_val < mean_vv_0_val else 0

    print(f"   Cluster 0 mean: {mean_vv_0_val:.2f} dB")
    print(f"   Cluster 1 mean: {mean_vv_1_val:.2f} dB")
    print(f"   Water cluster: {water_label}")

    water_mask_kmeans = class_img.select('cluster').eq(water_label).And(combined_mask)
    water_masks['kmeans'] = water_mask_kmeans

    water_area_kmeans = water_mask_kmeans.updateMask(water_mask_kmeans).multiply(pixel_area).reduceRegion(
        reducer=ee.Reducer.sum(), geometry=geometry, scale=10, maxPixels=1e13
    ).get('cluster')

    total_water_area_kmeans = ee.Number(water_area_kmeans).getInfo() if water_area_kmeans else 0
    area_km2_kmeans = total_water_area_kmeans / 1e6
    percent_kmeans = (total_water_area_kmeans / geometry.area(1).getInfo()) * 100

    print(f"Diện tích nước: {area_km2_kmeans:.2f} km²")
    print(f"Tỷ lệ ngập: {percent_kmeans:.2f}%")
else:
    print("Không có dữ liệu training")
    water_mask_kmeans = ee.Image(0)
    water_masks['kmeans'] = water_mask_kmeans
    total_water_area_kmeans = 0

"""###  4. Adaptive Threshold (Mean-Std)"""

# Tính thống kê VV trên baseline image
baseline_stats = baseline_image.select('VV') \
    .reduceRegion(
        reducer=ee.Reducer.mean().combine(ee.Reducer.stdDev(), '', True),
        geometry=geometry,
        scale=10,
        maxPixels=1e13
    )

baseline_mean = ee.Number(baseline_stats.get('VV_mean'))
baseline_std = ee.Number(baseline_stats.get('VV_stdDev'))

# Hệ số k (tùy chỉnh để tăng/giảm ngưỡng)
k = config.adaptive_k
adaptive_threshold = baseline_mean.subtract(baseline_std.multiply(k))

# Giới hạn threshold trong khoảng [-30, -5] dB
adaptive_threshold = ee.Number(adaptive_threshold).max(-30).min(-5)

# Tạo mask nước: pixel VV < adaptive_threshold và trong combined mask
vv = event_image_filtered.select('VV')
water_mask_adaptive_meanstd = vv.lt(adaptive_threshold).And(combined_mask)
water_masks['adaptive_meanstd'] = water_mask_adaptive_meanstd

# Tính diện tích nước (km²)
total_water_area_adaptive_meanstd, area_km2_adaptive_meanstd = calculate_water_area(water_mask_adaptive_meanstd, geometry)
percent_adaptive_meanstd = (total_water_area_adaptive_meanstd / geometry.area(1).getInfo()) * 100

print(f"Baseline Mean: {baseline_mean.getInfo():.2f} dB")
print(f"Baseline Std: {baseline_std.getInfo():.2f} dB")
print(f"Adaptive Threshold (Mean - {k}*Std) = {adaptive_threshold.getInfo():.2f} dB")
print(f"Diện tích nước: {area_km2_adaptive_meanstd:.2f} km²")
print(f"Tỷ lệ ngập: {percent_adaptive_meanstd:.2f}%")

"""### 5. Change Detection (Log-Ratio + Otsu)"""

# Log-ratio (event / baseline)
log_ratio = baseline_image.select('VV').divide(event_image_filtered.select('VV')).log()
log_ratio_clipped = log_ratio.clamp(-1, 1)

# Histogram server-side
hist_dict = log_ratio_clipped.reduceRegion(
    reducer=ee.Reducer.histogram(50),  # 50 bins
    geometry=geometry,
    scale=10,
    maxPixels=1e13,
    bestEffort=True
).get('VV')

# Otsu threshold function server-side
def otsu_threshold_server(hist_dict):
    hist = ee.Dictionary(hist_dict)
    counts = ee.List(hist.get('histogram'))
    means = ee.List(hist.get('bucketMeans'))

    total = ee.Number(counts.reduce(ee.Reducer.sum()))
    sum_total = ee.Number(
        ee.List.sequence(0, counts.size().subtract(1))
        .map(lambda i: ee.Number(counts.get(i)).multiply(ee.Number(means.get(i))))
        .reduce(ee.Reducer.sum())
    )

    def iterate_fn(i, acc):
        i = ee.Number(i)
        acc = ee.Dictionary(acc)
        sumB = ee.Number(acc.get('sumB'))
        wB = ee.Number(acc.get('wB'))
        maximum = ee.Number(acc.get('maximum'))
        threshold = ee.Number(acc.get('threshold'))

        count_i = ee.Number(counts.get(i))
        mean_i = ee.Number(means.get(i))

        wB_new = wB.add(count_i)
        wF = total.subtract(wB_new)
        sumB_new = sumB.add(count_i.multiply(mean_i))

        mB = sumB_new.divide(wB_new)
        mF = sum_total.subtract(sumB_new).divide(wF)
        between = wB_new.multiply(wF).multiply(mB.subtract(mF).pow(2))

        threshold = ee.Algorithms.If(between.gt(maximum), mean_i, threshold)
        maximum = ee.Algorithms.If(between.gt(maximum), between, maximum)

        return ee.Dictionary({
            'sumB': sumB_new,
            'wB': wB_new,
            'maximum': maximum,
            'threshold': threshold
        })

    init = ee.Dictionary({'sumB': 0, 'wB': 0, 'maximum': 0, 'threshold': 0})
    result = ee.List.sequence(0, counts.size().subtract(1)).iterate(iterate_fn, init)

    return ee.Number(ee.Dictionary(result).get('threshold'))

# Tính threshold
threshold_value = otsu_threshold_server(hist_dict)
print("Otsu Threshold (log-ratio):", threshold_value.getInfo())

# Mask phát hiện thay đổi
water_mask_change = log_ratio_clipped.gt(threshold_value).And(combined_mask)

# Kết hợp adaptive thresholds theo landcover (nếu có)
urban_water    = event_image_filtered.lt(-14).And(urban_mask)
rural_water    = event_image_filtered.lt(-17).And(rural_mask)
coastal_water  = event_image_filtered.lt(-15).And(coastal_mask)
mountain_water = event_image_filtered.lt(-19).And(mountain_mask)

adaptive_mask = water_mask_change.Or(urban_water).Or(rural_water).Or(coastal_water).Or(mountain_water)

# Lọc nhiễu nhỏ
adaptive_mask_clean = adaptive_mask.focal_max(1).focal_min(1)

# Tính diện tích
pixel_area = ee.Image.pixelArea()
water_area_change = adaptive_mask_clean.multiply(pixel_area).reduceRegion(
    reducer=ee.Reducer.sum(),
    geometry=geometry,
    scale=10,
    maxPixels=1e13
).get('VV')

total_water_area_change = ee.Number(water_area_change).getInfo() if water_area_change else 0
area_km2_change = total_water_area_change / 1e6
percent_change = (total_water_area_change / geometry.area(1).getInfo()) * 100

print(f"Diện tích nước: {area_km2_change:.2f} km²")
print(f"Tỷ lệ ngập: {percent_change:.2f}%")

"""### 6. Ensemble Method (Majority Vote >=3 methods)"""

# Tạo ảnh tổng số phương pháp phát hiện nước
vote_sum = ee.Image(0)
for method_name, mask in water_masks.items():
    vote_sum = vote_sum.add(mask)

# Áp dụng majority vote (>=3/5 phương pháp đồng ý)
ensemble_mask = vote_sum.gte(3).And(combined_mask)
water_masks['ensemble'] = ensemble_mask

# Tính diện tích
water_area_ensemble = ensemble_mask.updateMask(ensemble_mask) \
    .multiply(pixel_area) \
    .reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=geometry,
        scale=10,
        maxPixels=1e13
    ).get('VV')

total_water_area_ensemble = ee.Number(water_area_ensemble).getInfo() if water_area_ensemble else 0
area_km2_ensemble = total_water_area_ensemble / 1e6
percent_ensemble = (total_water_area_ensemble / geometry.area(1).getInfo()) * 100

print(f"Số phương pháp đồng ý: >=3/5")
print(f"Diện tích nước: {area_km2_ensemble:.2f} km²")
print(f"Tỷ lệ ngập: {percent_ensemble:.2f}%")

"""### Tổng kết kết quả"""

summary_data = {
    'Method': ['1. Adaptive Landcover', '2. EMS (-18 dB)', '3. K-means',
               '4. Adaptive Mean-Std', '5. Change Detection', '6. Ensemble (≥3)'],
    'Area (km²)': [area_km2, area_km2_ems, area_km2_kmeans,
                   area_km2_adaptive_meanstd, area_km2_change, area_km2_ensemble],
    'Flood (%)': [percent, percent_ems, percent_kmeans,
                  percent_adaptive_meanstd, percent_change, percent_ensemble]
}

for i in range(len(summary_data['Method'])):
    print(f"{summary_data['Method'][i]:<25} {summary_data['Area (km²)'][i]:>10.2f} km² {summary_data['Flood (%)'][i]:>10.2f}%")

"""## Post-processing (Morphological Filtering)

### 1. Morphological Filtering
"""

def clean_water_mask(mask, kernel_size=1):
    """
    Morphological opening/closing để loại bỏ noise
    kernel_size: 1-3 pixels (theo khuyến nghị ESA)
    """
    # Opening: erosion → dilation (loại bỏ vật thể nhỏ)
    opened = mask.focal_min(kernel_size, 'circle', 'pixels') \
                 .focal_max(kernel_size, 'circle', 'pixels')
    # Closing: dilation → erosion (lấp các lỗ nhỏ)
    cleaned = opened.focal_max(kernel_size, 'circle', 'pixels') \
                    .focal_min(kernel_size, 'circle', 'pixels')
    return cleaned

# Áp dụng morphological filtering
kernel_size = 1  # 1-3 pixels

# Clean tất cả các masks
water_masks_clean = {}
for method_name, mask in water_masks.items():
    water_masks_clean[method_name] = clean_water_mask(mask, kernel_size)
    print(f"Cleaned {method_name}")

print(f"\nMorphological filtering completed (kernel: {kernel_size} pixel)")

"""### 2. Recalculate areas after cleaning"""

all_results_clean = {}

for method_name, mask_clean in water_masks_clean.items():
    mask_band = mask_clean.select(0)  # Chỉ lấy band đầu tiên

    area_result = mask_band.updateMask(mask_band) \
        .multiply(pixel_area) \
        .reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=geometry,
            scale=10,
            maxPixels=1e13
        )

    # Lấy giá trị đầu tiên trong dictionary
    area_m2 = ee.Number(area_result.values().get(0)).getInfo() if area_result.values().size().getInfo() > 0 else 0
    all_results_clean[method_name] = area_m2 / 1e6  # Convert to km²

# Tính lại ensemble mask
vote_sum = ee.Image(0)
num_methods = len(water_masks_clean)

for mask in water_masks_clean.values():
    vote_sum = vote_sum.add(mask.select(0))

ensemble_mask_clean = vote_sum.gte(3).And(combined_mask)
water_masks_clean['ensemble'] = ensemble_mask_clean

# Final water mask
water_mask_final = ensemble_mask_clean.select(0)

# Tính diện tích final
water_area_final = water_mask_final.updateMask(water_mask_final) \
    .multiply(pixel_area) \
    .reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=geometry,
        scale=10,
        maxPixels=1e13
    )

total_water_area_final = ee.Number(water_area_final.values().get(0)).getInfo() if water_area_final else 0
area_final_km2 = total_water_area_final / 1e6

print(f"\n✓ Final flood area (ensemble): {area_final_km2:.2f} km²")

# In kết quả từng phương pháp
print("\nFlood area by method (after cleaning):")
for method_name, area_km2 in all_results_clean.items():
    print(f"  {method_name:<25}: {area_km2:>8.2f} km²")

"""### 3. Confidence score"""

# Tính confidence (% methods đồng ý)
confidence = vote_sum.divide(num_methods).multiply(100)

reducer = ee.Reducer.mean().combine(
    ee.Reducer.percentile([0, 25, 50, 75, 100]),
    '',
    True
)

confidence_stats = confidence.reduceRegion(
    reducer=reducer,
    geometry=geometry,
    scale=10,
    maxPixels=1e13
)

# In keys để theo dõi
stats_dict = confidence_stats.getInfo()
print("DEBUG keys:", stats_dict.keys())

# Hàm tự tìm key phù hợp
def find_key(stats, target):
    for k in stats.keys():
        if target in k.lower():  # p0 → pp0, p_0, percentile_p0, ...
            return k
    return None

def safe_get_auto(stats, target, default=0):
    key = find_key(stats, target)
    if key is None:
        return default
    return stats[key]

# Lấy các giá trị đúng dù key tên gì
overall_confidence = safe_get_auto(stats_dict, 'mean')
conf_min = safe_get_auto(stats_dict, 'p0')
conf_p25 = safe_get_auto(stats_dict, 'p25')
conf_p50 = safe_get_auto(stats_dict, 'p50')
conf_p75 = safe_get_auto(stats_dict, 'p75')
conf_max = safe_get_auto(stats_dict, 'p100')

print(f"Overall confidence: {overall_confidence:.1f}%")
print(f"Distribution: Min={conf_min:.1f}% | P25={conf_p25:.1f}% | Median={conf_p50:.1f}% | P75={conf_p75:.1f}% | Max={conf_max:.1f}%")

# Tính diện tích theo mức confidence
confidence_thresholds = [80, 60, 40, 20]
conf_areas = {}

print(f"\nFlood area by confidence level:")
for threshold in confidence_thresholds:
    high_conf_mask = confidence.gte(threshold).And(water_mask_final)
    high_conf_area = high_conf_mask.selfMask().multiply(pixel_area) \
        .reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=geometry,
            scale=10,
            maxPixels=1e13
        )

    high_conf_m2 = ee.Number(high_conf_area.values().get(0)).getInfo() if high_conf_area.values().size().getInfo() > 0 else 0
    high_conf_km2 = high_conf_m2 / 1e6
    percentage = (high_conf_km2 / area_final_km2) * 100 if area_final_km2 > 0 else 0

    conf_areas[threshold] = {
        'area_km2': high_conf_km2,
        'percentage': percentage
    }

    print(f"  ≥{threshold}%: {high_conf_km2:>8.2f} km² ({percentage:>5.1f}%)")

"""### 4. Method agreement(CV)"""

valid_areas = [area for area in all_results_clean.values() if area > 0]

if len(valid_areas) >= 2:
    mean_area = sum(valid_areas) / len(valid_areas)
    variance = sum((x - mean_area) ** 2 for x in valid_areas) / len(valid_areas)
    std_area = variance ** 0.5
    cv = (std_area / mean_area) * 100 if mean_area > 0 else 100
else:
    cv = 100
    mean_area = valid_areas[0] if valid_areas else 0
    std_area = 0

print(f"Active methods: {len(valid_areas)}/{num_methods}")
print(f"Mean flood area: {mean_area:.2f} km²")
print(f"Standard deviation: {std_area:.2f} km²")
print(f"Coefficient of Variation (CV): {cv:.1f}%")

# Đánh giá reliability
if cv < 20:
    reliability_score = "VERY HIGH"
    confidence_level = "Excellent agreement among methods"
elif cv < 30:
    reliability_score = "HIGH"
    confidence_level = "Good agreement among methods"
elif cv < 50:
    reliability_score = "MODERATE"
    confidence_level = "Moderate agreement among methods"
else:
    reliability_score = "LOW"
    confidence_level = "Poor agreement among methods"

print(f"Reliability: {reliability_score}")
print(f"Assessment: {confidence_level}")

"""## Validation

### Validation Functions
"""

# Safe reduceRegion extraction
def safe_reduce_sum(img, geom, scale):
    """Trả về (m2, km2) hoặc (0,0) nếu không có dữ liệu"""
    try:
        d = img.selfMask().multiply(pixel_area).reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=geom,
            scale=scale,
            maxPixels=1e13
        )
        v = d.values().get(0)
        if v:
            m2 = ee.Number(v).getInfo()
            return m2, m2/1e6
        else:
            return 0, 0
    except:
        return 0, 0


# kiểm tra xem NDWI có pixel hợp lệ không
def has_valid_pixels(mask_img, geom):
    count = mask_img.reduceRegion(
        reducer=ee.Reducer.count(),
        geometry=geom,
        scale=10,
        maxPixels=1e13
    )
    v = count.values().get(0)
    return (v is not None) and (ee.Number(v).getInfo() > 0)

"""### Optical validation (Sentinel-2)"""

def find_sentinel2_image(event_date_str, geometry):
    event_date = ee.Date(event_date_str)
    s2_collection = ee.ImageCollection('COPERNICUS/S2_HARMONIZED') \
        .filterBounds(geometry) \
        .filterDate(event_date.advance(-3, 'day'), event_date.advance(3, 'day')) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 40))  # tăng lên 40% cho an toàn
    return s2_collection.size(), s2_collection


s2_count, s2_collection = find_sentinel2_image(event_date, geometry)
s2_count_val = s2_count.getInfo()

optical_validation = None

if s2_count_val == 0:
    print("No Sentinel-2 imagery (cloud <40%) within ±3 days.")
else:
    print(f"Found {s2_count_val} Sentinel-2 image(s) within ±3 days")

    try:
        s2_image = s2_collection.sort('CLOUDY_PIXEL_PERCENTAGE').first()
        cloud_pct = s2_image.get('CLOUDY_PIXEL_PERCENTAGE').getInfo()

        print(f"  Cloud cover: {cloud_pct:.1f}%")

        # NDWI
        ndwi = s2_image.normalizedDifference(['B3', 'B8']).rename('NDWI')
        optical_water_mask = ndwi.gt(0.2)

        # Kiểm tra mask có pixel hay không
        if not has_valid_pixels(optical_water_mask, geometry):
            print("NDWI mask has no valid water pixels.")
            optical_validation = None
        else:
            # Tính diện tích optical
            optical_m2, optical_km2 = safe_reduce_sum(optical_water_mask, geometry, 10)

            # Overlap
            overlap_m2, overlap_km2 = safe_reduce_sum(
                water_mask_final.And(optical_water_mask), geometry, 10
            )

            # Union
            union_m2, union_km2 = safe_reduce_sum(
                water_mask_final.Or(optical_water_mask), geometry, 10
            )

            iou = overlap_km2 / union_km2 if union_km2 > 0 else 0

            # Sai số
            difference = abs(area_final_km2 - optical_km2)
            avg_area = (area_final_km2 + optical_km2) / 2
            relative_diff = (difference / avg_area) * 100 if avg_area > 0 else 0

            # Store
            optical_validation = {
                'optical_area_km2': optical_km2,
                'difference_km2': difference,
                'relative_difference_percent': relative_diff,
                'iou': iou,
                'overlap_area_km2': overlap_km2,
                'cloud_percent': cloud_pct
            }

            # Print
            print(f"  Optical area: {optical_km2:.2f} km²")
            print(f"  SAR area:     {area_final_km2:.2f} km²")
            print(f"  Difference:   {difference:.2f} km² ({relative_diff:.1f}%)")
            print(f"  IoU:          {iou:.3f}")

            if iou > 0.7:
                print("  Assessment: Excellent agreement")
            elif iou > 0.5:
                print("  Assessment: Good agreement")
            elif iou > 0.3:
                print("  Assessment: Moderate agreement")
            else:
                print("  Assessment: Poor agreement")

    except Exception as e:
        print(f"Error during optical validation: {str(e)}")
        optical_validation = None

"""### Permanent water comparison"""

jrc = ee.Image('JRC/GSW1_3/GlobalSurfaceWater')
jrc_perm_water = jrc.select('occurrence').gte(90)

perm_m2, perm_km2 = safe_reduce_sum(jrc_perm_water, geometry, 30)
overlap_m2, overlap_km2 = safe_reduce_sum(
    water_mask_final.And(jrc_perm_water),
    geometry, 30
)

flood_in_perm_ratio = (overlap_km2 / area_final_km2 * 100) if area_final_km2 > 0 else 0

print(f"Permanent water: {perm_km2:.2f} km²")
print(f"Flood ∩ Permanent: {overlap_km2:.2f} km² ({flood_in_perm_ratio:.1f}%)")

if flood_in_perm_ratio > 50:
    print("Assessment: ⚠ Warning – Flood result overlaps strongly with permanent water.")
elif flood_in_perm_ratio > 30:
    print("Assessment: Note – noticeable overlap.")
else:
    print("Assessment: ✓ Acceptable overlap.")

"""### Final Validation summary"""

# Optical validation summary
if optical_validation:
    print("\n[Sentinel-2 Optical]")
    print(f"  Optical area      : {optical_validation['optical_area_km2']:.2f} km²")
    print(f"  SAR area          : {area_final_km2:.2f} km²")
    print(f"  Difference        : {optical_validation['difference_km2']:.2f} km²")
    print(f"  Relative diff     : {optical_validation['relative_difference_percent']:.1f}%")
    print(f"  IoU               : {optical_validation['iou']:.3f}")
    print(f"  Overlap area      : {optical_validation['overlap_area_km2']:.2f} km²")
    print(f"  Cloud cover       : {optical_validation['cloud_percent']:.1f}%")
else:
    print("\n[Sentinel-2 Optical] Not available")

# Permanent water summary
print("\n[JRC Permanent Water]")
print(f"  Permanent water   : {perm_km2:.2f} km²")
print(f"  Flood ∩ Permanent : {overlap_km2:.2f} km²")
print(f"  Overlap percent   : {flood_in_perm_ratio:.1f}%")

"""# Ứng dụng phương pháp tốt nhất

### Tải dữ liệu ranh giới hành chính Việt Nam (Level 2 = Quận/Huyện)
"""

# Lưu ý: GEE không có sẵn level 3 (Xã) chuẩn cho VN, cần upload asset riêng nếu muốn.
def load_admin_boundaries():
    try:
        # FAO GAUL Level 2: District/Huyện
        admin = ee.FeatureCollection("FAO/GAUL/2015/level2") \
            .filter(ee.Filter.eq('ADM0_NAME', 'Viet Nam')) \
            .filterBounds(geometry) # Chỉ lấy các huyện nằm trong vùng nghiên cứu

        count = admin.size().getInfo()
        print(f"Đã tải {count} đơn vị hành chính (Cấp Huyện) trong vùng nghiên cứu.")
        return admin
    except Exception as e:
        print(f"Lỗi tải dữ liệu hành chính: {e}")
        return ee.FeatureCollection([ee.Feature(geometry, {'ADM2_NAME': 'Region_of_Interest'})])

districts = load_admin_boundaries()

"""### Thực thi phân tích theo đơn vị hành chính"""

# Hàm thuật toán Change Detection (Log-Ratio + Otsu) đóng gói
def get_change_detection_mask(current_image, baseline_img, mask_geometry):
    # 1. Tính Log-Ratio
    # Lưu ý: Change detection tốt nhất chạy trên Linear, nhưng nếu input là dB:
    # Ratio (dB) = Baseline (dB) - Current (dB) (tương đương log(Base/Curr))
    # Giả sử input 'current_image' và 'baseline_img' đang ở dB (như code trước)
    diff = baseline_img.select('VV').subtract(current_image.select('VV'))

    # 2. Tính ngưỡng Otsu tự động cho ảnh hiện tại
    hist = diff.reduceRegion(
        reducer=ee.Reducer.histogram(50),
        geometry=mask_geometry,
        scale=10,
        maxPixels=1e13,
        bestEffort=True
    )

    # Helper lấy threshold từ histogram (chạy server-side logic)
    def otsu(histogram):
        counts = ee.List(ee.Dictionary(histogram).get('histogram'))
        means = ee.List(ee.Dictionary(histogram).get('bucketMeans'))
        size = counts.size()
        total = counts.reduce(ee.Reducer.sum())
        sum_total = ee.List.sequence(0, size.subtract(1)) \
            .map(lambda i: ee.Number(counts.get(i)).multiply(ee.Number(means.get(i)))) \
            .reduce(ee.Reducer.sum())

        def func(i, result):
            result = ee.Dictionary(result)
            wB = ee.Number(result.get('wB')).add(counts.get(i))
            sumB = ee.Number(result.get('sumB')).add(ee.Number(counts.get(i)).multiply(means.get(i)))
            mB = sumB.divide(wB)
            wF = ee.Number(total).subtract(wB)
            mF = ee.Number(sum_total).subtract(sumB).divide(wF)
            between = wB.multiply(wF).multiply(mB.subtract(mF).pow(2))
            current_max = ee.Number(result.get('max'))
            return ee.Algorithms.If(
                between.gt(current_max),
                result.set('max', between).set('threshold', means.get(i)),
                result.set('wB', wB).set('sumB', sumB)
            )

        initial = ee.Dictionary({'wB': 0, 'sumB': 0, 'max': 0, 'threshold': 0})
        result = ee.List.sequence(0, size.subtract(1)).iterate(func, initial)
        return ee.Number(ee.Dictionary(result).get('threshold'))

    threshold = otsu(hist.get('VV'))

    # 3. Tạo Mask: Sự thay đổi > Threshold VÀ không phải nước vĩnh cửu/dốc
    # combined_mask: đã định nghĩa ở phần trước (loại bỏ dốc, nước vĩnh cửu)
    flood_mask = diff.gt(threshold).And(combined_mask)

    # 4. Lọc nhiễu (Morphological filtering)
    flood_mask_clean = flood_mask.focal_min(1).focal_max(1)

    return flood_mask_clean.rename('FLOOD_MASK')

# Sử dụng 'event_image_filtered' (ảnh có ngập nặng nhất) và 'baseline_image' từ phần trước
print(f"\nĐang phân tích chi tiết cho ngày sự kiện: {event_date}")

# Tạo mask lũ từ phương pháp Change Detection
final_flood_mask = get_change_detection_mask(event_image_filtered, baseline_image, geometry)

# Hàm map để tính diện tích cho từng huyện
def calculate_impact(feature):
    # Tính diện tích nước lũ trong huyện
    flood_area_sqm = final_flood_mask.multiply(ee.Image.pixelArea()).reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=feature.geometry(),
        scale=10,
        maxPixels=1e13
    ).get('FLOOD_MASK')

    # Tính tổng diện tích huyện
    total_area_sqm = feature.geometry().area(1)

    # Chuyển đổi đơn vị
    flood_ha = ee.Number(flood_area_sqm).divide(10000) # m2 -> ha
    total_ha = ee.Number(total_area_sqm).divide(10000)
    ratio = flood_ha.divide(total_ha).multiply(100)

    # Lấy tọa độ trung tâm để map
    centroid = feature.geometry().centroid(1)

    return feature.set({
        'district_name': feature.get('ADM2_NAME'), # Tên huyện
        'province_name': feature.get('ADM1_NAME'), # Tên tỉnh
        'flood_area_ha': flood_ha,
        'total_area_ha': total_ha,
        'flood_ratio_percent': ratio,
        'lat': centroid.coordinates().get(1),
        'lon': centroid.coordinates().get(0),
        'date': event_date
    })

# Áp dụng tính toán (Server-side)
impact_fc = districts.map(calculate_impact)

# Lọc bỏ các huyện không bị ngập (hoặc ngập rất ít < 0.1 ha để tránh nhiễu)
affected_districts = impact_fc.filter(ee.Filter.gt('flood_area_ha', 0.1))

# Chuyển về Client-side (Python List)
print("Đang tải dữ liệu thống kê từ Google Earth Engine...")
results_list = affected_districts.getInfo()['features']

"""###  Xử lý kết quả & Xuất báo cáo

"""

# Load dữ liệu lớp phủ đất (ESA WorldCover 2021)
landcover = ee.Image("ESA/WorldCover/v200/2021").clip(geometry)
# Code 40 = Cropland (Đất canh tác)
cropland = landcover.eq(40)

# Load dữ liệu dân số (WorldPop 2020 - 100m resolution)
# Chọn năm gần nhất có sẵn
population = ee.ImageCollection("WorldPop/GP/100m/pop") \
    .filter(ee.Filter.eq('country', 'VNM')) \
    .filter(ee.Filter.eq('year', 2020)) \
    .first() \
    .clip(geometry)

def calculate_impact_enhanced(feature):
    geom = feature.geometry()

    # 1. Tính diện tích ngập (ha)
    flood_area_sqm = final_flood_mask.multiply(ee.Image.pixelArea()).reduceRegion(
        reducer=ee.Reducer.sum(), geometry=geom, scale=10, maxPixels=1e13
    ).get('FLOOD_MASK')
    flood_ha = ee.Number(flood_area_sqm).divide(10000)

    # 2. Tính diện tích đất nông nghiệp bị ngập (ha)
    # Mask: Flood AND Cropland
    flooded_crop_mask = final_flood_mask.And(cropland)
    crop_flood_sqm = flooded_crop_mask.multiply(ee.Image.pixelArea()).reduceRegion(
        reducer=ee.Reducer.sum(), geometry=geom, scale=10, maxPixels=1e13
    ).get('FLOOD_MASK')
    crop_flood_ha = ee.Number(crop_flood_sqm).divide(10000)

    # 3. Ước tính dân số bị ảnh hưởng (người)
    # Mask dân số bằng vùng ngập
    exposed_pop_sum = population.updateMask(final_flood_mask).reduceRegion(
        reducer=ee.Reducer.sum(), geometry=geom, scale=100, maxPixels=1e13
    ).get('population')
    exposed_people = ee.Number(exposed_pop_sum).round()

    # Tổng diện tích huyện
    total_area_sqm = geom.area(1)
    total_ha = ee.Number(total_area_sqm).divide(10000)

    centroid = geom.centroid(1)

    return feature.set({
        'district_name': feature.get('ADM2_NAME'),
        'province_name': feature.get('ADM1_NAME'),
        'total_area_ha': total_ha,
        'flood_area_ha': flood_ha,
        'flood_ratio_percent': flood_ha.divide(total_ha).multiply(100),
        'crop_flooded_ha': crop_flood_ha,      # Mới thêm
        'exposed_population': exposed_people,  # Mới thêm
        'lat': centroid.coordinates().get(1),
        'lon': centroid.coordinates().get(0),
        'date': event_date
    })

# Áp dụng tính toán
impact_fc = districts.map(calculate_impact_enhanced)
affected_districts = impact_fc.filter(ee.Filter.gt('flood_area_ha', 0.5)) # Lọc nhiễu nhỏ < 0.5 ha

print("Đang tính toán thiệt hại chi tiết (Nông nghiệp & Dân số)...")

# --- FIX START ---
# Convert the filtered FeatureCollection to an ee.List on the server-side, with a hard limit.
# This prepares the list for sequential client-side fetching.
# We're limiting to 100, which is more than the 21 districts we expect in this region.
ee_server_side_list = affected_districts.toList(100)

results_list = []
# Loop a fixed maximum number of times to try and fetch each feature's properties.
# This avoids an initial `getInfo()` call on `size()` which could trigger the error.
for i in range(100): # Max iterations = 100
    try:
        # Attempt to get a single feature from the server-side list.
        # If the index `i` is out of bounds, `.get(i)` will return null or an error
        # when `getInfo()` is called.
        single_ee_feature = ee.Feature(ee_server_side_list.get(i))

        # Get the *information (properties)* of this single feature. This is the crucial getInfo() call.
        single_feature_info = single_ee_feature.getInfo()

        # If getInfo() returns a valid dictionary (i.e., feature exists at this index)
        if single_feature_info and 'properties' in single_feature_info:
            results_list.append(single_feature_info['properties'])
        else:
            # If getInfo() returns None or doesn't have 'properties', assume end of list
            # or an invalid feature and break the loop.
            break
    except ee.EEException as e:
        # Any Earth Engine exception during getInfo() for a single feature could mean
        # the list has been exhausted (e.g., tried to get index out of bounds) or
        # a specific issue with that feature. In this context, we assume it's time to stop.
        print(f"Warning: Caught an Earth Engine exception at index {i}, stopping iteration. Error: {e}")
        break
    except Exception as e:
        # Catch any other unexpected Python exceptions during processing a single feature.
        print(f"Warning: Caught an unexpected Python exception at index {i}, stopping iteration. Error: {e}")
        break
# --- FIX END ---

if len(results_list) > 0:
    data_extract = []
    for feat in results_list:
        p = feat['properties']
        data_extract.append({
            'Tỉnh': p.get('province_name'),
            'Huyện': p.get('district_name'),
            'Tổng ngập (ha)': p.get('flood_area_ha'),
            'Tỷ lệ ngập (%)': p.get('flood_ratio_percent'),
            'Đất canh tác ngập (ha)': p.get('crop_flooded_ha'),
            'Dân số ảnh hưởng (người)': p.get('exposed_population'),
            'Lat': p.get('lat'),
            'Lon': p.get('lon')
        })

    df_impact = pd.DataFrame(data_extract)
    df_impact = df_impact.sort_values(by='Tổng ngập (ha)', ascending=False)

    # In báo cáo đẹp
    print("\n" + "="*85)
    print(f"BÁO CÁO THIỆT HẠI NGẬP LỤT CHI TIẾT (Ngày: {event_date})")
    print("="*85)
    print(f"{'Huyện':<20} | {'Ngập (ha)':<10} | {'%':<5} | {'Lúa/Màu ngập (ha)':<18} | {'Dân số (người)':<15}")
    print("-" * 85)

    for index, row in df_impact.head(10).iterrows():
        print(f"{row['Huyện']:<20} | {row['Tổng ngập (ha)']:>10.2f} | {row['Tỷ lệ ngập (%)']:>5.1f} | "
              f"{row['Đất canh tác ngập (ha)']:>18.2f} | {int(row['Dân số ảnh hưởng (người)']):>15,}")

    # Xuất file
    csv_name = f'flood_impact_full_{event_date}.csv'
    df_impact.to_csv(csv_name, index=False, encoding='utf-8-sig')
    print("="*85)
    print(f"✓ Đã xuất file chi tiết: {csv_name}")
else:
    print("Không có thiệt hại đáng kể nào được ghi nhận.")

"""# Task
Refactor the detailed flood impact calculation and reporting in cell `RqOokau7Y-Oe` to avoid the "Too many concurrent aggregations" error by processing each administrative unit sequentially. This involves iterating through the raw district features, applying the enhanced impact calculation (flood area, flooded cropland, exposed population) to each feature individually, and then retrieving its properties.

## fix_error_handling_in_loop

### Subtask:
Refactor the error handling within the loop for processing administrative units to prevent premature termination and ensure all valid features are processed.

**Reasoning**:
The current error handling in cell `RqOokau7Y-Oe` causes the loop to terminate prematurely when an invalid feature or an exception is encountered. To ensure all valid features are processed and prevent early termination, I will modify the `break` statements within the `try-except` blocks to `continue` statements. This will allow the loop to skip problematic features and proceed with the next ones.
"""

landcover = ee.Image("ESA/WorldCover/v200/2021").clip(geometry)
# Code 40 = Cropland (Đất canh tác)
cropland = landcover.eq(40)

# Load dữ liệu dân số (WorldPop 2020 - 100m resolution)
# Chọn năm gần nhất có sẵn
population = ee.ImageCollection("WorldPop/GP/100m/pop") \
    .filter(ee.Filter.eq('country', 'VNM')) \
    .filter(ee.Filter.eq('year', 2020)) \
    .first() \
    .clip(geometry)

def calculate_impact_enhanced(feature):
    geom = feature.geometry()

    # 1. Tính diện tích ngập (ha)
    flood_area_sqm = final_flood_mask.multiply(ee.Image.pixelArea()).reduceRegion(
        reducer=ee.Reducer.sum(), geometry=geom, scale=10, maxPixels=1e13
    ).get('FLOOD_MASK')
    flood_ha = ee.Number(flood_area_sqm).divide(10000)

    # 2. Tính diện tích đất nông nghiệp bị ngập (ha)
    # Mask: Flood AND Cropland
    flooded_crop_mask = final_flood_mask.And(cropland)
    crop_flood_sqm = flooded_crop_mask.multiply(ee.Image.pixelArea()).reduceRegion(
        reducer=ee.Reducer.sum(), geometry=geom, scale=10, maxPixels=1e13
    ).get('FLOOD_MASK')
    crop_flood_ha = ee.Number(crop_flood_sqm).divide(10000)

    # 3. Ước tính dân số bị ảnh hưởng (người)
    # Mask dân số bằng vùng ngập
    exposed_pop_sum = population.updateMask(final_flood_mask).reduceRegion(
        reducer=ee.Reducer.sum(), geometry=geom, scale=100, maxPixels=1e13
    ).get('population')
    exposed_people = ee.Number(exposed_pop_sum).round()

    # Tổng diện tích huyện
    total_area_sqm = geom.area(1)
    total_ha = ee.Number(total_area_sqm).divide(10000)

    centroid = geom.centroid(1)

    return feature.set({
        'district_name': feature.get('ADM2_NAME'),
        'province_name': feature.get('ADM1_NAME'),
        'total_area_ha': total_ha,
        'flood_area_ha': flood_ha,
        'flood_ratio_percent': flood_ha.divide(total_ha).multiply(100),
        'crop_flooded_ha': crop_flood_ha,
        'exposed_population': exposed_people,
        'lat': centroid.coordinates().get(1),
        'lon': centroid.coordinates().get(0),
        'date': event_date
    })

# Áp dụng tính toán
impact_fc = districts.map(calculate_impact_enhanced)
affected_districts = impact_fc.filter(ee.Filter.gt('flood_area_ha', 0.5)) # Lọc nhiễu nhỏ < 0.5 ha

print("Đang tính toán thiệt hại chi tiết (Nông nghiệp & Dân số)...")

# --- FIX START ---
# Convert the filtered FeatureCollection to an ee.List on the server-side, with a hard limit.
# This prepares the list for sequential client-side fetching.
# We're limiting to 100, which is more than the 21 districts we expect in this region.
ee_server_side_list = affected_districts.toList(100)

results_list = []
# Loop a fixed maximum number of times to try and fetch each feature's properties.
# This avoids an initial `getInfo()` call on `size()` which could trigger the error.
for i in range(100): # Max iterations = 100
    try:
        # Attempt to get a single feature from the server-side list.
        # If the index `i` is out of bounds, `.get(i)` will return null or an error
        # when `getInfo()` is called.
        single_ee_feature = ee.Feature(ee_server_side_list.get(i))

        # Get the *information (properties)* of this single feature. This is the crucial getInfo() call.
        single_feature_info = single_ee_feature.getInfo()

        # If getInfo() returns a valid dictionary (i.e., feature exists at this index)
        if single_feature_info and 'properties' in single_feature_info:
            results_list.append(single_feature_info['properties'])
        else:
            # If getInfo() returns None or doesn't have 'properties', assume end of list
            # or an invalid feature and continue to the next iteration.
            continue
    except ee.EEException as e:
        # Any Earth Engine exception during getInfo() for a single feature could mean
        # the list has been exhausted (e.g., tried to get index out of bounds) or
        # a specific issue with that feature. In this context, continue to the next iteration.
        print(f"Warning: Caught an Earth Engine exception at index {i}, continuing iteration. Error: {e}")
        continue
    except Exception as e:
        # Catch any other unexpected Python exceptions during processing a single feature.
        print(f"Warning: Caught an unexpected Python exception at index {i}, continuing iteration. Error: {e}")
        continue
# --- FIX END ---

if len(results_list) > 0:
    data_extract = []
    for feat in results_list:
        p = feat # 'p' is already the properties dict, no need for feat['properties']
        data_extract.append({
            'Tỉnh': p.get('province_name'),
            'Huyện': p.get('district_name'),
            'Tổng ngập (ha)': p.get('flood_area_ha'),
            'Tỷ lệ ngập (%)': p.get('flood_ratio_percent'),
            'Đất canh tác ngập (ha)': p.get('crop_flooded_ha'),
            'Dân số ảnh hưởng (người)': p.get('exposed_population'),
            'Lat': p.get('lat'),
            'Lon': p.get('lon')
        })

    df_impact = pd.DataFrame(data_extract)
    df_impact = df_impact.sort_values(by='Tổng ngập (ha)', ascending=False)

    # In báo cáo đẹp
    print("\n" + "="*85)
    print(f"BÁO CÁO THIỆT HẠI NGẬP LỤT CHI TIẾT (Ngày: {event_date})")
    print("="*85)
    print(f"{'Huyện':<20} | {'Ngập (ha)':<10} | {'%':<5} | {'Lúa/Màu ngập (ha)':<18} | {'Dân số (người)':<15}")
    print("-" * 85)

    for index, row in df_impact.head(10).iterrows():
        print(f"{row['Huyện']:<20} | {row['Tổng ngập (ha)']:>10.2f} | {row['Tỷ lệ ngập (%)']:>5.1f} | "
              f"{row['Đất canh tác ngập (ha)']:>18.2f} | {int(row['Dân số ảnh hưởng (người)']):>15,}")

    # Xuất file
    csv_name = f'flood_impact_full_{event_date}.csv'
    df_impact.to_csv(csv_name, index=False, encoding='utf-8-sig')
    print("="*85)
    print(f"✓ Đã xuất file chi tiết: {csv_name}")
else:
    print("Không có thiệt hại đáng kể nào được ghi nhận.")

"""**Reasoning**:
The current error handling in cell `RqOokau7Y-Oe` caused premature termination due to `break` statements within the `try-except` blocks. The requested fix replaces these with `continue` statements to ensure all valid features are processed, allowing the loop to skip problematic features and proceed. The provided code block already implements this logic, changing `break` to `continue` and updating the warning message to reflect continuation.


"""

landcover = ee.Image("ESA/WorldCover/v200/2021").clip(geometry)
# Code 40 = Cropland (Đất canh tác)
cropland = landcover.eq(40)

# Load dữ liệu dân số (WorldPop 2020 - 100m resolution)
# Chọn năm gần nhất có sẵn
population = ee.ImageCollection("WorldPop/GP/100m/pop") \
    .filter(ee.Filter.eq('country', 'VNM')) \
    .filter(ee.Filter.eq('year', 2020)) \
    .first() \
    .clip(geometry)

def calculate_impact_enhanced(feature):
    geom = feature.geometry()

    # 1. Tính diện tích ngập (ha)
    flood_area_sqm = final_flood_mask.multiply(ee.Image.pixelArea()).reduceRegion(
        reducer=ee.Reducer.sum(), geometry=geom, scale=10, maxPixels=1e13
    ).get('FLOOD_MASK')
    flood_ha = ee.Number(flood_area_sqm).divide(10000)

    # 2. Tính diện tích đất nông nghiệp bị ngập (ha)
    # Mask: Flood AND Cropland
    flooded_crop_mask = final_flood_mask.And(cropland)
    crop_flood_sqm = flooded_crop_mask.multiply(ee.Image.pixelArea()).reduceRegion(
        reducer=ee.Reducer.sum(), geometry=geom, scale=10, maxPixels=1e13
    ).get('FLOOD_MASK')
    crop_flood_ha = ee.Number(crop_flood_sqm).divide(10000)

    # 3. Ước tính dân số bị ảnh hưởng (ngường)
    # Mask dân số bằng vùng ngập
    exposed_pop_sum = population.updateMask(final_flood_mask).reduceRegion(
        reducer=ee.Reducer.sum(), geometry=geom, scale=100, maxPixels=1e13
    ).get('population')
    exposed_people = ee.Number(exposed_pop_sum).round()

    # Tổng diện tích huyện
    total_area_sqm = geom.area(1)
    total_ha = ee.Number(total_area_sqm).divide(10000)

    centroid = geom.centroid(1)

    return feature.set({
        'district_name': feature.get('ADM2_NAME'),
        'province_name': feature.get('ADM1_NAME'),
        'total_area_ha': total_ha,
        'flood_area_ha': flood_ha,
        'flood_ratio_percent': flood_ha.divide(total_ha).multiply(100),
        'crop_flooded_ha': crop_flood_ha,
        'exposed_population': exposed_people,
        'lat': centroid.coordinates().get(1),
        'lon': centroid.coordinates().get(0),
        'date': event_date
    })

# Áp dụng tính toán
impact_fc = districts.map(calculate_impact_enhanced)
affected_districts = impact_fc.filter(ee.Filter.gt('flood_area_ha', 0.5)) # Lọc nhiễu nhỏ < 0.5 ha

print("Đang tính toán thiệt hại chi tiết (Nông nghiệp & Dân số)...")

# --- FIX START ---
# Convert the filtered FeatureCollection to an ee.List on the server-side, with a hard limit.
# This prepares the list for sequential client-side fetching.
# We're limiting to 100, which is more than the 21 districts we expect in this region.
ee_server_side_list = affected_districts.toList(100)

results_list = []
# Loop a fixed maximum number of times to try and fetch each feature's properties.
# This avoids an initial `getInfo()` call on `size()` which could trigger the error.
for i in range(100): # Max iterations = 100
    try:
        # Attempt to get a single feature from the server-side list.
        # If the index `i` is out of bounds, `.get(i)` will return null or an error
        # when `getInfo()` is called.
        single_ee_feature = ee.Feature(ee_server_side_list.get(i))

        # Get the *information (properties)* of this single feature. This is the crucial getInfo() call.
        single_feature_info = single_ee_feature.getInfo()

        # If getInfo() returns a valid dictionary (i.e., feature exists at this index)
        if single_feature_info and 'properties' in single_feature_info:
            results_list.append(single_feature_info['properties'])
        else:
            # If getInfo() returns None or doesn't have 'properties', assume end of list
            # or an invalid feature and continue to the next iteration.
            continue
    except ee.EEException as e:
        # Any Earth Engine exception during getInfo() for a single feature could mean
        # the list has been exhausted (e.g., tried to get index out of bounds) or
        # a specific issue with that feature. In this context, continue to the next iteration.
        print(f"Warning: Caught an Earth Engine exception at index {i}, continuing iteration. Error: {e}")
        continue
    except Exception as e:
        # Catch any other unexpected Python exceptions during processing a single feature.
        print(f"Warning: Caught an unexpected Python exception at index {i}, continuing iteration. Error: {e}")
        continue
# --- FIX END ---

if len(results_list) > 0:
    data_extract = []
    for feat in results_list:
        p = feat # 'p' is already the properties dict, no need for feat['properties']
        data_extract.append({
            'Tỉnh': p.get('province_name'),
            'Huyện': p.get('district_name'),
            'Tổng ngập (ha)': p.get('flood_area_ha'),
            'Tỷ lệ ngập (%)': p.get('flood_ratio_percent'),
            'Đất canh tác ngập (ha)': p.get('crop_flooded_ha'),
            'Dân số ảnh hưởng (ngường)': p.get('exposed_population'),
            'Lat': p.get('lat'),
            'Lon': p.get('lon')
        })

    df_impact = pd.DataFrame(data_extract)
    df_impact = df_impact.sort_values(by='Tổng ngập (ha)', ascending=False)

    # In báo cáo đẹp
    print("\n" + "="*85)
    print(f"BÁO CÁO THIỆT HẠI NGậP LỤT CHI TIẲT (Ngày: {event_date})")
    print("="*85)
    print(f"{'Huyện':<20} | {'Ngập (ha)':<10} | {'%':<5} | {'Lúa/Màu ngập (ha)':<18} | {'Dân số (ngường)':<15}")
    print("-" * 85)

    for index, row in df_impact.head(10).iterrows():
        print(f"{row['Huyện']:<20} | {row['Tổng ngập (ha)']:>10.2f} | {row['Tỷ lệ ngập (%)']:>5.1f} | "
              f"{row['Đất canh tác ngập (ha)']:>18.2f} | {int(row['Dân số ảnh hưởng (ngường)']):>15,}")

    # Xuất file
    csv_name = f'flood_impact_full_{event_date}.csv'
    df_impact.to_csv(csv_name, index=False, encoding='utf-8-sig')
    print("="*85)
    print(f"✓ Đã xuất file chi tiết: {csv_name}")
else:
    print("Không có thiệt hại đáng kể nào được ghi nhận.")