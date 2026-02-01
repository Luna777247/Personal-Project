import json
import pandas as pd
import re
import glob
from datetime import datetime
import spacy
from transformers import pipeline
import unicodedata
import logging
from tqdm import tqdm
import multiprocessing as mp

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load spaCy model (Vietnamese primary, English fallback)
try:
    nlp_spacy = spacy.load("vi_core_news_lg")  # Vietnamese model as primary
    logger.info("Loaded Vietnamese spaCy model")
except:
    try:
        nlp_spacy = spacy.load("en_core_web_sm")  # English fallback
        logger.warning("Vietnamese spaCy model not found, using English fallback")
    except:
        nlp_spacy = None
        logger.warning("No spaCy model loaded")

# Load Hugging Face NER pipeline for Vietnamese (secondary)
try:
    ner_pipeline = pipeline("ner", model="NlpHUST/ner-vietnamese-electra-base", aggregation_strategy="simple")
    logger.info("Loaded Hugging Face NER model")
except:
    ner_pipeline = None
    logger.warning("Hugging Face NER model not loaded")

# Enhanced regex patterns from extract_data_rule_based.py
weather_patterns = {
    'wind_speed': re.compile(r'(?:suc gio|toc do gio|gio manh|gió mạnh)\s+(?:manh\s+)?(?:nhat\s+|cuc dai\s+)?(?:cap\s+|cấp\s+)?(\d+)', re.IGNORECASE),
    'gust_speed': re.compile(r'(?:giat cap|giat toc|toc do giat)\s+(\d+)', re.IGNORECASE),
    'pressure': re.compile(r'(?:ap suat|ap luc)\s*(?:thap|cao)?\s*(\d+)', re.IGNORECASE),
    'movement_speed': re.compile(r'(?:toc do|van toc)\s*(?:di chuyen|chuyen dong)?\s*(\d+)-?(\d+)?\s*km/gio', re.IGNORECASE),
    'direction': re.compile(r'(?:huong|phuong)\s+([^,\.\d]+)', re.IGNORECASE),
    'rain_amount': re.compile(r'(?:luong mua|mua|luong mua trung binh)\s+(?:pho bien|trung binh|trong|dat|dat duoc)?\s*(\d+)(?:-(\d+))?\s*mm', re.IGNORECASE),
    'earthquake_magnitude': re.compile(r'(?:do lon|richter|magnitude)\s+(\d+(?:\.\d+)?)', re.IGNORECASE),
    'earthquake_depth': re.compile(r'(?:do sau|chieu sau)\s*(\d+)-?(\d+)?\s*km', re.IGNORECASE),
    'tsunami_height': re.compile(r'(?:do cao|chieu cao song)\s*(\d+)-?(\d+)?\s*m', re.IGNORECASE),
    'fire_area': re.compile(r'(?:dien tich|mat do)\s*(?:chay|bi chay)\s*(\d+(?:\.\d+)?)\s*ha', re.IGNORECASE),
    'drought_duration': re.compile(r'(?:keo dai|thoi gian)\s*(\d+)-?(\d+)?\s*(?:ngay|thang|nam)', re.IGNORECASE),
    'water_shortage': re.compile(r'(?:thieu|han)\s*(\d+(?:\.\d+)?)%?\s*(?:nuoc|nguon nuoc)', re.IGNORECASE),
    'temperature': re.compile(r'(?:nhiet do|do nong)\s*(\d+)-?(\d+)?\s*°?c', re.IGNORECASE),
    'aqi_index': re.compile(r'(?:chi so|AQI|chat luong khong khi)\s*(\d+)', re.IGNORECASE),
    'pollutant_level': re.compile(r'(?:nong do|ham luong)\s*(\d+(?:\.\d+)?)\s*(?:ppm|mg/m³|µg/m³)', re.IGNORECASE),
    'volcanic_ash_height': re.compile(r'(?:tro vung|cot tro)\s*(?:cao|chieu cao)\s*(\d+)-?(\d+)?\s*(?:km|m)', re.IGNORECASE),
    'landslide_depth': re.compile(r'(?:do sau|chieu sau)\s*(?:sat lo|truot dat)\s*(\d+)-?(\d+)?\s*m', re.IGNORECASE),
    'oil_spill_area': re.compile(r'(?:dien tich|mat do)\s*(?:tran dau|ro ri dau)\s*(\d+(?:\.\d+)?)\s*(?:km²|hec-ta|m²)', re.IGNORECASE),
    'chemical_concentration': re.compile(r'(?:nong do|ham luong)\s*(?:hoa chat|phong xa)\s*(\d+(?:\.\d+)?)\s*(?:ppm|mg/l|µg/l)', re.IGNORECASE),
    'epidemic_cases': re.compile(r'(\d+)\s*(?:ca|nguoi|benh nhan)\s*(?:nhiem|mac|bi)\s*(?:benh|dich)', re.IGNORECASE),
    'invasive_species_area': re.compile(r'(?:dien tich|khu vuc)\s*(?:sinh vat ngoai lai|sinh vat xam hai)\s*(\d+(?:\.\d+)?)\s*(?:ha|km²)', re.IGNORECASE),
    'salinity_level': re.compile(r'(?:do man|ham luong man)\s*(\d+(?:\.\d+)?)\s*(?:g/l|ppt|%)', re.IGNORECASE),
    'frost_temperature': re.compile(r'(?:nhiet do|do lanh)\s*(?:suong muoi|ret dam|ret hai)\s*(-?\d+)-?(-?\d+)?\s*°?c', re.IGNORECASE)
}

damage_patterns = {
    'human_losses': re.compile(r'(\d+(?:[.,]\d+)?)\s*(?:nguoi|ca|người)\s*(?:chet|thiet mang|tu vong|mat mang|thuong vong|chết|tử vong|mất mạng|thương vong)', re.IGNORECASE),
    'injured': re.compile(r'(\d+)\s*(?:nguoi|ca)\s*(?:bi thuong|bi thuong tich|bi thuong nang|bi thuong trong)', re.IGNORECASE),
    'missing': re.compile(r'(\d+)\s*(?:nguoi|ca)\s*(?:mat tich|bi mat tich|thieu)', re.IGNORECASE),
    'economic_loss': re.compile(r'(?:gay|gay ra|ton that|mat mat|thiet hai)(?:\s*(?:kinh te|tai chinh|khoang|du kien|uoc tinh))?\s*[:;]?\s*(\d+(?:[.,]\d+)?)\s*(?:ty|trieu|nghin|tỷ|triệu|nghìn)?\s*(?:dong|usd|\$|đồng)?', re.IGNORECASE),
    'property_damage': re.compile(r'(\d+)\s*(?:ngoi nha|can nha|toa nha|co so ha tang)\s*(?:bi|thiet hai|sap|pha huy)', re.IGNORECASE),
    'evacuated': re.compile(r'(?:so tan|di cu|di dan)\s*(\d+)\s*(?:ho|nguoi|khau|gia dinh)', re.IGNORECASE),
    'houses_destroyed': re.compile(r'(\d+)\s*(?:ngoi nha|can nha|nha)\s*(?:sap|bi sap|bi pha huy|bi cuon troi|bi huy hoai)', re.IGNORECASE),
    'houses_damaged': re.compile(r'(\d+(?:[.,]\d+)?)\s*(?:ngoi nha|can nha|nha|toà nhà)\s*(?:bi|thiet hai|sap|pha huy|hu hong|hu hai|hư hỏng|tốc mái|ngập|anh huong|bị ảnh hưởng)', re.IGNORECASE),
    'roads_damaged': re.compile(r'(\d+(?:\.\d+)?)\s*(?:km|cay so)\s*(?:duong sa|quoc lo|tinh lo|duong)\s*(?:bi hu|bi sat|bi ngap|bi hong)', re.IGNORECASE),
    'bridges_damaged': re.compile(r'(\d+)\s*(?:cay|cau)\s*(?:cau|bac)\s*(?:bi hu|bi sap|bi cuon troi|bi pha huy)', re.IGNORECASE),
    'crops_damaged': re.compile(r'(\d+(?:\.\d+)?)\s*(?:ha|hec-ta)\s*(?:ruong|dong|lua|nong nghiep|dat trong)\s*(?:bi anh huong|bi thiet hai|bi mat)', re.IGNORECASE),
    'livestock_lost': re.compile(r'(\d+)\s*(?:con|vat nuoi|gia suc|gia cam)\s*(?:chet|mat|bi cuon troi|bi huy)', re.IGNORECASE),
    'infrastructure_damage': re.compile(r'(\d+)\s*(?:cong trinh|ha tang|co so)\s*(?:bi hu|bi pha huy|bi hong)', re.IGNORECASE),
    'forest_area_burned': re.compile(r'(\d+(?:\.\d+)?)\s*(?:ha|hec-ta)\s*(?:rung|khu rung)\s*(?:bi chay|chay|bi dot)', re.IGNORECASE),
    'water_shortage_households': re.compile(r'(\d+)\s*(?:ho|gia dinh|nguoi)\s*(?:thieu nuoc|kho han|han han)', re.IGNORECASE),
    'drought_affected_area': re.compile(r'(\d+(?:\.\d+)?)\s*(?:ha|hec-ta)\s*(?:dat|ruong|dien tich)\s*(?:kho han|bi han|han han)', re.IGNORECASE),
    'pollution_affected_people': re.compile(r'(\d+)\s*(?:nguoi|ca)\s*(?:bi anh huong|bi ngo doc|bi tac dong)\s*(?:o nhiem|khong khi)', re.IGNORECASE),
    'health_impact': re.compile(r'(\d+)\s*(?:nguoi|ca)\s*(?:vao vien|cap cuu|bi benh|bi nhiem)\s*(?:do|vi)\s*(?:o nhiem|khong khi)', re.IGNORECASE),
    'volcanic_victims': re.compile(r'(\d+)\s*(?:nguoi|ca)\s*(?:chet|bi hu|bi anh huong)\s*(?:do|vi)\s*(?:nui lua|tro vung)', re.IGNORECASE),
    'landslide_victims': re.compile(r'(\d+)\s*(?:nguoi|ca)\s*(?:chet|mat tich|bi hu)\s*(?:do|vi)\s*(?:sat lo|truot dat)', re.IGNORECASE),
    'epidemic_deaths': re.compile(r'(\d+)\s*(?:nguoi|ca)\s*(?:chet|tu vong)\s*(?:do|vi)\s*(?:dich benh|benh dich)', re.IGNORECASE),
    'epidemic_infected': re.compile(r'(\d+)\s*(?:nguoi|ca)\s*(?:nhiem|mac|bi)\s*(?:dich benh|benh dich)', re.IGNORECASE),
    'animal_epidemic': re.compile(r'(\d+)\s*(?:con|vat nuoi|gia suc)\s*(?:chet|bi huy)\s*(?:do|vi)\s*(?:dich benh)', re.IGNORECASE),
    'crop_epidemic': re.compile(r'(\d+(?:\.\d+)?)\s*(?:ha|hec-ta)\s*(?:ruong|dong|dat trong)\s*(?:bi huy|mat)\s*(?:do|vi)\s*(?:dich benh)', re.IGNORECASE),
    'oil_pollution_area': re.compile(r'(\d+(?:\.\d+)?)\s*(?:km²|hec-ta)\s*(?:bien|hai|song)\s*(?:bi o nhiem|tran dau)', re.IGNORECASE),
    'chemical_accident_victims': re.compile(r'(\d+)\s*(?:nguoi|ca)\s*(?:bi anh huong|nhiem doc|bi hu)\s*(?:do|vi)\s*(?:hoa chat|phong xa)', re.IGNORECASE),
    'marine_life_affected': re.compile(r'(\d+)\s*(?:con|loai)\s*(?:dong vat|ca|tom)\s*(?:bien|hai)\s*(?:chet|bi hu)\s*(?:do|vi)\s*(?:o nhiem)', re.IGNORECASE),
    'salinity_affected_area': re.compile(r'(\d+(?:\.\d+)?)\s*(?:ha|hec-ta)\s*(?:dat|ruong)\s*(?:bi anh huong|han han)\s*(?:do|vi)\s*(?:xam nhap man)', re.IGNORECASE),
    'frost_damage': re.compile(r'(\d+(?:\.\d+)?)\s*(?:ha|hec-ta)\s*(?:ruong|dong|nong nghiep)\s*(?:bi hu|mat)\s*(?:do|vi)\s*(?:suong muoi|ret hai)', re.IGNORECASE)
}

def extract_entities_hf(content):
    """Trích xuất entities bằng Hugging Face (LOC, MISC, PER, ORG) với cải thiện cleaning"""
    if ner_pipeline:
        # Làm sạch text để tránh lỗi tokenization
        content = unicodedata.normalize('NFKC', content)
        content = re.sub(r'[^\w\s.,!?;:\-()\'\"àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴĐ]', ' ', content)
        
        # Cắt content để tránh lỗi max length
        content = content[:1000]  # Approximate 512 tokens
        try:
            entities = ner_pipeline(content)
            locs = []
            for ent in entities:
                if ent['entity_group'] == 'LOC':
                    # Loại bỏ token fragments (bắt đầu bằng ##)
                    word = ent['word']
                    if not word.startswith('##'):
                        locs.append(word)
            return list(set(locs))  # Unique locations
        except Exception as e:
            logger.warning(f"NER extraction failed: {e}")
            return []
    return []

def extract_entities_spacy(content):
    """Trích xuất entities bằng spaCy (GPE, LOC, FAC, ORG) với mô hình tiếng Việt"""
    if not nlp_spacy:
        return []
    
    try:
        doc = nlp_spacy(content)
        # Extract location-related entities
        locs = []
        for ent in doc.ents:
            if ent.label_ in ['GPE', 'LOC', 'FAC']:  # GPE: geopolitical, LOC: location, FAC: facility
                locs.append(ent.text)
        return list(set(locs))
    except Exception as e:
        logger.warning(f"spaCy extraction failed: {e}")
        return []

def extract_location_rule_based(content):
    """Rule-based extraction cho địa danh phổ biến Việt Nam"""
    # Danh sách địa danh mở rộng
    vietnam_locations = [
        # Tỉnh thành
        'Hà Nội', 'Hồ Chí Minh', 'TP.HCM', 'Sài Gòn', 'Hải Phòng', 'Cần Thơ', 'Đà Nẵng',
        'Hà Giang', 'Cao Bằng', 'Bắc Kạn', 'Tuyên Quang', 'Lào Cai', 'Điện Biên', 'Lai Châu', 'Sơn La',
        'Yên Bái', 'Hòa Bình', 'Thái Nguyên', 'Lạng Sơn', 'Quảng Ninh', 'Bắc Giang', 'Phú Thọ',
        'Vĩnh Phúc', 'Bắc Ninh', 'Hải Dương', 'Hưng Yên', 'Thái Bình', 'Hà Nam', 'Nam Định', 'Ninh Bình',
        'Thanh Hóa', 'Nghệ An', 'Hà Tĩnh', 'Quảng Bình', 'Quảng Trị', 'Thừa Thiên Huế', 'Đà Nẵng',
        'Quảng Nam', 'Quảng Ngãi', 'Bình Định', 'Phú Yên', 'Khánh Hòa', 'Ninh Thuận', 'Bình Thuận',
        'Kon Tum', 'Gia Lai', 'Đắk Lắk', 'Đắk Nông', 'Lâm Đồng',
        'Bình Phước', 'Tây Ninh', 'Bình Dương', 'Đồng Nai', 'Bà Rịa Vũng Tàu', 'Long An', 'Tiền Giang',
        'Bến Tre', 'Trà Vinh', 'Vĩnh Long', 'Đồng Tháp', 'An Giang', 'Kiên Giang', 'Hậu Giang', 'Sóc Trăng', 'Bạc Liêu', 'Cà Mau',
        # Vùng miền
        'Miền Bắc', 'Miền Trung', 'Miền Nam', 'Đồng bằng sông Hồng', 'Đồng bằng sông Cửu Long',
        'Trung du miền núi Bắc Bộ', 'Duyên hải Nam Trung Bộ', 'Tây Nguyên', 'Đông Nam Bộ',
        # Biển đảo
        'Biển Đông', 'Hoàng Sa', 'Trường Sa', 'vịnh Bắc Bộ', 'vịnh Hạ Long',
        # Sông hồ
        'sông Hồng', 'sông Mã', 'sông Lam', 'sông Cả', 'sông Gianh', 'sông Hương', 'sông Thu Bồn',
        'sông Ba', 'sông Đồng Nai', 'sông Cửu Long', 'sông Tiền', 'sông Hậu', 'hồ Hoàn Kiếm', 'hồ Tây'
    ]
    
    found_locations = []
    content_lower = content.lower()
    for loc in vietnam_locations:
        if loc.lower() in content_lower:
            found_locations.append(loc)
    
    return list(set(found_locations))

def extract_location(content):
    """Trích xuất vị trí ảnh hưởng bằng NLP kết hợp (spaCy primary + Hugging Face secondary + rule-based)"""
    # spaCy primary (Vietnamese model)
    locs_spacy = extract_entities_spacy(content)
    
    # Hugging Face secondary
    locs_hf = extract_entities_hf(content)
    
    # Rule-based
    locs_rule = extract_location_rule_based(content)
    
    # Combine all
    all_locs = list(set(locs_spacy + locs_hf + locs_rule))
    
    # Filter to known provinces and regions
    provinces = [
        'Quảng Ninh', 'Hải Phòng', 'Hà Nội', 'Thanh Hóa', 'Nghệ An', 'Hà Tĩnh',
        'Quảng Bình', 'Quảng Trị', 'Thừa Thiên Huế', 'Đà Nẵng', 'Quảng Nam',
        'Quảng Ngãi', 'Bình Định', 'Phú Yên', 'Khánh Hòa', 'Ninh Thuận',
        'Bình Thuận', 'Bà Rịa Vũng Tàu', 'Tây Ninh', 'Đồng Nai', 'Gia Lai',
        'Đắk Lắk', 'Kon Tum', 'Lâm Đồng', 'Đắk Nông', 'Biển Đông', 'Hoàng Sa',
        'Trường Sa', 'vịnh Bắc Bộ', 'Miền Bắc', 'Miền Trung', 'Miền Nam',
        'Đồng bằng sông Cửu Long', 'Đồng bằng sông Hồng'
    ]
    
    # Additional filtering: remove fragments and short words
    filtered = []
    for loc in all_locs:
        loc_clean = loc.strip()
        if len(loc_clean) >= 2 and not loc_clean.startswith('##') and not re.match(r'^\W+$', loc_clean):
            # Check if it matches or contains known provinces
            if any(prov.lower() in loc_clean.lower() for prov in provinces) or loc_clean in provinces:
                filtered.append(loc_clean)
    
    return ', '.join(filtered) if filtered else None

def extract_quantities_with_ner(content):
    """Sử dụng NER để trích xuất quantities (số liệu)"""
    if not nlp_spacy:
        return {}
    
    try:
        doc = nlp_spacy(content)
        quantities = {}
        for ent in doc.ents:
            if ent.label_ == 'QUANTITY':
                # Parse quantity (e.g., "100 mm", "cấp 6")
                text = ent.text.lower()
                if 'mm' in text:
                    match = re.search(r'(\d+(?:\.\d+)?)', text)
                    if match:
                        quantities['rainfall'] = f"{match.group(1)} mm"
                elif 'km/h' in text or 'kmh' in text:
                    match = re.search(r'(\d+(?:\.\d+)?)', text)
                    if match:
                        quantities['wind_speed'] = f"cấp {min(17, max(1, round(float(match.group(1)) / 5)))}"  # Rough conversion
                elif 'người' in text and ('chết' in text or 'mất tích' in text):
                    match = re.search(r'(\d+(?:\.\d+)?)', text)
                    if match:
                        quantities['casualties'] = f"{match.group(1)} người chết"
                elif 'tỷ' in text and 'đồng' in text:
                    match = re.search(r'(\d+(?:\.\d+)?)', text)
                    if match:
                        quantities['damages'] = f"{match.group(1)} tỷ đồng"
        return quantities
    except Exception as e:
        logger.warning(f"NER quantity extraction failed: {e}")
        return {}

def extract_numbers(content):
    """Trích xuất số liệu (sức gió, mưa, thiệt hại) bằng enhanced regex patterns từ rule-based extraction"""
    # First, try NER for quantities
    ner_quantities = extract_quantities_with_ner(content)
    
    # Initialize results
    wind_speed = None
    rainfall = None
    casualties = None
    damages = None
    
def extract_numbers(content):
    """Trích xuất số liệu (sức gió, mưa, thiệt hại) bằng enhanced regex patterns từ rule-based extraction"""
    # First, try NER for quantities
    ner_quantities = extract_quantities_with_ner(content)

    # Initialize results
    wind_speed = None
    rainfall = None
    casualties = None
    damages = None

    # Extract wind speed using enhanced patterns with better coverage
    if not wind_speed:
        # Try multiple wind speed patterns
        wind_patterns = [
            weather_patterns['wind_speed'],
            re.compile(r'(?:gio|gió)\s+(?:manh|toc do)\s+(?:cap|cấp)\s*(\d+)', re.IGNORECASE),
            re.compile(r'(?:suc gio|sức gió)\s+(?:trung binh|trung bình|cuc dai|cực đại)\s+(?:cap|cấp)\s*(\d+)', re.IGNORECASE),
            re.compile(r'(?:gio|gió)\s+(?:cap|cấp)\s*(\d+)', re.IGNORECASE),
            re.compile(r'(?:toc do gio|tốc độ gió)\s+(\d+)(?:\s*km/h|\s*km/gio)', re.IGNORECASE)
        ]

        for pattern in wind_patterns:
            match = pattern.search(content)
            if match:
                if 'km/h' in content.lower() or 'km/gio' in content.lower():
                    # Convert km/h to wind scale (rough approximation)
                    kmh = int(match.group(1))
                    scale = min(17, max(1, round(kmh / 5)))
                    wind_speed = f"cấp {scale}"
                else:
                    wind_speed = f"cấp {match.group(1)}"
                break

        # Try gust_speed pattern
        match = weather_patterns['gust_speed'].search(content)
        if match and wind_speed:
            wind_speed += f", giật cấp {match.group(1)}"

    # Extract rainfall using enhanced patterns with better coverage
    if not rainfall:
        # Try multiple rainfall patterns
        rain_patterns = [
            weather_patterns['rain_amount'],
            re.compile(r'(?:mua|mưa)\s+(?:trong|dat|pho bien|trung binh)\s+(\d+)(?:-(\d+))?\s*mm', re.IGNORECASE),
            re.compile(r'(?:luong mua|lượng mưa)\s+(\d+)(?:-(\d+))?\s*mm', re.IGNORECASE),
            re.compile(r'(?:mua|mưa)\s+(\d+)(?:-(\d+))?\s*mm', re.IGNORECASE),
            re.compile(r'(?:tong luong|tổng lượng)\s+(?:mua|mưa)\s+(\d+)(?:-(\d+))?\s*mm', re.IGNORECASE)
        ]

        for pattern in rain_patterns:
            match = pattern.search(content)
            if match:
                amount = match.group(1)
                if match.group(2):  # Range
                    rainfall = f"{amount}-{match.group(2)} mm"
                else:
                    rainfall = f"{amount} mm"
                break

    # Extract casualties using enhanced damage patterns with better coverage
    if not casualties:
        # Try human_losses pattern
        match = damage_patterns['human_losses'].search(content)
        if match:
            casualties = f"{match.group(1)} người chết"

        # Try injured pattern
        match = damage_patterns['injured'].search(content)
        if match and casualties:
            casualties += f", {match.group(1)} người bị thương"
        elif match:
            casualties = f"{match.group(1)} người bị thương"

        # Try missing pattern
        match = damage_patterns['missing'].search(content)
        if match and casualties:
            casualties += f", {match.group(1)} người mất tích"
        elif match:
            casualties = f"{match.group(1)} người mất tích"

        # Additional casualty patterns for better coverage
        casualty_patterns = [
            re.compile(r'(\d+(?:[.,]\d+)?)\s*(?:nguoi|ca|người)\s*(?:thiet mang|thương vong|mất mạng)', re.IGNORECASE),
            re.compile(r'(?:gay|gay ra|gây ra)\s*(\d+(?:[.,]\d+)?)\s*(?:nguoi|ca|người)\s*(?:chet|chết)', re.IGNORECASE),
            re.compile(r'(?:so nguoi chet|số người chết)\s*[:;]?\s*(\d+(?:[.,]\d+)?)', re.IGNORECASE)
        ]

        for pattern in casualty_patterns:
            match = pattern.search(content)
            if match and not casualties:
                casualties = f"{match.group(1)} người chết"
                break

    # Extract damages using enhanced patterns with better coverage
    if not damages:
        # Try economic_loss pattern
        match = damage_patterns['economic_loss'].search(content)
        if match:
            amount = match.group(1)  # The number
            unit_parts = []
            if len(match.groups()) >= 2 and match.group(2):
                unit_parts.append(match.group(2))  # ty|trieu|nghin etc.
            if len(match.groups()) >= 3 and match.group(3):
                unit_parts.append(match.group(3))  # dong|usd|$ etc.
            unit = ' '.join(unit_parts) if unit_parts else "tỷ đồng"
            damages = f"{amount} {unit}"

        # Additional damage patterns
        damage_patterns_extra = [
            re.compile(r'(?:thiet hai|thiệt hại)\s*(?:kinh te|tai chinh)?\s*[:;]?\s*(\d+(?:[.,]\d+)?)\s*(?:ty|trieu|tỷ|triệu)\s*(?:dong|đồng)?', re.IGNORECASE),
            re.compile(r'(?:ton that|tổn thất)\s*[:;]?\s*(\d+(?:[.,]\d+)?)\s*(?:ty|trieu|tỷ|triệu)', re.IGNORECASE),
            re.compile(r'(?:mat mat|mất mát)\s*[:;]?\s*(\d+(?:[.,]\d+)?)\s*(?:ty|trieu|tỷ|triệu)', re.IGNORECASE)
        ]

        for pattern in damage_patterns_extra:
            match = pattern.search(content)
            if match and not damages:
                damages = f"{match.group(1)} tỷ đồng"
                break

    # Override with NER if available and more specific
    if ner_quantities.get('wind_speed') and not wind_speed:
        wind_speed = ner_quantities['wind_speed']
    if ner_quantities.get('rainfall') and not rainfall:
        rainfall = ner_quantities['rainfall']
    if ner_quantities.get('casualties') and not casualties:
        casualties = ner_quantities['casualties']
    if ner_quantities.get('damages') and not damages:
        damages = ner_quantities['damages']

    return wind_speed, rainfall, casualties, damages

def normalize_damages(damages_str):
    """Chuẩn hóa damages thành số (float)"""
    if damages_str:
        match = re.search(r'(\d+(?:\.\d+)?)', damages_str)
        if match:
            return float(match.group(1))
    return None

def extract_severity_level(wind_speed, disaster_type):
    """Tính mức độ nghiêm trọng dựa trên sức gió và loại thiên tai"""
    if not wind_speed:
        if disaster_type in ['lũ lụt', 'sạt lở đất']:
            return 'Trung bình'
        return 'Không xác định'
    
    match = re.search(r'cấp\s*(\d+)', wind_speed)
    if match:
        level = int(match.group(1))
        if level >= 12:
            return 'Rất nghiêm trọng'
        elif level >= 10:
            return 'Nghiêm trọng'
        elif level >= 8:
            return 'Trung bình'
        else:
            return 'Nhẹ'
    return 'Không xác định'

def extract_impact_area(content, location):
    """Trích xuất khu vực ảnh hưởng dựa trên content và location"""
    if location:
        return location
    
    # Nếu không có location, tìm trong content
    areas = []
    if 'miền Bắc' in content.lower():
        areas.append('Miền Bắc')
    if 'miền Trung' in content.lower():
        areas.append('Miền Trung')
    if 'miền Nam' in content.lower():
        areas.append('Miền Nam')
    if 'đồng bằng sông Cửu Long' in content.lower():
        areas.append('Đồng bằng sông Cửu Long')
    if 'đồng bằng sông Hồng' in content.lower():
        areas.append('Đồng bằng sông Hồng')
    
    return ', '.join(areas) if areas else 'Không xác định'

def extract_forecast(content):
    """Trích xuất phần dự báo (đơn giản: tìm câu chứa 'dự báo')"""
    sentences = re.split(r'[.!?]', content)
    for sent in sentences:
        if 'dự báo' in sent.lower():
            return sent.strip()
    return None

def extract_event_name(content):
    """Trích xuất tên riêng của thiên tai (như tên bão, tên cơn bão, động đất, sóng thần, v.v.)"""
    try:
        # Tìm tên bão riêng với validation chặt chẽ hơn
        storm_patterns = [
            r'mang\s+tên\s+([A-Z][a-z]+)',  # mang tên Danas (ưu tiên pattern này)
            r'mang\s+ten\s+([A-Z][a-z]+)',  # mang ten Danas
            r'bao\s+so\s+\d+\s+([^,\.\d]{3,15})',  # bão số 16 Danas
            r'con\s+bao\s+([^,\.\d]{3,15})',  # cơn bão Danas
            r'(?:bao|bão)\s+([A-Z][a-z]+)',  # bão/bao Yagi (chỉ bắt đầu bằng chữ hoa)
        ]
        for pattern in storm_patterns:
            storm_match = re.search(pattern, content, re.IGNORECASE)
            if storm_match:
                name = storm_match.group(1).strip()
                # Validation chặt chẽ hơn cho tên bão
                if (len(name) >= 3 and len(name) <= 15 and
                    not re.search(r'\d', name) and  # Không chứa số
                    not any(char in name for char in ['(', ')', '[', ']', '{', '}', '|', '\\', '/', '?', '*', '+', '^', '$']) and
                    name.lower() not in ['nhi', 'gồm', 'giông', 'quanh', 'la', 'phủ', 'rộng', 'tại', 'ở', 'vùng', 'khu', 'vực'] and
                    not any(word in name.lower() for word in ['bao', 'nhieu', 'giong', 'bao', 'quanh', 'la', 'luon', 'hinh', 'anh', 'phu', 'rong', 'nhom', 'tre', 'xe', 'oto', 'dac', 'khu', 'con', 'co', 'taluy', 'duong', 'khoang', 'giua', 'dem', 'hon', 'chua', 'nam', 'nao', 'ghi', 'nhan', 'lu', 'lich', 'su', 'dac', 'biet', 'lon', 'cung', 'xuat', 'hien', 'tai', 'song', 'bac', 'bo', 'nghiem', 'trong', 'vuot', 'muc', 'tren', 'cac', 'sông', 'như', 'hien', 'nay', 'tp', 'hcm', 'can', 'canh', 'lich', 'su', 'vua', 'qua'])):
                    return f"Bão {name}"

        # Tìm tên động đất hoặc trận động đất với validation chặt chẽ
        earthquake_match = re.search(r'(?:dong dat|động đất)\s+([^,\.\d]{3,30})', content, re.IGNORECASE)
        if earthquake_match:
            name = earthquake_match.group(1).strip()
            # Validation chặt chẽ: tránh false matches
            exclude_words = ['ở', 'tại', 'khu vực', 'vùng', 'tỉnh', 'thành phố', 'quận', 'huyện', 'xã', 'thị trấn',
                           'do', 'vì', 'của', 'trong', 'ngày', 'tháng', 'năm', 'lúc', 'khi']
            if (len(name) > 2 and len(name) < 25 and
                not any(word in name.lower() for word in exclude_words) and
                not re.search(r'\d', name) and  # Không chứa số
                not any(char in name for char in ['(', ')', '[', ']', '{', '}', '|', '\\', '/', '?', '*', '+', '^', '$']) and  # Không chứa ký tự đặc biệt
                name.lower() not in ['mạnh', 'yếu', 'nhẹ', 'nặng', 'lớn', 'nhỏ', 'tại', 'ở', 'vùng', 'khu vực']):
                return f"Động đất {name}"

        # Tìm tên sóng thần với validation
        tsunami_match = re.search(r'(?:song than|sóng thần)\s+([^,\.\d]{3,25})', content, re.IGNORECASE)
        if tsunami_match:
            name = tsunami_match.group(1).strip()
            exclude_words = ['ở', 'tại', 'khu vực', 'do', 'vì', 'của', 'trong', 'ngày', 'tháng', 'năm']
            if (len(name) > 2 and len(name) < 20 and
                not any(word in name.lower() for word in exclude_words) and
                not re.search(r'\d', name)):
                return f"Sóng thần {name}"

        # Tìm tên núi lửa phun trào với validation
        volcano_match = re.search(r'(?:nui lua|núi lửa)\s+([^,\.\d]{3,25})', content, re.IGNORECASE)
        if volcano_match:
            name = volcano_match.group(1).strip()
            exclude_words = ['ở', 'tại', 'khu vực', 'do', 'vì', 'của', 'trong']
            if (len(name) > 2 and len(name) < 20 and
                not any(word in name.lower() for word in exclude_words) and
                not re.search(r'\d', name)):
                return f"Núi lửa {name}"

        # Tìm tên cháy rừng với patterns mở rộng và validation chặt chẽ
        fire_patterns = [
            r'(?:chay rung|cháy rừng)\s+([^,\.\d]{3,30})',
            r'(?:vuc chay|vùng cháy|rừng cháy)\s+([^,\.\d]{3,30})',
            r'(?:cháy)\s+(?:rừng|khu vực)\s+([^,\.\d]{3,30})'
        ]
        for pattern in fire_patterns:
            fire_match = re.search(pattern, content, re.IGNORECASE)
            if fire_match:
                name = fire_match.group(1).strip()
                exclude_words = ['ở', 'tại', 'do', 'vì', 'của', 'tại', 'trong', 'ngày', 'tháng', 'năm', 'lúc', 'khi']
                if (len(name) > 2 and len(name) < 25 and
                    not any(word in name.lower() for word in exclude_words) and
                    not re.search(r'\d', name) and
                    name.lower() not in ['lớn', 'nhỏ', 'mạnh', 'yếu', 'nhiều', 'ít']):
                    return f"Cháy rừng {name}"

        # Tìm tên dịch bệnh với validation
        epidemic_match = re.search(r'(?:dich benh|dịch bệnh)\s+([^,\.\d]{3,25})', content, re.IGNORECASE)
        if epidemic_match:
            name = epidemic_match.group(1).strip()
            exclude_words = ['ở', 'tại', 'khu vực', 'do', 'vì', 'của', 'trong']
            if (len(name) > 2 and len(name) < 20 and
                not any(word in name.lower() for word in exclude_words) and
                not re.search(r'\d', name)):
                return f"Dịch bệnh {name}"

        # Tìm tên lũ lụt hoặc trận lũ với patterns mở rộng và validation chặt chẽ hơn
        flood_patterns = [
            r'(?:tran lu|trận lũ|lũ lụt)\s+([^,\.\d]{3,25})',
            r'(?:lu lut|lũ lụt|lũ)\s+(?:lớn|to|khổng lồ|khủng khiếp|vượt mức)\s+([^,\.\d]{3,25})',
            r'(?:lu lut|lũ lụt)\s+([^,\.\d]{3,25})'
        ]
        for pattern in flood_patterns:
            flood_match = re.search(pattern, content, re.IGNORECASE)
            if flood_match:
                name = flood_match.group(1).strip()
                exclude_words = ['ở', 'tại', 'khu vực', 'vùng', 'tỉnh', 'thành phố', 'quận', 'huyện', 'xã', 'thị trấn',
                               'do', 'vì', 'của', 'trong', 'ngày', 'tháng', 'năm', 'lúc', 'khi', 'cho', 'của', 'với',
                               'như', 'hiện', 'nay', 'tp', 'hcm', 'cận', 'cảnh', 'lịch', 'sử', 'vừa', 'qua', 'trên',
                               'các', 'sông', 'bắc', 'bộ', 'nghiệm', 'trong', 'vượt', 'mức', 'đặc', 'biệt', 'lớn',
                               'cùng', 'xuất', 'hiện', 'tại', 'sông', 'chưa', 'năm', 'nào', 'ghi', 'nhận', 'lũ',
                               'lịch', 'sử', 'đặc', 'biệt', 'lớn', 'cùng', 'xuất', 'hiện', 'tại', 'sông']
                if (len(name) > 2 and len(name) < 20 and
                    not any(word in name.lower() for word in exclude_words) and
                    not re.search(r'\d', name) and
                    name.lower() not in ['lớn', 'nhỏ', 'mạnh', 'yếu', 'nhiều', 'ít', 'tại', 'ở', 'vùng', 'khu vực', 'đất', 'biển', 'sông', 'suối'] and
                    not any(char in name for char in ['(', ')', '[', ']', '{', '}', '|', '\\', '/', '?', '*', '+', '^', '$'])):
                    return f"Lũ lụt {name}"

        # Tìm tên hạn hán với validation
        drought_match = re.search(r'(?:han han|hạn hán)\s+([^,\.\d]{3,25})', content, re.IGNORECASE)
        if drought_match:
            name = drought_match.group(1).strip()
            exclude_words = ['ở', 'tại', 'khu vực', 'do', 'vì', 'của', 'trong']
            if (len(name) > 2 and len(name) < 20 and
                not any(word in name.lower() for word in exclude_words) and
                not re.search(r'\d', name)):
                return f"Hạn hán {name}"

        # Tìm tên sạt lở/trượt đất với validation chặt chẽ hơn
        landslide_match = re.search(r'(?:sat lo|trượt đất|sạt lở)\s+([^,\.\d]{3,20})', content, re.IGNORECASE)
        if landslide_match:
            name = landslide_match.group(1).strip()
            exclude_words = ['ở', 'tại', 'khu vực', 'do', 'vì', 'của', 'trong', 'ngày', 'tháng', 'năm', 'lúc', 'khi',
                           'đất', 'bờ', 'suối', 'sông', 'biển', 'núi', 'đồi', 'dốc', 'sườn', 'taluy', 'dương', 'khoảng',
                           'giữa', 'đêm', 'hơn', 'chưa', 'năm', 'nào', 'ghi', 'nhận', 'nghiệm', 'trọng', 'bờ', 'sông',
                           'bồ', 'diễn', 'biến', 'phức', 'tạp', 'mùa', 'bão', 'lũ']
            if (len(name) > 2 and len(name) < 15 and
                not any(word in name.lower() for word in exclude_words) and
                not re.search(r'\d', name) and
                name.lower() not in ['lớn', 'nhỏ', 'mạnh', 'yếu', 'nhiều', 'ít', 'tại', 'ở', 'vùng', 'khu vực'] and
                not any(char in name for char in ['(', ')', '[', ']', '{', '}', '|', '\\', '/', '?', '*', '+', '^', '$'])):
                return f"Sạt lở {name}"

        return None
    except Exception as e:
        logger.warning(f"Error extracting event name: {e}")
        return None

def filter_relevant_articles(df):
    """Lọc bỏ các bài viết không liên quan đến thiên tai thực sự và quá cũ"""
    # Keywords indicating relevant disaster articles
    disaster_keywords = [
        'bão', 'áp thấp nhiệt đới', 'lũ', 'lũ quét', 'ngập úng', 'hạn hán', 'xâm nhập mặn',
        'động đất', 'sóng thần', 'núi lửa', 'sạt lở', 'trượt đất', 'cháy rừng', 'ô nhiễm',
        'tràn dầu', 'sự cố hóa chất', 'dịch bệnh', 'sinh vật ngoại lai', 'sương muối',
        'rét đậm', 'rét hại', 'nắng nóng', 'sóng nhiệt', 'mưa lớn', 'lốc xoáy', 'vòi rồng'
    ]
    
    def is_relevant(row):
        title = str(row.get('title', '')).lower()
        content = str(row.get('content', '')).lower()
        disaster_type = str(row.get('disaster_type', '')).lower()
        
        # Check if any keyword appears in title, content, or disaster_type
        text_to_check = title + ' ' + content + ' ' + disaster_type
        return any(keyword in text_to_check for keyword in disaster_keywords)
    
    initial_count = len(df)
    df_filtered = df[df.apply(is_relevant, axis=1)]
    
    # Filter by date: keep articles from 2020 onwards
    if 'date' in df_filtered.columns:
        df_filtered['date'] = pd.to_datetime(df_filtered['date'], errors='coerce')
        df_filtered = df_filtered[df_filtered['date'].dt.year >= 2020]
    
    filtered_count = len(df_filtered)
    logger.info(f"Filtered out {initial_count - filtered_count} irrelevant/old articles")
    
    return df_filtered

def clean_duplicates(df):
    """Loại bỏ duplicate dựa trên url và nội dung tương tự"""
    # First, remove exact URL duplicates
    df = df.drop_duplicates(subset=['url'], keep='first')
    
    # Then, remove near-duplicates based on title similarity (optional, can be expensive)
    # For now, just keep URL deduplication as it's fast and effective
    
    return df

def main():
    logger.info("Starting enhanced CSV export process")
    
    # Tìm file JSON mới nhất trong thư mục data
    json_files = glob.glob('data/disaster_data_multisource_*.json')
    if not json_files:
        logger.error("❌ Không tìm thấy file JSON disaster_data_multisource!")
        return
    
    json_file = max(json_files)  # Lấy file mới nhất
    logger.info(f"📂 Đang xử lý file: {json_file}")
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Tạo DataFrame từ JSON
    df = pd.DataFrame(data)
    logger.info(f"Loaded {len(df)} articles")

    # Lọc bỏ bài viết không liên quan
    df = filter_relevant_articles(df)

    # Làm sạch: Loại duplicate dựa trên url
    df = clean_duplicates(df)
    logger.info(f"After deduplication: {len(df)} articles")

    # Cập nhật cột với NLP using batch processing for better performance
    logger.info("Extracting locations...")
    df['location'] = [extract_location(content) for content in tqdm(df['content'], desc="Location extraction")]

    logger.info("Extracting numerical data...")
    numerical_data = [extract_numbers(content) for content in tqdm(df['content'], desc="Numerical extraction")]
    df[['wind_speed', 'rainfall', 'casualties', 'damages']] = pd.DataFrame(numerical_data, index=df.index)

    df['damages_normalized'] = df['damages'].apply(normalize_damages)
    df['forecast'] = df['content'].apply(extract_forecast)
    df['event_name'] = df['content'].apply(extract_event_name)

    # Thêm cột mới
    df['severity_level'] = df.apply(lambda row: extract_severity_level(row['wind_speed'], row['disaster_type']), axis=1)
    df['impact_area'] = df.apply(lambda row: extract_impact_area(row['content'], row['location']), axis=1)

    # Sắp xếp cột (loại bỏ content để nhẹ file)
    columns = [
        'date', 'disaster_type', 'event_name', 'location', 'impact_area', 'severity_level', 'title', 'source', 'category',
        'wind_speed', 'rainfall', 'casualties', 'damages', 'damages_normalized', 'forecast',
        'url', 'scrape_time'
    ]
    df = df[columns]

    # Xuất CSV mới
    new_csv_file = 'data/disaster_data_enhanced.csv'
    df.to_csv(new_csv_file, index=False, encoding='utf-8-sig')
    logger.info(f"Đã làm sạch và xuất CSV mới: {new_csv_file}")
    logger.info(f"Số dòng sau làm sạch: {len(df)}")

    # Hiển thị thống kê
    logger.info("Generating statistics...")
    print(f"\nData Quality Summary:")
    for col in ['event_name', 'location', 'wind_speed', 'rainfall', 'casualties', 'damages']:
        if col in df.columns:
            filled = df[col].notna().sum()
            rate = (filled / len(df)) * 100
            print(f"  {col}: {filled}/{len(df)} ({rate:.1f}%)")
    
    # Validation: Show sample of extracted data for manual check
    print(f"\nSample Validation (first 5 rows):")
    sample_cols = ['title', 'event_name', 'location', 'wind_speed', 'rainfall', 'casualties', 'damages']
    print(df[sample_cols].head().to_string(index=False))
    
    # Cross-reference check: Compare with known sources (placeholder)
    logger.info("Validation complete - manual review recommended for accuracy")

if __name__ == "__main__":
    try:
        main()
        logger.info("Process completed successfully")
    except Exception as e:
        logger.error(f"Process failed: {e}", exc_info=True)