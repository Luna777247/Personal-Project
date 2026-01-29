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
from functools import lru_cache
import numpy as np
from collections import Counter
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('disaster_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==================== MODEL INITIALIZATION ====================
class ModelManager:
    """Quản lý các models NLP với lazy loading"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.nlp_spacy = None
            self.ner_pipeline = None
            self.initialized = True
            self._load_models()
    
    def _load_models(self):
        """Load models với error handling"""
        # Load spaCy
        try:
            self.nlp_spacy = spacy.load("vi_core_news_lg")
            logger.info("✅ Loaded Vietnamese spaCy model")
        except:
            try:
                self.nlp_spacy = spacy.load("en_core_web_sm")
                logger.warning("⚠️ Using English spaCy fallback")
            except:
                logger.warning("❌ No spaCy model available")
        
        # Load Hugging Face NER
        try:
            self.ner_pipeline = pipeline(
                "ner",
                model="NlpHUST/ner-vietnamese-electra-base",
                aggregation_strategy="simple",
                device=-1  # CPU
            )
            logger.info("✅ Loaded Hugging Face NER model")
        except:
            logger.warning("❌ Hugging Face NER unavailable")

models = ModelManager()

# ==================== ENHANCED REGEX PATTERNS ====================
class PatternLibrary:
    """Thư viện patterns được tối ưu và mở rộng"""
    
    # Weather patterns
    WEATHER = {
        'wind_speed': [
            re.compile(r'(?:sức\s+gió|tốc\s+độ\s+gió|gió\s+mạnh)\s+(?:cấp\s+)?(\d+)', re.IGNORECASE),
            re.compile(r'gió\s+(?:cấp|cap)\s*(\d+)', re.IGNORECASE),
            re.compile(r'(?:mạnh|cực\s+đại)\s+cấp\s*(\d+)', re.IGNORECASE),
        ],
        'rainfall': [
            re.compile(r'(?:lượng\s+mưa|mưa)\s+(?:phổ\s+biến\s+)?(\d+)(?:-(\d+))?\s*mm', re.IGNORECASE),
            re.compile(r'tổng\s+lượng\s+mưa\s+(\d+)(?:-(\d+))?\s*mm', re.IGNORECASE),
            re.compile(r'mưa\s+từ\s+(\d+)\s+đến\s+(\d+)\s*mm', re.IGNORECASE),
        ],
        'temperature': [
            re.compile(r'nhiệt\s+độ\s+(?:cao\s+nhất\s+)?(\d+)(?:-(\d+))?\s*°?[Cc]', re.IGNORECASE),
            re.compile(r'nóng\s+(?:lên\s+)?(?:đến\s+)?(\d+)\s*độ', re.IGNORECASE),
        ]
    }
    
    # Damage patterns
    DAMAGE = {
        'casualties': [
            re.compile(r'(\d+(?:[.,]\d+)?)\s*(?:người|ca)\s*(?:chết|tử\s+vong|thiệt\s+mạng)', re.IGNORECASE),
            re.compile(r'(?:làm\s+)?chết\s+(\d+)\s*người', re.IGNORECASE),
            re.compile(r'số\s+người\s+chết\s*[:;]?\s*(\d+)', re.IGNORECASE),
        ],
        'injured': [
            re.compile(r'(\d+)\s*(?:người|ca)\s*(?:bị\s+thương|thương\s+vong)', re.IGNORECASE),
        ],
        'missing': [
            re.compile(r'(\d+)\s*người\s*(?:mất\s+tích|bị\s+mất\s+tích)', re.IGNORECASE),
        ],
        'economic': [
            re.compile(r'(?:thiệt\s+hại|tổn\s+thất)\s*(?:khoảng\s+)?(\d+(?:[.,]\d+)?)\s*(tỷ|triệu|nghìn)?\s*(?:đồng|USD|\$)?', re.IGNORECASE),
            re.compile(r'(\d+(?:[.,]\d+)?)\s*(tỷ|triệu)\s*đồng\s*thiệt\s+hại', re.IGNORECASE),
        ],
        'houses': [
            re.compile(r'(\d+(?:[.,]\d+)?)\s*(?:căn|ngôi)\s*nhà\s*(?:bị\s+)?(?:sập|hư\s+hỏng|tốc\s+mái)', re.IGNORECASE),
            re.compile(r'hư\s+hỏng\s+(\d+(?:[.,]\d+)?)\s*(?:căn|ngôi)\s*nhà', re.IGNORECASE),
        ]
    }
    
    # Location patterns (Vietnamese specific)
    LOCATIONS = {
        'provinces': [
            'Hà Nội', 'TP.HCM', 'Hồ Chí Minh', 'Đà Nẵng', 'Hải Phòng', 'Cần Thơ',
            'Quảng Ninh', 'Thanh Hóa', 'Nghệ An', 'Hà Tĩnh', 'Quảng Bình', 'Quảng Trị',
            'Thừa Thiên Huế', 'Quảng Nam', 'Quảng Ngãi', 'Bình Định', 'Phú Yên',
            'Khánh Hòa', 'Ninh Thuận', 'Bình Thuận', 'Kon Tum', 'Gia Lai', 'Đắk Lắk',
            'Đắk Nông', 'Lâm Đồng', 'Bình Phước', 'Tây Ninh', 'Bình Dương', 'Đồng Nai',
            'Bà Rịa Vũng Tàu', 'Long An', 'Tiền Giang', 'Bến Tre', 'Trà Vinh',
            'Vĩnh Long', 'Đồng Tháp', 'An Giang', 'Kiên Giang', 'Cà Mau', 'Bạc Liêu',
            'Sóc Trăng', 'Hậu Giang', 'Lào Cai', 'Yên Bái', 'Sơn La', 'Lai Châu',
            'Điện Biên', 'Hòa Bình', 'Thái Nguyên', 'Bắc Kạn', 'Cao Bằng', 'Lạng Sơn',
            'Hà Giang', 'Tuyên Quang', 'Phú Thọ', 'Vĩnh Phúc', 'Bắc Ninh', 'Bắc Giang',
            'Hải Dương', 'Hưng Yên', 'Thái Bình', 'Hà Nam', 'Nam Định', 'Ninh Bình'
        ],
        'regions': [
            'Miền Bắc', 'Miền Trung', 'Miền Nam', 'Bắc Bộ', 'Trung Bộ', 'Nam Bộ',
            'Đồng bằng sông Hồng', 'Đồng bằng sông Cửu Long', 'Tây Nguyên',
            'Trung du miền núi Bắc Bộ', 'Duyên hải Nam Trung Bộ', 'Đông Nam Bộ'
        ],
        'seas': ['Biển Đông', 'Hoàng Sa', 'Trường Sa', 'Vịnh Bắc Bộ', 'Vịnh Hạ Long']
    }

patterns = PatternLibrary()

# ==================== EXTRACTION FUNCTIONS ====================
class DataExtractor:
    """Class chứa các hàm trích xuất được tối ưu"""
    
    @staticmethod
    @lru_cache(maxsize=1000)
    def clean_text(text: str) -> str:
        """Làm sạch text với caching"""
        text = unicodedata.normalize('NFKC', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    @staticmethod
    def extract_location_nlp(content: str) -> List[str]:
        """Trích xuất location bằng NLP models"""
        locations = []
        
        # spaCy extraction
        if models.nlp_spacy:
            try:
                doc = models.nlp_spacy(content[:5000])  # Limit length
                for ent in doc.ents:
                    if ent.label_ in ['GPE', 'LOC', 'FAC']:
                        locations.append(ent.text)
            except Exception as e:
                logger.debug(f"spaCy extraction failed: {e}")
        
        # Hugging Face NER
        if models.ner_pipeline:
            try:
                clean_content = DataExtractor.clean_text(content[:1000])
                entities = models.ner_pipeline(clean_content)
                for ent in entities:
                    if ent['entity_group'] == 'LOC' and not ent['word'].startswith('##'):
                        locations.append(ent['word'])
            except Exception as e:
                logger.debug(f"HF NER failed: {e}")
        
        return list(set(locations))
    
    @staticmethod
    def extract_location_rule(content: str) -> List[str]:
        """Rule-based location extraction"""
        found = []
        content_lower = content.lower()
        
        # Check provinces
        for province in patterns.LOCATIONS['provinces']:
            if province.lower() in content_lower:
                found.append(province)
        
        # Check regions
        for region in patterns.LOCATIONS['regions']:
            if region.lower() in content_lower:
                found.append(region)
        
        # Check seas
        for sea in patterns.LOCATIONS['seas']:
            if sea.lower() in content_lower:
                found.append(sea)
        
        return list(set(found))
    
    @staticmethod
    def extract_location(content: str) -> Optional[str]:
        """Kết hợp NLP và rule-based để trích xuất location"""
        nlp_locs = DataExtractor.extract_location_nlp(content)
        rule_locs = DataExtractor.extract_location_rule(content)
        
        # Merge và ưu tiên rule-based (reliable hơn cho VN locations)
        all_locs = list(set(rule_locs + nlp_locs))
        
        # Filter: chỉ giữ locations hợp lệ
        valid_locs = []
        for loc in all_locs:
            loc_clean = loc.strip()
            if (len(loc_clean) >= 3 and 
                not loc_clean.startswith('##') and 
                not re.match(r'^\W+$', loc_clean)):
                valid_locs.append(loc_clean)
        
        return ', '.join(valid_locs[:5]) if valid_locs else None  # Limit to 5 locations
    
    @staticmethod
    def extract_with_patterns(content: str, pattern_list: List) -> Optional[str]:
        """Extract data using multiple patterns"""
        for pattern in pattern_list:
            match = pattern.search(content)
            if match:
                return match
        return None
    
    @staticmethod
    def extract_wind_speed(content: str) -> Optional[str]:
        """Trích xuất sức gió"""
        match = DataExtractor.extract_with_patterns(content, patterns.WEATHER['wind_speed'])
        if match:
            level = match.group(1)
            # Check for gust
            gust_match = re.search(r'giật\s+cấp\s*(\d+)', content, re.IGNORECASE)
            if gust_match:
                return f"cấp {level}, giật cấp {gust_match.group(1)}"
            return f"cấp {level}"
        return None
    
    @staticmethod
    def extract_rainfall(content: str) -> Optional[str]:
        """Trích xuất lượng mưa"""
        match = DataExtractor.extract_with_patterns(content, patterns.WEATHER['rainfall'])
        if match:
            if match.group(2):  # Range
                return f"{match.group(1)}-{match.group(2)} mm"
            return f"{match.group(1)} mm"
        return None
    
    @staticmethod
    def extract_casualties(content: str) -> Optional[str]:
        """Trích xuất thiệt hại về người"""
        parts = []
        
        # Deaths
        match = DataExtractor.extract_with_patterns(content, patterns.DAMAGE['casualties'])
        if match:
            num = match.group(1).replace(',', '.')
            parts.append(f"{num} người chết")
        
        # Injured
        match = DataExtractor.extract_with_patterns(content, patterns.DAMAGE['injured'])
        if match:
            parts.append(f"{match.group(1)} người bị thương")
        
        # Missing
        match = DataExtractor.extract_with_patterns(content, patterns.DAMAGE['missing'])
        if match:
            parts.append(f"{match.group(1)} người mất tích")
        
        return ', '.join(parts) if parts else None
    
    @staticmethod
    def extract_damages(content: str) -> Optional[str]:
        """Trích xuất thiệt hại kinh tế"""
        match = DataExtractor.extract_with_patterns(content, patterns.DAMAGE['economic'])
        if match:
            amount = match.group(1).replace(',', '.')
            unit = match.group(2) if len(match.groups()) >= 2 and match.group(2) else 'tỷ'
            return f"{amount} {unit} đồng"
        return None
    
    @staticmethod
    def extract_event_name(content: str) -> Optional[str]:
        """Trích xuất tên sự kiện với validation chặt chẽ"""
        # Storm names
        storm_patterns = [
            r'(?:bão|cơn\s+bão)\s+([A-Z][a-z]{2,12})',
            r'mang\s+tên\s+([A-Z][a-z]{2,12})',
        ]
        
        exclude_words = {'Nhi', 'Gồm', 'Giông', 'Vùng', 'Khu', 'Tại', 'Nhiều', 'Lớn'}
        
        for pattern in storm_patterns:
            match = re.search(pattern, content)
            if match:
                name = match.group(1).strip()
                if (len(name) >= 3 and 
                    name not in exclude_words and
                    not any(char.isdigit() for char in name)):
                    return f"Bão {name}"
        
        # Other disasters
        disaster_types = [
            ('động đất', 'Động đất'),
            ('sóng thần', 'Sóng thần'),
            ('núi lửa', 'Núi lửa'),
            ('cháy rừng', 'Cháy rừng'),
            ('lũ lụt', 'Lũ lụt'),
        ]
        
        for keyword, prefix in disaster_types:
            pattern = re.compile(rf'{keyword}\s+([^,\.\d]{{3,20}})', re.IGNORECASE)
            match = pattern.search(content)
            if match:
                name = match.group(1).strip()
                if len(name) >= 3 and len(name) <= 20:
                    return f"{prefix} {name}"
        
        return None
    
    @staticmethod
    def calculate_severity(wind_speed: Optional[str], disaster_type: str) -> str:
        """Tính mức độ nghiêm trọng"""
        if not wind_speed:
            return 'Trung bình' if disaster_type in ['lũ lụt', 'sạt lở'] else 'Không xác định'
        
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

extractor = DataExtractor()

# ==================== DATA PROCESSING ====================
def process_single_article(row: pd.Series) -> Dict:
    """Xử lý một article với error handling"""
    try:
        content = str(row.get('content', ''))
        disaster_type = str(row.get('disaster_type', ''))
        
        return {
            'location': extractor.extract_location(content),
            'wind_speed': extractor.extract_wind_speed(content),
            'rainfall': extractor.extract_rainfall(content),
            'casualties': extractor.extract_casualties(content),
            'damages': extractor.extract_damages(content),
            'event_name': extractor.extract_event_name(content),
            'severity_level': None  # Will be calculated after
        }
    except Exception as e:
        logger.warning(f"Error processing article: {e}")
        return {k: None for k in ['location', 'wind_speed', 'rainfall', 
                                   'casualties', 'damages', 'event_name', 'severity_level']}

def parallel_process_articles(df: pd.DataFrame, n_workers: int = None) -> pd.DataFrame:
    """Xử lý song song với multiprocessing"""
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)
    
    logger.info(f"Processing {len(df)} articles with {n_workers} workers...")
    
    with mp.Pool(n_workers) as pool:
        results = list(tqdm(
            pool.imap(process_single_article, [row for _, row in df.iterrows()]),
            total=len(df),
            desc="Processing articles"
        ))
    
    # Merge results back to dataframe
    result_df = pd.DataFrame(results)
    for col in result_df.columns:
        df[col] = result_df[col].values
    
    # Calculate severity
    df['severity_level'] = df.apply(
        lambda row: extractor.calculate_severity(row['wind_speed'], row.get('disaster_type', '')),
        axis=1
    )
    
    return df

def filter_relevant_articles(df: pd.DataFrame) -> pd.DataFrame:
    """Lọc bài viết liên quan với improved logic"""
    disaster_keywords = [
        'bão', 'áp thấp nhiệt đới', 'lũ', 'lũ quét', 'ngập', 'hạn hán',
        'động đất', 'sóng thần', 'núi lửa', 'sạt lở', 'trượt', 'cháy rừng',
        'ô nhiễm', 'dịch', 'rét', 'nắng nóng', 'mưa lớn', 'lốc'
    ]
    
    def is_relevant(row):
        text = f"{row.get('title', '')} {row.get('content', '')} {row.get('disaster_type', '')}".lower()
        return any(kw in text for kw in disaster_keywords)
    
    initial = len(df)
    df_filtered = df[df.apply(is_relevant, axis=1)].copy()
    
    # Filter by date (2020+)
    if 'date' in df_filtered.columns:
        df_filtered['date'] = pd.to_datetime(df_filtered['date'], errors='coerce')
        df_filtered = df_filtered[df_filtered['date'].dt.year >= 2020]
    
    logger.info(f"Filtered: {initial} → {len(df_filtered)} articles")
    return df_filtered

def clean_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Làm sạch duplicates"""
    initial = len(df)
    df = df.drop_duplicates(subset=['url'], keep='first')
    logger.info(f"Removed {initial - len(df)} duplicate URLs")
    return df

def generate_statistics(df: pd.DataFrame):
    """Generate comprehensive statistics"""
    print("\n" + "="*70)
    print("DATA QUALITY REPORT".center(70))
    print("="*70)
    
    print(f"\n📊 Total Articles: {len(df)}")
    
    print("\n🎯 Field Completion Rates:")
    for col in ['event_name', 'location', 'wind_speed', 'rainfall', 'casualties', 'damages']:
        if col in df.columns:
            filled = df[col].notna().sum()
            rate = (filled / len(df)) * 100
            bar = "█" * int(rate / 2) + "░" * (50 - int(rate / 2))
            print(f"  {col:20s} [{bar}] {rate:5.1f}% ({filled}/{len(df)})")
    
    print("\n📅 Date Distribution:")
    if 'date' in df.columns:
        df['year'] = pd.to_datetime(df['date'], errors='coerce').dt.year
        year_counts = df['year'].value_counts().sort_index()
        for year, count in year_counts.items():
            if pd.notna(year):
                print(f"  {int(year)}: {count} articles")
    
    print("\n🌪️  Disaster Type Distribution:")
    if 'disaster_type' in df.columns:
        type_counts = df['disaster_type'].value_counts().head(10)
        for dtype, count in type_counts.items():
            print(f"  {dtype:30s}: {count}")
    
    print("\n⚠️  Severity Level Distribution:")
    if 'severity_level' in df.columns:
        severity_counts = df['severity_level'].value_counts()
        for level, count in severity_counts.items():
            print(f"  {level:30s}: {count}")
    
    print("\n" + "="*70)

# ==================== MAIN FUNCTION ====================
def main():
    logger.info("="*70)
    logger.info("ENHANCED DISASTER DATA PROCESSING SYSTEM v2.0")
    logger.info("="*70)
    
    # Find latest JSON file
    json_files = glob.glob('data/disaster_data_multisource_*.json')
    if not json_files:
        logger.error("❌ No JSON file found!")
        return
    
    json_file = max(json_files)
    logger.info(f"📂 Processing: {json_file}")
    
    # Load data
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    logger.info(f"✅ Loaded {len(df)} articles")
    
    # Filter and clean
    df = filter_relevant_articles(df)
    df = clean_duplicates(df)
    
    # Process articles (with parallel processing)
    df = parallel_process_articles(df, n_workers=4)
    
    # Add computed columns
    df['impact_area'] = df.apply(
        lambda row: row['location'] if row['location'] else 'Không xác định',
        axis=1
    )
    
    df['damages_normalized'] = df['damages'].apply(
        lambda x: float(re.search(r'(\d+(?:\.\d+)?)', str(x)).group(1)) 
        if x and re.search(r'(\d+(?:\.\d+)?)', str(x)) else None
    )
    
    # Select columns
    columns = [
        'date', 'disaster_type', 'event_name', 'location', 'impact_area',
        'severity_level', 'title', 'source', 'category',
        'wind_speed', 'rainfall', 'casualties', 'damages', 'damages_normalized',
        'url', 'scrape_time'
    ]
    
    df = df[[col for col in columns if col in df.columns]]
    
    # Export
    output_file = 'data/disaster_data_enhanced_v2.csv'
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    logger.info(f"✅ Exported to: {output_file}")
    
    # Generate statistics
    generate_statistics(df)
    
    # Sample validation
    print("\n📋 Sample Data (First 5 rows):")
    sample_cols = ['title', 'event_name', 'location', 'wind_speed', 'casualties']
    print(df[[col for col in sample_cols if col in df.columns]].head().to_string(index=False))
    
    logger.info("\n✅ Processing completed successfully!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"❌ Process failed: {e}", exc_info=True)
        raise