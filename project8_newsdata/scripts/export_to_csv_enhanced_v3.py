import json
import pandas as pd
import re
import glob
from datetime import datetime
import spacy
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import unicodedata
import logging
from tqdm import tqdm
import multiprocessing as mp
from functools import lru_cache
import numpy as np
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional, Set
import warnings
from difflib import SequenceMatcher
import torch
import os
warnings.filterwarnings('ignore')

# ==================== ADVANCED CONFIGURATION ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('disaster_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==================== ENHANCED MODEL MANAGER ====================
class AdvancedModelManager:
    """Quản lý models với caching và fallback strategies"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.nlp_spacy = None
            self.ner_pipeline = None
            self.tokenizer = None
            self.confidence_threshold = 0.85
            self.initialized = True
            self._load_models()
    
    def _load_models(self):
        """Load models với progressive fallback"""
        # spaCy (primary for Vietnamese)
        for model_name in ["vi_core_news_lg", "vi_core_news_md", "en_core_web_sm"]:
            try:
                self.nlp_spacy = spacy.load(model_name)
                logger.info(f"✅ Loaded spaCy: {model_name}")
                break
            except:
                continue
        
        # Hugging Face NER (với confidence scoring)
        try:
            model_name = "NlpHUST/ner-vietnamese-electra-base"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForTokenClassification.from_pretrained(model_name)
            self.ner_pipeline = pipeline(
                "ner",
                model=model,
                tokenizer=self.tokenizer,
                aggregation_strategy="simple",
                device=-1
            )
            logger.info("✅ Loaded HF NER with confidence scoring")
        except Exception as e:
            logger.warning(f"⚠️ HF NER unavailable: {e}")

models = AdvancedModelManager()

# ==================== DYNAMIC KNOWLEDGE BASE ====================
class DynamicKnowledgeBase:
    """Dynamic Knowledge Base với auto-update capabilities"""
    
    def __init__(self, config_path: str = "config/knowledge_base.json"):
        self.config_path = config_path
        self._data = {}
        self._last_update = None
        self._update_interval = 86400  # 24 hours
        self._load_or_create_kb()
        
        # Auto-update sources
        self.storm_sources = [
            "https://www.nhc.noaa.gov/data/tstorms-atl.dat",  # Atlantic storms
            "https://www.nhc.noaa.gov/data/tstorms-epac.dat",  # Pacific storms
        ]
    
    def _load_or_create_kb(self):
        """Load từ file hoặc tạo default"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._data = json.load(f)
            logger.info(f"✅ Loaded knowledge base from {self.config_path}")
        except FileNotFoundError:
            logger.warning(f"⚠️ Knowledge base file not found, creating default")
            self._create_default_kb()
            self.save()
    
    def _create_default_kb(self):
        """Tạo knowledge base mặc định"""
        self._data = {
            "provinces": KnowledgeBase.PROVINCES.copy(),
            "regions": KnowledgeBase.REGIONS.copy(), 
            "storm_names": list(KnowledgeBase.STORM_NAMES.copy()),
            "severity_keywords": KnowledgeBase.SEVERITY_KEYWORDS.copy(),
            "metadata": {
                "version": "1.0",
                "last_updated": datetime.now().isoformat(),
                "auto_discovered_provinces": [],
                "auto_discovered_storms": [],
                "confidence_scores": {}
            }
        }
    
    def save(self):
        """Lưu knowledge base ra file"""
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)
        logger.info(f"💾 Saved knowledge base to {self.config_path}")
    
    # ==================== PROVINCES MANAGEMENT ====================
    def get_provinces(self) -> Dict[str, List[str]]:
        """Get provinces với aliases"""
        return self._data.get("provinces", {})
    
    def add_province(self, official_name: str, aliases: List[str] = None, confidence: float = 0.8):
        """Thêm province mới với validation"""
        if aliases is None:
            aliases = []
        
        # Validate format
        if not official_name or not isinstance(official_name, str):
            raise ValueError("Invalid province name")
        
        # Check if already exists
        if official_name in self._data["provinces"]:
            logger.warning(f"Province {official_name} already exists")
            return
        
        self._data["provinces"][official_name] = aliases
        self._data["metadata"]["auto_discovered_provinces"].append({
            "name": official_name,
            "aliases": aliases,
            "added_date": datetime.now().isoformat(),
            "confidence": confidence
        })
        self.save()
        logger.info(f"✅ Added province: {official_name}")
    
    def discover_provinces_from_data(self, df: pd.DataFrame):
        """Auto-discover provinces từ processed data"""
        discovered = set()
        
        # Extract potential locations từ NER results
        for _, row in df.iterrows():
            if pd.notna(row.get('location')):
                locations = str(row['location']).split(',')
                for loc in locations:
                    loc = loc.strip()
                    # Check if looks like Vietnamese location
                    if self._is_potential_province(loc):
                        discovered.add(loc)
        
        # Validate và add new provinces
        for loc in discovered:
            if loc not in self._data["provinces"]:
                # Auto-generate aliases
                aliases = self._generate_aliases(loc)
                self.add_province(loc, aliases, confidence=0.6)
    
    def _is_potential_province(self, text: str) -> bool:
        """Check if text có thể là province name"""
        # Vietnamese location patterns
        if re.search(r'(tỉnh|thành phố|tp\.?|thành phố)', text.lower()):
            return True
        
        # Length và character checks
        if 3 <= len(text) <= 30 and any(char in text for char in [' ', 'Đ', 'đ']):
            return True
        
        return False
    
    def _generate_aliases(self, province: str) -> List[str]:
        """Tự động generate aliases cho province"""
        aliases = []
        name = province.lower()
        
        # Remove common prefixes
        name = re.sub(r'^(tỉnh|thành phố|tp\.?)\s+', '', name)
        
        # Add common variations
        if 'đà nẵng' in name:
            aliases.extend(['Da Nang', 'Danang'])
        elif 'hồ chí minh' in name:
            aliases.extend(['HCM', 'Sài Gòn', 'Saigon', 'TP.HCM'])
        elif 'hà nội' in name:
            aliases.extend(['Ha Noi', 'Hanoi'])
        
        return aliases
    
    # ==================== STORM MANAGEMENT ====================
    def get_storm_names(self) -> Set[str]:
        """Get storm names"""
        return set(self._data.get("storm_names", []))
    
    def add_storm_name(self, name: str, source: str = "auto", confidence: float = 0.9):
        """Thêm storm name mới"""
        if not name or not isinstance(name, str):
            raise ValueError("Invalid storm name")
        
        name = name.strip().title()
        
        if name in self._data["storm_names"]:
            return
        
        self._data["storm_names"].append(name)
        self._data["metadata"]["auto_discovered_storms"].append({
            "name": name,
            "source": source,
            "added_date": datetime.now().isoformat(),
            "confidence": confidence
        })
        self.save()
        logger.info(f"✅ Added storm: {name}")
    
    def update_storm_names_from_web(self):
        """Auto-update storm names từ web sources"""
        try:
            import requests
            from bs4 import BeautifulSoup
            
            # Scrape từ WMO storm names
            url = "https://www.wmo.int/pages/prog/www/tcp/Storm-naming.html"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                # Extract storm names (simplified)
                storm_text = soup.get_text()
                potential_storms = re.findall(r'\b[A-Z][a-z]+\b', storm_text)
                
                for storm in potential_storms:
                    if len(storm) >= 3 and storm not in self.get_storm_names():
                        self.add_storm_name(storm, "wmo_web_scrape", 0.7)
                        
        except Exception as e:
            logger.warning(f"Failed to update storms from web: {e}")
    
    def discover_storms_from_data(self, df: pd.DataFrame):
        """Auto-discover storms từ processed data"""
        for _, row in df.iterrows():
            content = str(row.get('content', ''))
            title = str(row.get('title', ''))
            text = title + ' ' + content
            
            # Look for storm patterns
            storm_matches = re.findall(r'(?:bão|cơn bão)\s+([A-Z][a-z]{2,12})', text, re.IGNORECASE)
            
            for match in storm_matches:
                match = match.strip().title()
                if match not in self.get_storm_names():
                    # Validate it's likely a storm name
                    if self.validate_storm_name(match):
                        self.add_storm_name(match, "data_discovery", 0.8)
    
    def validate_storm_name(self, name: str) -> bool:
        """Validate if a name is a known storm name"""
        if not name:
            return False
        
        name = name.strip().title()
        storm_names = self.get_storm_names()
        
        # Exact match
        if name in storm_names:
            return True
        
        # Fuzzy match with high similarity
        return any(
            SequenceMatcher(None, name, storm).ratio() > 0.85 
            for storm in storm_names
        )
    
    # ==================== REGIONS MANAGEMENT ====================
    def get_regions(self) -> Dict[str, List[str]]:
        """Get regions"""
        return self._data.get("regions", {})
    
    def add_region(self, name: str, aliases: List[str] = None):
        """Thêm region mới"""
        if aliases is None:
            aliases = []
        
        if name in self._data["regions"]:
            return
        
        self._data["regions"][name] = aliases
        self.save()
        logger.info(f"✅ Added region: {name}")
    
    # ==================== AUTO-UPDATE SYSTEM ====================
    def auto_update(self, df: pd.DataFrame = None):
        """Auto-update knowledge base"""
        current_time = datetime.now()
        
        # Check if update needed
        if self._last_update and (current_time - self._last_update).seconds < self._update_interval:
            return
        
        logger.info("🔄 Auto-updating knowledge base...")
        
        try:
            # Update storms from web
            self.update_storm_names_from_web()
            
            # Discover from data if provided
            if df is not None:
                self.discover_provinces_from_data(df)
                self.discover_storms_from_data(df)
            
            # Update metadata
            self._data["metadata"]["last_updated"] = current_time.isoformat()
            self._last_update = current_time
            self.save()
            
            logger.info("✅ Knowledge base auto-updated")
            
        except Exception as e:
            logger.error(f"❌ Auto-update failed: {e}")
    
    # ==================== UTILITY METHODS ====================
    def get_stats(self) -> Dict:
        """Get knowledge base statistics"""
        return {
            "provinces_count": len(self._data.get("provinces", {})),
            "regions_count": len(self._data.get("regions", {})),
            "storm_names_count": len(self._data.get("storm_names", [])),
            "auto_discovered_provinces": len(self._data["metadata"].get("auto_discovered_provinces", [])),
            "auto_discovered_storms": len(self._data["metadata"].get("auto_discovered_storms", [])),
            "last_updated": self._data["metadata"].get("last_updated")
        }
    
    def export_for_review(self, output_path: str):
        """Export knowledge base for manual review"""
        review_data = {
            "auto_discovered_provinces": self._data["metadata"].get("auto_discovered_provinces", []),
            "auto_discovered_storms": self._data["metadata"].get("auto_discovered_storms", []),
            "confidence_scores": self._data["metadata"].get("confidence_scores", {}),
            "export_date": datetime.now().isoformat()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(review_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"📋 Exported knowledge base for review: {output_path}")

# ==================== BACKWARD COMPATIBILITY ====================
class KnowledgeBase:
    """Backward compatibility wrapper"""
    
    # Static data for fallback
    PROVINCES = {
        'Hà Nội': ['Ha Noi', 'Hanoi', 'Thủ đô'],
        'TP.HCM': ['Ho Chi Minh', 'Hồ Chí Minh', 'Sài Gòn', 'Saigon', 'TPHCM', 'HCM'],
        'Đà Nẵng': ['Da Nang', 'Danang'],
        'Hải Phòng': ['Hai Phong', 'Haiphong'],
        'Cần Thơ': ['Can Tho', 'Cantho'],
        'Quảng Ninh': ['Quang Ninh'],
        'Thanh Hóa': ['Thanh Hoa'],
        'Nghệ An': ['Nghe An'],
        'Hà Tĩnh': ['Ha Tinh'],
        'Quảng Bình': ['Quang Binh'],
        'Quảng Trị': ['Quang Tri'],
        'Thừa Thiên Huế': ['Thua Thien Hue', 'Huế', 'Hue'],
        'Quảng Nam': ['Quang Nam'],
        'Quảng Ngãi': ['Quang Ngai'],
        'Bình Định': ['Binh Dinh'],
        'Phú Yên': ['Phu Yen'],
        'Khánh Hòa': ['Khanh Hoa', 'Nha Trang'],
        'Ninh Thuận': ['Ninh Thuan'],
        'Bình Thuận': ['Binh Thuan'],
        'Kon Tum': ['Kontum'],
        'Gia Lai': ['Pleiku'],
        'Đắk Lắk': ['Dak Lak', 'Daklak', 'Buôn Ma Thuột'],
        'Đắk Nông': ['Dak Nong'],
        'Lâm Đồng': ['Lam Dong', 'Đà Lạt', 'Da Lat'],
        'Bình Phước': ['Binh Phuoc'],
        'Tây Ninh': ['Tay Ninh'],
        'Bình Dương': ['Binh Duong'],
        'Đồng Nai': ['Dong Nai'],
        'Bà Rịa Vũng Tàu': ['Ba Ria Vung Tau', 'Vung Tau'],
        'Long An': [],
        'Tiền Giang': ['Tien Giang'],
        'Bến Tre': ['Ben Tre'],
        'Trà Vinh': ['Tra Vinh'],
        'Vĩnh Long': ['Vinh Long'],
        'Đồng Tháp': ['Dong Thap'],
        'An Giang': [],
        'Kiên Giang': ['Kien Giang', 'Phú Quốc'],
        'Cà Mau': ['Ca Mau'],
        'Bạc Liêu': ['Bac Lieu'],
        'Sóc Trăng': ['Soc Trang'],
        'Hậu Giang': ['Hau Giang'],
    }
    
    REGIONS = {
        'Miền Bắc': ['Bắc Bộ', 'Bac Bo', 'phía Bắc'],
        'Miền Trung': ['Trung Bộ', 'Trung Bo', 'phía Trung'],
        'Miền Nam': ['Nam Bộ', 'Nam Bo', 'phía Nam'],
        'Đồng bằng sông Hồng': ['ĐBSH', 'delta song Hong'],
        'Đồng bằng sông Cửu Long': ['ĐBSCL', 'delta Mekong', 'Đồng bằng Mê Kông'],
        'Tây Nguyên': ['Tay Nguyen', 'Central Highlands'],
    }
    
    STORM_NAMES = {
        'Yagi', 'Damrey', 'Doksuri', 'Khanun', 'Lan', 'Saola', 'Haikui',
        'Kirogi', 'Kai-tak', 'Tembin', 'Bolaven', 'Sanba', 'Jelawat',
        'Usagi', 'Pabuk', 'Wutip', 'Sepat', 'Fitow', 'Danas', 'Nari',
        'Wipha', 'Francisco', 'Lekima', 'Krosa', 'Haiyan', 'Podul',
        'Lingling', 'Kajiki', 'Faxai', 'Peipah', 'Tapah', 'Mitag',
        'Hagibis', 'Neoguri', 'Bualoi', 'Matmo', 'Halong', 'Nakri',
        'Fengshen', 'Kalmaegi', 'Fung-wong', 'Kammuri', 'Phanfone',
        'Vongfong', 'Nuri', 'Sinlaku', 'Hagupit', 'Jangmi', 'Mekkhala',
        'Higos', 'Bavi', 'Maysak', 'Haishen', 'Noul', 'Dolphin', 'Kujira',
        'Chan-hom', 'Linfa', 'Nangka', 'Soudelor', 'Molave', 'Goni', 'Atsani',
        'Etau', 'Vamco', 'Krovanh', 'Dujuan', 'Mujigae', 'Koppu', 'Champ-mi',
    }
    
    SEVERITY_KEYWORDS = {
        'Rất nghiêm trọng': ['siêu bão', 'đại hồng thủy', 'thảm họa', 'khủng khiếp', 'tàn phá nặng nề'],
        'Nghiêm trọng': ['nghiêm trọng', 'mạnh', 'lớn', 'thiệt hại nặng', 'cấp độ cao'],
        'Trung bình': ['trung bình', 'vừa phải', 'ảnh hưởng đáng kể'],
        'Nhẹ': ['nhẹ', 'ít ảnh hưởng', 'thiệt hại nhỏ'],
    }
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = DynamicKnowledgeBase()
        return cls._instance
    
    @classmethod
    def normalize_location(cls, location: str) -> Optional[str]:
        return cls._instance.normalize_location(location)
    
    @classmethod
    def validate_storm_name(cls, name: str) -> bool:
        return cls._instance.validate_storm_name(name)

# Initialize global instance
kb = KnowledgeBase()

# ==================== CONTEXT-AWARE EXTRACTION ====================
class ContextAwareExtractor:
    """Trích xuất dựa trên context và semantic understanding"""
    
    @staticmethod
    def extract_with_context(content: str, target: str, window: int = 100) -> List[str]:
        """Trích xuất với context window"""
        results = []
        sentences = re.split(r'[.!?]+', content)
        
        for i, sentence in enumerate(sentences):
            if target.lower() in sentence.lower():
                # Get surrounding context
                start = max(0, i - 1)
                end = min(len(sentences), i + 2)
                context = ' '.join(sentences[start:end])
                results.append(context.strip())
        
        return results
    
    @staticmethod
    def extract_time_events(content: str) -> Dict[str, List[Dict[str, any]]]:
        """Trích xuất thời gian bắt đầu, đỉnh, và kết thúc của thiên tai"""
        time_patterns = {
            'start_time': [
                (r'(?:bắt đầu|bùng phát|xảy ra|diễn ra)\s+(?:từ\s+)?(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}\s+(?:tháng|giờ)\s+\d{1,2}[/-]\d{2,4}|\d{1,2}:\d{2}\s+\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', 'explicit_start'),
                (r'(?:khoảng|từ)\s+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\s+(?:đến|cho đến)', 'range_start'),
                (r'(?:sáng|chiều|tối|đêm)\s+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', 'time_of_day_start'),
                # Simple date patterns
                (r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', 'simple_date'),
                (r'tháng\s+(\d{1,2}[/-]\d{4})', 'month_year'),
                (r'(\d{4})', 'year_only'),  # Just year if mentioned
            ],
            'peak_time': [
                (r'(?:đạt đỉnh|cao nhất|mạnh nhất|gay gắt nhất)\s+(?:vào\s+)?(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}:\d{2}\s+\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', 'peak_explicit'),
                (r'(?:lúc|cách đây)\s+(\d+)\s+(?:giờ|tiếng)\s+(?:trước|qua)', 'peak_relative'),
                (r'(?:trong\s+)?(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\s+(?:đã|thì)\s+(?:đạt|mạnh)', 'peak_context'),
            ],
            'end_time': [
                (r'(?:kết thúc|chấm dứt|dừng lại)\s+(?:vào\s+)?(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}:\d{2}\s+\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', 'end_explicit'),
                (r'(?:đến\s+)?(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\s+(?:thì|đã)\s+(?:kết thúc|dừng)', 'end_context'),
                (r'(?:kéo dài\s+)?(?:đến\s+)?(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\s+(?:thì|đã)\s+(?:hết|dứt)', 'end_duration'),
            ]
        }
        
        results = defaultdict(list)
        
        for event_type, pattern_list in time_patterns.items():
            for pattern, context_type in pattern_list:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    time_str = match.group(1)
                    
                    # Get context for validation
                    start = max(0, match.start() - 100)
                    end = min(len(content), match.end() + 100)
                    context = content[start:end]
                    
                    # Parse time
                    parsed_time = ContextAwareExtractor._parse_time_string(time_str)
                    
                    if parsed_time:
                        results[event_type].append({
                            'time_string': time_str,
                            'parsed_time': parsed_time,
                            'context_type': context_type,
                            'context': context.strip(),
                            'confidence': ContextAwareExtractor._calculate_time_confidence(context_type, context)
                        })
        
        return results
    
    @staticmethod
    def _parse_time_string(time_str: str) -> Optional[datetime]:
        """Parse time string thành datetime object"""
        # Normalize separators
        time_str = re.sub(r'[/-]', '-', time_str)
        
        # Common Vietnamese date formats
        formats = [
            '%d-%m-%Y',      # 17-10-2025
            '%d-%m-%y',      # 17-10-25
            '%d/%m/%Y',      # 17/10/2025
            '%d/%m/%y',      # 17/10/25
            '%H:%M %d-%m-%Y', # 14:30 17-10-2025
            '%H:%M %d/%m/%Y', # 14:30 17/10/2025
            '%d %m %Y',      # 17 10 2025 (space separated)
            '%m-%Y',         # 10-2023 (month-year)
            '%m/%Y',         # 10/2023
            '%Y',            # 2023 (year only)
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(time_str, fmt)
            except ValueError:
                continue
        
        # Try with current year if only day-month
        if re.match(r'\d{1,2}-\d{1,2}$', time_str):
            current_year = datetime.now().year
            try:
                return datetime.strptime(f"{time_str}-{current_year}", '%d-%m-%Y')
            except ValueError:
                pass
        
        return None
    
    @staticmethod
    def _calculate_time_confidence(context_type: str, context: str) -> float:
        """Tính confidence cho time extraction"""
        base_confidence = {
            'explicit_start': 0.95,
            'explicit_end': 0.95,
            'peak_explicit': 0.90,
            'range_start': 0.85,
            'time_of_day_start': 0.80,
            'peak_relative': 0.75,
            'peak_context': 0.70,
            'end_context': 0.85,
            'end_duration': 0.80,
            'simple_date': 0.60,  # Lower confidence for generic dates
            'month_year': 0.55,   # Lower confidence for month-year only
        }
        
        confidence = base_confidence.get(context_type, 0.5)
        
        # Boost if context mentions disaster terms
        disaster_keywords = ['bão', 'lũ', 'động đất', 'hạn hán', 'cháy rừng', 'sóng thần']
        if any(keyword in context.lower() for keyword in disaster_keywords):
            confidence += 0.05
        
        # Boost if has specific time indicators
        time_indicators = ['lúc', 'vào', 'từ', 'đến', 'khoảng', 'xảy ra']
        if any(indicator in context.lower() for indicator in time_indicators):
            confidence += 0.03
        
        return min(confidence, 1.0)
    
    @staticmethod
    def extract_numbers_with_units(content: str) -> Dict[str, List[Tuple[float, Optional[str], str]]]:
        """Trích xuất số liệu với units và context"""
        patterns = {
            'wind': [
                (r'(?:gió|sức gió|tốc độ gió)\s+(?:mạnh\s+)?(?:cấp\s+)?(\d+)(?:-(\d+))?\s*(cấp|km/h|m/s)?', 'wind_speed'),
                (r'giật\s+(?:cấp\s+)?(\d+)', 'gust'),
            ],
            'rain': [
                (r'(?:mưa|lượng mưa)\s+(?:đạt\s+)?(\d+)(?:-(\d+))?\s*mm', 'rainfall'),
                (r'tổng\s+lượng\s+(?:mưa|nước)\s+(\d+)(?:-(\d+))?\s*mm', 'total_rainfall'),
            ],
            'casualties': [
                (r'(\d+)\s*người\s*(?:chết|tử vong|thiệt mạng)', 'deaths'),
                (r'(\d+)\s*người\s*(?:bị thương|thương tích)', 'injured'),
                (r'(\d+)\s*người\s*(?:mất tích|bị mất tích)', 'missing'),
            ],
            'economic': [
                (r'(?:thiệt hại|tổn thất|mất mát)\s*(?:ước tính\s+)?(\d+(?:[.,]\d+)?)\s*(tỷ|triệu|nghìn)?\s*(?:đồng|VND)', 'damage_vnd'),
                (r'(\d+(?:[.,]\d+)?)\s*(?:tỷ|triệu)\s*USD', 'damage_usd'),
            ],
            'area': [
                (r'(\d+(?:[.,]\d+)?)\s*(?:ha|hecta|héc-ta)\s*(?:ruộng|đất|rừng)', 'affected_area'),
            ],
            'structures': [
                (r'(\d+(?:[.,]\d+)?)\s*(?:căn|ngôi)\s*nhà\s*(?:bị\s+)?(?:sập|hư|tốc mái)', 'houses_damaged'),
                (r'(\d+)\s*cây\s*cầu\s*(?:bị\s+)?(?:sập|hư|cuốn trôi)', 'bridges_damaged'),
            ]
        }
        
        results = defaultdict(list)
        
        for category, pattern_list in patterns.items():
            for pattern, key in pattern_list:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    value = float(match.group(1).replace(',', '.'))
                    unit = match.group(2) if len(match.groups()) >= 2 else None
                    
                    # Get context for validation
                    start = max(0, match.start() - 50)
                    end = min(len(content), match.end() + 50)
                    context = content[start:end]
                    
                    results[key].append((value, unit, context))
        
        return results
    
    @staticmethod
    def resolve_ambiguity(candidates: List[Tuple[str, float]]) -> Optional[str]:
        """Giải quyết trường hợp nhiều candidates với confidence scores"""
        if not candidates:
            return None
        
        # Sort by confidence
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # If top candidate significantly better than others
        if len(candidates) == 1 or candidates[0][1] > candidates[1][1] + 0.15:
            return candidates[0][0]
        
        # Otherwise, use voting or return most common
        values = [c[0] for c in candidates]
        return max(set(values), key=values.count)

# ==================== MULTI-STRATEGY EXTRACTOR ====================
class MultiStrategyExtractor:
    """Kết hợp nhiều strategies để tăng accuracy"""
    
    def __init__(self):
        self.context_extractor = ContextAwareExtractor()
    
    def extract_location_multi(self, content: str) -> Dict[str, any]:
        """Multi-strategy location extraction với confidence"""
        strategies = {
            'ner_spacy': self._extract_loc_spacy,
            'ner_hf': self._extract_loc_hf,
            'rule_based': self._extract_loc_rules,
            'context': self._extract_loc_context,
        }
        
        all_results = []
        
        for strategy_name, strategy_func in strategies.items():
            try:
                locs = strategy_func(content)
                for loc in locs:
                    all_results.append({
                        'location': loc,
                        'strategy': strategy_name,
                        'confidence': self._calculate_confidence(loc, strategy_name)
                    })
            except Exception as e:
                logger.debug(f"Strategy {strategy_name} failed: {e}")
        
        # Aggregate and normalize
        return self._aggregate_locations(all_results)
    
    def _extract_loc_spacy(self, content: str) -> List[str]:
        """spaCy extraction"""
        if not models.nlp_spacy:
            return []
        
        doc = models.nlp_spacy(content[:5000])
        return [ent.text for ent in doc.ents if ent.label_ in ['GPE', 'LOC', 'FAC']]
    
    def _extract_loc_hf(self, content: str) -> List[str]:
        """HuggingFace extraction"""
        if not models.ner_pipeline:
            return []
        
        content_clean = re.sub(r'\s+', ' ', content[:1000])
        entities = models.ner_pipeline(content_clean)
        return [ent['word'] for ent in entities 
                if ent['entity_group'] == 'LOC' and ent['score'] > 0.8]
    
    def _extract_loc_rules(self, content: str) -> List[str]:
        """Rule-based extraction"""
        found = []
        content_lower = content.lower()
        
        # Check all known locations
        for official, aliases in kb.PROVINCES.items():
            if official.lower() in content_lower:
                found.append(official)
            for alias in aliases:
                if alias.lower() in content_lower:
                    found.append(official)
                    break
        
        for official, aliases in kb.REGIONS.items():
            if official.lower() in content_lower:
                found.append(official)
        
        return list(set(found))
    
    def _extract_loc_context(self, content: str) -> List[str]:
        """Context-based extraction"""
        location_contexts = self.context_extractor.extract_with_context(
            content, 'tại|ở|vùng|khu vực'
        )
        
        locations = []
        for context in location_contexts:
            # Extract locations from context
            for official in kb.PROVINCES.keys():
                if official.lower() in context.lower():
                    locations.append(official)
        
        return locations
    
    def _calculate_confidence(self, location: str, strategy: str) -> float:
        """Tính confidence score"""
        base_confidence = {
            'rule_based': 0.95,  # Highest for known locations
            'ner_spacy': 0.85,
            'ner_hf': 0.80,
            'context': 0.75,
        }
        
        confidence = base_confidence.get(strategy, 0.5)
        
        # Boost if in knowledge base
        if kb.normalize_location(location) in kb.PROVINCES:
            confidence += 0.05
        
        return min(confidence, 1.0)
    
    def _aggregate_locations(self, results: List[Dict]) -> Dict:
        """Aggregate results from multiple strategies"""
        if not results:
            return {'locations': [], 'confidence': 0.0}
        
        # Group by normalized location
        location_groups = defaultdict(list)
        for result in results:
            normalized = kb.normalize_location(result['location'])
            location_groups[normalized].append(result['confidence'])
        
        # Calculate aggregated confidence
        final_locations = []
        for loc, confidences in location_groups.items():
            avg_confidence = np.mean(confidences)
            num_strategies = len(confidences)
            
            # Boost confidence if multiple strategies agree
            if num_strategies > 1:
                avg_confidence = min(1.0, avg_confidence * (1 + 0.1 * (num_strategies - 1)))
            
            if avg_confidence > 0.6:  # Threshold
                final_locations.append({
                    'location': loc,
                    'confidence': avg_confidence,
                    'num_sources': num_strategies
                })
        
        # Sort by confidence
        final_locations.sort(key=lambda x: x['confidence'], reverse=True)
        
        return {
            'locations': [loc['location'] for loc in final_locations[:5]],  # Top 5
            'confidence': np.mean([loc['confidence'] for loc in final_locations]) if final_locations else 0.0,
            'details': final_locations
        }
    
    def extract_time_multi(self, content: str) -> Dict[str, any]:
        """Multi-strategy time event extraction"""
        # Use context-aware extractor as primary method
        time_events = self.context_extractor.extract_time_events(content)
        
        # Process and aggregate results
        aggregated = {}
        
        for event_type in ['start_time', 'peak_time', 'end_time']:
            if event_type in time_events:
                candidates = time_events[event_type]
                
                if candidates:
                    # Sort by confidence and select best
                    candidates.sort(key=lambda x: x['confidence'], reverse=True)
                    best_candidate = candidates[0]
                    
                    # Validate time is reasonable (not in future, not too old)
                    if self._validate_time_event(best_candidate['parsed_time']):
                        aggregated[event_type] = {
                            'time_string': best_candidate['time_string'],
                            'parsed_time': best_candidate['parsed_time'].isoformat(),
                            'confidence': best_candidate['confidence'],
                            'context': best_candidate['context'][:200]  # Limit context length
                        }
                    else:
                        # Use time string even if parsing failed
                        aggregated[event_type] = {
                            'time_string': best_candidate['time_string'],
                            'parsed_time': None,
                            'confidence': best_candidate['confidence'] * 0.8,  # Penalty for invalid time
                            'context': best_candidate['context'][:200]
                        }
        
        return aggregated
    
    def _validate_time_event(self, parsed_time: datetime) -> bool:
        """Validate extracted time is reasonable"""
        if not parsed_time:
            return False
        
        now = datetime.now()
        
        # Not in future (allow small tolerance for timezone)
        if parsed_time > now + pd.Timedelta(days=1):
            return False
        
        # Not too old (before 2000)
        if parsed_time.year < 2000:
            return False
        
        # Not too far in past (more than 25 years ago)
        if parsed_time < now - pd.Timedelta(days=365*25):
            return False
        
        return True
    
    def extract_event_name_advanced(self, content: str, disaster_type: str) -> Dict:
        """Advanced event name extraction với validation"""
        candidates = []
        
        # Strategy 1: Named storms
        if 'bão' in disaster_type.lower():
            storm_patterns = [
                r'(?:bão|cơn\s+bão)\s+([A-Z][a-z]{2,12})',
                r'(?:mang\s+tên|tên\s+là)\s+([A-Z][a-z]{2,12})',
                r'bão\s+số\s+\d+\s+\(([A-Z][a-z]+)\)',
            ]
            
            for pattern in storm_patterns:
                matches = re.finditer(pattern, content)
                for match in matches:
                    name = match.group(1).strip()
                    
                    # Validate against known storms
                    if kb.validate_storm_name(name):
                        candidates.append({
                            'name': f"Bão {name}",
                            'confidence': 0.95,
                            'method': 'validated_storm'
                        })
                    elif len(name) >= 3 and name.isalpha():
                        candidates.append({
                            'name': f"Bão {name}",
                            'confidence': 0.75,
                            'method': 'pattern_match'
                        })
        
        # Strategy 2: Date-based events
        date_match = re.search(r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})', content)
        if date_match:
            day, month, year = date_match.groups()
            candidates.append({
                'name': f"{disaster_type} {day}/{month}/{year}",
                'confidence': 0.70,
                'method': 'date_based'
            })
        
        # Strategy 3: Location-based
        location_result = self.extract_location_multi(content)
        if location_result['locations']:
            main_location = location_result['locations'][0]
            candidates.append({
                'name': f"{disaster_type} {main_location}",
                'confidence': 0.65,
                'method': 'location_based'
            })
        
        # Select best candidate
        if candidates:
            candidates.sort(key=lambda x: x['confidence'], reverse=True)
            return candidates[0]
        
        return {'name': None, 'confidence': 0.0, 'method': None}
    
    def extract_numbers_validated(self, content: str) -> Dict:
        """Extract numbers với cross-validation"""
        raw_extractions = self.context_extractor.extract_numbers_with_units(content)
        
        validated = {}
        
        # Wind speed validation
        if 'wind_speed' in raw_extractions:
            wind_values = [v[0] for v in raw_extractions['wind_speed']]
            # Validate range (1-17 for wind scale, 0-300 for km/h)
            valid_winds = [w for w in wind_values if 1 <= w <= 17 or 10 <= w <= 300]
            if valid_winds:
                avg_wind = np.median(valid_winds)
                if avg_wind > 17:  # Likely km/h, convert to scale
                    wind_scale = min(17, max(1, round(avg_wind / 10)))
                else:
                    wind_scale = int(avg_wind)
                validated['wind_speed'] = f"cấp {wind_scale}"
        
        # Rainfall validation
        if 'rainfall' in raw_extractions:
            rain_values = [v[0] for v in raw_extractions['rainfall']]
            valid_rain = [r for r in rain_values if 0 <= r <= 2000]  # Max reasonable rainfall
            if valid_rain:
                validated['rainfall'] = f"{int(np.median(valid_rain))} mm"
        
        # Casualties validation (cross-check deaths, injured, missing)
        casualties_parts = []
        if 'deaths' in raw_extractions:
            deaths = [v[0] for v in raw_extractions['deaths']]
            if deaths:
                casualties_parts.append(f"{int(np.median(deaths))} người chết")
        
        if 'injured' in raw_extractions:
            injured = [v[0] for v in raw_extractions['injured']]
            if injured:
                casualties_parts.append(f"{int(np.median(injured))} người bị thương")
        
        if 'missing' in raw_extractions:
            missing = [v[0] for v in raw_extractions['missing']]
            if missing:
                casualties_parts.append(f"{int(np.median(missing))} người mất tích")
        
        if casualties_parts:
            validated['casualties'] = ', '.join(casualties_parts)
        
        # Economic damages validation
        if 'damage_vnd' in raw_extractions:
            damages = raw_extractions['damage_vnd']
            if damages:
                value, unit, _ = damages[0]  # Take first/most prominent
                unit_text = unit if unit else 'tỷ'
                validated['damages'] = f"{value:.1f} {unit_text} đồng"
        
        return validated

# ==================== MAIN ENHANCED PROCESSOR ====================
class EnhancedProcessor:
    """Main processor với multi-strategy extraction"""
    
    def __init__(self):
        self.multi_extractor = MultiStrategyExtractor()
    
    def process_article(self, row: pd.Series) -> Dict:
        """Process single article với full validation"""
        try:
            content = str(row.get('content', ''))
            disaster_type = str(row.get('disaster_type', ''))
            
            # Multi-strategy extraction
            location_result = self.multi_extractor.extract_location_multi(content)
            event_result = self.multi_extractor.extract_event_name_advanced(content, disaster_type)
            numbers = self.multi_extractor.extract_numbers_validated(content)
            time_result = self.multi_extractor.extract_time_multi(content)
            
            return {
                'location': ', '.join(location_result['locations']) if location_result['locations'] else None,
                'location_confidence': location_result['confidence'],
                'event_name': event_result['name'],
                'event_confidence': event_result['confidence'],
                'wind_speed': numbers.get('wind_speed'),
                'rainfall': numbers.get('rainfall'),
                'casualties': numbers.get('casualties'),
                'damages': numbers.get('damages'),
                'start_time': time_result.get('start_time', {}).get('time_string') if time_result.get('start_time') else None,
                'start_time_confidence': time_result.get('start_time', {}).get('confidence') if time_result.get('start_time') else None,
                'peak_time': time_result.get('peak_time', {}).get('time_string') if time_result.get('peak_time') else None,
                'peak_time_confidence': time_result.get('peak_time', {}).get('confidence') if time_result.get('peak_time') else None,
                'end_time': time_result.get('end_time', {}).get('time_string') if time_result.get('end_time') else None,
                'end_time_confidence': time_result.get('end_time', {}).get('confidence') if time_result.get('end_time') else None,
                'extraction_quality': self._calculate_quality_score(location_result, event_result, numbers)
            }
        except Exception as e:
            logger.warning(f"Error processing article: {e}")
            return {k: None for k in ['location', 'event_name', 'wind_speed', 
                                      'rainfall', 'casualties', 'damages', 
                                      'location_confidence', 'event_confidence',
                                      'start_time', 'start_time_confidence',
                                      'peak_time', 'peak_time_confidence',
                                      'end_time', 'end_time_confidence',
                                      'extraction_quality']}
    
    def _calculate_quality_score(self, location_result, event_result, numbers) -> float:
        """Calculate overall extraction quality score"""
        scores = []
        
        # Location quality
        if location_result['locations']:
            scores.append(location_result['confidence'])
        
        # Event quality
        if event_result['name']:
            scores.append(event_result['confidence'])
        
        # Numbers completeness
        num_fields = len([v for v in numbers.values() if v is not None])
        scores.append(num_fields / 4)  # 4 possible fields
        
        return np.mean(scores) if scores else 0.0

# ==================== PARALLEL PROCESSING ====================
def parallel_process_enhanced(df: pd.DataFrame, n_workers: int = None) -> pd.DataFrame:
    """Enhanced parallel processing"""
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)
    
    processor = EnhancedProcessor()
    logger.info(f"Processing {len(df)} articles with {n_workers} workers...")
    
    # Process in batches
    batch_size = 100
    results = []
    
    for i in tqdm(range(0, len(df), batch_size), desc="Processing batches"):
        batch = df.iloc[i:i+batch_size]
        batch_results = [processor.process_article(row) for _, row in batch.iterrows()]
        results.extend(batch_results)
    
    # Merge results
    result_df = pd.DataFrame(results)
    for col in result_df.columns:
        df[col] = result_df[col].values
    
    return df

# ==================== QUALITY ASSURANCE ====================
def quality_check(df: pd.DataFrame) -> pd.DataFrame:
    """Perform quality checks và flag suspicious data"""
    df['qa_flags'] = ''
    
    # Flag 1: Low extraction quality
    if 'extraction_quality' in df.columns:
        df.loc[df['extraction_quality'] < 0.5, 'qa_flags'] += 'LOW_QUALITY;'
    
    # Flag 2: Missing critical fields
    critical_fields = ['location', 'disaster_type']
    for field in critical_fields:
        if field in df.columns:
            df.loc[df[field].isna(), 'qa_flags'] += f'MISSING_{field.upper()};'
    
    # Flag 3: Suspicious numbers
    if 'casualties' in df.columns:
        # Extract death count
        df['death_count'] = df['casualties'].apply(
            lambda x: int(re.search(r'(\d+)\s*người chết', str(x)).group(1)) 
            if x and re.search(r'(\d+)\s*người chết', str(x)) else 0
        )
        df.loc[df['death_count'] > 10000, 'qa_flags'] += 'SUSPICIOUS_CASUALTIES;'
    
    return df

# ==================== STATISTICS & REPORTING ====================
def generate_advanced_statistics(df: pd.DataFrame):
    """Generate comprehensive statistics with quality metrics"""
    print("\n" + "="*80)
    print("ADVANCED DATA QUALITY REPORT".center(80))
    print("="*80)
    
    print(f"\n📊 Dataset Overview:")
    print(f"  Total articles: {len(df)}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}" if 'date' in df.columns else "")
    
    print(f"\n🎯 Extraction Quality Metrics:")
    if 'extraction_quality' in df.columns:
        # Filter out None values
        quality_values = df['extraction_quality'].dropna()
        if len(quality_values) > 0:
            quality_bins = pd.cut(quality_values, bins=[0, 0.5, 0.7, 0.85, 1.0],
                                 labels=['Poor', 'Fair', 'Good', 'Excellent'])
            quality_dist = quality_bins.value_counts().sort_index()
            for level, count in quality_dist.items():
                pct = (count / len(df)) * 100
            bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
            print(f"  {str(level):12s} [{bar}] {pct:5.1f}% ({count})")
    
    print(f"\n📍 Location Extraction:")
    if 'location' in df.columns:
        has_location = df['location'].notna().sum()
        print(f"  Articles with location: {has_location}/{len(df)} ({has_location/len(df)*100:.1f}%)")
        
        if 'location_confidence' in df.columns:
            avg_conf = df[df['location'].notna()]['location_confidence'].mean()
            print(f"  Average confidence: {avg_conf:.2f}")
        
        # Top locations
        all_locs = []
        for locs in df['location'].dropna():
            all_locs.extend([l.strip() for l in str(locs).split(',')])
        
        if all_locs:
            loc_counts = Counter(all_locs).most_common(10)
            print(f"\n  Top 10 affected locations:")
            for loc, count in loc_counts:
                print(f"    {loc:30s}: {count:3d} articles")
    
    print(f"\n🌪️  Event Names:")
    if 'event_name' in df.columns:
        has_event = df['event_name'].notna().sum()
        print(f"  Articles with event name: {has_event}/{len(df)} ({has_event/len(df)*100:.1f}%)")
        
        if 'event_confidence' in df.columns:
            avg_conf = df[df['event_name'].notna()]['event_confidence'].mean()
            print(f"  Average confidence: {avg_conf:.2f}")
        
        # Most mentioned events
        event_counts = df['event_name'].value_counts().head(10)
        print(f"\n  Top 10 events:")
        for event, count in event_counts.items():
            print(f"    {event:40s}: {count:3d} articles")
    
    print(f"\n💰 Economic Impact:")
    if 'damages' in df.columns:
        has_damage = df['damages'].notna().sum()
        print(f"  Articles with damage data: {has_damage}/{len(df)} ({has_damage/len(df)*100:.1f}%)")
    
    print(f"\n⚠️  Quality Assurance Flags:")
    if 'qa_flags' in df.columns:
        flagged = (df['qa_flags'] != '').sum()
        print(f"  Articles flagged: {flagged}/{len(df)} ({flagged/len(df)*100:.1f}%)")
        
        # Count flag types
        flag_types = defaultdict(int)
        for flags in df['qa_flags']:
            if flags:
                for flag in str(flags).split(';'):
                    if flag:
                        flag_types[flag] += 1
        
        if flag_types:
            print(f"\n  Flag distribution:")
            for flag, count in sorted(flag_types.items(), key=lambda x: x[1], reverse=True):
                print(f"    {flag:30s}: {count:3d}")
    
    print("\n" + "="*80)

# ==================== MAIN FUNCTION ====================
def main():
    logger.info("="*80)
    logger.info("ADVANCED MULTI-STRATEGY DISASTER DATA EXTRACTION v3.0")
    logger.info("="*80)
    
    # Find latest JSON
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
    
    # Filter relevant articles
    disaster_keywords = [
        'bão', 'lũ', 'ngập', 'hạn', 'động đất', 'sóng thần', 
        'sạt lở', 'cháy', 'dịch', 'rét', 'nắng', 'mưa'
    ]
    
    def is_relevant(row):
        text = f"{row.get('title', '')} {row.get('content', '')}".lower()
        return any(kw in text for kw in disaster_keywords)
    
    df = df[df.apply(is_relevant, axis=1)].copy()
    df = df.drop_duplicates(subset=['url'], keep='first')
    logger.info(f"✅ After filtering: {len(df)} articles")
    
    # Process with advanced extraction
    df = parallel_process_enhanced(df, n_workers=4)
    
    # Auto-update knowledge base with new discoveries
    logger.info("🔄 Updating knowledge base with new discoveries...")
    kb.auto_update(df)
    
    # Log knowledge base stats
    kb_stats = kb.get_stats()
    logger.info(f"📊 Knowledge Base Stats: {kb_stats}")
    
    # Export knowledge base for review if significant changes
    if kb_stats['auto_discovered_provinces'] > 0 or kb_stats['auto_discovered_storms'] > 0:
        review_file = 'data/knowledge_base_review.json'
        kb.export_for_review(review_file)
        logger.info(f"📋 New discoveries exported for review: {review_file}")
    
    # Quality assurance
    df = quality_check(df)
    
    # Calculate severity
    def calc_severity(row):
        wind = row.get('wind_speed', '')
        if wind and 'cấp' in str(wind):
            level = int(re.search(r'cấp\s*(\d+)', str(wind)).group(1))
            if level >= 12:
                return 'Rất nghiêm trọng'
            elif level >= 10:
                return 'Nghiêm trọng'
            elif level >= 8:
                return 'Trung bình'
            else:
                return 'Nhẹ'
        return 'Không xác định'
    
    df['severity_level'] = df.apply(calc_severity, axis=1)
    
    # Select columns for export
    export_cols = [
        'date', 'disaster_type', 'event_name', 'event_confidence',
        'location', 'location_confidence', 'severity_level',
        'start_time', 'start_time_confidence',
        'peak_time', 'peak_time_confidence',
        'end_time', 'end_time_confidence',
        'title', 'source', 'category',
        'wind_speed', 'rainfall', 'casualties', 'damages',
        'extraction_quality', 'qa_flags',
        'url', 'scrape_time'
    ]
    
    df_export = df[[col for col in export_cols if col in df.columns]]
    
    # Export
    output_file = 'data/disaster_data_enhanced_v3.csv'
    df_export.to_csv(output_file, index=False, encoding='utf-8-sig')
    logger.info(f"✅ Exported to: {output_file}")
    
    # Generate statistics
    generate_advanced_statistics(df_export)
    
    # Export high-quality subset
    if 'extraction_quality' in df_export.columns:
        high_quality = df_export[df_export['extraction_quality'] >= 0.7]
        hq_file = 'data/disaster_data_high_quality_v3.csv'
        high_quality.to_csv(hq_file, index=False, encoding='utf-8-sig')
        logger.info(f"✅ High-quality subset ({len(high_quality)} articles): {hq_file}")
    
    logger.info("\n✅ Processing completed successfully!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"❌ Process failed: {e}", exc_info=True)
        raise