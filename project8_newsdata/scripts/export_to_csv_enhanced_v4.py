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
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import numpy as np
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional, Set
import warnings
from difflib import SequenceMatcher
import torch
import os
import sqlite3
from contextlib import contextmanager
import hashlib
from simhash import Simhash
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity

# Try import Vietnamese NLP tools
try:
    from underthesea import ner as underthesea_ner
    UNDERTHESEA_AVAILABLE = True
except:
    UNDERTHESEA_AVAILABLE = False

try:
    from pyvi import ViTokenizer
    PYVI_AVAILABLE = True
except:
    PYVI_AVAILABLE = False

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

# ==================== COMPILED REGEX PATTERNS ====================
class CompiledPatterns:
    """Pre-compiled regex patterns for performance"""
    
    # Wind patterns
    WIND_PATTERNS = [
        re.compile(r'(?:gió|sức gió|tốc độ gió)\s+(?:mạnh\s+)?(?:cấp\s+)?(\d+)(?:-(\d+))?\s*(cấp|km/h|m/s)?', re.IGNORECASE),
        re.compile(r'giật\s+(?:cấp\s+)?(\d+)', re.IGNORECASE),
        re.compile(r'(?:gió\s+)?(?:mạnh\s+)?cấp\s+(\d+)(?:\s*-\s*(\d+))?', re.IGNORECASE),
        re.compile(r'(?:vận tốc|tốc độ)\s+(\d+)\s*(?:km/h|m/s)', re.IGNORECASE),
    ]
    
    # Rain patterns
    RAIN_PATTERNS = [
        re.compile(r'(?:mưa|lượng mưa)\s+(?:đạt\s+)?(\d+)(?:-(\d+))?\s*mm', re.IGNORECASE),
        re.compile(r'tổng\s+lượng\s+(?:mưa|nước)\s+(\d+)(?:-(\d+))?\s*mm', re.IGNORECASE),
        re.compile(r'(?:lượng\s+)?mưa\s+(?:lớn|to|rất to)\s+(\d+)\s*mm', re.IGNORECASE),
        re.compile(r'(\d+)\s*mm\s+(?:mưa|nước)', re.IGNORECASE),
    ]
    
    # Casualties patterns
    DEATH_PATTERNS = [
        re.compile(r'(\d+)\s*người\s*(?:chết|tử vong|thiệt mạng|qua đời)', re.IGNORECASE),
        re.compile(r'(?:số\s+)?(?:người\s+)?(?:chết|tử vong|thiệt mạng)[\s:]+(\d+)', re.IGNORECASE),
        re.compile(r'làm\s+(\d+)\s*người\s*(?:chết|tử vong)', re.IGNORECASE),
    ]
    
    INJURED_PATTERNS = [
        re.compile(r'(\d+)\s*người\s*(?:bị thương|thương tích|bị nạn)', re.IGNORECASE),
        re.compile(r'(?:số\s+)?(?:người\s+)?(?:bị\s+)?thương[\s:]+(\d+)', re.IGNORECASE),
    ]
    
    MISSING_PATTERNS = [
        re.compile(r'(\d+)\s*người\s*(?:mất tích|bị mất tích)', re.IGNORECASE),
        re.compile(r'(?:số\s+)?(?:người\s+)?mất\s+tích[\s:]+(\d+)', re.IGNORECASE),
    ]
    
    # Economic patterns
    DAMAGE_VND_PATTERNS = [
        re.compile(r'(?:thiệt hại|tổn thất|mất mát)\s*(?:ước tính\s+)?(?:khoảng\s+)?(\d+(?:[.,]\d+)?)\s*(tỷ|triệu|nghìn)?\s*(?:đồng|VND)', re.IGNORECASE),
        re.compile(r'(?:tổng\s+)?(?:thiệt hại|tổn thất)[\s:]+(\d+(?:[.,]\d+)?)\s*(tỷ|triệu)', re.IGNORECASE),
    ]
    
    # Storm patterns
    STORM_PATTERNS = [
        re.compile(r'(?:bão|cơn\s+bão)\s+([A-Z][a-z]{2,12})', re.IGNORECASE),
        re.compile(r'(?:mang\s+tên|tên\s+là)\s+([A-Z][a-z]{2,12})', re.IGNORECASE),
        re.compile(r'bão\s+số\s+\d+\s+\(([A-Z][a-z]+)\)', re.IGNORECASE),
    ]

COMPILED = CompiledPatterns()

# ==================== DATABASE MANAGER ====================
class DatabaseManager:
    """SQLite database for caching and storage"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, db_path: str = 'data/disaster_cache.db'):
        if not hasattr(self, 'initialized'):
            self.db_path = db_path
            self.initialized = True
            self._create_tables()
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()
    
    def _create_tables(self):
        """Create database tables if not exist"""
        os.makedirs(os.path.dirname(self.db_path) if os.path.dirname(self.db_path) else '.', exist_ok=True)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Processed articles cache
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS processed_articles (
                    article_hash TEXT PRIMARY KEY,
                    url TEXT,
                    location TEXT,
                    location_confidence REAL,
                    event_name TEXT,
                    event_confidence REAL,
                    wind_speed TEXT,
                    rainfall TEXT,
                    casualties TEXT,
                    damages TEXT,
                    extraction_quality REAL,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Location cache
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS location_cache (
                    content_hash TEXT PRIMARY KEY,
                    locations TEXT,
                    confidence REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_url ON processed_articles(url)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_quality ON processed_articles(extraction_quality)')
            
            logger.info(f"✅ Database initialized: {self.db_path}")
    
    def get_cached_article(self, content: str) -> Optional[Dict]:
        """Get cached processing result"""
        article_hash = hashlib.md5(content.encode()).hexdigest()
        
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    'SELECT * FROM processed_articles WHERE article_hash = ?',
                    (article_hash,)
                )
                row = cursor.fetchone()
                return dict(row) if row else None
        except sqlite3.Error as e:
            logger.warning(f"Cache lookup failed: {e}")
            return None
    
    def cache_article(self, content: str, url: str, result: Dict):
        """Cache processing result"""
        article_hash = hashlib.md5(content.encode()).hexdigest()
        
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO processed_articles
                    (article_hash, url, location, location_confidence, event_name, 
                     event_confidence, wind_speed, rainfall, casualties, damages, extraction_quality)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    article_hash, url, result.get('location'), result.get('location_confidence'),
                    result.get('event_name'), result.get('event_confidence'),
                    result.get('wind_speed'), result.get('rainfall'),
                    result.get('casualties'), result.get('damages'),
                    result.get('extraction_quality')
                ))
        except sqlite3.Error as e:
            logger.warning(f"Cache write failed: {e}")
    
    def get_statistics(self) -> Dict:
        """Get database statistics"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT COUNT(*) as total FROM processed_articles')
                total = cursor.fetchone()['total']
                
                cursor.execute('SELECT AVG(extraction_quality) as avg_quality FROM processed_articles')
                avg_quality = cursor.fetchone()['avg_quality'] or 0
                
                return {
                    'total_cached': total,
                    'avg_quality': avg_quality
                }
        except sqlite3.Error as e:
            logger.error(f"Stats query failed: {e}")
            return {'total_cached': 0, 'avg_quality': 0}

db_manager = DatabaseManager()

# ==================== DEDUPLICATION MANAGER ====================
class DeduplicationManager:
    """3-tier deduplication: Time Window → SimHash → Semantic Embedding"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.time_window_days = 2  # +/- 2 days
            self.simhash_threshold = 0.90  # 90% similarity
            self.semantic_threshold = 0.85  # 85% cosine similarity
            self.embedding_model = None
            self.initialized = True
            self._load_embedding_model()
    
    def _load_embedding_model(self):
        """Load lightweight multilingual embedding model"""
        try:
            # Use lightweight model for speed
            self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            logger.info("✅ Loaded embedding model for deduplication")
        except Exception as e:
            logger.warning(f"⚠️ Could not load embedding model: {e}")
            logger.warning("⚠️ Semantic deduplication will be disabled")
    
    # ==================== TIER 1: TIME WINDOW FILTER ====================
    def filter_by_time_window(self, articles: pd.DataFrame) -> pd.DataFrame:
        """Tier 1: Lọc thô - Chỉ so sánh articles trong cùng time window"""
        if 'date' not in articles.columns or 'scrape_time' not in articles.columns:
            logger.warning("⚠️ Missing date columns for time window filtering")
            return articles
        
        try:
            # Parse dates
            articles['_parsed_date'] = pd.to_datetime(articles['date'], errors='coerce')
            articles['_parsed_scrape'] = pd.to_datetime(articles['scrape_time'], errors='coerce')
            
            # Use scrape_time if date is invalid
            articles['_effective_date'] = articles['_parsed_date'].fillna(articles['_parsed_scrape'])
            
            # Sort by date for efficient comparison
            articles = articles.sort_values('_effective_date')
            
            logger.info(f"📅 Time window filtering: {self.time_window_days} days")
            return articles
            
        except Exception as e:
            logger.error(f"❌ Time window filtering failed: {e}")
            return articles
    
    # ==================== TIER 2: SIMHASH FILTER ====================
    def calculate_simhash(self, text: str) -> int:
        """Calculate SimHash for text"""
        try:
            if not text or len(text) < 10:
                return 0
            # SimHash với width=64 bits
            return Simhash(text, f=64).value
        except Exception as e:
            logger.debug(f"SimHash calculation error: {e}")
            return 0
    
    def simhash_similarity(self, hash1: int, hash2: int) -> float:
        """Calculate similarity between two SimHash values"""
        if hash1 == 0 or hash2 == 0:
            return 0.0
        
        # Hamming distance
        xor = hash1 ^ hash2
        hamming_dist = bin(xor).count('1')
        
        # Convert to similarity (0-1)
        similarity = 1.0 - (hamming_dist / 64.0)
        return similarity
    
    def filter_by_simhash(self, articles: pd.DataFrame) -> pd.DataFrame:
        """Tier 2: Lọc nhanh - Remove exact/near-exact duplicates using SimHash"""
        logger.info("🔍 Tier 2: SimHash filtering (threshold: {:.0%})...".format(self.simhash_threshold))
        
        if 'title' not in articles.columns or 'content' not in articles.columns:
            logger.warning("⚠️ Missing title/content for SimHash")
            return articles
        
        try:
            # Calculate SimHash for all articles
            articles['_simhash'] = articles.apply(
                lambda row: self.calculate_simhash(
                    f"{row.get('title', '')} {row.get('content', '')}"
                ),
                axis=1
            )
            
            # Track duplicates to remove
            duplicates_to_remove = set()
            total_comparisons = 0
            duplicates_found = 0
            
            # Compare articles within time window
            for idx, row in articles.iterrows():
                if idx in duplicates_to_remove:
                    continue
                
                current_date = row['_effective_date']
                current_hash = row['_simhash']
                
                if current_hash == 0:
                    continue
                
                # Find articles in time window
                time_window_start = current_date - pd.Timedelta(days=self.time_window_days)
                time_window_end = current_date + pd.Timedelta(days=self.time_window_days)
                
                candidates = articles[
                    (articles['_effective_date'] >= time_window_start) &
                    (articles['_effective_date'] <= time_window_end) &
                    (articles.index > idx) &  # Only compare with later articles
                    (~articles.index.isin(duplicates_to_remove))
                ]
                
                # Compare SimHash
                for cand_idx, cand_row in candidates.iterrows():
                    total_comparisons += 1
                    
                    similarity = self.simhash_similarity(current_hash, cand_row['_simhash'])
                    
                    if similarity >= self.simhash_threshold:
                        duplicates_to_remove.add(cand_idx)
                        duplicates_found += 1
                        logger.debug(f"  SimHash duplicate: {similarity:.2%} similarity")
            
            # Remove duplicates
            original_count = len(articles)
            articles = articles[~articles.index.isin(duplicates_to_remove)]
            removed_count = original_count - len(articles)
            
            logger.info(f"  ✅ SimHash: Removed {removed_count}/{original_count} duplicates")
            logger.info(f"  📊 Comparisons: {total_comparisons}, Efficiency: {len(articles)}/{original_count}")
            
            # Clean up temp columns
            articles = articles.drop('_simhash', axis=1)
            
            return articles
            
        except Exception as e:
            logger.error(f"❌ SimHash filtering failed: {type(e).__name__}: {e}")
            return articles
    
    # ==================== TIER 3: SEMANTIC EMBEDDING FILTER ====================
    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get semantic embedding for text"""
        if not self.embedding_model:
            return None
        
        try:
            if not text or len(text) < 10:
                return None
            
            # Truncate long texts for speed
            text = text[:1000]
            
            embedding = self.embedding_model.encode(text, convert_to_tensor=False)
            return embedding
            
        except Exception as e:
            logger.debug(f"Embedding error: {e}")
            return None
    
    def filter_by_semantic(self, articles: pd.DataFrame) -> pd.DataFrame:
        """Tier 3: Lọc tinh - Remove semantic duplicates using embeddings"""
        if not self.embedding_model:
            logger.info("⚠️ Tier 3: Semantic filtering skipped (model not available)")
            return articles
        
        logger.info("🧠 Tier 3: Semantic filtering (threshold: {:.0%})...".format(self.semantic_threshold))
        
        if 'title' not in articles.columns or 'content' not in articles.columns:
            return articles
        
        try:
            # Calculate embeddings (only for first 500 chars for speed)
            logger.info("  Computing embeddings...")
            articles['_embedding'] = articles.apply(
                lambda row: self.get_embedding(
                    f"{row.get('title', '')} {str(row.get('content', ''))[:500]}"
                ),
                axis=1
            )
            
            # Filter out None embeddings
            valid_articles = articles[articles['_embedding'].notna()].copy()
            
            if len(valid_articles) == 0:
                logger.warning("⚠️ No valid embeddings computed")
                return articles.drop('_embedding', axis=1, errors='ignore')
            
            # Track duplicates
            duplicates_to_remove = set()
            total_comparisons = 0
            duplicates_found = 0
            
            # Compare articles within time window
            for idx, row in valid_articles.iterrows():
                if idx in duplicates_to_remove:
                    continue
                
                current_date = row['_effective_date']
                current_emb = row['_embedding']
                
                # Find articles in time window
                time_window_start = current_date - pd.Timedelta(days=self.time_window_days)
                time_window_end = current_date + pd.Timedelta(days=self.time_window_days)
                
                candidates = valid_articles[
                    (valid_articles['_effective_date'] >= time_window_start) &
                    (valid_articles['_effective_date'] <= time_window_end) &
                    (valid_articles.index > idx) &
                    (~valid_articles.index.isin(duplicates_to_remove))
                ]
                
                # Compare embeddings
                for cand_idx, cand_row in candidates.iterrows():
                    total_comparisons += 1
                    
                    cand_emb = cand_row['_embedding']
                    
                    # Cosine similarity
                    similarity = util.cos_sim(current_emb, cand_emb).item()
                    
                    if similarity >= self.semantic_threshold:
                        duplicates_to_remove.add(cand_idx)
                        duplicates_found += 1
                        logger.debug(f"  Semantic duplicate: {similarity:.2%} similarity")
            
            # Remove duplicates
            original_count = len(articles)
            articles = articles[~articles.index.isin(duplicates_to_remove)]
            removed_count = original_count - len(articles)
            
            logger.info(f"  ✅ Semantic: Removed {removed_count}/{original_count} duplicates")
            logger.info(f"  📊 Comparisons: {total_comparisons}")
            
            # Clean up temp columns
            articles = articles.drop('_embedding', axis=1, errors='ignore')
            
            return articles
            
        except Exception as e:
            logger.error(f"❌ Semantic filtering failed: {type(e).__name__}: {e}")
            # Clean up on error
            return articles.drop('_embedding', axis=1, errors='ignore')
    
    # ==================== FULL PIPELINE ====================
    def deduplicate(self, articles: pd.DataFrame) -> pd.DataFrame:
        """Run full 3-tier deduplication pipeline"""
        logger.info("\n" + "="*80)
        logger.info("🚀 DEDUPLICATION PIPELINE (3-Tier Funnel)")
        logger.info("="*80)
        
        original_count = len(articles)
        logger.info(f"📊 Input: {original_count} articles")
        
        # Tier 1: Time Window
        articles = self.filter_by_time_window(articles)
        
        # Tier 2: SimHash (Fast)
        articles = self.filter_by_simhash(articles)
        
        # Tier 3: Semantic (Slow but accurate)
        articles = self.filter_by_semantic(articles)
        
        # Clean up temporary columns
        temp_cols = ['_parsed_date', '_parsed_scrape', '_effective_date']
        articles = articles.drop(columns=temp_cols, errors='ignore')
        
        final_count = len(articles)
        removed_count = original_count - final_count
        
        logger.info("\n" + "="*80)
        logger.info(f"✅ DEDUPLICATION COMPLETE")
        logger.info(f"   Original: {original_count} articles")
        logger.info(f"   Final: {final_count} articles")
        logger.info(f"   Removed: {removed_count} duplicates ({removed_count/original_count*100:.1f}%)")
        logger.info("="*80 + "\n")
        
        return articles

dedup_manager = DeduplicationManager()

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
        """Load models với progressive fallback - IMPROVED for Vietnamese"""
        # Priority 1: Vietnamese spaCy models
        for model_name in ["vi_core_news_lg", "vi_core_news_md", "en_core_web_sm"]:
            try:
                self.nlp_spacy = spacy.load(model_name)
                logger.info(f"✅ Loaded spaCy: {model_name}")
                break
            except OSError as e:
                logger.debug(f"Model {model_name} not found: {e}")
                continue
            except Exception as e:
                logger.warning(f"Failed to load {model_name}: {e}")
                continue
        
        # Priority 2: Vietnamese HuggingFace NER models
        vietnamese_ner_models = [
            "NlpHUST/ner-vietnamese-electra-base",
            "uitnlp/vibert4news-base-cased",
            "vinai/phobert-base"
        ]
        
        for model_name in vietnamese_ner_models:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForTokenClassification.from_pretrained(model_name)
                self.ner_pipeline = pipeline(
                    "ner",
                    model=model,
                    tokenizer=self.tokenizer,
                    aggregation_strategy="simple",
                    device=-1
                )
                logger.info(f"✅ Loaded Vietnamese NER: {model_name}")
                break
            except OSError as e:
                logger.debug(f"Model {model_name} not available: {e}")
                continue
            except Exception as e:
                logger.warning(f"Failed to load NER model {model_name}: {type(e).__name__}: {e}")
                continue
        
        if not self.ner_pipeline:
            logger.warning("⚠️ No Vietnamese NER model available")
        
        # Log available Vietnamese tools
        if UNDERTHESEA_AVAILABLE:
            logger.info("✅ Underthesea Vietnamese NLP available")
        if PYVI_AVAILABLE:
            logger.info("✅ PyVi tokenizer available")

models = AdvancedModelManager()

# ==================== VIETNAMESE LOCATION DICTIONARY ====================
class VietnameseLocationDict:
    """Comprehensive Vietnamese location dictionary"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.provinces = {}
            self.cities = []
            self.districts = {}
            self.geo_features = {}
            self.variations = {}
            self.region_keywords = {}
            self.initialized = True
            self._load_locations()
    
    def _load_locations(self):
        """Load Vietnamese locations from JSON"""
        try:
            with open('config/vietnam_locations.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.provinces = data.get('provinces', {})
                self.cities = data.get('major_cities', [])
                self.districts = data.get('districts', {})
                self.geo_features = data.get('geographic_features', {})
                self.variations = data.get('location_variations', {})
                self.region_keywords = data.get('region_keywords', {})
                logger.info(f"✅ Loaded {len(self.provinces)} provinces, {len(self.cities)} cities")
        except FileNotFoundError:
            logger.warning(f"⚠️ Vietnamese locations file not found, using fallback")
            self._create_minimal_dict()
        except json.JSONDecodeError as e:
            logger.error(f"❌ Invalid JSON in locations file: {e}")
            self._create_minimal_dict()
        except Exception as e:
            logger.error(f"❌ Unexpected error loading locations: {type(e).__name__}: {e}")
            self._create_minimal_dict()
    
    def _create_minimal_dict(self):
        """Create minimal location dictionary as fallback"""
        self.provinces = {
            "Hà Nội": {"region": "Đồng bằng sông Hồng"},
            "Hồ Chí Minh": {"region": "Đông Nam Bộ"},
            "Đà Nẵng": {"region": "Nam Trung Bộ"},
            "Hải Phòng": {"region": "Đồng bằng sông Hồng"},
            "Cần Thơ": {"region": "Đồng bằng sông Cửu Long"}
        }
        self.cities = ["Hà Nội", "Hồ Chí Minh", "Đà Nẵng"]
        self.variations = {"TPHCM": "Hồ Chí Minh", "Sài Gòn": "Hồ Chí Minh"}
    
    @lru_cache(maxsize=1000)
    def normalize_location(self, text: str) -> str:
        """Normalize location variations (cached)"""
        text = text.strip()
        return self.variations.get(text, text)
    
    def is_province(self, text: str) -> bool:
        """Check if text is a valid province"""
        normalized = self.normalize_location(text)
        return normalized in self.provinces
    
    def is_city(self, text: str) -> bool:
        """Check if text is a major city"""
        normalized = self.normalize_location(text)
        return normalized in self.cities
    
    def get_region(self, province: str) -> Optional[str]:
        """Get region for a province"""
        province_data = self.provinces.get(province, {})
        return province_data.get('region')
    
    def extract_locations_fuzzy(self, text: str, threshold: float = 0.85) -> List[Tuple[str, float]]:
        """Extract locations with fuzzy matching"""
        results = []
        text_lower = text.lower()
        
        # Exact matches
        for province in self.provinces.keys():
            if province.lower() in text_lower:
                results.append((province, 1.0))
        
        # Variations
        for var, canonical in self.variations.items():
            if var.lower() in text_lower:
                results.append((canonical, 0.95))
        
        # Geographic features
        for category, features in self.geo_features.items():
            for feature in features:
                if feature.lower() in text_lower:
                    results.append((feature, 0.90))
        
        return results

vn_locations = VietnameseLocationDict()

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
    def normalize_location(self, location: str) -> str:
        """Normalize location name - CRITICAL for aggregation"""
        if not location:
            return location
        
        location = location.strip()
        
        # Check if it's an alias for a province
        provinces = self.get_provinces()
        for official, aliases in provinces.items():
            if location.lower() == official.lower():
                return official
            for alias in aliases:
                if location.lower() == alias.lower():
                    return official
        
        # Check Vietnamese location dictionary
        normalized = vn_locations.normalize_location(location)
        if normalized != location:
            return normalized
        
        # Return as-is if no match
        return location
    
    @property
    def PROVINCES(self) -> Dict[str, List[str]]:
        """Property for backward compatibility"""
        return self.get_provinces()
    
    @property
    def REGIONS(self) -> Dict[str, List[str]]:
        """Property for backward compatibility"""
        return self.get_regions()
    
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
        """EXPANDED regex patterns for quantitative data extraction"""
        patterns = {
            'wind': [
                (r'(?:gió|sức gió|tốc độ gió)\s+(?:mạnh\s+)?(?:cấp\s+)?(\d+)(?:-(\d+))?\s*(cấp|km/h|m/s)?', 'wind_speed'),
                (r'giật\s+(?:cấp\s+)?(\d+)', 'gust'),
                (r'(?:gió\s+)?(?:mạnh\s+)?cấp\s+(\d+)(?:\s*-\s*(\d+))?', 'wind_level'),
                (r'(?:vận tốc|tốc độ)\s+(\d+)\s*(?:km/h|m/s)', 'wind_velocity'),
            ],
            'rain': [
                (r'(?:mưa|lượng mưa)\s+(?:đạt\s+)?(\d+)(?:-(\d+))?\s*mm', 'rainfall'),
                (r'tổng\s+lượng\s+(?:mưa|nước)\s+(\d+)(?:-(\d+))?\s*mm', 'total_rainfall'),
                (r'(?:lượng\s+)?mưa\s+(?:lớn|to|rất to)\s+(\d+)\s*mm', 'heavy_rain'),
                (r'(\d+)\s*mm\s+(?:mưa|nước)', 'rain_amount'),
                (r'(?:mưa|nước)\s+(?:cao|lên đến)\s+(\d+)\s*mm', 'max_rainfall'),
            ],
            'casualties': [
                # Deaths - EXPANDED patterns
                (r'(\d+)\s*người\s*(?:chết|tử vong|thiệt mạng|qua đời)', 'deaths'),
                (r'(?:số\s+)?(?:người\s+)?(?:chết|tử vong|thiệt mạng)[\s:]+(\d+)', 'deaths'),
                (r'(?:có\s+)?(\d+)\s*người\s*(?:đã\s+)?(?:chết|tử vong)', 'deaths'),
                (r'làm\s+(\d+)\s*người\s*(?:chết|tử vong)', 'deaths'),
                
                # Injured - EXPANDED patterns
                (r'(\d+)\s*người\s*(?:bị thương|thương tích|bị nạn)', 'injured'),
                (r'(?:số\s+)?(?:người\s+)?(?:bị\s+)?thương[\s:]+(\d+)', 'injured'),
                (r'làm\s+(\d+)\s*người\s*bị\s*thương', 'injured'),
                
                # Missing - EXPANDED patterns
                (r'(\d+)\s*người\s*(?:mất tích|bị mất tích|mất tích)', 'missing'),
                (r'(?:số\s+)?(?:người\s+)?mất\s+tích[\s:]+(\d+)', 'missing'),
                
                # Affected - NEW patterns
                (r'(\d+)\s*người\s*(?:bị ảnh hưởng|ảnh hưởng|chịu ảnh hưởng)', 'affected'),
                (r'(?:khoảng|gần|hơn)\s+(\d+)\s*người\s*dân', 'people_affected'),
            ],
            'economic': [
                # Damage VND - EXPANDED patterns
                (r'(?:thiệt hại|tổn thất|mất mát|thiệt hại)\s*(?:ước tính\s+)?(?:khoảng\s+)?(\d+(?:[.,]\d+)?)\s*(tỷ|triệu|nghìn)?\s*(?:đồng|VND)', 'damage_vnd'),
                (r'(?:tổng\s+)?(?:thiệt hại|tổn thất)[\s:]+(\d+(?:[.,]\d+)?)\s*(tỷ|triệu)', 'total_damage_vnd'),
                (r'(?:ước tính\s+)?(\d+(?:[.,]\d+)?)\s*(tỷ|triệu)\s*(?:đồng|VND)\s*(?:thiệt hại|tổn thất)', 'damage_vnd'),
                (r'gây\s+(?:thiệt hại|tổn thất)\s+(\d+(?:[.,]\d+)?)\s*(tỷ|triệu)', 'caused_damage_vnd'),
                
                # Damage USD - EXPANDED patterns
                (r'(\d+(?:[.,]\d+)?)\s*(?:tỷ|triệu|nghìn)?\s*USD', 'damage_usd'),
                (r'(?:thiệt hại|tổn thất)\s+(\d+(?:[.,]\d+)?)\s*(?:triệu|tỷ)\s*USD', 'damage_usd'),
            ],
            'area': [
                # Area - EXPANDED patterns
                (r'(\d+(?:[.,]\d+)?)\s*(?:ha|hecta|héc-ta)\s*(?:ruộng|đất|rừng|diện tích)?', 'affected_area'),
                (r'(?:diện tích|khoảng)\s+(\d+(?:[.,]\d+)?)\s*(?:ha|hecta)', 'area_ha'),
                (r'(\d+(?:[.,]\d+)?)\s*(?:km2|km²|kilômét vuông)', 'area_km2'),
                (r'(?:hơn|khoảng|gần)\s+(\d+(?:[.,]\d+)?)\s*(?:ha|hecta)\s*(?:lúa|rừng|đất)', 'crop_area'),
            ],
            'structures': [
                # Houses - EXPANDED patterns
                (r'(\d+(?:[.,]\d+)?)\s*(?:căn|ngôi|hộ)\s*nhà\s*(?:bị\s+)?(?:sập|hư|tốc mái|hư hỏng|đổ)', 'houses_damaged'),
                (r'(?:có\s+)?(\d+)\s*(?:căn|ngôi)\s*nhà\s*(?:bị\s+)?(?:ảnh hưởng|thiệt hại)', 'houses_affected'),
                (r'làm\s+(\d+)\s*(?:căn|ngôi)\s*nhà\s*(?:sập|hư)', 'houses_collapsed'),
                
                # Infrastructure - EXPANDED patterns
                (r'(\d+)\s*(?:cây\s+)?cầu\s*(?:bị\s+)?(?:sập|hư|cuốn trôi|hư hỏng)', 'bridges_damaged'),
                (r'(\d+)\s*(?:km|kilômét)\s*đường\s*(?:bị\s+)?(?:ngập|hư|sạt lở)', 'roads_damaged'),
                (r'(\d+)\s*trường\s*(?:học|mầm non|cấp \d+)\s*(?:bị\s+)?(?:hư|ngập|tốc mái)', 'schools_damaged'),
                (r'(\d+)\s*(?:trạm|cơ sở)\s*y tế\s*(?:bị\s+)?(?:hư|ngập)', 'health_facilities_damaged'),
            ],
            'agriculture': [
                # NEW category for agriculture
                (r'(\d+(?:[.,]\d+)?)\s*(?:ha|hecta)\s*(?:lúa|hoa màu|cây trồng)\s*(?:bị\s+)?(?:thiệt hại|ngập|hư)', 'crops_damaged'),
                (r'(\d+(?:[.,]\d+)?)\s*(?:con|tấn)\s*(?:gia súc|gia cầm|thủy sản)', 'livestock_affected'),
                (r'(\d+)\s*(?:cây|tấn)\s*(?:gỗ|tre)', 'trees_damaged'),
            ],
            'flooding': [
                # NEW category for flooding
                (r'(?:nước\s+)?(?:ngập|lụt)\s+(?:sâu|cao)\s+(\d+(?:[.,]\d+)?)\s*(?:m|mét)', 'flood_depth'),
                (r'(\d+)\s*(?:xã|phường|thôn|thị trấn)\s*(?:bị\s+)?(?:ngập|lụt)', 'flooded_areas'),
                (r'(\d+)\s*(?:hộ|gia đình)\s*(?:bị\s+)?(?:ngập|lụt|cô lập)', 'households_flooded'),
            ]
        }
        
        results = defaultdict(list)
        
        for category, pattern_list in patterns.items():
            for pattern, key in pattern_list:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    # Handle flexible number extraction
                    value_str = match.group(1).replace(',', '.')
                    try:
                        value = float(value_str)
                    except:
                        continue
                    
                    # Handle unit if present
                    unit = None
                    if len(match.groups()) >= 2 and match.group(2):
                        unit = match.group(2)
                    
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
        """Multi-strategy location extraction WITH VIETNAMESE DICTIONARY"""
        strategies = {
            'dict_fuzzy': self._extract_loc_dict_fuzzy,  # NEW: Priority 1
            'rule_based': self._extract_loc_rules,  # Priority 2
            'context': self._extract_loc_context,  # Priority 3
            # 'underthesea': self._extract_loc_underthesea,  # Disabled for speed
            # 'ner_hf': self._extract_loc_hf,  # Disabled for speed in parallel
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
        
        if all_results:
            logger.debug(f"[MULTI] Total {len(all_results)} location candidates before aggregation")
        
        # Aggregate and normalize
        result = self._aggregate_locations(all_results)
        if result['locations']:
            logger.info(f"[MULTI] ✅ Final locations after aggregation: {result['locations']}")
        else:
            logger.debug(f"[MULTI] ❌ No locations after aggregation (had {len(all_results)} candidates)")
        return result
    
    def _extract_loc_dict_fuzzy(self, content: str) -> List[str]:
        """Vietnamese dictionary-based fuzzy extraction - HIGHEST PRIORITY"""
        locations = vn_locations.extract_locations_fuzzy(content)
        result = [loc for loc, conf in locations if conf >= 0.85]
        if result:
            logger.debug(f"[DICT_FUZZY] Found {len(result)} locations: {result[:3]}")
        return result
    
    def _extract_loc_underthesea(self, content: str) -> List[str]:
        """Underthesea Vietnamese NER extraction"""
        if not UNDERTHESEA_AVAILABLE:
            return []
        
        try:
            ner_results = underthesea_ner(content[:2000])
            locations = []
            for token, label in ner_results:
                if label.startswith('B-LOC') or label.startswith('I-LOC'):
                    locations.append(token)
            return locations
        except:
            return []
    
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
                if ent['entity_group'] in ['LOC', 'LOCATION'] and ent['score'] > 0.7]
    
    def _extract_loc_rules(self, content: str) -> List[str]:
        """Rule-based extraction WITH VIETNAMESE DICTIONARY"""
        found = []
        content_lower = content.lower()
        
        # Check Vietnamese location dictionary
        for province in vn_locations.provinces.keys():
            if province.lower() in content_lower:
                found.append(province)
        
        # Check variations
        for alias, canonical in vn_locations.variations.items():
            if alias.lower() in content_lower:
                found.append(canonical)
        
        # Check old knowledge base for compatibility
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
        
        result = list(set(found))
        if result:
            logger.debug(f"[RULES] Found {len(result)} locations: {result[:3]}")
        return result
    
    def _extract_loc_context(self, content: str) -> List[str]:
        """Context-based extraction WITH VIETNAMESE DICTIONARY"""
        location_contexts = self.context_extractor.extract_with_context(
            content, 'tại|ở|vùng|khu vực|địa phương|tỉnh|thành phố|huyện|xã'
        )
        
        locations = []
        for context in location_contexts:
            # Check Vietnamese dictionary
            for province in vn_locations.provinces.keys():
                if province.lower() in context.lower():
                    locations.append(province)
            
            # Check variations
            for alias, canonical in vn_locations.variations.items():
                if alias.lower() in context.lower():
                    locations.append(canonical)
        
        return locations
    
    def _calculate_confidence(self, location: str, strategy: str) -> float:
        """Tính confidence score WITH VIETNAMESE DICT BOOST"""
        base_confidence = {
            'dict_fuzzy': 0.95,  # HIGHEST for Vietnamese dictionary
            'underthesea': 0.90,  # Vietnamese-specific NER
            'rule_based': 0.85,
            'ner_hf': 0.80,
            'context': 0.75,
        }
        
        confidence = base_confidence.get(strategy, 0.5)
        
        # Boost if in Vietnamese location dictionary
        if vn_locations.is_province(location) or vn_locations.is_city(location):
            confidence += 0.05
        
        # Boost if in old knowledge base
        if kb.normalize_location(location) in kb.PROVINCES:
            confidence += 0.03
        
        return min(confidence, 1.0)
    
    def _aggregate_locations(self, results: List[Dict]) -> Dict:
        """Aggregate results from multiple strategies WITH VIETNAMESE NORMALIZATION"""
        if not results:
            return {'locations': [], 'confidence': 0.0}
        
        # Group by normalized location
        location_groups = defaultdict(list)
        for result in results:
            # Try Vietnamese dict normalization first
            normalized = vn_locations.normalize_location(result['location'])
            if normalized not in vn_locations.provinces and normalized not in vn_locations.cities:
                # Fallback to old KB normalization
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
            
            # Lower threshold for Vietnamese dictionary matches
            threshold = 0.5 if (vn_locations.is_province(loc) or vn_locations.is_city(loc)) else 0.6
            
            if avg_confidence > threshold:
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
        """Process single article với full validation and caching"""
        content = str(row.get('content', ''))
        url = str(row.get('url', ''))
        
        # Check cache first
        cached = db_manager.get_cached_article(content)
        if cached:
            logger.debug(f"✅ Using cached result for: {url[:50]}")
            return cached
        
        try:
            disaster_type = str(row.get('disaster_type', ''))
            title = str(row.get('title', '')[:50])  # For debugging
            
            # Multi-strategy extraction
            location_result = self.multi_extractor.extract_location_multi(content)
            
            # DEBUG: Log location extraction result
            if location_result['locations']:
                logger.info(f"✅ [ARTICLE: {title}...] Locations: {location_result['locations']}")
            
            event_result = self.multi_extractor.extract_event_name_advanced(content, disaster_type)
            numbers = self.multi_extractor.extract_numbers_validated(content)
            time_result = self.multi_extractor.extract_time_multi(content)
            
            result = {
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
            
            # Cache result
            db_manager.cache_article(content, url, result)
            
            return result
            
        except KeyError as e:
            logger.error(f"❌ Missing required field in article: {e}")
            return {k: None for k in ['location', 'event_name', 'wind_speed', 
                                      'rainfall', 'casualties', 'damages', 
                                      'location_confidence', 'event_confidence',
                                      'start_time', 'start_time_confidence',
                                      'peak_time', 'peak_time_confidence',
                                      'end_time', 'end_time_confidence',
                                      'extraction_quality']}
        except ValueError as e:
            logger.error(f"❌ Invalid data format in article: {e}")
            return {k: None for k in ['location', 'event_name', 'wind_speed', 
                                      'rainfall', 'casualties', 'damages', 
                                      'location_confidence', 'event_confidence',
                                      'start_time', 'start_time_confidence',
                                      'peak_time', 'peak_time_confidence',
                                      'end_time', 'end_time_confidence',
                                      'extraction_quality']}
        except Exception as e:
            logger.exception(f"❌ Unexpected error processing article {url[:50]}: {type(e).__name__}: {e}")
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

# ==================== TRULY PARALLEL PROCESSING ====================
def process_single_row(row_tuple):
    """Process single row - for multiprocessing"""
    idx, row = row_tuple
    processor = EnhancedProcessor()
    result = processor.process_article(row)
    result['_index'] = idx
    return result

def parallel_process_enhanced(df: pd.DataFrame, n_workers: int = None) -> pd.DataFrame:
    """TRULY parallel processing with ProcessPoolExecutor"""
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)
    
    logger.info(f"🚀 Processing {len(df)} articles with {n_workers} workers (TRUE parallel)...")
    
    results = []
    
    try:
        # Use ProcessPoolExecutor for true parallelism
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Submit all tasks
            futures = {executor.submit(process_single_row, (idx, row)): idx 
                      for idx, row in df.iterrows()}
            
            # Collect results with progress bar
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
                try:
                    result = future.result(timeout=60)  # 60s timeout per article
                    results.append(result)
                except TimeoutError:
                    idx = futures[future]
                    logger.warning(f"⏱️ Timeout processing article at index {idx}")
                    results.append({'_index': idx, **{k: None for k in ['location', 'event_name']}})
                except Exception as e:
                    idx = futures[future]
                    logger.error(f"❌ Error processing article at index {idx}: {type(e).__name__}: {e}")
                    results.append({'_index': idx, **{k: None for k in ['location', 'event_name']}})
    
    except Exception as e:
        logger.error(f"❌ Parallel processing failed: {type(e).__name__}: {e}")
        logger.info("⚠️ Falling back to sequential processing...")
        # Fallback to sequential
        processor = EnhancedProcessor()
        results = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Sequential processing"):
            try:
                result = processor.process_article(row)
                result['_index'] = idx
                results.append(result)
            except Exception as e:
                logger.error(f"Error at {idx}: {e}")
                results.append({'_index': idx, **{k: None for k in ['location', 'event_name']}})
    
    # Sort by original index
    results.sort(key=lambda x: x.get('_index', 0))
    
    # Merge results
    result_df = pd.DataFrame(results)
    if '_index' in result_df.columns:
        result_df = result_df.drop('_index', axis=1)
    
    for col in result_df.columns:
        if col in result_df.columns:
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
    logger.info("ADVANCED MULTI-STRATEGY DISASTER DATA EXTRACTION v4.0 (OPTIMIZED)")
    logger.info("="*80)
    
    # Database stats
    db_stats = db_manager.get_statistics()
    logger.info(f"📊 Database: {db_stats['total_cached']} cached articles, avg quality: {db_stats['avg_quality']:.2f}")
    
    # Find latest JSON
    json_files = glob.glob('data/disaster_data_multisource_*.json')
    if not json_files:
        logger.error("❌ No JSON file found!")
        return
    
    json_file = max(json_files)
    logger.info(f"📂 Processing: {json_file}")
    
    # Load data with error handling
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.error(f"❌ File not found: {json_file}")
        return
    except json.JSONDecodeError as e:
        logger.error(f"❌ Invalid JSON file: {e}")
        return
    except Exception as e:
        logger.error(f"❌ Error loading file: {type(e).__name__}: {e}")
        return
    
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
    
    # ==================== DEDUPLICATION PIPELINE ====================
    logger.info("\n🔄 Running deduplication pipeline...")
    df = dedup_manager.deduplicate(df)
    logger.info(f"✅ After deduplication: {len(df)} unique articles")
    
    # Process with TRUE parallel extraction
    n_workers = max(1, mp.cpu_count() - 1)
    logger.info(f"💻 Using {n_workers} CPU cores for parallel processing")
    df = parallel_process_enhanced(df, n_workers=n_workers)
    
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
    
    # Export with error handling
    output_file = 'data/disaster_data_enhanced_v4.csv'
    try:
        df_export.to_csv(output_file, index=False, encoding='utf-8-sig')
        logger.info(f"✅ Exported to: {output_file}")
    except IOError as e:
        logger.error(f"❌ Failed to export CSV: {e}")
    except Exception as e:
        logger.error(f"❌ Unexpected export error: {type(e).__name__}: {e}")
    
    # Generate statistics
    generate_advanced_statistics(df_export)
    
    # Export high-quality subset
    if 'extraction_quality' in df_export.columns:
        high_quality = df_export[df_export['extraction_quality'] >= 0.7]
        hq_file = 'data/disaster_data_high_quality_v4.csv'
        try:
            high_quality.to_csv(hq_file, index=False, encoding='utf-8-sig')
            logger.info(f"✅ High-quality subset ({len(high_quality)} articles): {hq_file}")
        except Exception as e:
            logger.error(f"❌ Failed to export high-quality subset: {e}")
    
    # Final database stats
    final_stats = db_manager.get_statistics()
    logger.info(f"\n📊 Final database stats: {final_stats['total_cached']} cached, avg quality: {final_stats['avg_quality']:.2f}")
    logger.info("\n✅ Processing completed successfully!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"❌ Process failed: {e}", exc_info=True)
        raise