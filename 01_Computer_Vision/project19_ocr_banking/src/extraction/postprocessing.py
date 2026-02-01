"""
Post-processing utilities for OCR results
"""
import re
from typing import List, Dict, Optional, Tuple
from difflib import SequenceMatcher
import logging

try:
    from fuzzywuzzy import fuzz
    FUZZYWUZZY_AVAILABLE = True
except ImportError:
    FUZZYWUZZY_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextPostProcessor:
    """Post-process OCR text for better accuracy"""
    
    def __init__(self, dictionary: Optional[List[str]] = None):
        """
        Initialize text post-processor
        
        Args:
            dictionary: Optional dictionary for spell correction
        """
        self.dictionary = dictionary or self._default_dictionary()
    
    @staticmethod
    def _default_dictionary() -> List[str]:
        """Default dictionary for Vietnamese banking terms"""
        return [
            # Common words
            'căn cước công dân', 'chứng minh nhân dân', 'quốc tịch',
            'họ và tên', 'ngày sinh', 'giới tính', 'nam', 'nữ',
            'số', 'ngày', 'tháng', 'năm', 'địa chỉ', 'thường trú',
            
            # Banking terms
            'tài khoản', 'số tài khoản', 'chủ tài khoản',
            'số dư', 'giao dịch', 'chuyển khoản', 'rút tiền', 'nạp tiền',
            'lãi suất', 'kỳ hạn', 'ngân hàng', 'chi nhánh',
            'sao kê', 'biên lai', 'hóa đơn', 'hợp đồng',
            
            # Document types
            'bản sao', 'bản chính', 'công chứng', 'xác nhận'
        ]
    
    def clean_text(self, text: str) -> str:
        """
        Clean OCR text by removing noise
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters (keep Vietnamese)
        text = re.sub(r'[^\w\s\-/.,àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđĐ]', '', text)
        
        # Fix common OCR errors
        text = self._fix_common_errors(text)
        
        return text.strip()
    
    @staticmethod
    def _fix_common_errors(text: str) -> str:
        """Fix common OCR recognition errors"""
        corrections = {
            '0': 'O',  # Zero to O (context-dependent)
            'l': '1',  # l to 1 in numbers
            'I': '1',  # I to 1 in numbers
            'S': '5',  # S to 5 in numbers (context-dependent)
            'B': '8',  # B to 8 in numbers (context-dependent)
        }
        
        # Apply corrections (simplified - should be context-aware)
        for old, new in corrections.items():
            # Only replace in numeric contexts
            text = re.sub(rf'(?<=\d){old}(?=\d)', new, text)
        
        return text
    
    def correct_spelling(self, text: str, threshold: float = 0.8) -> str:
        """
        Correct spelling using fuzzy matching
        
        Args:
            text: Input text
            threshold: Similarity threshold
            
        Returns:
            Corrected text
        """
        if not FUZZYWUZZY_AVAILABLE:
            logger.warning("fuzzywuzzy not available, skipping spelling correction")
            return text
        
        words = text.split()
        corrected_words = []
        
        for word in words:
            # Skip if too short or looks like a number
            if len(word) < 3 or word.isdigit():
                corrected_words.append(word)
                continue
            
            # Find best match in dictionary
            best_match = self._find_best_match(word.lower(), threshold)
            
            if best_match:
                # Preserve original case
                if word.isupper():
                    corrected_words.append(best_match.upper())
                elif word[0].isupper():
                    corrected_words.append(best_match.capitalize())
                else:
                    corrected_words.append(best_match)
            else:
                corrected_words.append(word)
        
        return ' '.join(corrected_words)
    
    def _find_best_match(self, word: str, threshold: float) -> Optional[str]:
        """Find best matching word from dictionary"""
        if not FUZZYWUZZY_AVAILABLE:
            return None
        
        best_score = 0
        best_match = None
        
        for dict_word in self.dictionary:
            score = fuzz.ratio(word, dict_word.lower()) / 100
            
            if score > best_score and score >= threshold:
                best_score = score
                best_match = dict_word
        
        return best_match
    
    def format_id_number(self, id_number: str) -> Optional[str]:
        """
        Format and validate ID number
        
        Args:
            id_number: Raw ID number
            
        Returns:
            Formatted ID number or None if invalid
        """
        # Remove non-digits
        digits = re.sub(r'\D', '', id_number)
        
        # Check length (9 for CMND, 12 for CCCD)
        if len(digits) == 9 or len(digits) == 12:
            return digits
        
        return None
    
    def format_date(self, date_str: str, input_format: str = 'dmy') -> Optional[str]:
        """
        Format and validate date
        
        Args:
            date_str: Raw date string
            input_format: Input format ('dmy', 'ymd')
            
        Returns:
            Formatted date (DD/MM/YYYY) or None if invalid
        """
        # Extract digits and separators
        parts = re.findall(r'\d+', date_str)
        
        if len(parts) != 3:
            return None
        
        try:
            if input_format == 'dmy':
                day, month, year = parts
            elif input_format == 'ymd':
                year, month, day = parts
            else:
                return None
            
            # Validate
            day = int(day)
            month = int(month)
            year = int(year)
            
            if not (1 <= day <= 31 and 1 <= month <= 12 and 1900 <= year <= 2100):
                return None
            
            return f"{day:02d}/{month:02d}/{year:04d}"
            
        except ValueError:
            return None
    
    def format_money(self, amount_str: str) -> Optional[str]:
        """
        Format money amount
        
        Args:
            amount_str: Raw amount string
            
        Returns:
            Formatted amount or None if invalid
        """
        # Remove currency symbols and whitespace
        amount_str = re.sub(r'[^\d.,]', '', amount_str)
        
        # Replace comma with dot (if used as decimal separator)
        # This is simplified - should be more sophisticated
        amount_str = amount_str.replace(',', '.')
        
        try:
            amount = float(amount_str)
            return f"{amount:,.0f}"
        except ValueError:
            return None


class ConfidenceCalculator:
    """Calculate confidence scores for extracted fields"""
    
    @staticmethod
    def calculate_field_confidence(value: str, field_type: str) -> float:
        """
        Calculate confidence score for extracted field
        
        Args:
            value: Extracted value
            field_type: Type of field
            
        Returns:
            Confidence score (0-1)
        """
        if not value:
            return 0.0
        
        if field_type in ['cccd', 'cmnd', 'id_number']:
            return ConfidenceCalculator._id_confidence(value)
        elif field_type in ['date_dmy', 'date_ymd']:
            return ConfidenceCalculator._date_confidence(value)
        elif field_type in ['money_vnd', 'money_number']:
            return ConfidenceCalculator._money_confidence(value)
        elif field_type == 'vietnamese_name':
            return ConfidenceCalculator._name_confidence(value)
        
        # Default confidence based on length
        return min(len(value) / 20, 1.0)
    
    @staticmethod
    def _id_confidence(id_number: str) -> float:
        """Calculate confidence for ID number"""
        digits = re.sub(r'\D', '', id_number)
        
        if len(digits) == 12:  # CCCD
            return 0.95
        elif len(digits) == 9:  # CMND
            return 0.90
        
        return 0.5
    
    @staticmethod
    def _date_confidence(date_str: str) -> float:
        """Calculate confidence for date"""
        parts = re.findall(r'\d+', date_str)
        
        if len(parts) != 3:
            return 0.3
        
        try:
            day, month, year = map(int, parts)
            
            if 1 <= day <= 31 and 1 <= month <= 12 and 1900 <= year <= 2100:
                return 0.85
        except ValueError:
            pass
        
        return 0.5
    
    @staticmethod
    def _money_confidence(amount_str: str) -> float:
        """Calculate confidence for money amount"""
        digits = re.sub(r'\D', '', amount_str)
        
        if len(digits) >= 3:
            return 0.80
        
        return 0.5
    
    @staticmethod
    def _name_confidence(name: str) -> float:
        """Calculate confidence for Vietnamese name"""
        # Check for at least 2 words
        words = name.split()
        
        if len(words) >= 2:
            # Check if words start with uppercase
            if all(w[0].isupper() for w in words):
                return 0.85
            return 0.70
        
        return 0.5


if __name__ == "__main__":
    # Test post-processing
    processor = TextPostProcessor()
    
    # Test text cleaning
    dirty_text = "Họ  và   tên:  NGUYỄN   VĂN  A@#"
    clean_text = processor.clean_text(dirty_text)
    print(f"Cleaned: '{clean_text}'")
    
    # Test ID formatting
    raw_id = "001-234-567-890"
    formatted_id = processor.format_id_number(raw_id)
    print(f"ID: {formatted_id}")
    
    # Test date formatting
    raw_date = "15-3-1990"
    formatted_date = processor.format_date(raw_date)
    print(f"Date: {formatted_date}")
    
    # Test confidence
    calculator = ConfidenceCalculator()
    confidence = calculator.calculate_field_confidence("001234567890", "cccd")
    print(f"CCCD Confidence: {confidence:.2f}")
