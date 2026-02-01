"""
Information Extraction from OCR Results
Extract structured data from recognized text
"""
import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FieldExtractor:
    """Extract structured fields from OCR text"""
    
    def __init__(self, patterns_config: Optional[Dict] = None):
        """
        Initialize field extractor
        
        Args:
            patterns_config: Custom regex patterns for extraction
        """
        self.patterns = patterns_config or self._default_patterns()
    
    @staticmethod
    def _default_patterns() -> Dict[str, str]:
        """Default extraction patterns for Vietnamese banking documents"""
        return {
            # ID numbers
            'cccd': r'\b\d{12}\b',  # 12-digit CCCD
            'cmnd': r'\b\d{9}\b',   # 9-digit CMND
            'id_number': r'\b\d{9,12}\b',  # General ID
            
            # Dates
            'date_dmy': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b',  # DD/MM/YYYY or DD-MM-YYYY
            'date_ymd': r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',  # YYYY/MM/DD or YYYY-MM-DD
            
            # Money amounts
            'money_vnd': r'\b\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?\s*(?:VND|VNĐ|đ|₫)?\b',
            'money_number': r'\b\d{1,3}(?:[.,]\d{3})*\b',
            
            # Account numbers
            'account_number': r'\b\d{10,16}\b',
            
            # Phone numbers
            'phone_vn': r'\b(?:0|\+84)[1-9]\d{8,9}\b',
            
            # Vietnamese names (basic pattern)
            'vietnamese_name': r'\b[A-ZÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ][a-zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]+(?:\s+[A-ZÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ][a-zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]+)+\b',
            
            # Gender
            'gender': r'\b(?:Nam|Nữ|Male|Female|M|F)\b',
            
            # Address (basic pattern)
            'address': r'(?:Số|số)\s+\d+[^,\n]*(?:,\s*[^,\n]*){0,3}',
        }
    
    def extract_field(self, text: str, field_type: str) -> Optional[str]:
        """
        Extract single field from text
        
        Args:
            text: Input text
            field_type: Type of field to extract
            
        Returns:
            Extracted value or None
        """
        if field_type not in self.patterns:
            logger.warning(f"Unknown field type: {field_type}")
            return None
        
        pattern = self.patterns[field_type]
        match = re.search(pattern, text, re.IGNORECASE)
        
        if match:
            return match.group(0).strip()
        
        return None
    
    def extract_all_fields(self, text: str, field_types: List[str]) -> Dict[str, Optional[str]]:
        """
        Extract multiple fields from text
        
        Args:
            text: Input text
            field_types: List of field types to extract
            
        Returns:
            Dictionary of extracted fields
        """
        results = {}
        
        for field_type in field_types:
            results[field_type] = self.extract_field(text, field_type)
        
        return results
    
    def extract_multiple_values(self, text: str, field_type: str) -> List[str]:
        """
        Extract all occurrences of a field type
        
        Args:
            text: Input text
            field_type: Type of field to extract
            
        Returns:
            List of extracted values
        """
        if field_type not in self.patterns:
            logger.warning(f"Unknown field type: {field_type}")
            return []
        
        pattern = self.patterns[field_type]
        matches = re.findall(pattern, text, re.IGNORECASE)
        
        return [m.strip() for m in matches]


class DocumentTypeClassifier:
    """Classify document type based on content"""
    
    def __init__(self, keywords_config: Optional[Dict] = None):
        """
        Initialize document classifier
        
        Args:
            keywords_config: Custom keywords for document types
        """
        self.keywords = keywords_config or self._default_keywords()
    
    @staticmethod
    def _default_keywords() -> Dict[str, List[str]]:
        """Default keywords for Vietnamese banking documents"""
        return {
            'cccd': [
                'căn cước công dân', 'cccd', 'citizen id',
                'quốc tịch', 'nationality', 'số cccd'
            ],
            'cmnd': [
                'chứng minh nhân dân', 'cmnd', 'identity card',
                'số cmnd', 'cmnd số'
            ],
            'bank_statement': [
                'sao kê', 'bank statement', 'transaction history',
                'lịch sử giao dịch', 'số dư', 'balance',
                'số tài khoản', 'account number'
            ],
            'loan_document': [
                'hợp đồng vay', 'loan agreement', 'contract',
                'giải ngân', 'disbursement', 'khoản vay',
                'lãi suất', 'interest rate', 'thời hạn'
            ],
            'payslip': [
                'phiếu lương', 'payslip', 'salary',
                'thu nhập', 'income', 'lương cơ bản'
            ]
        }
    
    def classify(self, text: str, threshold: float = 0.3) -> str:
        """
        Classify document type
        
        Args:
            text: Document text
            threshold: Minimum keyword match ratio
            
        Returns:
            Document type
        """
        text_lower = text.lower()
        scores = {}
        
        for doc_type, keywords in self.keywords.items():
            matches = sum(1 for keyword in keywords if keyword.lower() in text_lower)
            score = matches / len(keywords)
            scores[doc_type] = score
        
        # Get best match
        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]
        
        if best_score >= threshold:
            return best_type
        
        return 'unknown'
    
    def get_confidence(self, text: str, doc_type: str) -> float:
        """
        Get confidence score for document type
        
        Args:
            text: Document text
            doc_type: Document type to check
            
        Returns:
            Confidence score (0-1)
        """
        if doc_type not in self.keywords:
            return 0.0
        
        text_lower = text.lower()
        keywords = self.keywords[doc_type]
        
        matches = sum(1 for keyword in keywords if keyword.lower() in text_lower)
        confidence = matches / len(keywords)
        
        return confidence


class CCCDExtractor:
    """Extract information from CCCD (Citizen ID Card)"""
    
    def __init__(self):
        self.field_extractor = FieldExtractor()
    
    def extract(self, text: str) -> Dict[str, Optional[str]]:
        """
        Extract CCCD information
        
        Args:
            text: OCR text from CCCD
            
        Returns:
            Dictionary with extracted fields
        """
        result = {
            'id_number': None,
            'full_name': None,
            'date_of_birth': None,
            'gender': None,
            'nationality': None,
            'place_of_origin': None,
            'place_of_residence': None,
            'expiry_date': None
        }
        
        # Extract ID number (12 digits for CCCD)
        result['id_number'] = self.field_extractor.extract_field(text, 'cccd')
        
        # Extract name
        result['full_name'] = self._extract_name(text)
        
        # Extract dates
        dates = self.field_extractor.extract_multiple_values(text, 'date_dmy')
        if len(dates) >= 1:
            result['date_of_birth'] = dates[0]
        if len(dates) >= 2:
            result['expiry_date'] = dates[-1]
        
        # Extract gender
        result['gender'] = self.field_extractor.extract_field(text, 'gender')
        
        return result
    
    def _extract_name(self, text: str) -> Optional[str]:
        """Extract full name from CCCD text"""
        # Look for name after keywords
        name_keywords = ['họ và tên', 'họ tên', 'full name', 'name']
        
        for keyword in name_keywords:
            pattern = rf'{keyword}[:\s]*([A-ZÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ][a-zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]+(?:\s+[A-ZÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ][a-zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]+)+)'
            match = re.search(pattern, text, re.IGNORECASE)
            
            if match:
                return match.group(1).strip()
        
        return None


class BankStatementExtractor:
    """Extract information from bank statements"""
    
    def __init__(self):
        self.field_extractor = FieldExtractor()
    
    def extract(self, text: str) -> Dict:
        """
        Extract bank statement information
        
        Args:
            text: OCR text from bank statement
            
        Returns:
            Dictionary with extracted fields
        """
        result = {
            'account_number': None,
            'account_holder': None,
            'statement_period': None,
            'opening_balance': None,
            'closing_balance': None,
            'transactions': []
        }
        
        # Extract account number
        result['account_number'] = self.field_extractor.extract_field(text, 'account_number')
        
        # Extract account holder name
        result['account_holder'] = self._extract_account_holder(text)
        
        # Extract balances
        money_values = self.field_extractor.extract_multiple_values(text, 'money_number')
        if len(money_values) >= 2:
            result['opening_balance'] = money_values[0]
            result['closing_balance'] = money_values[-1]
        
        # Extract transaction lines (simplified)
        result['transactions'] = self._extract_transactions(text)
        
        return result
    
    def _extract_account_holder(self, text: str) -> Optional[str]:
        """Extract account holder name"""
        keywords = ['chủ tài khoản', 'account holder', 'tên chủ tk']
        
        for keyword in keywords:
            pattern = rf'{keyword}[:\s]*([A-ZÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ][a-zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]+(?:\s+[A-ZÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ][a-zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]+)+)'
            match = re.search(pattern, text, re.IGNORECASE)
            
            if match:
                return match.group(1).strip()
        
        return None
    
    def _extract_transactions(self, text: str) -> List[Dict]:
        """Extract transaction lines from statement"""
        transactions = []
        
        # Split into lines
        lines = text.split('\n')
        
        for line in lines:
            # Look for lines with date and amount
            date_match = re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{4}', line)
            amount_match = re.search(r'\d{1,3}(?:[.,]\d{3})*', line)
            
            if date_match and amount_match:
                transaction = {
                    'date': date_match.group(0),
                    'amount': amount_match.group(0),
                    'description': line.strip()
                }
                transactions.append(transaction)
        
        return transactions


if __name__ == "__main__":
    # Test extraction
    test_text = """
    CĂN CƯỚC CÔNG DÂN
    Số: 001234567890
    Họ và tên: NGUYỄN VĂN A
    Ngày sinh: 15/03/1990
    Giới tính: Nam
    Quốc tịch: Việt Nam
    """
    
    # Test CCCD extraction
    cccd_extractor = CCCDExtractor()
    result = cccd_extractor.extract(test_text)
    
    print("CCCD Extraction Result:")
    for field, value in result.items():
        print(f"  {field}: {value}")
    
    # Test document classification
    classifier = DocumentTypeClassifier()
    doc_type = classifier.classify(test_text)
    confidence = classifier.get_confidence(test_text, doc_type)
    
    print(f"\nDocument Type: {doc_type}")
    print(f"Confidence: {confidence:.2f}")
