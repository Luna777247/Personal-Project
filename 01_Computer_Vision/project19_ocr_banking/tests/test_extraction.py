"""
Test Extraction Module
"""
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.extraction.field_extractor import (
    FieldExtractor, DocumentTypeClassifier,
    CCCDExtractor, BankStatementExtractor
)
from src.extraction.postprocessing import TextPostProcessor, ConfidenceCalculator


def test_field_extractor():
    """Test FieldExtractor class"""
    extractor = FieldExtractor()
    
    # Test CCCD number extraction
    text = "Số CCCD: 001234567890"
    cccd = extractor.extract_field(text, 'cccd')
    assert cccd == '001234567890'
    
    # Test date extraction
    text = "Ngày sinh: 15/03/1990"
    date = extractor.extract_field(text, 'date_dmy')
    assert date == '15/03/1990'
    
    # Test money extraction
    text = "Số tiền: 1,000,000 VND"
    money = extractor.extract_field(text, 'money_number')
    assert money is not None


def test_document_classifier():
    """Test DocumentTypeClassifier class"""
    classifier = DocumentTypeClassifier()
    
    # Test CCCD classification
    cccd_text = "CĂN CƯỚC CÔNG DÂN Số: 001234567890 Họ và tên: NGUYỄN VĂN A"
    doc_type = classifier.classify(cccd_text)
    assert doc_type == 'cccd'
    
    # Test confidence
    confidence = classifier.get_confidence(cccd_text, 'cccd')
    assert 0 <= confidence <= 1


def test_cccd_extractor():
    """Test CCCDExtractor class"""
    extractor = CCCDExtractor()
    
    text = """
    CĂN CƯỚC CÔNG DÂN
    Số: 001234567890
    Họ và tên: NGUYỄN VĂN A
    Ngày sinh: 15/03/1990
    Giới tính: Nam
    """
    
    result = extractor.extract(text)
    
    assert isinstance(result, dict)
    assert 'id_number' in result
    assert 'full_name' in result
    assert result['id_number'] == '001234567890'


def test_bank_statement_extractor():
    """Test BankStatementExtractor class"""
    extractor = BankStatementExtractor()
    
    text = """
    SAO KÊ TÀI KHOẢN
    Số tài khoản: 1234567890123456
    Chủ tài khoản: NGUYỄN VĂN A
    Số dư đầu kỳ: 10,000,000
    Số dư cuối kỳ: 12,500,000
    """
    
    result = extractor.extract(text)
    
    assert isinstance(result, dict)
    assert 'account_number' in result
    assert 'account_holder' in result


def test_text_postprocessor():
    """Test TextPostProcessor class"""
    processor = TextPostProcessor()
    
    # Test cleaning
    dirty_text = "Họ  và   tên:  NGUYỄN   VĂN  A@#"
    clean_text = processor.clean_text(dirty_text)
    assert "NGUYỄN VĂN A" in clean_text
    
    # Test ID formatting
    raw_id = "001-234-567-890"
    formatted_id = processor.format_id_number(raw_id)
    assert formatted_id == "001234567890"
    
    # Test date formatting
    raw_date = "15-3-1990"
    formatted_date = processor.format_date(raw_date)
    assert formatted_date == "15/03/1990"


def test_confidence_calculator():
    """Test ConfidenceCalculator class"""
    calculator = ConfidenceCalculator()
    
    # Test CCCD confidence
    cccd_conf = calculator.calculate_field_confidence("001234567890", "cccd")
    assert cccd_conf > 0.9
    
    # Test CMND confidence
    cmnd_conf = calculator.calculate_field_confidence("012345678", "cmnd")
    assert cmnd_conf > 0.85
    
    # Test date confidence
    date_conf = calculator.calculate_field_confidence("15/03/1990", "date_dmy")
    assert date_conf > 0.8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
