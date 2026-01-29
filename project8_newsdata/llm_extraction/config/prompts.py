"""
Prompt Templates for LLM-Based Disaster Information Extraction

This module contains carefully crafted prompts for extracting structured
disaster information from Vietnamese news articles using Large Language Models.
"""

from typing import Dict, Any, List


# Main extraction prompt template
DISASTER_EXTRACTION_PROMPT = """
Bạn là chuyên gia phân tích thông tin thiên tai từ bài báo tiếng Việt.
Hãy trích xuất thông tin chính xác từ bài báo dưới đây và trả về dưới dạng JSON.

**YÊU CẦU:**
- Chỉ trả về JSON, không có text khác
- Sử dụng tiếng Việt cho các giá trị text
- Để trống (null hoặc "") nếu không có thông tin
- Sử dụng format chuẩn cho ngày tháng (DD/MM/YYYY)
- Ước tính số liệu nếu bài báo không nêu rõ

**THÔNG TIN CẦN TRÍCH XUẤT:**
- type: Loại thiên tai (bão, lũ, động đất, sạt lở, v.v.)
- location: Địa điểm xảy ra (tỉnh/thành phố/huyện)
- time: Thời gian xảy ra (ngày/tháng/năm)
- severity: Mức độ nghiêm trọng (nhẹ, trung bình, nặng, rất nặng)
- damage: Mô tả thiệt hại về vật chất
- deaths: Số người chết (số nguyên)
- injured: Số người bị thương (số nguyên)
- missing: Số người mất tích (số nguyên)
- affected_people: Tổng số người bị ảnh hưởng
- affected_area: Diện tích bị ảnh hưởng
- organizations: Các cơ quan/tổ chức tham gia ứng cứu (mảng)
- forecast: Dự báo thời tiết tiếp theo
- response_actions: Các biện pháp ứng cứu đã thực hiện
- source: Nguồn thông tin/cơ quan phát hành

**BÀI BÁO:**
{text}

**JSON OUTPUT:**
"""

# Simplified prompt for basic extraction
BASIC_EXTRACTION_PROMPT = """
Trích xuất thông tin thiên tai từ bài báo tiếng Việt dưới đây.
Chỉ trả về JSON với các trường: type, location, time, damage, deaths, missing, forecast.

**BÀI BÁO:**
{text}

**JSON:**
"""

# Detailed extraction with confidence scores
DETAILED_EXTRACTION_PROMPT = """
Phân tích bài báo thiên tai tiếng Việt và trích xuất thông tin chi tiết.
Đánh giá độ tin cậy cho mỗi thông tin (0.0-1.0).

**BÀI BÁO:**
{text}

**YÊU CẦU OUTPUT JSON:**
{
  "extraction": {
    "type": {"value": "loại thiên tai", "confidence": 0.9},
    "location": {"value": "địa điểm", "confidence": 0.8},
    "time": {"value": "thời gian", "confidence": 0.95},
    "damage": {"value": "thiệt hại", "confidence": 0.7},
    "casualties": {
      "deaths": {"value": 5, "confidence": 0.95},
      "injured": {"value": 12, "confidence": 0.9},
      "missing": {"value": 2, "confidence": 0.8}
    }
  },
  "metadata": {
    "model_used": "gpt-4",
    "processing_time": 2.3,
    "text_length": 1500
  }
}
"""

# Multi-article batch processing prompt
BATCH_EXTRACTION_PROMPT = """
Xử lý nhiều bài báo thiên tai và trích xuất thông tin từ mỗi bài.
Trả về mảng JSON với thông tin của từng bài báo.

**BÀI BÁO (cách nhau bởi ---):**
{text}

**OUTPUT JSON ARRAY:**
"""

# Validation and correction prompt
VALIDATION_PROMPT = """
Kiểm tra và sửa chữa thông tin thiên tai đã trích xuất.
So sánh với bài báo gốc và điều chỉnh nếu cần.

**BÀI BÁO GỐC:**
{text}

**THÔNG TIN HIỆN TẠI:**
{current_extraction}

**HƯỚNG DẪN:**
- Giữ nguyên thông tin chính xác
- Sửa thông tin sai lệch
- Bổ sung thông tin bị thiếu
- Giảm confidence cho thông tin không chắc chắn

**JSON SỬA CHỮA:**
"""

# Quality assessment prompt
QUALITY_CHECK_PROMPT = """
Đánh giá chất lượng thông tin thiên tai đã trích xuất.
Xác định các vấn đề tiềm ẩn như hallucination, inconsistency, missing info.

**BÀI BÁO:**
{text}

**THÔNG TIN TRÍCH XUẤT:**
{extraction}

**ĐÁNH GIÁ (JSON):**
{
  "overall_quality": "high/medium/low",
  "issues_found": ["hallucination", "missing_data", "inconsistency"],
  "confidence_adjustment": 0.1,
  "recommendations": ["kiểm tra lại số liệu", "bổ sung thông tin"]
}
"""

# Entity relationship extraction prompt
RELATIONSHIP_EXTRACTION_PROMPT = """
Trích xuất các mối quan hệ giữa các thực thể trong bài báo thiên tai.
Xác định quan hệ như: thiên tai xảy ra tại địa điểm, gây thiệt hại, ảnh hưởng đến người.

**BÀI BÁO:**
{text}

**RELATIONSHIPS JSON:**
[
  {
    "subject": "bão số 12",
    "relation": "xảy ra tại",
    "object": "tỉnh Quảng Nam",
    "confidence": 0.9
  }
]
"""

# Prompt templates dictionary
PROMPT_TEMPLATES = {
    "basic": BASIC_EXTRACTION_PROMPT,
    "detailed": DETAILED_EXTRACTION_PROMPT,
    "batch": BATCH_EXTRACTION_PROMPT,
    "validation": VALIDATION_PROMPT,
    "quality_check": QUALITY_CHECK_PROMPT,
    "relationships": RELATIONSHIP_EXTRACTION_PROMPT,
    "full": DISASTER_EXTRACTION_PROMPT
}

# Prompt configurations
PROMPT_CONFIGS = {
    "basic": {
        "max_tokens": 1000,
        "temperature": 0.1,
        "description": "Basic disaster information extraction"
    },
    "detailed": {
        "max_tokens": 2000,
        "temperature": 0.1,
        "description": "Detailed extraction with confidence scores"
    },
    "batch": {
        "max_tokens": 3000,
        "temperature": 0.1,
        "description": "Batch processing multiple articles"
    },
    "validation": {
        "max_tokens": 1500,
        "temperature": 0.2,
        "description": "Validation and correction of extractions"
    },
    "quality_check": {
        "max_tokens": 1000,
        "temperature": 0.1,
        "description": "Quality assessment of extracted information"
    },
    "relationships": {
        "max_tokens": 2000,
        "temperature": 0.1,
        "description": "Entity relationship extraction"
    },
    "full": {
        "max_tokens": 2500,
        "temperature": 0.1,
        "description": "Complete disaster information extraction"
    }
}

# Vietnamese-specific prompt enhancements
VIETNAMESE_ENHANCEMENTS = {
    "date_formats": "DD/MM/YYYY, DD-MM-YYYY, ngày DD tháng MM năm YYYY",
    "number_formats": "sử dụng dấu chấm làm hàng nghìn (1.200), không dùng comma",
    "location_hierarchy": "tỉnh > huyện > xã, ưu tiên địa điểm cụ thể nhất",
    "disaster_types": "bão, lũ, lũ quét, động đất, sạt lở, cháy rừng, hạn hán",
    "severity_levels": "nhẹ, trung bình, nặng, rất nặng, thảm họa"
}

def get_prompt_template(template_name: str = "full", **kwargs) -> str:
    """Get a prompt template with optional customization"""
    if template_name not in PROMPT_TEMPLATES:
        raise ValueError(f"Unknown prompt template: {template_name}")

    template = PROMPT_TEMPLATES[template_name]

    # Apply customizations
    for key, value in kwargs.items():
        template = template.replace(f"{{{key}}}", str(value))

    return template

def create_custom_prompt(fields: List[str], language: str = "vietnamese") -> str:
    """Create a custom prompt for specific fields"""
    field_descriptions = {
        "type": "Loại thiên tai (bão, lũ, động đất, v.v.)",
        "location": "Địa điểm xảy ra thiên tai",
        "time": "Thời gian xảy ra",
        "severity": "Mức độ nghiêm trọng",
        "damage": "Thiệt hại về vật chất",
        "deaths": "Số người chết",
        "injured": "Số người bị thương",
        "missing": "Số người mất tích",
        "organizations": "Cơ quan/tổ chức tham gia",
        "forecast": "Dự báo thời tiết tiếp theo"
    }

    field_list = []
    for field in fields:
        if field in field_descriptions:
            field_list.append(f"- {field}: {field_descriptions[field]}")

    fields_text = "\n".join(field_list)

    prompt = f"""
Bạn là chuyên gia phân tích thông tin thiên tai từ bài báo {language}.
Hãy trích xuất thông tin từ bài báo và trả về JSON.

**THÔNG TIN CẦN TRÍCH XUẤT:**
{fields_text}

**BÀI BÁO:**
{{text}}

**JSON OUTPUT:**
"""

    return prompt