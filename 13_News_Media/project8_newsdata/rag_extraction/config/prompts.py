"""
RAG Prompts for Disaster Information Extraction

This module contains prompt templates for the RAG-based extraction system,
including query generation, context processing, and information extraction.
"""

from typing import Dict, Any, List

# Disaster Query Generation Prompts
DISASTER_QUERY_PROMPTS = {
    "vietnamese_basic": """
Dựa trên chủ đề thiên tai sau: "{topic}"
Hãy tạo ra các câu truy vấn tìm kiếm phù hợp bằng tiếng Việt để tìm thông tin liên quan trong cơ sở dữ liệu bài báo.

Yêu cầu:
- Tạo 3-5 câu truy vấn khác nhau
- Tập trung vào các khía cạnh: loại thiên tai, địa điểm, thời gian, thiệt hại, ứng phó
- Sử dụng ngôn ngữ tự nhiên, gần gũi với cách viết báo Việt Nam
- Mỗi câu truy vấn nên ngắn gọn nhưng đầy đủ thông tin

Ví dụ cho chủ đề "Bão lũ tại Quảng Bình":
- "Thông tin về bão lũ gây thiệt hại tại Quảng Bình"
- "Thiệt hại do bão lũ ở Quảng Bình tháng 10 năm 2023"
- "Công tác cứu hộ bão lũ Quảng Bình"

Chủ đề: {topic}
Câu truy vấn:
""",

    "vietnamese_detailed": """
Bạn là chuyên gia phân tích thông tin thiên tai. Dựa trên chủ đề: "{topic}"

Hãy tạo ra các câu truy vấn chi tiết để tìm kiếm thông tin trong kho dữ liệu bài báo Việt Nam.

Yêu cầu tạo truy vấn cho:
1. Thông tin chung về thiên tai
2. Thiệt hại và số liệu cụ thể
3. Công tác ứng phó và cứu hộ
4. Dự báo và cảnh báo
5. Tác động lâu dài

Mỗi truy vấn nên:
- Sử dụng tiếng Việt
- Cụ thể và có tính truy vấn cao
- Bao gồm các từ khóa quan trọng
- Phù hợp với phong cách viết báo Việt Nam

Chủ đề: {topic}
Kết quả trả về dưới dạng danh sách các câu truy vấn:
""",

    "english_fallback": """
Generate search queries for disaster information based on topic: "{topic}"

Create 3-5 different search queries in Vietnamese that would be effective for finding relevant information in Vietnamese news articles.

Focus on:
- Disaster type and location
- Damage and casualties
- Response and rescue operations
- Timeline and forecasts

Queries should be natural and match Vietnamese news writing style.
"""
}

# Context Processing Prompts
CONTEXT_PROCESSING_PROMPTS = {
    "chunk_relevance": """
Đánh giá mức độ liên quan của đoạn văn bản sau với câu truy vấn: "{query}"

Đoạn văn bản:
{text_chunk}

Yêu cầu:
- Đánh giá mức độ liên quan (0-10)
- Giải thích lý do
- Xác định thông tin quan trọng về thiên tai nếu có

Định dạng trả về:
Điểm liên quan: [số]
Giải thích: [văn bản]
Thông tin thiên tai: [tóm tắt nếu có]
""",

    "chunk_summarization": """
Tóm tắt nội dung chính của đoạn văn bản sau, tập trung vào thông tin thiên tai:

Đoạn văn bản:
{text_chunk}

Yêu cầu tóm tắt:
- Loại thiên tai
- Địa điểm ảnh hưởng
- Thời gian xảy ra
- Thiệt hại và số liệu
- Công tác ứng phó

Tóm tắt ngắn gọn:
""",

    "multi_chunk_synthesis": """
Kết hợp thông tin từ nhiều đoạn văn bản liên quan để tạo ra bức tranh tổng thể về thiên tai.

Các đoạn văn bản:
{chunks_text}

Câu truy vấn gốc: "{query}"

Yêu cầu:
- Tổng hợp thông tin từ tất cả các đoạn
- Loại bỏ thông tin trùng lặp
- Tạo ra bản tóm tắt mạch lạc
- Tập trung vào các khía cạnh quan trọng của thiên tai

Kết quả tổng hợp:
"""
}

# Information Extraction Prompts
EXTRACTION_PROMPTS = {
    "disaster_extraction_rag": """
Dựa trên các đoạn văn bản liên quan được truy xuất từ câu truy vấn: "{query}"

Nội dung các đoạn:
{context_chunks}

Yêu cầu trích xuất thông tin thiên tai theo định dạng JSON:

{{
  "type": "loại thiên tai (Bão, Lũ, Động đất, v.v.)",
  "location": "địa điểm ảnh hưởng",
  "time": "thời gian xảy ra",
  "severity": "mức độ nghiêm trọng",
  "damage": "thiệt hại về vật chất",
  "deaths": "số người chết",
  "injured": "số người bị thương",
  "missing": "số người mất tích",
  "organizations": ["tên các tổ chức tham gia"],
  "forecast": "dự báo nếu có",
  "response_actions": "công tác ứng phó",
  "source": "nguồn thông tin"
}}

Hướng dẫn:
- Chỉ trích xuất thông tin có trong văn bản
- Nếu không có thông tin, để trống hoặc null
- Sử dụng tiếng Việt cho các giá trị
- Đảm bảo định dạng JSON hợp lệ
- Tập trung vào thông tin chính xác và có thể kiểm chứng

Kết quả JSON:
""",

    "disaster_extraction_structured": """
Phân tích và trích xuất thông tin có cấu trúc từ các đoạn văn bản về thiên tai.

Văn bản nguồn:
{context_chunks}

Truy vấn: "{query}"

Trích xuất thông tin theo cấu trúc sau:

**THÔNG TIN THIÊN TAI**
- Loại: [loại thiên tai]
- Vị trí: [địa điểm]
- Thời gian: [khoảng thời gian]

**THIỆT HẠI**
- Vật chất: [mô tả thiệt hại]
- Nhân mạng: [số liệu người chết, bị thương, mất tích]

**PHẢN ỨNG & ỨNG PHÓ**
- Tổ chức tham gia: [danh sách]
- Biện pháp: [công tác cứu hộ]

**DỰ BÁO**
- Tình hình tương lai: [nếu có]

**NGUỒN GỐC**
- Nguồn tin: [nguồn bài báo]

Đảm bảo thông tin chính xác và có căn cứ từ văn bản.
""",

    "confidence_scoring": """
Đánh giá độ tin cậy của thông tin thiên tai đã trích xuất.

Thông tin đã trích xuất:
{extracted_info}

Văn bản nguồn:
{context_chunks}

Yêu cầu đánh giá:
1. Độ chính xác của thông tin (0-10)
2. Mức độ chi tiết (0-10)
3. Tính nhất quán với văn bản nguồn (0-10)
4. Khả năng kiểm chứng (0-10)

Điểm tổng thể: [tính trung bình]

Giải thích đánh giá:
"""
}

# Query Expansion Prompts
QUERY_EXPANSION_PROMPTS = {
    "synonym_expansion": """
Mở rộng câu truy vấn về thiên tai bằng các từ đồng nghĩa và biến thể.

Câu truy vấn gốc: "{query}"

Tạo ra các biến thể:
1. Thay thế từ đồng nghĩa
2. Sử dụng cách diễn đạt khác
3. Thêm ngữ cảnh địa phương Việt Nam

Ví dụ cho "bão lũ":
- "bão kèm theo lũ lụt"
- "thiên tai bão và lũ"
- "bão gây ngập lụt"

Các câu truy vấn mở rộng:
""",

    "temporal_expansion": """
Mở rộng truy vấn theo khía cạnh thời gian.

Truy vấn gốc: "{query}"

Tạo truy vấn cho:
- Thời gian cụ thể
- Khoảng thời gian
- Thời gian tương đối (hôm qua, tuần này, v.v.)

Ví dụ cho "bão tại Quảng Ninh":
- "bão Quảng Ninh tháng 10 năm 2023"
- "bão xảy ra ở Quảng Ninh tuần trước"
- "thông tin bão Quảng Ninh mới nhất"
"""
}

# Vietnamese Language Enhancements
VIETNAMESE_ENHANCEMENTS = {
    "disaster_keywords": [
        "thiên tai", "bão", "lũ", "lũ lụt", "động đất", "sạt lở",
        "hạn hán", "mưa lớn", "ngập lụt", "thiệt hại", "cứu hộ",
        "ứng phó", "thảm họa", "tại nạn", "bị thương", "chết", "mất tích"
    ],

    "location_keywords": [
        "tỉnh", "thành phố", "huyện", "xã", "thôn", "khu vực",
        "miền", "vùng", "khu", "làng", "ấp"
    ],

    "temporal_keywords": [
        "hôm qua", "hôm nay", "tuần này", "tháng này", "năm nay",
        "sáng", "chiều", "tối", "đêm", "ngày", "tháng", "năm"
    ]
}

# Default Prompt Selection
DEFAULT_PROMPTS = {
    "query_generation": "vietnamese_detailed",
    "context_processing": "chunk_relevance",
    "extraction": "disaster_extraction_rag",
    "confidence": "confidence_scoring"
}

def get_prompt(prompt_type: str, prompt_name: str = None) -> str:
    """Get prompt template by type and name"""
    if prompt_name is None:
        prompt_name = DEFAULT_PROMPTS.get(prompt_type, "")

    prompt_collections = {
        "query_generation": DISASTER_QUERY_PROMPTS,
        "context_processing": CONTEXT_PROCESSING_PROMPTS,
        "extraction": EXTRACTION_PROMPTS,
        "confidence": EXTRACTION_PROMPTS  # confidence_scoring is in EXTRACTION_PROMPTS
    }

    collection = prompt_collections.get(prompt_type, {})
    return collection.get(prompt_name, "")

def format_prompt(prompt_template: str, **kwargs) -> str:
    """Format prompt template with variables"""
    try:
        return prompt_template.format(**kwargs)
    except KeyError as e:
        raise ValueError(f"Missing required variable for prompt: {e}")

def create_extraction_prompt(query: str, context_chunks: str, prompt_type: str = "disaster_extraction_rag") -> str:
    """Create formatted extraction prompt"""
    template = get_prompt("extraction", prompt_type)
    return format_prompt(template, query=query, context_chunks=context_chunks)

def create_query_prompt(topic: str, prompt_type: str = "vietnamese_detailed") -> str:
    """Create formatted query generation prompt"""
    template = get_prompt("query_generation", prompt_type)
    return format_prompt(template, topic=topic)

if __name__ == "__main__":
    # Test prompt formatting
    test_query = "bão lũ tại Quảng Bình"
    test_chunks = "Đoạn văn bản về bão lũ..."

    prompt = create_extraction_prompt(test_query, test_chunks)
    print("Test extraction prompt:")
    print(prompt[:200] + "...")

    query_prompt = create_query_prompt("động đất Hà Giang")
    print("\nTest query prompt:")
    print(query_prompt[:200] + "...")