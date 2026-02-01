# Cấu hình từ khóa cho trích xuất thông tin thiên tai
# Keyword-based Extraction Configuration

# Danh sách từ khóa thiên tai theo loại
DISASTER_KEYWORDS = {
    # Thiên tai khí tượng - thủy văn
    "storm": [
        "bão", "áp thấp nhiệt đới", "lốc xoáy", "vòi rồng", "mưa lớn",
        "mưa kéo dài", "lũ", "lũ lụt", "lũ quét", "ngập úng", "hạn hán",
        "xâm nhập mặn", "sương muối", "rét đậm", "rét hại", "nắng nóng",
        "sóng nhiệt", "cảnh báo bão", "bão số", "siêu bão"
    ],

    # Thiên tai địa chất
    "geological": [
        "động đất", "sóng thần", "núi lửa phun", "sạt lở đất", "trượt đất",
        "sụt lún", "hang động karst sụp đổ", "động đất mạnh", "động đất nhẹ",
        "rung chấn", "núi lửa", "phun trào"
    ],

    # Thiên tai sinh học
    "biological": [
        "dịch bệnh", "dịch bệnh ở người", "dịch bệnh ở động vật",
        "dịch bệnh cây trồng", "sinh vật ngoại lai", "xâm hại",
        "dịch tả", "dịch cúm", "dịch h5n1", "dịch covid"
    ],

    # Thiên tai môi trường - con người gây ra
    "environmental": [
        "cháy rừng", "ô nhiễm môi trường", "tràn dầu", "sự cố hóa chất",
        "phóng xạ", "sự cố phóng xạ", "cháy lớn", "hỏa hoạn"
    ],

    # Từ khóa tác động và hậu quả
    "impact": [
        "thiệt hại", "thiệt hại nặng", "thiệt hại lớn", "tổn thất",
        "mất mát", "chết", "tử vong", "bị thương", "mất tích",
        "bị ảnh hưởng", "bị tác động", "hậu quả", "tác động",
        "người chết", "người bị thương", "người mất tích"
    ],

    # Từ khóa thời tiết và cảnh báo
    "weather": [
        "gió mạnh", "gió giật", "sóng lớn", "sóng cao", "cảnh báo",
        "cảnh báo cấp", "cảnh báo đỏ", "cảnh báo cam", "cảnh báo vàng",
        "cảnh báo xanh", "đề phòng", "phòng tránh", "ứng phó",
        "giảm thiểu thiệt hại"
    ],

    # Từ khóa địa điểm và quy mô
    "location": [
        "tỉnh", "thành phố", "quận", "huyện", "xã", "phường",
        "khu vực", "vùng", "miền", "toàn quốc", "cả nước",
        "miền bắc", "miền trung", "miền nam", "đồng bằng",
        "trung du", "vùng núi", "đồi núi"
    ]
}

# Danh sách từ khóa kết hợp (phrases)
DISASTER_PHRASES = [
    "cảnh báo bão", "bão mạnh", "siêu bão", "bão nhiệt đới",
    "lũ lụt lớn", "lũ quét", "ngập úng nặng", "hạn hán kéo dài",
    "động đất mạnh", "động đất 7.0", "sóng thần cao",
    "sạt lở đất", "trượt lở", "núi lửa phun trào",
    "dịch bệnh covid", "dịch tả lợn", "cháy rừng lớn",
    "ô nhiễm nghiêm trọng", "sự cố hóa chất", "tràn dầu",
    "thiệt hại nặng nề", "thiệt hại hàng tỷ", "nhiều người chết",
    "hàng trăm người", "hàng nghìn người", "tổn thất lớn",
    "mưa lớn kéo dài", "gió mạnh cấp", "sóng biển cao",
    "cảnh báo đỏ", "cảnh báo cấp 4", "cảnh báo cấp 5"
]

# Cấu hình trích xuất
EXTRACTION_CONFIG = {
    "min_sentence_length": 10,  # Độ dài tối thiểu của câu (ký tự)
    "max_sentence_length": 500, # Độ dài tối đa của câu (ký tự)
    "context_window": 2,        # Số câu xung quanh để lấy context
    "case_sensitive": False,    # Có phân biệt hoa thường không
    "remove_duplicates": True,  # Loại bỏ câu trùng lặp
    "min_keyword_matches": 1,   # Số từ khóa tối thiểu để match
}

# Mapping loại thiên tai
DISASTER_TYPE_MAPPING = {
    "Bão, áp thấp nhiệt đới": "storm",
    "Lốc xoáy, vòi rồng": "storm",
    "Mưa lớn kéo dài": "storm",
    "Lũ, lũ quét": "storm",
    "Ngập úng": "storm",
    "Hạn hán": "storm",
    "Xâm nhập mặn": "storm",
    "Sương muối, rét đậm – rét hại": "storm",
    "Nắng nóng, sóng nhiệt": "storm",
    "Động đất": "geological",
    "Sóng thần": "geological",
    "Núi lửa phun": "geological",
    "Sạt lở đất, trượt đất, sụt lún": "geological",
    "Hang động karst sụp đổ": "geological",
    "Dịch bệnh ở người": "biological",
    "Dịch bệnh ở động vật": "biological",
    "Dịch bệnh cây trồng": "biological",
    "Sinh vật ngoại lai xâm hại": "biological",
    "Cháy rừng": "environmental",
    "Ô nhiễm môi trường nghiêm trọng": "environmental",
    "Tràn dầu": "environmental",
    "Sự cố hóa chất, phóng xạ": "environmental"
}