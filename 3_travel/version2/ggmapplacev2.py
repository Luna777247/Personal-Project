import requests

def post_search_nearby(latitude, longitude, radius, language_code="", region_code="", 
                  included_types=None, excluded_types=None, included_primary_types=None, 
                  excluded_primary_types=None, max_result_count=1, rank_preference=0):
    """
    Gửi yêu cầu tìm kiếm địa điểm gần đây tới Google Maps Places (New) API sử dụng phương thức POST.
    
    Args:
        latitude (float): Vĩ độ của tâm vòng tròn tìm kiếm (ví dụ: 40).
        longitude (float): Kinh độ của tâm vòng tròn tìm kiếm (ví dụ: -110).
        radius (float): Bán kính tìm kiếm (mét, ví dụ: 10000).
        language_code (str): Mã ngôn ngữ trả về kết quả (mặc định: "").
        region_code (str): Mã vùng (mặc định: "").
        included_types (list): Danh sách các loại địa điểm để bao gồm (tùy chọn).
        excluded_types (list): Danh sách các loại địa điểm để loại trừ (tùy chọn).
        included_primary_types (list): Danh sách các loại chính để bao gồm (tùy chọn).
        excluded_primary_types (list): Danh sách các loại chính để loại trừ (tùy chọn).
        max_result_count (int): Số lượng kết quả tối đa (mặc định: 1).
        rank_preference (int): Thứ tự ưu tiên xếp hạng (0: ưu tiên khoảng cách, 1: ưu tiên mức độ liên quan) (mặc định: 0).
    
    Returns:
        dict: Kết quả JSON từ API hoặc thông báo lỗi nếu yêu cầu thất bại.
    """
    url = "https://google-map-places-new-v2.p.rapidapi.com/v1/places:searchNearby"
    
    # Khởi tạo payload
    payload = {
        "languageCode": language_code,
        "regionCode": region_code,
        "includedTypes": included_types if included_types is not None else [],
        "excludedTypes": excluded_types if excluded_types is not None else [],
        "includedPrimaryTypes": included_primary_types if included_primary_types is not None else [],
        "excludedPrimaryTypes": excluded_primary_types if excluded_primary_types is not None else [],
        "maxResultCount": max_result_count,
        "locationRestriction": {
            "circle": {
                "center": {
                    "latitude": latitude,
                    "longitude": longitude
                },
                "radius": radius
            }
        },
        "rankPreference": rank_preference
    }
    
    # Khởi tạo headers
    headers = {
        "x-rapidapi-key": "02ad4fd6f3msh1f0390da51ae627p19a5cfjsn7f2b23cadfdb",
        "x-rapidapi-host": "google-map-places-new-v2.p.rapidapi.com",
        "Content-Type": "application/json",
        "X-Goog-FieldMask": "*"
    }
    
    try:
        # Gửi yêu cầu POST tới API
        response = requests.post(url, json=payload, headers=headers)
        
        # Kiểm tra trạng thái phản hồi
        response.raise_for_status()
        
        # Trả về kết quả JSON
        return response.json()
    
    except requests.exceptions.RequestException as e:
        # Xử lý lỗi và trả về thông báo lỗi
        return {"error": f"Yêu cầu API thất bại: {str(e)}"}



def post_search_text_places(
    text_query,
    latitude=40.0,
    longitude=-110.0,
    radius=10000,
    language_code="",
    region_code="",
    rank_preference=0,
    included_type="",
    open_now=True,
    min_rating=0,
    max_result_count=100,
    price_levels=None,
    strict_type_filtering=True,
    minimum_charging_rate_kw=0,
    connector_types=None
):
    """
    Gửi yêu cầu tìm kiếm địa điểm bằng text query tới Google Maps Places (New) API.

    Args:
        text_query (str): Chuỗi tìm kiếm (ví dụ: "restaurants" hoặc "123 Main Street").
        latitude (float): Vĩ độ tâm tìm kiếm (mặc định 40.0).
        longitude (float): Kinh độ tâm tìm kiếm (mặc định -110.0).
        radius (int): Bán kính tìm kiếm (mặc định 10000 mét, tối đa 50000).
        language_code (str): Mã ngôn ngữ trả về kết quả.
        region_code (str): Mã vùng (ví dụ: "US", "VN").
        rank_preference (int): Ưu tiên xếp hạng (0: RELEVANCE, 1: DISTANCE).
        included_type (str): Loại địa điểm cần lọc (restaurant, cafe, hospital...).
        open_now (bool): Chỉ trả về địa điểm đang mở cửa.
        min_rating (int): Điểm đánh giá tối thiểu (0–5).
        max_result_count (int): Số kết quả tối đa trả về.
        price_levels (list): Danh sách mức giá (PRICE_LEVEL_...).
        strict_type_filtering (bool): Có lọc chặt theo loại địa điểm hay không.
        minimum_charging_rate_kw (int): Tốc độ sạc tối thiểu (nếu tìm trạm EV).
        connector_types (list): Danh sách loại cổng sạc EV.

    Returns:
        dict: Kết quả JSON từ API hoặc thông báo lỗi.
    """
    url = "https://google-map-places-new-v2.p.rapidapi.com/v1/places:searchText"

    payload = {
        "textQuery": text_query,
        "languageCode": language_code,
        "regionCode": region_code,
        "rankPreference": rank_preference,
        "includedType": included_type,
        "openNow": open_now,
        "minRating": min_rating,
        "maxResultCount": max_result_count,
        "priceLevels": price_levels if price_levels else [],
        "strictTypeFiltering": strict_type_filtering,
        "locationBias": {
            "circle": {
                "center": {"latitude": latitude, "longitude": longitude},
                "radius": radius
            }
        },
        "evOptions": {
            "minimumChargingRateKw": minimum_charging_rate_kw,
            "connectorTypes": connector_types if connector_types else []
        }
    }

    headers = {
        "x-rapidapi-key": "02ad4fd6f3msh1f0390da51ae627p19a5cfjsn7f2b23cadfdb",
        "x-rapidapi-host": "google-map-places-new-v2.p.rapidapi.com",
        "Content-Type": "application/json",
        "X-Goog-FieldMask": "*"
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Yêu cầu API thất bại: {str(e)}"}


def get_place_details(
    place_id: str,
    session_token: str = "",
    language_code: str = "",
    region_code: str = "",
    field_mask: str = "*"
):
    """
    Lấy chi tiết địa điểm từ Google Maps Places (New) API bằng phương thức GET.

    Args:
        place_id (str): ID của địa điểm cần lấy chi tiết (bắt buộc).
        session_token (str): Token phiên autocomplete (tùy chọn).
        language_code (str): Mã ngôn ngữ (tùy chọn, ví dụ: "en", "vi").
        region_code (str): Mã vùng quốc gia (tùy chọn, ví dụ: "US", "VN").
        field_mask (str): Các trường cần lấy từ kết quả (mặc định "*": tất cả).

    Returns:
        dict: Kết quả JSON từ API hoặc thông báo lỗi.
    """
    if not place_id:
        return {"error": "Thiếu place_id. Đây là tham số bắt buộc."}

    # Endpoint URL
    url = f"https://google-map-places-new-v2.p.rapidapi.com/v1/places/{place_id}"

    # Query parameters
    params = {}
    if session_token:
        params["sessionToken"] = session_token
    if language_code:
        params["languageCode"] = language_code
    if region_code:
        params["regionCode"] = region_code

    # Headers
    headers = {
        "x-rapidapi-key": "02ad4fd6f3msh1f0390da51ae627p19a5cfjsn7f2b23cadfdb",
        "x-rapidapi-host": "google-map-places-new-v2.p.rapidapi.com",
        "X-Goog-FieldMask": field_mask
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Yêu cầu API thất bại: {str(e)}"}


def get_place_photo(
    place_id: str,
    photo_resource: str,
    max_width_px: int = 400,
    max_height_px: int = 400,
    skip_http_redirect: bool = True
):
    """
    Lấy ảnh từ Google Maps Places (New) API.

    Args:
        place_id (str): ID của địa điểm (bắt buộc).
        photo_resource (str): Photo resource token (bắt buộc, lấy từ kết quả Place Details).
        max_width_px (int): Chiều rộng tối đa (1-4800, mặc định: 400).
        max_height_px (int): Chiều cao tối đa (1-4800, mặc định: 400).
        skip_http_redirect (bool): Nếu True, trả về JSON thay vì redirect sang ảnh.

    Returns:
        dict hoặc bytes:
            - Nếu skip_http_redirect=True: trả về JSON chứa thông tin ảnh (link download).
            - Nếu skip_http_redirect=False: trả về raw image bytes (ảnh thực tế).
    """
    if not place_id or not photo_resource:
        return {"error": "Thiếu place_id hoặc photo_resource"}

    url = f"https://google-map-places-new-v2.p.rapidapi.com/v1/places/{place_id}/photos/{photo_resource}/media"

    querystring = {}
    if max_width_px:
        querystring["maxWidthPx"] = str(max_width_px)
    if max_height_px:
        querystring["maxHeightPx"] = str(max_height_px)
    if skip_http_redirect:
        querystring["skipHttpRedirect"] = "true"

    headers = {
        "x-rapidapi-key": "02ad4fd6f3msh1f0390da51ae627p19a5cfjsn7f2b23cadfdb",
        "x-rapidapi-host": "google-map-places-new-v2.p.rapidapi.com"
    }

    try:
        response = requests.get(url, headers=headers, params=querystring)

        # Nếu skipHttpRedirect=False => đây là ảnh binary
        if not skip_http_redirect:
            if response.status_code == 200:
                return response.content  # raw image bytes
            else:
                return {"error": f"Lỗi khi tải ảnh: {response.status_code}"}

        # Nếu skipHttpRedirect=True => trả về JSON chứa link ảnh
        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as e:
        return {"error": f"Yêu cầu API thất bại: {str(e)}"}


def post_autocomplete_places(
    input_text: str,
    latitude: float = None,
    longitude: float = None,
    radius: float = 10000,
    included_primary_types=None,
    included_region_codes=None,
    language_code: str = "",
    region_code: str = "",
    origin_lat: float = 0,
    origin_lng: float = 0,
    input_offset: int = 0,
    include_query_predictions: bool = True,
    session_token: str = ""
):
    """
    Gọi API Google Places Autocomplete (New) để gợi ý địa điểm theo từ khóa.

    Args:
        input_text (str): Chuỗi đầu vào cần autocomplete (ví dụ: "Restaurant").
        latitude (float): Vĩ độ tâm locationBias (tùy chọn).
        longitude (float): Kinh độ tâm locationBias (tùy chọn).
        radius (float): Bán kính tìm kiếm (mặc định: 10000 mét).
        included_primary_types (list): Danh sách các loại chính để lọc (tùy chọn).
        included_region_codes (list): Danh sách mã vùng (tùy chọn).
        language_code (str): Ngôn ngữ trả về kết quả.
        region_code (str): Mã vùng hiển thị kết quả.
        origin_lat (float): Vĩ độ gốc (mặc định 0).
        origin_lng (float): Kinh độ gốc (mặc định 0).
        input_offset (int): Offset ký tự đang nhập (mặc định 0).
        include_query_predictions (bool): Có bao gồm gợi ý truy vấn không.
        session_token (str): Token cho phiên autocomplete (để tối ưu billing).

    Returns:
        dict: JSON phản hồi từ API hoặc thông báo lỗi.
    """
    url = "https://google-map-places-new-v2.p.rapidapi.com/v1/places:autocomplete"

    payload = {
        "input": input_text,
        "locationBias": None,
        "includedPrimaryTypes": included_primary_types if included_primary_types is not None else [],
        "includedRegionCodes": included_region_codes if included_region_codes is not None else [],
        "languageCode": language_code,
        "regionCode": region_code,
        "origin": {"latitude": origin_lat, "longitude": origin_lng},
        "inputOffset": input_offset,
        "includeQueryPredictions": include_query_predictions,
        "sessionToken": session_token
    }

    # Nếu có locationBias thì thêm vào payload
    if latitude is not None and longitude is not None:
        payload["locationBias"] = {
            "circle": {
                "center": {"latitude": latitude, "longitude": longitude},
                "radius": radius
            }
        }

    headers = {
        "x-rapidapi-key": "02ad4fd6f3msh1f0390da51ae627p19a5cfjsn7f2b23cadfdb",
        "x-rapidapi-host": "google-map-places-new-v2.p.rapidapi.com",
        "Content-Type": "application/json",
        "X-Goog-FieldMask": "*"
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Yêu cầu API thất bại: {str(e)}"}



# Sử dụng hàm
if __name__ == "__main__":
    
    result = post_search_nearby(
        latitude=40,
        longitude=-110,
        radius=10000,
        language_code="en",
        region_code="en",
        included_types=["restaurant", "cafe"],
        max_result_count=1,
        rank_preference=0
    )
    print(result)
    
    result = post_search_text_places(
        text_query="restaurants",
        latitude=40,
        longitude=-110,
        radius=5000,
        open_now=True,
        max_result_count=3
    )
    print(result)

    result = get_place_details(
        place_id="ChIJj61dQgK6j4AR4GeTYWZsKWw",
        language_code="en"
    )
    print(result)
    
    result = get_place_photo(
        place_id="ChIJ2fzCmcW7j4AR2JzfXBBoh6E",
        photo_resource="AUacShh3_Dd8yvV2JZMtNjjbbSbFhSv-0VmUN-uasQ2Oj00XB63irPTks0-A_1rMNfdTunoOVZfVOExRRBNrupUf8TY4Kw5iQNQgf2rwcaM8hXNQg7KDyvMR5B-HzoCE1mwy2ba9yxvmtiJrdV-xBgO8c5iJL65BCd0slyI1",
        max_width_px=600,
        max_height_px=600,
        skip_http_redirect=True
    )
    print(result)

    result = post_autocomplete_places(
        input_text="Restaurant",
        latitude=40,
        longitude=-110,
        radius=10000,
        language_code="en",
        region_code="us"
    )
    print(result)