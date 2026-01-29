import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get RapidAPI credentials from environment variables
RAPIDAPI_KEY = os.environ.get("RAPIDAPI_KEY")
RAPIDAPI_HOST = os.environ.get("RAPIDAPI_HOST")

def get_text_search(query, radius=1000, opennow=True, location=None, language="en", 
                region="en", minprice=None, maxprice=None, type=None, pagetoken=None):
    """
    Gửi yêu cầu tìm kiếm văn bản tới Google Map Places API (Text Search) sử dụng phương thức GET.
    
    Args:
        query (str): Chuỗi truy vấn văn bản (ví dụ: "restaurants in Sydney").
        radius (int): Bán kính tìm kiếm (mét, mặc định: 1000).
        opennow (bool): Chỉ trả về các địa điểm đang mở (mặc định: True).
        location (str): Tọa độ vĩ độ,kinh độ (ví dụ: "40,-110") (tùy chọn).
        language (str): Mã ngôn ngữ trả về kết quả (mặc định: "en").
        region (str): Mã vùng (mặc định: "en").
        minprice (str): Mức giá tối thiểu (0-4) (tùy chọn).
        maxprice (str): Mức giá tối đa (0-4) (tùy chọn).
        type (str): Loại địa điểm cụ thể (tùy chọn).
        pagetoken (str): Token để lấy trang kết quả tiếp theo (tùy chọn).
    
    Returns:
        dict: Kết quả JSON từ API hoặc thông báo lỗi nếu yêu cầu thất bại.
    """
    url = "https://google-map-places.p.rapidapi.com/maps/api/place/textsearch/json"
    
    # Khởi tạo query parameters
    querystring = {
        "query": query,
        "radius": str(radius),
        "opennow": str(opennow).lower(),
        "language": language,
        "region": region
    }
    
    # Thêm các tham số tùy chọn nếu có
    if location:
        querystring["location"] = location
    if minprice is not None:
        querystring["minprice"] = str(minprice)
    if maxprice is not None:
        querystring["maxprice"] = str(maxprice)
    if type:
        querystring["type"] = type
    if pagetoken:
        querystring["pagetoken"] = pagetoken
    
    # Khởi tạo headers
    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": RAPIDAPI_HOST
    }
    
    try:
        # Gửi yêu cầu GET tới API
        response = requests.get(url, headers=headers, params=querystring)
        
        # Kiểm tra trạng thái phản hồi
        response.raise_for_status()
        
        # Trả về kết quả JSON
        return response.json()
    
    except requests.exceptions.RequestException as e:
        # Xử lý lỗi và trả về thông báo lỗi
        return {"error": f"Yêu cầu API thất bại: {str(e)}"}



def get_place_details(place_id, region="en", fields="all", language="en", 
                  reviews_no_translations=True, reviews_sort=None, sessiontoken=None):
    """
    Gửi yêu cầu lấy chi tiết địa điểm tới Google Map Places API (Place Details) sử dụng phương thức GET.
    
    Args:
        place_id (str): Mã định danh duy nhất của địa điểm (ví dụ: "ChIJN1t_tDeuEmsRUsoyG83frY4").
        region (str): Mã vùng (mặc định: "en").
        fields (str): Danh sách các trường dữ liệu cần trả về, phân tách bằng dấu phẩy (mặc định: "all").
        language (str): Mã ngôn ngữ trả về kết quả (mặc định: "en").
        reviews_no_translations (bool): Tắt dịch đánh giá nếu True, bật nếu False (mặc định: True).
        reviews_sort (str): Phương thức sắp xếp đánh giá ("most_relevant" hoặc "newest") (tùy chọn).
        sessiontoken (str): Token phiên tự hoàn thành cho mục đích thanh toán (tùy chọn).
    
    Returns:
        dict: Kết quả JSON từ API hoặc thông báo lỗi nếu yêu cầu thất bại.
    """
    url = "https://google-map-places.p.rapidapi.com/maps/api/place/details/json"
    
    # Khởi tạo query parameters
    querystring = {
        "place_id": place_id,
        "region": region,
        "fields": fields,
        "language": language,
        "reviews_no_translations": str(reviews_no_translations).lower()
    }
    
    # Thêm các tham số tùy chọn nếu có
    if reviews_sort:
        querystring["reviews_sort"] = reviews_sort
    if sessiontoken:
        querystring["sessiontoken"] = sessiontoken
    
    # Khởi tạo headers
    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": RAPIDAPI_HOST
    }
    
    try:
        # Gửi yêu cầu GET tới API
        response = requests.get(url, headers=headers, params=querystring)
        
        # Kiểm tra trạng thái phản hồi
        response.raise_for_status()
        
        # Trả về kết quả JSON
        return response.json()
    
    except requests.exceptions.RequestException as e:
        # Xử lý lỗi và trả về thông báo lỗi
        return {"error": f"Yêu cầu API thất bại: {str(e)}"}


def get_find_place_from_text(input_text, input_type="textquery", fields="all", language="en", locationbias=None):
    """
    Gửi yêu cầu tìm kiếm địa điểm từ văn bản tới Google Map Places API (Find Place from Text) sử dụng phương thức GET.
    
    Args:
        input_text (str): Chuỗi văn bản để tìm kiếm (ví dụ: "Museum of Contemporary Art Australia").
        input_type (str): Loại đầu vào, có thể là "textquery" hoặc "phonenumber" (mặc định: "textquery").
        fields (str): Danh sách các trường dữ liệu cần trả về, phân tách bằng dấu phẩy (mặc định: "all").
        language (str): Mã ngôn ngữ trả về kết quả (mặc định: "en").
        locationbias (str): Khu vực ưu tiên kết quả (ví dụ: "circle:1000@40,-110" hoặc "ipbias") (tùy chọn).
    
    Returns:
        dict: Kết quả JSON từ API hoặc thông báo lỗi nếu yêu cầu thất bại.
    """
    url = "https://google-map-places.p.rapidapi.com/maps/api/place/findplacefromtext/json"
    
    # Khởi tạo query parameters
    querystring = {
        "input": input_text,
        "inputtype": input_type,
        "fields": fields,
        "language": language
    }
    
    # Thêm tham số tùy chọn nếu có
    if locationbias:
        querystring["locationbias"] = locationbias
    
    # Khởi tạo headers
    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": RAPIDAPI_HOST
    }
    
    try:
        # Gửi yêu cầu GET tới API
        response = requests.get(url, headers=headers, params=querystring)
        
        # Kiểm tra trạng thái phản hồi
        response.raise_for_status()
        
        # Trả về kết quả JSON
        return response.json()
    
    except requests.exceptions.RequestException as e:
        # Xử lý lỗi và trả về thông báo lỗi
        return {"error": f"Yêu cầu API thất bại: {str(e)}"}


def get_nearby_search(location, radius=1000, type=None, language="en", opennow=True, 
                  minprice=None, maxprice=None, name=None, keyword=None, 
                  rankby="prominence", pagetoken=None):
    """
    Gửi yêu cầu tìm kiếm địa điểm gần đây tới Google Map Places API (Nearby Search) sử dụng phương thức GET.
    
    Args:
        location (str): Tọa độ vĩ độ,kinh độ (ví dụ: "40,-110").
        radius (int): Bán kính tìm kiếm (mét, mặc định: 1000). Không được sử dụng nếu rankby="distance".
        type (str): Loại địa điểm cụ thể (tùy chọn).
        language (str): Mã ngôn ngữ trả về kết quả (mặc định: "en").
        opennow (bool): Chỉ trả về các địa điểm đang mở (mặc định: True).
        minprice (str): Mức giá tối thiểu (0-4) (tùy chọn).
        maxprice (str): Mức giá tối đa (0-4) (tùy chọn).
        name (str): Tên địa điểm hoặc từ khóa bổ sung (tương tự keyword) (tùy chọn).
        keyword (str): Chuỗi văn bản để tìm kiếm (ví dụ: "restaurant") (tùy chọn).
        rankby (str): Phương thức sắp xếp ("prominence" hoặc "distance") (mặc định: "prominence").
        pagetoken (str): Token để lấy trang kết quả tiếp theo (tùy chọn).
    
    Returns:
        dict: Kết quả JSON từ API hoặc thông báo lỗi nếu yêu cầu thất bại.
    """
    url = "https://google-map-places.p.rapidapi.com/maps/api/place/nearbysearch/json"
    
    # Khởi tạo query parameters
    querystring = {
        "location": location,
        "language": language,
        "opennow": str(opennow).lower(),
        "rankby": rankby
    }
    
    # Chỉ thêm radius nếu rankby không phải là "distance"
    if rankby != "distance":
        querystring["radius"] = str(radius)
    
    # Thêm các tham số tùy chọn nếu có
    if type:
        querystring["type"] = type
    if minprice is not None:
        querystring["minprice"] = str(minprice)
    if maxprice is not None:
        querystring["maxprice"] = str(maxprice)
    if name:
        querystring["name"] = name
    if keyword:
        querystring["keyword"] = keyword
    if pagetoken:
        querystring["pagetoken"] = pagetoken
    
    # Khởi tạo headers
    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": RAPIDAPI_HOST
    }
    
    try:
        # Gửi yêu cầu GET tới API
        response = requests.get(url, headers=headers, params=querystring)
        
        # Kiểm tra trạng thái phản hồi
        response.raise_for_status()
        
        # Trả về kết quả JSON
        return response.json()
    
    except requests.exceptions.RequestException as e:
        # Xử lý lỗi và trả về thông báo lỗi
        return {"error": f"Yêu cầu API thất bại: {str(e)}"}



def get_autocomplete(input_text, radius=1000, types=None, strictbounds=True, offset=3, 
                 location=None, origin=None, components=None, language="en", 
                 region="en", locationrestriction=None, locationbias=None, sessiontoken=None):
    """
    Gửi yêu cầu tự động hoàn thành tới Google Map Places API (Place Autocomplete) sử dụng phương thức GET.
    
    Args:
        input_text (str): Chuỗi văn bản để tìm kiếm (ví dụ: "amoeba").
        radius (int): Bán kính tìm kiếm (mét, mặc định: 1000).
        types (str): Loại địa điểm hoặc bộ sưu tập loại (ví dụ: "book_store|cafe") (tùy chọn).
        strictbounds (bool): Chỉ trả về kết quả trong vùng xác định bởi location và radius (mặc định: True).
        offset (int): Vị trí ký tự cuối cùng để khớp gợi ý (mặc định: 3).
        location (str): Tọa độ vĩ độ,kinh độ (ví dụ: "40,-110") (tùy chọn).
        origin (str): Tọa độ gốc để tính khoảng cách thẳng (ví dụ: "40,-110") (tùy chọn).
        components (str): Giới hạn kết quả theo quốc gia (ví dụ: "country:us|country:pr") (tùy chọn).
        language (str): Mã ngôn ngữ trả về kết quả (mặc định: "en").
        region (str): Mã vùng (mặc định: "en").
        locationrestriction (str): Giới hạn kết quả trong khu vực (ví dụ: "circle:1000@40,-110") (tùy chọn).
        locationbias (str): Ưu tiên kết quả trong khu vực (ví dụ: "ipbias" hoặc "circle:1000@40,-110") (tùy chọn).
        sessiontoken (str): Token phiên tự hoàn thành cho mục đích thanh toán (tùy chọn).
    
    Returns:
        dict: Kết quả JSON từ API hoặc thông báo lỗi nếu yêu cầu thất bại.
    """
    url = "https://google-map-places.p.rapidapi.com/maps/api/place/autocomplete/json"
    
    # Khởi tạo query parameters
    querystring = {
        "input": input_text,
        "radius": str(radius),
        "strictbounds": str(strictbounds).lower(),
        "offset": str(offset),
        "language": language,
        "region": region
    }
    
    # Thêm các tham số tùy chọn nếu có
    if types:
        querystring["types"] = types
    if location:
        querystring["location"] = location
    if origin:
        querystring["origin"] = origin
    if components:
        querystring["components"] = components
    if locationrestriction:
        querystring["locationrestriction"] = locationrestriction
    if locationbias:
        querystring["locationbias"] = locationbias
    if sessiontoken:
        querystring["sessiontoken"] = sessiontoken
    
    # Khởi tạo headers
    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": RAPIDAPI_HOST
    }
    
    try:
        # Gửi yêu cầu GET tới API
        response = requests.get(url, headers=headers, params=querystring)
        
        # Kiểm tra trạng thái phản hồi
        response.raise_for_status()
        
        # Trả về kết quả JSON
        return response.json()
    
    except requests.exceptions.RequestException as e:
        # Xử lý lỗi và trả về thông báo lỗi
        return {"error": f"Yêu cầu API thất bại: {str(e)}"}



def get_query_autocomplete(input_text, radius=1000, language="en", location=None, offset=3):
    """
    Gửi yêu cầu tự động hoàn thành truy vấn tới Google Map Places API (Query Autocomplete) sử dụng phương thức GET.
    
    Args:
        input_text (str): Chuỗi văn bản để tìm kiếm (ví dụ: "pizza near par").
        radius (int): Bán kính tìm kiếm (mét, mặc định: 1000).
        language (str): Mã ngôn ngữ trả về kết quả (mặc định: "en").
        location (str): Tọa độ vĩ độ,kinh độ (ví dụ: "40,-110") (tùy chọn).
        offset (int): Vị trí ký tự cuối cùng để khớp gợi ý (mặc định: 3).
    
    Returns:
        dict: Kết quả JSON từ API hoặc thông báo lỗi nếu yêu cầu thất bại.
    """
    url = "https://google-map-places.p.rapidapi.com/maps/api/place/queryautocomplete/json"
    
    # Khởi tạo query parameters
    querystring = {
        "input": input_text,
        "radius": str(radius),
        "language": language,
        "offset": str(offset)
    }
    
    # Thêm tham số tùy chọn nếu có
    if location:
        querystring["location"] = location
    
    # Khởi tạo headers
    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": RAPIDAPI_HOST
    }
    
    try:
        # Gửi yêu cầu GET tới API
        response = requests.get(url, headers=headers, params=querystring)
        
        # Kiểm tra trạng thái phản hồi
        response.raise_for_status()
        
        # Trả về kết quả JSON
        return response.json()
    
    except requests.exceptions.RequestException as e:
        # Xử lý lỗi và trả về thông báo lỗi
        return {"error": f"Yêu cầu API thất bại: {str(e)}"}



def get_place_photo(photo_reference, maxheight=None, maxwidth=None):
    """
    Gửi yêu cầu lấy ảnh địa điểm tới Google Map Places API (Place Photo) sử dụng phương thức GET.
    
    Args:
        photo_reference (str): Mã định danh duy nhất của ảnh (trả về từ Place Search hoặc Place Details).
        maxheight (int): Chiều cao tối đa mong muốn của ảnh (pixel, từ 1 đến 1600) (tùy chọn).
        maxwidth (int): Chiều rộng tối đa mong muốn của ảnh (pixel, từ 1 đến 1600) (tùy chọn).
    
    Returns:
        dict: Kết quả JSON từ API (nếu có) hoặc thông báo lỗi nếu yêu cầu thất bại.
              Lưu ý: API Place Photo thường trả về nội dung ảnh trực tiếp hoặc chuyển hướng đến URL ảnh.
    """
    url = "https://google-map-places.p.rapidapi.com/maps/api/place/photo"
    
    # Khởi tạo query parameters
    querystring = {
        "photo_reference": photo_reference
    }
    
    # Thêm các tham số tùy chọn nếu có
    if maxheight is not None:
        querystring["maxheight"] = str(maxheight)
    if maxwidth is not None:
        querystring["maxwidth"] = str(maxwidth)
    
    # Khởi tạo headers
    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": RAPIDAPI_HOST
    }
    
    try:
        # Gửi yêu cầu GET tới API
        response = requests.get(url, headers=headers, params=querystring, allow_redirects=True)
        
        # Kiểm tra trạng thái phản hồi
        response.raise_for_status()
        
        # Kiểm tra xem phản hồi có phải là JSON hay không
        content_type = response.headers.get("Content-Type", "")
        if "application/json" in content_type:
            return response.json()
        else:
            # API Place Photo thường trả về nội dung ảnh hoặc URL chuyển hướng
            return {
                "url": response.url,  # URL ảnh sau khi chuyển hướng
                "content_type": content_type,
                "status_code": response.status_code
            }
    
    except requests.exceptions.RequestException as e:
        # Xử lý lỗi và trả về thông báo lỗi
        return {"error": f"Yêu cầu API thất bại: {str(e)}"}

    
def get_directions(origin, destination, mode="driving", departure_time=None, traffic_model="best_guess",
               region="en", transit_routing_preference=None, avoid=None, alternatives=True,
               units="metric", transit_mode=None, waypoints=None, language="en"):
    """
    Gửi yêu cầu chỉ đường tới Google Maps Directions API sử dụng phương thức GET.
    
    Args:
        origin (str): Điểm xuất phát (place_id, địa chỉ, hoặc tọa độ vĩ độ,kinh độ).
        destination (str): Điểm đích (place_id, địa chỉ, hoặc tọa độ vĩ độ,kinh độ).
        mode (str): Chế độ di chuyển ("driving", "walking", "bicycling", "transit") (mặc định: "driving").
        departure_time (str): Thời gian khởi hành (giây kể từ 1970-01-01 UTC hoặc "now") (tùy chọn).
        traffic_model (str): Mô hình giao thông ("best_guess", "pessimistic", "optimistic") (mặc định: "best_guess").
        region (str): Mã vùng (mặc định: "en").
        transit_routing_preference (str): Ưu tiên tuyến đường công cộng ("less_walking", "fewer_transfers") (tùy chọn).
        avoid (str): Các yếu tố cần tránh ("tolls", "highways", "ferries", "indoor") (tùy chọn).
        alternatives (bool): Trả về nhiều tuyến đường thay thế nếu True (mặc định: True).
        units (str): Hệ đơn vị ("metric" hoặc "imperial") (mặc định: "metric").
        transit_mode (str): Chế độ công cộng ("bus", "subway", "train", "tram", "rail") (tùy chọn).
        waypoints (str): Các điểm trung gian (place_id, địa chỉ, tọa độ, hoặc "optimize:true|...") (tùy chọn).
        language (str): Mã ngôn ngữ trả về kết quả (mặc định: "en").
    
    Returns:
        dict: Kết quả JSON từ API hoặc thông báo lỗi nếu yêu cầu thất bại.
    """
    url = "https://google-map-places.p.rapidapi.com/maps/api/directions/json"
    
    # Khởi tạo query parameters
    querystring = {
        "origin": origin,
        "destination": destination,
        "mode": mode,
        "alternatives": str(alternatives).lower(),
        "units": units,
        "language": language,
        "region": region
    }
    
    # Thêm các tham số tùy chọn nếu có
    if departure_time:
        querystring["departure_time"] = str(departure_time)
    if traffic_model:
        querystring["traffic_model"] = traffic_model
    if transit_routing_preference:
        querystring["transit_routing_preference"] = transit_routing_preference
    if avoid:
        querystring["avoid"] = avoid
    if transit_mode:
        querystring["transit_mode"] = transit_mode
    if waypoints:
        querystring["waypoints"] = waypoints
    
    # Khởi tạo headers
    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": RAPIDAPI_HOST
    }
    
    try:
        # Gửi yêu cầu GET tới API
        response = requests.get(url, headers=headers, params=querystring)
        
        # Kiểm tra trạng thái phản hồi
        response.raise_for_status()
        
        # Trả về kết quả JSON
        return response.json()
    
    except requests.exceptions.RequestException as e:
        # Xử lý lỗi và trả về thông báo lỗi
        return {"error": f"Yêu cầu API thất bại: {str(e)}"}


def get_geocode(address=None, place_id=None, latlng=None, bounds=None, language="en", 
            region="en", result_type=None, location_type=None, components=None):
    """
    Gửi yêu cầu mã hóa địa lý tới Google Maps Geocoding API sử dụng phương thức GET.
    
    Args:
        address (str): Địa chỉ hoặc mã Plus Code để mã hóa địa lý (ví dụ: "1600 Amphitheatre Parkway, Mountain View, CA") (tùy chọn).
        place_id (str): Mã định danh duy nhất của địa điểm (tùy chọn).
        latlng (str): Tọa độ vĩ độ,kinh độ (ví dụ: "40,-110") (tùy chọn).
        bounds (str): Hộp giới hạn để ưu tiên kết quả (ví dụ: "rectangle:south,west|north,east") (tùy chọn).
        language (str): Mã ngôn ngữ trả về kết quả (mặc định: "en").
        region (str): Mã vùng (mặc định: "en").
        result_type (str): Loại kết quả (ví dụ: "administrative_area_level_1") (tùy chọn).
        location_type (str): Loại vị trí (ví dụ: "APPROXIMATE") (tùy chọn).
        components (str): Bộ lọc thành phần (ví dụ: "country:us|postal_code:94043") (tùy chọn).
    
    Returns:
        dict: Kết quả JSON từ API hoặc thông báo lỗi nếu yêu cầu thất bại.
    """
    url = "https://google-map-places.p.rapidapi.com/maps/api/geocode/json"
    
    # Khởi tạo query parameters
    querystring = {
        "language": language,
        "region": region
    }
    
    # Thêm ít nhất một trong các tham số address, place_id, hoặc latlng
    if address:
        querystring["address"] = address
    if place_id:
        querystring["place_id"] = place_id
    if latlng:
        querystring["latlng"] = latlng
    
    # Thêm các tham số tùy chọn nếu có
    if bounds:
        querystring["bounds"] = bounds
    if result_type:
        querystring["result_type"] = result_type
    if location_type:
        querystring["location_type"] = location_type
    if components:
        querystring["components"] = components
    
    # Kiểm tra xem có ít nhất một trong address, place_id, hoặc latlng được cung cấp
    if not any([address, place_id, latlng]):
        return {"error": "Phải cung cấp ít nhất một trong address, place_id, hoặc latlng."}
    
    # Khởi tạo headers
    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": RAPIDAPI_HOST
    }
    
    try:
        # Gửi yêu cầu GET tới API
        response = requests.get(url, headers=headers, params=querystring)
        
        # Kiểm tra trạng thái phản hồi
        response.raise_for_status()
        
        # Trả về kết quả JSON
        return response.json()
    
    except requests.exceptions.RequestException as e:
        # Xử lý lỗi và trả về thông báo lỗi
        return {"error": f"Yêu cầu API thất bại: {str(e)}"}


def get_timezone(location, timestamp="1331161200", language="en"):
    """
    Gửi yêu cầu lấy thông tin múi giờ tới Google Maps Time Zone API sử dụng phương thức GET.
    
    Args:
        location (str): Tọa độ vĩ độ,kinh độ (ví dụ: "39.6034810,-119.6822510").
        timestamp (str): Thời gian tính bằng giây kể từ 1970-01-01 UTC (mặc định: "1331161200").
        language (str): Mã ngôn ngữ trả về kết quả (mặc định: "en").
    
    Returns:
        dict: Kết quả JSON từ API hoặc thông báo lỗi nếu yêu cầu thất bại.
    """
    url = "https://google-map-places.p.rapidapi.com/maps/api/timezone/json"
    
    # Khởi tạo query parameters
    querystring = {
        "location": location,
        "timestamp": str(timestamp),
        "language": language
    }
    
    # Khởi tạo headers
    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": RAPIDAPI_HOST
    }
    
    try:
        # Gửi yêu cầu GET tới API
        response = requests.get(url, headers=headers, params=querystring)
        
        # Kiểm tra trạng thái phản hồi
        response.raise_for_status()
        
        # Trả về kết quả JSON
        return response.json()
    
    except requests.exceptions.RequestException as e:
        # Xử lý lỗi và trả về thông báo lỗi
        return {"error": f"Yêu cầu API thất bại: {str(e)}"}



def get_distance_matrix(origins, destinations, mode="driving", transit_routing_preference=None, 
                    traffic_model="best_guess", avoid=None, language="en", departure_time=None, 
                    arrival_time=None, transit_mode=None, units="metric", region="en"):
    """
    Gửi yêu cầu tính khoảng cách và thời gian di chuyển tới Google Maps Distance Matrix API sử dụng phương thức GET.
    
    Args:
        origins (str): Các điểm xuất phát, phân tách bằng dấu | (ví dụ: "Vancouver BC|Seattle").
        destinations (str): Các điểm đích, phân tách bằng dấu | (ví dụ: "San Francisco|Victoria BC").
        mode (str): Chế độ di chuyển ("driving", "walking", "bicycling", "transit") (mặc định: "driving").
        transit_routing_preference (str): Ưu tiên tuyến đường công cộng ("less_walking", "fewer_transfers") (tùy chọn).
        traffic_model (str): Mô hình giao thông ("best_guess", "pessimistic", "optimistic") (mặc định: "best_guess").
        avoid (str): Các yếu tố cần tránh ("tolls", "highways", "ferries", "indoor") (tùy chọn).
        language (str): Mã ngôn ngữ trả về kết quả (mặc định: "en").
        departure_time (str): Thời gian khởi hành (giây kể từ 1970-01-01 UTC hoặc "now") (tùy chọn).
        arrival_time (str): Thời gian đến (giây kể từ 1970-01-01 UTC) (tùy chọn, chỉ cho transit).
        transit_mode (str): Chế độ công cộng ("bus", "subway", "train", "tram", "rail") (tùy chọn).
        units (str): Hệ đơn vị ("metric" hoặc "imperial") (mặc định: "metric").
        region (str): Mã vùng (mặc định: "en").
    
    Returns:
        dict: Kết quả JSON từ API hoặc thông báo lỗi nếu yêu cầu thất bại.
    """
    url = "https://google-map-places.p.rapidapi.com/maps/api/distancematrix/json"
    
    # Khởi tạo query parameters
    querystring = {
        "origins": origins,
        "destinations": destinations,
        "mode": mode,
        "language": language,
        "units": units,
        "region": region
    }
    
    # Thêm các tham số tùy chọn nếu có
    if transit_routing_preference:
        querystring["transit_routing_preference"] = transit_routing_preference
    if traffic_model:
        querystring["traffic_model"] = traffic_model
    if avoid:
        querystring["avoid"] = avoid
    if departure_time:
        querystring["departure_time"] = str(departure_time)
    if arrival_time:
        querystring["arrival_time"] = str(arrival_time)
    if transit_mode:
        querystring["transit_mode"] = transit_mode
    
    # Khởi tạo headers
    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": RAPIDAPI_HOST
    }
    
    try:
        # Gửi yêu cầu GET tới API
        response = requests.get(url, headers=headers, params=querystring)
        
        # Kiểm tra trạng thái phản hồi
        response.raise_for_status()
        
        # Trả về kết quả JSON
        return response.json()
    
    except requests.exceptions.RequestException as e:
        # Xử lý lỗi và trả về thông báo lỗi
        return {"error": f"Yêu cầu API thất bại: {str(e)}"}


def get_street_view(size="600x400", location=None, pano=None, heading=None, pitch=None, fov=None,
                source="default", radius=50, return_error_code=True, signature=None):
    """
    Gửi yêu cầu lấy hình ảnh Street View tới Google Maps Street View API sử dụng phương thức GET.
    
    Args:
        size (str): Kích thước ảnh (width x height, ví dụ: "600x400"). Không vượt quá 640 pixel.
        location (str): Vị trí để lấy hình ảnh (địa chỉ hoặc tọa độ vĩ độ,kinh độ, ví dụ: "Chagrin Falls, OH") (tùy chọn).
        pano (str): ID panorama cụ thể (tùy chọn).
        heading (float): Góc định hướng của camera (0-360 độ) (tùy chọn).
        pitch (float): Góc nghiêng của camera (-90 đến 90 độ) (tùy chọn).
        fov (float): Góc nhìn ngang của ảnh (tối đa 120 độ) (tùy chọn).
        source (str): Nguồn ảnh ("default" hoặc "outdoor") (mặc định: "default").
        radius (int): Bán kính tìm kiếm panorama (mét, mặc định: 50).
        return_error_code (bool): Trả về mã lỗi HTTP thay vì ảnh xám nếu không tìm thấy (mặc định: True).
        signature (str): Chữ ký số để xác thực yêu cầu (tùy chọn).
    
    Returns:
        dict: Kết quả JSON (nếu return_error_code=True và có lỗi) hoặc thông tin về URL ảnh.
    """
    url = "https://google-map-places.p.rapidapi.com/maps/api/streetview"
    
    # Khởi tạo query parameters
    querystring = {
        "size": size,
        "source": source,
        "return_error_code": str(return_error_code).lower(),
        "radius": str(radius)
    }
    
    # Thêm ít nhất một trong location hoặc pano
    if location:
        querystring["location"] = location
    if pano:
        querystring["pano"] = pano
    
    # Thêm các tham số tùy chọn nếu có
    if heading is not None:
        querystring["heading"] = str(heading)
    if pitch is not None:
        querystring["pitch"] = str(pitch)
    if fov is not None:
        querystring["fov"] = str(fov)
    if signature:
        querystring["signature"] = signature
    
    # Kiểm tra xem có ít nhất một trong location hoặc pano được cung cấp
    if not any([location, pano]):
        return {"error": "Phải cung cấp ít nhất một trong location hoặc pano."}
    
    # Khởi tạo headers
    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": RAPIDAPI_HOST
    }
    
    try:
        # Gửi yêu cầu GET tới API
        response = requests.get(url, headers=headers, params=querystring, allow_redirects=True)
        
        # Kiểm tra trạng thái phản hồi
        response.raise_for_status()
        
        # Kiểm tra xem phản hồi có phải là JSON hay không
        content_type = response.headers.get("Content-Type", "")
        if "application/json" in content_type:
            return response.json()  # Trả về JSON nếu có lỗi (khi return_error_code=True)
        else:
            # Trả về thông tin về URL ảnh
            return {
                "url": response.url,
                "content_type": content_type,
                "status_code": response.status_code
            }
    
    except requests.exceptions.RequestException as e:
        # Xử lý lỗi và trả về thông báo lỗi
        return {"error": f"Yêu cầu API thất bại: {str(e)}"}



def get_street_view_metadata(location=None, pano=None, size=None, source="default", pitch=None, 
                        radius=50, return_error_code=True, signature=None, heading=None):
    """
    Gửi yêu cầu lấy siêu dữ liệu Street View tới Google Maps Street View Metadata API sử dụng phương thức GET.
    
    Args:
        location (str): Vị trí để lấy siêu dữ liệu (địa chỉ hoặc tọa độ vĩ độ,kinh độ, ví dụ: "Chagrin Falls, OH") (tùy chọn).
        pano (str): ID panorama cụ thể (tùy chọn).
        size (str): Kích thước ảnh (width x height, ví dụ: "600x400") (tùy chọn).
        source (str): Nguồn ảnh ("default" hoặc "outdoor") (mặc định: "default").
        pitch (float): Góc nghiêng của camera (-90 đến 90 độ) (tùy chọn).
        radius (int): Bán kính tìm kiếm panorama (mét, mặc định: 50).
        return_error_code (bool): Trả về mã lỗi HTTP nếu không tìm thấy hình ảnh (mặc định: True).
        signature (str): Chữ ký số để xác thực yêu cầu (tùy chọn).
        heading (float): Góc định hướng của camera (0-360 độ) (tùy chọn).
    
    Returns:
        dict: Kết quả JSON từ API chứa siêu dữ liệu hoặc thông báo lỗi nếu yêu cầu thất bại.
    """
    url = "https://google-map-places.p.rapidapi.com/maps/api/streetview/metadata"
    
    # Khởi tạo query parameters
    querystring = {
        "source": source,
        "return_error_code": str(return_error_code).lower(),
        "radius": str(radius)
    }
    
    # Thêm ít nhất một trong location hoặc pano
    if location:
        querystring["location"] = location
    if pano:
        querystring["pano"] = pano
    
    # Thêm các tham số tùy chọn nếu có
    if size:
        querystring["size"] = size
    if pitch is not None:
        querystring["pitch"] = str(pitch)
    if signature:
        querystring["signature"] = signature
    if heading is not None:
        querystring["heading"] = str(heading)
    
    # Kiểm tra xem có ít nhất một trong location hoặc pano được cung cấp
    if not any([location, pano]):
        return {"error": "Phải cung cấp ít nhất một trong location hoặc pano."}
    
    # Khởi tạo headers
    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": RAPIDAPI_HOST
    }
    
    try:
        # Gửi yêu cầu GET tới API
        response = requests.get(url, headers=headers, params=querystring)
        
        # Kiểm tra trạng thái phản hồi
        response.raise_for_status()
        
        # Trả về kết quả JSON
        return response.json()
    
    except requests.exceptions.RequestException as e:
        # Xử lý lỗi và trả về thông báo lỗi
        return {"error": f"Yêu cầu API thất bại: {str(e)}"}



# Sử dụng hàm
if __name__ == "__main__":
    result = get_text_search(
        query="restaurants in Sydney",
        radius=1000,
        opennow=True,
        location="40,-110",
        language="en",
        region="en"
    )
    print(result)

    result = get_place_details(
        place_id="ChIJN1t_tDeuEmsRUsoyG83frY4",
        region="en",
        fields="all",
        language="en",
        reviews_no_translations=True
    )
    print(result)

    result = get_find_place_from_text(
        input_text="Museum of Contemporary Art Australia",
        input_type="textquery",
        fields="all",
        language="en"
    )
    print(result)
    
    result = get_nearby_search(
        location="40,-110",
        radius=1000,
        language="en",
        opennow=True,
        rankby="prominence"
    )
    print(result)

    result = get_autocomplete(
        input_text="amoeba",
        radius=1000,
        strictbounds=True,
        offset=3,
        location="40,-110",
        origin="40,-110",
        components="country:us|country:pr",
        language="en",
        region="en"
    )
    print(result)

    result = get_query_autocomplete(
        input_text="pizza near par",
        radius=1000,
        language="en",
        location="40,-110",
        offset=3
    )
    print(result)

    result = get_place_photo(
        photo_reference="ATJ83zhSSAtkh5LTozXMhBghqubeOxnZWUV2m7Hv2tQaIzKQJgvZk9yCaEjBW0r0Zx1oJ9RF1G7oeM34sQQMOv8s2zA0sgGBiyBgvdyMxeVByRgHUXmv-rkJ2wyvNv17jyTSySm_-_6R2B0v4eKX257HOxvXlx_TSwp2NrICKrZM2d5d2P4q",
        maxheight=400,
        maxwidth=400
    )
    print(result)

    result = get_directions(
        origin="Vancouver, BC",
        destination="Victoria, BC",
        mode="driving",
        departure_time="1782624107",
        traffic_model="pessimistic",
        region="en",
        transit_routing_preference="less_walking",
        alternatives=True,
        units="metric",
        transit_mode="train|tram|subway",
        language="en"
    )
    print(result)

    result = get_geocode(
        address="1600 Amphitheatre Parkway, Mountain View, CA",
        language="en",
        region="en",
        result_type="administrative_area_level_1",
        location_type="APPROXIMATE"
    )
    print(result)  

    result = get_timezone(
        location="39.6034810,-119.6822510",
        timestamp="1331161200",
        language="en"
    )
    print(result)

    result = get_distance_matrix(
        origins="Vancouver BC|Seattle",
        destinations="San Francisco|Victoria BC",
        mode="driving",
        transit_routing_preference="less_walking",
        traffic_model="pessimistic",
        avoid="highways",
        language="en",
        departure_time="1782624107",
        units="metric",
        region="en",
        transit_mode="train|tram|subway"
    )
    print(result)
    
    result = get_street_view(
        size="600x400",
        location="Chagrin Falls, OH",
        source="default",
        return_error_code=True
    )
    print(result)
    
    result = get_street_view_metadata(
        location="Chagrin Falls, OH",
        source="default",
        return_error_code=True,
        size="600x400"
    )
    print(result)
    