import streamlit as st
import requests
import os
import json
from datetime import datetime, timezone
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Any
import pymongo

from ggmapplacev1 import (
    get_text_search,
    get_place_details,
    get_place_photo,
    get_timezone,
    get_directions
)
from mongo_utils import save_place_to_mongodb

# Constants
PLACE_TYPES = [
    "accounting", "airport", "amusement_park", "aquarium", "art_gallery", "atm", 
    "bakery", "bank", "bar", "beauty_salon", "bicycle_store", "book_store", 
    "bowling_alley", "bus_station", "cafe", "campground", "car_dealer", "car_rental", 
    "car_repair", "car_wash", "casino", "cemetery", "church", "city_hall", 
    "clothing_store", "convenience_store", "courthouse", "dentist", "department_store", 
    "doctor", "drugstore", "electrician", "electronics_store", "embassy", 
    "fire_station", "florist", "funeral_home", "furniture_store", "gas_station", 
    "gym", "hair_care", "hardware_store", "hindu_temple", "home_goods_store", 
    "hospital", "insurance_agency", "jewelry_store", "laundry", "lawyer", 
    "library", "light_rail_station", "liquor_store", "local_government_office", 
    "locksmith", "lodging", "meal_delivery", "meal_takeaway", "mosque", 
    "movie_rental", "movie_theater", "moving_company", "museum", "night_club", 
    "painter", "park", "parking", "pet_store", "pharmacy", "physiotherapist", 
    "plumber", "police", "post_office", "primary_school", "real_estate_agency", 
    "restaurant", "roofing_contractor", "rv_park", "school", "secondary_school", 
    "shoe_store", "shopping_mall", "spa", "stadium", "storage", "store", 
    "subway_station", "supermarket", "synagogue", "taxi_stand", "tourist_attraction", 
    "train_station", "transit_station", "travel_agency", "university", 
    "veterinary_care", "zoo"
]

MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://mongodb:12345@cluster0.olqzq.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")

# Pure functions for data processing
def format_type_display(place_type: str) -> str:
    """Convert place type to display format."""
    return place_type.replace('_', ' ').title()

def get_rating_string(rating: Any, user_ratings_total: Optional[int] = None) -> str:
    """Format rating string with total reviews."""
    if rating == 'Chưa có' or rating is None:
        return 'Chưa có đánh giá'
    rating_str = f"{rating}/5"
    if user_ratings_total:
        rating_str += f" ({user_ratings_total} đánh giá)"
    return rating_str

def get_open_status(opening_hours: Optional[Dict]) -> str:
    """Get open/closed status from opening hours."""
    if not opening_hours or not isinstance(opening_hours, dict):
        return 'Không rõ'
    
    open_now = opening_hours.get('open_now')
    return 'Đang mở cửa' if open_now else ('Đang đóng cửa' if open_now is not None else 'Không rõ')

def extract_place_info(place: Dict[str, Any]) -> Dict[str, str]:
    """Extract and format place information."""
    return {
        'name': place.get('name', 'Không có tên'),
        'address': place.get('formatted_address', 'Không có địa chỉ'),
        'business_status': place.get('business_status', 'Không rõ'),
        'open_status': get_open_status(place.get('opening_hours')),
        'types': ', '.join(place.get('types', [])) or 'Không rõ',
        'rating': get_rating_string(place.get('rating'), place.get('user_ratings_total')),
        'search_type': place.get('search_type', ', '.join(place.get('types', [])) or 'Không rõ')
    }

# Database connection and caching
@st.cache_resource
def get_mongo_client():
    """Get MongoDB client with caching."""
    return pymongo.MongoClient(MONGO_URI)

@st.cache_data
def get_city_list():
    """Get list of cities from database."""
    client = get_mongo_client()
    db = client["smart_travel"]
    city_docs = list(db.world_cities.find({}, {"city_ascii": 1, "_id": 0}))
    return sorted(list({doc["city_ascii"] for doc in city_docs}))

@st.cache_data
def get_city_coordinates(city: str) -> Optional[str]:
    """Get city coordinates from database."""
    client = get_mongo_client()
    db = client["smart_travel"]
    city_doc = db.world_cities.find_one({"city_ascii": city.strip()})
    return f"{city_doc['lat']},{city_doc['lng']}" if city_doc else None

# Session state management
def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        'request_count': 0,
        'restaurants': [],
        'selected_restaurant': None,
        'details_cache': {},
        'photo_cache': {},
        'timezone_cache': {},
        'directions_cache': {},
        'last_search_params': None,
        'user_location': None
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def increment_request_count():
    """Increment and display request count."""
    st.session_state.request_count += 1
    st.sidebar.info(f"Đã sử dụng khoảng {st.session_state.request_count}/100 request hôm nay.")

# Data fetching functions
def search_from_database(selected_type: str, city: str) -> List[Dict]:
    """Search places from database first."""
    client = get_mongo_client()
    db = client["smart_travel"]
    places_col = db["places"]
    
    db_query = {
        "search_type": selected_type,
        "formatted_address": {"$regex": city, "$options": "i"}
    }
    return list(places_col.find(db_query))

def search_from_api(selected_type: str, city: str, radius: int, user_location: Optional[str]) -> Dict:
    """Search places from Google Places API."""
    return get_text_search(
        query=f"{selected_type.replace('_', ' ')} in {city}",
        radius=radius,
        opennow=True,
        location=user_location,
        language="vi",
        region="vi",
        type=selected_type
    )

def save_places_to_db(places: List[Dict], selected_type: str):
    """Save places to database."""
    client = get_mongo_client()
    db = client["smart_travel"]
    places_col = db["places"]
    
    for place in places:
        try:
            place_doc = dict(place)
            place_doc["search_type"] = selected_type
            places_col.update_one(
                {"place_id": place_doc.get("place_id")},
                {"$set": place_doc},
                upsert=True
            )
        except Exception as e:
            st.error(f"MongoDB error: {e}")

def log_user_action(action: str, **kwargs):
    """Log user action to database."""
    client = get_mongo_client()
    db = client["smart_travel"]
    user_log_col = db["user_log"]
    
    log_data = {
        "action": action,
        "timestamp": datetime.now(timezone.utc),
        **kwargs
    }
    user_log_col.insert_one(log_data)

# Place details functions
def get_cached_or_fetch_details(place_id: str) -> Optional[Dict]:
    """Get place details from cache, database, or API."""
    client = get_mongo_client()
    db = client["smart_travel"]
    place_details_col = db["place_details"]
    
    # Check database first
    db_detail = place_details_col.find_one({"place_id": place_id})
    if db_detail and "result" in db_detail:
        st.session_state.details_cache[place_id] = {"result": db_detail["result"]}
        return db_detail["result"]
    
    # Check session cache
    if place_id in st.session_state.details_cache:
        return st.session_state.details_cache[place_id].get("result")
    
    # Fetch from API
    with st.spinner("Đang lấy chi tiết địa điểm..."):
        api_details = get_place_details(
            place_id=place_id,
            region="vi",
            language="vi",
            reviews_sort="most_relevant"
        )
        increment_request_count()
        
        if "error" in api_details:
            st.error(f"Lỗi khi lấy chi tiết: {api_details['error']}")
            return None
        elif api_details.get("status") != "OK":
            st.error(f"Không lấy được chi tiết: {api_details.get('status')}")
            return None
        
        # Save to database and cache
        place_details_col.update_one(
            {"place_id": place_id},
            {"$set": api_details},
            upsert=True
        )
        st.session_state.details_cache[place_id] = api_details
        return api_details.get("result")

def get_cached_or_fetch_photo(place_id: str, photo_ref: str) -> Optional[Dict]:
    """Get place photo from cache, database, or API."""
    client = get_mongo_client()
    db = client["smart_travel"]
    place_details_col = db["place_details"]
    
    # Check database first
    db_photo = place_details_col.find_one(
        {"place_id": place_id, "photo.photo_reference": photo_ref}, 
        {"photo": 1}
    )
    if db_photo and db_photo.get("photo"):
        st.session_state.photo_cache[photo_ref] = db_photo["photo"]
        return db_photo["photo"]
    
    # Check session cache
    if photo_ref in st.session_state.photo_cache:
        return st.session_state.photo_cache[photo_ref]
    
    # Fetch from API
    with st.spinner("Đang lấy ảnh địa điểm..."):
        photo_result = get_place_photo(
            photo_reference=photo_ref,
            maxheight=300,
            maxwidth=300
        )
        increment_request_count()
        
        if "error" in photo_result:
            st.error(f"Lỗi khi lấy ảnh: {photo_result['error']}")
            return None
        
        # Save to database and cache
        st.session_state.photo_cache[photo_ref] = photo_result
        place_details_col.update_one(
            {"place_id": place_id},
            {"$set": {"photo": photo_result}},
            upsert=True
        )
        return photo_result

def get_cached_or_fetch_timezone(place_id: str, latlng: str) -> Optional[Dict]:
    """Get timezone info from cache, database, or API."""
    client = get_mongo_client()
    db = client["smart_travel"]
    place_details_col = db["place_details"]
    
    # Check database first
    db_tz = place_details_col.find_one(
        {"place_id": place_id, "timezone.location": latlng}, 
        {"timezone": 1}
    )
    if db_tz and db_tz.get("timezone"):
        st.session_state.timezone_cache[latlng] = db_tz["timezone"]
        return db_tz["timezone"]
    
    # Check session cache
    if latlng in st.session_state.timezone_cache:
        return st.session_state.timezone_cache[latlng]
    
    # Fetch from API
    with st.spinner("Đang lấy thông tin múi giờ..."):
        timezone_result = get_timezone(
            location=latlng,
            timestamp=int(datetime.now(timezone.utc).timestamp()),
            language="vi"
        )
        increment_request_count()
        
        if "error" in timezone_result:
            st.error(f"Lỗi khi lấy múi giờ: {timezone_result['error']}")
            return None
        
        # Save to database and cache
        st.session_state.timezone_cache[latlng] = timezone_result
        place_details_col.update_one(
            {"place_id": place_id},
            {"$set": {"timezone": timezone_result}},
            upsert=True
        )
        return timezone_result

def get_cached_or_fetch_directions(place_id: str, user_location: str) -> Optional[Dict]:
    """Get directions from cache, database, or API."""
    client = get_mongo_client()
    db = client["smart_travel"]
    place_details_col = db["place_details"]
    direction_key = f"{user_location}_{place_id}"
    
    # Check database first
    db_dir = place_details_col.find_one(
        {"place_id": place_id, "directions.origin": user_location}, 
        {"directions": 1}
    )
    if db_dir and db_dir.get("directions"):
        directions = db_dir["directions"]
        st.session_state.directions_cache[direction_key] = directions
        return directions.get("data", directions)
    
    # Check session cache
    if direction_key in st.session_state.directions_cache:
        cached = st.session_state.directions_cache[direction_key]
        return cached.get("data", cached) if isinstance(cached, dict) else cached
    
    # Fetch from API
    with st.spinner("Đang lập kế hoạch di chuyển..."):
        directions = get_directions(
            origin=user_location,
            destination=place_id,
            mode="driving",
            departure_time="now",
            traffic_model="best_guess",
            language="vi",
            units="metric"
        )
        increment_request_count()
        
        if "error" in directions:
            st.error(f"Lỗi khi lấy chỉ đường: {directions['error']}")
            return None
        
        # Save to database and cache
        st.session_state.directions_cache[direction_key] = directions
        place_details_col.update_one(
            {"place_id": place_id},
            {"$set": {"directions": {"origin": user_location, "data": directions}}},
            upsert=True
        )
        return directions

# UI Rendering functions
def render_place_card(place: Dict[str, Any], idx: int) -> str:
    """Render place information card."""
    info = extract_place_info(place)
    
    return f"""
    <div style='border:2px solid #e0e0e0; border-radius:12px; padding:16px; margin-bottom:16px; background-color:#fafbfc;'>
        <h4 style='margin-bottom:8px'>{info['name']}</h4>
        <div><b>Địa chỉ:</b> {info['address']}</div>
        <div><b>Trạng thái:</b> {info['business_status']} - {info['open_status']}</div>
        <div><b>Loại dịch vụ:</b> {info['search_type']}</div>
        <div><b>Đánh giá:</b> {info['rating']}</div>
    </div>
    """

def render_place_details(result: Dict[str, Any]):
    """Render detailed place information."""
    st.markdown(f"#### Thông tin chi tiết: {result.get('name')}")
    details = [
        ("Địa chỉ", result.get('formatted_address', 'Không có')),
        ("Số điện thoại", result.get('formatted_phone_number', 'Không có')),
        ("Website", result.get('website', 'Không có')),
        ("Đánh giá", f"{result.get('rating', 'Chưa có')}/5")
    ]
    
    for label, value in details:
        st.write(f"**{label}:** {value}")

def render_reviews(reviews: List[Dict]):
    """Render place reviews."""
    if not reviews:
        return
    
    st.subheader("Đánh giá nổi bật")
    for review in reviews[:3]:
        author = review.get('author_name', 'Ẩn danh')
        rating = review.get('rating', 'N/A')
        text = review.get("text", "Không có nội dung")
        
        with st.expander(f"{author} ({rating}/5)"):
            st.write(text)

def render_directions_info(directions: Dict):
    """Render directions information."""
    routes = directions.get("routes", [])
    if not routes:
        st.warning("Không tìm thấy tuyến đường.")
        return
    
    route = routes[0]
    leg = route.get("legs", [{}])[0]
    
    distance = leg.get("distance", {}).get("text", "N/A")
    duration = leg.get("duration", {}).get("text", "N/A")
    
    st.subheader("Kế hoạch di chuyển")
    st.write(f"**Khoảng cách**: {distance}")
    st.write(f"**Thời gian di chuyển**: {duration}")
    
    steps = leg.get("steps", [])
    if steps:
        with st.expander("Hướng dẫn chi tiết"):
            for step in steps:
                instruction = step.get("html_instructions", "Không có hướng dẫn")
                st.write(f"- {instruction}")

# Main search logic
def perform_search(city: str, user_location: Optional[str], radius: int, selected_type: str) -> Tuple[bool, List[Dict]]:
    """Perform place search with database-first approach."""
    # Try database first
    db_results = search_from_database(selected_type, city)
    if db_results:
        log_user_action(
            "search", 
            city=city, 
            type=selected_type, 
            user_location=user_location, 
            radius=radius, 
            result_count=len(db_results)
        )
        st.info(f"Đã lấy {len(db_results)} kết quả từ cơ sở dữ liệu.")
        return True, db_results
    
    # Fallback to API
    search_result = search_from_api(selected_type, city, radius, user_location)
    increment_request_count()
    
    if "error" in search_result:
        st.error(f"Lỗi khi tìm kiếm {format_type_display(selected_type)}: {search_result['error']}")
        return False, []
    elif search_result.get("status") != "OK":
        st.error(f"Không tìm thấy kết quả: {search_result.get('status')}")
        return False, []
    
    places = search_result.get("results", [])
    if not places:
        st.warning(f"Không tìm thấy {format_type_display(selected_type)} nào.")
        return False, []
    
    # Save to database
    save_places_to_db(places, selected_type)
    log_user_action(
        "search", 
        city=city, 
        type=selected_type, 
        user_location=user_location, 
        radius=radius, 
        result_count=len(places)
    )
    st.success(f"Đã tìm thấy {len(places)} {format_type_display(selected_type)}!")
    return True, places

# Main app
def main():
    """Main application function."""
    # Page config
    st.set_page_config(page_title="Tìm Địa Điểm Tối Ưu", page_icon="🗺️")
    st.title("🗺️ Tìm Địa Điểm và Lập Kế Hoạch Di Chuyển")
    st.markdown("Tìm địa điểm và lập kế hoạch di chuyển với số lượng request API tối thiểu!")
    
    # Initialize session state
    init_session_state()
    
    # Sidebar configuration
    st.sidebar.header("Thông tin tìm kiếm")
    
    # Get city list and user selection
    city_list = get_city_list()
    default_city = "Hanoi" if "Hanoi" in city_list else (city_list[0] if city_list else "")
    
    city = st.sidebar.selectbox(
        "Chọn thành phố", 
        options=city_list, 
        index=city_list.index(default_city) if default_city in city_list else 0
    )
    
    radius = st.sidebar.slider("Bán kính tìm kiếm (mét)", 500, 5000, 2000)
    
    # Handle user location
    user_location = st.session_state.get('user_location')
    if not user_location:
        user_location = get_city_coordinates(city)
        if user_location:
            st.session_state['user_location'] = user_location
            st.info(f"Sử dụng vị trí trung tâm thành phố: {city}")
    
    # Place type selection
    selected_type = st.sidebar.selectbox(
        "Chọn loại địa điểm",
        options=PLACE_TYPES,
        index=PLACE_TYPES.index("tourist_attraction") if "tourist_attraction" in PLACE_TYPES else 0,
        help="Chọn loại địa điểm để tìm kiếm (ví dụ: tourist_attraction, restaurant, cafe, bank, ...)."
    )
    
    # Search buttons
    col1, col2 = st.sidebar.columns(2)
    search_clicked = col1.button("Tìm địa điểm")
    refresh_clicked = col2.button("Làm mới")
    
    # Perform search
    if search_clicked or refresh_clicked:
        current_search_params = (city, user_location, radius, selected_type)
        if refresh_clicked or st.session_state.last_search_params != current_search_params:
            with st.spinner(f"Đang tìm kiếm {format_type_display(selected_type)}..."):
                success, results = perform_search(city, user_location, radius, selected_type)
                if success:
                    st.session_state.restaurants = results
                    st.session_state.last_search_params = current_search_params
    
    # Display results
    if st.session_state.restaurants:
        st.header(f"Danh sách {format_type_display(selected_type)}")
        
        for idx, place in enumerate(st.session_state.restaurants):
            with st.container():
                # Render place card
                st.markdown(render_place_card(place, idx), unsafe_allow_html=True)
                
                # Detail button
                if st.button(f"Xem chi tiết #{idx+1}", key=f"detail_{place['place_id']}"):
                    place_id = place["place_id"]
                    
                    # Get place details
                    result = get_cached_or_fetch_details(place_id)
                    if not result:
                        continue
                    
                    log_user_action(
                        "view_detail", 
                        place_id=place_id, 
                        city=city, 
                        type=selected_type, 
                        user_location=user_location
                    )
                    
                    # Render details
                    render_place_details(result)
                    
                    # Handle photos
                    photos = result.get("photos", [])
                    if photos:
                        photo_ref = photos[0].get("photo_reference")
                        if photo_ref:
                            photo_data = get_cached_or_fetch_photo(place_id, photo_ref)
                            if photo_data and photo_data.get("url"):
                                st.image(photo_data.get("url"), caption="Ảnh địa điểm", use_container_width=True)
                    
                    # Render reviews
                    render_reviews(result.get("reviews", []))        
                
    else:
        st.info("Vui lòng nhập thông tin và nhấn 'Tìm địa điểm' để bắt đầu.")
    
    # Footer
    st.markdown("---")
    st.markdown("Được xây dựng với Streamlit và Google Maps API. Tối ưu hóa để giảm số lượng request API.")

if __name__ == "__main__":
    main()