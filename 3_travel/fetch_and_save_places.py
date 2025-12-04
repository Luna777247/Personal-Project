import os
import requests
import pymongo
from datetime import datetime, timezone
import time
import csv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# MongoDB setup
MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://nguyenanhilu9785_db_user:12345@cluster0.olqzq.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
mongo_client = pymongo.MongoClient(MONGO_URI)
db = mongo_client["smart_travel"]
places_collection = db["places"]
metadata_collection = db["place_metadata"]
worldcities_collection = db["worldcities"]

# API setup
RAPIDAPI_KEY = "cf1b379a98msh116f2d78aa3d55ep1a4602jsndbd91f1a8bb4"  # Replace with your key if needed
url = "https://google-map-places-new-v2.p.rapidapi.com/v1/places:searchText"

headers = {
    "x-rapidapi-key": RAPIDAPI_KEY,
    "x-rapidapi-host": "google-map-places-new-v2.p.rapidapi.com",
    "Content-Type": "application/json",
    "X-Goog-FieldMask": "*"
}

def check_connections():
    """Check MongoDB and API connections."""
    logging.info("Checking connections...")
    
    # # Check MongoDB
    # try:
    #     mongo_client.admin.command('ping')
    #     logging.info("MongoDB connection successful.")
    #     # Drop old index if exists
    #     try:
    #         db["places"].drop_index("place_id_1")
    #         logging.info("Dropped old index place_id_1")
    #     except Exception as e:
    #         logging.info(f"Index place_id_1 not found or already dropped: {e}")
    # except Exception as e:
    #     logging.error(f"MongoDB connection failed: {e}")
    #     return False
    
    # # Check API
    # try:
    #     test_payload = {
    #         "textQuery": "hotel in Bangkok",
    #         "maxResultCount": 1
    #     }
    #     response = requests.post(url, json=test_payload, headers=headers, timeout=10)
    #     if response.status_code == 200:
    #         logging.info("API connection successful.")
    #     else:
    #         logging.warning(f"API returned status code: {response.status_code}")
    # except Exception as e:
    #     logging.error(f"API connection failed: {e}")
    #     return False
    
    return True

# Check connections before proceeding
if not check_connections():
    logging.error("Connection checks failed. Exiting.")
    exit(1)

# Data
types = ["hotel", "restaurant", "attraction"]
cities = [
    # "Bangkok",
    # "Tokyo",
    # "Seoul",
    # "Hong Kong",
    # "Singapore",
    # "Kuala Lumpur",
    # "Bali",
    # "Dubai",
    # "Istanbul",
    # "Macau",
    # "Phuket",
    # "Paris",
    # "London",
    # "Rome",
    # "Barcelona",
    # "Amsterdam",
    # "Prague",
    # "Vienna",
    # "Berlin",
    # "Madrid",
    # "New York",
    # "Los Angeles",
    # "Las Vegas",
    # "Orlando",
    # "Cancun",
    # "Mexico City",
    # "Rio de Janeiro",
    # "Cape Town",
    # "Marrakech",
    # "Cairo",
    # "Sydney",
    # "Melbourne",
    # "Auckland",
    # "San Francisco",
    # "Miami",
    # "Chicago",
    # "Toronto",
    # "Vancouver",
    # "Buenos Aires",
    # "Lima",
    # "Havana",
    # "Lisbon",
    # "Venice",
    # "Athens",
    # "Budapest",
    # "Dubrovnik",
    # "Zurich",
    # "Edinburgh",
    # "Dublin",
    # "Doha",
    # "Abu Dhabi",
    # "Beijing",
    # "Shanghai",
    # "Taipei",
    # "Kyoto",
    # "Osaka",
    # "Hanoi",
    # "Da Nang",
    # "Ho Chi Minh City",
    # "San Diego",
    # "Seattle",
    # "Boston",
    # "Washington",
    # "Philadelphia",
    # "Montreal",
    # "Quebec City",
    "Sao Paulo",
    "Brasilia",
    "Bogota",
    "Cartagena",
    "Santiago",
    "Cusco",
    "Panama City",
    "San Jose",
    "Manila",
    "Chiang Mai",
    "Siem Reap",
    "Kolkata",
    "Mumbai",
    "New Delhi",
    "Kathmandu",
    "Doha",
    "Muscat",
    "Tel Aviv",
    "Jerusalem",
    "Johannesburg",
    "Nairobi",
    "Casablanca",
    "Hobart",
    "Brisbane",
    "Perth"
]


# Load city coordinates from worldcities.csv
city_coords = {}
csv_path = "worldcities.csv"
try:
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            city_name = row['city_ascii']  # Use city_ascii for matching
            if city_name in cities:  # Only load and upsert cities in the list
                lat = float(row['lat'])
                lng = float(row['lng'])
                city_coords[city_name] = {"latitude": lat, "longitude": lng}
                # Upsert to worldcities collection
                worldcities_collection.update_one(
                    {"city_ascii": city_name},
                    {"$set": row},
                    upsert=True
                )
    logging.info(f"Loaded {len(city_coords)} cities from {csv_path} and upserted to worldcities collection.")
except Exception as e:
    logging.error(f"Error loading cities from CSV: {e}")
    # Fallback to default
    city_coords = {"default": {"latitude": 40.0, "longitude": -110.0}}

# Default place schema to ensure all fields are present
DEFAULT_PLACE_SCHEMA = {
    "name": "",
    "id": "",
    "displayName": {"text": "", "languageCode": ""},
    "types": [],
    "primaryType": "",
    "primaryTypeDisplayName": {"text": "", "languageCode": ""},
    "nationalPhoneNumber": "",
    "internationalPhoneNumber": "",
    "formattedAddress": "",
    "shortFormattedAddress": "",
    "addressComponents": [{"longText": "", "shortText": "", "types": [], "languageCode": ""}],
    "plusCode": {"globalCode": "", "compoundCode": ""},
    "location": {"latitude": 0.0, "longitude": 0.0},
    "viewport": {"low": {"latitude": 0.0, "longitude": 0.0}, "high": {"latitude": 0.0, "longitude": 0.0}},
    "rating": 0.0,
    "googleMapsUri": "",
    "websiteUri": "",
    "reviews": [{"name": "", "relativePublishTimeDescription": "", "rating": 0, "publishTime": ""}],
    "regularOpeningHours": {
        "openNow": False,
        "periods": [{"open": {"day": 0, "hour": 0, "minute": 0, "date": "", "truncated": False}, "close": {"day": 0, "hour": 0, "minute": 0, "date": "", "truncated": False}}],
        "weekdayDescriptions": [],
        "secondaryHoursType": 0,
        "specialDays": [{"date": ""}]
    },
    "utcOffsetMinutes": 0,
    "photos": [{"name": "", "widthPx": 0, "heightPx": 0, "authorAttributions": [{"displayName": "", "uri": "", "photoUri": ""}]}],
    "adrFormatAddress": "",
    "businessStatus": 0,
    "priceLevel": 0,
    "attributions": [{"provider": "", "providerUri": ""}],
    "userRatingCount": 0,
    "iconMaskBaseUri": "",
    "iconBackgroundColor": "",
    "takeout": False,
    "delivery": False,
    "dineIn": False,
    "curbsidePickup": False,
    "reservable": False,
    "servesBreakfast": False,
    "servesLunch": False,
    "servesDinner": False,
    "servesBeer": False,
    "servesWine": False,
    "servesBrunch": False,
    "servesVegetarianFood": False,
    "currentOpeningHours": {
        "openNow": False,
        "periods": [{"open": {"day": 0, "hour": 0, "minute": 0, "date": "", "truncated": False}, "close": {"day": 0, "hour": 0, "minute": 0, "date": "", "truncated": False}}],
        "weekdayDescriptions": [],
        "secondaryHoursType": 0,
        "specialDays": [{"date": ""}]
    },
    "currentSecondaryOpeningHours": [{
        "openNow": False,
        "periods": [{"open": {"day": 0, "hour": 0, "minute": 0, "date": "", "truncated": False}, "close": {"day": 0, "hour": 0, "minute": 0, "date": "", "truncated": False}}],
        "weekdayDescriptions": [],
        "secondaryHoursType": 0,
        "specialDays": [{"date": ""}]
    }],
    "regularSecondaryOpeningHours": [{
        "openNow": False,
        "periods": [{"open": {"day": 0, "hour": 0, "minute": 0, "date": "", "truncated": False}, "close": {"day": 0, "hour": 0, "minute": 0, "date": "", "truncated": False}}],
        "weekdayDescriptions": [],
        "secondaryHoursType": 0,
        "specialDays": [{"date": ""}]
    }],
    "editorialSummary": {"text": "", "languageCode": ""},
    "outdoorSeating": False,
    "liveMusic": False,
    "menuForChildren": False,
    "servesCocktails": False,
    "servesDessert": False,
    "servesCoffee": False,
    "goodForChildren": False,
    "allowsDogs": False,
    "restroom": False,
    "goodForGroups": False,
    "goodForWatchingSports": False,
    "paymentOptions": {
        "acceptsCreditCards": False,
        "acceptsDebitCards": False,
        "acceptsCashOnly": False,
        "acceptsNfc": False
    },
    "parkingOptions": {
        "freeParkingLot": False,
        "paidParkingLot": False,
        "freeStreetParking": False,
        "paidStreetParking": False,
        "valetParking": False,
        "freeGarageParking": False,
        "paidGarageParking": False
    },
    "subDestinations": [{"name": "", "id": ""}],
    "accessibilityOptions": {
        "wheelchairAccessibleParking": False,
        "wheelchairAccessibleEntrance": False,
        "wheelchairAccessibleRestroom": False,
        "wheelchairAccessibleSeating": False
    },
    "fuelOptions": {"fuelPrices": [{"type": 0, "price": {"currencyCode": "", "units": "", "nanos": 0}, "updateTime": ""}]},
    "evChargeOptions": {
        "connectorCount": 0,
        "connectorAggregation": [{"type": 0, "maxChargeRateKw": 0.0, "count": 0, "availableCount": 0, "outOfServiceCount": 0, "availabilityLastUpdateTime": ""}]
    },
    "generativeSummary": {
        "overview": {"text": "", "languageCode": ""},
        "description": {"text": "", "languageCode": ""},
        "references": {"reviews": [], "places": []}
    },
    "areaSummary": {"contentBlocks": [{"topic": "", "content": {"text": "", "languageCode": ""}, "references": {"reviews": [], "places": []}}]}
}

def merge_dicts(default, update):
    """Recursively merge update into default, keeping defaults for missing keys."""
    for key, value in update.items():
        if isinstance(value, dict) and key in default and isinstance(default[key], dict):
            merge_dicts(default[key], value)
        else:
            default[key] = value
    return default

def normalize_place(place):
    """Ensure place has all fields from DEFAULT_PLACE_SCHEMA."""
    normalized = DEFAULT_PLACE_SCHEMA.copy()
    return merge_dicts(normalized, place)


def fetch_and_save_places():
    logging.info("Starting fetch_and_save_places function")
    total_saved = 0
    for city in cities:
        logging.info(f"Processing city: {city}")
        for place_type in types:
            text_query = f"{place_type} in {city}"
            logging.info(f"Fetching: {text_query}")

            payload = {
                "textQuery": text_query,
                "languageCode": "",
                "regionCode": "",
                "rankPreference": 0,
                "includedType": "",
                "openNow": True,
                "minRating": 0,
                "maxResultCount": 50,
                "priceLevels": [],
                "strictTypeFiltering": True,
                "locationBias": {
                    "circle": {
                        "center": city_coords.get(city, {"latitude": 40.0, "longitude": -110.0}),
                        "radius": 10000
                    }
                },
                "evOptions": {
                    "minimumChargingRateKw": 0,
                    "connectorTypes": []
                }
            }

            try:
                response = requests.post(url, json=payload, headers=headers)
                data = response.json()

                if "places" in data:
                    places = data["places"]
                    logging.info(f"Found {len(places)} places for {text_query}")

                    for place in places:
                        # Normalize place to ensure all fields are present
                        normalized_place = normalize_place(place)
                        
                        # Add city and city_id from worldcities collection
                        city_doc = worldcities_collection.find_one({"city_ascii": city})
                        if city_doc:
                            normalized_place["city"] = city
                            normalized_place["city_id"] = city_doc.get("id", "")
                        else:
                            normalized_place["city"] = city
                            normalized_place["city_id"] = ""
                        
                        # Lưu place data vào collection "places"
                        place_id = normalized_place.get("id")
                        if not place_id:
                            logging.warning(f"Skipping place without id for {text_query}")
                            continue
                        
                        places_collection.update_one({"id": place_id}, {"$set": normalized_place}, upsert=True)
                        
                        # Lưu metadata vào collection "place_metadata"
                        metadata_doc = {
                            "place_id": place_id,
                            "search_type": place_type,
                            "city": city,
                            "text_query": text_query,
                            "fetch_time": datetime.now(timezone.utc).isoformat()
                        }
                        metadata_collection.insert_one(metadata_doc)
                        
                        total_saved += 1
                        logging.info(f"Saved place: {place_id} for {text_query}")

                else:
                    logging.warning(f"No places found for {text_query}")

                # Rate limiting
                time.sleep(1)  # Adjust as needed

            except Exception as e:
                logging.error(f"Error fetching {text_query}: {e}")

    logging.info(f"Total places saved: {total_saved}")

if __name__ == "__main__":
    if check_connections():
        fetch_and_save_places()
    else:
        logging.error("Connection checks failed. Exiting.")