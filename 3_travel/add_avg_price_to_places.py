import pymongo
import random

# MongoDB setup
MONGO_URI = "mongodb+srv://nguyenanhilu9785_db_user:12345@cluster0.olqzq.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = pymongo.MongoClient(MONGO_URI)
db = client["smart_travel"]
places = db["places"]

# Mapping priceLevel to avg_price
PRICE_MAP = {
    "PRICE_LEVEL_FREE": 0,
    "PRICE_LEVEL_INEXPENSIVE": random.randint(5, 15),
    "PRICE_LEVEL_MODERATE": random.randint(15, 40),
    "PRICE_LEVEL_EXPENSIVE": random.randint(40, 100),
    "PRICE_LEVEL_VERY_EXPENSIVE": random.randint(100, 500)
}

def estimate_avg_price(types, city):
    # City tiers (big cities have higher prices)
    expensive_cities = {"Paris", "London", "New York", "Tokyo", "Singapore", "Zurich", "Dubai", "Hong Kong"}
    mid_cities = {"Hanoi", "Bangkok", "Ho Chi Minh City", "Istanbul", "Kuala Lumpur", "Lisbon", "Berlin", "Toronto"}
    city = (city or "").strip()
    city_tier = "expensive" if city in expensive_cities else ("mid" if city in mid_cities else "cheap")

    # Type-based price ranges (USD)
    if "restaurant" in types:
        if city_tier == "expensive":
            return random.randint(25, 60)
        elif city_tier == "mid":
            return random.randint(10, 30)
        else:
            return random.randint(5, 15)
    elif "hotel" in types:
        if city_tier == "expensive":
            return random.randint(120, 300)
        elif city_tier == "mid":
            return random.randint(50, 120)
        else:
            return random.randint(20, 60)
    elif "attraction" in types:
        if city_tier == "expensive":
            return random.randint(20, 50)
        elif city_tier == "mid":
            return random.randint(5, 20)
        else:
            return random.randint(0, 10)
    else:
        # Default for other types
        if city_tier == "expensive":
            return random.randint(10, 40)
        elif city_tier == "mid":
            return random.randint(5, 20)
        else:
            return random.randint(0, 10)

def main():
    count_update = 0
    count_insert = 0
    for doc in places.find({}):
        price_level = doc.get("priceLevel")
        # Luôn update lại avg_price theo priceLevel hoặc estimate
        if price_level in PRICE_MAP:
            avg_price_new = PRICE_MAP[price_level]
        else:
            types = doc.get("types", [])
            city = doc.get("city", "")
            avg_price_new = estimate_avg_price(types, city)
        result = places.update_one({"_id": doc["_id"]}, {"$set": {"avg_price": avg_price_new}})
        if result.modified_count:
            count_update += 1
        
        print(f"Processed place_id: {doc.get('place_id')}, priceLevel: {price_level}, avg_price set to: {avg_price_new}")
    print(f"Updated avg_price for {count_update} places.")

if __name__ == "__main__":
    main()
