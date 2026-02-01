import os
import pymongo
from datetime import datetime, timezone
import random
import uuid
from faker import Faker
import hashlib

# MongoDB setup
MONGO_URI = "mongodb+srv://nguyenanhilu9785_db_user:12345@cluster0.olqzq.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
mongo_client = pymongo.MongoClient(MONGO_URI)
db = mongo_client["smart_travel"]
places_collection = db["places"]
tours_collection = db["tours"]
users_collection = db["users"]

fake = Faker()

def update_existing_users():
    """Update existing users to hash their passwords."""
    users = list(users_collection.find({}))
    for user in users:
        if 'password' in user and not user['password'].startswith('$2y$'):  # Assuming bcrypt starts with $2y$
            # If not hashed, hash it
            plain_password = user['password']
            hashed_password = hashlib.sha256(plain_password.encode()).hexdigest()
            users_collection.update_one(
                {"_id": user["_id"]},
                {"$set": {"password": hashed_password}}
            )
    print("Updated existing users' passwords.")

def generate_users(num_users=100):
    """Generate and insert fake users."""
    users = []
    for i in range(num_users):
        plain_password = fake.password()
        hashed_password = hashlib.sha256(plain_password.encode()).hexdigest()
        # Generate unique email by adding a random suffix
        base_email = fake.email()
        unique_email = f"{base_email.split('@')[0]}_{i}_{random.randint(1000, 9999)}@{base_email.split('@')[1]}"
        user = {
            "name": fake.name(),
            "email": unique_email,
            "password": hashed_password,
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc)
        }
        users.append(user)
    
    try:
        result = users_collection.insert_many(users)
        print(f"Inserted {len(result.inserted_ids)} users successfully.")
        return result.inserted_ids
    except Exception as e:
        print(f"Error inserting users: {e}")
        return []

def generate_tour_for_city(city, places):
    """Generate a sample tour for a given city using available places."""
    # Filter places for the city
    city_places = [p for p in places if p.get("city") == city]
    if len(city_places) < 1:
        return None  # No places for the city

    # Select random places for the tour (up to 12 for 3 days, or all if less)
    num_places = min(12, len(city_places))
    selected_places = random.sample(city_places, num_places)

    # Adjust itinerary based on number of places
    days = min(3, (len(selected_places) + 3) // 4)  # At least 1 day, up to 3
    places_per_day = len(selected_places) // days
    extra = len(selected_places) % days

    itinerary = []
    place_index = 0
    for day in range(1, days + 1):
        day_places_count = places_per_day + (1 if day <= extra else 0)
        day_places = []
        for i in range(day_places_count):
            if place_index < len(selected_places):
                place = selected_places[place_index]
                day_places.append({
                    "place_id": place.get("id"),
                    "name": place.get("displayName", {}).get("text", place.get("name", "")),
                    "start_time": f"{9 + i*3:02d}:00",  # 9:00, 12:00, 15:00, 18:00
                    "duration_hours": random.uniform(1.5, 3.0),
                    "types": place.get("types", []),
                    "rating": place.get("rating", 0.0),
                    "tips": ["Highly rated by visitors", "Very popular destination"]
                })
                place_index += 1

        itinerary.append({
            "day_number": day,
            "date": f"2025-10-{27 + day - 1:02d}",
            "theme": random.choice(["Food & Entertainment", "Nature & Relaxation", "Cultural Experience", "Historical Sites"]),
            "places": day_places,
            "meals": [
                {"type": "breakfast", "time": "08:00", "estimated_cost": 15},
                {"type": "lunch", "time": "13:00", "estimated_cost": 25},
                {"type": "dinner", "time": "19:00", "estimated_cost": 35}
            ],
            "transportation": [
                {"type": "walking", "distance_km": random.uniform(5, 15)},
                {"type": "subway", "cost_per_day": 5}
            ],
            "total_distance_km": random.uniform(5, 15),
            "estimated_cost": 80
        })

    # Generate tour data
    tour = {
        "tour_id": f"{city.lower().replace(' ', '_')}_tour_{str(uuid.uuid4())[:8]}",
        "title": f"3-Day {city} Adventure",
        "description": f"Discover the magic of {city} in 3 unforgettable days! This curated tour features amazing places perfect for exploration.",
        "destination": city,
        "duration_days": 3,
        "participants": [],  # List of user IDs
        "user_preferences": {
            "destination_city": city,
            "trip_duration_days": 3,
            "budget_range": "medium",
            "interests": ["landmarks", "parks"],
            "travel_party": "solo",
            "accommodation_type": "hotel",
            "dietary_restrictions": [],
            "accessibility_needs": []
        },
        "description": f"Discover the magic of {city} in 3 unforgettable days! This curated tour features amazing places perfect for exploration.",
        "destination": city,
        "duration_days": 3,
        "user_preferences": {
            "destination_city": city,
            "trip_duration_days": 3,
            "budget_range": "medium",
            "interests": ["landmarks", "parks"],
            "travel_party": "solo",
            "accommodation_type": "hotel",
            "dietary_restrictions": [],
            "accessibility_needs": []
        },
        "itinerary": itinerary,
        "flights": {
            "outbound": {
                "departure_city": "Chicago (ORD)",
                "arrival_city": f"{city} Airport",
                "airline": "Spirit",
                "flight_number": f"AS{random.randint(100, 999)}",
                "departure_time": "08:00",
                "arrival_time": "13:48",
                "duration_hours": 5.5,
                "price_usd": 554,
                "class_type": "economy"
            },
            "return": {
                "departure_city": f"{city} Airport",
                "arrival_city": "Chicago (ORD)",
                "airline": "United",
                "flight_number": f"AS{random.randint(100, 999)}",
                "departure_time": "18:00",
                "arrival_time": "23:46",
                "duration_hours": 5.5,
                "price_usd": 554,
                "class_type": "economy"
            }
        },
        "pricing": {
            "flights": 1108,
            "accommodation": 600,
            "activities": 240,
            "meals": 225,
            "transportation": 150,
            "insurance": 50,
            "misc": 100,
            "total_usd": 2473,
            "currency": "USD",
            "price_per_person": 2473
        },
        "tags": [city.lower().replace(" ", "-"), "medium", "solo", "3-days", "landmarks", "parks"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "user_feedback": None
    }

    return tour

def main():
    # Update existing users' passwords
    update_existing_users()

    # Generate and insert 100 users
    user_ids = generate_users(100)
    if not user_ids:
        print("Failed to generate users, exiting.")
        return

    # Get unique cities using aggregation (more efficient)
    pipeline = [
        {"$match": {"city": {"$exists": True, "$ne": None}}},
        {"$group": {"_id": "$city"}},
        {"$project": {"_id": 0, "city": "$_id"}}
    ]
    cities_result = list(places_collection.aggregate(pipeline))
    cities = [doc["city"] for doc in cities_result]
    print(f"Unique cities: {cities}")

    # Load all places into memory for efficiency
    all_places = list(places_collection.find({}))
    places_by_city = {}
    for place in all_places:
        city = place.get("city")
        if city:
            if city not in places_by_city:
                places_by_city[city] = []
            places_by_city[city].append(place)

    # For each city, generate multiple tours
    tour_ids = []
    for city in cities:
        city_places = places_by_city.get(city, [])
        if len(city_places) < 1:
            print(f"No places for {city}, skipping.")
            continue

        # Generate 5-10 tours per city to reach at least 500 tours
        num_tours_per_city = random.randint(5, 10)
        for _ in range(num_tours_per_city):
            tour = generate_tour_for_city(city, city_places)
            if tour:
                try:
                    result = tours_collection.insert_one(tour)
                    tour_ids.append(result.inserted_id)
                    print(f"Tour for {city} inserted successfully.")
                except Exception as e:
                    print(f"Error inserting tour for {city}: {e}")
            else:
                print(f"Failed to generate tour for {city}.")

    # Assign tours to users (each user 1-20 tours, tours can overlap)
    all_users = list(users_collection.find({}, {"_id": 1}))
    user_ids = [u["_id"] for u in all_users]
    for user_id in user_ids:
        num_tours = random.randint(1, 20)
        selected_tours = random.sample(tour_ids, min(num_tours, len(tour_ids)))
        for tour_id in selected_tours:
            try:
                tours_collection.update_one(
                    {"_id": tour_id},
                    {"$addToSet": {"participants": user_id}}  # Add user to participants if not already
                )
            except Exception as e:
                print(f"Error updating tour {tour_id} with user {user_id}: {e}")

    print("All tours assigned to users successfully.")

if __name__ == "__main__":
    main()