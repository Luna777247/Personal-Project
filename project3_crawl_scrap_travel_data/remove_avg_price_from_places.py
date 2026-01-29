import pymongo

# MongoDB setup
MONGO_URI = "mongodb+srv://nguyenanhilu9785_db_user:12345@cluster0.olqzq.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = pymongo.MongoClient(MONGO_URI)
db = client["smart_travel"]
places = db["places"]

def main():
    count = 0
    for doc in places.find({"price_avg": {"$exists": True}}):
        result = places.update_one({"_id": doc["_id"]}, {"$unset": {"price_avg": ""}})
        if result.modified_count:
            count += 1
    print(f"Removed price_avg from {count} places.")

if __name__ == "__main__":
    main()
