import os
import pymongo
from datetime import datetime, timezone

MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://nguyenanhilu9785_db_user:12345@cluster0.olqzq.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
mongo_client = pymongo.MongoClient(MONGO_URI)
db = mongo_client["smart_travel"]

def save_place_to_mongodb(place_data, search_type):
    collection = db["places"]
    doc = {
        **place_data,
        "search_type": search_type,
        "fetch_time": datetime.now(timezone.utc).isoformat()
    }
    collection.update_one({"place_id": place_data.get("place_id")}, {"$set": doc}, upsert=True)

def delete_all_places():
    """
    Xóa tất cả documents trong collection 'places'.
    Sử dụng delete_many({}) để xóa dữ liệu mà không xóa collection.
    """
    collection = db["places"]
    result = collection.delete_many({})  # Xóa tất cả documents
    print(f"Đã xóa {result.deleted_count} documents trong collection 'places'.")
    return result.deleted_count
