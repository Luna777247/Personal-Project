"""
Test script để kiểm tra search results flow
"""

import requests
import time
import json

# Configuration
BACKEND_URL = "http://localhost:8000"

def test_search_flow():
    """Test complete search flow"""
    print("🧪 Testing Search Results Flow...")
    
    # 1. Test backend health
    try:
        response = requests.get(f"{BACKEND_URL}/health")
        print(f"✅ Backend health: {response.status_code}")
        print(f"📊 Components: {response.json()}")
    except Exception as e:
        print(f"❌ Backend not running: {e}")
        return
    
    # 2. Start search
    search_data = {
        "query": "restaurants in Ha Noi",
        "location_type": "restaurant",
        "max_results": 10,
        "language": "en",
        "api_source": "google_places"
    }
    
    print(f"\n🔍 Starting search: {search_data['query']}")
    try:
        response = requests.post(f"{BACKEND_URL}/search/places", json=search_data)
        if response.status_code == 200:
            search_result = response.json()
            search_id = search_result['data']['search_id']
            print(f"✅ Search initiated: {search_id}")
        else:
            print(f"❌ Search failed: {response.status_code} - {response.text}")
            return
    except Exception as e:
        print(f"❌ Search request failed: {e}")
        return
    
    # 3. Poll for status
    print(f"\n⏳ Polling for search status...")
    max_attempts = 30
    attempt = 0
    
    while attempt < max_attempts:
        try:
            response = requests.get(f"{BACKEND_URL}/search/status/{search_id}")
            if response.status_code == 200:
                status_data = response.json()['data']
                status = status_data['status']
                print(f"📊 Status: {status}")
                
                if status == 'completed':
                    print(f"✅ Search completed!")
                    break
                elif 'failed' in status:
                    print(f"❌ Search failed: {status}")
                    return
                    
            else:
                print(f"⚠️ Status check failed: {response.status_code}")
                
        except Exception as e:
            print(f"⚠️ Status check error: {e}")
        
        attempt += 1
        time.sleep(2)
    
    if attempt >= max_attempts:
        print(f"⏰ Timeout waiting for search completion")
        return
    
    # 4. Get results
    print(f"\n📊 Getting search results...")
    try:
        response = requests.get(f"{BACKEND_URL}/search/results/{search_id}")
        if response.status_code == 200:
            results = response.json()['data']
            print(f"✅ Results retrieved!")
            print(f"📍 Total places: {results.get('total_places', 0)}")
            
            if 'processed_data' in results:
                processed_data = results['processed_data']
                print(f"🔧 Processed data: {len(processed_data)} places")
                
                # Show sample data
                if len(processed_data) > 0:
                    print(f"\n📋 Sample result:")
                    sample = processed_data[0]
                    for key, value in sample.items():
                        print(f"  {key}: {value}")
                    
                    # Check required fields for frontend
                    required_fields = ['name', 'category', 'rating', 'city', 'address']
                    print(f"\n🔍 Field check:")
                    for field in required_fields:
                        if field in sample:
                            print(f"  ✅ {field}: {sample[field]}")
                        else:
                            print(f"  ❌ {field}: Missing")
                            
            else:
                print(f"⚠️ No processed_data found in results")
                print(f"📋 Results structure: {list(results.keys())}")
        else:
            print(f"❌ Results retrieval failed: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Results request failed: {e}")
    
    print(f"\n🎉 Test completed!")

if __name__ == "__main__":
    test_search_flow()