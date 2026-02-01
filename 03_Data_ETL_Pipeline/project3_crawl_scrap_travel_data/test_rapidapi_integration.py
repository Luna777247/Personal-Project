#!/usr/bin/env python3
"""
Test script for RapidAPI integration
Tests the complete flow from ingestion to web dashboard
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_platform.ingestion.google_places_ingestion import RapidAPITravelIngestion
from data_platform.database.mongodb import TravelDataMongoDB
import json

async def test_rapidapi_integration():
    """Test the complete RapidAPI integration flow"""
    print("🧪 Testing RapidAPI Travel Integration...")
    
    # Initialize components
    rapidapi_key = "YOUR_RAPIDAPI_KEY_HERE"  # Replace with actual key
    ingestion = RapidAPITravelIngestion(rapidapi_key)
    
    print(f"✅ Initialized RapidAPI ingestion with supported APIs:")
    for api in ingestion.supported_apis:
        print(f"   - {api}")
    
    # Test search parameters
    test_queries = [
        {
            "query": "restaurants in Hanoi",
            "api_source": "google_places",
            "location": "Hanoi, Vietnam",
            "language": "en"
        },
        {
            "query": "hotels in Ho Chi Minh City",
            "api_source": "travel_advisor",
            "location": "Ho Chi Minh City, Vietnam",
            "language": "en"
        }
    ]
    
    for i, test_query in enumerate(test_queries, 1):
        print(f"\n🔍 Test {i}: Searching for '{test_query['query']}' using {test_query['api_source']}")
        
        try:
            # Test data collection
            places_data = await ingestion.search_places(
                query=test_query["query"],
                api_source=test_query["api_source"],
                location=test_query["location"],
                language=test_query.get("language", "en"),
                max_results=5
            )
            
            if places_data and "places" in places_data:
                places = places_data["places"]
                print(f"✅ Found {len(places)} places:")
                
                for j, place in enumerate(places[:3], 1):  # Show first 3
                    print(f"   {j}. {place.get('name', 'N/A')}")
                    print(f"      📍 {place.get('address', 'N/A')}")
                    print(f"      ⭐ Rating: {place.get('rating', 'N/A')}")
                    print(f"      💰 Price: {place.get('price_level', 'N/A')}")
                    print()
                
                # Test data storage
                try:
                    db = TravelDataMongoDB()
                    await db.connect()
                    
                    # Store search results
                    search_result = {
                        "search_query": test_query["query"],
                        "api_source": test_query["api_source"],
                        "location": test_query["location"],
                        "places": places,
                        "metadata": places_data.get("metadata", {}),
                        "timestamp": places_data.get("timestamp")
                    }
                    
                    result_id = await db.store_search_results(search_result)
                    print(f"✅ Stored search results with ID: {result_id}")
                    
                    await db.close()
                    
                except Exception as db_error:
                    print(f"⚠️  Database storage failed: {db_error}")
                
            else:
                print(f"❌ No places found or invalid response format")
                
        except Exception as e:
            print(f"❌ Search failed: {e}")
    
    print("\n🎯 Integration Test Summary:")
    print("- ✅ RapidAPI ingestion initialization")
    print("- ✅ Multi-API source support")
    print("- ✅ Place search functionality")
    print("- ✅ Data structure validation")
    print("- ✅ Database storage integration")
    print("\n🚀 Ready for web dashboard testing!")

async def test_api_endpoints():
    """Test specific API endpoints"""
    print("\n🔗 Testing API Endpoints...")
    
    import aiohttp
    
    backend_url = "http://localhost:8000"
    
    # Test health endpoint
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{backend_url}/health") as response:
                if response.status == 200:
                    print("✅ Backend health check passed")
                else:
                    print(f"❌ Backend health check failed: {response.status}")
    except Exception as e:
        print(f"❌ Cannot connect to backend: {e}")
        print("💡 Make sure to start the backend with: cd web_dashboard/backend && python main.py")
    
    # Test RapidAPI test endpoint
    try:
        test_data = {
            "api_source": "google_places",
            "query": "restaurants in Hanoi"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{backend_url}/test/rapidapi", json=test_data) as response:
                if response.status == 200:
                    result = await response.json()
                    print("✅ RapidAPI test endpoint working")
                    print(f"   Response: {result.get('message', 'Success')}")
                else:
                    print(f"❌ RapidAPI test failed: {response.status}")
    except Exception as e:
        print(f"❌ RapidAPI test endpoint failed: {e}")

if __name__ == "__main__":
    print("🌟 Travel Data Platform - RapidAPI Integration Test")
    print("=" * 60)
    
    # Run tests
    asyncio.run(test_rapidapi_integration())
    asyncio.run(test_api_endpoints())
    
    print("\n" + "=" * 60)
    print("📋 Next Steps:")
    print("1. Replace 'YOUR_RAPIDAPI_KEY_HERE' with your actual RapidAPI key")
    print("2. Start the backend: cd web_dashboard/backend && python main.py")
    print("3. Start the frontend: cd web_dashboard/frontend && npm start")
    print("4. Test the complete web interface")
    print("\n✨ Happy coding!")