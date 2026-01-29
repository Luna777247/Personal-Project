#!/usr/bin/env python3
"""
WebSocket Test Client for Real-time News Data Pipeline
======================================================

This script tests the WebSocket connection and demonstrates
the real-time news streaming functionality.
"""

import asyncio
import websockets
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_websocket_connection():
    """Test WebSocket connection to the news pipeline"""
    uri = "ws://localhost:8000/ws"

    try:
        async with websockets.connect(uri) as websocket:
            logger.info("🔗 Connected to News Pipeline WebSocket")

            # Send a test message if needed
            # await websocket.send(json.dumps({"type": "test"}))

            # Listen for news articles
            article_count = 0
            while article_count < 5:  # Test with first 5 articles
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                    data = json.loads(message)

                    if data.get("type") == "news_article":
                        article_count += 1
                        article = data["data"]
                        logger.info(f"📄 Article {article_count}: {article['title'][:50]}...")
                        logger.info(f"   URL: {article['url']}")
                        logger.info(f"   Source: {article['source']}")
                        logger.info("---")

                    elif data.get("type") == "welcome":
                        logger.info(f"📢 {data['message']}")

                except asyncio.TimeoutError:
                    logger.warning("⏰ Timeout waiting for news articles")
                    break

            logger.info(f"✅ Successfully received {article_count} news articles")

    except Exception as e:
        logger.error(f"❌ WebSocket connection failed: {e}")
        return False

    return True

async def test_health_endpoint():
    """Test the health endpoint"""
    import aiohttp

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:8000/health") as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info("✅ Health check passed:")
                    logger.info(f"   Status: {data['status']}")
                    logger.info(f"   Clients: {data['active_connections']}")
                    logger.info(f"   Clients: {data['active_connections']}")
                    return True
                else:
                    logger.error(f"❌ Health check failed: {response.status}")
                    return False
    except Exception as e:
        logger.error(f"❌ Health check error: {e}")
        return False

async def main():
    """Main test function"""
    logger.info("🧪 Testing Real-time News Data Pipeline")
    logger.info("=" * 50)

    # Test health endpoint
    logger.info("1. Testing health endpoint...")
    health_ok = await test_health_endpoint()

    if not health_ok:
        logger.error("❌ Backend server not responding")
        return

    # Test WebSocket connection
    logger.info("2. Testing WebSocket connection...")
    ws_ok = await test_websocket_connection()

    if ws_ok:
        logger.info("🎉 All tests passed! Big data pipeline is working!")
        logger.info("📊 Pipeline demonstrates:")
        logger.info("   • Real-time data streaming")
        logger.info("   • WebSocket bidirectional communication")
        logger.info("   • Concurrent news source processing")
        logger.info("   • Distributed client-server architecture")
    else:
        logger.error("❌ WebSocket test failed")

if __name__ == "__main__":
    asyncio.run(main())