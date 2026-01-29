#!/usr/bin/env python3
"""
ETL Pipeline Demo Script
========================

This script demonstrates how to:
1. Configure API keys for real data extraction
2. Run the enhanced ETL pipeline
3. View the extracted data

Usage:
    python demo.py

Requirements:
    - API keys configured in .env file
    - Dependencies installed (pip install -r requirements.txt)
"""

import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from scripts.enhanced_etl_pipeline import EnhancedETLPipeline, APIExtractor
import sqlite3

# Load environment variables
load_dotenv()

def check_api_keys():
    """Check which API keys are configured"""
    print("🔑 API Keys Status:")
    print("-" * 30)

    apis = {
        'OpenWeatherMap': 'OPENWEATHER_API_KEY',
        'Alpha Vantage': 'ALPHA_VANTAGE_API_KEY',
        'NewsAPI': 'NEWS_API_KEY'
    }

    configured = 0
    for api_name, env_var in apis.items():
        key = os.getenv(env_var)
        status = "✅ Configured" if key else "❌ Missing"
        print(f"  {api_name}: {status}")
        if key:
            configured += 1

    print(f"\n📊 {configured}/3 API keys configured")
    return configured

def test_api_connections():
    """Test API connections with configured keys"""
    print("\n🌐 Testing API Connections:")
    print("-" * 35)

    extractor = APIExtractor()

    # Test weather API
    weather_key = os.getenv('OPENWEATHER_API_KEY')
    if weather_key:
        print("  Testing OpenWeatherMap API...")
        data = extractor.extract_weather_data("London")
        if data:
            print("    ✅ Weather data extracted successfully")
            print(f"       City: {data['city']}, Temp: {data['temperature']}°C")
        else:
            print("    ❌ Weather API test failed")
    else:
        print("  ⏭️  Skipping OpenWeatherMap API test (no key)")

    # Test financial API
    financial_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    if financial_key:
        print("  Testing Alpha Vantage API...")
        data = extractor.extract_financial_data("AAPL")
        if data:
            print("    ✅ Financial data extracted successfully")
            print(f"       Symbol: {data['symbol']}, Price: ${data['close']}")
        else:
            print("    ❌ Financial API test failed")
    else:
        print("  ⏭️  Skipping Alpha Vantage API test (no key)")

    # Test news API
    news_key = os.getenv('NEWS_API_KEY')
    if news_key:
        print("  Testing NewsAPI...")
        data = extractor.extract_news_data("us")
        if data and len(data) > 0:
            print("    ✅ News data extracted successfully")
            print(f"       Articles found: {len(data)}")
        else:
            print("    ❌ News API test failed")
    else:
        print("  ⏭️  Skipping NewsAPI test (no key)")

def run_demo_etl():
    """Run a demo ETL pipeline"""
    print("\n🚀 Running Demo ETL Pipeline:")
    print("-" * 35)

    try:
        pipeline = EnhancedETLPipeline(db_type='sqlite')
        results = pipeline.run_etl_pipeline()

        if results['success']:
            print("✅ Demo ETL completed successfully!")
            print(f"   Execution time: {results['duration_seconds']:.2f} seconds")
            stats = results['extraction_stats']
            print(f"   Extracted: {stats['extracted']} records")
            print(f"   Cleaned: {stats['cleaned']} records")
            print(f"   Inserted: {stats['inserted']} records")

            return True
        else:
            print(f"❌ Demo ETL failed: {results.get('error', 'Unknown error')}")
            return False

    except Exception as e:
        print(f"❌ Error running demo ETL: {e}")
        return False

def show_sample_data():
    """Show sample data from the database"""
    print("\n📊 Sample Data from Database:")
    print("-" * 35)

    try:
        # Connect to database
        db_path = project_root / 'enhanced_etl.db'
        if not db_path.exists():
            print("❌ Database file not found")
            return

        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Show news data
        cursor.execute("SELECT title, source, published_at FROM news_data ORDER BY timestamp DESC LIMIT 3")
        news_rows = cursor.fetchall()

        if news_rows:
            print("📰 Recent News Articles:")
            for i, (title, source, published_at) in enumerate(news_rows, 1):
                print(f"   {i}. {title[:50]}... ({source})")
        else:
            print("📰 No news data found")

        # Show country data
        cursor.execute("SELECT name, capital, population FROM country_data ORDER BY population DESC LIMIT 3")
        country_rows = cursor.fetchall()

        if country_rows:
            print("\n🌍 Country Information:")
            for i, (name, capital, population) in enumerate(country_rows, 1):
                print(f"   {i}. {name} - Capital: {capital}, Population: {population:,}")
        else:
            print("\n🌍 No country data found")

        # Show database stats
        tables = ['weather_data', 'financial_data', 'news_data', 'country_data']
        print("\n📈 Database Summary:")
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            table_name = table.replace('_', ' ').title()
            print(f"   {table_name}: {count} records")

        conn.close()

    except Exception as e:
        print(f"❌ Error reading database: {e}")

def show_next_steps():
    """Show next steps for the user"""
    print("\n🎯 Next Steps:")
    print("-" * 15)
    print("1. Get API keys from the services mentioned above")
    print("2. Add keys to your .env file")
    print("3. Run: python scripts/enhanced_etl_pipeline.py")
    print("4. Schedule daily runs with: python scripts/scheduler.py run")
    print("5. Monitor with: python scripts/scheduler.py status")
    print("6. Set up automated scheduling (cron/Windows Task Scheduler)")

    print("\n🔧 Advanced Features:")
    print("• Switch to PostgreSQL/MongoDB in .env")
    print("• Configure email/Slack alerts")
    print("• Add data visualization dashboard")
    print("• Implement custom data sources")

def main():
    """Main demo function"""
    print("=" * 60)
    print("🚀 Enhanced ETL Pipeline Demo")
    print("=" * 60)

    # Check API keys
    configured_keys = check_api_keys()

    if configured_keys == 0:
        print("\n⚠️  No API keys configured. The demo will use fallback data.")
        print("   To get real data, configure API keys in your .env file.")
    elif configured_keys < 3:
        print(f"\n⚠️  {3-configured_keys} API key(s) missing. Some data sources will use fallback.")

    # Test API connections
    if configured_keys > 0:
        test_api_connections()

    # Run demo ETL
    success = run_demo_etl()

    if success:
        # Show sample data
        show_sample_data()

    # Show next steps
    show_next_steps()

    print("\n" + "=" * 60)
    print("✨ Demo completed! Check the README.md for detailed documentation.")
    print("=" * 60)

if __name__ == "__main__":
    main()