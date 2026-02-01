import sqlite3
from datetime import datetime
import json

def create_mock_weather_data(city):
    """Create mock weather data for demonstration"""
    mock_data = {
        'London': {'temp': 15.5, 'humidity': 72, 'pressure': 1013, 'description': 'cloudy'},
        'New York': {'temp': 22.3, 'humidity': 65, 'pressure': 1018, 'description': 'sunny'},
        'Tokyo': {'temp': 18.7, 'humidity': 78, 'pressure': 1009, 'description': 'rainy'}
    }
    data = mock_data.get(city, {'temp': 20.0, 'humidity': 70, 'pressure': 1013, 'description': 'clear'})
    return {
        'city': city,
        'temperature': data['temp'],
        'humidity': data['humidity'],
        'pressure': data['pressure'],
        'weather_description': data['description'],
        'timestamp': datetime.now().isoformat()
    }

def create_mock_financial_data(symbol):
    """Create mock financial data for demonstration"""
    mock_data = {
        'IBM': {'open': 145.20, 'high': 147.80, 'low': 144.50, 'close': 146.75, 'volume': 2456789},
        'AAPL': {'open': 182.50, 'high': 185.20, 'low': 181.80, 'close': 184.30, 'volume': 45678912},
        'GOOGL': {'open': 2750.00, 'high': 2785.50, 'low': 2740.00, 'close': 2775.25, 'volume': 1234567}
    }
    data = mock_data.get(symbol, {'open': 100.0, 'high': 105.0, 'low': 99.0, 'close': 102.5, 'volume': 1000000})
    return {
        'symbol': symbol,
        'date': datetime.now().date().isoformat(),
        'open': data['open'],
        'high': data['high'],
        'low': data['low'],
        'close': data['close'],
        'volume': data['volume'],
        'timestamp': datetime.now().isoformat()
    }

def create_database():
    """Create SQLite database and tables"""
    conn = sqlite3.connect('etl_demo.db')
    cursor = conn.cursor()

    # Weather data table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS weather_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            city TEXT,
            temperature REAL,
            humidity REAL,
            pressure REAL,
            weather_description TEXT,
            timestamp TEXT,
            UNIQUE(city, timestamp)
        )
    ''')

    # Financial data table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS financial_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            date TEXT,
            open_price REAL,
            high_price REAL,
            low_price REAL,
            close_price REAL,
            volume INTEGER,
            timestamp TEXT,
            UNIQUE(symbol, date)
        )
    ''')

    conn.commit()
    return conn

def insert_weather_data(conn, data):
    """Insert weather data"""
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT OR IGNORE INTO weather_data (city, temperature, humidity, pressure, weather_description, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (data['city'], data['temperature'], data['humidity'], data['pressure'],
              data['weather_description'], data['timestamp']))
        conn.commit()
        print(f"Inserted weather data for {data['city']}")
    except Exception as e:
        print(f"Error inserting weather data: {e}")

def insert_financial_data(conn, data):
    """Insert financial data"""
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT OR IGNORE INTO financial_data (symbol, date, open_price, high_price, low_price, close_price, volume, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (data['symbol'], data['date'], data['open'], data['high'], data['low'],
              data['close'], data['volume'], data['timestamp']))
        conn.commit()
        print(f"Inserted financial data for {data['symbol']}")
    except Exception as e:
        print(f"Error inserting financial data: {e}")

def run_etl():
    """Main ETL pipeline with mock data"""
    print("Starting ETL Pipeline (Demo Mode)...")

    # Create database
    conn = create_database()

    # Define test data
    cities = ['London', 'New York', 'Tokyo']
    symbols = ['IBM', 'AAPL', 'GOOGL']

    # Weather data ETL
    print("\nProcessing weather data...")
    for city in cities:
        data = create_mock_weather_data(city)
        insert_weather_data(conn, data)

    # Financial data ETL
    print("\nProcessing financial data...")
    for symbol in symbols:
        data = create_mock_financial_data(symbol)
        insert_financial_data(conn, data)

    # Show results
    print("\n=== ETL Results ===")

    cursor = conn.cursor()

    # Weather data count
    cursor.execute("SELECT COUNT(*) FROM weather_data")
    weather_count = cursor.fetchone()[0]
    print(f"Weather records inserted: {weather_count}")

    # Financial data count
    cursor.execute("SELECT COUNT(*) FROM financial_data")
    financial_count = cursor.fetchone()[0]
    print(f"Financial records inserted: {financial_count}")

    # Sample weather data
    cursor.execute("SELECT city, temperature, weather_description FROM weather_data LIMIT 3")
    weather_samples = cursor.fetchall()
    print("\nSample weather data:")
    for row in weather_samples:
        print(f"  {row[0]}: {row[1]}°C, {row[2]}")

    # Sample financial data
    cursor.execute("SELECT symbol, close_price, volume FROM financial_data LIMIT 3")
    financial_samples = cursor.fetchall()
    print("\nSample financial data:")
    for row in financial_samples:
        print(f"  {row[0]}: ${row[1]:.2f}, Volume: {row[2]:,}")

    conn.close()
    print("\nETL pipeline completed successfully!")

if __name__ == "__main__":
    run_etl()