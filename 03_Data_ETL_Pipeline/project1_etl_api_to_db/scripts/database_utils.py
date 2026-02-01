import psycopg2
from config.database_config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD

def create_connection():
    """Create database connection"""
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )

def create_tables():
    """Create necessary tables"""
    conn = create_connection()
    cursor = conn.cursor()
    
    # Weather data table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS weather_data (
            id SERIAL PRIMARY KEY,
            city VARCHAR(100),
            temperature FLOAT,
            humidity FLOAT,
            pressure FLOAT,
            weather_description TEXT,
            timestamp TIMESTAMP,
            UNIQUE(city, timestamp)
        )
    ''')
    
    # Financial data table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS financial_data (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(10),
            date DATE,
            open_price FLOAT,
            high_price FLOAT,
            low_price FLOAT,
            close_price FLOAT,
            volume BIGINT,
            timestamp TIMESTAMP,
            UNIQUE(symbol, date)
        )
    ''')
    
    conn.commit()
    cursor.close()
    conn.close()

def insert_weather_data(data):
    """Insert weather data with deduplication"""
    conn = create_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO weather_data (city, temperature, humidity, pressure, weather_description, timestamp)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (city, timestamp) DO NOTHING
    ''', (data['city'], data['temperature'], data['humidity'], data['pressure'], 
          data['weather_description'], data['timestamp']))
    
    conn.commit()
    cursor.close()
    conn.close()

def insert_financial_data(data):
    """Insert financial data with deduplication"""
    conn = create_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO financial_data (symbol, date, open_price, high_price, low_price, close_price, volume, timestamp)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (symbol, date) DO NOTHING
    ''', (data['symbol'], data['date'], data['open'], data['high'], data['low'], 
          data['close'], data['volume'], data['timestamp']))
    
    conn.commit()
    cursor.close()
    conn.close()