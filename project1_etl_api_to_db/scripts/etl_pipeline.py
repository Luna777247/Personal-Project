from scripts.weather_api_extractor import extract_weather_data
from scripts.financial_api_extractor import extract_financial_data
from scripts.database_utils import create_tables, insert_weather_data, insert_financial_data
import pandas as pd

def clean_weather_data(data):
    """Clean and normalize weather data"""
    if data:
        # Normalize temperature to Celsius if needed
        # Add data quality checks
        return data
    return None

def clean_financial_data(data):
    """Clean and normalize financial data"""
    if data:
        # Remove outliers, normalize volume
        return data
    return None

def run_etl():
    """Main ETL pipeline"""
    # Create tables if not exist
    create_tables()
    
    # Extract data
    cities = ['London', 'New York', 'Tokyo']
    symbols = ['IBM', 'AAPL', 'GOOGL']
    
    # Weather data ETL
    for city in cities:
        raw_data = extract_weather_data(city)
        cleaned_data = clean_weather_data(raw_data)
        if cleaned_data:
            insert_weather_data(cleaned_data)
            print(f"Inserted weather data for {city}")
    
    # Financial data ETL
    for symbol in symbols:
        raw_data = extract_financial_data(symbol)
        cleaned_data = clean_financial_data(raw_data)
        if cleaned_data:
            insert_financial_data(cleaned_data)
            print(f"Inserted financial data for {symbol}")
    
    print("ETL pipeline completed successfully")

if __name__ == "__main__":
    run_etl()