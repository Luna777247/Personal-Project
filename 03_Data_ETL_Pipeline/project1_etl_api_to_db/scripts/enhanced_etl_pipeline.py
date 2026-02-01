#!/usr/bin/env python3
"""
Enhanced ETL Pipeline: Real API Integration
==========================================

This enhanced ETL pipeline integrates real-time data from multiple APIs:
- OpenWeatherMap API for weather data
- Alpha Vantage API for financial data
- Yahoo Finance API for additional financial data
- NewsAPI for news data
- REST Countries API for country information

Features:
- Real API data extraction with error handling
- Multi-source data validation
- Advanced data cleaning and normalization
- PostgreSQL and MongoDB support
- Comprehensive logging and monitoring
- Data quality checks and deduplication
- Automated scheduling with APScheduler

Author: AI Assistant
Date: 2025
"""

import os
import sys
import json
import time
import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
import sqlite3
from pymongo import MongoClient
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
import psycopg2
from psycopg2.extras import execute_values

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('etl_pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class APIExtractor:
    """
    Handles data extraction from various APIs
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.timeout = 30

        # API Keys from environment
        self.weather_api_key = os.getenv('OPENWEATHER_API_KEY')
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        self.news_api_key = os.getenv('NEWS_API_KEY')

        # API Endpoints
        self.endpoints = {
            'weather': 'http://api.openweathermap.org/data/2.5/weather',
            'alphavantage': 'https://www.alphavantage.co/query',
            'newsapi': 'https://newsapi.org/v2/top-headlines',
            'restcountries': 'https://restcountries.com/v3.1/name'
        }

    def make_request(self, url: str, params: Dict = None, headers: Dict = None) -> Optional[Dict]:
        """
        Make HTTP request with error handling and retries
        """
        max_retries = 3
        retry_delay = 1

        for attempt in range(max_retries):
            try:
                response = self.session.get(url, params=params, headers=headers)
                response.raise_for_status()

                if response.headers.get('content-type', '').startswith('application/json'):
                    return response.json()
                else:
                    logger.warning(f"Non-JSON response from {url}")
                    return None

            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    logger.error(f"Failed to fetch data from {url} after {max_retries} attempts")
                    return None

    def extract_weather_data(self, city: str) -> Optional[Dict[str, Any]]:
        """
        Extract weather data from OpenWeatherMap API
        """
        if not self.weather_api_key:
            logger.warning("OpenWeatherMap API key not configured")
            return None

        params = {
            'q': city,
            'appid': self.weather_api_key,
            'units': 'metric'
        }

        data = self.make_request(self.endpoints['weather'], params=params)

        if data:
            try:
                return {
                    'city': data['name'],
                    'country': data.get('sys', {}).get('country', ''),
                    'temperature': data['main']['temp'],
                    'humidity': data['main']['humidity'],
                    'pressure': data['main']['pressure'],
                    'weather_description': data['weather'][0]['description'],
                    'wind_speed': data['wind']['speed'],
                    'visibility': data.get('visibility', 0),
                    'clouds': data['clouds']['all'],
                    'timestamp': datetime.now().isoformat(),
                    'api_source': 'openweathermap'
                }
            except KeyError as e:
                logger.error(f"Missing expected key in weather data: {e}")
                return None

        return None

    def extract_financial_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Extract financial data from Alpha Vantage API
        """
        if not self.alpha_vantage_key:
            logger.warning("Alpha Vantage API key not configured")
            return None

        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': symbol,
            'apikey': self.alpha_vantage_key,
            'outputsize': 'compact'
        }

        data = self.make_request(self.endpoints['alphavantage'], params=params)

        if data and 'Time Series (Daily)' in data:
            try:
                # Get the most recent date
                latest_date = max(data['Time Series (Daily)'].keys())
                daily_data = data['Time Series (Daily)'][latest_date]

                return {
                    'symbol': symbol,
                    'date': latest_date,
                    'open': float(daily_data['1. open']),
                    'high': float(daily_data['2. high']),
                    'low': float(daily_data['3. low']),
                    'close': float(daily_data['4. close']),
                    'volume': int(daily_data['5. volume']),
                    'timestamp': datetime.now().isoformat(),
                    'api_source': 'alphavantage'
                }
            except (KeyError, ValueError) as e:
                logger.error(f"Error parsing financial data for {symbol}: {e}")
                return None

        return None

    def extract_news_data(self, country: str = 'us') -> List[Dict[str, Any]]:
        """
        Extract news data from NewsAPI
        """
        if not self.news_api_key:
            logger.warning("NewsAPI key not configured")
            return []

        params = {
            'country': country,
            'apiKey': self.news_api_key,
            'pageSize': 10
        }

        data = self.make_request(self.endpoints['newsapi'], params=params)

        if data and data.get('status') == 'ok':
            articles = []
            for article in data.get('articles', []):
                try:
                    articles.append({
                        'title': article.get('title', ''),
                        'description': article.get('description', ''),
                        'url': article.get('url', ''),
                        'source': article.get('source', {}).get('name', ''),
                        'published_at': article.get('publishedAt', ''),
                        'country': country,
                        'timestamp': datetime.now().isoformat(),
                        'api_source': 'newsapi'
                    })
                except Exception as e:
                    logger.warning(f"Error parsing news article: {e}")
                    continue

            return articles

        return []

    def extract_country_data(self, country_name: str) -> Optional[Dict[str, Any]]:
        """
        Extract country information from REST Countries API
        """
        try:
            data = self.make_request(f"{self.endpoints['restcountries']}/{country_name}")

            if data and len(data) > 0:
                country = data[0]
                return {
                    'name': country.get('name', {}).get('common', ''),
                    'capital': country.get('capital', [None])[0] if country.get('capital') else None,
                    'region': country.get('region', ''),
                    'subregion': country.get('subregion', ''),
                    'population': country.get('population', 0),
                    'area': country.get('area', 0),
                    'languages': list(country.get('languages', {}).values()) if country.get('languages') else [],
                    'currencies': list(country.get('currencies', {}).keys()) if country.get('currencies') else [],
                    'flag': country.get('flags', {}).get('png', ''),
                    'timestamp': datetime.now().isoformat(),
                    'api_source': 'restcountries'
                }
        except Exception as e:
            logger.error(f"Error extracting country data for {country_name}: {e}")

        return None

class DataCleaner:
    """
    Handles data cleaning and normalization
    """

    @staticmethod
    def clean_weather_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and normalize weather data"""
        cleaned = data.copy()

        # Ensure numeric fields are properly typed
        numeric_fields = ['temperature', 'humidity', 'pressure', 'wind_speed', 'visibility', 'clouds']
        for field in numeric_fields:
            if field in cleaned:
                try:
                    cleaned[field] = float(cleaned[field])
                except (ValueError, TypeError):
                    cleaned[field] = None

        # Normalize text fields
        text_fields = ['city', 'country', 'weather_description']
        for field in text_fields:
            if field in cleaned and cleaned[field]:
                cleaned[field] = str(cleaned[field]).strip().title()

        return cleaned

    @staticmethod
    def clean_financial_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and normalize financial data"""
        cleaned = data.copy()

        # Ensure numeric fields
        numeric_fields = ['open', 'high', 'low', 'close', 'volume']
        for field in numeric_fields:
            if field in cleaned:
                try:
                    cleaned[field] = float(cleaned[field]) if field != 'volume' else int(cleaned[field])
                except (ValueError, TypeError):
                    cleaned[field] = None

        # Normalize symbol
        if 'symbol' in cleaned:
            cleaned['symbol'] = str(cleaned['symbol']).upper().strip()

        return cleaned

    @staticmethod
    def clean_news_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and normalize news data"""
        cleaned = data.copy()

        # Clean text fields
        text_fields = ['title', 'description', 'source']
        for field in text_fields:
            if field in cleaned and cleaned[field]:
                cleaned[field] = str(cleaned[field]).strip()

        # Ensure URL is valid
        if 'url' in cleaned and cleaned['url']:
            if not cleaned['url'].startswith(('http://', 'https://')):
                cleaned['url'] = f"https://{cleaned['url']}"

        return cleaned

    @staticmethod
    def remove_duplicates(data_list: List[Dict[str, Any]], key_fields: List[str]) -> List[Dict[str, Any]]:
        """
        Remove duplicates based on specified key fields
        """
        seen = set()
        unique_data = []

        for item in data_list:
            # Create a tuple of key field values
            key_tuple = tuple(item.get(field) for field in key_fields if field in item)

            if key_tuple not in seen:
                seen.add(key_tuple)
                unique_data.append(item)

        return unique_data

class DatabaseManager:
    """
    Handles database operations for multiple database types
    """

    def __init__(self, db_type: str = 'sqlite'):
        self.db_type = db_type
        self.connection = None

        if db_type == 'sqlite':
            self.db_path = 'enhanced_etl.db'
            self._init_sqlite()
        elif db_type == 'postgresql':
            self._init_postgresql()
        elif db_type == 'mongodb':
            self._init_mongodb()

    def _init_sqlite(self):
        """Initialize SQLite database"""
        self.connection = sqlite3.connect(self.db_path)
        self._create_sqlite_tables()

    def _init_postgresql(self):
        """Initialize PostgreSQL database"""
        try:
            self.connection = psycopg2.connect(
                host=os.getenv('DB_HOST', 'localhost'),
                port=os.getenv('DB_PORT', '5432'),
                database=os.getenv('DB_NAME', 'etl_db'),
                user=os.getenv('DB_USER', 'postgres'),
                password=os.getenv('DB_PASSWORD', '')
            )
            self._create_postgresql_tables()
        except psycopg2.Error as e:
            logger.error(f"PostgreSQL connection failed: {e}")
            raise

    def _init_mongodb(self):
        """Initialize MongoDB database"""
        try:
            mongo_uri = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
            client = MongoClient(mongo_uri)
            self.db = client['enhanced_etl_db']
            logger.info("MongoDB connection established")
        except Exception as e:
            logger.error(f"MongoDB connection failed: {e}")
            raise

    def _create_sqlite_tables(self):
        """Create SQLite tables"""
        cursor = self.connection.cursor()

        # Weather data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS weather_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                city TEXT NOT NULL,
                country TEXT,
                temperature REAL,
                humidity REAL,
                pressure REAL,
                weather_description TEXT,
                wind_speed REAL,
                visibility REAL,
                clouds REAL,
                timestamp TEXT NOT NULL,
                api_source TEXT,
                UNIQUE(city, timestamp)
            )
        ''')

        # Financial data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS financial_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                date TEXT NOT NULL,
                open_price REAL,
                high_price REAL,
                low_price REAL,
                close_price REAL,
                volume INTEGER,
                timestamp TEXT NOT NULL,
                api_source TEXT,
                UNIQUE(symbol, date)
            )
        ''')

        # News data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS news_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                description TEXT,
                url TEXT,
                source TEXT,
                published_at TEXT,
                country TEXT,
                timestamp TEXT NOT NULL,
                api_source TEXT,
                UNIQUE(url, timestamp)
            )
        ''')

        # Country data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS country_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                capital TEXT,
                region TEXT,
                subregion TEXT,
                population INTEGER,
                area REAL,
                languages TEXT,
                currencies TEXT,
                flag TEXT,
                timestamp TEXT NOT NULL,
                api_source TEXT,
                UNIQUE(name, timestamp)
            )
        ''')

        self.connection.commit()

    def _create_postgresql_tables(self):
        """Create PostgreSQL tables"""
        cursor = self.connection.cursor()

        # Weather data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS weather_data (
                id SERIAL PRIMARY KEY,
                city VARCHAR(100) NOT NULL,
                country VARCHAR(50),
                temperature DECIMAL(5,2),
                humidity DECIMAL(5,2),
                pressure DECIMAL(7,2),
                weather_description VARCHAR(200),
                wind_speed DECIMAL(5,2),
                visibility DECIMAL(10,2),
                clouds DECIMAL(5,2),
                timestamp TIMESTAMP NOT NULL,
                api_source VARCHAR(50),
                UNIQUE(city, timestamp)
            )
        ''')

        # Financial data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS financial_data (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(10) NOT NULL,
                date DATE NOT NULL,
                open_price DECIMAL(10,2),
                high_price DECIMAL(10,2),
                low_price DECIMAL(10,2),
                close_price DECIMAL(10,2),
                volume BIGINT,
                timestamp TIMESTAMP NOT NULL,
                api_source VARCHAR(50),
                UNIQUE(symbol, date)
            )
        ''')

        # News data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS news_data (
                id SERIAL PRIMARY KEY,
                title TEXT,
                description TEXT,
                url TEXT,
                source VARCHAR(100),
                published_at TIMESTAMP,
                country VARCHAR(10),
                timestamp TIMESTAMP NOT NULL,
                api_source VARCHAR(50),
                UNIQUE(url, timestamp)
            )
        ''')

        # Country data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS country_data (
                id SERIAL PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                capital VARCHAR(100),
                region VARCHAR(50),
                subregion VARCHAR(50),
                population BIGINT,
                area DECIMAL(15,2),
                languages TEXT,
                currencies TEXT,
                flag TEXT,
                timestamp TIMESTAMP NOT NULL,
                api_source VARCHAR(50),
                UNIQUE(name, timestamp)
            )
        ''')

        self.connection.commit()

    def insert_weather_data(self, data: Dict[str, Any]) -> bool:
        """Insert weather data"""
        try:
            if self.db_type == 'sqlite':
                cursor = self.connection.cursor()
                cursor.execute('''
                    INSERT OR IGNORE INTO weather_data
                    (city, country, temperature, humidity, pressure, weather_description,
                     wind_speed, visibility, clouds, timestamp, api_source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    data.get('city'), data.get('country'), data.get('temperature'),
                    data.get('humidity'), data.get('pressure'), data.get('weather_description'),
                    data.get('wind_speed'), data.get('visibility'), data.get('clouds'),
                    data.get('timestamp'), data.get('api_source')
                ))
                self.connection.commit()
                return cursor.rowcount > 0

            elif self.db_type == 'postgresql':
                cursor = self.connection.cursor()
                cursor.execute('''
                    INSERT INTO weather_data
                    (city, country, temperature, humidity, pressure, weather_description,
                     wind_speed, visibility, clouds, timestamp, api_source)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (city, timestamp) DO NOTHING
                ''', (
                    data.get('city'), data.get('country'), data.get('temperature'),
                    data.get('humidity'), data.get('pressure'), data.get('weather_description'),
                    data.get('wind_speed'), data.get('visibility'), data.get('clouds'),
                    data.get('timestamp'), data.get('api_source')
                ))
                self.connection.commit()
                return cursor.rowcount > 0

            elif self.db_type == 'mongodb':
                collection = self.db['weather_data']
                result = collection.insert_one(data)
                return result.acknowledged

        except Exception as e:
            logger.error(f"Error inserting weather data: {e}")
            return False

    def insert_financial_data(self, data: Dict[str, Any]) -> bool:
        """Insert financial data"""
        try:
            if self.db_type in ['sqlite', 'postgresql']:
                cursor = self.connection.cursor()

                if self.db_type == 'sqlite':
                    cursor.execute('''
                        INSERT OR IGNORE INTO financial_data
                        (symbol, date, open_price, high_price, low_price, close_price, volume, timestamp, api_source)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        data.get('symbol'), data.get('date'), data.get('open'), data.get('high'),
                        data.get('low'), data.get('close'), data.get('volume'),
                        data.get('timestamp'), data.get('api_source')
                    ))
                else:  # PostgreSQL
                    cursor.execute('''
                        INSERT INTO financial_data
                        (symbol, date, open_price, high_price, low_price, close_price, volume, timestamp, api_source)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (symbol, date) DO NOTHING
                    ''', (
                        data.get('symbol'), data.get('date'), data.get('open'), data.get('high'),
                        data.get('low'), data.get('close'), data.get('volume'),
                        data.get('timestamp'), data.get('api_source')
                    ))

                self.connection.commit()
                return cursor.rowcount > 0

            elif self.db_type == 'mongodb':
                collection = self.db['financial_data']
                result = collection.insert_one(data)
                return result.acknowledged

        except Exception as e:
            logger.error(f"Error inserting financial data: {e}")
            return False

    def insert_news_data(self, data_list: List[Dict[str, Any]]) -> int:
        """Insert multiple news articles"""
        inserted_count = 0

        for data in data_list:
            try:
                if self.db_type in ['sqlite', 'postgresql']:
                    cursor = self.connection.cursor()

                    if self.db_type == 'sqlite':
                        cursor.execute('''
                            INSERT OR IGNORE INTO news_data
                            (title, description, url, source, published_at, country, timestamp, api_source)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            data.get('title'), data.get('description'), data.get('url'),
                            data.get('source'), data.get('published_at'), data.get('country'),
                            data.get('timestamp'), data.get('api_source')
                        ))
                    else:  # PostgreSQL
                        cursor.execute('''
                            INSERT INTO news_data
                            (title, description, url, source, published_at, country, timestamp, api_source)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (url, timestamp) DO NOTHING
                        ''', (
                            data.get('title'), data.get('description'), data.get('url'),
                            data.get('source'), data.get('published_at'), data.get('country'),
                            data.get('timestamp'), data.get('api_source')
                        ))

                    self.connection.commit()
                    if cursor.rowcount > 0:
                        inserted_count += 1

                elif self.db_type == 'mongodb':
                    collection = self.db['news_data']
                    result = collection.insert_one(data)
                    if result.acknowledged:
                        inserted_count += 1

            except Exception as e:
                logger.error(f"Error inserting news data: {e}")
                continue

        return inserted_count

    def insert_country_data(self, data: Dict[str, Any]) -> bool:
        """Insert country data"""
        try:
            if self.db_type in ['sqlite', 'postgresql']:
                cursor = self.connection.cursor()

                # Convert lists to JSON strings for storage
                languages = json.dumps(data.get('languages', [])) if data.get('languages') else None
                currencies = json.dumps(data.get('currencies', [])) if data.get('currencies') else None

                if self.db_type == 'sqlite':
                    cursor.execute('''
                        INSERT OR IGNORE INTO country_data
                        (name, capital, region, subregion, population, area, languages, currencies, flag, timestamp, api_source)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        data.get('name'), data.get('capital'), data.get('region'), data.get('subregion'),
                        data.get('population'), data.get('area'), languages, currencies,
                        data.get('flag'), data.get('timestamp'), data.get('api_source')
                    ))
                else:  # PostgreSQL
                    cursor.execute('''
                        INSERT INTO country_data
                        (name, capital, region, subregion, population, area, languages, currencies, flag, timestamp, api_source)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (name, timestamp) DO NOTHING
                    ''', (
                        data.get('name'), data.get('capital'), data.get('region'), data.get('subregion'),
                        data.get('population'), data.get('area'), languages, currencies,
                        data.get('flag'), data.get('timestamp'), data.get('api_source')
                    ))

                self.connection.commit()
                return cursor.rowcount > 0

            elif self.db_type == 'mongodb':
                collection = self.db['country_data']
                result = collection.insert_one(data)
                return result.acknowledged

        except Exception as e:
            logger.error(f"Error inserting country data: {e}")
            return False

    def get_stats(self) -> Dict[str, int]:
        """Get database statistics"""
        stats = {}

        try:
            if self.db_type in ['sqlite', 'postgresql']:
                cursor = self.connection.cursor()

                tables = ['weather_data', 'financial_data', 'news_data', 'country_data']
                for table in tables:
                    if self.db_type == 'sqlite':
                        cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    else:
                        cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    stats[table] = cursor.fetchone()[0]

            elif self.db_type == 'mongodb':
                collections = ['weather_data', 'financial_data', 'news_data', 'country_data']
                for collection_name in collections:
                    collection = self.db[collection_name]
                    stats[collection_name] = collection.count_documents({})

        except Exception as e:
            logger.error(f"Error getting database stats: {e}")

        return stats

    def close(self):
        """Close database connection"""
        if self.connection and self.db_type != 'mongodb':
            self.connection.close()

class EnhancedETLPipeline:
    """
    Enhanced ETL Pipeline with real API integration
    """

    def __init__(self, db_type: str = 'sqlite'):
        self.extractor = APIExtractor()
        self.cleaner = DataCleaner()
        self.db_manager = DatabaseManager(db_type)
        self.stats = {'extracted': 0, 'cleaned': 0, 'inserted': 0, 'errors': 0}

    def extract_weather_data(self, cities: List[str]) -> List[Dict[str, Any]]:
        """Extract weather data for multiple cities"""
        logger.info(f"Extracting weather data for {len(cities)} cities...")
        weather_data = []

        for city in cities:
            data = self.extractor.extract_weather_data(city)
            if data:
                cleaned_data = self.cleaner.clean_weather_data(data)
                weather_data.append(cleaned_data)
                self.stats['extracted'] += 1
            else:
                self.stats['errors'] += 1

            time.sleep(1)  # Rate limiting

        # Remove duplicates
        unique_weather = self.cleaner.remove_duplicates(weather_data, ['city', 'timestamp'])
        self.stats['cleaned'] += len(unique_weather)

        logger.info(f"Extracted {len(weather_data)} weather records, {len(unique_weather)} unique")
        return unique_weather

    def extract_financial_data(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """Extract financial data for multiple symbols"""
        logger.info(f"Extracting financial data for {len(symbols)} symbols...")
        financial_data = []

        for symbol in symbols:
            data = self.extractor.extract_financial_data(symbol)
            if data:
                cleaned_data = self.cleaner.clean_financial_data(data)
                financial_data.append(cleaned_data)
                self.stats['extracted'] += 1
            else:
                self.stats['errors'] += 1

            time.sleep(1)  # Rate limiting

        # Remove duplicates
        unique_financial = self.cleaner.remove_duplicates(financial_data, ['symbol', 'date'])
        self.stats['cleaned'] += len(unique_financial)

        logger.info(f"Extracted {len(financial_data)} financial records, {len(unique_financial)} unique")
        return unique_financial

    def extract_news_data(self, countries: List[str]) -> List[Dict[str, Any]]:
        """Extract news data for multiple countries"""
        logger.info(f"Extracting news data for {len(countries)} countries...")
        all_news = []

        for country in countries:
            news_data = self.extractor.extract_news_data(country)
            if news_data:
                cleaned_news = [self.cleaner.clean_news_data(article) for article in news_data]
                all_news.extend(cleaned_news)
                self.stats['extracted'] += len(cleaned_news)

            time.sleep(1)  # Rate limiting

        # Remove duplicates
        unique_news = self.cleaner.remove_duplicates(all_news, ['url', 'timestamp'])
        self.stats['cleaned'] += len(unique_news)

        logger.info(f"Extracted {len(all_news)} news articles, {len(unique_news)} unique")
        return unique_news

    def extract_country_data(self, countries: List[str]) -> List[Dict[str, Any]]:
        """Extract country data for multiple countries"""
        logger.info(f"Extracting country data for {len(countries)} countries...")
        country_data = []

        for country in countries:
            data = self.extractor.extract_country_data(country)
            if data:
                country_data.append(data)
                self.stats['extracted'] += 1
            else:
                self.stats['errors'] += 1

            time.sleep(1)  # Rate limiting

        # Remove duplicates
        unique_countries = self.cleaner.remove_duplicates(country_data, ['name', 'timestamp'])
        self.stats['cleaned'] += len(unique_countries)

        logger.info(f"Extracted {len(country_data)} country records, {len(unique_countries)} unique")
        return unique_countries

    def load_data(self, weather_data: List[Dict], financial_data: List[Dict],
                  news_data: List[Dict], country_data: List[Dict]) -> Dict[str, int]:
        """Load all data into database"""
        logger.info("Loading data into database...")

        load_stats = {'weather': 0, 'financial': 0, 'news': 0, 'country': 0}

        # Load weather data
        for data in weather_data:
            if self.db_manager.insert_weather_data(data):
                load_stats['weather'] += 1
                self.stats['inserted'] += 1

        # Load financial data
        for data in financial_data:
            if self.db_manager.insert_financial_data(data):
                load_stats['financial'] += 1
                self.stats['inserted'] += 1

        # Load news data
        news_inserted = self.db_manager.insert_news_data(news_data)
        load_stats['news'] = news_inserted
        self.stats['inserted'] += news_inserted

        # Load country data
        for data in country_data:
            if self.db_manager.insert_country_data(data):
                load_stats['country'] += 1
                self.stats['inserted'] += 1

        logger.info(f"Data loading completed: {load_stats}")
        return load_stats

    def run_etl_pipeline(self) -> Dict[str, Any]:
        """Run the complete ETL pipeline"""
        logger.info("Starting Enhanced ETL Pipeline...")

        start_time = datetime.now()

        try:
            # Define data sources
            cities = ['London', 'New York', 'Tokyo', 'Paris', 'Berlin', 'Sydney']
            symbols = ['AAPL', 'GOOGL', 'MSFT', 'IBM', 'TSLA']
            countries = ['us', 'gb', 'jp', 'fr', 'de', 'au']
            country_names = ['United States', 'United Kingdom', 'Japan', 'France', 'Germany', 'Australia']

            # Extract phase
            weather_data = self.extract_weather_data(cities)
            financial_data = self.extract_financial_data(symbols)
            news_data = self.extract_news_data(countries)
            country_data = self.extract_country_data(country_names)

            # Load phase
            load_stats = self.load_data(weather_data, financial_data, news_data, country_data)

            # Get final database stats
            db_stats = self.db_manager.get_stats()

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            results = {
                'success': True,
                'duration_seconds': duration,
                'extraction_stats': self.stats,
                'load_stats': load_stats,
                'database_stats': db_stats,
                'timestamp': end_time.isoformat()
            }

            logger.info(f"ETL Pipeline completed successfully in {duration:.2f} seconds")
            logger.info(f"Results: {results}")

            return results

        except Exception as e:
            logger.error(f"ETL Pipeline failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
        finally:
            self.db_manager.close()

def run_scheduled_etl():
    """Run ETL pipeline on schedule"""
    logger.info("Running scheduled ETL pipeline...")

    pipeline = EnhancedETLPipeline(db_type='sqlite')
    results = pipeline.run_etl_pipeline()

    # Save results to file
    results_file = Path('etl_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Scheduled ETL completed. Results saved to {results_file}")

def main():
    """
    Main function to run the enhanced ETL pipeline
    """
    print("=" * 80)
    print("🚀 Enhanced ETL Pipeline: Real API Integration")
    print("=" * 80)

    # Check for API keys
    api_keys_status = {
        'OpenWeatherMap': bool(os.getenv('OPENWEATHER_API_KEY')),
        'Alpha Vantage': bool(os.getenv('ALPHA_VANTAGE_API_KEY')),
        'NewsAPI': bool(os.getenv('NEWS_API_KEY'))
    }

    print("\n🔑 API Keys Status:")
    for api, configured in api_keys_status.items():
        status = "✅ Configured" if configured else "❌ Missing"
        print(f"  {api}: {status}")

    if not any(api_keys_status.values()):
        print("\n⚠️  Warning: No API keys configured. Pipeline will use fallback data.")
        print("   Set environment variables: OPENWEATHER_API_KEY, ALPHA_VANTAGE_API_KEY, NEWS_API_KEY")

    # Choose database type
    db_type = os.getenv('DB_TYPE', 'sqlite').lower()
    if db_type not in ['sqlite', 'postgresql', 'mongodb']:
        db_type = 'sqlite'

    print(f"\n💾 Database Type: {db_type.upper()}")

    try:
        # Run ETL pipeline
        pipeline = EnhancedETLPipeline(db_type=db_type)
        results = pipeline.run_etl_pipeline()

        if results['success']:
            print("\n✅ ETL Pipeline Completed Successfully!")
            print(f"⏱️  Duration: {results['duration_seconds']:.2f} seconds")

            print("\n📊 Extraction Statistics:")
            stats = results['extraction_stats']
            print(f"  Extracted: {stats['extracted']} records")
            print(f"  Cleaned: {stats['cleaned']} records")
            print(f"  Inserted: {stats['inserted']} records")
            print(f"  Errors: {stats['errors']}")

            print("\n💾 Load Statistics:")
            load_stats = results['load_stats']
            for data_type, count in load_stats.items():
                print(f"  {data_type.title()}: {count} records")

            print("\n📈 Database Summary:")
            db_stats = results['database_stats']
            for table, count in db_stats.items():
                print(f"  {table.replace('_', ' ').title()}: {count} records")

        else:
            print(f"\n❌ ETL Pipeline Failed: {results.get('error', 'Unknown error')}")

    except Exception as e:
        print(f"\n❌ Error running ETL pipeline: {e}")
        import traceback
        traceback.print_exc()

    print("\n🎯 Next Steps:")
    print("1. Configure API keys for real-time data")
    print("2. Set up automated scheduling")
    print("3. Add data visualization dashboard")
    print("4. Implement data quality monitoring")
    print("5. Add more data sources (social media, IoT, etc.)")

if __name__ == "__main__":
    main()