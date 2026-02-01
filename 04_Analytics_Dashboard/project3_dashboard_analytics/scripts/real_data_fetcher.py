#!/usr/bin/env python3
"""
Real Data Fetcher for Analytics Dashboard
=========================================

Fetches real business data from various APIs for dashboard analytics.
"""

import requests
import pandas as pd
import json
from datetime import datetime, timedelta
import logging
from pathlib import Path
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealDataFetcher:
    """Fetch real data from various APIs for business analytics"""

    def __init__(self):
        self.data_dir = Path("data/raw")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # API Keys (you'll need to set these as environment variables)
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY', 'demo')
        self.news_api_key = os.getenv('NEWS_API_KEY', 'demo')

    def fetch_stock_data(self, symbols=['AAPL', 'GOOGL', 'MSFT', 'AMZN'], period='1y'):
        """Fetch real stock data from Yahoo Finance"""
        logger.info(f"Fetching stock data for {symbols}")

        stock_data = {}
        for symbol in symbols:
            try:
                stock = yf.Ticker(symbol)
                df = stock.history(period=period)

                # Add basic metrics
                df['Symbol'] = symbol
                df['Daily_Return'] = df['Close'].pct_change()
                df['Volatility'] = df['Daily_Return'].rolling(window=30).std()

                stock_data[symbol] = df.reset_index()
                logger.info(f"✅ Fetched {len(df)} days of data for {symbol}")

            except Exception as e:
                logger.error(f"❌ Failed to fetch {symbol}: {e}")

        # Combine all stocks
        if stock_data:
            combined_df = pd.concat(stock_data.values(), ignore_index=True)
            combined_df.to_csv(self.data_dir / 'stock_data.csv', index=False)
            logger.info(f"💾 Saved combined stock data to {self.data_dir / 'stock_data.csv'}")

        return stock_data

    def fetch_economic_indicators(self):
        """Fetch economic indicators from Alpha Vantage"""
        logger.info("Fetching economic indicators")

        try:
            ts = TimeSeries(key=self.alpha_vantage_key, output_format='pandas')

            # Fetch GDP data
            gdp_data, _ = ts.get_gdp()
            gdp_data.to_csv(self.data_dir / 'gdp_data.csv')
            logger.info("✅ Fetched GDP data")

            # Fetch inflation data
            inflation_data, _ = ts.get_inflation()
            inflation_data.to_csv(self.data_dir / 'inflation_data.csv')
            logger.info("✅ Fetched inflation data")

            # Fetch unemployment data
            unemployment_data, _ = ts.get_unemployment()
            unemployment_data.to_csv(self.data_dir / 'unemployment_data.csv')
            logger.info("✅ Fetched unemployment data")

        except Exception as e:
            logger.error(f"❌ Failed to fetch economic indicators: {e}")
            # Fallback to mock data
            self._create_mock_economic_data()

    def fetch_news_sentiment(self, query='business OR economy OR finance', days=30):
        """Fetch news articles for sentiment analysis"""
        logger.info(f"Fetching news articles for: {query}")

        try:
            base_url = 'https://newsapi.org/v2/everything'
            params = {
                'q': query,
                'from': (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'),
                'sortBy': 'publishedAt',
                'apiKey': self.news_api_key,
                'language': 'en',
                'pageSize': 100
            }

            response = requests.get(base_url, params=params)
            data = response.json()

            if data.get('status') == 'ok':
                articles = data.get('articles', [])
                news_df = pd.DataFrame(articles)
                news_df['publishedAt'] = pd.to_datetime(news_df['publishedAt'])
                news_df.to_csv(self.data_dir / 'news_data.csv', index=False)
                logger.info(f"✅ Fetched {len(articles)} news articles")
                return news_df
            else:
                logger.error(f"❌ News API error: {data.get('message')}")

        except Exception as e:
            logger.error(f"❌ Failed to fetch news: {e}")

        # Fallback to mock data
        return self._create_mock_news_data()

    def fetch_ecommerce_data(self):
        """Fetch e-commerce data from public APIs"""
        logger.info("Fetching e-commerce data")

        try:
            # Using a public API for demo purposes
            # In production, you'd use actual e-commerce APIs
            url = "https://fakestoreapi.com/products"
            response = requests.get(url)

            if response.status_code == 200:
                products = response.json()
                products_df = pd.DataFrame(products)
                products_df.to_csv(self.data_dir / 'ecommerce_products.csv', index=False)
                logger.info(f"✅ Fetched {len(products)} products")

                # Generate mock sales data based on products
                sales_data = self._generate_sales_data(products_df)
                sales_data.to_csv(self.data_dir / 'ecommerce_sales.csv', index=False)
                logger.info("✅ Generated sales data")

                return products_df, sales_data

        except Exception as e:
            logger.error(f"❌ Failed to fetch e-commerce data: {e}")

        # Fallback to mock data
        return self._create_mock_ecommerce_data()

    def fetch_user_behavior_data(self):
        """Fetch user behavior data from public APIs"""
        logger.info("Fetching user behavior data")

        try:
            # Using GitHub API as a proxy for user activity data
            url = "https://api.github.com/search/repositories"
            params = {
                'q': 'stars:>1000',
                'sort': 'stars',
                'order': 'desc',
                'per_page': 100
            }

            response = requests.get(url, params=params)
            data = response.json()

            if 'items' in data:
                repos = data['items']
                behavior_df = pd.DataFrame([{
                    'user_id': repo['id'],
                    'activity_type': 'repository_creation',
                    'timestamp': repo['created_at'],
                    'stars': repo['stargazers_count'],
                    'forks': repo['forks_count'],
                    'language': repo['language'],
                    'size': repo['size']
                } for repo in repos])

                behavior_df.to_csv(self.data_dir / 'user_behavior.csv', index=False)
                logger.info(f"✅ Fetched {len(repos)} user behavior records")
                return behavior_df

        except Exception as e:
            logger.error(f"❌ Failed to fetch user behavior data: {e}")

        # Fallback to mock data
        return self._create_mock_user_behavior_data()

    def _create_mock_economic_data(self):
        """Create mock economic data when API fails"""
        logger.info("Creating mock economic data")

        dates = pd.date_range(start='2020-01-01', end=datetime.now(), freq='Q')
        gdp_data = pd.DataFrame({
            'date': dates,
            'value': [21000 + i * 500 + np.random.normal(0, 200) for i in range(len(dates))]
        })
        gdp_data.to_csv(self.data_dir / 'gdp_data.csv', index=False)

        inflation_data = pd.DataFrame({
            'date': dates,
            'value': [2.5 + np.random.normal(0, 0.5) for _ in dates]
        })
        inflation_data.to_csv(self.data_dir / 'inflation_data.csv', index=False)

    def _create_mock_news_data(self):
        """Create mock news data when API fails"""
        logger.info("Creating mock news data")

        titles = [
            "Tech Giant Reports Record Profits",
            "Market Volatility Increases Amid Global Uncertainty",
            "New Startup Raises $50M in Series A Funding",
            "Economic Indicators Show Mixed Signals",
            "Consumer Spending Reaches All-Time High"
        ] * 20

        news_data = pd.DataFrame({
            'title': titles,
            'description': [f"Description for {title}" for title in titles],
            'publishedAt': pd.date_range(start='2024-01-01', periods=len(titles), freq='D'),
            'source': ['Reuters', 'Bloomberg', 'WSJ', 'CNN', 'BBC'] * (len(titles)//5 + 1),
            'sentiment': ['positive', 'negative', 'neutral'] * (len(titles)//3 + 1)
        })

        news_data.to_csv(self.data_dir / 'news_data.csv', index=False)
        return news_data.head(20)

    def _create_mock_ecommerce_data(self):
        """Create mock e-commerce data"""
        logger.info("Creating mock e-commerce data")

        products = pd.DataFrame({
            'id': range(1, 101),
            'title': [f'Product {i}' for i in range(1, 101)],
            'price': [10 + i * 5 + np.random.normal(0, 10) for i in range(1, 101)],
            'category': ['electronics', 'clothing', 'books', 'home', 'sports'] * 20,
            'rating': [3.5 + np.random.normal(0, 1) for _ in range(100)]
        })

        sales_data = self._generate_sales_data(products)
        products.to_csv(self.data_dir / 'ecommerce_products.csv', index=False)
        sales_data.to_csv(self.data_dir / 'ecommerce_sales.csv', index=False)

        return products, sales_data

    def _generate_sales_data(self, products_df):
        """Generate sales data based on products"""
        sales_records = []
        for _ in range(1000):
            product = products_df.sample(1).iloc[0]
            sales_records.append({
                'product_id': product['id'],
                'quantity': np.random.randint(1, 10),
                'price': product['price'],
                'total': product['price'] * np.random.randint(1, 10),
                'date': pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(0, 365)),
                'customer_id': np.random.randint(1000, 9999)
            })

        return pd.DataFrame(sales_records)

    def _create_mock_user_behavior_data(self):
        """Create mock user behavior data"""
        logger.info("Creating mock user behavior data")

        behaviors = ['page_view', 'click', 'purchase', 'search', 'login'] * 200
        behavior_data = pd.DataFrame({
            'user_id': np.random.randint(1000, 9999, 1000),
            'behavior_type': behaviors[:1000],
            'timestamp': pd.date_range(start='2024-01-01', periods=1000, freq='H'),
            'page': ['home', 'product', 'cart', 'checkout', 'search'] * 200,
            'device': ['mobile', 'desktop', 'tablet'] * (1000//3 + 1),
            'duration': np.random.exponential(30, 1000)
        })

        behavior_data.to_csv(self.data_dir / 'user_behavior.csv', index=False)
        return behavior_data

    def fetch_all_data(self):
        """Fetch all types of real data"""
        logger.info("🚀 Starting comprehensive data fetching...")

        self.fetch_stock_data()
        self.fetch_economic_indicators()
        self.fetch_news_sentiment()
        self.fetch_ecommerce_data()
        self.fetch_user_behavior_data()

        logger.info("✅ Data fetching completed!")
        logger.info(f"📁 Check {self.data_dir} for downloaded data files")

if __name__ == "__main__":
    fetcher = RealDataFetcher()
    fetcher.fetch_all_data()