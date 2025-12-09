import requests
import pandas as pd
from datetime import datetime
from config.api_config import FINANCIAL_API_KEY, FINANCIAL_BASE_URL

def extract_financial_data(symbol='IBM'):
    """Extract financial data from Alpha Vantage API"""
    params = {
        'function': 'TIME_SERIES_DAILY',
        'symbol': symbol,
        'apikey': FINANCIAL_API_KEY
    }
    
    response = requests.get(FINANCIAL_BASE_URL, params=params)
    if response.status_code == 200:
        data = response.json()
        if 'Time Series (Daily)' in data:
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
                'timestamp': datetime.now().isoformat()
            }
    return None