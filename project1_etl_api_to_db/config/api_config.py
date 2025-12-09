import os
from dotenv import load_dotenv

load_dotenv()

# Weather API (OpenWeatherMap)
WEATHER_API_KEY = os.getenv('WEATHER_API_KEY')
WEATHER_BASE_URL = 'http://api.openweathermap.org/data/2.5/weather'

# Financial API (Alpha Vantage)
FINANCIAL_API_KEY = os.getenv('FINANCIAL_API_KEY')
FINANCIAL_BASE_URL = 'https://www.alphavantage.co/query'