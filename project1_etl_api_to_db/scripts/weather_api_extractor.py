import requests
import pandas as pd
from datetime import datetime
from config.api_config import WEATHER_API_KEY, WEATHER_BASE_URL

def extract_weather_data(city='London'):
    """Extract weather data from OpenWeatherMap API"""
    params = {
        'q': city,
        'appid': WEATHER_API_KEY,
        'units': 'metric'
    }
    
    response = requests.get(WEATHER_BASE_URL, params=params)
    if response.status_code == 200:
        data = response.json()
        return {
            'city': data['name'],
            'temperature': data['main']['temp'],
            'humidity': data['main']['humidity'],
            'pressure': data['main']['pressure'],
            'weather_description': data['weather'][0]['description'],
            'timestamp': datetime.now().isoformat()
        }
    return None