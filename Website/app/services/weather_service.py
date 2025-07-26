"""
Weather Service Module for AgroVista
Handles weather API integration and forecast processing
"""

import requests
import json
import logging
from datetime import datetime, timedelta
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WeatherService:
    def __init__(self):
        self.api_key = "b5e34b8e7d89c0c5cf3b7d5e8e53f9b7"  # OpenWeatherMap API key
        self.base_url = "https://api.openweathermap.org/data/2.5"
        self.onecall_url = "https://api.openweathermap.org/data/3.0/onecall"
        self.cache = {}
        self.cache_duration = 1800  # 30 minutes
    
    def get_current_weather(self, lat, lon):
        """Get current weather conditions"""
        try:
            cache_key = f"current_{lat}_{lon}"
            
            # Check cache
            if self.is_cached(cache_key):
                return self.cache[cache_key]['data']
            
            url = f"{self.base_url}/weather"
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.api_key,
                'units': 'metric'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            weather_data = {
                'temperature': data['main']['temp'],
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'wind_speed': data['wind']['speed'],
                'description': data['weather'][0]['description'],
                'icon': data['weather'][0]['icon']
            }
            
            # Cache the result
            self.cache[cache_key] = {
                'data': weather_data,
                'timestamp': time.time()
            }
            
            logger.info(f"✅ Current weather fetched for ({lat}, {lon})")
            return weather_data
            
        except Exception as e:
            logger.error(f"❌ Error fetching current weather: {e}")
            return self.get_fallback_weather()
    
    def get_forecast(self, lat, lon, days=7):
        """Get weather forecast"""
        try:
            cache_key = f"forecast_{lat}_{lon}_{days}"
            
            # Check cache
            if self.is_cached(cache_key):
                return self.cache[cache_key]['data']
            
            url = f"{self.base_url}/forecast"
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.api_key,
                'units': 'metric',
                'cnt': min(days * 8, 40)  # 8 forecasts per day, max 40
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            forecast_data = []
            
            # Process forecast data
            for item in data['list']:
                forecast_data.append({
                    'datetime': item['dt_txt'],
                    'temperature': item['main']['temp'],
                    'humidity': item['main']['humidity'],
                    'pressure': item['main']['pressure'],
                    'wind_speed': item['wind']['speed'],
                    'description': item['weather'][0]['description'],
                    'icon': item['weather'][0]['icon'],
                    'rain': item.get('rain', {}).get('3h', 0)
                })
            
            # Cache the result
            self.cache[cache_key] = {
                'data': forecast_data,
                'timestamp': time.time()
            }
            
            logger.info(f"✅ Forecast fetched for ({lat}, {lon})")
            return forecast_data
            
        except Exception as e:
            logger.error(f"❌ Error fetching forecast: {e}")
            return self.get_fallback_forecast()
    
    def get_extended_forecast(self, lat, lon):
        """Get 6-month extended forecast (simplified)"""
        try:
            # For now, generate a simplified 6-month forecast
            # In a production system, this would use seasonal forecasting APIs
            
            current_date = datetime.now()
            forecast_data = []
            
            for i in range(6):
                month_date = current_date + timedelta(days=30*i)
                
                # Generate seasonal averages (simplified)
                month = month_date.month
                if month in [6, 7, 8, 9]:  # Monsoon
                    temp_avg = 28
                    humidity_avg = 75
                    rainfall_avg = 150
                elif month in [12, 1, 2]:  # Winter
                    temp_avg = 20
                    humidity_avg = 60
                    rainfall_avg = 20
                else:  # Other months
                    temp_avg = 32
                    humidity_avg = 55
                    rainfall_avg = 50
                
                forecast_data.append({
                    'month': month_date.strftime('%B %Y'),
                    'temperature': temp_avg,
                    'humidity': humidity_avg,
                    'rainfall': rainfall_avg,
                    'description': 'Seasonal average'
                })
            
            logger.info(f"✅ Extended forecast generated for ({lat}, {lon})")
            return forecast_data
            
        except Exception as e:
            logger.error(f"❌ Error generating extended forecast: {e}")
            return []
    
    def process_daily_forecast(self, forecast_data):
        """Process hourly forecast into daily summaries"""
        try:
            daily_forecast = {}
            
            for item in forecast_data:
                date = item['datetime'].split(' ')[0]
                
                if date not in daily_forecast:
                    daily_forecast[date] = {
                        'temperatures': [],
                        'humidity': [],
                        'rainfall': 0,
                        'descriptions': []
                    }
                
                daily_forecast[date]['temperatures'].append(item['temperature'])
                daily_forecast[date]['humidity'].append(item['humidity'])
                daily_forecast[date]['rainfall'] += item['rain']
                daily_forecast[date]['descriptions'].append(item['description'])
            
            # Calculate daily averages
            processed_forecast = []
            for date, data in daily_forecast.items():
                processed_forecast.append({
                    'date': date,
                    'temp_max': max(data['temperatures']),
                    'temp_min': min(data['temperatures']),
                    'temp_avg': sum(data['temperatures']) / len(data['temperatures']),
                    'humidity_avg': sum(data['humidity']) / len(data['humidity']),
                    'rainfall_total': data['rainfall'],
                    'description': max(set(data['descriptions']), key=data['descriptions'].count)
                })
            
            return processed_forecast
            
        except Exception as e:
            logger.error(f"❌ Error processing daily forecast: {e}")
            return []
    
    def is_cached(self, cache_key):
        """Check if data is cached and still valid"""
        if cache_key not in self.cache:
            return False
        
        cached_time = self.cache[cache_key]['timestamp']
        return (time.time() - cached_time) < self.cache_duration
    
    def get_fallback_weather(self):
        """Fallback weather data when API fails"""
        return {
            'temperature': 25.0,
            'humidity': 65,
            'pressure': 1013,
            'wind_speed': 5.0,
            'description': 'Data unavailable',
            'icon': '01d'
        }
    
    def get_fallback_forecast(self):
        """Fallback forecast data when API fails"""
        forecast = []
        for i in range(5):
            date = (datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d %H:%M:%S')
            forecast.append({
                'datetime': date,
                'temperature': 25.0,
                'humidity': 65,
                'pressure': 1013,
                'wind_speed': 5.0,
                'description': 'Data unavailable',
                'icon': '01d',
                'rain': 0
            })
        return forecast

# Global instance
weather_service = WeatherService()
