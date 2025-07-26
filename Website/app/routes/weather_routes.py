"""
Weather Routes Module for AgroVista
Handles weather-related routes and API endpoints
"""

from flask import Blueprint, render_template, request, jsonify
import logging
from datetime import datetime
import sys
import os

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app.services.weather_service import weather_service

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create blueprint
weather_bp = Blueprint('weather', __name__)

@weather_bp.route('/weather')
def weather_page():
    """Weather page"""
    try:
        return render_template('Weather.html')
    except Exception as e:
        logger.error(f"❌ Error in weather route: {e}")
        return "Error loading weather page", 500

@weather_bp.route('/api/weather/current')
def current_weather():
    """Get current weather data"""
    try:
        lat = float(request.args.get('lat', 28.6139))  # Default to Delhi
        lon = float(request.args.get('lon', 77.2090))
        
        weather_data = weather_service.get_current_weather(lat, lon)
        
        return jsonify({
            'success': True,
            'data': weather_data,
            'location': {'lat': lat, 'lon': lon},
            'timestamp': datetime.now().isoformat()
        })
        
    except ValueError:
        return jsonify({
            'success': False,
            'error': 'Invalid coordinates'
        }), 400
        
    except Exception as e:
        logger.error(f"❌ Error getting current weather: {e}")
        return jsonify({
            'success': False,
            'error': 'Weather service unavailable'
        }), 500

@weather_bp.route('/api/weather/forecast')
def weather_forecast():
    """Get weather forecast"""
    try:
        lat = float(request.args.get('lat', 28.6139))
        lon = float(request.args.get('lon', 77.2090))
        days = int(request.args.get('days', 5))
        
        # Limit days to reasonable range
        days = max(1, min(days, 7))
        
        forecast_data = weather_service.get_forecast(lat, lon, days)
        daily_forecast = weather_service.process_daily_forecast(forecast_data)
        
        return jsonify({
            'success': True,
            'data': {
                'hourly': forecast_data,
                'daily': daily_forecast
            },
            'location': {'lat': lat, 'lon': lon},
            'days': days,
            'timestamp': datetime.now().isoformat()
        })
        
    except ValueError:
        return jsonify({
            'success': False,
            'error': 'Invalid parameters'
        }), 400
        
    except Exception as e:
        logger.error(f"❌ Error getting weather forecast: {e}")
        return jsonify({
            'success': False,
            'error': 'Forecast service unavailable'
        }), 500

@weather_bp.route('/api/weather/extended')
def extended_forecast():
    """Get 6-month extended forecast"""
    try:
        lat = float(request.args.get('lat', 28.6139))
        lon = float(request.args.get('lon', 77.2090))
        
        extended_data = weather_service.get_extended_forecast(lat, lon)
        
        return jsonify({
            'success': True,
            'data': extended_data,
            'location': {'lat': lat, 'lon': lon},
            'type': 'seasonal_average',
            'timestamp': datetime.now().isoformat()
        })
        
    except ValueError:
        return jsonify({
            'success': False,
            'error': 'Invalid coordinates'
        }), 400
        
    except Exception as e:
        logger.error(f"❌ Error getting extended forecast: {e}")
        return jsonify({
            'success': False,
            'error': 'Extended forecast service unavailable'
        }), 500

@weather_bp.route('/api/weather/historical')
def historical_weather():
    """Get historical weather data (placeholder)"""
    try:
        # This would implement historical weather data retrieval
        # For now, return a placeholder response
        
        return jsonify({
            'success': True,
            'message': 'Historical weather data not yet implemented',
            'data': [],
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Error getting historical weather: {e}")
        return jsonify({
            'success': False,
            'error': 'Historical weather service unavailable'
        }), 500

@weather_bp.route('/api/weather/alerts')
def weather_alerts():
    """Get weather alerts and warnings"""
    try:
        lat = float(request.args.get('lat', 28.6139))
        lon = float(request.args.get('lon', 77.2090))
        
        # Get current weather to check for alert conditions
        current = weather_service.get_current_weather(lat, lon)
        
        alerts = []
        
        # Simple alert logic (can be enhanced)
        if current.get('temperature', 0) > 40:
            alerts.append({
                'type': 'heat_wave',
                'severity': 'high',
                'message': 'Extreme heat warning - protect crops from heat stress'
            })
        
        if current.get('wind_speed', 0) > 50:
            alerts.append({
                'type': 'high_wind',
                'severity': 'medium',
                'message': 'High wind speeds may affect crop protection'
            })
        
        return jsonify({
            'success': True,
            'alerts': alerts,
            'count': len(alerts),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Error getting weather alerts: {e}")
        return jsonify({
            'success': False,
            'error': 'Weather alerts service unavailable'
        }), 500
