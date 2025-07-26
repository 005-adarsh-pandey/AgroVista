"""
Weather-related functions and routes
"""
import base64
import os
import xarray as xr
import pandas as pd
import calendar
from flask import jsonify, request
import warnings
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import logging

# Suppress matplotlib warnings about categorical units - more aggressive approach
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', message='Using categorical units to plot a list of strings')
warnings.filterwarnings('ignore', category=UserWarning, message='.*categorical units.*')

# Also suppress matplotlib logging
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('matplotlib.category').setLevel(logging.ERROR)


def weather_forecast_handler(app):
    """Weather forecast endpoint"""
    print("🌤️ WEATHER FORECAST CHART REQUEST RECEIVED!")
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
    from cds_download import download_forecast_nc
    from plot_nc_file import plot_forecast

    try:
        data = request.get_json()
        city = data.get("city")
        state_code = data.get("state_code")
        lat = float(data.get("lat"))
        lon = float(data.get("lon"))

        if not all([city, state_code, lat, lon]):
            return jsonify({"error": "Missing required fields"}), 400

        nc_file = download_forecast_nc(state_code, city, lat, lon, 
                                       os.path.join(os.path.dirname(__file__), '..', 'weather_data'))
        image_path = plot_forecast(nc_file, city, state_code)

        full_path = os.path.join(app.root_path, image_path)
        if not os.path.exists(full_path):
            return jsonify({"error": "Forecast image not found"}), 404

        with open(full_path, "rb") as img_file:
            encoded_img = base64.b64encode(img_file.read()).decode('utf-8')

        return jsonify({"image": f"data:image/png;base64,{encoded_img}"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def get_weather_data_handler(app):
    """Pre-fetch weather data for a location using CDS API"""
    data = request.json
    city = data.get('city')
    state = data.get('state')
    latitude = data.get('latitude')
    longitude = data.get('longitude')
    
    if not city or not state:
        return jsonify({'error': 'City and state required'}), 400
    
    if not latitude or not longitude:
        return jsonify({'error': 'Coordinates required'}), 400
    
    try:
        # Import weather functions
        from .enhanced_ml_inputs import get_weather_data_for_prediction
        
        # print(f"🌤️ PRE-FETCHING weather data for {city}, {state}")
        # print(f"   📍 Coordinates: ({latitude}, {longitude})")
        
        # Pre-fetch weather data using the same function as prediction
        weather_data = get_weather_data_for_prediction(
            state_name=state.upper(),
            district_name=city.upper(),  # Use city as district for consistency
            lat=float(latitude),
            lon=float(longitude),
            city_name=city
        )
        
        if weather_data:
            # print(f"✅ Pre-fetched weather data successfully!")
            # print(f"   📂 File: {weather_data.get('file_used', 'N/A')}")
            # print(f"   🌧️ Total rainfall: {weather_data.get('total_rainfall', 0):.1f} mm")
            
            # Return summary data for UI
            return jsonify({
                'success': True,
                'location': f"{city}, {state}",
                'temperature': 28.5,  # Mock temperature for UI
                'humidity': 65,       # Mock humidity for UI
                'pressure': 1013.2,   # Mock pressure for UI
                'rainfall_forecast': weather_data.get('total_rainfall', 0),
                'kharif_rainfall': weather_data.get('kharif_rainfall', 0),
                'rabi_rainfall': weather_data.get('rabi_rainfall', 0),
                'zaid_rainfall': weather_data.get('zaid_rainfall', 0),
                'data_source': 'CDS API',
                'file_cached': weather_data.get('file_used', 'N/A'),
                'reference_time': weather_data.get('reference_time', 'N/A')
            })
        else:
            # print("⚠️ Failed to pre-fetch weather data")
            return jsonify({
                'success': False,
                'location': f"{city}, {state}",
                'error': 'Failed to fetch weather data',
                'temperature': 28.5,  # Mock fallback for UI
                'humidity': 65,
                'pressure': 1013.2,
                'rainfall_forecast': 0,
                'data_source': 'Fallback'
            })
            
    except Exception as e:
        print(f"❌ Error pre-fetching weather data: {e}")
        return jsonify({
            'success': False,
            'location': f"{city}, {state}",
            'error': str(e),
            'temperature': 28.5,  # Mock fallback for UI
            'humidity': 65,
            'pressure': 1013.2,
            'rainfall_forecast': 0,
            'data_source': 'Error fallback'
        })


def get_previous_rainfall_handler(app):
    """Get previous rainfall data for a location (mock implementation)"""
    data = request.json
    district = data.get('district')
    state = data.get('state')
    year = data.get('year', 2024)
    
    if not district or not state:
        return jsonify({'error': 'District and state required'}), 400
    
    try:
        # Mock rainfall data - replace with actual API integration
        prev_rainfall = {
            'annual_total': 1250.7,    # Total annual rainfall
            'kharif_total': 850.5,     # Total kharif season rainfall
            'monsoon_avg': 142.3,      # Average monthly monsoon rainfall
            'year': year
        }
        
        return jsonify(prev_rainfall)
        
    except Exception as e:
        return jsonify({'error': f'Previous rainfall data fetch failed: {str(e)}'}), 500


def weather_cards_data_handler(app):
    """Extract monthly weather data from .nc files for weather cards display"""
    print("🌤️ WEATHER CARDS SUMMARY REQUEST RECEIVED!")
    
    try:
        data = request.get_json()
        city = data.get("city")
        state_code = data.get("state_code")
        lat = float(data.get("lat"))
        lon = float(data.get("lon"))

        if not all([city, state_code, lat, lon]):
            return jsonify({"error": "Missing required fields"}), 400

        # Use the same download logic as prediction system
        import sys
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
        from cds_download import download_forecast_nc
        
        # Use the same weather data folder as the original system - avoid duplication
        # This ensures all weather data is stored in one location: Website/weather_data/
        data_folder = os.path.join(os.path.dirname(__file__), '..', 'weather_data')
        
        # Ensure location naming is consistent with prediction system
        location_clean = city.lower().replace(' ', '_')
        
        # Check if file already exists in the original data folder
        expected_filename = f"{state_code.lower()}_{location_clean}.nc"
        expected_filepath = os.path.join(data_folder, expected_filename)
        
        # Debug messages disabled for cleaner logs
        print(f"🔍 Looking for cached file: {expected_filename}")
        
        if os.path.exists(expected_filepath) and os.path.getsize(expected_filepath) > 1000:
            print(f"🚀 Using cached weather data: {expected_filename}")
            nc_file_path = expected_filepath
        else:
            print(f"⏳ Downloading weather data for weather cards...")
            # Download/get the .nc file from the same location as prediction system
            nc_file_path = download_forecast_nc(state_code, location_clean, lat, lon, data_folder)
        
        # Extract weather data using same logic as plot_nc_file.py
        ds = xr.open_dataset(nc_file_path)
        
        # Unit conversions (same as plot function)
        ds["t2m_C"] = ds["t2m"] - 273.15
        ds["precip_mm_month"] = ds["tprate"] * 30 * 24 * 60 * 60 * 1000
        ds["solar_mj_month"] = ds["msnsrf"] * 30 * 24 * 60 * 60 / 1_000_000
        
        # Use center coordinates (same as plot_nc_file.py)
        lat_center = float(ds.latitude.values.mean())
        lon_center = float(ds.longitude.values.mean())
        
        # Extract monthly values (same logic as plot_nc_file.py)
        temp_vals = ds["t2m_C"].sel(latitude=lat_center, longitude=lon_center, method="nearest").mean(dim="forecast_reference_time").values.flatten()
        rain_vals = ds["precip_mm_month"].sel(latitude=lat_center, longitude=lon_center, method="nearest").mean(dim="forecast_reference_time").values.flatten()
        solar_vals = ds["solar_mj_month"].sel(latitude=lat_center, longitude=lon_center, method="nearest").mean(dim="forecast_reference_time").values.flatten()
        
        # Get forecast months
        base_time = pd.to_datetime(ds['forecast_reference_time'].values[0])
        forecast_months = ds["forecastMonth"].values
        actual_months = [(base_time + pd.DateOffset(months=int(m - 1))).month for m in forecast_months]
        month_names = [calendar.month_name[m] for m in actual_months]
        
        # Format data for weather cards
        weather_cards = []
        for i, (month_name, temp, rainfall, solar) in enumerate(zip(month_names, temp_vals, rain_vals, solar_vals)):
            
            # Handle NaN values
            if pd.isna(temp) or temp is None:
                temp = 0.0
            if pd.isna(rainfall) or rainfall is None:
                rainfall = 0.0
            if pd.isna(solar) or solar is None:
                solar = 0.0
            # Generate weather conditions based on rainfall
            if rainfall > 150:
                condition = "Heavy Rain"
                icon = "🌧️"
            elif rainfall > 80:
                condition = "Moderate Rain"
                icon = "🌦️"
            elif rainfall > 30:
                condition = "Light Rain"
                icon = "🌤️"
            elif rainfall > 10:
                condition = "Partly Cloudy"
                icon = "⛅"
            else:
                condition = "Clear"
                icon = "☀️"
            
            weather_cards.append({
                "month": month_name,
                "temperature": round(float(temp), 1) if not pd.isna(temp) else 0.0,
                "rainfall": round(float(rainfall), 1) if not pd.isna(rainfall) else 0.0,
                "solar_radiation": round(float(solar), 1) if not pd.isna(solar) else 0.0,
                "condition": condition,
                "icon": icon
            })
        
        return jsonify({
            "success": True,
            "weather_cards": weather_cards,
            "reference_time": base_time.strftime('%Y-%m-%d'),
            "location": f"{city}, {state_code}",
            "coordinates": f"({lat}, {lon})",
            "data_source": "CDS API"
        })

    except Exception as e:
        print(f"❌ Error getting weather cards data: {e}")
        return jsonify({"error": str(e)}), 500
