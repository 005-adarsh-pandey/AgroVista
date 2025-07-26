"""
Enhanced ML input preparation for the crop prediction system.
This module provides improved rainfall data handling with CDS API integration
and fallback to historical averages from the main dataset.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os
import json


def get_season_for_date(date=None):
    """Determine the crop season based on the date"""
    if date is None:
        date = datetime.now()
    
    month = date.month
    if month in [4, 5, 6, 7, 8, 9]:  # Apr-Sep
        return 'Kharif'
    elif month in [10, 11, 12, 1, 2, 3]:  # Oct-Mar
        return 'Rabi'
    else:
        return 'Kharif'  # Default fallback

def get_weather_data_for_prediction(state_name, district_name, lat, lon, city_name=None):
    """
    Get weather data for prediction using the same logic as weather page.
    Downloads data if not exists, then extracts rainfall information.
    Uses actual city coordinates and name instead of district.
    """
    try:
        # Import the existing utilities
        sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
        from cds_download import download_forecast_nc
        import xarray as xr
        import pandas as pd
        import calendar
        
        # Map state names to state codes (same as weather.html)
        state_codes = {
            'ANDHRA PRADESH': '28', 'ASSAM': '18', 'BIHAR': '10', 'CHANDIGARH': '04',
            'CHHATTISGARH': '22', 'DELHI': '07', 'GOA': '30', 'GUJARAT': '24',
            'HARYANA': '06', 'HIMACHAL PRADESH': '02', 'JAMMU AND KASHMIR': '01',
            'JHARKHAND': '20', 'KARNATAKA': '29', 'KERALA': '32', 'MADHYA PRADESH': '23',
            'MAHARASHTRA': '27', 'MANIPUR': '14', 'MEGHALAYA': '17', 'MIZORAM': '15',
            'NAGALAND': '13', 'ODISHA': '21', 'PUNJAB': '03', 'RAJASTHAN': '08',
            'SIKKIM': '11', 'TAMIL NADU': '33', 'TELANGANA': '36', 'TRIPURA': '16',
            'UTTAR PRADESH': '09', 'UTTARAKHAND': '05', 'WEST BENGAL': '19',
            'ANDAMAN AND NICOBAR ISLANDS': '35', 'DADRA AND NAGAR HAVELI': '26',
            'DAMAN AND DIU': '25', 'LAKSHADWEEP': '31', 'PUDUCHERRY': '34'
        }
        
        # Get state code
        state_code = state_codes.get(state_name.upper())
        if not state_code:
            print(f"❌ State code not found for {state_name}")
            return None
        
        # If coordinates are None, try to get them from india_places.json (same as weather.html)
        if lat is None or lon is None:
            print(f"🔍 Coordinates not provided, looking up in india_places.json")
            print(f"   Looking for city: {city_name}, state: {state_name}")
            
            # Load india_places.json to get city coordinates
            try:
                india_places_path = os.path.join(os.path.dirname(__file__), 'static', 'india_places.json')
                with open(india_places_path, 'r', encoding='utf-8') as f:
                    places_data = json.load(f)
                
                # Search for city in the state (same logic as weather.html)
                city_found = False
                for place in places_data:
                    if (place.get('state', '').upper() == state_name.upper() and 
                        place.get('city', '').upper() == city_name.upper()):
                        lat = float(place['lat'])
                        lon = float(place['lon'])
                        city_found = True
                        print(f"✅ Found city coordinates: {city_name} -> ({lat}, {lon})")
                        break
                
                if not city_found:
                    print(f"❌ City {city_name} not found in {state_name}")
                    return None
                    
            except Exception as e:
                print(f"❌ Error loading india_places.json: {e}")
                return None
        
        # Use city name if provided, otherwise use district name
        location_name = city_name if city_name else district_name
        location_clean = location_name.lower().replace(' ', '_')
        
        print(f"🌤️ Getting weather data for {location_name}, {state_name}")
        print(f"   🎯 Using city coordinates: ({lat}, {lon})")
        print(f"   📊 Model will use district: {district_name}")
        print(f"   💡 This gives more accurate weather data while keeping model compatibility")
        
        # Use existing CDS download function (same as weather page)
        data_folder = os.path.join(os.path.dirname(__file__), 'weather_data')
        
        # Check if file already exists (optimization for pre-fetched data)
        expected_filename = f"{state_code.lower()}_{location_clean}.nc"
        expected_filepath = os.path.join(data_folder, expected_filename)
        was_cached = False
        
        if os.path.exists(expected_filepath) and os.path.getsize(expected_filepath) > 1000:
            print(f"🚀 Using CACHED weather data: {expected_filename}")
            nc_file_path = expected_filepath
            was_cached = True
        else:
            print(f"⏳ Downloading weather data for the first time...")
            nc_file_path = download_forecast_nc(state_code, location_clean, lat, lon, data_folder)
            was_cached = False
        
        # Extract rainfall data using same logic as plot_nc_file.py
        ds = xr.open_dataset(nc_file_path)
        
        # Unit conversion (same as plot function)
        ds["precip_mm_month"] = ds["tprate"] * 30 * 24 * 60 * 60 * 1000
        
        # Use center coordinates
        lat_center = float(ds.latitude.values.mean())
        lon_center = float(ds.longitude.values.mean())
        
        # Extract monthly rainfall values
        rain_vals = ds["precip_mm_month"].sel(
            latitude=lat_center, 
            longitude=lon_center, 
            method="nearest"
        ).mean(dim="forecast_reference_time").values.flatten()
        
        # Get forecast months (same logic as plot function)
        base_time = pd.to_datetime(ds['forecast_reference_time'].values[0])
        forecast_months = ds["forecastMonth"].values
        actual_months = [(base_time + pd.DateOffset(months=int(m - 1))).month for m in forecast_months]
        
        # Calculate seasonal totals
        kharif_total = 0  # June-September (6,7,8,9)
        rabi_total = 0    # October-March (10,11,12,1,2,3)
        zaid_total = 0    # April-May (4,5)
        
        monthly_data = {}
        for i, (month, rainfall) in enumerate(zip(actual_months, rain_vals)):
            monthly_data[month] = rainfall
            
            if month in [6, 7, 8, 9]:  # Kharif
                kharif_total += rainfall
            elif month in [10, 11, 12, 1, 2, 3]:  # Rabi
                rabi_total += rainfall
            elif month in [4, 5]:  # Zaid
                zaid_total += rainfall
        
        result = {
            'kharif_rainfall': float(kharif_total),
            'rabi_rainfall': float(rabi_total),
            'zaid_rainfall': float(zaid_total),
            'total_rainfall': float(kharif_total + rabi_total + zaid_total),
            'monthly_data': {k: float(v) for k, v in monthly_data.items()},
            'source': 'CDS_DOWNLOAD',
            'reference_time': base_time.strftime('%Y-%m-%d'),
            'file_used': os.path.basename(nc_file_path),
            'location_used': location_name,
            'coordinates_used': f"({lat}, {lon})"
        }
        
        print(f"✅ Weather data ready from CDS API:")
        cache_status = "🚀 CACHED (instant)" if was_cached else "⏳ DOWNLOADED (new)"
        print(f"   📁 Status: {cache_status}")
        print(f"   🌍 Weather Location: {location_name} (city-level coordinates)")
        print(f"   📊 Model Input: {district_name} (district-level for ML model)")
        print(f"   📂 File: {os.path.basename(nc_file_path)}")
        print(f"   📅 Reference: {base_time.strftime('%Y-%m-%d')}")
        print(f"   🎯 Coordinates: ({lat}, {lon})")
        print(f"   🌧️ Kharif: {kharif_total:.1f} mm")
        print(f"   🌧️ Rabi: {rabi_total:.1f} mm")
        print(f"   🌧️ Zaid: {zaid_total:.1f} mm")
        print(f"   🌧️ Total: {result['total_rainfall']:.1f} mm")
        
        return result
        
    except Exception as e:
        print(f"❌ Error getting weather data: {e}")
        return None
    """
    Get seasonal forecast data from CDS API based on current month and season.
    Uses intelligent seasonal mapping for better accuracy.
    """
    try:
        current_month = datetime.now().month
        
        # Determine if we should get current season data
        if season.upper() == 'KHARIF':
            # For Kharif (Jun-Oct), get data if we're in the season or just before
            if current_month >= 6 and current_month <= 10:
                # We're in Kharif season - get current data
                print(f"CDS API: Getting current Kharif season data for ({lat}, {lon})")
                # TODO: Implement actual CDS API call for current season
                return get_cds_current_season_data(lat, lon, 'kharif')
            elif current_month >= 4 and current_month <= 5:
                # Pre-Kharif season - get forecast
                print(f"CDS API: Getting Kharif forecast for ({lat}, {lon})")
                return get_cds_seasonal_forecast(lat, lon, 'kharif')
        
        elif season.upper() == 'RABI':
            # For Rabi (Nov-Mar), get data if we're in the season or just before
            if current_month >= 11 or current_month <= 3:
                # We're in Rabi season - get current data
                print(f"CDS API: Getting current Rabi season data for ({lat}, {lon})")
                return get_cds_current_season_data(lat, lon, 'rabi')
            elif current_month >= 9 and current_month <= 10:
                # Pre-Rabi season - get forecast
                print(f"CDS API: Getting Rabi forecast for ({lat}, {lon})")
                return get_cds_seasonal_forecast(lat, lon, 'rabi')
        
        # Fall back to historical averages if not in season
        print(f"CDS API: Season {season} not active, using historical averages")
        return None
        
    except Exception as e:
        print(f"CDS API error: {e}")
        return None

def get_rainfall_averages_from_dataset(state=None, district=None):
    """Get historical rainfall averages with fallback hierarchy: district -> state -> country"""
    
    # Use the correct dataset file path relative to the web app
    possible_paths = [
        os.path.join(os.path.dirname(__file__), '..', 'Models', 'main_dataset_final_cleaned.csv'),
        os.path.join(os.path.dirname(__file__), '..', 'Final_Data', 'final_rainfall_data_combined.csv'),
        os.path.join(os.path.dirname(__file__), '..', 'Final_Data', 'main_dataset_final_cleaned.csv'),
        r"d:\Git Hub\Maths Assignment\Models\main_dataset_final_cleaned.csv"
    ]
    
    dataset_path = None
    for path in possible_paths:
        if os.path.exists(path):
            dataset_path = path
            break
    
    if not dataset_path:
        print("Warning: Cannot find rainfall dataset, using default values")
        return {
            'Rainfall - Kharif (mm)': 850,
            'Rainfall - Rabi (mm)': 180,
            'Rainfall - Zaid (mm)': 80,
            'Rainfall - Whole Year (mm)': 1100,
            'level': 'default'
        }
    
    try:
        df = pd.read_csv(dataset_path)
        rainfall_cols = ['Rainfall - Kharif (mm)', 'Rainfall - Rabi (mm)', 
                        'Rainfall - Zaid (mm)', 'Rainfall - Whole Year (mm)']
        
        # Check if required columns exist
        missing_cols = [col for col in rainfall_cols if col not in df.columns]
        if missing_cols:
            print(f"Warning: Missing columns {missing_cols}, using defaults")
            return {
                'Rainfall - Kharif (mm)': 850,
                'Rainfall - Rabi (mm)': 180,
                'Rainfall - Zaid (mm)': 80,
                'Rainfall - Whole Year (mm)': 1100,
                'level': 'default'
            }
        
        df_clean = df.dropna(subset=rainfall_cols)
        
        # Try district-level first
        if state and district:
            district_data = df_clean[
                (df_clean['State_Name'].str.upper() == state.upper()) & 
                (df_clean['District_Name'].str.upper() == district.upper())
            ]
            
            if not district_data.empty:
                result = district_data[rainfall_cols].mean().to_dict()
                result['level'] = 'district'
                return result
        
        # Try state-level fallback
        if state:
            state_data = df_clean[df_clean['State_Name'].str.upper() == state.upper()]
            
            if not state_data.empty:
                result = state_data[rainfall_cols].mean().to_dict()
                result['level'] = 'state'
                return result
        
        # Country-level fallback
        result = df_clean[rainfall_cols].mean().to_dict()
        result['level'] = 'country'
        return result
        
    except Exception as e:
        print(f"Error reading rainfall dataset: {e}")
        return {
            'Rainfall - Kharif (mm)': 850,
            'Rainfall - Rabi (mm)': 180,
            'Rainfall - Zaid (mm)': 80,
            'Rainfall - Whole Year (mm)': 1100,
            'level': 'default'
        }

def prepare_enhanced_ml_inputs(data, lat=None, lon=None):
    """
    Prepare enhanced ML model inputs with improved rainfall handling.
    
    Args:
        data: Request data from Flask
        lat: Latitude (optional)
        lon: Longitude (optional)
    
    Returns:
        Dictionary with enhanced ML model inputs
    """
    
    # Basic inputs
    state_name = data.get('state_name', '').upper()
    district = data.get('district', '').upper()
    crop = data.get('crop', 'Rice').upper()
    season = data.get('season', 'Rabi').upper()
    crop_year = int(data.get('crop_year', datetime.now().year))
    
    # Area handling - frontend already handles conversion, use directly
    area_hectares = float(data.get('area', 1.0))
    print(f"Using area directly from frontend: {area_hectares:.4f} hectares")
    
    # Determine current season if not provided
    if not season or season == 'AUTO':
        season = get_season_for_date().upper()
    
    print(f"Preparing enhanced inputs for {crop} in {district}, {state_name}")
    print(f"Season: {season}, Area: {area_hectares:.4f} hectares")
    
    # Enhanced rainfall handling
    current_season = get_season_for_date()
    
    # Try to get CDS forecast for current season if coordinates are available
    forecast_success = {}
    rainfall_inputs = {
        'Rainfall - Kharif (mm)': None,
        'Rainfall - Rabi (mm)': None,
        'Rainfall - Zaid (mm)': None,
        'Rainfall - Whole Year (mm)': None
    }
    
    # Extract city name from request data if available
    city_name = data.get('city_name', data.get('city', ''))
    print(f"🔍 City name from request: '{city_name}'")
    print(f"🔍 Coordinates from request: lat={lat}, lon={lon}")
    
    # Try to get weather data using existing CDS download infrastructure
    # Use city coordinates for accurate weather data, but keep district for model
    nc_weather_data = get_weather_data_for_prediction(state_name, district, lat, lon, city_name)
    
    if nc_weather_data:
        # Use real weather data from .nc files
        rainfall_inputs['Rainfall - Kharif (mm)'] = nc_weather_data['kharif_rainfall']
        rainfall_inputs['Rainfall - Rabi (mm)'] = nc_weather_data['rabi_rainfall']
        rainfall_inputs['Rainfall - Zaid (mm)'] = nc_weather_data['zaid_rainfall']
        rainfall_inputs['Rainfall - Whole Year (mm)'] = nc_weather_data['total_rainfall']
        
        print(f"🎉 Using REAL weather data from {nc_weather_data['file_used']}")
        print(f"   📅 Reference time: {nc_weather_data['reference_time']}")
        print(f"   🌍 Weather source: {nc_weather_data.get('location_used', 'City coordinates')}")
        print(f"   📊 Model input: District-level ({district})")
        
        # Set source info
        weather_data_source = f"CDS .nc file: {nc_weather_data['file_used']}"
        
    else:
        # Fall back to historical averages
        print("⚠️ No .nc file found, using historical averages")
        
        # Get historical averages for any missing values
        historical_averages = get_rainfall_averages_from_dataset(state_name, district)
        
        # Fill missing values with historical averages
        for column_name in rainfall_inputs:
            if rainfall_inputs[column_name] is None:
                rainfall_inputs[column_name] = historical_averages[column_name]
        
        weather_data_source = f"Historical averages ({historical_averages['level']} level)"
    
    # Validate and correct unrealistic seasonal rainfall patterns
    # Expected hierarchy: Kharif > Rabi > Zaid
    kharif_val = rainfall_inputs['Rainfall - Kharif (mm)']
    rabi_val = rainfall_inputs['Rainfall - Rabi (mm)']
    zaid_val = rainfall_inputs['Rainfall - Zaid (mm)']
    
    # Check for unrealistic patterns and apply corrections
    if zaid_val > rabi_val or zaid_val > kharif_val:
        print(f"⚠️ Unrealistic rainfall pattern detected: Kharif={kharif_val:.1f}, Rabi={rabi_val:.1f}, Zaid={zaid_val:.1f}")
        
        # Apply realistic seasonal distribution (as percentages of total)
        total_rainfall = kharif_val + rabi_val + zaid_val
        
        # Typical Indian seasonal distribution: Kharif=70%, Rabi=20%, Zaid=10%
        corrected_kharif = total_rainfall * 0.70
        corrected_rabi = total_rainfall * 0.20  
        corrected_zaid = total_rainfall * 0.10
        
        rainfall_inputs['Rainfall - Kharif (mm)'] = corrected_kharif
        rainfall_inputs['Rainfall - Rabi (mm)'] = corrected_rabi
        rainfall_inputs['Rainfall - Zaid (mm)'] = corrected_zaid
        
        print(f"✅ Corrected to realistic pattern: Kharif={corrected_kharif:.1f}, Rabi={corrected_rabi:.1f}, Zaid={corrected_zaid:.1f}")
    
    # Handle legacy rainfall data format if provided
    try:
        legacy_rainfall_data = json.loads(data.get('rainfall_data', '{}'))
        if legacy_rainfall_data:
            print("Processing legacy rainfall data format")
            # If legacy data is provided, use it but apply our seasonal logic
            kharif_rainfall = legacy_rainfall_data.get('jun_sep', 0) + legacy_rainfall_data.get('oct', 0)
            rabi_rainfall = (legacy_rainfall_data.get('nov_dec', 0) + 
                           legacy_rainfall_data.get('jan_feb', 0) + 
                           legacy_rainfall_data.get('mar', 0))
            zaid_rainfall = legacy_rainfall_data.get('apr', 0) + legacy_rainfall_data.get('may', 0)
            annual_rainfall = legacy_rainfall_data.get('annual_total', 0)
            
            # Use legacy data if it seems valid (non-zero)
            if kharif_rainfall > 0:
                rainfall_inputs['Rainfall - Kharif (mm)'] = kharif_rainfall
            if rabi_rainfall > 0:
                rainfall_inputs['Rainfall - Rabi (mm)'] = rabi_rainfall
            if zaid_rainfall > 0:
                rainfall_inputs['Rainfall - Zaid (mm)'] = zaid_rainfall
            if annual_rainfall > 0:
                rainfall_inputs['Rainfall - Whole Year (mm)'] = annual_rainfall
            
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Legacy rainfall data parsing error: {e}")
    
    # Recalculate whole year rainfall
    rainfall_inputs['Rainfall - Whole Year (mm)'] = (
        rainfall_inputs['Rainfall - Kharif (mm)'] +
        rainfall_inputs['Rainfall - Rabi (mm)'] +
        rainfall_inputs['Rainfall - Zaid (mm)']
    )
    
    # Prepare final ML inputs
    ml_inputs = {
        'State_Name': state_name,
        'District_Name': district,
        'Crop_Year': crop_year,
        'Season': season,
        'Crop': crop,
        'Area (Hectares)': float(area_hectares),
        'Rainfall - Kharif (mm)': float(rainfall_inputs['Rainfall - Kharif (mm)']),
        'Rainfall - Rabi (mm)': float(rainfall_inputs['Rainfall - Rabi (mm)']),
        'Rainfall - Zaid (mm)': float(rainfall_inputs['Rainfall - Zaid (mm)']),
        'Rainfall - Whole Year (mm)': float(rainfall_inputs['Rainfall - Whole Year (mm)'])
    }
    
    # Add coordinates if available
    if lat and lon:
        ml_inputs['Latitude'] = float(lat)
        ml_inputs['Longitude'] = float(lon)
    
    # Print summary
    print(f"\n📊 Enhanced ML Input Summary:")
    print(f"   🌍 Weather Data Source: {weather_data_source}")
    print(f"   📍 Model Location: {district}, {state_name} (district-level)")
    if nc_weather_data:
        print(f"   🎯 Weather Coordinates: {nc_weather_data.get('coordinates_used', 'N/A')}")
    print(f"   🌧️ Rainfall Data:")
    for season_name in ['Kharif', 'Rabi', 'Zaid']:
        value = ml_inputs[f'Rainfall - {season_name} (mm)']
        source = "Real CDS Data" if nc_weather_data else "Historical Average"
        print(f"      {season_name}: {value:.2f} mm ({source})")
    print(f"      Whole Year: {ml_inputs['Rainfall - Whole Year (mm)']:.2f} mm")
    
    return ml_inputs
