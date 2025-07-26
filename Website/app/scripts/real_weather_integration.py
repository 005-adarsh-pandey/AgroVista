"""
Real weather data integration for crop prediction.
Uses existing .nc files from CDS API downloads.
"""

import os
import sys
import xarray as xr
import pandas as pd
import calendar
from datetime import datetime

def get_nc_file_for_location(state_code, city):
    """Find the appropriate .nc file for a given location"""
    # Convert to the naming convention used in your data folder
    filename = f"{state_code.lower()}_{city.lower().replace(' ', '_')}.nc"
    filepath = os.path.join("data", filename)
    
    if os.path.exists(filepath):
        return filepath
    
    # Try some common variations
    variations = [
        f"{state_code.lower()}_{city.lower().replace(' ', '')}.nc",
        f"{state_code.lower()}_{city.lower().replace(' ', '_').replace('-', '_')}.nc",
    ]
    
    for variation in variations:
        filepath = os.path.join("data", variation)
        if os.path.exists(filepath):
            return filepath
    
    return None

def extract_seasonal_rainfall_from_nc(nc_file_path):
    """Extract seasonal rainfall data from .nc file"""
    try:
        # Load the dataset
        ds = xr.open_dataset(nc_file_path)
        
        # Convert precipitation rate to mm/month (same as plot_nc_file.py)
        ds["precip_mm_month"] = ds["tprate"] * 30 * 24 * 60 * 60 * 1000
        
        # Use center coordinates
        lat = float(ds.latitude.values.mean())
        lon = float(ds.longitude.values.mean())
        
        # Extract monthly rainfall values
        rain_vals = ds["precip_mm_month"].sel(
            latitude=lat, 
            longitude=lon, 
            method="nearest"
        ).mean(dim="forecast_reference_time").values.flatten()
        
        # Get forecast months
        base_time = pd.to_datetime(ds['forecast_reference_time'].values[0])
        forecast_months = ds["forecastMonth"].values
        actual_months = [(base_time + pd.DateOffset(months=int(m - 1))).month for m in forecast_months]
        month_names = [calendar.month_name[m] for m in actual_months]
        
        # Create month-rainfall mapping
        monthly_rainfall = {}
        for i, month_name in enumerate(month_names):
            monthly_rainfall[month_name] = float(rain_vals[i])
        
        # Calculate seasonal totals
        seasonal_rainfall = {
            'kharif': 0,    # June-September
            'rabi': 0,      # November-March
            'zaid': 0,      # April-May
            'annual': 0
        }
        
        # Map months to seasons
        season_mapping = {
            'June': 'kharif', 'July': 'kharif', 'August': 'kharif', 'September': 'kharif',
            'November': 'rabi', 'December': 'rabi', 'January': 'rabi', 'February': 'rabi', 'March': 'rabi',
            'April': 'zaid', 'May': 'zaid',
            'October': 'kharif'  # Sometimes October is included in Kharif
        }
        
        # Calculate seasonal totals
        for month_name, rainfall in monthly_rainfall.items():
            if month_name in season_mapping:
                season = season_mapping[month_name]
                seasonal_rainfall[season] += rainfall
            seasonal_rainfall['annual'] += rainfall
        
        ds.close()
        
        print(f"✅ Extracted rainfall from {nc_file_path}")
        print(f"Base time: {base_time}")
        print(f"Monthly data: {monthly_rainfall}")
        print(f"Seasonal totals: {seasonal_rainfall}")
        
        return seasonal_rainfall, monthly_rainfall, base_time
        
    except Exception as e:
        print(f"❌ Error extracting rainfall from {nc_file_path}: {e}")
        return None, None, None

def get_real_weather_data(state_name, district, lat=None, lon=None):
    """Get real weather data for crop prediction"""
    
    # Try to find the appropriate .nc file
    # First, try to map state name to state code
    state_code_mapping = {
        'MADHYA PRADESH': '23',
        'UTTAR PRADESH': '09',
        'PUNJAB': '03',
        'HARYANA': '06',
        'RAJASTHAN': '08',
        'GUJARAT': '24',
        'MAHARASHTRA': '27',
        'KARNATAKA': '29',
        'ANDHRA PRADESH': '28',
        'TAMIL NADU': '32',
        'WEST BENGAL': '19',
        'BIHAR': '10',
        'JHARKHAND': '20',
        'ODISHA': '21',
        'CHHATTISGARH': '22'
    }
    
    state_code = state_code_mapping.get(state_name.upper())
    if not state_code:
        print(f"⚠️ No state code mapping for {state_name}")
        return None
    
    # Try to find .nc file for the district
    nc_file = get_nc_file_for_location(state_code, district)
    
    if not nc_file:
        print(f"⚠️ No .nc file found for {district}, {state_name}")
        return None
    
    print(f"📁 Found .nc file: {nc_file}")
    
    # Extract rainfall data
    seasonal_rainfall, monthly_rainfall, base_time = extract_seasonal_rainfall_from_nc(nc_file)
    
    if not seasonal_rainfall:
        return None
    
    # Current date context
    current_date = datetime.now()
    current_month = current_date.month
    
    # Determine what data is available based on current season
    result = {
        'source': 'CDS_API_NetCDF',
        'file_used': nc_file,
        'base_time': base_time,
        'data_type': 'forecast',  # These are forecast files
        'kharif_mm': seasonal_rainfall['kharif'],
        'rabi_mm': seasonal_rainfall['rabi'],
        'zaid_mm': seasonal_rainfall['zaid'],
        'annual_mm': seasonal_rainfall['annual'],
        'monthly_breakdown': monthly_rainfall,
        'current_season': 'kharif' if 6 <= current_month <= 9 else 'rabi' if 11 <= current_month <= 3 else 'zaid'
    }
    
    return result

# Test function
if __name__ == "__main__":
    print("=== Testing Real Weather Data Integration ===")
    
    # Test with Sagar, MP (same as our prediction example)
    result = get_real_weather_data("MADHYA PRADESH", "SAGAR")
    
    if result:
        print("✅ Successfully extracted weather data:")
        print(f"  Kharif rainfall: {result['kharif_mm']:.1f} mm")
        print(f"  Rabi rainfall: {result['rabi_mm']:.1f} mm")
        print(f"  Zaid rainfall: {result['zaid_mm']:.1f} mm")
        print(f"  Annual rainfall: {result['annual_mm']:.1f} mm")
        print(f"  Current season: {result['current_season']}")
        print(f"  Data source: {result['source']}")
    else:
        print("❌ Failed to extract weather data")
