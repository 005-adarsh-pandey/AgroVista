import cdsapi
import os
from datetime import datetime, timedelta

def get_area_box(lat, lon, delta=0.25):
    # Add small perturbation to ensure unique area boxes for nearby cities
    import hashlib
    city_hash = int(hashlib.md5(f"{lat}_{lon}".encode()).hexdigest()[:8], 16)
    perturbation = (city_hash % 100) * 0.001  # 0.001 to 0.099 degree perturbation
    
    adj_lat = lat + perturbation
    adj_lon = lon + perturbation
    
    return [adj_lat + delta, adj_lon - delta, adj_lat - delta, adj_lon + delta]  # [N, W, S, E]

def download_forecast_nc(state_code, city, lat, lon, output_folder='data'):
    area = get_area_box(lat, lon)
    os.makedirs(output_folder, exist_ok=True)

    filename = f"{state_code.lower()}_{city.lower().replace(' ', '_')}.nc"
    filepath = os.path.join(output_folder, filename)

    if os.path.exists(filepath) and os.path.getsize(filepath) > 1000:
        print(f"[INFO] Using cached forecast: {filename}")
        return filepath

    # Attempt current month first, fallback if needed
    now = datetime.utcnow()
    tried_months = []

    for i in range(2):  # Try current and previous month
        attempt_date = now - timedelta(days=30 * i)
        year = str(attempt_date.year)
        month = f"{attempt_date.month:02d}"
        tried_months.append(f"{year}-{month}")

        print(f"[INFO] Attempting forecast for {year}-{month}")

        try:
            c = cdsapi.Client()

            c.retrieve(
                'seasonal-monthly-single-levels',
                {
                    'format': 'netcdf',
                    'originating_centre': 'ecmwf',
                    'system': '51',
                    'variable': [
                        '2m_temperature',
                        'total_precipitation',
                        'surface_solar_radiation'
                    ],
                    'product_type': 'ensemble_mean',
                    'year': year,
                    'month': month,
                    'leadtime_month': ['1', '2', '3', '4', '5', '6'],
                    'area': area,
                },
                filepath
            )

            print(f"[✅] Forecast downloaded using data from {year}-{month}")
            return filepath

        except Exception as e:
            print(f"[⚠️] Failed to fetch data for {year}-{month}: {e}")

    raise RuntimeError(f"❌ No forecast data available for months: {tried_months}")
