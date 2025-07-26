import matplotlib
matplotlib.use('Agg')  # <-- Force non-GUI backend (for headless/server environments)

import xarray as xr
import matplotlib.pyplot as plt
import calendar
import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
import logging

# Suppress matplotlib warnings about categorical units - more aggressive approach
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', message='Using categorical units to plot a list of strings')
warnings.filterwarnings('ignore', category=UserWarning, message='.*categorical units.*')

# Also suppress matplotlib logging
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('matplotlib.category').setLevel(logging.ERROR)

# Set matplotlib to not show warnings
matplotlib.rcParams['axes.formatter.use_mathtext'] = False
matplotlib.rcParams['axes.unicode_minus'] = False

def plot_forecast(nc_file_path, city="city", state_code=None):
    # Suppress warnings within this function
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        ds = xr.open_dataset(nc_file_path)

        # Unit conversions
        ds["t2m_C"] = ds["t2m"] - 273.15
        ds["precip_mm_month"] = ds["tprate"] * 30 * 24 * 60 * 60 * 1000
        ds["solar_mj_month"] = ds["msnsrf"] * 30 * 24 * 60 * 60 / 1_000_000

        # Use center coordinates
        lat = float(ds.latitude.values.mean())
        lon = float(ds.longitude.values.mean())

        # Extract monthly values
        temp_vals = ds["t2m_C"].sel(latitude=lat, longitude=lon, method="nearest").mean(dim="forecast_reference_time").values.flatten()
        rain_vals = ds["precip_mm_month"].sel(latitude=lat, longitude=lon, method="nearest").mean(dim="forecast_reference_time").values.flatten()
        solar_vals = ds["solar_mj_month"].sel(latitude=lat, longitude=lon, method="nearest").mean(dim="forecast_reference_time").values.flatten()

        # Get forecast months
        base_time = pd.to_datetime(ds['forecast_reference_time'].values[0])
        forecast_months = ds["forecastMonth"].values
        actual_months = [(base_time + pd.DateOffset(months=int(m - 1))).month for m in forecast_months]
        month_names = [calendar.month_name[m] for m in actual_months]    # Plot setup
    colors = {"temp": "#E74C3C", "rain": "#3498DB", "solar": "#F1C40F"}
    plt.figure(figsize=(16, 6))
    font_opts = {'fontsize': 12, 'fontweight': 'bold'}

    # Temperature
    plt.subplot(1, 3, 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bars = plt.bar(month_names, temp_vals, color=colors["temp"])
    plt.title("Avg Temperature (°C)", **font_opts)
    plt.ylabel("°C")
    plt.xticks(rotation=45)
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f"{bar.get_height():.1f}", ha='center', fontsize=10)

    # Rainfall
    plt.subplot(1, 3, 2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bars = plt.bar(month_names, rain_vals, color=colors["rain"])
    plt.title("Monthly Rainfall (mm)", **font_opts)
    plt.ylabel("mm")
    plt.xticks(rotation=45)
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, f"{bar.get_height():.0f}", ha='center', fontsize=10)

    # Solar Radiation
    plt.subplot(1, 3, 3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bars = plt.bar(month_names, solar_vals, color=colors["solar"])
    plt.title("Solar Radiation (MJ/m²)", **font_opts)
    plt.ylabel("MJ/m²")
    plt.xticks(rotation=45)
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{bar.get_height():.0f}", ha='center', fontsize=10)

    plt.tight_layout()

    # Save to static/images
    output_dir = os.path.join("static", "images")
    os.makedirs(output_dir, exist_ok=True)

    # Filename (without timestamp for caching)
    base_name = f"{state_code.lower()}_{city.lower().replace(' ', '_')}_forecast.png" if state_code else f"{city.lower().replace(' ', '_')}_forecast.png"
    image_path = os.path.join(output_dir, base_name)

    # Save and return
    plt.savefig(image_path)
    plt.close()

    return image_path
