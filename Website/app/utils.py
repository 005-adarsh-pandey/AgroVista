"""
Utility functions for the Flask application
"""
import os
import glob
import datetime
import time
import pandas as pd
import re
import threading

# Global reference to mandi downloader
mandi_downloader = None


def find_latest_mandi_file(app):
    """Find the latest available mandi data file, with graceful fallback"""
    data_dir = os.path.join(app.static_folder, 'data')
    
    # First try to find today's file
    today = datetime.datetime.now().strftime('%Y_%m_%d')
    today_file = os.path.join(data_dir, f"mandi_prices_{today}.csv")
    
    if os.path.exists(today_file):
        return today_file
    
    # Look for mandi_prices_*.csv files
    pattern = os.path.join(data_dir, "mandi_prices_*.csv")
    files = glob.glob(pattern)
    
    if not files:
        # No error messages on frontend - silent fallback
        return None
    
    # Sort files by modification time (newest first)
    files.sort(key=os.path.getmtime, reverse=True)
    latest_file = files[0]
    
    return latest_file


def find_latest_msp_file(app):
    """Find the latest available MSP data files (both Kharif and Rabi)"""
    
    # Look for MSP files in order of preference (newest first)
    kharif_files = [
        'MSP_Kharif_2013-25.csv',  # Has 2024-25 data for Kharif crops
        'MSP_Kharif_2024-25.csv',
        'MSP_Kharif_2023-24.csv'
    ]
    
    rabi_files = [
        'MSP_Rabi_2024-25.csv',    # Would have 2024-25 Rabi data if available
        'MSP_2023-24.csv',         # Has both Kharif and Rabi crops for 2023-24
        'MSP_2022-23.csv'
    ]
    
    result = {}
    
    # Find Kharif file
    for filename in kharif_files:
        file_path = os.path.join(app.root_path, filename)
        if os.path.exists(file_path):
            result['kharif'] = file_path
            print(f"Using Kharif MSP data: {filename}")
            break
    
    # Find Rabi file  
    for filename in rabi_files:
        file_path = os.path.join(app.root_path, filename)
        if os.path.exists(file_path):
            result['rabi'] = file_path
            print(f"Using Rabi MSP data: {filename}")
            break
    
    return result if result else None


def prepare_today_mandi_data(app):
    """Initialize mandi downloader and check for existing data"""
    global mandi_downloader
    
    data_dir = os.path.join(app.static_folder, "data")
    json_path = os.path.join(data_dir, "mandi_crop_hierarchy.json")

    # Check if we have recent data
    latest_mandi = find_latest_mandi_file(app)
    if latest_mandi and os.path.exists(json_path):
        # Check if data is recent (within 1 hour)
        file_age_hours = (time.time() - os.path.getmtime(latest_mandi)) / 3600
        if file_age_hours < 1:
            print(f"Recent mandi data available ({file_age_hours:.1f}h old)")
            return

    # Initialize downloader for background operation
    try:
        # Import from the same directory first, then try parent directory
        try:
            from .mandi_auto_downloader import MandiDataDownloader
        except ImportError:
            import sys
            sys.path.append(os.path.dirname(__file__))
            from mandi_auto_downloader import MandiDataDownloader
        
        mandi_downloader = MandiDataDownloader()
        mandi_downloader.set_server_active(True)  # Activate when Flask starts
        
        # Start downloader in background thread
        downloader_thread = threading.Thread(target=mandi_downloader.start_scheduler, daemon=True)
        downloader_thread.start()
        
        print("Mandi auto-downloader initialized")
        
    except Exception as e:
        print(f"Warning: Downloader init failed: {str(e)[:50]}...")  # Truncated error


def get_mandi_status(app):
    """Get status of mandi downloader and data freshness"""
    global mandi_downloader
    
    try:
        data_dir = os.path.join(app.static_folder, "data")
        latest_file = find_latest_mandi_file(app)
        
        status = {
            'downloader_active': mandi_downloader and mandi_downloader.server_active if mandi_downloader else False,
            'latest_data_available': latest_file is not None,
            'data_freshness': 'unknown',
            'next_download': 'unknown',
            'total_files': 0,
            'market_hours': False,
            'downloader_stats': {}
        }
        
        # Get downloader stats if available
        if mandi_downloader:
            try:
                status['downloader_stats'] = mandi_downloader.get_status()
            except:
                pass
        
        if latest_file:
            # Calculate data age
            file_age_hours = (time.time() - os.path.getmtime(latest_file)) / 3600
            status['data_age_hours'] = round(file_age_hours, 1)
            
            if file_age_hours < 2:
                status['data_freshness'] = 'very_fresh'
            elif file_age_hours < 6:
                status['data_freshness'] = 'fresh'
            elif file_age_hours < 24:
                status['data_freshness'] = 'acceptable'
            else:
                status['data_freshness'] = 'stale'
            
            # Count total mandi files
            mandi_files = [f for f in os.listdir(data_dir) if f.startswith('mandi_prices_')]
            status['total_files'] = len(mandi_files)
        
        # Check if in market hours
        current_hour = datetime.datetime.now().hour
        status['market_hours'] = 6 <= current_hour <= 18
        
        # Calculate next download time
        if status['market_hours']:
            next_hour = current_hour + 1
            if next_hour <= 18:
                status['next_download'] = f"{next_hour:02d}:00"
            else:
                status['next_download'] = "06:00 (next day)"
        else:
            status['next_download'] = "06:00" if current_hour < 6 else "06:00 (next day)"
        
        return status
        
    except Exception as e:
        return {'error': f'Status check failed: {str(e)}'}
