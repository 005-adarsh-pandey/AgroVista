"""
Market Data Service Module for AgroVista
Handles market price data, MSP data, and price trend analysis
"""

import pandas as pd
import os
import glob
import logging
import requests
from datetime import datetime
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketDataService:
    def __init__(self):
        self.msp_data_dir = "../MSP_Data"
        self.mandi_data_dir = "../static/data"
        self.api_base_url = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
        self.api_key = "579b464db66ec23bdd000001c967da9cd4b24a5d4bfe4601e18d2521"
        self.cache = {}
    
    def get_msp_data(self, crop_name):
        """Get MSP (Minimum Support Price) data for a crop"""
        try:
            # Search for MSP data in CSV files
            msp_files = glob.glob(os.path.join(self.msp_data_dir, "*.csv"))
            
            for file_path in msp_files:
                try:
                    df = pd.read_csv(file_path)
                    msp_data = self.search_crop_in_msp(df, crop_name)
                    if msp_data:
                        logger.info(f"✅ MSP data found for {crop_name}")
                        return msp_data
                except Exception as e:
                    logger.warning(f"Error reading MSP file {file_path}: {e}")
                    continue
            
            logger.warning(f"No MSP data found for {crop_name}")
            return None
            
        except Exception as e:
            logger.error(f"❌ Error getting MSP data: {e}")
            return None
    
    def search_crop_in_msp(self, df, search_crop):
        """Search for crop in MSP dataframe"""
        try:
            search_crop = search_crop.lower().strip()
            
            # Try different column names
            commodity_cols = ['Commodities', 'Sub Commodity', 'Commodity']
            commodity_col = None
            
            for col in commodity_cols:
                if col in df.columns:
                    commodity_col = col
                    break
            
            if not commodity_col:
                return None
            
            # Search for crop with fuzzy matching
            for idx, row in df.iterrows():
                crop_name = str(row[commodity_col]).lower().strip()
                
                if search_crop in crop_name or crop_name in search_crop:
                    # Find the latest year column
                    year_cols = [col for col in df.columns if any(yr in str(col) for yr in ['2024', '2023', '2022'])]
                    
                    if year_cols:
                        latest_col = max(year_cols)
                        price = row[latest_col]
                        
                        return {
                            'crop': row[commodity_col],
                            'msp_price': price,
                            'year': latest_col,
                            'unit': 'Rs. per Quintal'
                        }
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error searching crop in MSP: {e}")
            return None
    
    def get_mandi_prices(self, state=None, limit=100):
        """Get current mandi prices from data.gov.in API"""
        try:
            cache_key = f"mandi_{state}_{limit}"
            
            # Check cache (valid for 1 hour)
            if self.is_cached(cache_key, 3600):
                return self.cache[cache_key]['data']
            
            params = {
                'api-key': self.api_key,
                'format': 'json',
                'limit': limit
            }
            
            if state:
                params['filters[state]'] = state
            
            response = requests.get(self.api_base_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if 'records' in data:
                records = data['records']
                
                # Process and format the data
                mandi_data = []
                for record in records:
                    mandi_data.append({
                        'state': record.get('state', 'N/A'),
                        'district': record.get('district', 'N/A'),
                        'market': record.get('market', 'N/A'),
                        'commodity': record.get('commodity', 'N/A'),
                        'variety': record.get('variety', 'N/A'),
                        'min_price': record.get('min_price', 0),
                        'max_price': record.get('max_price', 0),
                        'modal_price': record.get('modal_price', 0),
                        'price_date': record.get('price_date', 'N/A')
                    })
                
                # Cache the result
                self.cache[cache_key] = {
                    'data': mandi_data,
                    'timestamp': datetime.now().timestamp()
                }
                
                logger.info(f"✅ Mandi prices fetched: {len(mandi_data)} records")
                return mandi_data
            else:
                logger.warning("No records found in mandi data API response")
                return []
                
        except Exception as e:
            logger.error(f"❌ Error fetching mandi prices: {e}")
            return self.get_fallback_mandi_data()
    
    def get_price_trends(self, crop_name, days=30):
        """Get price trends for a crop (simplified)"""
        try:
            # In a production system, this would analyze historical price data
            # For now, generate a simplified trend
            
            trend_data = []
            base_price = 2000  # Base price in Rs per quintal
            
            for i in range(days):
                date = (datetime.now() - timedelta(days=days-i)).strftime('%Y-%m-%d')
                # Simple price variation
                price_variation = base_price + (i * 10) + (20 * (0.5 - random.random()))
                
                trend_data.append({
                    'date': date,
                    'price': max(1000, price_variation),
                    'volume': 100 + (i * 5)
                })
            
            return trend_data
            
        except Exception as e:
            logger.error(f"❌ Error getting price trends: {e}")
            return []
    
    def find_latest_mandi_file(self):
        """Find the latest mandi data file"""
        try:
            if not os.path.exists(self.mandi_data_dir):
                return None
            
            files = [f for f in os.listdir(self.mandi_data_dir) if f.startswith('mandi_prices_')]
            
            if not files:
                return None
            
            # Get the latest file by modification time
            latest_file = max(files, key=lambda x: os.path.getmtime(os.path.join(self.mandi_data_dir, x)))
            return os.path.join(self.mandi_data_dir, latest_file)
            
        except Exception as e:
            logger.error(f"❌ Error finding latest mandi file: {e}")
            return None
    
    def get_local_mandi_data(self, limit=100):
        """Get mandi data from local CSV files"""
        try:
            latest_file = self.find_latest_mandi_file()
            
            if not latest_file:
                return []
            
            df = pd.read_csv(latest_file)
            
            # Convert to list of dictionaries
            return df.head(limit).to_dict('records')
            
        except Exception as e:
            logger.error(f"❌ Error reading local mandi data: {e}")
            return []
    
    def is_cached(self, cache_key, duration):
        """Check if data is cached and still valid"""
        if cache_key not in self.cache:
            return False
        
        cached_time = self.cache[cache_key]['timestamp']
        return (datetime.now().timestamp() - cached_time) < duration
    
    def get_fallback_mandi_data(self):
        """Fallback mandi data when API fails"""
        return [
            {
                'state': 'Data Unavailable',
                'district': 'N/A',
                'market': 'N/A',
                'commodity': 'N/A',
                'variety': 'N/A',
                'min_price': 0,
                'max_price': 0,
                'modal_price': 0,
                'price_date': 'N/A'
            }
        ]

# Global instance
market_service = MarketDataService()
