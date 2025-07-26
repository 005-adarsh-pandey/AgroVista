"""
Price data functions for MSP and market prices
"""
import os
import pandas as pd
import re
from .utils import find_latest_mandi_file, find_latest_msp_file


def get_msp_price_from_csv(app, crop_name):
    """Get MSP price from CSV file"""
    try:
        # Try multiple possible paths for the MSP CSV file
        possible_paths = [
            os.path.join(app.root_path, 'MSP_Data', 'comprehensive_msp_data.csv'),
            os.path.join(app.root_path, 'MSP_Data', 'enhanced_msp_data.csv'),
            os.path.join(app.static_folder, 'data', 'comprehensive_msp_data.csv'),
            'MSP_Data/comprehensive_msp_data.csv'
        ]
        
        msp_df = None
        for path in possible_paths:
            if os.path.exists(path):
                msp_df = pd.read_csv(path)
                break
        
        if msp_df is None:
            print("MSP CSV file not found in any of the expected locations")
            return None
        
        # Normalize crop name for search
        crop_normalized = crop_name.lower().strip()
        
        # Create search variants for better matching
        search_variants = [
            crop_normalized,
            crop_normalized.replace('(', '').replace(')', ''),  # Remove parentheses
            crop_normalized.split('(')[0].strip() if '(' in crop_normalized else crop_normalized,
        ]
        
        # Add common crop name mappings
        crop_mappings = {
            'masoor': 'masur (lentil)',
            'lentil': 'masur (lentil)',
            'masur': 'masur (lentil)',
            'rice': 'rice',
            'paddy': 'rice',
            'wheat': 'wheat',
            'cotton': 'cotton',
            'sugarcane': 'sugarcane',
            'maize': 'maize',
            'corn': 'maize',
            'tur': 'tur',
            'arhar': 'tur',
            'moong': 'moong',
            'urad': 'urad',
            'groundnut': 'groundnut',
            'peanut': 'groundnut',
            'sunflower': 'sunflower',
            'sesame': 'sesame',
            'sesamum': 'sesame',
            'mustard': 'rapeseed & mustard',
            'rapeseed': 'rapeseed & mustard',
            'gram': 'chana',
            'chana': 'gram',
            'masoor': 'lentil',
            'lentil': 'masur'
        }
        
        # Add mapped variants
        for variant in search_variants.copy():
            if variant in crop_mappings:
                search_variants.append(crop_mappings[variant])
        
        # Remove duplicates while preserving order
        search_variants = list(dict.fromkeys(search_variants))
        
        # Search for the crop using all variants
        for variant in search_variants:
            if not variant:
                continue
                
            # Try exact match first
            exact_match = msp_df[msp_df['Crop'].str.lower().str.strip() == variant.lower()]
            if not exact_match.empty:
                best_match = exact_match.iloc[0]
                return {
                    'price': best_match['MSP_Price'],
                    'year': best_match['Year'],
                    'source': best_match['Source']
                }
            
            # Try partial match (contains)
            partial_match = msp_df[msp_df['Crop'].str.lower().str.contains(variant.lower(), na=False)]
            if not partial_match.empty:
                best_match = partial_match.iloc[0]
                return {
                    'price': best_match['MSP_Price'],
                    'year': best_match['Year'],
                    'source': best_match['Source']
                }
        
        return None
        
    except Exception as e:
        print(f"Error loading MSP data: {e}")
        return None


def get_market_price_from_csv(app, crop_name, state=None, district=None, mandi=None):
    """Get market price from current mandi data file (latest available)"""
    try:
        # First try to get data from the latest mandi file
        latest_mandi_file = find_latest_mandi_file(app)
        if latest_mandi_file and os.path.exists(latest_mandi_file):
            print(f"Using latest mandi file: {latest_mandi_file}")
            mandi_df = pd.read_csv(latest_mandi_file)
            
            # Normalize column names to handle variations
            mandi_df.columns = [col.strip().replace(' ', '_').replace('x0020_', '').replace('__', '_') for col in mandi_df.columns]
            
            # Find the commodity column (could be Commodity, commodity, etc.)
            commodity_col = None
            for col in mandi_df.columns:
                if 'commodity' in col.lower():
                    commodity_col = col
                    break
            
            if commodity_col is None:
                print("No commodity column found in mandi data")
                return None
            
            # Enhanced search for the crop with multiple variations
            crop_normalized = crop_name.lower().strip()
            
            # Create search variants for better matching
            search_variants = [
                crop_normalized,
                crop_normalized.replace('(', '').replace(')', ''),  # Remove parentheses
                crop_normalized.split('(')[0].strip() if '(' in crop_normalized else crop_normalized,
                crop_normalized.split()[0],  # First word only
            ]
            
            # Add common crop name mappings
            crop_mappings = {
                'mousambi': 'mousambi',
                'mosambi': 'mousambi', 
                'sweet lime': 'mousambi',
                'lime': 'mousambi',
                'rice': 'rice',
                'paddy': 'rice',
                'wheat': 'wheat',
                'tomato': 'tomato',
                'potato': 'potato',
                'onion': 'onion',
                'brinjal': 'brinjal',
                'bhindi': 'bhindi',
                'green chilli': 'green chilli',
                'maize': 'maize',
                'cotton': 'cotton'
            }
            
            # Add mapped variants
            for variant in search_variants[:]:
                if variant in crop_mappings:
                    search_variants.append(crop_mappings[variant])
            
            # Remove duplicates while preserving order
            search_variants = list(dict.fromkeys(search_variants))
            
            # Search for crop matches
            crop_matches = pd.DataFrame()
            for variant in search_variants:
                if not variant:
                    continue
                    
                # Try exact match first
                exact_match = mandi_df[mandi_df[commodity_col].str.lower().str.strip() == variant.lower()]
                if not exact_match.empty:
                    crop_matches = pd.concat([crop_matches, exact_match])
                    break
                
                # Try contains match
                contains_match = mandi_df[mandi_df[commodity_col].str.lower().str.contains(variant.lower(), na=False)]
                if not contains_match.empty:
                    crop_matches = pd.concat([crop_matches, contains_match])
                    break
            
            if crop_matches.empty:
                print(f"No matches found for crop: {crop_name}")
                return None
            
            # Filter by location if provided
            if state:
                state_matches = crop_matches[crop_matches['state'].str.lower().str.contains(state.lower(), na=False)]
                if not state_matches.empty:
                    crop_matches = state_matches
            
            if district:
                district_matches = crop_matches[crop_matches['district'].str.lower().str.contains(district.lower(), na=False)]
                if not district_matches.empty:
                    crop_matches = district_matches
            
            if mandi:
                mandi_matches = crop_matches[crop_matches['market'].str.lower().str.contains(mandi.lower(), na=False)]
                if not mandi_matches.empty:
                    crop_matches = mandi_matches
            
            # Get the first match
            best_match = crop_matches.iloc[0]
            
            # Extract price information
            def get_numeric_value(value):
                if pd.isna(value) or value is None:
                    return None
                try:
                    if isinstance(value, str):
                        cleaned = ''.join(c for c in value if c.isdigit() or c == '.')
                        if cleaned:
                            return float(cleaned)
                    else:
                        return float(value)
                except:
                    return None
            
            # Extract price information
            min_price = get_numeric_value(best_match.get('min_price'))
            max_price = get_numeric_value(best_match.get('max_price'))
            modal_price = get_numeric_value(best_match.get('modal_price'))
            
            return {
                'min_price': min_price,
                'max_price': max_price,
                'modal_price': modal_price,
                'state': best_match.get('state'),
                'district': best_match.get('district'),
                'market': best_match.get('market'),
                'arrival_date': best_match.get('arrival_date')
            }
        
        return None
        
    except Exception as e:
        print(f"Error in get_market_price_from_csv: {e}")
        return None


def get_latest_msp_price(app, crop, msp_files=None):
    """Get the latest MSP price for a crop from CSV file"""
    
    # Get MSP data from CSV
    msp_data = get_msp_price_from_csv(app, crop)
    
    if msp_data:
        return msp_data['price'], msp_data['year']
    else:
        return None, "Crop not found in MSP data"


def get_original_msp_price(app, crop, msp_files):
    """Original MSP price lookup method as fallback"""
    
    if not msp_files:
        return None, "No MSP files available"
    
    # Determine crop season to search in appropriate file
    kharif_crops = ['paddy', 'rice', 'jowar', 'bajra', 'ragi', 'maize', 'arhar', 'tur', 
                   'moong', 'urad', 'cotton', 'groundnut', 'sunflower', 'soyabean', 
                   'sesamum', 'nigerseed']
    
    rabi_crops = ['wheat', 'barley', 'gram', 'masur', 'lentil', 'rapeseed', 'mustard', 
                 'safflower']
    
    crop_lower = crop.lower()
    crop_mapping = {
        'rice': 'paddy',
        'arhar': 'arhar',
        'tur': 'arhar', 
        'moong(green gram)': 'moong',
        'green gram': 'moong',
        'mustard': 'rapeseed & mustard',
        'lentil': 'masur (lentil)',
        'masoor': 'masur (lentil)',
        'soyabean': 'soyabean',
        'soybean': 'soyabean',
        'soya': 'soyabean',
        'sunflower': 'sunflower seed',
        'groundnut': 'groundnut',
        'peanut': 'groundnut'
    }
    
    search_crop = crop_mapping.get(crop_lower, crop_lower)
    
    # Determine which file to search first
    files_to_search = []
    if any(krop in search_crop for krop in kharif_crops) and 'kharif' in msp_files:
        files_to_search.append(('kharif', msp_files['kharif']))
    if any(rrop in search_crop for rrop in rabi_crops) and 'rabi' in msp_files:
        files_to_search.append(('rabi', msp_files['rabi']))
    
    # If not categorized, search both files
    if not files_to_search:
        if 'kharif' in msp_files:
            files_to_search.append(('kharif', msp_files['kharif']))
        if 'rabi' in msp_files:
            files_to_search.append(('rabi', msp_files['rabi']))
    
    # Search each file
    for season, file_path in files_to_search:
        try:
            msp_df = pd.read_csv(file_path)
            
            # Check file format and get the latest year column
            if 'KMS 2024-25' in msp_df.columns:
                # This is the 2013-25 format file - can calculate 2025 prices
                latest_year_col = 'KMS 2024-25'
                commodity_col = 'Commodities'
                base_year = '2024-25'
                
                # Check if percentage increase column exists for extrapolation
                has_percentage = '% increase 201314 to 2024-25' in msp_df.columns
                
            elif '2023-24 - (Rs. per Quintal)' in msp_df.columns:
                # This is the 2023-24 format file  
                latest_year_col = '2023-24 - (Rs. per Quintal)'
                commodity_col = 'Sub Commodity'
                base_year = '2023-24'
                has_percentage = '% increase in MSP in 2023-24 over 2014-15' in msp_df.columns
                
            else:
                # Try to find the most recent year column automatically
                year_cols = [col for col in msp_df.columns if any(yr in col for yr in ['2024', '2023', '2022'])]
                if year_cols:
                    latest_year_col = max(year_cols)  # Get the latest year
                    commodity_col = 'Commodities' if 'Commodities' in msp_df.columns else 'Sub Commodity'
                    base_year = latest_year_col.split()[-1] if 'KMS' in latest_year_col else latest_year_col.split()[0]
                    has_percentage = False
                else:
                    continue
            
            # Enhanced search in MSP data with better crop matching
            
            # Create multiple search variants for better matching
            search_variants = [
                search_crop,  # Original
                search_crop.replace('(', '').replace(')', ''),  # Remove parentheses
                search_crop.split('(')[0].strip(),  # Before parentheses
                search_crop.split()[0],  # First word only
            ]
            
            # Add common crop name mappings
            crop_mappings = {
                'paddy': 'rice',
                'rice': 'paddy',
                'arhar': 'tur',
                'tur': 'arhar',
                'moong': 'green gram',
                'green gram': 'moong',
                'urad': 'black gram',
                'black gram': 'urad',
                'mustard': 'rapeseed',
                'rapeseed': 'mustard',
                'gram': 'chana',
                'chana': 'gram',
                'masoor': 'lentil',
                'lentil': 'masur'
            }
            
            # Add mapped variants
            for variant in search_variants[:]:  # Copy to avoid modifying during iteration
                if variant.lower() in crop_mappings:
                    search_variants.append(crop_mappings[variant.lower()])
            
            # Search for crop in MSP data
            msp_match = None
            for variant in search_variants:
                if variant:  # Skip empty variants
                    mask = msp_df[commodity_col].str.contains(variant, case=False, na=False)
                    if mask.any():
                        msp_match = msp_df[mask].iloc[0]
                        break
            
            if msp_match is not None:
                msp_price = msp_match.get(latest_year_col, 0)
                
                # Clean and convert price
                if pd.notna(msp_price):
                    if isinstance(msp_price, str):
                        msp_price = re.sub(r'[^\d.]', '', str(msp_price))
                    try:
                        msp_price = float(msp_price) if msp_price else 0
                        if msp_price > 0:
                            return {
                                'msp_price': msp_price,
                                'year': base_year,
                                'source': f'MSP {season} {base_year}',
                                'commodity': msp_match.get(commodity_col, search_crop)
                            }
                    except (ValueError, TypeError):
                        continue
                        
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
    
    # Return None if no MSP data found
    return None
