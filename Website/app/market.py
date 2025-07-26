"""
Market-related functions and routes
"""
import pandas as pd
from flask import jsonify, request
from .price_data import get_market_price_from_csv, get_msp_price_from_csv


def market_price_handler(app):
    """Get market price using the simplified CSV approach"""
    print("🏪 MARKET PRICE REQUEST RECEIVED!")
    data = request.get_json()
    state = data.get("state", "").strip()
    district = data.get("district", "").strip()
    mandi = data.get("mandi", "").strip()
    crop = data.get("crop", "").strip()

    if not crop:
        return jsonify({"error": "Crop name is required", "data_found": False}), 200

    try:
        # Use the CSV-based market price function
        market_data = get_market_price_from_csv(app, crop, state, district, mandi)
        
        if market_data:
            return jsonify({
                "data_found": True,
                "min_price": market_data.get('min_price'),
                "max_price": market_data.get('max_price'),
                "modal_price": market_data.get('modal_price'),
                "market_state": market_data.get('state', ''),
                "market_district": market_data.get('district', ''),
                "market_name": market_data.get('market', ''),
                "arrival_date": market_data.get('arrival_date'),
                "price_currency": "Rs. per quintal"
            })
        else:
            return jsonify({
                "error": f"No market data found for {crop}",
                "data_found": False
            }), 200
            
    except Exception as e:
        return jsonify({
            "error": f"Error processing request: {str(e)}",
            "data_found": False
        }), 200


def get_price_data_handler(app):
    """Get comprehensive MSP and market price data for a crop using CSV files"""
    data = request.json
    crop = data.get('crop')
    state = data.get('state')
    district = data.get('district')
    
    if not crop:
        return jsonify({'error': 'Crop name required'}), 400
    
    try:
        response = {
            'crop': crop,
            'msp_available': False,
            'market_available': False,
            'currency': 'Rs. per quintal',
            'data_found': False
        }
        
        # Get MSP data from CSV
        msp_data = get_msp_price_from_csv(app, crop)
        if msp_data:
            # Convert to Python types for JSON serialization
            def convert_to_native(value):
                if pd.isna(value) or value is None:
                    return None
                try:
                    if hasattr(value, 'item'):  # numpy types
                        return value.item()
                    elif isinstance(value, (int, float)):
                        return float(value)
                    else:
                        return str(value)
                except:
                    return str(value)
            
            msp_price = convert_to_native(msp_data.get('price'))
            
            response.update({
                'msp_available': True,
                'msp_price': msp_price,
                'msp_year': str(msp_data.get('year', '')),
                'msp_source': str(msp_data.get('source', '')),
                'data_found': True
            })
        
        # Get market data from CSV
        market_data = get_market_price_from_csv(app, crop, state, district)
        if market_data:
            # Convert numpy types to Python types for JSON serialization
            def convert_to_native(value):
                if pd.isna(value) or value is None:
                    return None
                try:
                    # Handle numpy/pandas numeric types
                    if hasattr(value, 'item'):  # numpy types
                        return value.item()
                    elif isinstance(value, (int, float)):
                        return float(value)
                    else:
                        return str(value)
                except:
                    return None
            
            min_price = convert_to_native(market_data.get('min_price'))
            max_price = convert_to_native(market_data.get('max_price'))
            modal_price = convert_to_native(market_data.get('modal_price'))
            
            response.update({
                'market_available': True,
                'market_price': modal_price or min_price,
                'min_price': min_price,
                'max_price': max_price,
                'modal_price': modal_price,
                'market_location': {
                    'state': str(market_data.get('state', '')),
                    'district': str(market_data.get('district', '')),
                    'market': str(market_data.get('market', ''))
                },
                'data_found': True
            })
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'error': f'Error processing request: {str(e)}',
            'data_found': False
        }), 500
