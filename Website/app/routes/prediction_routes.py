"""
Prediction Routes Module for AgroVista
Handles crop yield prediction routes and API endpoints
"""

from flask import Blueprint, render_template, request, jsonify
import logging
from datetime import datetime
import sys
import os

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from app.services.prediction_service import prediction_service
from app.services.weather_service import weather_service
from enhanced_ml_inputs import prepare_enhanced_ml_inputs

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create blueprint
prediction_bp = Blueprint('prediction', __name__)

@prediction_bp.route('/predictor')
def predictor():
    """Crop predictor page"""
    try:
        return render_template('predictor.html')
    except Exception as e:
        logger.error(f"❌ Error in predictor route: {e}")
        return "Error loading predictor page", 500

@prediction_bp.route('/predict', methods=['POST'])
def predict():
    """Main prediction API endpoint"""
    try:
        # Get form data
        data = request.get_json() if request.is_json else request.form
        
        # Extract and validate input parameters
        state = data.get('state', '').strip()
        district = data.get('district', '').strip()
        crop = data.get('crop', '').strip()
        season = data.get('season', '').strip()
        area = float(data.get('area', 0))
        lat = float(data.get('latitude', 0))
        lon = float(data.get('longitude', 0))
        year = int(data.get('year', datetime.now().year))
        
        # Validation
        if not all([state, district, crop, season]) or area <= 0:
            return jsonify({
                'success': False,
                'error': 'Missing or invalid input parameters'
            }), 400
        
        logger.info(f"🔮 Making prediction for {crop} in {district}, {state}")
        
        # Prepare enhanced ML inputs (includes rainfall data)
        ml_inputs = prepare_enhanced_ml_inputs(
            state=state,
            district=district,
            crop=crop,
            season=season,
            area=area,
            latitude=lat,
            longitude=lon,
            year=year
        )
        
        # Prepare features for prediction
        features = prediction_service.prepare_prediction_features(
            state=state,
            district=district,
            crop=crop,
            season=season,
            year=year,
            area=area,
            lat=lat,
            lon=lon,
            rainfall_data=ml_inputs['rainfall_data']
        )
        
        # Make prediction
        prediction = prediction_service.make_prediction(features)
        confidence = prediction_service.get_prediction_confidence(features)
        
        # Calculate yield per hectare
        yield_per_hectare = prediction / area if area > 0 else 0
        
        # Get current weather for context
        current_weather = weather_service.get_current_weather(lat, lon)
        
        # Prepare response
        response = {
            'success': True,
            'prediction': {
                'total_production': round(prediction, 2),
                'yield_per_hectare': round(yield_per_hectare, 2),
                'confidence': round(confidence, 1),
                'unit': 'Metric Tonnes'
            },
            'inputs': {
                'state': state,
                'district': district,
                'crop': crop,
                'season': season,
                'area': area,
                'year': year
            },
            'weather_context': current_weather,
            'rainfall_data': ml_inputs['rainfall_data'],
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"✅ Prediction successful: {prediction:.2f} MT")
        return jsonify(response)
        
    except ValueError as e:
        logger.error(f"❌ Validation error in prediction: {e}")
        return jsonify({
            'success': False,
            'error': f'Invalid input: {str(e)}'
        }), 400
        
    except Exception as e:
        logger.error(f"❌ Error in prediction: {e}")
        return jsonify({
            'success': False,
            'error': 'Prediction service temporarily unavailable'
        }), 500

@prediction_bp.route('/api/validate-inputs', methods=['POST'])
def validate_inputs():
    """Validate prediction inputs"""
    try:
        data = request.get_json()
        
        errors = []
        
        # Validate required fields
        required_fields = ['state', 'district', 'crop', 'season', 'area']
        for field in required_fields:
            if not data.get(field):
                errors.append(f'{field} is required')
        
        # Validate numeric fields
        try:
            area = float(data.get('area', 0))
            if area <= 0:
                errors.append('Area must be greater than 0')
        except (ValueError, TypeError):
            errors.append('Area must be a valid number')
        
        try:
            lat = float(data.get('latitude', 0))
            if not (-90 <= lat <= 90):
                errors.append('Latitude must be between -90 and 90')
        except (ValueError, TypeError):
            errors.append('Latitude must be a valid number')
        
        try:
            lon = float(data.get('longitude', 0))
            if not (-180 <= lon <= 180):
                errors.append('Longitude must be between -180 and 180')
        except (ValueError, TypeError):
            errors.append('Longitude must be a valid number')
        
        return jsonify({
            'valid': len(errors) == 0,
            'errors': errors
        })
        
    except Exception as e:
        logger.error(f"❌ Error validating inputs: {e}")
        return jsonify({
            'valid': False,
            'errors': ['Validation service error']
        }), 500

@prediction_bp.route('/api/crop-info/<crop_name>')
def get_crop_info(crop_name):
    """Get information about a specific crop"""
    try:
        # Basic crop information (can be expanded)
        crop_info = {
            'rice': {
                'season': 'Kharif',
                'growing_period': '120-150 days',
                'water_requirement': 'High',
                'soil_type': 'Clay, loam'
            },
            'wheat': {
                'season': 'Rabi',
                'growing_period': '120-130 days',
                'water_requirement': 'Medium',
                'soil_type': 'Loam, sandy loam'
            },
            'cotton': {
                'season': 'Kharif',
                'growing_period': '150-180 days',
                'water_requirement': 'Medium to High',
                'soil_type': 'Black cotton soil'
            }
        }
        
        info = crop_info.get(crop_name.lower(), {
            'season': 'Variable',
            'growing_period': 'Variable',
            'water_requirement': 'Variable',
            'soil_type': 'Variable'
        })
        
        return jsonify({
            'crop': crop_name,
            'info': info
        })
        
    except Exception as e:
        logger.error(f"❌ Error getting crop info: {e}")
        return jsonify({'error': 'Crop information unavailable'}), 500
