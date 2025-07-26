"""
Prediction Service Module for AgroVista
Handles ML model loading, prediction logic, and yield calculations
"""

import pandas as pd
import numpy as np
import joblib
import os
import logging
from datetime import datetime
from robust_encoding import robust_encode_inputs, load_or_create_encoders

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PredictionService:
    def __init__(self):
        self.model = None
        self.encoders = None
        self.feature_columns = None
        self.load_model_and_encoders()
    
    def load_model_and_encoders(self):
        """Load the trained model and encoders"""
        try:
            # Load model
            model_path = "../Models/quick_rf_model.pkl"
            if os.path.exists(model_path):
                model_data = joblib.load(model_path)
                self.model = model_data.get('model')
                self.encoders = model_data.get('label_encoders', {})
                self.feature_columns = model_data.get('feature_columns', [])
                logger.info("✅ Model and encoders loaded successfully")
            else:
                # Fallback to separate files
                self.model = joblib.load("../Models/crop_prediction_rf_model.pkl")
                self.encoders = load_or_create_encoders()
                self.feature_columns = [
                    'Crop_Year', 'Area (Hectares)', 'Latitude', 'Longitude',
                    'Rainfall - Kharif (mm)', 'Rainfall - Rabi (mm)', 
                    'Rainfall - Zaid (mm)', 'Rainfall - Whole Year (mm)',
                    'State_Name_encoded', 'District_Name_encoded', 
                    'Crop_encoded', 'Season_encoded'
                ]
                logger.info("✅ Model loaded from separate files")
                
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            raise
    
    def prepare_prediction_features(self, state, district, crop, season, year, area, lat, lon, rainfall_data, area_unit="hectare"):
        """Prepare features for model prediction, with correct area conversion and encoding check"""
        try:
            # Area conversion: only convert if input is in acres
            if area_unit.lower() == "acre":
                area = float(area) * 0.40468564224  # 1 acre = 0.40468564224 hectares
                logger.info(f"Converted area from acres to hectares: {area:.4f}")
            else:
                area = float(area)

            # Encode categorical variables
            encoded = robust_encode_inputs(state, district, crop, season, self.encoders)

            # Check encoding matches model's encoder classes
            for key, val in encoded.items():
                encoder_name = key.replace('_encoded', '')
                if encoder_name in self.encoders:
                    encoder = self.encoders[encoder_name]
                    if val not in encoder.transform(encoder.classes_):
                        logger.warning(f"Encoding mismatch for {encoder_name}: {val} not in model encoder classes")

            # Create feature dataframe
            features = pd.DataFrame({
                'Crop_Year': [year],
                'Area (Hectares)': [area],
                'Latitude': [lat],
                'Longitude': [lon],
                'Rainfall - Kharif (mm)': [rainfall_data.get('Kharif', 0)],
                'Rainfall - Rabi (mm)': [rainfall_data.get('Rabi', 0)],
                'Rainfall - Zaid (mm)': [rainfall_data.get('Zaid', 0)],
                'Rainfall - Whole Year (mm)': [rainfall_data.get('Whole Year', 0)],
                'State_Name_encoded': [encoded['State_Name_encoded']],
                'District_Name_encoded': [encoded['District_Name_encoded']],
                'Crop_encoded': [encoded['Crop_encoded']],
                'Season_encoded': [encoded['Season_encoded']]
            })

            return features[self.feature_columns]
        except Exception as e:
            logger.error(f"❌ Error preparing features: {e}")
            raise

    def make_prediction(self, features):
        """Make prediction using the loaded model (no conversion factors)"""
        try:
            if self.model is None:
                raise ValueError("Model not loaded")
            prediction = self.model.predict(features)[0]
            # No conversion factor applied, return raw model output
            logger.info(f"Raw model prediction: {prediction}")
            return max(0, prediction)  # Ensure non-negative
        except Exception as e:
            logger.error(f"❌ Error making prediction: {e}")
            raise

    def apply_yield_adjustment(self, raw_prediction, features):
        """Yield calculation only, no conversion factor"""
        try:
            area = features.iloc[0]['Area (Hectares)']
            yield_per_hectare = raw_prediction / area if area > 0 else 0
            logger.info(f"Yield per hectare: {yield_per_hectare:.2f}")
            return yield_per_hectare
        except Exception as e:
            logger.error(f"❌ Error in yield adjustment: {e}")
            return raw_prediction
    
    def get_prediction_confidence(self, features):
        """Calculate prediction confidence based on feature values"""
        try:
            # Use tree-based model's prediction variance as confidence measure
            if hasattr(self.model, 'estimators_'):
                predictions = [tree.predict(features)[0] for tree in self.model.estimators_]
                variance = np.var(predictions)
                
                # Convert variance to confidence percentage (0-100)
                # Lower variance = higher confidence
                confidence = max(0, min(100, 100 - (variance / 1000000)))
                return confidence
            else:
                return 85.0  # Default confidence
                
        except Exception as e:
            logger.error(f"❌ Error calculating confidence: {e}")
            return 75.0  # Default fallback confidence

# Global instance
prediction_service = PredictionService()
