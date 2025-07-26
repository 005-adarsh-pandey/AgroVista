#!/usr/bin/env python3
"""
Robust Label Encoding Handler for Crop Prediction System
Handles unseen districts, crops, and states with intelligent fallbacks
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
import re
from typing import Dict, Any, Tuple, Optional

class RobustLabelEncoder:
    """Enhanced label encoder with fallback mechanisms for unseen labels"""
    
    def __init__(self, encoders_path: str):
        """Initialize with path to existing encoders"""
        self.encoders_path = encoders_path
        self.label_encoders = {}
        self.fallback_mappings = {}
        self.load_encoders()
        self.setup_fallback_mappings()
    
    def load_encoders(self):
        """Load existing label encoders"""
        try:
            with open(self.encoders_path, 'rb') as f:
                self.label_encoders = pickle.load(f)
            print(f"Loaded encoders for: {list(self.label_encoders.keys())}")
        except Exception as e:
            print(f"Error loading encoders: {e}")
            self.label_encoders = {}
    
    def setup_fallback_mappings(self):
        """Setup intelligent fallback mappings for common variations"""
        
        # District fallback mappings (normalized name -> likely encoded name)
        self.district_fallbacks = {
            'ahmedabad': ['AHMADABAD', 'AHMED ABAD', 'AHMEDABAD'],
            'ahmadabad': ['AHMADABAD', 'AHMEDABAD'],
            'mumbai': ['MUMBAI', 'GREATER MUMBAI'],
            'bangalore': ['BANGALORE', 'BENGALURU', 'BANGALORE URBAN'],
            'chennai': ['CHENNAI', 'MADRAS'],
            'hyderabad': ['HYDERABAD', 'RANGAREDDY'],
            'delhi': ['DELHI', 'NEW DELHI', 'CENTRAL DELHI'],
            'kolkata': ['KOLKATA', 'CALCUTTA'],
            'pune': ['PUNE', 'POONA'],
            'lucknow': ['LUCKNOW'],
            'kanpur': ['KANPUR', 'KANPUR NAGAR'],
        }
        
        # Crop fallback mappings
        self.crop_fallbacks = {
            'cotton': ['COTTON(LINT)', 'Cotton(lint)', 'COTTON', 'Cotton', 'AMERICAN COTTON', 'DESI COTTON'],
            'wheat': ['WHEAT', 'Wheat'],
            'rice': ['RICE', 'Rice', 'PADDY', 'Paddy'],
            'sugarcane': ['SUGARCANE', 'Sugarcane', 'SUGAR CANE'],
            'maize': ['MAIZE', 'Maize', 'CORN', 'Corn'],
            'bajra': ['BAJRA', 'Bajra', 'PEARL MILLET'],
            'jowar': ['JOWAR', 'Jowar', 'SORGHUM'],
            'groundnut': ['GROUNDNUT', 'Groundnut', 'PEANUT'],
            'soybean': ['SOYBEAN', 'Soybean', 'SOYA BEAN'],
            'onion': ['ONION', 'Onion'],
            'potato': ['POTATO', 'Potato'],
            'tomato': ['TOMATO', 'Tomato'],
        }
        
        # State fallback mappings
        self.state_fallbacks = {
            'gujarat': ['GUJARAT', 'Gujarat'],
            'maharashtra': ['MAHARASHTRA', 'Maharashtra'],
            'karnataka': ['KARNATAKA', 'Karnataka'],
            'tamil nadu': ['TAMIL NADU', 'Tamil Nadu', 'TAMILNADU'],
            'telangana': ['TELANGANA', 'Telangana', 'ANDHRA PRADESH'],
            'andhra pradesh': ['ANDHRA PRADESH', 'Andhra Pradesh', 'TELANGANA'],
            'uttar pradesh': ['UTTAR PRADESH', 'Uttar Pradesh'],
            'west bengal': ['WEST BENGAL', 'West Bengal'],
            'rajasthan': ['RAJASTHAN', 'Rajasthan'],
            'madhya pradesh': ['MADHYA PRADESH', 'Madhya Pradesh'],
            'haryana': ['HARYANA', 'Haryana'],
            'punjab': ['PUNJAB', 'Punjab'],
            'bihar': ['BIHAR', 'Bihar'],
            'odisha': ['ODISHA', 'Odisha', 'ORISSA'],
            'kerala': ['KERALA', 'Kerala'],
        }
        
        # Season fallback mappings
        self.season_fallbacks = {
            'kharif': ['KHARIF', 'Kharif', 'MONSOON', 'RAINY'],
            'rabi': ['RABI', 'Rabi', 'WINTER', 'Winter', 'POST-MONSOON'],
            'zaid': ['SUMMER', 'Summer', 'ZAID', 'SUMMER CROP'],
            'summer': ['SUMMER', 'Summer', 'ZAID'],
            'autumn': ['AUTUMN', 'Autumn'],
            'winter': ['WINTER', 'Winter', 'RABI', 'Rabi'],
            'whole year': ['WHOLE YEAR', 'Whole Year', 'ANNUAL', 'YEAR ROUND']
        }
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for better matching"""
        if pd.isna(text) or text is None:
            return ""
        
        # Convert to string and normalize
        text = str(text).strip()
        text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
        text = re.sub(r'\s+', ' ', text)     # Normalize whitespace
        return text.lower()
    
    def find_best_match(self, input_value: str, encoder: LabelEncoder, fallback_dict: Dict) -> Tuple[Optional[str], float]:
        """Find best match for input value in encoder classes"""
        if not input_value or not hasattr(encoder, 'classes_'):
            return None, 0.0
        
        normalized_input = self.normalize_text(input_value)
        
        # Direct match check
        for class_name in encoder.classes_:
            if self.normalize_text(class_name) == normalized_input:
                return class_name, 1.0
        
        # Fallback mapping check
        if normalized_input in fallback_dict:
            for candidate in fallback_dict[normalized_input]:
                if candidate in encoder.classes_:
                    return candidate, 0.9
        
        # Fuzzy matching - check if input is contained in any class name
        best_match = None
        best_score = 0.0
        
        for class_name in encoder.classes_:
            normalized_class = self.normalize_text(class_name)
            
            # Check substring matches
            if normalized_input in normalized_class:
                score = len(normalized_input) / len(normalized_class)
                if score > best_score:
                    best_match = class_name
                    best_score = score
            elif normalized_class in normalized_input:
                score = len(normalized_class) / len(normalized_input)
                if score > best_score:
                    best_match = class_name
                    best_score = score
        
        return best_match, best_score
    
    def encode_with_fallback(self, column_name: str, value: str, use_similar_features: bool = True) -> Tuple[int, Dict[str, Any]]:
        """
        Encode value with intelligent fallback
        Returns: (encoded_value, metadata)
        """
        metadata = {
            'original_value': value,
            'encoded_value': None,
            'match_method': 'none',
            'match_score': 0.0,
            'fallback_used': False
        }
        
        if column_name not in self.label_encoders:
            print(f"Warning: No encoder found for column '{column_name}'")
            metadata['encoded_value'] = 0
            metadata['fallback_used'] = True
            return 0, metadata
        
        encoder = self.label_encoders[column_name]
        
        # Choose appropriate fallback dictionary
        fallback_dict = {}
        if column_name.lower() in ['district_name', 'district']:
            fallback_dict = self.district_fallbacks
        elif column_name.lower() in ['crop']:
            fallback_dict = self.crop_fallbacks
        elif column_name.lower() in ['state_name', 'state']:
            fallback_dict = self.state_fallbacks
        elif column_name.lower() in ['season']:
            fallback_dict = self.season_fallbacks
        
        # Try to find best match
        best_match, score = self.find_best_match(value, encoder, fallback_dict)
        
        if best_match and score > 0.7:  # High confidence match
            try:
                encoded_value = encoder.transform([best_match])[0]
                metadata.update({
                    'encoded_value': encoded_value,
                    'match_method': 'direct' if score == 1.0 else 'fallback',
                    'match_score': score,
                    'matched_to': best_match
                })
                return encoded_value, metadata
            except Exception as e:
                print(f"Error encoding '{best_match}': {e}")
        
        # Fallback strategies
        if use_similar_features:
            # Use most common value (mode) as fallback
            if hasattr(encoder, 'classes_') and len(encoder.classes_) > 0:
                # Use middle value as a reasonable default
                fallback_encoded = len(encoder.classes_) // 2
                metadata.update({
                    'encoded_value': fallback_encoded,
                    'match_method': 'mode_fallback',
                    'fallback_used': True,
                    'fallback_to': encoder.classes_[fallback_encoded]
                })
                return fallback_encoded, metadata
        
        # Final fallback - use 0
        metadata.update({
            'encoded_value': 0,
            'match_method': 'zero_fallback',
            'fallback_used': True
        })
        return 0, metadata
    
    def encode_features(self, features: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Encode all categorical features with fallbacks
        Returns: (encoded_features, encoding_metadata)
        """
        encoded_features = features.copy()
        encoding_metadata = {}
        
        categorical_columns = ['State_Name', 'District_Name', 'Crop', 'Season']
        
        for col in categorical_columns:
            if col in features:
                encoded_value, metadata = self.encode_with_fallback(col, features[col])
                encoded_features[col] = encoded_value
                encoding_metadata[col] = metadata
                
                if metadata['fallback_used']:
                    print(f"Warning: Used fallback for {col}='{features[col]}' -> {encoded_value}")
        
        return encoded_features, encoding_metadata

# Utility function for the main app
def create_robust_encoder(models_path: str = None) -> RobustLabelEncoder:
    """Create a robust encoder instance"""
    if models_path is None:
        models_path = os.path.join(os.path.dirname(__file__), '..', 'Models')
    
    encoders_path = os.path.join(models_path, 'label_encoders.pkl')
    return RobustLabelEncoder(encoders_path)

if __name__ == "__main__":
    # Test the robust encoder
    encoder = create_robust_encoder()
    
    # Test encoding
    test_features = {
        'State_Name': 'GUJARAT',
        'District_Name': 'AHMEDABAD',
        'Crop': 'COTTON',
        'Season': 'KHARIF'
    }
    
    encoded_features, metadata = encoder.encode_features(test_features)
    
    print("=== ROBUST ENCODING TEST ===")
    print(f"Original features: {test_features}")
    print(f"Encoded features: {encoded_features}")
    print(f"Encoding metadata:")
    for col, meta in metadata.items():
        print(f"  {col}: {meta}")
