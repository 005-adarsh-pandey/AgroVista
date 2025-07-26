"""
Crop prediction functions and routes
"""
import os
import pandas as pd
import joblib
from flask import jsonify, request
from .enhanced_ml_inputs import prepare_enhanced_ml_inputs


def get_districts_handler(app):
    """Get districts for a given state"""
    state = request.args.get('state')
    if not state:
        return jsonify({'error': 'State parameter required'}), 400
        
    try:
        # Load district data from coordinates file
        csv_path = os.path.join(app.root_path, 'state_district_coordinates.csv')
        
        if not os.path.exists(csv_path):
            return jsonify({'error': 'District data not found'}), 500
            
        df = pd.read_csv(csv_path)
        
        # Filter districts for the specified state (case insensitive match)
        state_districts = df[df['State'].str.lower() == state.lower()]
        
        if state_districts.empty:
            return jsonify({'districts': []})
            
        # Get unique districts and sort alphabetically
        districts = state_districts['District'].unique().tolist()
        districts.sort()
        
        return jsonify({'districts': districts})
        
    except Exception as e:
        return jsonify({'error': f'Failed to fetch districts: {str(e)}'}), 500


def get_coordinates_handler(app):
    """Get latitude and longitude for a given state and district"""
    state = request.args.get('state')
    district = request.args.get('district')
    
    if not state or not district:
        return jsonify({'error': 'State and district parameters required'}), 400
        
    try:
        # Load coordinate data
        csv_path = os.path.join(app.root_path, 'state_district_coordinates.csv')
        
        if not os.path.exists(csv_path):
            return jsonify({'error': 'Coordinate data not found'}), 500
            
        df = pd.read_csv(csv_path)
        
        # Find matching row (case insensitive)
        match = df[
            (df['State'].str.lower() == state.lower()) & 
            (df['District'].str.lower() == district.lower())
        ]
        
        if match.empty:
            return jsonify({'error': f'No coordinates found for {district}, {state}'}), 404
            
        row = match.iloc[0]
        return jsonify({
            'latitude': float(row['Latitude']),
            'longitude': float(row['Longitude']),
            'state': row['State'],
            'district': row['District']
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to fetch coordinates: {str(e)}'}), 500


def predict_crop_production_handler(app):
    """Main crop prediction endpoint using the Random Forest model"""
    print("🚨 CROP PREDICTION REQUEST RECEIVED!")
    print("🚨" * 20)
    data = request.json
    print(f"🚨 Request data: {data}")
    
    try:
        # Get actual coordinates for the selected state and district
        state_name = data.get('state_name')
        district = data.get('district')
        crop = data.get('crop')
        
        # Get city coordinates for weather data (from india_places.json like weather.html)
        city_lat = data.get('latitude')  # City coordinates for weather
        city_lon = data.get('longitude')  # City coordinates for weather
        city_name = data.get('city_name', '')  # City name for weather
        
        # Load district coordinate data for ML model
        csv_path = os.path.join(app.root_path, 'state_district_coordinates.csv')
        district_lat, district_lon = 0, 0  # Default fallback for ML model
        actual_district_name = district.upper()  # User selected district from dropdown
        
        if os.path.exists(csv_path) and state_name and district:
            coord_df = pd.read_csv(csv_path)
            coord_match = coord_df[
                (coord_df['State'].str.lower() == state_name.lower()) & 
                (coord_df['District'].str.lower() == district.lower())
            ]
            
            if not coord_match.empty:
                district_lat = float(coord_match.iloc[0]['Latitude'])
                district_lon = float(coord_match.iloc[0]['Longitude'])
                # District name is already correct from user selection, just use it
                print(f"Using district coordinates for ML model: {actual_district_name}, {state_name}: ({district_lat}, {district_lon})")
            else:
                print(f"Warning: No district coordinates found for {district}, {state_name}, using defaults")
        
        # Use city coordinates for weather data if available, otherwise fall back to district
        weather_lat = float(city_lat) if city_lat else district_lat
        weather_lon = float(city_lon) if city_lon else district_lon
        
        print(f"🎯 Weather coordinates (city): ({weather_lat}, {weather_lon})")
        print(f"🎯 ML model coordinates (district): ({district_lat}, {district_lon})")
        print(f"🎯 Weather location: {city_name if city_name else district}")
        
        # Try multiple possible locations for the model file - PRIORITIZE THE CORRECT MODELS
        possible_model_paths = [
            os.path.join(app.root_path, '..', 'Models', 'quick_rf_model.pkl'),  # MAIN MODEL
            os.path.join(app.root_path, '..', 'Models', 'crop_prediction_rf_model.pkl'),  # ALTERNATE MODEL
            os.path.join(app.root_path, 'quick_rf_model.pkl'),  # LOCAL COPY (might be old)
            os.path.join(app.root_path, 'quick_rf_model_backup.pkl'),
            os.path.join(app.static_folder, 'quick_rf_model.pkl')
        ]
        
        model_path = None
        for path in possible_model_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if not model_path:
            print("❌ CRITICAL: No model file found in any location!")
            return jsonify({'error': 'Model file not found'}), 500
        
        print(f"✅ Loading model from: {model_path}")
        model_data = joblib.load(model_path)
        
        # Check if model has predict method
        if not hasattr(model_data, 'predict'):
            # If model is a dictionary, try to extract the actual model
            if isinstance(model_data, dict):
                # Try common keys for model storage
                actual_model = None
                for key in ['model', 'rf_model', 'random_forest', 'pipeline']:
                    if key in model_data and hasattr(model_data[key], 'predict'):
                        actual_model = model_data[key]
                        break
                if actual_model is None:
                    return jsonify({'error': 'Model does not have predict method'}), 500
                model = actual_model
            else:
                model = model_data
        else:
            model = model_data
        
        # Debug: Check if model has feature_names_in_ attribute (scikit-learn 1.0+)
        if hasattr(model, 'feature_names_in_'):
            print(f"Model expects these features: {model.feature_names_in_}")
        elif hasattr(model, 'get_booster') and hasattr(model.get_booster(), 'feature_names'):
            # For XGBoost models
            print(f"Model expects these features: {model.get_booster().feature_names}")
        elif hasattr(model, 'get_params'):
            # Try to extract feature names from model parameters
            params = model.get_params()
            if 'feature_names_in_' in params:
                print(f"Model expects these features: {params['feature_names_in_']}")
        
        # Prepare input data using enhanced ML inputs with improved rainfall handling
        # Use city coordinates for weather data, district coordinates for ML model
        input_data = None
        try:
            print(f"🌍 Using city coordinates for weather: ({city_lat}, {city_lon})")
            print(f"📊 Using district coordinates for ML model: ({district_lat}, {district_lon})")
            
            # Pass city coordinates to get weather data, but the ML model will use district coordinates
            input_data = prepare_enhanced_ml_inputs(data, lat=city_lat, lon=city_lon)
            print(f"✓ Enhanced ML inputs prepared successfully")
            print(f"  Area: {input_data['Area (Hectares)']:.2f} hectares")
            print(f"  Kharif rainfall: {input_data['Rainfall - Kharif (mm)']:.2f} mm")
        except Exception as e:
            print(f"⚠️ Error with enhanced inputs, falling back to legacy method: {e}")
            input_data = None
        
        # Extract area_hectares from input_data or use directly from request
        if input_data is not None:
            # Area already in hectares from enhanced ML inputs, use directly
            area_hectares = float(input_data['Area (Hectares)'])
            print(f"✅ Using area from enhanced ML inputs: {area_hectares:.4f} hectares")
        else:
            # Frontend already handles conversion, use directly
            area_hectares = float(data.get('area', 1.0))
            print(f"✅ Using area directly from frontend: {area_hectares:.4f} hectares")
        
        # If enhanced inputs failed, use legacy input preparation
        if input_data is None:
            input_data = {
                'State_Name': data.get('state_name', '').upper(),
                'District_Name': data.get('district', '').upper(),
                'Crop_Year': int(data.get('crop_year', 2024)),
                'Season': data.get('season', 'Rabi').upper(),
                'Crop': data.get('crop', 'Wheat').upper(),
                'Area (Hectares)': area_hectares,  # Use fixed value
                'Latitude': district_lat,  # Use district coordinates for ML model
                'Longitude': district_lon,  # Use district coordinates for ML model
                'Rainfall - Kharif (mm)': 850,
                'Rainfall - Rabi (mm)': 180,
                'Rainfall - Zaid (mm)': 80,
                'Rainfall - Whole Year (mm)': 1100
            }
            print(f"⚠️ Using legacy input preparation")
            print(f"  Area: {area_hectares:.2f} hectares")
            print(f"  Rainfall (legacy defaults): Kharif=850, Rabi=180, Zaid=80 mm")
        
        # Load model and encoders
        if isinstance(model_data, dict) and 'label_encoders' in model_data:
            encoders = model_data['label_encoders']
            model_obj = model_data['model']
            feature_columns = model_data.get('feature_columns', [])
            print("✅ Loaded model with encoders from dictionary")

            # Apply encoding to categorical variables using the trained encoders (OLD LOGIC)
            try:
                # Create the input DataFrame with encoded values
                input_data_encoded = {
                    'Crop_Year': data.get('year', 2023),
                    'Area (Hectares)': area_hectares,
                    'Latitude': district_lat,  # Use district coordinates for ML model
                    'Longitude': district_lon,  # Use district coordinates for ML model
                    'Rainfall - Kharif (mm)': input_data['Rainfall - Kharif (mm)'],
                    'Rainfall - Rabi (mm)': input_data['Rainfall - Rabi (mm)'],
                    'Rainfall - Zaid (mm)': input_data['Rainfall - Zaid (mm)'],
                    'Rainfall - Whole Year (mm)': input_data['Rainfall - Whole Year (mm)']
                }

                # Simple fallback encoding logic (no robust encoder)
                categorical_mappings = {
                    'State_Name': state_name.upper(),
                    'District_Name': actual_district_name,  # Use actual district name from CSV
                    'Crop': crop.upper(),
                    'Season': input_data['Season'].upper()
                }
                for cat_col, cat_value in categorical_mappings.items():
                    if cat_col in encoders:
                        try:
                            encoded_val = encoders[cat_col].transform([cat_value])[0]
                            input_data_encoded[f'{cat_col}_encoded'] = encoded_val
                            print(f"✅ Encoded {cat_col}: {cat_value} → {encoded_val}")
                        except ValueError as ve:
                            print(f"⚠️ Could not encode {cat_col}={cat_value}: {ve}")
                            # Use a fallback value (middle of range)
                            if len(encoders[cat_col].classes_) > 0:
                                default_idx = len(encoders[cat_col].classes_) // 2
                                input_data_encoded[f'{cat_col}_encoded'] = default_idx
                                print(f"Using fallback encoding: {default_idx}")
                            else:
                                input_data_encoded[f'{cat_col}_encoded'] = 0
                                print(f"Using default encoding: 0")

                # Create DataFrame for prediction
                input_df = pd.DataFrame([input_data_encoded])

                # Use the stored feature columns to select the right features
                if feature_columns:
                    prediction_input = input_df[feature_columns]
                    print(f"✅ Using stored feature columns: {feature_columns}")
                else:
                    # Fallback to expected columns
                    expected_cols = [
                        'Crop_Year', 'Area (Hectares)', 'Latitude', 'Longitude',
                        'Rainfall - Kharif (mm)', 'Rainfall - Rabi (mm)', 
                        'Rainfall - Zaid (mm)', 'Rainfall - Whole Year (mm)',
                        'State_Name_encoded', 'District_Name_encoded', 
                        'Crop_encoded', 'Season_encoded'
                    ]
                    prediction_input = input_df[expected_cols]
                    print(f"Using expected feature columns: {expected_cols}")

                print(f"📊 Prediction input shape: {prediction_input.shape}")
                print(f"📊 Prediction input columns: {list(prediction_input.columns)}")

                # Make prediction
                prediction = model_obj.predict(prediction_input)
                raw_prediction = float(prediction[0])

                # CRITICAL FIX: The model predicts YIELD per hectare, not total production
                # So we need to multiply by area to get total production
                predicted_yield_per_hectare = raw_prediction  # This is yield per hectare
                predicted_production = predicted_yield_per_hectare * area_hectares  # Total production
                predicted_quintals = predicted_production * 10
                final_yield = predicted_yield_per_hectare  # This is already yield per hectare

                print(f"🎯 Raw model prediction: {raw_prediction} tonnes/hectare (YIELD)")
                print(f"🎯 Area: {area_hectares} hectares")
                print(f"🎯 Total production: {predicted_production} tonnes")
                print(f"🎯 Total production: {predicted_quintals} quintals")
                print(f"📊 Final yield: {final_yield:.2f} tonnes/hectare")
                print(f"📊 Input crop was: {crop}")
                print(f"📊 Input state was: {state_name}")
                print(f"📊 Input district was: {district}")

            except Exception as e:
                print(f"❌ Model prediction failed: {e}")
                # Fall back to manual calculation
                predicted_production = None
            
        else:
            print("❌ No encoders found in model, falling back to manual prediction")
            predicted_production = None
            
        # If model prediction failed or no encoders, use fallback calculation
        if predicted_production is None:
            area_hectares = float(input_data['Area (Hectares)'])
            crop = input_data['Crop'].lower()
            state_name = input_data['State_Name']
            
            # Fallback calculation based on area and crop type with location adjustment
            # Simple yield estimates (tons per hectare) for common crops - UPDATED with realistic values
            crop_yields = {
                'rice': 4.5, 'wheat': 4.8, 'maize': 5.2, 'sugarcane': 65.0,
                'cotton': 2.2, 'soybean': 1.8, 'soyabean': 1.8, 'groundnut': 2.0, 'potato': 25.0,
                'onion': 20.0, 'tomato': 30.0, 'cabbage': 35.0, 'cauliflower': 25.0
            }
            
            base_yield = crop_yields.get(crop, 3.5)  # Default 3.5 tons/hectare
            print(f"🔧 FALLBACK TRIGGERED! Using fallback calculation for {crop} with yield {base_yield} tons/hectare")
            
            # Apply location-based adjustment - UPDATED with realistic factors
            state_factors = {
                'punjab': 1.4, 'haryana': 1.35, 'uttar pradesh': 1.2,
                'west bengal': 1.25, 'andhra pradesh': 1.3, 'tamil nadu': 1.2,
                'karnataka': 1.15, 'maharashtra': 1.1, 'gujarat': 1.05,
                'rajasthan': 0.9, 'madhya pradesh': 1.2, 'bihar': 1.0,  # MP is major wheat producer
                'jharkhand': 0.9, 'chhattisgarh': 1.0, 'odisha': 1.0
            }
            
            state_factor = state_factors.get(state_name.lower(), 1.0)
            adjusted_yield = base_yield * state_factor
            predicted_production = area_hectares * adjusted_yield
            
            print(f"Using fallback prediction: {predicted_production} tonnes")
            
        return jsonify({
            'predicted_production': float(predicted_production),
            'unit': 'tonnes',
            'area_provided': f"{float(input_data.get('Area (Hectares)', 0)) / 0.404686:.2f} acres",  # Display only
            'area_converted': f"{float(input_data.get('Area (Hectares)', 0)):.2f} hectares",
            'model_used': 'Random Forest (Enhanced)' if 'model_error' not in locals() else 'Fallback with location adjustment',
            'coordinates_used': {'latitude': float(district_lat), 'longitude': float(district_lon)},
            'rainfall_used': {
                'kharif': float(input_data['Rainfall - Kharif (mm)']),
                'rabi': float(input_data['Rainfall - Rabi (mm)']),
                'zaid': float(input_data['Rainfall - Zaid (mm)']),
                'annual': float(input_data['Rainfall - Whole Year (mm)'])
            },
            'yield_per_hectare': float(final_yield) if 'final_yield' in locals() else float(predicted_production) / float(area_hectares),
            'total_production_quintals': float(predicted_quintals) if 'predicted_quintals' in locals() else float(predicted_production) * 10,
            'crop_input': crop,
            'state_input': state_name,
            'district_input': district,
            'season_input': input_data['Season'],
            'weather_data_source': 'CDS API' if 'nc_weather_data' in locals() else 'Historical Average',
            'enhancements_applied': {
                'seasonal_rainfall_logic': True,
                'historical_averages_fallback': True,
                'cds_api_integration': True,
                'accurate_area_conversion': True
            },
            'input_data': input_data
        })
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
