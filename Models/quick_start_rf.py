# QUICK START: RANDOM FOREST MODEL
# ================================

"""
QUICK START SCRIPT: Train a Random Forest model on your cleaned dataset
Run this script to get a working model in under 5 minutes!
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib


print("Quick Start: Random Forest Crop Prediction Model")
print("=" * 60)


# 1. Load the dataset
print("Loading dataset...")


# Try to load the cleaned dataset first, fallback to original
import os
if os.path.exists('dataset_improved_cleaned.csv'):
    print("Using cleaned dataset: dataset_improved_cleaned.csv")
    df = pd.read_csv('dataset_improved_cleaned.csv')
    print(f"Loaded {len(df):,} rows from cleaned dataset")
else:
    print("Cleaned dataset not found, using original...")
    df = pd.read_csv('main_dataset_final_cleaned.csv')
    print(f"Loaded {len(df):,} rows from original dataset")


# 2. Prepare features for the model
print("Preparing features...")


# Encode categorical variables (convert text to numbers)
label_encoders = {}
categorical_cols = ['State_Name', 'District_Name', 'Crop', 'Season']
for col in categorical_cols:
    le = LabelEncoder()
    df[f'{col}_encoded'] = le.fit_transform(df[col])
    label_encoders[col] = le


# Select features for training
feature_columns = [
    'Crop_Year', 'Area (Hectares)', 'Latitude', 'Longitude',
    'Rainfall - Kharif (mm)', 'Rainfall - Rabi (mm)', 
    'Rainfall - Zaid (mm)', 'Rainfall - Whole Year (mm)',
    'State_Name_encoded', 'District_Name_encoded', 
    'Crop_encoded', 'Season_encoded'
]
X = df[feature_columns]


# Find the correct production column (target variable)
production_col = None
for col in df.columns:
    if 'production' in col.lower():
        production_col = col
        break
if production_col is None:
    production_col = 'Production (Metric Tons)'
y = df[production_col]
print(f"Using target column: '{production_col}'")
print(f"Number of features: {len(feature_columns)}")


# 3. Split data into training and testing sets
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training samples: {len(X_train):,}, Testing samples: {len(X_test):,}")


# 4. Train the Random Forest model
print("Training Random Forest model...")
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
print("Model training complete.")


# 5. Evaluate the model
print("Evaluating model performance...")
y_train_pred = rf_model.predict(X_train)
y_test_pred = rf_model.predict(X_test)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_mae = mean_absolute_error(y_test, y_test_pred)

print("\nModel Results:")
print(f"Training R^2: {train_r2:.4f}")
print(f"Testing R^2:  {test_r2:.4f}")
print(f"RMSE:         {test_rmse:,.0f} Metric Tons")
print(f"MAE:          {test_mae:,.0f} Metric Tons")


# 6. Show the most important features
print("\nTop 5 Important Features:")
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)
for i, row in feature_importance.head(5).iterrows():
    print(f"{row['feature']:<25} {row['importance']:.4f}")


# 7. Save the trained model and encoders
print("\nSaving model to file...")
model_data = {
    'model': rf_model,
    'label_encoders': label_encoders,
    'feature_columns': feature_columns
}
joblib.dump(model_data, 'quick_rf_model.pkl')
print("Model saved as 'quick_rf_model.pkl'")

print("\nQuick Start Complete!")
print("Your Random Forest model is ready to use!")


# Example prediction function for using the trained model
def make_prediction(state, district, crop, season, year, area, lat, lon, 
                   rainfall_k, rainfall_r, rainfall_z, rainfall_w):
    """
    Make a prediction using the trained model and label encoders.
    Returns the predicted value for the given input.
    """
    input_data = pd.DataFrame({
        'Crop_Year': [year],
        'Area (Hectares)': [area],
        'Latitude': [lat],
        'Longitude': [lon],
        'Rainfall - Kharif (mm)': [rainfall_k],
        'Rainfall - Rabi (mm)': [rainfall_r],
        'Rainfall - Zaid (mm)': [rainfall_z],
        'Rainfall - Whole Year (mm)': [rainfall_w],
        'State_Name_encoded': [label_encoders['State_Name'].transform([state])[0]],
        'District_Name_encoded': [label_encoders['District_Name'].transform([district])[0]],
        'Crop_encoded': [label_encoders['Crop'].transform([crop])[0]],
        'Season_encoded': [label_encoders['Season'].transform([season])[0]]
    })
    prediction = rf_model.predict(input_data)[0]
    return prediction

print("\nYou can use make_prediction() to predict crop production.")
print("Example: make_prediction('UTTAR PRADESH', 'AGRA', 'WHEAT', 'RABI', 2020, 1000, 27.2, 78.0, 100, 50, 0, 800)")

print("\n" + "="*60)
print("Happy Crop Prediction!")
print("="*60)
