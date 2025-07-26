# COMPREHENSIVE RANDOM FOREST MODEL PIPELINE
# ==========================================

"""
EFFICIENT RANDOM FOREST MODEL DEVELOPMENT PIPELINE
for Agricultural Crop Production Prediction

This guide provides a complete, optimized workflow for building
a Random Forest model on your cleaned agricultural dataset.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class CropProductionPredictor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.label_encoders = {}
        self.scaler = None
        
    def load_and_explore_data(self):
        """Step 1: Load and explore the dataset"""
        print("=" * 50)
        print("STEP 1: LOADING AND EXPLORING DATA")
        print("=" * 50)
        
        self.df = pd.read_csv(self.data_path)
        print(f"✅ Dataset loaded: {len(self.df):,} rows, {len(self.df.columns)} columns")
        
        # Basic info
        print(f"\n📊 Basic Information:")
        print(f"Date range: {self.df['Crop_Year'].min()} - {self.df['Crop_Year'].max()}")
        print(f"States: {self.df['State_Name'].nunique()}")
        print(f"Districts: {self.df['District_Name'].nunique()}")
        print(f"Crops: {self.df['Crop'].nunique()}")
        print(f"Seasons: {self.df['Season'].nunique()}")
        
        # Target variable statistics
        production = self.df['Production (Metric Tonnes)']
        print(f"\n🎯 Target Variable (Production) Statistics:")
        print(f"Range: {production.min():,.0f} - {production.max():,.0f} MT")
        print(f"Mean: {production.mean():,.0f} MT")
        print(f"Median: {production.median():,.0f} MT")
        print(f"Skewness: {production.skew():.2f}")
        
        return self.df.head()
    
    def feature_engineering(self):
        """Step 2: Feature engineering and preprocessing"""
        print("\n" + "=" * 50)
        print("STEP 2: FEATURE ENGINEERING")
        print("=" * 50)
        
        # Create a copy for processing
        df_processed = self.df.copy()
        
        # 1. Create additional features
        print("🔧 Creating additional features...")
        
        # Yield feature (Production per unit area)
        df_processed['Yield'] = df_processed['Production (Metric Tonnes)'] / df_processed['Area (Hectares)']
        
        # Total rainfall
        df_processed['Total_Rainfall'] = (df_processed['Rainfall - Kharif (mm)'] + 
                                        df_processed['Rainfall - Rabi (mm)'] + 
                                        df_processed['Rainfall - Zaid (mm)'])
        
        # Rainfall efficiency (production per mm of rainfall)
        df_processed['Rainfall_Efficiency'] = df_processed['Production (Metric Tonnes)'] / (df_processed['Total_Rainfall'] + 1)
        
        # Year-based features
        df_processed['Years_Since_Start'] = df_processed['Crop_Year'] - df_processed['Crop_Year'].min()
        
        # 2. Handle categorical variables
        print("🏷️ Encoding categorical variables...")
        
        categorical_cols = ['State_Name', 'District_Name', 'Crop', 'Season']
        
        for col in categorical_cols:
            le = LabelEncoder()
            df_processed[f'{col}_encoded'] = le.fit_transform(df_processed[col])
            self.label_encoders[col] = le
            print(f"   {col}: {len(le.classes_)} categories")
        
        # 3. Feature selection
        print("📋 Selecting features for modeling...")
        
        feature_columns = [
            # Original numerical features
            'Crop_Year', 'Area (Hectares)', 'Latitude', 'Longitude',
            'Rainfall - Kharif (mm)', 'Rainfall - Rabi (mm)', 
            'Rainfall - Zaid (mm)', 'Rainfall - Whole Year (mm)',
            
            # Engineered features
            'Total_Rainfall', 'Years_Since_Start',
            
            # Encoded categorical features
            'State_Name_encoded', 'District_Name_encoded', 
            'Crop_encoded', 'Season_encoded'
        ]
        
        # Prepare X and y
        X = df_processed[feature_columns]
        y = df_processed['Production (Metric Tonnes)']
        
        print(f"✅ Features selected: {len(feature_columns)}")
        print(f"✅ Target variable: Production (Metric Tonnes)")
        
        return X, y
    
    def split_and_scale_data(self, X, y, test_size=0.2, random_state=42):
        """Step 3: Split data and scale features"""
        print("\n" + "=" * 50)
        print("STEP 3: DATA SPLITTING AND SCALING")
        print("=" * 50)
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=None
        )
        
        print(f"✅ Data split:")
        print(f"   Training set: {len(self.X_train):,} samples")
        print(f"   Test set: {len(self.X_test):,} samples")
        print(f"   Split ratio: {(1-test_size)*100:.0f}% train, {test_size*100:.0f}% test")
        
        # Scale features (optional for Random Forest, but can help with interpretability)
        print("📏 Scaling features...")
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Convert back to DataFrame for easier handling
        self.X_train_scaled = pd.DataFrame(self.X_train_scaled, columns=self.X_train.columns)
        self.X_test_scaled = pd.DataFrame(self.X_test_scaled, columns=self.X_test.columns)
        
        print("✅ Features scaled using RobustScaler")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_random_forest(self, use_grid_search=True):
        """Step 4: Train Random Forest model with hyperparameter tuning"""
        print("\n" + "=" * 50)
        print("STEP 4: RANDOM FOREST TRAINING")
        print("=" * 50)
        
        if use_grid_search:
            print("🔍 Performing hyperparameter tuning with GridSearchCV...")
            
            # Define parameter grid
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }
            
            # Create base model
            rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)
            
            # Grid search with cross-validation
            grid_search = GridSearchCV(
                estimator=rf_base,
                param_grid=param_grid,
                cv=5,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=1
            )
            
            # Fit grid search
            grid_search.fit(self.X_train, self.y_train)
            
            # Get best model
            self.model = grid_search.best_estimator_
            
            print(f"✅ Best parameters found:")
            for param, value in grid_search.best_params_.items():
                print(f"   {param}: {value}")
            
            print(f"✅ Best cross-validation score: {-grid_search.best_score_:.2f}")
            
        else:
            print("🚀 Training Random Forest with default parameters...")
            
            # Train with reasonable default parameters
            self.model = RandomForestRegressor(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            )
            
            self.model.fit(self.X_train, self.y_train)
            print("✅ Model trained successfully")
        
        return self.model
    
    def evaluate_model(self):
        """Step 5: Evaluate model performance"""
        print("\n" + "=" * 50)
        print("STEP 5: MODEL EVALUATION")
        print("=" * 50)
        
        # Make predictions
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)
        
        # Calculate metrics
        train_r2 = r2_score(self.y_train, y_train_pred)
        test_r2 = r2_score(self.y_test, y_test_pred)
        
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_test_pred))
        
        train_mae = mean_absolute_error(self.y_train, y_train_pred)
        test_mae = mean_absolute_error(self.y_test, y_test_pred)
        
        print("📊 Model Performance Metrics:")
        print(f"{'Metric':<15} {'Training':<15} {'Testing':<15}")
        print("-" * 45)
        print(f"{'R² Score':<15} {train_r2:<15.4f} {test_r2:<15.4f}")
        print(f"{'RMSE':<15} {train_rmse:<15.2f} {test_rmse:<15.2f}")
        print(f"{'MAE':<15} {train_mae:<15.2f} {test_mae:<15.2f}")
        
        # Cross-validation score
        cv_scores = cross_val_score(self.model, self.X_train, self.y_train, cv=5, scoring='r2')
        print(f"\n🔄 Cross-validation R² scores: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Check for overfitting
        if train_r2 - test_r2 > 0.1:
            print("⚠️  Warning: Model might be overfitting (training R² - test R² > 0.1)")
        else:
            print("✅ Model shows good generalization")
        
        return {
            'train_r2': train_r2, 'test_r2': test_r2,
            'train_rmse': train_rmse, 'test_rmse': test_rmse,
            'train_mae': train_mae, 'test_mae': test_mae,
            'cv_r2_mean': cv_scores.mean(), 'cv_r2_std': cv_scores.std()
        }
    
    def analyze_feature_importance(self):
        """Step 6: Analyze feature importance"""
        print("\n" + "=" * 50)
        print("STEP 6: FEATURE IMPORTANCE ANALYSIS")
        print("=" * 50)
        
        # Get feature importance
        feature_importance = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("🔝 Top 10 Most Important Features:")
        print("-" * 40)
        for i, row in feature_importance.head(10).iterrows():
            print(f"{row['feature']:<25} {row['importance']:.4f}")
        
        return feature_importance
    
    def save_model(self, model_path='crop_prediction_rf_model.pkl'):
        """Step 7: Save the trained model"""
        print("\n" + "=" * 50)
        print("STEP 7: SAVING MODEL")
        print("=" * 50)
        
        # Save model and preprocessors
        model_data = {
            'model': self.model,
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'feature_columns': list(self.X_train.columns)
        }
        
        joblib.dump(model_data, model_path)
        print(f"✅ Model saved to: {model_path}")
        
        return model_path

# Example usage function
def run_complete_pipeline():
    """Run the complete Random Forest pipeline"""
    
    print("🌾 AGRICULTURAL CROP PRODUCTION PREDICTION")
    print("🤖 Random Forest Model Development Pipeline")
    print("=" * 60)
    
    # Initialize predictor
    predictor = CropProductionPredictor('main_dataset_final_cleaned.csv')
    
    # Run the complete pipeline
    try:
        # Step 1: Load and explore
        sample_data = predictor.load_and_explore_data()
        
        # Step 2: Feature engineering
        X, y = predictor.feature_engineering()
        
        # Step 3: Split and scale
        X_train, X_test, y_train, y_test = predictor.split_and_scale_data(X, y)
        
        # Step 4: Train model (set use_grid_search=False for faster training)
        model = predictor.train_random_forest(use_grid_search=False)
        
        # Step 5: Evaluate
        metrics = predictor.evaluate_model()
        
        # Step 6: Feature importance
        feature_importance = predictor.analyze_feature_importance()
        
        # Step 7: Save model
        model_path = predictor.save_model()
        
        print("\n" + "🎉" * 20)
        print("MODEL DEVELOPMENT COMPLETED SUCCESSFULLY!")
        print("🎉" * 20)
        
        return predictor, metrics, feature_importance
        
    except Exception as e:
        print(f"❌ Error in pipeline: {str(e)}")
        return None, None, None

if __name__ == "__main__":
    # Run the complete pipeline
    predictor, metrics, feature_importance = run_complete_pipeline()
