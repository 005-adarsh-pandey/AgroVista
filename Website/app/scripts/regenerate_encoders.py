#!/usr/bin/env python3
"""
Regenerate Label Encoders from Training Data
This script ensures the encoders include all possible values seen in the training data
"""

import pandas as pd
import pickle
import os
from sklearn.preprocessing import LabelEncoder

def regenerate_encoders():
    """Regenerate encoders from the training dataset"""
    
    # Load the training dataset
    models_path = os.path.join('..', 'Models')
    dataset_path = os.path.join(models_path, 'main_dataset_final_cleaned.csv')
    
    if not os.path.exists(dataset_path):
        print(f"❌ Training dataset not found at: {dataset_path}")
        return False
    
    print(f"📂 Loading training dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    
    print(f"📊 Dataset shape: {df.shape}")
    print(f"📊 Columns: {list(df.columns)}")
    
    # Define categorical columns to encode
    categorical_columns = []
    
    # Check which categorical columns exist in the dataset
    possible_cat_cols = ['State', 'State_Name', 'District', 'District_Name', 'Crop', 'Season']
    for col in possible_cat_cols:
        if col in df.columns:
            categorical_columns.append(col)
            print(f"✅ Found categorical column: {col}")
    
    if not categorical_columns:
        print("❌ No categorical columns found in dataset")
        return False
    
    # Create new label encoders
    label_encoders = {}
    
    for col in categorical_columns:
        print(f"\n🔄 Processing column: {col}")
        
        # Clean the column data
        df[col] = df[col].astype(str).str.strip()
        unique_values = df[col].unique()
        
        print(f"  📈 Unique values: {len(unique_values)}")
        print(f"  📝 Sample values: {list(unique_values[:10])}")
        
        # Create and fit encoder
        encoder = LabelEncoder()
        encoder.fit(unique_values)
        
        label_encoders[col] = encoder
        print(f"  ✅ Created encoder for {col} with {len(encoder.classes_)} classes")
    
    # Save the new encoders
    encoders_path = os.path.join(models_path, 'label_encoders.pkl')
    backup_path = os.path.join(models_path, 'label_encoders_backup.pkl')
    
    # Backup existing encoders if they exist
    if os.path.exists(encoders_path):
        print(f"📋 Backing up existing encoders to: {backup_path}")
        import shutil
        shutil.copy2(encoders_path, backup_path)
    
    # Save new encoders
    with open(encoders_path, 'wb') as f:
        pickle.dump(label_encoders, f)
    
    print(f"💾 Saved new encoders to: {encoders_path}")
    
    # Verify the saved encoders
    print(f"\n🔍 Verifying saved encoders...")
    with open(encoders_path, 'rb') as f:
        loaded_encoders = pickle.load(f)
    
    for col, encoder in loaded_encoders.items():
        print(f"  ✅ {col}: {len(encoder.classes_)} classes")
        
        # Test with some common values
        test_values = {
            'State': ['GUJARAT', 'MAHARASHTRA', 'UTTAR PRADESH'],
            'State_Name': ['GUJARAT', 'MAHARASHTRA', 'UTTAR PRADESH'],
            'District': ['AHMEDABAD', 'PUNE', 'LUCKNOW'],
            'District_Name': ['AHMEDABAD', 'PUNE', 'LUCKNOW'],
            'Crop': ['WHEAT', 'RICE', 'COTTON'],
            'Season': ['KHARIF', 'RABI']
        }
        
        if col in test_values:
            for test_val in test_values[col]:
                if test_val in encoder.classes_:
                    encoded = encoder.transform([test_val])[0]
                    print(f"    ✅ {test_val} → {encoded}")
                else:
                    print(f"    ⚠️ {test_val} not found in encoder")
    
    return True

if __name__ == "__main__":
    print("🔄 REGENERATING LABEL ENCODERS FROM TRAINING DATA")
    print("=" * 60)
    
    success = regenerate_encoders()
    
    if success:
        print("\n✅ Successfully regenerated label encoders!")
        print("   The new encoders should handle all values seen in training data.")
    else:
        print("\n❌ Failed to regenerate label encoders.")
    
    print("=" * 60)
