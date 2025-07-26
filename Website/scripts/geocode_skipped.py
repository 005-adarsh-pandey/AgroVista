import pandas as pd
import requests
from datetime import date
import os
import json

# Configuration
API_KEY = "579b464db66ec23bdd0000015129dfcfbc1e44966f3b7f30f62a3b53"
API_URL = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"

# Dynamically get path to static/data (even if this script is inside static/)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Website/
data_dir = os.path.join(project_root, "static", "data")

# File paths
today = date.today().strftime("%Y-%m-%d")
filename = f"mandi_prices_{today}.csv"
save_path = os.path.join(data_dir, filename)

# Ensure folder exists
os.makedirs(data_dir, exist_ok=True)

# API request parameters
params = {
    "api-key": API_KEY,
    "format": "csv",
    "limit": 10000
}

print("🔽 Downloading...")
response = requests.get(API_URL, params=params)
if response.status_code == 200:
    with open(save_path, "wb") as f:
        f.write(response.content)
    print(f"✅ File saved: {save_path}")

    # Load CSV and extract unique crops
    df = pd.read_csv(save_path)
    crops = sorted(df["Commodity"].dropna().unique().tolist())

    # Save crop list to JSON for frontend dropdowns
    crop_json_path = os.path.join(data_dir, "crop_list.json")
    with open(crop_json_path, "w", encoding="utf-8") as f:
        json.dump(crops, f, indent=2, ensure_ascii=False)
    print("🌾 Crops list generated.")
else:
    print(f"❌ Failed to download. Status: {response.status_code}")
