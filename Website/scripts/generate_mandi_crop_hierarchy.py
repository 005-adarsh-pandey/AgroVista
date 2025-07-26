import os
import pandas as pd
import json
from datetime import datetime, timedelta

# Get absolute path to the root of the project (where app.py is)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Construct correct path to static/data folder
data_dir = os.path.join(project_root, "static", "data")

# Ensure data folder exists
os.makedirs(data_dir, exist_ok=True)

# Try to find the latest mandi file with correct format
today = datetime.now().strftime("%Y_%m_%d")
csv_filename = f"mandi_prices_{today}.csv"
csv_path = os.path.join(data_dir, csv_filename)

# If today's file doesn't exist, try yesterday's
if not os.path.exists(csv_path):
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y_%m_%d")
    csv_filename = f"mandi_prices_{yesterday}.csv"
    csv_path = os.path.join(data_dir, csv_filename)

# Output JSON path
output_path = os.path.join(data_dir, "mandi_crop_hierarchy.json")

# Validate CSV exists
if not os.path.exists(csv_path):
    print(f"❌ File not found: {csv_path}")
    print(f"❌ Also tried: {os.path.join(data_dir, f'mandi_prices_{yesterday}.csv')}")
    exit(1)

# Load the data
df = pd.read_csv(csv_path)

# Prepare hierarchy: State > District > Market > List of Crops
hierarchy = {}

for _, row in df.iterrows():
    state = str(row.get("State")).strip()
    district = str(row.get("District")).strip()
    market = str(row.get("Market")).strip()
    commodity = str(row.get("Commodity")).strip()

    if not all([state, district, market, commodity]):
        continue

    # Build nested structure
    hierarchy.setdefault(state, {}).setdefault(district, {}).setdefault(market, set()).add(commodity)

# Convert sets to sorted lists
for state in hierarchy:
    for district in hierarchy[state]:
        for market in hierarchy[state][district]:
            hierarchy[state][district][market] = sorted(hierarchy[state][district][market])

# Save to JSON
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(hierarchy, f, indent=2, ensure_ascii=False)

print(f"✅ JSON hierarchy saved to: {output_path}")
