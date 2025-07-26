import os
import requests
import pandas as pd
from datetime import datetime, timedelta
import io

API_KEY = "579b464db66ec23bdd000001c967da9cd4b24a5d4bfe4601e18d2521"
RESOURCE_ID = "35985678-0d79-46b4-9ed6-6f13308a1d24"
BASE_URL = f"https://api.data.gov.in/resource/{RESOURCE_ID}"
LIMIT = 1000
DATA_DIR = os.path.join(os.getcwd(), "data")
os.makedirs(DATA_DIR, exist_ok=True)

def fetch_data_for_date(arrival_date):
    offset = 0
    all_data = []

    while True:
        print(f"📦 Downloading page {(offset // LIMIT) + 1} with offset {offset} for date {arrival_date}...")

        params = {
            "api-key": API_KEY,
            "format": "csv",
            "limit": LIMIT,
            "offset": offset,
            "filters[Arrival_Date]": arrival_date
        }

        try:
            response = requests.get(BASE_URL, params=params)
            response.raise_for_status()

            csv_data = response.text.strip()
            if not csv_data or "No records found" in csv_data or "html" in csv_data.lower():
                print("⚠️ No data downloaded or invalid response.")
                break

            df = pd.read_csv(io.StringIO(csv_data))
            if df.empty:
                break

            all_data.append(df)
            if len(df) < LIMIT:
                break

            offset += LIMIT

        except Exception as e:
            print(f"❌ Error during request: {e}")
            break

    if all_data:
        full_df = pd.concat(all_data, ignore_index=True)
        file_path = os.path.join(DATA_DIR, f"mandi_prices_{arrival_date.replace('-', '')}.csv")
        full_df.to_csv(file_path, index=False)
        print(f"✅ Saved {len(full_df)} records to {file_path}")
        return full_df
    else:
        print("⚠️ No data collected.")
        return None

def fetch_today_or_yesterday():
    for day_shift in [0, 1]:  # 0 = today, 1 = yesterday
        date_str = (datetime.now() - timedelta(days=day_shift)).strftime("%d-%m-%Y")
        print(f"\n🔎 Trying to download data for {date_str}...")
        df = fetch_data_for_date(date_str)
        if df is not None and len(df) >= 100:  # 👈 You can adjust this threshold
            break
        else:
            print(f"ℹ️ Only {len(df) if df is not None else 0} records for {date_str}, trying previous day...")

if __name__ == "__main__":
    fetch_today_or_yesterday()
