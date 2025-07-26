"""
Automated Mandi Data Downloader
Downloads fresh mandi data every hour during market operating hours (6 AM to 6 PM)
"""
import schedule
import time
import requests
import pandas as pd
import os
import logging
from datetime import datetime, timedelta
import json
from concurrent.futures import ThreadPoolExecutor
import threading

# Setup logging with proper Unicode handling
import sys
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mandi_downloader.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Ensure stdout can handle Unicode
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except:
        pass  # Fallback for older Python versions

class MandiDataDownloader:
    def __init__(self):
        self.data_dir = "static/data"
        self.base_url = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
        self.api_key = "579b464db66ec23bdd000001c967da9cd4b24a5d4bfe4601e18d2521"
        self.max_workers = 5  # Increased for better throughput with larger data
        self.running = False
        self.server_active = False
        self.last_download_time = None
        self.download_success_count = 0
        
        # Ensure data directory exists
        os.makedirs(self.data_dir, exist_ok=True)
        
    def get_states_list(self):
        """Get list of all states from existing data or API"""
        try:
            existing_files = [f for f in os.listdir(self.data_dir) if f.startswith('mandi_prices_') and f.endswith('.csv')]
            
            if existing_files:
                # ⬇️ Extract latest file based on date in filename (YYYY_MM_DD)
                from datetime import datetime

                def extract_date_from_filename(filename):
                    try:
                        date_part = filename.replace('mandi_prices_', '').replace('.csv', '')
                        return datetime.strptime(date_part, '%Y_%m_%d')
                    except:
                        return datetime.min  # Skip badly named files

                latest_file = max(existing_files, key=lambda x: extract_date_from_filename(x))
                df = pd.read_csv(os.path.join(self.data_dir, latest_file))
                
                if 'state' in df.columns:
                    return df['state'].dropna().unique().tolist()
            
            # 🔁 Fallback: full list of known states
            return [
                'Uttar Pradesh', 'Maharashtra', 'Bihar', 'West Bengal', 'Madhya Pradesh',
                'Tamil Nadu', 'Rajasthan', 'Karnataka', 'Gujarat', 'Andhra Pradesh',
                'Odisha', 'Telangana', 'Kerala', 'Jharkhand', 'Assam', 'Punjab',
                'Chhattisgarh', 'Haryana', 'Delhi', 'Jammu and Kashmir', 'Uttarakhand',
                'Himachal Pradesh', 'Tripura', 'Meghalaya', 'Manipur', 'Nagaland',
                'Goa', 'Arunachal Pradesh', 'Mizoram', 'Sikkim'
            ]
        except Exception as e:
            logging.error(f"Error getting states list: {e}")
            return ['Uttar Pradesh', 'Maharashtra', 'Bihar']  # Minimal fallback

    
    def download_state_data(self, state):
        """Download mandi data for a specific state with pagination for complete data"""
        try:
            all_records = []
            offset = 0
            limit = 5000  # Reduced for better reliability
            max_retries = 3
            
            while True:
                params = {
                    'api-key': self.api_key,
                    'format': 'json',
                    'limit': limit,
                    'offset': offset,
                    'filters[state]': state
                }
                
                # Retry logic for API calls
                for attempt in range(max_retries):
                    try:
                        response = requests.get(self.base_url, params=params, timeout=120)
                        response.raise_for_status()
                        
                        data = response.json()
                        if 'records' in data and data['records']:
                            records = data['records']
                            all_records.extend(records)
                            logging.info(f"{state}: Downloaded {len(records)} records (offset {offset})")
                            
                            # If we got less than the limit, we've reached the end
                            if len(records) < limit:
                                logging.info(f"{state}: Completed with {len(all_records)} total records")
                                return all_records
                            
                            offset += limit
                            # Add delay to be nice to the API
                            time.sleep(1)
                            break  # Success, exit retry loop
                        else:
                            logging.info(f"{state}: No more records at offset {offset}")
                            return all_records
                            
                    except (requests.exceptions.Timeout, requests.exceptions.RequestException) as e:
                        logging.warning(f"{state}: API error attempt {attempt+1}/{max_retries}: {e}")
                        if attempt < max_retries - 1:
                            time.sleep(2 ** attempt)  # Exponential backoff
                        else:
                            logging.error(f"{state}: Failed after {max_retries} attempts")
                            return all_records
                    except Exception as e:
                        logging.error(f"{state}: Unexpected error: {e}")
                        return all_records
                
                # Safety check - don't get stuck in infinite loop
                if offset > 50000:  # Max 50k records per state
                    logging.warning(f"{state}: Reached max offset limit")
                    break
            
            return all_records
                
        except Exception as e:
            # Log more specific errors for debugging
            logging.warning(f"State {state} download failed: {str(e)[:50]}...")
            return []
    
    def download_all_mandi_data(self):
        """Download fresh mandi data for all states - only if server is active"""
        if not self.server_active:
            return False
            
        try:
            current_time = datetime.now()
            logging.info(f"Starting mandi data download...")
            
            states = self.get_states_list()
            all_records = []
            successful_states = 0
            
            # Use ThreadPoolExecutor for parallel downloads
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_state = {executor.submit(self.download_state_data, state): state for state in states}
                
                for future in future_to_state:
                    try:
                        records = future.result(timeout=120)  # Increased timeout for larger data
                        if records:
                            all_records.extend(records)
                            successful_states += 1
                    except Exception:
                        pass  # Silent fail - don't spam logs
            
            if all_records:
                # Create DataFrame and save
                new_df = pd.DataFrame(all_records)
                
                # Clean and standardize column names - handle x0020 encoding properly
                new_df.columns = [col.strip().replace(' ', '_').replace('x0020_', '').lower() for col in new_df.columns]
                
                # Generate filename for today's date
                date_str = current_time.strftime('%Y_%m_%d')
                filename = f"mandi_prices_{date_str}.csv"
                filepath = os.path.join(self.data_dir, filename)
                
                # Combine with existing data for today if it exists
                combined_df = self.combine_with_existing_data(new_df, filepath)
                
                # Save the combined data
                combined_df.to_csv(filepath, index=False)
                
                # Update tracking
                self.last_download_time = current_time
                self.download_success_count += 1
                
                logging.info(f"Combined {len(new_df)} new records with existing data - Total: {len(combined_df)} records from {successful_states}/{len(states)} states")
                
                # Update the hierarchy file
                self.update_crop_hierarchy(combined_df)
                
                # Clean up old files - remove other files for today and files older than 7 days
                self.cleanup_old_files()
                
                return True
            else:
                logging.warning("No data downloaded from any state")
                return False
                
        except Exception as e:
            logging.error(f"Download failed: {str(e)[:50]}...")  # Truncated error message
            return False
    
    def update_crop_hierarchy(self, df):
        """Update the mandi crop hierarchy JSON file"""
        try:
            hierarchy = {}
            
            for _, row in df.iterrows():
                state = str(row.get('state', '')).strip()
                district = str(row.get('district', '')).strip()
                market = str(row.get('market', '')).strip()
                commodity = str(row.get('commodity', '')).strip()
                
                if state and district and market and commodity:
                    if state not in hierarchy:
                        hierarchy[state] = {}
                    if district not in hierarchy[state]:
                        hierarchy[state][district] = {}
                    if market not in hierarchy[state][district]:
                        hierarchy[state][district][market] = set()
                    
                    hierarchy[state][district][market].add(commodity)
            
            # Convert sets to sorted lists
            for state in hierarchy:
                for district in hierarchy[state]:
                    for market in hierarchy[state][district]:
                        hierarchy[state][district][market] = sorted(list(hierarchy[state][district][market]))
            
            # Save hierarchy
            hierarchy_file = os.path.join(self.data_dir, 'mandi_crop_hierarchy.json')
            with open(hierarchy_file, 'w', encoding='utf-8') as f:
                json.dump(hierarchy, f, indent=2, ensure_ascii=False)
            
        except Exception:
            pass  # Silent fail
    
    def cleanup_old_files(self):
        """Remove duplicate files and files older than 7 days - keep only one file per day"""
        try:
            current_time = datetime.now()
            current_date_str = current_time.strftime('%Y_%m_%d')
            files_by_date = {}
            
            # Group files by date
            for filename in os.listdir(self.data_dir):
                if filename.startswith('mandi_prices_') and filename.endswith('.csv'):
                    try:
                        # Extract date from filename
                        parts = filename.replace('mandi_prices_', '').replace('.csv', '').split('_')
                        if len(parts) >= 3:
                            date_str = f"{parts[0]}_{parts[1]}_{parts[2]}"  # YYYY_MM_DD
                            filepath = os.path.join(self.data_dir, filename)
                            
                            if os.path.exists(filepath):
                                file_time = os.path.getmtime(filepath)
                                if date_str not in files_by_date:
                                    files_by_date[date_str] = []
                                files_by_date[date_str].append((filename, file_time))
                    except:
                        continue
            
            files_removed = 0
            
            # For each date, keep only the main date file (without timestamp)
            for date_str, files in files_by_date.items():
                main_file = f"mandi_prices_{date_str}.csv"
                files_to_remove = []
                
                # If we have multiple files for the same date
                if len(files) > 1:
                    # Sort files to identify which ones to keep/remove
                    for filename, file_time in files:
                        # Remove files with timestamps (keep only the main daily file)
                        if filename != main_file:
                            files_to_remove.append(filename)
                
                # Remove unnecessary files for this date
                for filename in files_to_remove:
                    try:
                        filepath = os.path.join(self.data_dir, filename)
                        if os.path.exists(filepath):
                            os.remove(filepath)
                            files_removed += 1
                    except:
                        pass
            
            # Remove files older than 7 days
            cutoff_time = current_time - timedelta(days=7)
            for date_str, files in files_by_date.items():
                # Check if this date is older than 7 days
                try:
                    date_parts = date_str.split('_')
                    file_date = datetime(int(date_parts[0]), int(date_parts[1]), int(date_parts[2]))
                    
                    if file_date < cutoff_time:
                        # Remove all files for this old date
                        for filename, _ in files:
                            try:
                                filepath = os.path.join(self.data_dir, filename)
                                if os.path.exists(filepath):
                                    os.remove(filepath)
                                    files_removed += 1
                            except:
                                pass
                except:
                    pass
            
            if files_removed > 0:
                logging.info(f"[CLEANUP] Cleaned {files_removed} duplicate/old files")
                        
        except Exception:
            pass  # Silent fail on cleanup errors
    
    def is_market_hours(self):
        """Check if current time is within market hours (6 AM to 6 PM)"""
        current_hour = datetime.now().hour
        return 6 <= current_hour <= 18
    
    def start_scheduler(self):
        """Start the hourly download scheduler - only when server is active"""
        logging.info("Mandi downloader ready")
        
        # Schedule downloads every hour during market hours
        schedule.every().hour.do(self.scheduled_download)
        
        self.running = True
        
        while self.running:
            try:
                schedule.run_pending()
                time.sleep(300)  # Check every 5 minutes
            except KeyboardInterrupt:
                logging.info("[SCHEDULER] Scheduler stopped")
                self.running = False
                break
            except Exception:
                time.sleep(600)  # Wait 10 minutes on error
    
    def scheduled_download(self):
        """Scheduled download job - only runs during market hours and when server is active"""
        if self.server_active and self.is_market_hours():
            success = self.download_all_mandi_data()
            # Silent operation - no additional logging
    
    def set_server_active(self, active=True):
        """Set server active status"""
        self.server_active = active
        if active:
            logging.info("Server active - mandi downloads enabled")
            # Perform initial download if within market hours
            if self.is_market_hours():
                self.download_all_mandi_data()
        else:
            logging.info("Server inactive - mandi downloads disabled")
    
    def get_status(self):
        """Get downloader status information"""
        return {
            'running': self.running,
            'server_active': self.server_active,
            'last_download': self.last_download_time.isoformat() if self.last_download_time else None,
            'success_count': self.download_success_count,
            'market_hours': self.is_market_hours()
        }
    
    def stop_scheduler(self):
        """Stop the scheduler"""
        self.running = False
        logging.info("[SCHEDULER] Scheduler stop requested")
    
    def combine_with_existing_data(self, new_df, filepath):
        """Combine new data with existing data for the same day, updating prices where newer"""
        try:
            if os.path.exists(filepath):
                # Load existing data
                existing_df = pd.read_csv(filepath)
                
                # Ensure column consistency - handle x0020 encoding
                existing_df.columns = [col.strip().replace(' ', '_').replace('x0020_', '').lower() for col in existing_df.columns]
                
                # Create a unique key for each record (state + district + market + commodity)
                def create_key(df):
                    if all(col in df.columns for col in ['state', 'district', 'market', 'commodity']):
                        return df['state'].astype(str) + '|' + df['district'].astype(str) + '|' + df['market'].astype(str) + '|' + df['commodity'].astype(str)
                    else:
                        # Fallback if columns are missing
                        return df.index.astype(str)
                
                existing_df['_key'] = create_key(existing_df)
                new_df['_key'] = create_key(new_df)
                
                # Convert arrival_date to datetime for comparison
                if 'arrival_date' in existing_df.columns:
                    existing_df['arrival_date'] = pd.to_datetime(existing_df['arrival_date'], errors='coerce')
                if 'arrival_date' in new_df.columns:
                    new_df['arrival_date'] = pd.to_datetime(new_df['arrival_date'], errors='coerce')
                
                # Find new records and updates
                existing_keys = set(existing_df['_key'].values)
                new_keys = set(new_df['_key'].values)
                
                # Records that are completely new
                new_records = new_df[~new_df['_key'].isin(existing_keys)].copy()
                
                # Records that might be updates (same key but potentially newer data)
                potential_updates = new_df[new_df['_key'].isin(existing_keys)].copy()
                
                # Keep existing records that aren't being updated
                unchanged_records = existing_df[~existing_df['_key'].isin(new_keys)].copy()
                
                # For potential updates, keep the one with the latest arrival_date or newer data
                updated_records = []
                for key in existing_keys.intersection(new_keys):
                    existing_record = existing_df[existing_df['_key'] == key].iloc[0]
                    new_record = new_df[new_df['_key'] == key].iloc[0]
                    
                    # Compare by arrival_date if available, otherwise take the new record
                    if ('arrival_date' in existing_df.columns and 'arrival_date' in new_df.columns and 
                        pd.notnull(existing_record['arrival_date']) and pd.notnull(new_record['arrival_date'])):
                        if new_record['arrival_date'] >= existing_record['arrival_date']:
                            updated_records.append(new_record)
                        else:
                            updated_records.append(existing_record)
                    else:
                        # If no date comparison possible, take the new record (assume it's fresher)
                        updated_records.append(new_record)
                
                # Combine all records
                combined_records = []
                if not unchanged_records.empty:
                    combined_records.append(unchanged_records)
                if updated_records:
                    combined_records.append(pd.DataFrame(updated_records))
                if not new_records.empty:
                    combined_records.append(new_records)
                
                if combined_records:
                    combined_df = pd.concat(combined_records, ignore_index=True)
                else:
                    combined_df = new_df.copy()
                
                # Remove the temporary key column
                combined_df = combined_df.drop(columns=['_key'], errors='ignore')
                
                logging.info(f"Combined: {len(new_records)} new, {len(updated_records)} updated, {len(unchanged_records)} unchanged")
                return combined_df
            
            else:
                # No existing file, return new data as is
                return new_df
                
        except Exception as e:
            logging.error(f"Error combining data: {str(e)[:50]}...")
            # Fallback to new data only
            return new_df

def run_as_service():
    """Run the downloader as a background service"""
    downloader = MandiDataDownloader()
    
    try:
        downloader.start_scheduler()
    except Exception as e:
        logging.error(f"Service error: {e}")
    finally:
        downloader.stop_scheduler()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--service":
        # Run as background service
        run_as_service()
    else:
        # Run single download for testing
        downloader = MandiDataDownloader()
        print("🧪 Testing single mandi data download...")
        success = downloader.download_all_mandi_data()
        if success:
            print("Test download completed successfully!")
        else:
            print("Test download failed!")
