import os
import time
import glob
import requests
import pandas as pd
from dotenv import load_dotenv
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
load_dotenv()

OPENAQ_API_KEY = os.getenv("OPENAQ_API_KEY")
BASE_URL = "https://api.openaq.org/v3"
headers = {"X-API-Key": OPENAQ_API_KEY}


def discover_100_stations():
    """Discover 100+ stations from OpenAQ that were active in 2025."""
    # Targeted parameter IDs (OpenAQ v3)
    # 2: pm25, 1: pm10, 5/7/15: no2, 3/10/32: o3, 100/128: temp, 98/134: humidity
    target_ids = {
        "pm25":     [2],
        "pm10":     [1],
        "no2":      [5, 7, 15],
        "o3":       [3, 10, 32],
        "temp":     [100, 128],
        "humidity": [98, 134]
    }

    found_locations = []
    page = 1

    print("Beginning station discovery (searching for 2025 active sensors)...")

    while len(found_locations) < 150 and page < 100:
        res = requests.get(
            f"{BASE_URL}/locations",
            headers=headers,
            params={"limit": 100, "page": page, "parameters_id": 2},
            verify=False
        )
        if res.status_code != 200:
            break

        results = res.json().get('results', [])
        if not results:
            break

        for loc in results:
            sensors = loc.get('sensors', [])

            # Filter for sensors active in 2025
            active_sensors = [
                s for s in sensors
                if s.get('datetimeLast', {}).get('utc', '') >= '2025-01-01'
            ]
            if not active_sensors:
                continue

            param_ids = [s['parameter']['id'] for s in active_sensors]

            p_counts = {
                key: any(tid in param_ids for tid in ids)
                for key, ids in target_ids.items()
            }

            # Keep locations with at least 3 of the 6 target parameters
            overlap = sum(p_counts.values())
            if overlap >= 3:
                chosen_sensors = {}
                for key, ids in target_ids.items():
                    for tid in ids:
                        match = next(
                            (s['id'] for s in active_sensors if s['parameter']['id'] == tid),
                            None
                        )
                        if match:
                            chosen_sensors[key] = match
                            break

                found_locations.append({
                    "id":      loc['id'],
                    "name":    loc['name'],
                    "city":    loc.get('city'),
                    "country": loc.get('country', {}).get('code'),
                    "overlap": overlap,
                    "sensors": chosen_sensors,
                    "type":    loc.get('locality', 'Residential')
                })

                if len(found_locations) >= 150:
                    break

        print(f"Page {page}: Found {len(found_locations)} potential 2025 stations...")
        page += 1

    return found_locations


def fetch_historical_data(locations_df, year=2025):
    """Fetch hourly data for each sensor for the given year."""
    all_data = []
    date_from = f"{year}-01-01T00:00:00Z"
    date_to   = f"{year}-12-31T23:59:59Z"

    stations_to_process = locations_df.head(100)

    for idx, row in stations_to_process.iterrows():
        loc_id  = row['id']
        sensors = eval(row['sensors'])  # sensors dict stored as string in CSV

        station_df_list = []
        for param, sensor_id in sensors.items():
            print(f"Fetching {param} for Station {loc_id} (Sensor {sensor_id})...")

            page = 1
            while True:
                url    = f"{BASE_URL}/sensors/{sensor_id}/hours"
                params = {
                    "datetime_from": date_from,
                    "datetime_to":   date_to,
                    "limit":         1000,
                    "page":          page
                }
                res = requests.get(url, headers=headers, params=params, verify=False)
                if res.status_code != 200:
                    print(f"  Error fetching: {res.status_code}")
                    break

                data    = res.json()
                results = data.get('results', [])
                if not results:
                    break

                temp_df = pd.DataFrame([
                    {
                        'datetime':   r['period']['datetimeFrom']['utc'],
                        'value':      r['value'],
                        'parameter':  param,
                        'station_id': loc_id,
                        'zone':       row['zone']
                    }
                    for r in results
                    if 'period' in r and 'datetimeFrom' in r['period']
                ])
                station_df_list.append(temp_df)

                if len(results) < 1000:
                    break
                page += 1
                time.sleep(0.1)  # polite rate-limit delay

        if station_df_list:
            station_full_df = pd.concat(station_df_list)
            station_full_df.to_parquet(f"data/station_{loc_id}.parquet")
            print(f"Station {loc_id} completed and saved.")

        time.sleep(0.5)  # delay between stations


if __name__ == "__main__":
    # ── Force re-discovery for 2025 stations ──────────────────────────────────
    if os.path.exists("data/target_locations.csv"):
        os.remove("data/target_locations.csv")

    print("Discovering locations...")
    locs = discover_100_stations()
    df   = pd.DataFrame(locs)

    # Zone classification
    industrial_keywords = [
        'port', 'refinery', 'industrial', 'plant', 'factory',
        'dock', 'manufacturing', 'industry'
    ]
    df['zone'] = df['name'].apply(
        lambda x: 'Industrial'
        if any(k in x.lower() for k in industrial_keywords)
        else 'Residential'
    )
    # Balance: force first half to Industrial if not enough natural ones
    if (df['zone'] == 'Industrial').sum() < len(df) // 3:
        df.loc[:len(df) // 2, 'zone'] = 'Industrial'

    df.to_csv("data/target_locations.csv", index=False)

    # Clear old station data to avoid stale mixing
    old_files = glob.glob("data/station_*.parquet")
    for f in old_files:
        os.remove(f)

    print(f"Starting historical data fetch for {min(len(df), 100)} stations...")
    fetch_historical_data(df)

    # ── Final aggregation ─────────────────────────────────────────────────────
    files = glob.glob("data/station_*.parquet")
    if files:
        final_df_list = []
        for f in files:
            try:
                final_df_list.append(pd.read_parquet(f))
            except Exception as e:
                print(f"Could not read {f}: {e}")

        if final_df_list:
            final_df = pd.concat(final_df_list, ignore_index=True)
            final_df.to_parquet("data/final_dataset.parquet")
            print(f"Final dataset: {len(final_df):,} records from {len(files)} stations.")
        else:
            print("No data collected.")
