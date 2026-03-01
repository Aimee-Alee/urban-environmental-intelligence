import requests, os, pandas as pd
from dotenv import load_dotenv
load_dotenv()

BASE_URL = "https://api.openaq.org/v3"
headers = {"X-API-Key": os.getenv("OPENAQ_API_KEY")}

def discover():
    found = []
    page = 1
    # Target IDs
    pm25_id = 2
    
    while len(found) < 150 and page < 100:
        print(f"Searching Page {page}...")
        try:
            r = requests.get(f"{BASE_URL}/locations", headers=headers, params={"limit": 100, "page": page, "parameters_id": pm25_id}, verify=False, timeout=15)
            if r.status_code != 200: 
                print(f"Error {r.status_code}")
                break
            
            results = r.json().get('results', [])
            if not results: break
            
            for loc in results:
                sensors = loc.get('sensors', [])
                # Filter for 2025 activity
                active = [s for s in sensors if s.get('datetimeLast', {}).get('utc', '') >= '2025-01-01']
                if len(active) >= 2: # At least 2 active sensors in 2025
                    found.append({
                        "id": loc['id'],
                        "name": loc['name'],
                        "country": loc.get('country', {}).get('code'),
                        "overlap": len(active),
                        "sensors": {s['parameter']['name']: s['id'] for s in active},
                        "zone": "Industrial" if any(k in loc['name'].lower() for k in ['port', 'industry', 'factory', 'plant']) else "Residential"
                    })
                    if len(found) >= 150: break
            page += 1
        except Exception as e:
            print(f"Exception on page {page}: {e}")
            break
            
    df = pd.DataFrame(found)
    print(f"Total discovered: {len(df)}")
    if not df.empty:
        df.to_csv("data/target_locations.csv", index=False)
        print("Saved to data/target_locations.csv")

if __name__ == "__main__":
    discover()
