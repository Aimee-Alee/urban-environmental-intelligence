import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAQ_API_KEY")

headers = {"X-API-Key": api_key}
# We'll try sensor 3123 (McMillan Reservoir PM2.5 based on my research)
# If that doesn't work, we'll try sensor 2 (Generic PM2.5)
sensor_id = 3123 
url = f"https://api.openaq.org/v3/sensors/{sensor_id}/hours?limit=1&datetime_from=2025-01-01T00:00:00Z"

print(f"Testing URL: {url}")
try:
    response = requests.get(url, headers=headers)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print("Success! First result sample:")
        if data['results']:
            print(json.dumps(data['results'][0], indent=2))
        else:
            print("No data found for 2025 yet. Let's try to fetch recent data instead.")
            url_recent = f"https://api.openaq.org/v3/sensors/{sensor_id}/hours?limit=1"
            response_recent = requests.get(url_recent, headers=headers)
            print(f"Recent data status: {response_recent.status_code}")
            if response_recent.status_code == 200 and response_recent.json()['results']:
                print(json.dumps(response_recent.json()['results'][0], indent=2))
    else:
        print(f"Error: {response.text}")
except Exception as e:
    print(f"Request failed: {e}")
