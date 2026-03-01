import os
import requests
from dotenv import load_dotenv

load_dotenv()
OPENAQ_API_KEY = os.getenv("OPENAQ_API_KEY")
BASE_URL = "https://api.openaq.org/v3"

def list_params():
    url = f"{BASE_URL}/parameters"
    headers = {"X-API-Key": OPENAQ_API_KEY}
    res = requests.get(url, headers=headers)
    if res.status_code == 200:
        for p in res.json()['results']: # Search for key air quality parameters
             print(f"ID: {p['id']}, Name: {p['name']}, DisplayName: {p.get('displayName')}")
    else:
        print(f"Error: {res.status_code}")

if __name__ == "__main__":
    list_params()
