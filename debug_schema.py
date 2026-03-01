import os
import requests
from dotenv import load_dotenv
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
load_dotenv()
OPENAQ_API_KEY = os.getenv("OPENAQ_API_KEY")

headers = {"X-API-Key": OPENAQ_API_KEY}
url = "https://api.openaq.org/v3/sensors/261/hours?limit=1"
res = requests.get(url, headers=headers, verify=False)
if res.status_code == 200:
    data = res.json()
    results = data.get('results', [])
    if results:
        print(f"Schema keys for results[0]: {results[0].keys()}")
        print(f"Full object: {results[0]}")
    else:
        print("No results found.")
else:
    print(f"Error {res.status_code}: {res.text}")
