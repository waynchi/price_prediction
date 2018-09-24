#!/usr/bin/python3

import requests
from pprint import pprint

def get_price_history(url, payload):
    r = requests.get(url, params=payload)
    return r['data']['final']

def main():
    payload = {'appid': '49520', 'cc': 'us'}
    url = 'https://steamdb.info/api/GetPriceHistory'

    r = get_price_history(url, payload)

    print(r.url)
    pprint(r.json())

if __name__ == "__main__":
    main()
