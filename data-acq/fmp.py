from urllib.request import urlopen
import os
import certifi
import json

key = os.getenv("FMP_API_KEY")

def get_jsonparsed_data(url):
    response = urlopen(url)
    data = response.read().decode("utf-8")
    return data

url = (f"https://financialmodelingprep.com/stable/profile?symbol=AAPL&apikey={key}")
print(get_jsonparsed_data(url))
