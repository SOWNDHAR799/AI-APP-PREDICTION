import requests, re
r = requests.get('https://www.google.com/finance/quote/TATAGOLD:NSE')
m = re.search(r'data-last-price="([0-9.]+)"', r.text)
print(m.group(1) if m else 'NOT FOUND')
