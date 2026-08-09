import urllib.request
import re

req = urllib.request.Request('https://similarpng.com/swimming-logo-on-transparent-background-png/', headers={'User-Agent': 'Mozilla/5.0'})
html = urllib.request.urlopen(req).read().decode('utf-8')
urls = re.findall(r'"previewUrl":"([^"]+)"', html)
for u in urls:
    print(u)
