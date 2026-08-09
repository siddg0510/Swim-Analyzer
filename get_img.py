import urllib.request
import re

req = urllib.request.Request('https://similarpng.com/swimming-logo-on-transparent-background-png/', headers={'User-Agent': 'Mozilla/5.0'})
html = urllib.request.urlopen(req).read().decode('utf-8')

urls = re.findall(r'https://image.similarpng.com/file/similarpng/[^\s"]+\.png', html)
for u in set(urls):
    if 'wim' in u.lower():
        print(u)
        req2 = urllib.request.Request(u.replace('\\/', '/'), headers={'User-Agent': 'Mozilla/5.0'})
        img_data = urllib.request.urlopen(req2).read()
        with open('c:/Users/SiddG/swim_analyzer/assets/icon.png', 'wb') as f:
            f.write(img_data)
        print("Saved!")
        break
