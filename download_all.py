import urllib.request
import re
import os

req = urllib.request.Request('https://similarpng.com/swimming-logo-on-transparent-background-png/', headers={'User-Agent': 'Mozilla/5.0'})
html = urllib.request.urlopen(req).read().decode('utf-8')

urls = re.findall(r'<img[^>]+src="([^"]+)"', html)
os.makedirs('c:/Users/SiddG/swim_analyzer/assets/temp', exist_ok=True)
count = 0
for u in urls:
    if 'similarpng' in u and (u.endswith('.png') or u.endswith('.jpg')):
        print(u)
        try:
            req2 = urllib.request.Request(u, headers={'User-Agent': 'Mozilla/5.0'})
            img_data = urllib.request.urlopen(req2).read()
            with open(f'c:/Users/SiddG/swim_analyzer/assets/temp/img_{count}.png', 'wb') as f:
                f.write(img_data)
            count += 1
        except Exception as e:
            print("Failed:", u, e)
