import requests
from bs4 import BeautifulSoup
import re
from tqdm import tqdm
import json

base_urls = [
    'https://www.lyrics.com/style/Country',
    'https://www.lyrics.com/genre/Blues',
    'https://www.lyrics.com/genre/Pop',
    'https://www.lyrics.com/genre/Rock',
    'https://www.lyrics.com/genre/Hip+Hop',
    'https://www.lyrics.com/style/Heavy+Metal'
]

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

found_urls = {}

# Gather all links for each song per genre url
for url in tqdm(base_urls, desc="Base URLs Progress"):
    genre = url.split('/')[-1]
    found_urls[genre] = []
    # used to increment page number on genre website
    i = 1
    while True:
        # add page number to url
        new_url = url + f'/{i}'
        response = requests.get(new_url, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')
        text = soup.prettify()
        # pattern to find the link to lyrics
        pattern = r'<p class="lyric-meta-title">\s+<a href="(.*)">\s+(.*)\s'

        matches = re.findall(pattern, text)
        if len(matches) == 0:
            break
        
        # tuple of url and title
        for match in matches:
            found_urls[genre].append(('https://www.lyrics.com'+match[0], match[1]))

        i += 1

# write results to json
with open("found_urls.json", "w") as json_file:
    json.dump(found_urls, json_file, indent=4)
