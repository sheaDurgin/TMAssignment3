import json
import requests
from bs4 import BeautifulSoup
import re
from tqdm import tqdm
import random

random.seed(42)

def get_lyrics(soup):
    lyrics = soup.find('pre', id='lyric-body-text')
    if not lyrics:
        return None
    lyrics_text = lyrics.text.strip()
    return re.sub(r'\s+', ' ', lyrics_text)

fix_genres = {
    'Hip+Hop': 'Rap',
    'Heavy+Metal': 'Metal'
}

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# load all urls and titles
with open('found_urls.json', 'r') as file:
    data = json.load(file)

# write to tsv
with open('unfiltered_genre_to_lyrics.tsv', 'w', encoding='utf-8') as tsv_file:
    # row -> Genre URL Lyrics
    tsv_file.write('Genre\tTitle\tURL\tLyrics\n')
    
    # get smallest genre, and use this to truncate others
    min_len = min(len(url_and_titles) for url_and_titles in data.values())

    # loop through all urls for each genre
    for genre, url_and_titles in tqdm(data.items(), desc='Genres'):
        random.shuffle(url_and_titles)
        for url_and_title in tqdm(url_and_titles[:min_len], desc='URLs', total=min_len):
        # for url_and_title in tqdm(url_and_titles, desc='URLs', total=len(url_and_titles)):
            url, title = url_and_title
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.content, 'html.parser')

            # find lyrics using BeautifulSoup
            lyrics = get_lyrics(soup)
            if not lyrics:
                continue
            
            # reformat specific genre names
            if genre in fix_genres:
                genre = fix_genres[genre]

            # write to tsv
            tsv_file.write(f"{genre.lower()}\t{title}\t{url}\t{lyrics}\n")