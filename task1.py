import json
import requests
from bs4 import BeautifulSoup
import re
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
import random
import os
import string

random.seed(42)

fix_genres = {
    'Hip+Hop': 'Rap',
    'Heavy+Metal': 'Metal',
    'Rock+__+Roll': 'Rock'
}

base_urls = [
    'https://www.lyrics.com/style/Rock+__+Roll',
    'https://www.lyrics.com/style/Country',
    'https://www.lyrics.com/genre/Blues',
    'https://www.lyrics.com/genre/Pop',
    'https://www.lyrics.com/genre/Hip+Hop',
    'https://www.lyrics.com/style/Heavy+Metal'
]

urls_last_i = [
    144,
    410,
    216,
    1886,
    595,
    117
]

min_len = min(urls_last_i)

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

test_dir = 'Test Songs'

def write_to_tsvs(df):
    # Separate data based on Genre
    genres = df['genre'].unique()

    all_train = pd.DataFrame()
    all_val = pd.DataFrame()

    for genre in genres:
        genre_data = df[df['genre'] == genre]
        
        # Shuffle the data for randomness
        genre_data = genre_data.sample(frac=1).reset_index(drop=True)
        
        # Split into train and validation sets
        train, val = train_test_split(genre_data, test_size=0.1, random_state=42)

        all_train = pd.concat([all_train, train], ignore_index=True)
        all_val = pd.concat([all_val, val], ignore_index=True)
        
    all_train.to_csv(f"train.tsv", sep='\t', index=False)
    all_val.to_csv(f"validation.tsv", sep='\t', index=False)

def get_lyrics(soup):
    lyrics = soup.find('pre', id='lyric-body-text')
    if not lyrics:
        return None
    lyrics_text = lyrics.text.strip()
    return re.sub(r'\s+', ' ', lyrics_text)

def get_titles_to_remove():
    titles_to_remove = {}
    for genre_dir_path in os.listdir(test_dir):
        titles_to_remove[genre_dir_path.lower()] = set()
        for song_filename in os.listdir(test_dir+ '/' + genre_dir_path):
            song = song_filename[:-4]
            song = ''.join([c.lower() for c in song if c not in string.punctuation])
            titles_to_remove[genre_dir_path.lower()].add(song)
    return titles_to_remove

def get_lyrics_links():
    found_urls = {}

    # Gather all links for each song per genre url
    for index, url in tqdm(enumerate(base_urls), desc="Base URLs Progress"):
        genre = url.split('/')[-1]
        found_urls[genre] = []
        genre_pages = [x+1 for x in range(urls_last_i[index])]
        cnt = 0
        # used to increment page number on genre website
        while cnt < min_len and genre_pages:
            i = random.choice(genre_pages)
            genre_pages.remove(i)
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

            cnt += 1
            
    return found_urls

def get_all_data(data):
    titles_to_remove = get_titles_to_remove()
    genre_lyrics = set()
    genres = []
    lyrics = []

    # loop through all urls for each genre
    for genre, url_and_titles in tqdm(data.items(), desc='Genres'):
        genre_lyrics = set()
        # reformat specific genre names
        if genre in fix_genres:
            genre = fix_genres[genre]
        genre = genre.lower()

        random.shuffle(url_and_titles)
        for url_and_title in tqdm(url_and_titles, desc='URLs', total=min_len*24):
            url, title = url_and_title

            processed_title = ''.join([c.lower() for c in title if c not in string.punctuation])
            if processed_title in titles_to_remove[genre]:
                continue

            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.content, 'html.parser')

            # find lyrics using BeautifulSoup
            lyric = get_lyrics(soup)
            if not lyric:
                continue

            if lyric not in genre_lyrics:
                # write to tsv
                genres.append(genre)
                lyrics.append(lyric)
                genre_lyrics.add(lyric)
    
    return pd.DataFrame({'genre': genres, 'lyric': lyrics})

if __name__ == '__main__':
    data = get_lyrics_links()
    df = get_all_data(data)
    write_to_tsvs(df)