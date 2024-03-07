import csv
import os
import string

genres = []
titles = []
urls = []
lyrics = []

with open('unfiltered_genre_to_lyrics.tsv', 'r', encoding='utf-8') as tsvfile:
    tsvreader = csv.reader(tsvfile, delimiter='\t')
    next(tsvreader, None)
    for row in tsvreader:
        genre, title, url, lyric = row

        genres.append(genre)
        titles.append(title)
        urls.append(url)
        lyrics.append(lyric)

titles_to_remove = set()
for genre_dir_path in os.listdir('Test Songs'):
    for song_filename in os.listdir('Test Songs/'+genre_dir_path):
        song = song_filename[:-4]
        song = ''.join([c.lower() for c in song if c not in string.punctuation])
        titles_to_remove.add(song)

with open('genre_to_lyrics.tsv', 'w', encoding='utf-8') as tsv_file:
    tsv_file.write('Genre\tTitle\tURL\tLyrics\n')
    for genre, title, url, lyric in zip(genres, titles, urls, lyrics):
        processed_title = ''.join([c.lower() for c in title if c not in string.punctuation])
        if processed_title in titles_to_remove:
            continue
        tsv_file.write(f"{genre}\t{title}\t{url}\t{lyric}\n")