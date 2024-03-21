import os
import re

def get_test_data(file_path):
    genre_to_songs = {}
    for genre in os.listdir(file_path):
        genre_dir_path = os.path.join(file_path, genre)
        genre = genre.lower()
        files = os.listdir(genre_dir_path)
        song_paths = [os.path.join(genre_dir_path, file) for file in files]
        genre_to_songs[genre] = []
        for song in song_paths:
            with open(song, 'r') as f:
                text = f.read()
            text = re.sub(r'\s+', ' ', text).replace('[^\w\s]', '').lower().strip()
            genre_to_songs[genre].append(text)

    lyrics = []
    genres = []

    for genre, songs in genre_to_songs.items():
        lyrics.extend(songs)
        genres.extend([genre] * len(songs))
    
    return lyrics, genres

lyrics, genres = get_test_data('Test Songs')

with open('test.tsv', 'w') as tsv_file:
    tsv_file.write('genre\tlyric\n')
    for lyric, genre in zip(lyrics, genres):
        tsv_file.write(f'{genre}\t{lyric}\n')