import csv


def read_tsv_file(file_path): 
    genre_to_lyrics = []
    with open(file_path, 'r', newline='', encoding='utf-8') as tsv:
        reader = csv.reader(tsv, delimiter='\t')
        next(reader)
        for row in reader:
            Genre, Title, URL, Lyrics = row
            genre_to_lyrics.append({'Genre': Genre, 'Title': Title, 'URL': URL, 'Lyrics': Lyrics})
        
        
        return genre_to_lyrics
    

def main():
    tsv_file_path = "unfiltered_genre_to_lyrics.tsv"

    res = read_tsv_file(tsv_file_path)
    for song in res: 
        print(song)

main()