import csv
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# whatever this does
nltk.download('punkt')
nltk.download('stopwords')


def read_tsv_file(file_path): 
    genre_to_lyrics = []
    with open(file_path, 'r', newline='', encoding='utf-8') as tsv:
        reader = csv.reader(tsv, delimiter='\t')
        next(reader)
        for row in reader:
            Genre, _, _, Lyrics = row
            
            # Tokenize lyrics before loading array
            preprocessed_lyrics = preprocess_lyrics(Lyrics)

            genre_to_lyrics.append({'Genre': Genre, 'Lyrics': Lyrics})
        
        
        return genre_to_lyrics
    
def preprocess_lyrics(lyrics):
    
    lyrics = lyrics.lower()
    lyrics = lyrics.translated(str.maketrans('', '', string.punctuation))

    words = word_tokenize(lyrics)

    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    preprocessed_lyrics = ' '.join(words)

    return preprocessed_lyrics
    
    
def main():
    tsv_file_path = "unfiltered_genre_to_lyrics.tsv"

    res = read_tsv_file(tsv_file_path)

main()