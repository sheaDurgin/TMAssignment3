import csv
import nltk
import certifi
import ssl
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
import gensim.downloader as api
import string

# whatever this does
ssl._create_default_https_context = ssl._create_unverified_context


nltk.download('punkt')
nltk.download('stopwords')
word2vec_model = api.load('word2vec-google-news-300')


def read_tsv_file(file_path): 
    genre_to_lyrics = []
    with open(file_path, 'r', newline='', encoding='utf-8') as tsv:
        reader = csv.reader(tsv, delimiter='\t')
        next(reader)
        for row in reader:
            Genre, _, _, Lyrics = row
            
            # Tokenize lyrics and gen embeddings before loading array
            preprocessed_lyrics = preprocess_lyrics(Lyrics)
            
            lyric_embeddings = generate_lyric_embeddings(preprocessed_lyrics)

            genre_to_lyrics.append({'Genre': Genre, 'Lyrics': lyric_embeddings})
        
        
        return genre_to_lyrics

# 80 testing 20 training  
def split_data(data):
    
    X = [entry['Lyrics'] for entry in data]
    y = [entry['Genre'] for entry in data]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

# Remove stop words, punctuation, and tokenize
def preprocess_lyrics(lyrics):
    
    lyrics = lyrics.lower()
    lyrics = lyrics.translate(str.maketrans('', '', string.punctuation))

    words = word_tokenize(lyrics)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    preprocessed_lyrics = ' '.join(words)

    return preprocessed_lyrics
    

def generate_lyric_embeddings(preprocessed_lyrics):

    word_tokens = preprocessed_lyrics.split()

    word_embeddings = []

    for word in word_tokens:
        if word in word2vec_model:
            word_embeddings.append(word2vec_model[word])
    
    # Average embeddings for reduced dimensionality
    if word_embeddings:
        avg_word_embedding = sum(word_embeddings) / len(word_embeddings)
    else:
        avg_word_embedding = None
    
    return avg_word_embedding


def main():
    tsv_file_path = "unfiltered_genre_to_lyrics.tsv"

    unsplit_data = read_tsv_file(tsv_file_path)

    X_train, X_test, y_train, y_test = split_data(unsplit_data)
    
    print(X_train)

# :-) 
if __name__ == '__main__':
    main()
