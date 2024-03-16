import csv
import nltk
import certifi
import ssl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from FFNN_task3 import FFNN
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

# 90 training 10 validation  
def split_data(data):
    
    X = [entry['Lyrics'] for entry in data]
    y = [entry['Genre'] for entry in data]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

    return X_train, X_val, y_train, y_val 

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
        avg_word_embedding = np.zeros_like(word2vec_model['word']) 
    
    return avg_word_embedding


def main():
    tsv_file_path = "unfiltered_genre_to_lyrics.tsv"

    unsplit_data = read_tsv_file(tsv_file_path)

    X_train, X_val, y_train, y_val = split_data(unsplit_data)
    
    #print(X_train)
    print("Training set size:", len(X_train))
    print("Validation set size:", len(X_val))
    print("Total size:", len(X_train) + len(X_val))

    # Convert string label to numerical representation
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.fit_transform(y_val)

    # Data to tensor conversion
    X_train_tensor = torch.FloatTensor(np.array(X_train))
    X_val_tensor = torch.FloatTensor(np.array(X_val))
    y_train_tensor = torch.LongTensor(y_train_encoded)
    y_val_tensor = torch.LongTensor(y_val_encoded)
    
    # DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=64)

    # model, loss function, and optimizer
    input_size = len(X_train[0]) 
    hidden_size = 128 
    num_classes = len(np.unique(y_train)) 
    model = FFNN(input_size, hidden_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    
    # Loss
    print(f'Epoch [{epoch+1}/{num_epochs}], Validation Accuracy: {accuracy:.4f}')

    # Save model
    torch.save(model.state_dict(), 'lyric_classification_model.pth')


# :-) 
if __name__ == '__main__':
    main()
