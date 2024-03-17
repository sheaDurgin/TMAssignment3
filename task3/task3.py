import csv
import nltk
import certifi
import ssl
import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid
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

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

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

def train_and_validate(train_loader, val_loader, input_size, num_classes, hidden_size, num_epochs, learning_rate):

    model = FFNN(input_size, hidden_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        scheduler.step()
    
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += len(labels)  

    accuracy = correct / total
    
    return accuracy

# Permute over hyperparameter setups to determine best combination 
def tune_hyperparameters(train_loader, val_loader, input_size, num_classes, param_grid):
    best_accuracy = 0
    best_params = {}

    for params in ParameterGrid(param_grid):
        learning_rate = params['learning_rate']
        hidden_size = params['hidden_size']
        num_epochs = params['num_epochs']

        model = FFNN(input_size, hidden_size, num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

        for epoch in range(num_epochs):
            model.train()
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            scheduler.step()
        
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += len(labels)  

        accuracy = correct / total
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = params
    
    return best_accuracy, best_params


def main():
    # Seed model 
    seed_value = 42
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    tsv_file_path = "unfiltered_genre_to_lyrics.tsv"
    unsplit_data = read_tsv_file(tsv_file_path)
    X_train, X_val, y_train, y_val = split_data(unsplit_data)

    # Model parameters
    input_size = len(X_train[0])
    hidden_size = 128
    num_classes = len(np.unique(y_train)) 
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.fit_transform(y_val)


    train_dataset = TensorDataset(torch.FloatTensor(np.array(X_train)), torch.LongTensor(y_train_encoded))
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    val_dataset = TensorDataset(torch.FloatTensor(np.array(X_val)), torch.LongTensor(y_val_encoded))
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Hyperparamters to test
    param_grid = {
     'learning_rate': [0.001, 0.01, 0.1],
     'hidden_size': [64, 128, 256],
     'num_epochs': [10, 20, 30]  
    }

    # Hyperparameter tuning
    best_accuracy, best_params = tune_hyperparameters(train_loader, val_loader, input_size, num_classes, param_grid)
    print("Best validation accuracy:", best_accuracy)


    # Slow so maybe just put in manually later 
    accuracy = train_and_validate(train_loader, val_loader, input_size, num_classes, best_params['hidden_size'], best_params['num_epochs'], best_params['learning_rate'])

    
    print(accuracy)

# :-) 
if __name__ == '__main__':
    main()
