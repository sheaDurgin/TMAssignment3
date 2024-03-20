import csv
import nltk
import certifi
import pandas as pd
import ssl
import torch
import os
import re
import matplotlib.pyplot as plt
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
from sklearn.metrics import f1_score, classification_report
from tabulate import tabulate
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


def get_test_data(dir_path):
    genre_to_songs = {}
    for genre in os.listdir(dir_path):
        genre_dir_path = os.path.join(dir_path, genre)
        if os.path.isdir(genre_dir_path):
            genre = genre.lower()

            files = os.listdir(genre_dir_path) 
            song_paths = [os.path.join(genre_dir_path, file) for file in files]
            genre_to_songs[genre] = []
            for song in song_paths:
                with open(song, 'r') as f:
                    text = f.read()
                preprocessed_lyrics = preprocess_lyrics(text)
                lyric_embeddings = generate_lyric_embeddings(preprocessed_lyrics)
                genre_to_songs[genre].append(lyric_embeddings)
    
    X_test = []
    y_test = []
    for genre, songs in genre_to_songs.items():
        X_test.extend(songs)
        y_test.extend([genre] * len(songs))

    # Encode labels
    label_encoder = LabelEncoder()
    y_test_encoded = label_encoder.fit_transform(y_test)
    
    return X_test, y_test_encoded


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


# Takes in val_loader or trainloader (val_loader for when val loss per epoch 
# needs to be calculated)
def train_model(train_loader, loader, model, criterion, optimizer, scheduler, num_epochs):
    train_losses = []
    val_losses = []
    accuracy_per_epoch = []

    for epoch in range(num_epochs):
        model.train()  
        running_train_loss = 0.0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item() * inputs.size(0)

        scheduler.step()  


        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        accuracy, epoch_val_loss, y_pred, y_true = evaluate_model(loader, model, criterion)
        val_losses.append(epoch_val_loss)
        accuracy_per_epoch.append(accuracy)

    return model, train_losses, val_losses, accuracy_per_epoch, y_pred, y_true


# For both testing on validation and testing data 
def train_and_test(train_loader, test_loader, input_size, num_classes, hidden_size, num_epochs, learning_rate):
    seed_value = 42
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    
    model = FFNN(input_size, hidden_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    trained_model, train_losses, val_losses, accuracy_per_epoch, y_pred, y_true = train_model(train_loader, test_loader, model, criterion, optimizer, scheduler, num_epochs)
    
    # Don't need other return values here
    accuracy, _, _, _= evaluate_model(test_loader, trained_model, criterion)
    
    return accuracy, train_losses, val_losses, accuracy_per_epoch, y_pred, y_true


# Function to evaluate the model on validation data
def evaluate_model(data_loader, model, criterion):
    model.eval()
    running_val_loss = 0.0
    correct = 0
    total = 0
    # For f1 scores
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item() * inputs.size(0)
            
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += len(labels)  
            y_true.extend(labels.numpy())
            y_pred.extend(predicted.numpy())

        accuracy = correct / total
        epoch_val_loss = running_val_loss / len(data_loader.dataset)

    
    return accuracy, epoch_val_loss, y_pred, y_true
    

# Permute over hyperparameter setups to determine best combination 
def tune_hyperparameters(train_loader, val_loader, input_size, num_classes, param_grid):
    seed_value = 42
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    
    best_accuracy = 0
    best_params = {}

    for params in ParameterGrid(param_grid):
        learning_rate = params['learning_rate']
        hidden_size = params['hidden_size']
        num_epochs = params['num_epochs']

        accuracy = train_and_test(train_loader, val_loader, input_size, num_classes, hidden_size, num_epochs, learning_rate)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = params
    
    return best_accuracy, best_params


def main():

    tsv_file_path = "genre_to_lyrics.tsv"
    unsplit_data = read_tsv_file(tsv_file_path)
    X_train, X_val, y_train, y_val = split_data(unsplit_data)

    # Model parameters
    input_size = len(X_train[0])
    num_classes = len(np.unique(y_train)) 
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.fit_transform(y_val)
    X_test, y_test = get_test_data("Test Songs")

    # Loaders
    train_dataset = TensorDataset(torch.FloatTensor(np.array(X_train)), torch.LongTensor(y_train_encoded))
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    val_dataset = TensorDataset(torch.FloatTensor(np.array(X_val)), torch.LongTensor(y_val_encoded))
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    test_dataset = TensorDataset(torch.FloatTensor(np.array(X_test)), torch.LongTensor(y_test))
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Validation Data - no need for predicted/true lables here
    #val_accuracy, train_losses, val_losses, accuracy_per_epoch, _, _ = train_and_test(train_loader, val_loader, input_size, num_classes, 128, 20, 0.01) 
    
    # Test Data - 3 parameters not relevent here for using testing data
    test_accuracy, _, _, _, y_pred, y_true = train_and_test(train_loader, test_loader, input_size, num_classes, 128, 20, 0.01)

    # Convert numerical notation back to string notation  
    y_pred_genre_names = label_encoder.inverse_transform(y_pred)
    y_true_genre_names = label_encoder.inverse_transform(y_true)

    report = classification_report(y_true_genre_names, y_pred_genre_names, output_dict=True, zero_division=0)

    genre_f1_scores = {}
    genres = ["blues", "country", "metal", "pop", "rap", "rock"]
    for genre in genres:
        genre_f1_scores[genre] = report[genre]['f1-score']

    overall_f1_score = report['weighted avg']['f1-score']

    table_data = [(genre, f1_score) for genre, f1_score in genre_f1_scores.items()]
    table = tabulate(table_data, headers=['Genre', 'F1 Score'], tablefmt='grid')
    print(f'\nOverall f1 score: {overall_f1_score}\n')
    print(f'F1 Scores Per Genre')
    print(table)

    # Train loss validation loss per epoch diagram
    #plot_losses(train_losses, val_losses)

    # Hyperparamters to test
    #param_grid = {
    # 'learning_rate': [0.001, 0.01, 0.1],
    # 'hidden_size': [64, 128, 256],
    # 'num_epochs': [10, 20, 30]  
    #}

    # Hyperparameter tuning
    #best_accuracy, best_params = tune_hyperparameters(train_loader, val_loader, input_size, num_classes, param_grid)
    #print("Best validation accuracy:", best_accuracy)
    #print("Best params:", best_params)

    # {'hidden_size': 128, 'learning_rate': 0.01, 'num_epochs': 20}
    # Slow so maybe just put in manually later 
    
    # accuracy = train_and_validate(train_loader, val_loader, input_size, num_classes, 128, 20, 0.01)

def plot_losses(train_losses, val_losses):
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss vs. Epoch')
    plt.legend()
    plt.show()

# :-) 
if __name__ == '__main__':
    main()