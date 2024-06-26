import nltk
import pandas as pd
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import random
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from tabulate import tabulate
import gensim.downloader as api
import string

class FFNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, p=0.5):
        super(FFNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=p)
        self.fc2 = nn.Linear(hidden_size, hidden_size//2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=p)
        self.fc3 = nn.Linear(hidden_size//2, num_classes)

        torch.manual_seed(42)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        out = self.fc3(out)
        return out
    
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)

label_encoder = LabelEncoder()

hidden_size = 128
num_epochs = 100
lr = 0.01

criterion = nn.CrossEntropyLoss()

nltk.download('punkt')
nltk.download('stopwords')
word2vec_model = api.load('word2vec-google-news-300')

def get_train_data(file_path):
    df = pd.read_csv(file_path, sep='\t')
    genres = df['genre'].tolist()
    lyrics = df['lyric'].tolist()

    lyrics = [generate_lyric_embeddings(lyric) for lyric in lyrics]

    return lyrics, genres

def get_test_data(file_path):
    df = pd.read_csv(file_path, sep='\t')
    
    lyrics = df['lyric'].tolist()
    lyrics = [generate_lyric_embeddings(lyric) for lyric in lyrics]

    genre = df['genre'].tolist()
    label_encoder = LabelEncoder()
    y_test_encoded = label_encoder.fit_transform(genre)

    return lyrics, y_test_encoded


# Remove stop words, punctuation, and tokenize
def preprocess_lyrics(lyrics):
    lyrics = lyrics.lower()
    lyrics = lyrics.translate(str.maketrans('', '', string.punctuation))

    words = word_tokenize(lyrics)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    preprocessed_lyrics = ' '.join(words)

    return preprocessed_lyrics
    

def generate_lyric_embeddings(lyrics):
    preprocessed_lyrics = preprocess_lyrics(lyrics)
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
def train_model(train_loader, loader, model):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
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

        accuracy, epoch_val_loss, _, _ = evaluate_model(loader, model, criterion)
        val_losses.append(epoch_val_loss)
        accuracy_per_epoch.append(accuracy)
    
    # Train loss validation loss per epoch diagram
    plot_losses(train_losses, val_losses)


# Function to see how model preforms with predictions
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

def main():
    train_file_path = 'train.tsv'
    val_file_path = 'validation.tsv'
    test_file_path = 'test.tsv'

    X_train, y_train = get_train_data(train_file_path)
    X_val, y_val = get_train_data(val_file_path)

    # Model parameters
    input_size = len(X_train[0])
    num_classes = len(np.unique(y_train)) 
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.fit_transform(y_val)

    X_test, y_test = get_test_data(test_file_path)

    # Loaders
    train_dataset = TensorDataset(torch.FloatTensor(np.array(X_train)), torch.LongTensor(y_train_encoded))
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    val_dataset = TensorDataset(torch.FloatTensor(np.array(X_val)), torch.LongTensor(y_val_encoded))
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    test_dataset = TensorDataset(torch.FloatTensor(np.array(X_test)), torch.LongTensor(y_test))
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = FFNN(input_size, hidden_size, num_classes)

    train_model(train_loader, val_loader, model)

    _, _, y_pred, y_true = evaluate_model(test_loader, model, criterion)
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