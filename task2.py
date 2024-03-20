import torch
import torch.nn as nn
from torch import optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, DataLoader

from tqdm import tqdm
import os
import re

from sklearn.metrics import f1_score, classification_report
from tabulate import tabulate


batch_size = 128
num_epochs = 100
learning_rate = 0.001

def extract_features(lyrics):
    num_words = len(lyrics.split())
    unique_words = len(set(lyrics.split()))
    return [num_words, unique_words]

def get_train_data(file_path):
    df = pd.read_csv(file_path, sep='\t')
    df['Lyrics'] = df['Lyrics'].str.lower().str.replace('[^\w\s]', '')

    genres = df['Genre'].unique()
    songs_per_genre_dict = {}
    for genre in genres:
        songs_per_genre_dict[genre] = df[df['Genre'] == genre]['Lyrics']

    X_train = []
    y_train = []
    X_val = []
    y_val = []

    for genre, songs in songs_per_genre_dict.items():

        genre_features = [extract_features(lyric) for lyric in songs]

        num_songs = len(songs)
        num_train = int(num_songs * 0.9)
        
        # Assign songs to train and validation sets
        X_train.extend(genre_features[:num_train])
        y_train.extend([genre] * len(genre_features[:num_train]))
        X_val.extend(genre_features[num_train:])
        y_val.extend([genre] * len(genre_features[num_train:]))

    return X_train, y_train, X_val, y_val

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

    X_test = []
    y_test = []

    for genre, songs in genre_to_songs.items():
        X_test.extend([extract_features(lyric) for lyric in songs])
        y_test.extend([genre] * len(songs))
    
    return X_test, y_test


def create_dataloader(X, y, label_encoder, shuffle=True):

    # Convert lists to tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(label_encoder.fit_transform(y))
    dataset = TensorDataset(X, y)

    dataloader = DataLoader(dataset, batch_size=batch_size)

    return dataloader

torch.manual_seed(42)
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, output)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out
    
def train_model(model, train_dataloader, validation_dataloader):

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_func = torch.nn.CrossEntropyLoss()
    loss_epoch = {}
    loss_values = []
    accuracy_epoch = {}
    loss_epoch_valid = {}
    accuracy_epoch_valid = {}

    for epoch in tqdm(range(num_epochs)):
        model.train()
        correct_train = 0
        total_train = 0

        # Training phase
        for X, y in train_dataloader:
            optimizer.zero_grad()
            pred = model(X)
            loss = loss_func(pred, y)
            loss_values.append(loss.item())
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(pred.data, 1)
            total_train += y.size(0)
            correct_train += (predicted == y).sum().item()
            
        accuracy_epoch[epoch] = correct_train / total_train
        loss_epoch[epoch] = sum(loss_values) / len(loss_values)
        
        # Validation phase
        running_vloss = 0
        correct_valid = 0
        total_valid = 0
        model.eval()
        with torch.no_grad():
            for vdata in validation_dataloader:
                vinputs, vlabels = vdata
                voutputs = model(vinputs)
                vloss = loss_func(voutputs, vlabels)
                running_vloss += vloss.item()
                _, predicted = torch.max(voutputs.data, 1)
                total_valid += vlabels.size(0)
                correct_valid += (predicted == vlabels).sum().item()
            loss_epoch_valid[epoch] = running_vloss / len(validation_dataloader)
            accuracy_epoch_valid[epoch] = correct_valid / total_valid

    plt.plot(loss_epoch.keys(), loss_epoch.values(), 'r--', label='Training Loss')
    plt.plot(loss_epoch_valid.keys(), loss_epoch_valid.values(), 'b--', label='Validation Loss')
    plt.plot(accuracy_epoch.keys(), accuracy_epoch.values(), 'g--', label='Training Accuracy')
    plt.plot(accuracy_epoch_valid.keys(), accuracy_epoch_valid.values(), 'y--', label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Loss / Accuracy')
    #plt.savefig('Task2_epoch_diagram.pdf')
    plt.legend()
    plt.show()


def test_model(model, test_dataloader):
    model.eval()
    y_true = []
    y_pred = []
    correct = 0
    total = 0
    with torch.no_grad():
        for tdata in test_dataloader:
            tinputs, tlabels = tdata
            toutputs = model(tinputs)
            _, predicted = torch.max(toutputs.data, 1)
            y_true.extend(tlabels.numpy())
            y_pred.extend(predicted.numpy())
            total += tlabels.size(0)
            correct += (predicted == tlabels).sum().item()

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')
    return y_true, y_pred

def calculate_f1(y_true, y_pred, label_encoder):

    # Transform into string
    decoded_true_labels = label_encoder.inverse_transform(y_true)
    decoded_pred_labels = label_encoder.inverse_transform(y_pred)
    
    f1 = f1_score(decoded_true_labels, decoded_pred_labels, average='macro', zero_division=0)
    
    # Generate report for f1 score per genre
    report = classification_report(decoded_true_labels, decoded_pred_labels, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose().round(4)
    
    report_df.drop(columns=['precision', 'recall', 'support'], inplace=True)
    report_df = report_df.drop(['macro avg', 'weighted avg', 'accuracy'], axis=0, errors='ignore')
    
    table = tabulate(report_df, headers='keys', tablefmt='pretty')
    
    print(table)
    print(f"Overall F1 Score: {f1:.4f}")


def main():
    train_file_path = 'genre_to_lyrics.tsv'
    test_file_path = 'Test Songs/'
    X_train, y_train, X_val, y_val = get_train_data(train_file_path)
    X_test, y_test = get_test_data(test_file_path)

    label_encoder = LabelEncoder()
    train_dataloader = create_dataloader(X_train, y_train, label_encoder)
    validation_dataloader = create_dataloader(X_val, y_val, label_encoder)
    test_dataloader = create_dataloader(X_test, y_test, label_encoder)
    
    input_size = len(X_train[0])
    hidden_size = 32
    output = len(np.unique(y_train))
    
    model = NeuralNet(input_size, hidden_size, output)
    train_model(model, train_dataloader, validation_dataloader)
    y_true, y_pred = test_model(model, test_dataloader)

    calculate_f1(y_true, y_pred, label_encoder)


if __name__ == '__main__':
    main()