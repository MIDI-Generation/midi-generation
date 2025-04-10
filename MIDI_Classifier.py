import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import sys
from torch.utils.data import Dataset, DataLoader


class MidiGRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, bidirectional=False):
        super(MidiGRUClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, bidirectional=bidirectional)
        
        direction_factor = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_size * direction_factor, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1),
                         x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.gru(x, h0)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


class PreprocessedMidiDataset(Dataset):
    def __init__(self, features_path, labels_path=None):

        if features_path.endswith('.npy'):
            self.sequences = np.load(features_path)
        elif features_path.endswith('.csv'):
            df = pd.read_csv(features_path, header=None)
            flat_array = df.values
            num_samples = flat_array.shape[0]
            # Assume seq_len and input_size:
            seq_len = 1024
            input_size = 3
            self.sequences = flat_array.reshape((num_samples, seq_len, input_size))
        else:
            raise ValueError("Unsupported format. Use .npy or .csv.")

        self.sequences = torch.tensor(self.sequences, dtype=torch.float)

        if labels_path:
            self.labels = torch.tensor(np.loadtxt(labels_path, dtype=int))
        else:
            self.labels = None

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.sequences[idx], self.labels[idx]
        else:
            return self.sequences[idx]


def train_model(model, train_loader, num_epochs=10, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if(torch.cuda.is_available()):
        print("GPU is available")
        device = torch.device('cuda')
    else:
        print("GPU not found...\nEnding runtime...")
        sys.exit(1)

    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)

            outputs = model(sequences)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        mean_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {mean_loss:.4f}")

    torch.save(model.state_dict(), 'midi_gru_classifier.pth')
    print("Training complete. Model saved as 'midi_gru_classifier.pth'")


def predict_artist(model, tensor_sequence, label_map):
    model.eval()
    device = next(model.parameters()).device

    input_tensor = tensor_sequence.unsqueeze(0).to(device)  # [1, seq_len, input_size]
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_class = torch.max(output, dim=1)

    return label_map[predicted_class.item()]
