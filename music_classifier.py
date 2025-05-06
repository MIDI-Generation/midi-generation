import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import json
import sys

with open('composer_label_map.json', 'r', encoding='utf-8') as f:
    label_map = json.load(f)


class TokenDataset(Dataset):
    def __init__(self, sequences, composers):
        self.data = [torch.tensor(seq, dtype=torch.long) for seq in sequences]
        self.labels = [torch.tensor(label_map[composer], dtype=torch.long) for composer in composers]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def collate_fn(batch):
    sequences, labels = zip(*batch)
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    return padded_sequences, labels


# Tried 256 hidden layers initiallty, but overfitted... :( (Also increased dropout from 0.3 to 0.5)
class MidiGRUClassifier(nn.Module):
    def __init__(self, vocab_size=131, embed_dim=128, hidden_size=128, num_layers=2, num_classes=len(label_map), bidirectional=True, dropout=0.5):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1  

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_size, num_layers,
                          batch_first=True,
                          dropout=dropout if num_layers > 1 else 0,
                          bidirectional=bidirectional)
        self.norm = nn.LayerNorm(hidden_size * self.num_directions)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * self.num_directions, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        x = self.embedding(x)
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.norm(out[:, -1, :])
        return self.fc(out)


def load_data_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    df = df[df['composer'].isin(label_map)]
    df['sequence'] = df['sequence'].apply(lambda s: list(map(int, s.strip().split())))
    return df


def train_model(model, train_loader, val_loader=None, num_epochs=10, lr=0.001):
    print("Beginning Training...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_loss = float('inf')
    best_epoch = -1

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for sequences, labels in pbar:
            sequences, labels = sequences.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix(loss=loss.item())

        epoch_loss = total_loss / len(train_loader)
        epoch_acc = (correct / total) * 100
        print(f"Epoch {epoch+1}: Loss = {epoch_loss:.4f}, Accuracy = {epoch_acc:.2f}%")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_epoch = epoch + 1
            torch.save(model, "midi_gru_classifier_v2.pth")
            print(f"Best model saved at Epoch {epoch+1} (Loss = {epoch_loss:.4f})")

    print(f"\nTraining complete. Best model was from Epoch {best_epoch} with Loss = {best_loss:.4f}")


def evaluate_model(model, loader):
    model.eval()
    preds, targets = [], []
    correct_top1 = 0
    correct_top3 = 0
    total = 0
    device = next(model.parameters()).device

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            
           
            top3_preds = torch.topk(outputs, k=3, dim=1).indices  # shape: (batch_size, 3)
            
            # Top-1 accuracy
            top1_preds = top3_preds[:, 0]
            correct_top1 += (top1_preds == y).sum().item()
            
            # Top-3 accuracy
            for idx in range(y.size(0)):
                if y[idx] in top3_preds[idx]:
                    correct_top3 += 1
            
            total += y.size(0)

    acc_top1 = (correct_top1 / total) * 100
    acc_top3 = (correct_top3 / total) * 100

    print(f"Top-1 Validation Accuracy: {acc_top1:.2f}%")
    print(f"Top-3 Validation Accuracy: {acc_top3:.2f}%")

