import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import json
import ast
from tqdm import tqdm

with open("composer_label_map.json", "r", encoding="utf-8") as f:
    composer_label_map = json.load(f)

idx_to_composer = {int(v): k for k, v in composer_label_map.items()}

class TokenDataset(Dataset):
    def __init__(self, sequences, composers):
        self.data = [torch.tensor(seq, dtype=torch.long) for seq in sequences]
        self.labels = [torch.tensor(composer_label_map[composer], dtype=torch.long) for composer in composers]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def collate_fn(batch):
    sequences, labels = zip(*batch)
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    return padded_sequences, labels

class MidiGRUClassifier(nn.Module):
    def __init__(self, vocab_size=131, embed_dim=128, hidden_size=128, num_layers=2, num_classes=len(composer_label_map), bidirectional=True, dropout=0.5):
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

def evaluate_model(model, loader):
    model.eval()
    correct_top1 = 0
    correct_top3 = 0
    total = 0
    device = next(model.parameters()).device

    with torch.no_grad():
        pbar = tqdm(loader, desc="Evaluating", leave=False)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            outputs = model(x)

            top3_preds = torch.topk(outputs, k=3, dim=1).indices
            top1_preds = top3_preds[:, 0]
            correct_top1 += (top1_preds == y).sum().item()

            for idx in range(y.size(0)):
                if y[idx] in top3_preds[idx]:
                    correct_top3 += 1

            total += y.size(0)

    acc_top1 = (correct_top1 / total) * 100
    acc_top3 = (correct_top3 / total) * 100

    print(f"Top-1 Validation Accuracy: {acc_top1:.2f}%")
    print(f"Top-3 Validation Accuracy: {acc_top3:.2f}%")

if __name__ == "__main__":
    print("Loading dataset...")
    df = pd.read_csv("filtered_dataset.csv")
    df['sequence'] = df['sequence_vector'].apply(ast.literal_eval)

    val_df = df[df['split'] == 'validation']

    val_dataset = TokenDataset(val_df['sequence'].tolist(), val_df['composer'].tolist())
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    model = MidiGRUClassifier()
    if torch.cuda.is_available():
        model.load_state_dict(torch.load("midi_gru_classifier_v7.pth"))
    else:
        model.load_state_dict(torch.load("midi_gru_classifier_v7.pth", map_location=torch.device('cpu')))

    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    evaluate_model(model, val_loader)
