import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import sys
import ast
import os



csv_file = "filtered_dataset.csv"
print("Loading CSV...")
df = pd.read_csv(csv_file)
df['sequence'] = df['sequence_vector'].apply(lambda s: ast.literal_eval(s))  

composers = sorted(df['composer'].unique())  
composer_label_map = {composer: idx for idx, composer in enumerate(composers)}
idx_to_composer = {idx: composer for composer, idx in composer_label_map.items()}

with open("composer_label_map.json", "w", encoding="utf-8") as f:
    json.dump(composer_label_map, f, ensure_ascii=False, indent=2)

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
    def __init__(self, vocab_size=131, embed_dim=128, hidden_size=128, num_layers=2,
                 num_classes=len(composer_label_map), bidirectional=True, dropout=0.5):
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

        self.attn = nn.Linear(hidden_size * self.num_directions, 1)
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

        gru_out, _ = self.gru(x, h0)

        attn_weights = torch.softmax(self.attn(gru_out).squeeze(-1), dim=1)
        context = torch.sum(gru_out * attn_weights.unsqueeze(-1), dim=1)

        context = self.norm(context)
        return self.fc(context)

def train_model(model, train_loader, val_loader=None, num_epochs=10, lr=0.001):
    print("Beginning Training...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    val_f1s = []

    best_loss = float('inf')
    best_epoch = -1

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds, all_labels = [], []

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
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix(loss=loss.item())

        epoch_loss = total_loss / len(train_loader)
        epoch_acc = (correct / total) * 100
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)

        print(f"[Train] Epoch {epoch+1}: Loss = {epoch_loss:.4f}, Accuracy = {epoch_acc:.2f}%")
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        val_preds, val_targets = [], []

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                loss = criterion(outputs, y)

                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == y).sum().item()
                total += y.size(0)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(y.cpu().numpy())

        val_epoch_loss = val_loss / len(val_loader)
        val_epoch_acc = (correct / total) * 100
        val_epoch_f1 = f1_score(val_targets, val_preds, average='weighted')

        val_losses.append(val_epoch_loss)
        val_accs.append(val_epoch_acc)
        val_f1s.append(val_epoch_f1)

        print(f"[Val]   Epoch {epoch+1}: Loss = {val_epoch_loss:.4f}, Accuracy = {val_epoch_acc:.2f}%, F1 Score = {val_epoch_f1:.4f}")

        if val_epoch_loss < best_loss:
            best_loss = val_epoch_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), "midi_gru_classifier_v7.pth")
            print(f"Best model saved at Epoch {epoch+1} (Val Loss = {val_epoch_loss:.4f})")

    print(f"Training complete. Best model was from Epoch {best_epoch} with Val Loss = {best_loss:.4f}")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.title('Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(val_f1s, label='Val F1 Score', color='purple')
    plt.title('Validation F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_metrics.png")
    plt.show()

def evaluate_model(model, loader):
    model.eval()
    correct_top1 = 0
    correct_top3 = 0
    total = 0
    device = next(model.parameters()).device
    val_preds, val_targets = [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)

            top3_preds = torch.topk(outputs, k=3, dim=1).indices
            top1_preds = top3_preds[:, 0]
            correct_top1 += (top1_preds == y).sum().item()

            for idx in range(y.size(0)):
                if y[idx] in top3_preds[idx]:
                    correct_top3 += 1

            val_preds.extend(top1_preds.cpu().numpy())
            val_targets.extend(y.cpu().numpy())
            total += y.size(0)

    acc_top1 = (correct_top1 / total) * 100
    acc_top3 = (correct_top3 / total) * 100
    f1 = f1_score(val_targets, val_preds, average='weighted')

    print(f"Top-1 Validation Accuracy: {acc_top1:.2f}%")
    print(f"Top-3 Validation Accuracy: {acc_top3:.2f}%")
    print(f"Validation F1 Score: {f1:.4f}")

if __name__ == "__main__":
    print("Checking labels...")
    print("Unique labels in the dataset:")
    print(df['composer'].unique())

    train_df = df[df['split'] == 'test']
    val_df = df[df['split'] == 'validation']

    train_dataset = TokenDataset(train_df['sequence'].tolist(), train_df['composer'].tolist())
    val_dataset = TokenDataset(val_df['sequence'].tolist(), val_df['composer'].tolist())

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    model = MidiGRUClassifier()
    train_model(model, train_loader, val_loader)

    print("\nEvaluating best saved model on validation set...")
    model = MidiGRUClassifier()
    if os.path.exists("midi_gru_classifier_v7.pth"):
        model.load_state_dict(torch.load("midi_gru_classifier_v7.pth"))
        model.eval()
        evaluate_model(model, val_loader)
    else:
        print("No trained model found. Please ensure the training completes properly.")

    sys.exit(0)
