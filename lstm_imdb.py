# LSTM model with IMDb database
# before running install and virtual environment:
# python3 -m venv .venv
# source .venv/bin/activate
# pip install datasets transformers torch scikit-learn pandas

from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from collections import Counter
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 1. Load dataset
dataset = load_dataset("stanfordnlp/imdb")
train_data = dataset["train"].shuffle(seed=42).select(range(5000))  # lower range due to testability and storage
test_data = dataset["test"].shuffle(seed=42).select(range(1000))

# 2. Simple preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text

train_texts = [clean_text(x["text"]) for x in train_data]
test_texts = [clean_text(x["text"]) for x in test_data]
train_labels = [x["label"] for x in train_data]
test_labels = [x["label"] for x in test_data]

# 3. Build vocabulary
counter = Counter()
for text in train_texts:
    counter.update(text.split())

vocab = {"<PAD>": 0, "<UNK>": 1}
for word, _ in counter.most_common(10000):
    vocab[word] = len(vocab)

def encode_text(text, max_len=200):
    tokens = text.split()
    encoded = [vocab.get(token, vocab["<UNK>"]) for token in tokens[:max_len]]
    if len(encoded) < max_len:
        encoded += [vocab["<PAD>"]] * (max_len - len(encoded))
    return encoded

train_encoded = [encode_text(text) for text in train_texts]
test_encoded = [encode_text(text) for text in test_texts]

# 4. Dataset class
class IMDBDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = torch.tensor(texts, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

train_dataset = IMDBDataset(train_encoded, train_labels)
test_dataset = IMDBDataset(test_encoded, test_labels)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# 5. LSTM model
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, output_dim=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        output = self.fc(hidden[-1])
        return output

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMClassifier(len(vocab)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 6. Train
for epoch in range(3):
    model.train()
    total_loss = 0
    for texts, labels in train_loader:
        texts, labels = texts.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

# 7. Evaluate
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for texts, labels in test_loader:
        texts = texts.to(device)
        outputs = model(texts)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

acc = accuracy_score(all_labels, all_preds)
precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="binary")

print("\nLSTM Results:")
print("accuracy:", acc)
print("precision:", precision)
print("recall:", recall)
print("f1:", f1)

# Outputs: 
# Epoch 1, Loss: 0.6948
# Epoch 2, Loss: 0.6731
# Epoch 3, Loss: 0.6343

# LSTM Results:
# accuracy: 0.524
# precision: 0.53125
# recall: 0.20901639344262296
# f1: 0.3
