# 🕵️ Fake News Detection with BERT-LSTM Model

This repository contains a complete implementation of a fake news detection system using a hybrid BERT-LSTM architecture. The model combines the contextual understanding of BERT with the sequence modeling capabilities of LSTM for high-accuracy classification of news articles.

## ✨ Features

- **Hybrid BERT-LSTM Architecture** - Combines transformer power with sequential modeling
- **Comprehensive Text Preprocessing** - Cleans and normalizes news text
- **Efficient Training Pipeline** - Optimized for Google Colab with GPU support
- **High Accuracy Classification** - Achieves state-of-the-art performance on Fake News Dataset
- **End-to-End Solution** - Complete workflow from data loading to evaluation

## 🚀 Complete Implementation

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Install dependencies
!pip install torch pandas numpy scikit-learn transformers nltk requests beautifulsoup4

import pandas as pd
from sklearn.model_selection import train_test_split
import re
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.metrics import accuracy_score

# Load and preprocess data
true_news = pd.read_csv('/content/drive/MyDrive/News_dataset/True.csv')
fake_news = pd.read_csv('/content/drive/MyDrive/News_dataset/Fake.csv')

# Add labels
true_news['label'] = 1  # Real news
fake_news['label'] = 0  # Fake news

# Combine and shuffle datasets
df = pd.concat([true_news, fake_news]).sample(frac=1).reset_index(drop=True)

# Text cleaning function
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)       # Remove digits
    text = text.lower()                   # Convert to lowercase
    return text

# Apply cleaning to text column
df['text'] = df['text'].apply(clean_text)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)

# Define BERT-LSTM model architecture
class FakeNewsClassifier(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', lstm_units=128):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.lstm = nn.LSTM(
            input_size=self.bert.config.hidden_size,
            hidden_size=lstm_units,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(lstm_units * 2, 1)  # *2 for bidirectional
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        lstm_out, _ = self.lstm(sequence_output)
        lstm_out = lstm_out[:, -1, :]  # Take last hidden state
        
        x = self.dropout(lstm_out)
        return torch.sigmoid(self.classifier(x))

# Custom Dataset class
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(self.labels[idx], dtype=torch.float)
        }

# Initialize tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = FakeNewsClassifier()
optimizer = AdamW(model.parameters(), lr=2e-5)
criterion = nn.BCELoss()

# Create DataLoaders
train_dataset = NewsDataset(X_train.tolist(), y_train.tolist(), tokenizer)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Set device (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Training loop
for epoch in range(3):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs.view(-1), labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f'Epoch {epoch+1}, Average Loss: {total_loss/len(train_loader):.4f}')

# Save trained model and tokenizer
torch.save(model.state_dict(), 'fakenews_model.pt')
tokenizer.save_pretrained('tokenizer')

# Evaluation on test set
test_dataset = NewsDataset(X_test.tolist(), y_test.tolist(), tokenizer)
test_loader = DataLoader(test_dataset, batch_size=16)

model.eval()
predictions = []
true_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].cpu().numpy()
        
        outputs = model(input_ids, attention_mask).cpu().numpy()
        predictions.extend(outputs > 0.5)
        true_labels.extend(labels)

# Calculate and print accuracy
accuracy = accuracy_score(true_labels, predictions)
print(f'Test Accuracy: {accuracy:.4f}')
```
# Dataset
![You can download dataset from:](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets)
