import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import argparse

# 1. Load and preprocess data
df = pd.read_csv("../ml-data/all_articles.csv")
df = df.dropna(subset=['urlToImage', 'reliability_score', 'title'])

# Text feature: length of title only
df['title_len'] = df['title'].apply(lambda x: len(str(x)))

# Train/test split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Normalize text feature
scaler = StandardScaler()
train_text_feats = scaler.fit_transform(train_df[['title_len']])
test_text_feats = scaler.transform(test_df[['title_len']])

# Image transform
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# 2. Dataset
class NewsDataset(Dataset):
    def __init__(self, df, text_feats, transform):
        self.df = df.reset_index(drop=True)
        self.text_feats = text_feats
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = torch.tensor(self.text_feats[idx], dtype=torch.float)
        # Load image
        try:
            response = requests.get(row['urlToImage'], timeout=5)
            image = Image.open(BytesIO(response.content)).convert('RGB')
            image = self.transform(image)
        except:
            image = torch.zeros((3, 64, 64))
        label = torch.tensor(row['reliability_score'], dtype=torch.float)
        return text, image, label

# DataLoaders
train_dataset = NewsDataset(train_df, train_text_feats, transform)
test_dataset = NewsDataset(test_df, test_text_feats, transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8)

# 3. Model
class MultimodalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_fc = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU()
        )
        self.img_cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * 31 * 31, 32),
            nn.ReLU()
        )
        self.combined = nn.Sequential(
            nn.Linear(8 + 32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, text, image):
        text_out = self.text_fc(text)
        image_out = self.img_cnn(image)
        combined = torch.cat((text_out, image_out), dim=1)
        return self.combined(combined).squeeze(1)

# 4. Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultimodalModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

for epoch in range(5):
    model.train()
    total_loss = 0
    for text, image, label in train_loader:
        text, image, label = text.to(device), image.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(text, image)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

# 5. Predict reliability for a new post
def predict_reliability(title, image_url):
    title_len = len(str(title))
    text_feats = scaler.transform([[title_len]])
    text_tensor = torch.tensor(text_feats, dtype=torch.float).to(device)

    try:
        response = requests.get(image_url, timeout=5)
        image = Image.open(BytesIO(response.content)).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
    except:
        image_tensor = torch.zeros((1, 3, 64, 64)).to(device)

    text_tensor = text_tensor if text_tensor.ndim == 2 else text_tensor.unsqueeze(0)
    model.eval()
    with torch.no_grad():
        pred = model(text_tensor, image_tensor)
        return float(pred.cpu().numpy()[0])

# Example usage:
if __name__ == "__main__":
    # Example post
    new_title = "Following PM Sheikh Hasina's resignation and departure, madrasa students are guarding the Hindu temples and ensuring safety against fundamentalist threats."
    new_image_url = "../ml-data/test-pic.png"
    score = predict_reliability(new_title, new_image_url)
    print(f"Percentage fake score: 100 - {score:.2f}")
    
