"""
train_test.py

This script performs binary classification (normal vs. abnormal) using CLIP visual embeddings.
It includes model training, validation, testing, and metric visualization (ROC and PR curves).

Steps:
    1. Load CLIP features and labels.
    2. Split into train, validation, and test sets.
    3. Train a simple feedforward neural network.
    4. Evaluate on validation and test sets.
    5. Plot ROC and Precision-Recall curves.
    6. Save the trained model.

Args:
    - clip_features_final.npy: Extracted CLIP features from clips.
    - clip_labels_final.npy: Binary labels (0 = normal, 1 = abnormal).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    auc
)
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# ==== Load CLIP Features and Labels ====
X = np.load(r"E:\score__fclayers\work\clip_features_final.npy")
y = np.load(r"E:\score__fclayers\work\clip_labels_final.npy")

# ==== Split into Train (60%), Val (20%), Test (20%) ====
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# ==== Convert to Torch Tensors ====
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# ==== Dataloaders ====
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=128, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=128, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=128, shuffle=False)

# ==== Define Model ====
class AnomalyDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# ==== Setup ====
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AnomalyDetector().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
epochs = 10

# ==== Training Loop ====
def train():
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs} | Loss: {total_loss:.4f}")
        evaluate()

# ==== Validation Evaluation ====
def evaluate():
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            preds = (outputs > 0.5).cpu().numpy()
            all_preds.extend(preds.flatten())
            all_labels.extend(batch_y.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    auc_score = roc_auc_score(all_labels, all_preds)
    print(f"✅ Accuracy: {acc:.4f} | 🔥 AUC: {auc_score:.4f}")

# ==== Run Training ====
train()

# ==== Save Trained Model ====
torch.save(model.state_dict(), "detector3.pth")
print("✅ Model saved as 'detector3.pth'")

# ==== Test on Unseen Data ====
model.load_state_dict(torch.load("detector3.pth", map_location=device))
model.to(device)
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        outputs = model(batch_x)
        preds = (outputs > 0.5).cpu().numpy()
        all_preds.extend(preds.flatten())
        all_labels.extend(batch_y.cpu().numpy())

acc = accuracy_score(all_labels, all_preds)
auc_score = roc_auc_score(all_labels, all_preds)
print(f"🎯 Test Accuracy: {acc:.4f} | 🌟 Test AUC: {auc_score:.4f}")

# ==== Plot ROC & PR Curves ====
fpr, tpr, _ = roc_curve(all_labels, all_preds)
roc_auc = auc(fpr, tpr)

precision, recall, _ = precision_recall_curve(all_labels, all_preds)
pr_auc = auc(recall, precision)

plt.figure(figsize=(12, 6))

# ROC Curve
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")

# Precision-Recall Curve
plt.subplot(1, 2, 2)
plt.plot(recall, precision, color='green', lw=2, label=f'PR curve (AUC = {pr_auc:.4f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")

plt.tight_layout()
plt.show()
