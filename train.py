import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# CNN Model Definition
# -----------------------------
class EEG_CNN(nn.Module):
    def __init__(self, in_channels, seq_length):
        super(EEG_CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(2)
        pooled_size = seq_length // 8
        self.fc1 = nn.Linear(256 * pooled_size, 512)
        self.fc2 = nn.Linear(512, in_channels * seq_length)  # predict for each channel & timestep
        self.in_channels = in_channels
        self.seq_length = seq_length

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x.view(x.size(0), self.in_channels, self.seq_length)

# -----------------------------
# Dataset Class
# -----------------------------
class EEGDataset(Dataset):
    def __init__(self, data_root, label_root, split, num_channels=22, allow_pickle=False):
        self.data_files = []
        self.label_files = []
        split_data_dir = os.path.join(data_root, split)
        split_label_dir = os.path.join(label_root, split)

        # Pair by filename present in data dir
        names = sorted([f for f in os.listdir(split_data_dir) if f.endswith(".npy")])
        if len(names) == 0:
            raise FileNotFoundError(f"No .npy files found in {split_data_dir}")

        for name in names:
            dpath = os.path.join(split_data_dir, name)
            lpath = os.path.join(split_label_dir, name)
            if not os.path.isfile(lpath):
                raise FileNotFoundError(f"Missing label file for {name}: expected at {lpath}")
            self.data_files.append(dpath)
            self.label_files.append(lpath)

        # Eager-load and concatenate (keeps behavior similar to your old code)
        segments_list, labels_list = [], []
        for dpath, lpath in zip(self.data_files, self.label_files):
            data = np.load(dpath, allow_pickle=allow_pickle)
            labels = np.load(lpath, allow_pickle=allow_pickle)

            if data.ndim != 3:
                raise ValueError(f"Data {dpath} must be 3D (N,C,T), got shape {data.shape}")
            if labels.shape != data.shape:
                raise ValueError(f"Shape mismatch: {dpath} {data.shape} vs {lpath} {labels.shape}")

            # Truncate/validate channels
            if data.shape[1] < num_channels:
                raise ValueError(f"{dpath}: channels {data.shape[1]} < required {num_channels}")
            data = data[:, :num_channels, :]
            labels = labels[:, :num_channels, :]

            segments_list.append(data)
            labels_list.append(labels)

        self.segments = np.concatenate(segments_list, axis=0)
        self.labels = np.concatenate(labels_list, axis=0)

    def __len__(self):
        return self.segments.shape[0]

    def __getitem__(self, idx):
        x = torch.tensor(self.segments[idx], dtype=torch.float32)   # (C,T)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)     # (C,T) binary 0/1
        return x, y

# -----------------------------
# Training function (unchanged)
# -----------------------------
def train_model(train_loader, val_loader, model, optimizer, criterion, num_epochs, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = criterion.to(device)

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss_total = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss_total += loss.item()

        avg_train_loss = train_loss_total / len(train_loader)

        model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss_total += criterion(outputs, labels).item()

        avg_val_loss = val_loss_total / len(val_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs("./models", exist_ok=True)
            torch.save(model.state_dict(), "./models/best_model4.pth")
            print(f"Saved new best model at epoch {epoch+1} with Val Loss = {avg_val_loss:.4f}")

# -----------------------------
# Main execution
# -----------------------------
if __name__ == "__main__":
    data_root  = "./data/preprocessed"
    label_root = "./data/labels"         
    train_split = "train"
    val_split   = "eval"

    batch_size = 32
    num_epochs = 30
    seq_length = 2560
    input_channels = 22

    # Load datasets from the new structure
    train_dataset = EEGDataset(data_root, label_root, train_split, num_channels=input_channels)
    val_dataset   = EEGDataset(data_root, label_root, val_split,   num_channels=input_channels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

    # Class balance for BCEWithLogitsLoss
    y_flat = train_dataset.labels.flatten()
    n_pos = np.sum(y_flat == 1)
    n_neg = np.sum(y_flat == 0)
    print(n_pos)
    print(n_neg)
    if n_pos == 0:
        raise ValueError("No positive labels found; pos_weight would be inf.")
    pos_weight = torch.tensor([n_neg/n_pos], dtype=torch.float32)

    model = EEG_CNN(in_channels=input_channels, seq_length=seq_length)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    print(f"Train segments: {len(train_dataset)} | Val segments: {len(val_dataset)}")
    train_model(train_loader, val_loader, model, optimizer, criterion, num_epochs=num_epochs)
