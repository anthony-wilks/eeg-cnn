import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

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
# Dataset Class for train/validation
# -----------------------------
class EEGDataset(Dataset):
    def __init__(self, folder, num_channels=21):
        self.files = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.npy')])
        self.segments = []
        self.labels = []

        for fpath in self.files:
            arr = np.load(fpath, allow_pickle=True)
            if isinstance(arr, np.ndarray) and arr.dtype == 'O':
                arr = arr.item()
            data = arr['data'][:, :num_channels, :]
            labels = arr['labels'][:, :num_channels, :]

            # Add data and labels to lists
            self.segments.append(data)
            self.labels.append(labels)

        # Flatten across all filess
        self.segments = np.concatenate(self.segments, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        return torch.tensor(self.segments[idx], dtype=torch.float32), \
               torch.tensor(self.labels[idx], dtype=torch.float32)

# -----------------------------
# Training function
# -----------------------------
def train_model(train_loader, val_loader, model, optimizer, criterion, num_epochs, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = criterion.to(device)

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss_total = 0

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
        val_loss_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss = criterion(outputs, labels)
                val_loss_total += val_loss.item()

        avg_val_loss = val_loss_total / len(val_loader)

        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "./models/best_model.pth")
            print(f"Saved new best model at epoch {epoch+1} with Val Loss = {avg_val_loss:.4f}")

# -----------------------------
# Main execution
# -----------------------------
if __name__ == "__main__":
    train_folder = './data/processed_data/train'
    val_folder   = './data/processed_data/eval'   

    batch_size = 32
    num_epochs = 30
    seq_length = 2560
    input_channels = 21

    # Load datasets
    train_dataset = EEGDataset(train_folder)
    val_dataset   = EEGDataset(val_folder)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # y_flat = train_dataset.labels.flatten()
    # n_total = len(y_flat)
    # n_pos = np.sum(y_flat == 1)
    # n_neg = np.sum(y_flat == 0)
    pos_weight = torch.tensor([10], dtype=torch.float32)

    model = EEG_CNN(in_channels=input_channels, seq_length=seq_length)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_model(train_loader, val_loader, model, optimizer, criterion, num_epochs=num_epochs)
