import torch
import torch.nn as nn
import numpy as np
import os
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


def concat_segments_collate(batch):
    datas, labels = zip(*batch)  # list of (N,C,T)
    data_cat = torch.cat(datas, dim=0)   # (ΣN, C, T)
    label_cat = torch.cat(labels, dim=0)
    return data_cat, label_cat


# CNN Model Definition 
class EEG_CNN(nn.Module):
    def __init__(self, in_channels, seq_length):
        super(EEG_CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(2)
        pooled_size = seq_length // 8
        self.fc1 = nn.Linear(256 * pooled_size, 512)
        self.fc2 = nn.Linear(512, in_channels * seq_length)  
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


class EEGDataset(Dataset):
    def __init__(self, data_folder, label_folder, num_channels=22):
        self.data_folder = Path(data_folder)
        self.label_folder = Path(label_folder)
        self.num_channels = num_channels

        # match files by stem
        data_files = {f.stem: f for f in self.data_folder.glob("*.npy")}
        label_files = {f.stem: f for f in self.label_folder.glob("*.npy")}
        common = sorted(set(data_files.keys()) & set(label_files.keys()))
        if not common:
            raise RuntimeError("No matching data/label files found.")

        self.pairs = [(data_files[stem], label_files[stem]) for stem in common]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        data_path, label_path = self.pairs[idx]
        data = np.load(data_path)     # (N,C,T)
        labels = np.load(label_path)  # (N,C,T)

        # ensure consistent channel count
        if data.shape[1] > self.num_channels:
            data = data[:, :self.num_channels, :]
            labels = labels[:, :self.num_channels, :]

        return torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)


# Evaluate the model on the test dataset
def evaluate_model(test_data_folder, test_label_folder, model, batch_size=32, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    test_dataset = EEGDataset(test_data_folder, test_label_folder)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=concat_segments_collate,
        pin_memory=True
    )


    all_probs = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(inputs)                   # (N,C,T) logits
            probs = torch.sigmoid(logits)            # (N,C,T) probabilities in [0,1]
            preds = (probs >= 0.5).float()           # threshold in prob space

            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    probs  = np.concatenate(all_probs, axis=0).reshape(-1)
    preds  = np.concatenate(all_preds, axis=0).reshape(-1).astype(np.int32)
    labels = np.concatenate(all_labels, axis=0).reshape(-1).astype(np.int32)

    accuracy  = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall    = recall_score(labels, preds, zero_division=0)
    f1        = f1_score(labels, preds, zero_division=0)

    # AUC requires both classes present
    if (labels.max() == labels.min()):
        auc_roc = float('nan')
    else:
        auc_roc = roc_auc_score(labels, probs)

    # Confusion matrix → TN, FP, FN, TP
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0,1]).ravel()

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"AUC-ROC   : {auc_roc:.4f}" if np.isfinite(auc_roc) else "AUC-ROC   : N/A (single class in labels)")
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

# Main function for evaluating the model
if __name__ == "__main__":
    test_data_folder = "./data/preprocessed/test"
    test_label_folder = "./data/labels/test"

    model = EEG_CNN(in_channels=22, seq_length=2560)
    model.load_state_dict(torch.load("./models/best_model4.pth"))

    evaluate_model(test_data_folder, test_label_folder, model)
