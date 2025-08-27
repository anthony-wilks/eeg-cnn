import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def concat_segments_collate(batch):
    datas, labels = zip(*batch)  # tuples of tensors
    data_cat = torch.cat(datas, dim=0)
    label_cat = torch.cat(labels, dim=0)
    return data_cat, label_cat
    
# CNN Model Definition (same as before)
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
    

class EEGDataset(Dataset):
    def __init__(self, data_folder, label_folder, num_channels=21):
        self.data_files = sorted([os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.npy')])
        self.label_files = sorted([os.path.join(label_folder, f) for f in os.listdir(label_folder) if f.endswith('.npy')])

        self.num_channels = num_channels

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        data = np.load(self.data_files[idx])  
        labels = np.load(self.label_files[idx])  

        # Ensure the data, labels have 21 channels
        if data.shape[1] > self.num_channels:
            data = data[:, :self.num_channels, :]  
            labels = labels[:, :self.num_channels, :]

        # Convert to torch tensors
        return torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)


# Evaluate the model on the test dataset
def evaluate_model(test_data_folder, test_label_folder, model, batch_size=32):
    test_dataset = EEGDataset(test_data_folder, test_label_folder)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=concat_segments_collate, pin_memory=True)

    model.eval()


    all_preds = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for inputs, labels in test_loader:  

            outputs = model(inputs)                
            preds = (outputs >= 0.5).float()

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    preds  = np.concatenate(all_preds,  axis=0).reshape(-1)
    labels = np.concatenate(all_labels, axis=0).reshape(-1)

    accuracy  = accuracy_score(labels, preds)
    precision = precision_score(labels, preds)
    recall    = recall_score(labels, preds)
    f1        = f1_score(labels, preds)
    auc_roc   = roc_auc_score(labels, preds)  

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"AUC-ROC   : {auc_roc:.4f}")


# Main function for evaluating the model
if __name__ == "__main__":
    test_data_folder = './data/processed_data/test'  
    test_label_folder = './data/processed_data/test_labels'  


    model = EEG_CNN(in_channels=21, seq_length=2560)
    model.load_state_dict(torch.load("./models/best_model.pth"))  # Load the best model

    evaluate_model(test_data_folder, test_label_folder, model)
