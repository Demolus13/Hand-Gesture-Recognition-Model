import os
import numpy as np
import pandas as pd
import pickle

import torch
from scripts.model import HGRModel

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to load and combine CSV files
def load_data_from_csv(directory):
    data = []
    labels = []
    for file_name in os.listdir(directory):
        if file_name.endswith('.csv'):
            file_path = os.path.join(directory, file_name)
            df = pd.read_csv(file_path)
            data.append(df.values)
            label = file_name.split('.')[0]
            labels.extend([label] * len(df))
    return np.vstack(data), labels

# Function to encode labels
def encode_labels(labels):
    unique_labels = list(set(labels))
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    return [label_to_index[label] for label in labels], len(unique_labels), label_to_index

# Load and preprocess data
data_dir = os.path.join("data", "processed")
data, labels = load_data_from_csv(data_dir)
labels, num_classes, label_to_index = encode_labels(labels)

# Convert data to PyTorch tensors
X = torch.tensor(data, dtype=torch.float32).to(device)
Y = torch.tensor(labels, dtype=torch.long).to(device)

# Initialize the model
in_features = X.shape[1]
model = HGRModel(in_features, num_classes).to(device)

# Train the model
model.fit(X, Y, epochs=100, lr=0.01)

# Save the model and label encodings
os.makedirs("models", exist_ok=True)
model_file_path = os.path.join("models", "hgr_model.pth")
label_encodings_file_path = os.path.join("models", "label_encodings.pkl")

model.save_model(model_file_path)

with open(label_encodings_file_path, 'wb') as f:
    pickle.dump(label_to_index, f)