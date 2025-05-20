import pandas as pd
import numpy as np
import torch

def prepare_data(csv_name) -> tuple:
    df = pd.read_csv(csv_name, header=None)
    # Extract time series data as numpy array
    time_series = df.iloc[:, 1:1228].values
    # Extract labels and convert 0 to -1 for binary classification
    labels = df.iloc[:, 0].values
    labels = np.where(labels == 0, -1, 1)

    # Store FFT of time-series data
    fft_data = np.fft.fft(time_series, axis = -1)
    fft_data = np.abs(fft_data)
    fft_normalized = (fft_data - fft_data.min(axis=1, keepdims=True)) / (fft_data.max(axis=1, keepdims=True) - fft_data.min(axis=1, keepdims=True))

    # Create 3D tensor with original signal and FFT as two spatial dimensions
    data_3d = np.zeros((time_series.shape[0], 2, time_series.shape[1]))
    data_3d[:, 0, :] = time_series
    data_3d[:, 1, :] = fft_normalized

    # Convert to PyTorch tensors
    data_tensor = torch.tensor(data_3d, dtype=torch.float64, requires_grad=False)
    label_tensor = torch.tensor(labels, requires_grad=False)

    # Split into training and validation sets
    train_ratio = 0.7
    n_samples = data_tensor.shape[0]
    n_train = int(train_ratio * n_samples)
    #Shuffling the samples
    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_data = data_tensor[train_indices]
    train_label = label_tensor[train_indices]
    val_data = data_tensor[val_indices]
    val_label = label_tensor[val_indices]

    return (train_data, train_label, val_data, val_label)

if __name__ == "__main__":
    train_data, train_label, val_data, val_label = prepare_data('sag_healthy.csv')
    print("Train data shape:", train_data.shape)
    print("Train label shape:", train_label.shape)
    print("Validation data shape:", val_data.shape)
    print("Validation label shape:", val_label.shape)