import numpy as np
import pyedflib

def read_edf(file_path):
    """
    Read an EDF file and return the signal data and header information.
    """
    with pyedflib.EdfReader(file_path) as f:
        n = f.signals_in_file
        signal_labels = f.getSignalLabels()
        data = np.zeros((n, f.getNSamples()[0]))
        for i in np.arange(n):
            data[i, :] = f.readSignal(i)
    return data, signal_labels

def preprocess_data(data):
    """
    Preprocess the EEG data. This can include filtering, normalization, etc.
    """
    # Example: Normalize data
    data_normalized = (data - np.mean(data, axis=1).reshape(-1, 1)) / np.std(data, axis=1).reshape(-1, 1)
    return data_normalized

def extract_features(data):
    """
    Extract features from the EEG data.
    """
    # Example: Extract mean and standard deviation as features
    features = np.array([np.mean(data, axis=1), np.std(data, axis=1)]).T
    return features

# Replace this with the path to your EDF file
file_path = 'chb01_01.edf'

# Read the EDF file
data, labels = read_edf(file_path)

# Preprocess the data
preprocessed_data = preprocess_data(data)

# Extract features
features = extract_features(preprocessed_data)

# Print the extracted features
print("Extracted Features:", features)
