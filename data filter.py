import os
import numpy as np
import pyedflib
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Read seizure information from a .seizures file with error handling
def read_seizure_file(seizure_file):
    seizure_info = []
    with open(seizure_file, 'r') as file:
        for line in file:
            try:
                start, end = map(int, line.split())
                seizure_info.append((start, end))
            except ValueError:
                print(f"Warning: Skipping unparseable line in {seizure_file}: {line.strip()}")
    return seizure_info

# Read an EDF file and return the signal data and header information
def read_edf(file_path):
    with pyedflib.EdfReader(file_path) as f:
        n = f.signals_in_file
        signal_labels = f.getSignalLabels()
        data = np.zeros((n, f.getNSamples()[0]))
        for i in np.arange(n):
            data[i, :] = f.readSignal(i)
    return data, signal_labels

# Preprocessing steps for the EEG data
def preprocess_data(data):
    return data  # Modify as needed

# LSTM model for EEG classification
def create_model(input_shape, num_classes):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(100, input_shape=input_shape, return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.LSTM(100))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))  # Use 'sigmoid' for binary classification
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Data generator for EEG data
class EEGDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, file_paths, labels, batch_size, sample_size, num_classes):
        self.file_paths = file_paths
        self.labels = labels
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.num_classes = num_classes

    def __len__(self):
        return int(np.ceil(len(self.file_paths) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_file_paths = self.file_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_labels = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_data = []
        for file_path in batch_file_paths:
            data, _ = read_edf(file_path)
            processed_data = preprocess_data(data)
            batch_data.append(processed_data[:self.sample_size])

        X = np.array(batch_data)
        y = tf.keras.utils.to_categorical(batch_labels, self.num_classes)

        return X, y

# Process the EEG data files
def process_folder(folder_path, num_classes, sample_size):
    all_data = []
    all_labels = []

    for file in os.listdir(folder_path):
        if file.endswith(".edf"):
            edf_path = os.path.join(folder_path, file)
            data, _ = read_edf(edf_path)
            processed_data = preprocess_data(data)

            num_samples = processed_data.shape[1] // sample_size
            for i in range(num_samples):
                sample = processed_data[:, i * sample_size:(i + 1) * sample_size]
                # Reshape sample to add a 'feature' dimension
                sample = sample.reshape((sample.shape[0], sample.shape[1], 1))
                all_data.append(sample)
                has_seizure = int(os.path.exists(edf_path + '.seizures'))
                all_labels.append(has_seizure)

    # Convert to numpy arrays
    X = np.array(all_data)
    y = np.array(all_labels)
    y = tf.keras.utils.to_categorical(y, num_classes)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = create_model(input_shape=(X_train.shape[1], X_train.shape[2]), num_classes=num_classes)
    model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2)
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test loss: {loss}, Test accuracy: {accuracy}')


# Define sample_size, folder_path, and num_classes
sample_size = 256  # Adjust as needed
folder_path = 'chb01'  # Update this to your folder path
num_classes = 2  # Update based on your classification task

# Run the processing
process_folder(folder_path, num_classes, sample_size)
