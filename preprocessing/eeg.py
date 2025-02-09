import mne
import numpy as np
from preprocessing.filter import butter_bandpass_filter, notch_filter

def remove_movement_artifacts(data, n_components=5):
    """Remove movement artifacts using PCA."""
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(data.T)
    transformed[:, :n_components] = 0  # Remove first few components
    return pca.inverse_transform(transformed).T

def preprocess_eeg(file_path):
    """Load and preprocess EEG data."""
    raw = mne.io.read_raw(file_path, preload=True)
    sfreq = raw.info['sfreq']
    data = raw.get_data()
    
    # Apply filters
    data = butter_bandpass_filter(data, sfreq)
    data = notch_filter(data, sfreq)
    data = remove_movement_artifacts(data)
    
    raw._data = data  # Update raw object with cleaned data
    return raw
