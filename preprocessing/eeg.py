import numpy as np
import matplotlib.pyplot as plt
from preprocessing.filter import butter_bandpass_filter, notch_filter
from analysis.frequencies import compute_psd, compute_fft
from sklearn.decomposition import PCA, FastICA
from scipy.integrate import simpson
from wrappers.eeg import EEG

def remove_movement_artifacts(data, n_components=5):
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(data.T)
    transformed[:, :n_components] = 0  # Remove first few components
    return pca.inverse_transform(transformed).T

def trim(eeg:EEG, cut_from_start:float, cut_from_end:float):
    print(f"Trimming EEG : [ {-cut_from_start}, {cut_from_end} ]")
    s_freq = eeg.sampling_frequency
    
    sample_num = eeg.channel_data.shape[1]
    
    start = int(cut_from_start * s_freq)
    end = sample_num - int(cut_from_end * s_freq)
    
    eeg.channel_data = eeg.channel_data[:, start:end]
    
    valid_indices = [i for i, onset in enumerate(eeg.feature_onsets) if start <= onset < end]
    eeg.feature_onsets = np.array([eeg.feature_onsets[i] - start for i in valid_indices])
    eeg.feature_descriptions = [eeg.feature_descriptions[i] for i in valid_indices]
    
def bandpass_channels(data, s_freq, lowcut, highcut, order):
    filtered = np.zeros((data.shape))
    
    for idx in range(len(data)):
        filtered[idx] = butter_bandpass_filter(data[idx], lowcut, highcut, s_freq, order)
    
    return filtered
def common_average_reference_filter(data):
    average_reference = np.mean(data, axis=0)
    averaged = data - average_reference
    return averaged

def normalize(data, znorm=False, baseline_correction=False, log=False):
    normalized = data
    # Z-Normalization
    if znorm:
        normalized = (data - np.mean(data, axis=1, keepdims=True)) / np.std(data, axis=1, keepdims=True)
    
    # Baseline Correction
    if baseline_correction:
        baseline = data[:, :2560]  # Assume first 50 samples are pre-stimulus
        baseline_mean = np.mean(baseline, axis=1, keepdims=True)
        normalized = data - baseline_mean
    # Log Transformation
    if log:
        normalized = np.log1p(np.abs(data))
        
    return normalized

def motion_correction(data):
    motion_corrected = data
    n = 5
    #pca = PCA(n_components=n)
    #ica = FastICA(n_components=n)
    #transformed_pca = pca.fit_transform(filtered.transpose())  # Transpose so PCA works on timepoints
    #transformed_ica = ica.fit_transform(filtered.transpose())  # Transpose so PCA works on timepoints
    #transformed_pca[:, :n] = 0  
    #transformed_ica[:, :2] = 0  
    #cleaned_pca = pca.inverse_transform(transformed_pca).transpose()  # Transpose back
    #cleaned_ica = ica.inverse_transform(transformed_ica).transpose()  # Transpose back
    return motion_corrected
    

def preprocess(eeg:EEG, bandpass=True, normalization=True, ica=True): 
    
    # Copy data to modify
    processed_data = eeg.channel_data.copy()
    if ica:
        ica = FastICA(n_components=processed_data.shape[0])
        transformed_ica = ica.fit_transform(processed_data.transpose())  # Transpose so PCA works on timepoints
        transformed_ica[:, :processed_data.shape[0]-1] = 0  
        cleaned_ica = ica.inverse_transform(transformed_ica).transpose()  # Transpose back
        eeg.preprocessing_history.append({'ica': {}})
        processed_data = cleaned_ica
        
        
    if bandpass: # Apply Bandpass filtering -> 1 to 100 Hz, 50, 60, 100 Hz notch filter
        for idx in range(len(processed_data)):
            filtered_time_series = butter_bandpass_filter(processed_data[idx], 8, 30, eeg.sampling_frequency, 10)
            notched = notch_filter(filtered_time_series, eeg.sampling_frequency, freqs=[50, 60, 100])
            processed_data[idx] = notched
              
        eeg.preprocessing_history.append({'bandpass': {'lowcut': 1, 'highcut': 100, 'notch_freqs': [50, 60, 100]}})
        
    
    # Motion Correction -> Not implemented
    processed_data = motion_correction(processed_data) 
    eeg.preprocessing_history.append({'motion_correction': {}})
    
    # Common Average Refrence 
    processed_data = common_average_reference_filter(processed_data)
    eeg.preprocessing_history.append({'common_average_reference': {}})
    
    # Normalization
    if normalization:
        processed_data = normalize(processed_data, znorm=True)
        eeg.preprocessing_history.append({'normalize': {'znorm': True}})
    
    # Replace channel data
    eeg.channel_data =  processed_data
    return eeg

band_ranges_spec = {
        "Delta (0.5-4 Hz)": (0.5, 4),
        "Theta (4-8 Hz)": (4, 8),
        "Alpha (8-12 Hz)": (8, 12),
        "Beta (12-30 Hz)": (12, 30),
        "Gamma (30-100 Hz)": (30, 100)
    }
band_power_colors = ['blue', 'green', 'orange', 'red', 'purple']

def compute_band_power(spectra, freqs):
    """
    Computes band power for predefined EEG frequency bands.
    
    Args:
        spectra (numpy.ndarray): Power spectral density values.
        freqs (numpy.ndarray): Corresponding frequency values.
    
    Returns:
        dict: Band powers for Delta, Theta, Alpha, Beta, and Gamma bands.
    """
    spectra_res = spectra[1] - spectra[0] 
    spectral_density = simpson(freqs, dx=spectra_res)
    
    band_powers = {}
    for band, (low, high) in band_ranges_spec.items():
        
        band_mask = np.logical_and(spectra >= low, spectra <= high) # TRUE when the sepctra Hz is within low and high,  FALSE where spectra is below low or above high
        band_powers[band]  = simpson(freqs[band_mask], dx=spectra_res) / spectral_density
       
    return band_powers   
