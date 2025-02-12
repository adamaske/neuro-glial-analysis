import numpy as np
import matplotlib.pyplot as plt
from preprocessing.filter import butter_bandpass_filter, notch_filter
from analysis.frequencies import compute_psd, compute_fft
from sklearn.decomposition import PCA, FastICA
from scipy.integrate import simpson

def remove_movement_artifacts(data, n_components=5):
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(data.T)
    transformed[:, :n_components] = 0  # Remove first few components
    return pca.inverse_transform(transformed).T

def trim(data, s_freq, cut_from_start, cut_from_end):
    sample_num = len(data[0])
    
    start = int(cut_from_start * s_freq)
    end = sample_num - int(cut_from_end * s_freq)
    
    cropped = data[:, start:end]
    return cropped
    
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
    

def preprocess(data, s_freq): 
    filtered = np.zeros((data.shape))
    
    for idx in range(len(data)):
        filtered_time_series = butter_bandpass_filter(data[idx], 3, 100, s_freq, 5)
        notched = notch_filter(filtered_time_series, s_freq, freqs=[50, 60, 100])
        filtered[idx] = notched

    motion_corrected = motion_correction(filtered) 

    averaged = common_average_reference_filter(motion_corrected)
    
    normalized = normalize(averaged, znorm=True)

    return normalized

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
