import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from os.path import split, splitext, join
from os import remove

from datasets.fnirs import read_snirf, find_snirf_in_folder
from visualization.fnirs import plot_snirf, plot_psd_snirf
from preprocessing.filter import butter_bandpass_filter, notch_filter

from mne.io import read_raw_snirf
from mne_nirs.io import write_raw_snirf
from mne.preprocessing.nirs import optical_density, beer_lambert_law,  temporal_derivative_distribution_repair
from mne.io.snirf._snirf import RawSNIRF
from snirf import validateSnirf

from wrappers.fnirs import fNIRS

def wl_to_od(snirf:RawSNIRF):
    return optical_density(snirf)

def od_to_hb(snirf:RawSNIRF):
    return beer_lambert_law(snirf)

def wl_to_hb(snirf:RawSNIRF):
    od = wl_to_od(snirf)
    return od_to_hb(od)
 
def motion_correction(snirf:RawSNIRF) -> "RawSNIRF":
    """
    This removes baseline shift and spike artifacts.\n
    It is recommended to apply this to optical density data, however hemoglobin will also work. 
   
    Args:
        snirf (): mne.raw snrif object or filepath to ".snirf" with either optical density of hemoglobin_concentration
    Returns:
        raw (): mne.raw snirf object with baseline
    
    """

    corrected = temporal_derivative_distribution_repair(snirf)
    return corrected

def channel_rejection(snirf:RawSNIRF, threshold=0.1) -> "RawSNIRF":
    """Remove channels with coefficient of variation (CoV) above threshold."""
    data = snirf.get_data()  # Extract fNIRS data
    mean_vals = np.mean(data, axis=1)  # Compute mean per channel
    std_vals = np.std(data, axis=1)  # Compute std per channel

    cov = std_vals / mean_vals  # Compute CoV
    valid_channels = cov <= threshold  # Keep channels with CoV <= 10%
    ch_names = snirf.info['channels'] 
    snirf.drop_channels()
    # Apply mask to remove bad channels
    snirf._data = data[valid_channels]
    snirf.info['channels'] = [ch for i, ch in enumerate(snirf.info['channels']) if valid_channels[i]]

    return snirf

def bandpass_filter_snirf(snirf:RawSNIRF, l_freq=0.01, h_freq=0.1, n=5) -> RawSNIRF:
    """
    Applies a digital bandpass filter to all channels. Returns filtered snirf object. 
    
    Args:
        snirf (RawSNIRF) : RawSNIRF object
        l_freq : Lowcut frequency, the lower edge of passband
        h_freq : Highcut frequency, the high edge of passband  
        n : Filter order, higher means small transition band
        
    Returns:
        filtered (RawSNIRF) : New RawSNIRF object with filtered channels
    """

    channels = snirf.get_data()
    s_freq = snirf.info["sfreq"]

    filtered_channels = np.zeros_like(channels)  # Ensures matching shape and type

    for idx in range(channels.shape[0]):  # Iterate over each channel
        filtered_channels[idx] = butter_bandpass_filter(channels[idx], l_freq, h_freq, s_freq, n)

    filtered = snirf.copy()
    filtered._data = filtered_channels.astype(np.float64)  # Ensure proper update

    return filtered

def z_normalize_snirf(snirf: RawSNIRF) -> RawSNIRF:
    """
    Applies Z-Normalization to all channels. Returns a normalized snirf object.
    
    Args:
        snirf (RawSNIRF): RawSNIRF object
    
    Returns:
        normalized (RawSNIRF): New RawSNIRF object with Z-normalized channels
    """
    
    channels = snirf.get_data()
    
    mean_vals = np.mean(channels, axis=1, keepdims=True)  # Compute mean per channel
    std_vals = np.std(channels, axis=1, keepdims=True)  # Compute standard deviation per channel
    
    # Avoid division by zero
    std_vals[std_vals == 0] = 1
    
    normalized_channels = (channels - mean_vals) / std_vals  # Apply Z-Normalization
    
    normalized = snirf.copy()
    normalized._data = normalized_channels.astype(np.float64)  # Ensure proper update
    
    return normalized

def spike_removal_snirf(snirf: RawSNIRF, n_components=5) -> RawSNIRF:
    """
    Removes spikes/artifacts using PCA by eliminating the first few components.
    Returns a cleaned snirf object.
    
    Args:
        snirf (RawSNIRF): RawSNIRF object
        n_components (int): Number of principal components to remove
    
    Returns:
        cleaned (RawSNIRF): New RawSNIRF object with spike-reduced channels
    """

    channels = snirf.get_data()
    
    # Apply PCA
    pca = PCA()
    transformed = pca.fit_transform(channels.T)  # Transpose so PCA works on timepoints
    
    # Zero out the first few components (assumed to be noise/artifacts)
    transformed[:, :n_components] = 0  
    
    # Reconstruct the signal
    cleaned_channels = pca.inverse_transform(transformed).T  # Transpose back
    
    cleaned = snirf.copy()
    cleaned._data = cleaned_channels.astype(np.float64)  # Ensure proper update
    
    return cleaned

def preprocess_snirf(snirf, od=True, hb=True, filter=True, cv=False, tddr=False, z_norm=False, spike_removal=False):
    current = snirf.copy()
    
    # Optical density
    if od:
        current = wl_to_od(current)
    
    # Channel rejection  
    if cv:
        current = channel_rejection(current, 0.1) # 10% CV
    
    # Normalization -> Z-Normalization
    if z_norm:
        current = z_normalize_snirf(current)
        pass

    # Spike Removal -> PCA 
    if spike_removal:
        current = spike_removal_snirf(current, n_components=51)

    # TDDR
    if tddr:
        current = motion_correction(current)


    # Bandpass Filtering
    if filter:
        lowcut = 0.01 
        highcut = 0.1
        order = 10
        current = bandpass_filter_snirf(current, lowcut, highcut, order)
        
    # Hemoglobin
    if hb:
        current = od_to_hb(current)
    
    return current
