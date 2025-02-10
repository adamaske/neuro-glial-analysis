import numpy as np
from scipy.signal import butter, lfilter
from mne.io.snirf._snirf import RawSNIRF
from scipy.signal import butter, freqz, sosfreqz, sosfilt, sosfiltfilt
from sklearn.decomposition import PCA

def butter_bandpass(lowcut, highcut, fs, freqs=512, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    w, h = sosfreqz(sos, worN=None, whole=True, fs=fs)
    return sos, w, h

def butter_bandpass_filter(time_series, lowcut, highcut, fs, order):
    sos, w, h = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfiltfilt(sos, time_series)
    return np.array(y)

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
