import numpy as np
from scipy.signal import butter, lfilter
from mne.io.snirf._snirf import RawSNIRF
from scipy.signal import butter, freqz, sosfreqz, sosfilt, sosfiltfilt

def z_normalize():
    import numpy as np

    # Example time series
    time_series = np.array([10, 20, 15, 25, 30])

    # Calculate mean and standard deviation
    mean = np.mean(time_series)
    std = np.std(time_series)

    # Z-normalize the time series
    z_normalized_series = (time_series - mean) / std

    print("Original time series:", time_series)
    print("Z-normalized time series:", z_normalized_series)
    pass

def butter_bandpass(lowcut, highcut, fs, freqs=512, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    w, h = sosfreqz(sos, worN=2000, whole=True, fs=fs)
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
