import numpy as np
from scipy.signal import butter, lfilter
from scipy.signal import butter, freqz, sosfreqz, sosfilt, sosfiltfilt, iirnotch, filtfilt

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

def notch_filter(data, sfreq, freqs=[50, 60]):
    """Apply notch filters at specified frequencies."""
    for freq in freqs:
        b, a = iirnotch(freq, 30, sfreq)
        data = filtfilt(b, a, data, axis=-1)
    return data