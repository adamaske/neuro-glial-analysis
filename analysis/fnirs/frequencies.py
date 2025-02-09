import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz, sosfreqz

def compute_fft(time_series, fs, freq_limit:float|None):
    # Compute FFT
    N = len(time_series)  # Length of the signal
    fft_result = np.fft.fft(time_series)
    fft_freq = np.fft.fftfreq(N, d=1/fs)  # Frequency axis

    # Take the positive half of the spectrum
    positive_freqs = fft_freq[:N // 2]
    positive_spectrum = np.abs(fft_result[:N // 2]) * (2 / N)  # Normalize for one-sided
    
    if freq_limit is None:
        return positive_freqs, positive_spectrum

    # Filter frequencies to only include up to freq_limit
    indices = positive_freqs <= freq_limit
    limited_freqs = positive_freqs[indices]
    limited_spectrum = positive_spectrum[indices]
    return limited_freqs, limited_spectrum

def compute_psd(time_series, fs, freq_limit:float|None):
    # Compute FFT
    freqs, spectrum = compute_fft(time_series, fs, freq_limit)
    # Normalize to get power spectral density
    psd = np.square(spectrum) / (fs * len(time_series))  
    # Double the PSD for one-sided spectrum (except at DC and Nyquist)
    psd[1:] = 2 * psd[1:]
    return freqs, psd

def display_fft(frequencies, spectrum):
    
    
    return 

def plot_frequency_response(b, a, fs, worN=512):
    w, h = freqz(b, a, fs=fs, worN=512)
    plt.plot(w, abs(h))
    return


def plot_sos_frequency_response(sos, fs, worN=512):
    w, h = sosfreqz(sos,worN=worN, whole=True,fs=fs)
    plt.plot(w, abs(h))
    return

def channel_wavelet_transform():
    
    
    
    pass