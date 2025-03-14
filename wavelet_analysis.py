import pywt
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch #for checking frequency content

def continuous_wavelet_transform_plot_eeg(signal, wavelet='morl', scales=None, title='Continuous Wavelet Transform (EEG)'):
    """
    Performs Continuous Wavelet Transform (CWT) on an EEG signal and plots the result.

    Args:
        signal (array-like): The input EEG signal.
        wavelet (str): The wavelet function to use (e.g., 'morl', 'cmor', 'gaus1').
        scales (array-like, optional): The scales to use for the CWT. If None,
                                        scales are automatically generated.
        title (str): Title of the plot.
    """

    if scales is None:
        scales = np.arange(1, len(signal) / 10) #adjust scale range for eeg

    #pywt.scale2frequency(wavelet, scales)
    coefficients, frequencies = pywt.cwt(signal, scales, wavelet)
    
    plt.figure(figsize=(12, 6))
    plt.imshow(np.abs(coefficients), extent=[0, len(signal), scales[-1], scales[0]], aspect='auto',
               cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Magnitude')
    plt.xlabel('Time (samples)')
    plt.ylabel('Scale')
    plt.title(title)
    plt.show()

if __name__ == "__main__":
   # Parameters
    fs = 256  # Sampling frequency (Hz)
    duration = 5  # Signal duration (seconds)
    t = np.arange(0, duration, 1 / fs)  # Time vector

    # Synthetic signal with 10 Hz and 20 Hz components
    f1 = 10  # Frequency 1 (Hz)
    f2 = 20  # Frequency 2 (Hz)
    signal = np.sin(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t)

    #check the frequency content with welch
    freqs, psd = welch(signal, fs=fs, nperseg=256)
    plt.figure()
    plt.plot(freqs, psd)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD")
    plt.title("Frequency content of EEG signal")
    plt.show()

    # Perform and plot the CWT
    continuous_wavelet_transform_plot_eeg(signal)

    # Example with different wavelet and custom scales
    scales = np.arange(1, 256) #adjust for eeg
    continuous_wavelet_transform_plot_eeg(signal, wavelet='cmor', scales=scales, title='CWT with cmor wavelet (EEG)')

    # Example with gaus1 wavelet
    continuous_wavelet_transform_plot_eeg(signal, wavelet='gaus1', title='CWT with gaus1 wavelet (EEG)')