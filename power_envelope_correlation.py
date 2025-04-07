import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.stats import pearsonr

from datasets.fnirs import read_snirf
from wrappers.eeg import EEG
from preprocessing.eeg import preprocess
from preprocessing.fnirs import preprocess_snirf

def power_envelope_correlation(signal1, signal2, plot=False):
    """
    Calculates the Pearson correlation of the power envelopes of two signals,
    and optionally plots the signals and their envelopes.

    Args:
        signal1 (array-like): The first signal.
        signal2 (array-like): The second signal.
        plot (bool, optional): Whether to plot the signals and envelopes. Defaults to False.

    Returns:
        float: The Pearson correlation coefficient.
        float: The p-value of the Pearson correlation.
    """

    analytic_signal1 = hilbert(signal1)
    analytic_signal2 = hilbert(signal2)

    power_envelope1 = np.abs(analytic_signal1)
    power_envelope2 = np.abs(analytic_signal2)

    correlation, p_value = pearsonr(power_envelope1, power_envelope2)


    print(f"Pearson correlation coefficient: {correlation}")
    print(f"P-value: {p_value}")

    if plot:
        time = np.arange(len(signal1))  # Assuming signals have the same length

        plt.figure(figsize=(12, 6))

        plt.subplot(2, 1, 1)
        plt.plot(time, signal1, label='Signal 1')
        plt.plot(time, power_envelope1, label='Envelope 1', linestyle='--')
        plt.title('Signal 1 and Envelope')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(time, signal2, label='Signal 2')
        plt.plot(time, power_envelope2, label='Envelope 2', linestyle='--')
        plt.title('Signal 2 and Envelope')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        plt.legend()

        plt.tight_layout()
        plt.show()

    return correlation, p_value

eeg_files = ["data/Subject01/Trial 1 - Supination/heel2025.03.24_14.27.28.hdf5",
                "data/Subject01/Trial 2 - Pronation/heel2025.03.24_14.31.33.hdf5",
                "data/Subject01/Trial 3 - Supination/heel2025.03.24_14.36.01.hdf5",
                "data/Subject01/Trial 4 - Pronation/heel2025.03.24_14.40.18.hdf5",
                "data/Subject01/Trial 5 - Supination/heel2025.03.24_14.45.30.hdf5",
                "data/Subject01/Trial 6 - Pronation/heel2025.03.24_14.50.12.hdf5",
                "data/Subject02/Trial 1/HeelSubject22025.03.27_11.14.27.hdf5",
                "data/Subject02/Trial 2/HeelSubject22025.03.27_11.17.47.hdf5",
                "data/Subject02/Trial 3/HeelSubject22025.03.27_11.21.29.hdf5",
                "data/Subject02/Trial 4/HeelSubject22025.03.27_11.25.31.hdf5",
                "data/Subject02/Trial 5/HeelSubject22025.03.27_11.29.06.hdf5",
                "data/Subject02/Trial 6/HeelSubject22025.03.27_11.32.54.hdf5",
                ]

#eegs = [EEG(path) for path in eeg_files]

eeg = EEG(eeg_files[0])
snirf = read_snirf("data/Subject01/Trial 1 - Supination/2025-03-24_001.snirf")

# Example usage:
np.random.seed(42)
signal1 = np.sin(np.linspace(0, 10 * np.pi, 1000)) + 0.1 * np.random.randn(1000)
signal2 = np.sin(np.linspace(0, 10 * np.pi + np.pi / 4, 1000)) + 0.1 * np.random.randn(1000)

correlation, p_value = power_envelope_correlation(signal1, signal2, plot=True) #plot=True

print(f"Pearson correlation coefficient: {correlation}")
print(f"P-value: {p_value}")

signal3 = np.sin(np.linspace(0, 10 * np.pi, 1000)) + 0.1 * np.random.randn(1000)
signal4 = signal3 + 0.05 * np.random.randn(1000)

correlation2, p_value2 = power_envelope_correlation(signal3, signal4, plot=True)

print(f"Pearson correlation coefficient (nearly identical signals): {correlation2}")
print(f"P-value (nearly identical signals): {p_value2}")

signal5 = np.sin(np.linspace(0, 10 * np.pi, 1000)) + 0.1 * np.random.randn(1000)
signal6 = np.cos(np.linspace(0, 10 * np.pi, 1000)) + 0.1 * np.random.randn(1000)

correlation3, p_value3 = power_envelope_correlation(signal5, signal6, plot=True)

print(f"Pearson correlation coefficient (uncorrelated signals): {correlation3}")
print(f"P-value (uncorrelated signals): {p_value3}")