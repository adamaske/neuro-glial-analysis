import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

def instantaneous_phase_frequency_plot(time_series_A, sampling_rate, title="Instantaneous Phase, Frequency, and Signal"):
    """
    Calculates and plots the instantaneous phase and frequency of a time series,
    along with a phase wrapping degree plot and the original signal.

    Args:
        time_series_A: NumPy array representing the time series.
        sampling_rate: Sampling rate of the time series (Hz).
        title: Title of the plot.
    """

    analytic_signal = hilbert(time_series_A)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi) * sampling_rate

    time = np.arange(len(time_series_A)) / sampling_rate

    plt.figure(figsize=(12, 10)) #increased figure size

    # Plot Original Signal
    plt.subplot(3, 1, 1)
    plt.plot(time, time_series_A)
    plt.title(f"{title} - Original Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    # Plot Instantaneous Phase
    plt.subplot(3, 1, 2)
    plt.plot(time, instantaneous_phase % (np.pi * 2))
    plt.title(f"{title} - Instantaneous Phase")
    plt.xlabel("Time (s)")
    plt.ylabel("Phase (radians)")

    # Plot Instantaneous Frequency
    plt.subplot(3, 1, 3)
    plt.plot(time[1:], instantaneous_frequency)
    plt.title(f"{title} - Instantaneous Frequency")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")

    plt.tight_layout()
    plt.show()

# Example Usage:
# Generate a sample time series
sampling_rate = 100  # Hz
time = np.arange(0, 10, 1/sampling_rate)
frequency = 5 # Hz
time_series_A = np.sin(2 * np.pi * frequency * time) + np.random.randn(len(time)) * 0.1

# Plot instantaneous phase and frequency
instantaneous_phase_frequency_plot(time_series_A, sampling_rate)