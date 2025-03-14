import numpy as np
import pywt
import matplotlib.pyplot as plt

def cwt_fnirs(fnirs_signal, sampling_rate, wavelet='morl', min_freq=0.01, max_freq=0.5):
    """
    Performs Continuous Wavelet Transform (CWT) on an fNIRS signal, focusing on frequencies
    between min_freq and max_freq.

    Args:
        fnirs_signal (numpy.ndarray): The fNIRS time series data.
        sampling_rate (float): The sampling rate of the fNIRS signal in Hz.
        wavelet (str): The wavelet to use (e.g., 'morl', 'cmor', 'gaus1').
        min_freq (float): The minimum frequency of interest (Hz).
        max_freq (float): The maximum frequency of interest (Hz).

    Returns:
        tuple: (cwt_coefficients, frequencies, time_axis)
            cwt_coefficients (numpy.ndarray): The CWT coefficients.
            frequencies (numpy.ndarray): The corresponding frequencies.
            time_axis (numpy.ndarray): The time axis.
    """

    dt = 1 / sampling_rate
    frequencies = np.arange(min_freq, max_freq + 0.001, 0.001)  # Adjust step as needed
    frequencies = np.maximum(frequencies, 1e-10) # Ensure no frequency is exactly zero
    scales = pywt.frequency2scale(wavelet, frequencies, dt)

    cwt_coefficients, _ = pywt.cwt(fnirs_signal, scales, wavelet, dt)

    time_axis = np.arange(0, len(fnirs_signal) / sampling_rate, dt)

    return cwt_coefficients, frequencies, time_axis

def plot_cwt_fnirs(cwt_coefficients, frequencies, time_axis, title='CWT of fNIRS'):
    """
    Plots the CWT coefficients for fNIRS.

    Args:
        cwt_coefficients (numpy.ndarray): The CWT coefficients.
        frequencies (numpy.ndarray): The corresponding frequencies.
        time_axis (numpy.ndarray): The time axis for the fNIRS signal.
        title (str): The title of the plot.
    """

    plt.figure(figsize=(12, 6))
    plt.imshow(np.abs(cwt_coefficients), extent=[time_axis[0], time_axis[-1], frequencies[-1], frequencies[0]],
               aspect='auto', cmap='jet')
    plt.colorbar(label='Magnitude')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title(title)
    plt.show()

# Example usage:
if __name__ == "__main__":
    # Generate a synthetic fNIRS signal (replace with your actual fNIRS data)
    sampling_rate = 10  # Hz (typical fNIRS sampling rate)
    duration = 100  # seconds
    time_axis = np.arange(0, duration, 1 / sampling_rate)
    frequency1 = 0.1  # Hz
    frequency2 = 0.25 # Hz
    fnirs_signal = np.sin(2 * np.pi * frequency1 * time_axis) + 0.5 * np.sin(2 * np.pi * frequency2 * time_axis) + np.random.normal(0, 0.1, len(time_axis))

    # Perform CWT
    cwt_coeffs, freqs, time_axis = cwt_fnirs(fnirs_signal, sampling_rate)
    print(freqs)

    # Plot the CWT
    plot_cwt_fnirs(cwt_coeffs, freqs, time_axis)