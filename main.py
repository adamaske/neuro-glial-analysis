import numpy as np
import pywt
import matplotlib.pyplot as plt

# Load or simulate an example fNIRS time series
Fs = 1  # Example: 1 Hz sampling rate (adjust accordingly)
t = np.arange(0, 600, 1/Fs)  # 10-minute signal
signal = np.sin(2 * np.pi * 0.05 * t) + 0.5 * np.sin(2 * np.pi * 0.02 * t)  # Simulated fNIRS signal

# Define the wavelet parameters
wavelet = 'cmor'  # Complex Morlet wavelet

# Compute scales corresponding to the desired frequency range (0.01 to 0.08 Hz)
frequencies = np.linspace(0.01, 0.08, 100)  # Frequencies of interest
scales = pywt.frequency2scale(wavelet, frequencies)

# Compute the CWT
coefficients, freqs = pywt.cwt(signal, scales, wavelet, sampling_period=1/Fs)

# Plot the wavelet power spectrum
plt.figure(figsize=(10, 6))
plt.imshow(np.abs(coefficients), aspect='auto', extent=[t[0], t[-1], frequencies[-1], frequencies[0]], cmap='jet')
plt.colorbar(label='Power')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.title('Wavelet Transform (0.01 - 0.08 Hz)')
plt.show()
