import scipy.signal as signal
import matplotlib.pyplot as plt
import numpy as np
from wrappers.fnirs import fNIRS
import random
random.seed(42)
snirf = fNIRS()
snirf.read_snirf("C:/dev/neuro-glial-analysis/data/Subject01/Trial 4 - Pronation/2025-03-24_004.snirf")

snirf.preprocess(normalization=True, tddr=False, bandpass_low=0.02, bandpass_high=0.07, bandpass_order=10)

print("onsets : ", snirf.get_feature_onsets(desc=4))
channel_data = snirf.channel_data

sample_frequency = snirf.sampling_frequency

# Use the signals provided by the user
signal1 = channel_data[0][150:600]
signal2 = channel_data[12][150:600]
signal3 = channel_data[15][150:600]
signal4 = channel_data[32][150:600]

# Sampling frequency (Hz)
sampling_frequency = snirf.sampling_frequency

# Define the window and overlap parameters for STFT
window = 'hann'  # The type of window function to use (e.g., 'hann', 'hamming', 'blackman')
nperseg = 512  # Increased number of data points per segment for higher frequency resolution
noverlap = 256 # Increased overlap for higher time resolution (typically 75% overlap)

# Perform the Short-Time Fourier Transform (STFT) for each signal
f1, t1, Zxx1 = signal.stft(signal1, fs=sampling_frequency, window=window, nperseg=nperseg, noverlap=noverlap)
f2, t2, Zxx2 = signal.stft(signal2, fs=sampling_frequency, window=window, nperseg=nperseg, noverlap=noverlap)
f3, t3, Zxx3 = signal.stft(signal3, fs=sampling_frequency, window=window, nperseg=nperseg, noverlap=noverlap)
f4, t4, Zxx4 = signal.stft(signal4, fs=sampling_frequency, window=window, nperseg=nperseg, noverlap=noverlap)

# Plot the STFT results for the 4 signals
fig, axs = plt.subplots(2, 2, figsize=(12, 8))  # Create a 2x2 grid of subplots

# Plot STFT of signal 1
axs[0, 0].pcolormesh(t1, f1, np.abs(Zxx1), cmap='viridis')
axs[0, 0].set_title('STFT of Signal 1')
axs[0, 0].set_xlabel('Time (seconds)')
axs[0, 0].set_ylabel('Frequency (Hz)')
axs[0, 0].set_ylim(0, 0.1)  # Limit the frequency axis

# Plot STFT of signal 2
axs[0, 1].pcolormesh(t2, f2, np.abs(Zxx2), cmap='viridis')
axs[0, 1].set_title('STFT of Signal 2')
axs[0, 1].set_xlabel('Time (seconds)')
axs[0, 1].set_ylabel('Frequency (Hz)')
axs[0, 1].set_ylim(0, 0.1)  # Limit the frequency axis

# Plot STFT of signal 3
axs[1, 0].pcolormesh(t3, f3, np.abs(Zxx3), cmap='viridis')
axs[1, 0].set_title('STFT of Signal 3')
axs[1, 0].set_xlabel('Time (seconds)')
axs[1, 0].set_ylabel('Frequency (Hz)')
axs[1, 0].set_ylim(0, 0.1)  # Limit the frequency axis

# Plot STFT of signal 4
axs[1, 1].pcolormesh(t4, f4, np.abs(Zxx4), cmap='viridis')
axs[1, 1].set_title('STFT of Signal 4')
axs[1, 1].set_xlabel('Time (seconds)')
axs[1, 1].set_ylabel('Frequency (Hz)')
axs[1, 1].set_ylim(0, 0.1)  # Limit the frequency axis

plt.tight_layout()
plt.show()
