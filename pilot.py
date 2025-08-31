from wrappers.eeg import EEG
filepath = "data/Pilot2025.05.19_11.10.56.hdf5"

eeg = EEG(filepath)

from preprocessing.eeg import preprocess  

eeg = preprocess(eeg, bandpass=True, normalization=True, ica=False, rerefernce=True)

from visualization.eeg import inspect_channels

inspect_channels(eeg.channel_data, eeg.sampling_frequency)
eeg.trim_from_features(0, 0)
from scipy.signal import stft
import matplotlib.pyplot as plt
import numpy as np

# Assuming you have eeg.channel_data and eeg.sampling_frequency from your loaded EEG object

# Example for one channel, you'd likely loop through all channels or select specific ones
channel_index = 9  # Let's say you want to analyze the first channel
signal = eeg.channel_data[channel_index, :]
fs = eeg.sampling_frequency

# STFT parameters
nperseg = int(0.5 * fs)  # Window size (e.g., 0.5 seconds)
noverlap = int(0.25 * fs) # Overlap (e.g., 0.25 seconds)

f, t, Zxx = stft(signal, fs, nperseg=nperseg, noverlap=noverlap)

channel_names = [f"Ch {i}" for i in range(len(eeg.channel_data))]
# Plot the spectrogram
plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=np.max(np.abs(Zxx)))
plt.title(f'STFT Magnitude for Channel {channel_names[channel_index]}')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.colorbar(label='Magnitude')
plt.show()
