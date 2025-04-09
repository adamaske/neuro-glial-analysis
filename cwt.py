import pywt._extensions
import pywt._extensions._cwt
import pywt._extensions._pywt
from wrappers.fnirs import fNIRS
import pywt
import matplotlib.pyplot as plt
import numpy as np
import random
random.seed(42)
snirf = fNIRS()
snirf.read_snirf("C:/dev/neuro-glial-analysis/data/Subject01/Trial 4 - Pronation/2025-03-24_004.snirf")

snirf.preprocess(normalization=True, tddr=False, bandpass_low=0.005, bandpass_high=0.07, bandpass_order=50)

print("onsets : ", snirf.get_feature_onsets(desc=3))
channel_data = snirf.channel_data

sample_frequency = snirf.sampling_frequency

signal1 = channel_data[int(random.randrange(0, 51))]
signal2 = channel_data[int(random.randrange(0, 51))]
signal3 = channel_data[int(random.randrange(0, 51))]
signal4 = channel_data[int(random.randrange(0, 51))]

wavelets = pywt._extensions._pywt.wavelist()
print(wavelets)
# Desired frequency range (Hz)
f_min = 0.01
f_max = 0.07

# Set the wavelet function and scales
wavelet_function = "cmor    " # You can change this to other wavelets like 'gaus1', 'cmor'
# Calculate the scales corresponding to the desired frequency range
central_frequency = pywt.central_frequency(wavelet_function)  # Get the wavelet's central frequency
scales = central_frequency / (np.arange(f_max, f_min - 0.001, -0.001) * sample_frequency) # Vectorized scale calculatio

# Perform the Continuous Wavelet Transform (CWT) for each signal
cwt1, freqs1 = pywt.cwt(signal1, scales, wavelet_function, sampling_period=1/sample_frequency)
cwt2, freqs2 = pywt.cwt(signal2, scales, wavelet_function, sampling_period=1/sample_frequency)
cwt3, freqs3 = pywt.cwt(signal3, scales, wavelet_function, sampling_period=1/sample_frequency)
cwt4, freqs4 = pywt.cwt(signal4, scales, wavelet_function, sampling_period=1/sample_frequency)

# Calculate the frequencies corresponding to the scales
# Use the already calculated scales
frequencies = central_frequency / (scales * sample_frequency)

# Plot the CWT results for the 4 signals
fig, axs = plt.subplots(2, 2, figsize=(12, 8))  # Create a 2x2 grid of subplots

# Plot CWT of signal 1
axs[0, 0].imshow(np.abs(cwt1), extent=[0, len(signal1)/sample_frequency, frequencies[-1], frequencies[0]], aspect='auto',
                   vmax=abs(cwt1).max(), vmin=-abs(cwt1).max(), cmap='viridis')
axs[0, 0].set_title('Wavelet Transform of Signal 1')
axs[0, 0].set_xlabel('Time (seconds)')
axs[0, 0].set_ylabel('Frequency (Hz)')

# Plot CWT of signal 2
axs[0, 1].imshow(np.abs(cwt2), extent=[0, len(signal2)/sample_frequency, frequencies[-1], frequencies[0]], aspect='auto',
                   vmax=abs(cwt2).max(), vmin=-abs(cwt2).max(), cmap='viridis')
axs[0, 1].set_title('Wavelet Transform of Signal 2')
axs[0, 1].set_xlabel('Time (seconds)')
axs[0, 1].set_ylabel('Frequency (Hz)')

# Plot CWT of signal 3
axs[1, 0].imshow(np.abs(cwt3), extent=[0, len(signal3)/sample_frequency, frequencies[-1], frequencies[0]], aspect='auto',
                   vmax=abs(cwt3).max(), vmin=-abs(cwt3).max(), cmap='viridis')
axs[1, 0].set_title('Wavelet Transform of Signal 3')
axs[1, 0].set_xlabel('Time (seconds)')
axs[1, 0].set_ylabel('Frequency (Hz)')

# Plot CWT of signal 4
axs[1, 1].imshow(np.abs(cwt4), extent=[0, len(signal4)/sample_frequency, frequencies[-1], frequencies[0]], aspect='auto',
                   vmax=abs(cwt4).max(), vmin=-abs(cwt4).max(), cmap='viridis')
axs[1, 1].set_title('Wavelet Transform of Signal 4')
axs[1, 1].set_xlabel('Time (seconds)')
axs[1, 1].set_ylabel('Frequency (Hz)')

plt.tight_layout()
plt.show()
