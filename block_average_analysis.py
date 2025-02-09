import numpy as np
import matplotlib.pyplot as plt

# Example data: Simulated time series with multiple channels
np.random.seed(42)
n_channels = 3
n_timepoints = 10000
sampling_rate = 1000  # Hz
time_series = np.random.randn(n_channels, n_timepoints)  # Simulated data

# Example stimulus times (in seconds)
stimulus_times = np.array([1, 3, 5, 7, 9])  # Stimulus at 1s, 3s, etc.
stimulus_indices = (stimulus_times * sampling_rate).astype(int)  # Convert to index

# Define epoch window (pre- and post-stimulus duration in seconds)
pre_stimulus = 0.5  # 500 ms before stimulus
post_stimulus = 1.0  # 1000 ms after stimulus
epoch_length = int((pre_stimulus + post_stimulus) * sampling_rate)

# Extract epochs
epochs = []
for idx in stimulus_indices:
    start = idx - int(pre_stimulus * sampling_rate)
    end = idx + int(post_stimulus * sampling_rate)
    if start >= 0 and end < n_timepoints:  # Ensure within bounds
        epochs.append(time_series[:, start:end])

epochs = np.array(epochs)  # Shape: (n_trials, n_channels, epoch_length)

# Compute the block average across trials
block_avg = np.mean(epochs, axis=0)  # Shape: (n_channels, epoch_length)

# Create time vector for plotting
time_vector = np.linspace(-pre_stimulus, post_stimulus, epoch_length)

# Plot results
plt.figure(figsize=(10, 5))
for ch in range(n_channels):
    plt.plot(time_vector, block_avg[ch], label=f'Channel {ch+1}')
plt.axvline(0, color='k', linestyle='--', label='Stimulus onset')
plt.xlabel('Time (s)')
plt.ylabel('Signal Amplitude')
plt.title('Block-Averaged Response')
plt.legend()
plt.show()
