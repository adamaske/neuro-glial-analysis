import numpy as np
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt

def butter_bandpass(lowcut, highcut, fs, order):
    return butter(order, [lowcut, highcut], fs=fs, btype='bandpass')

def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


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
    freqs, spectrum = compute_fft(time_series, freq_limit)
    # Normalize to get power spectral density
    psd = np.square(spectrum) / (fs * len(time_series))  
    # Double the PSD for one-sided spectrum (except at DC and Nyquist)
    psd[1:] = 2 * psd[1:]
    return freqs, psd

def plot_frequency_response(b, a, fs, worN=512):
    w, h = freqz(b, a, fs=fs, worN=512)
    plt.plot(w, abs(h))
    return

fs = 5.0
duration = 400
t = np.linspace(0.0, duration, int(duration * fs))

n_samples = len(t)
freqs = np.fft.fftfreq(n_samples, d=1/fs)


# 1. Generate a 50 Hz signal
signal_frequency = 50
clean_signal = np.sin(2 * np.pi * signal_frequency * t)

# 2. Generate white noise and apply band-pass filter (0 to 5 Hz)
n_samples = len(t)
frequencies = np.fft.fftfreq(n_samples, d=1/fs)

# Generate random complex noise in frequency domain
noise_in_freq = np.random.normal(0, 1, n_samples) + 1j * np.random.normal(0, 1, n_samples)

# Apply band-pass filter (0 to 5 Hz)
freq_mask = np.logical_and(np.abs(frequencies) >= 0.04, np.abs(frequencies) <= 1.4)
noise_in_freq_filtered = noise_in_freq * freq_mask

# Convert the filtered noise back to the time domain
filtered_noise_time = np.fft.ifft(noise_in_freq_filtered)

# 3. Add the noise to the 50 Hz signal
noise_amplitude = 3
noisy_signal = clean_signal + np.real(filtered_noise_time) * noise_amplitude

freqs, sepctrum = compute_fft(noisy_signal, fs, None)

bandpassed_signal = butter_bandpass_filter(noisy_signal, 0.01, 1, fs=fs, order=5)
fig, axs = plt.subplots(1, 2)
axs[0].plot(t, noisy_signal, color ="green")
axs[1].plot(t, bandpassed_signal, color="blue")
    
plt.show()