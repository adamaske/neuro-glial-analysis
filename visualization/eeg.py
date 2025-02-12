import numpy as np
import matplotlib.pyplot as plt

from analysis.frequencies import compute_psd
from preprocessing.eeg import compute_band_power, band_power_colors, band_ranges_spec

def inspect_channels(data, s_freq):
    fig, axs = plt.subplots(1, 3)
        
    fig.suptitle(f"{len(data)} channels")
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("Voltage (uV)")
    axs[0].set_title("Time Series")
    
    axs[1].set_xlabel("Frequency (Hz)")
    axs[1].set_ylabel("Magnitude")
    axs[1].set_title("Power Spectral Density")
    
    total_band_power = np.array(((0, 0, 0, 0, 0)), dtype=float)
    for channel in data:
        spectra, freqs = compute_psd(channel, s_freq, int(s_freq/2))
        band_power = compute_band_power(spectra, freqs)
        total_band_power += list(band_power.values())
        
        axs[0].plot(channel)
        axs[1].plot(spectra, freqs)
    
    avg_band_power = total_band_power / len(data) 
    
    axs[2].bar(list(band_ranges_spec.keys()), list(avg_band_power), color=band_power_colors)
    axs[2].set_xlabel("Frequency Bands")
    axs[2].set_ylabel("Power (μV²/Hz)")
    axs[2].set_title("Band Powers")
    plt.show()
    
def inspect_channel_by_channel(data, s_freq):
    
    h_freq = int(s_freq/2)
    for idx in range(len(data)):
        channel = data[idx] # Get channel data

        fig, axs = plt.subplots(1, 3)
        fig.suptitle(f"Channel {idx + 1}")
        axs[0].plot(channel)
        axs[0].set_xlabel("Time")
        axs[0].set_ylabel("Voltage (uV)")
        axs[0].set_title("Time Series")
        
        spectra, freqs = compute_psd(channel, s_freq, h_freq)
        axs[1].plot(spectra, freqs)
        axs[1].set_xlabel("Frequency (Hz)")
        axs[1].set_ylabel("Magnitude")
        axs[1].set_title("Power Spectral Density")
        
        spectra, freqs = compute_psd(channel, s_freq, h_freq)
        band_powers = compute_band_power(spectra, freqs)
        axs[2].bar(list(band_powers.keys()), list(band_powers.values()), color=band_power_colors)
        axs[2].set_xlabel("Frequency Bands")
        axs[2].set_ylabel("Power (μV²/Hz)")
        axs[2].set_title("Band Powers")
        #axs[2].set_xticks(0)  # Rotate labels for readability
        axs[2].grid(axis='y', linestyle='--', alpha=0.6)
        plt.show()
        