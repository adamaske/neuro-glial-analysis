from preprocessing.fnirs.validation import validate_snirf 
from preprocessing.fnirs.conversion import light_intensity_to_hemoglobin_concentration
from preprocessing.fnirs.motion_correction import motion_correction
from preprocessing.fnirs.filtering import filter
from visualization.plotting import plot_raw_channels
from mne.io import read_raw_snirf
from mne_nirs.io import write_raw_snirf
import matplotlib.pyplot as plt
import numpy as np


filepath = "C:/dev/neuro-glial-analysis/data/balance-15-11/2024-11-15_005/2024-11-15_005.snirf"

raw = light_intensity_to_hemoglobin_concentration(read_raw_snirf(filepath).load_data()) #no filter no mc
mc = motion_correction(raw) #no filter w mc
filtered = filter(raw) # filtered no mc
filtered_mc = filter(motion_correction(raw)) #filtered mc

fs = raw.info["sfreq"]

no_filter_no_mc =  np.array(raw.get_data())
no_filter_mc = np.array(mc.get_data())
filter_no_mc =  np.array(filtered.get_data())
filter_mc = np.array(filtered_mc.get_data())

def compute_fft(time_series):
    # Compute FFT
    N = len(time_series)  # Length of the signal
    fft_result = np.fft.fft(time_series)
    fft_freq = np.fft.fftfreq(N, d=1/fs)  # Frequency axis

    # Take the positive half of the spectrum
    positive_freqs = fft_freq[:N // 2]
    positive_spectrum = np.abs(fft_result[:N // 2]) * (2 / N)  # Normalize for one-sided
    
    # Compute Power Spectral Density (PSD)
    fft_magnitude = np.abs(fft_result[:N // 2])  # Magnitude of FFT for positive frequencies
    psd = (positive_spectrum**2) / (fs * N)  # Normalize to get power spectral density

    
    # Double the PSD for one-sided spectrum (except at DC and Nyquist)
    psd[1:] = 2 * psd[1:]
    
    # Filter frequencies to only include up to 5 Hz
    freq_limit = 1.5  # Hz
    indices = positive_freqs <= freq_limit
    limited_freqs = positive_freqs[indices]
    limited_spectrum = psd[indices]
    return limited_freqs, limited_spectrum

#unfiltered
nf_nmc_s, nf_nmc_f =  compute_fft(no_filter_no_mc[0])
nf_mc_s, nf_mc_f = compute_fft(no_filter_mc[0])
#filtered
f_nmc_s, f_nmc_f=  compute_fft(filter_no_mc[0])
f_mc_s, f_mc_f =  compute_fft(filter_mc[0])


fig, axs = plt.subplots(2, 2)

axs[0][0].plot(nf_nmc_s, nf_nmc_f)
axs[0][1].plot(nf_mc_s, nf_mc_f)
axs[1][0].plot(f_nmc_s, f_nmc_f)
axs[1][1].plot(f_mc_s, f_mc_f)


axs[0][0].legend(["No Filter No Motion Correction"])
axs[0][1].legend(["No Filter W/ Motion Correctoin"])
axs[1][0].legend(["Filtered No Motion Correctoin"])
axs[1][1].legend(["Filtered W/ Motion Correctoin"])

new_filepath = "C:/dev/neuro-glial-analysis/data/balance-15-11/2024-11-15_005"
write_raw_snirf(raw, "C:/dev/neuro-glial-analysis/data/balance-15-11/2024-11-15_005/no_filter_no_motion_correction")
write_raw_snirf(mc, "C:/dev/neuro-glial-analysis/data/balance-15-11/2024-11-15_005/no_filter_motion_correction")
write_raw_snirf(filtered, "C:/dev/neuro-glial-analysis/data/balance-15-11/2024-11-15_005/filtered_no_motion_correction")
write_raw_snirf(filtered_mc, "C:/dev/neuro-glial-analysis/data/balance-15-11/2024-11-15_005/filtered_motion_correction")
plt.show()