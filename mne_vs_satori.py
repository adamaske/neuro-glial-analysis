from fnirs_preprocessing.validation import validate_snirf 
from fnirs_preprocessing.conversion import light_intensity_to_hemoglobin_concentration
from fnirs_preprocessing.motion_correction import motion_correction
from fnirs_preprocessing.filtering import filter_snirf
from visualization.plotting import plot_raw_channels
from mne.io import read_raw_snirf
from mne_nirs.io import write_raw_snirf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter,freqz


#Raw file
un_preprocessed_filepath = "C:/dev/neuro-glial-analysis/data/balance-22-11/2024-11-22_003/2024-11-22_003.snirf"

#File preprocssed from by Satori
#satori_filepath = "C:/dev/neuro-glial-analysis/data/balance-22-11/2024-11-22_003/satori-preprocessed/2024-11-22_003_CV10p_SRi10l5t3p5I0p5MON_TDDRHF_THPBW0p01s_TDTS0p4s_ZNORM.snirf"
#satori = read_raw_snirf(satori_filepath).load_data()
#validate_snirf(satori_filepath)

adam_preprocssed_filepath = "C:/dev/neuro-glial-analysis/data/balance-22-11/2024-11-22_003/adam-preprocessed/2024-11-22_003_ADAM-PREPROCSSED.snirf"

raw = read_raw_snirf(un_preprocessed_filepath).load_data()

hm = light_intensity_to_hemoglobin_concentration(raw)
hm = filter_snirf(hm)
fs = hm.info["sfreq"]

hm_data = np.array(hm.get_data())
raw_data = np.array(raw.get_data())

#CASES
s1d1_satori = np.array(pd.read_csv(
    "C:/dev/neuro-glial-analysis\data/balance-22-11/2024-11-22_003/satori-preprocessed\s1d1_satori_cc_001_007_znorm.csv")
                       ).transpose()[0] #0th channel is s1d1 oxy
#s1d1_satori = (s1d1_satori - np.mean(s1d1_satori)) / np.std(s1d1_satori)#z-normalization


s1d1_data = np.array(hm_data[0][:-1]) #adjust for different lengths
s1d1_data = (s1d1_data - np.mean(s1d1_data)) / np.std(s1d1_data)#z-nromailzation

def butter_bandpass(lowcut, highcut, fs, order):
    return butter(order, [lowcut, highcut], fs=fs, btype='bandpass')

def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y
use_order = 200
custom_filtered_data = butter_bandpass_filter(s1d1_data, 0.01, 0.07, hm.info["sfreq"], use_order)


def compute_fft(time_series, freq_limit:int|None):
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

def compute_psd(time_series, freq_limit):
    # Compute FFT
    freqs, spectrum = compute_fft(time_series, freq_limit)
    # Normalize to get power spectral density
    psd = (np.square(spectrum)) / (fs * len(time_series))  
    # Double the PSD for one-sided spectrum (except at DC and Nyquist)
    psd[1:] = 2 * psd[1:]
    return freqs, psd

def display_frequency_response():
    order_file = 300
    # Plot the frequency response for a few different orders.
    fig, axs = plt.subplots()
    fig.suptitle(f"{order_file}th \n@ fs=5 Hz, lowcut=0.01 Hz highcut=0.07")
    b100, a100 = butter_bandpass(lowcut=0.01, highcut=0.07, fs=5, order=order_file)

    w, h = freqz(b100, a100, fs=fs, worN=2000)
    plt.plot(w, abs(h))
    plt.show()
    #w, h = freqz(b50, a50, fs=fs, worN=2000)
    #plt.plot(w, abs(h))
    #w, h = freqz(b100, a100, fs=fs, worN=2000)
    #plt.plot(w, abs(h))
    #fig.legend(["n=5", "n=50", "n=100"])

def display_time_series_custom():
    fig, axs = plt.subplots(1, 2)
    axs[0].plot(custom_filtered_data, color="green")
    axs[0].set_title("S1-D1 HbO (Pinti)")
    axs[1].plot(s1d1_satori, color="red")
    axs[1].set_title("S1-D1 HbO (Satori)")
    fig.legend(["Pinti", "Satori"])
    fig.suptitle(f"Pinti Method BP @ 0.01 - 0.07 Hz n={use_order} Z-Normalized vs Pre-Processed Satori BP @ 0.01 - 0.07 Hz Z-Normalized : Hemoglobin Concentration ")

def display_frequency_spectra_custom(freq_limit):
    mne_freqs, mne_spectrum = compute_fft(custom_filtered_data, freq_limit)
    satori_freqs, satori_spectrum = compute_fft(s1d1_satori, freq_limit)

    fig, axs = plt.subplots(1, 2)
    axs[0].plot(mne_freqs, mne_spectrum, color="green")
    axs[0].set_title("S1-D1 HbO PSD (Pinti)")
    axs[1].plot(satori_freqs,satori_spectrum, color="red")
    axs[1].set_title("S1-D1 HbO PSD (Satori)")

    fig.legend(["Pinti", "Satori"])
    fig.suptitle(f"Power Spectral Density : \nPinti Method BP @ 0.01 - 0.07 Hz n={use_order} Z-Normalized vs Pre-Processed Satori BP @ 0.01 - 0.07 Hz Z-Normalized ")


def display_time_series():
    fig, axs = plt.subplots(1, 2)
    axs[0].plot(s1d1_data, color="green")
    axs[0].set_title("S1-D1 HbO (MNE)")
    axs[1].plot(s1d1_satori, color="red")
    axs[1].set_title("S1-D1 HbO (Satori)")
    fig.legend(["MNE", "Satori"])
    fig.suptitle("Pre-Processed MNE BP @ 0.01 - 0.07 Hz Z-Normalized vs Pre-Processed Satori BP @ 0.01 - 0.07 Hz Z-Normalized : Hemoglobin Concentration ")


def display_frequency_spectra(freq_limit):
    mne_freqs, mne_spectrum = compute_psd(s1d1_data, freq_limit)
    satori_freqs, satori_spectrum = compute_psd(s1d1_satori, freq_limit)

    fig, axs = plt.subplots(1, 2)
    axs[0].plot(mne_freqs, mne_spectrum, color="green")
    axs[0].set_title("S1-D1 HbO PSD (MNE)")
    axs[1].plot(satori_freqs,satori_spectrum, color="red")
    axs[1].set_title("S1-D1 HbO PSD (Satori)")

    fig.legend(["MNE", "Satori"])
    fig.suptitle("Power Spectral Density : \nPre-Processed MNE BP @ 0.01 - 0.07 Hz Z-Normalized vs Pre-Processed Satori BP @ 0.01 - 0.07 Hz Z-Normalized ")

#display_time_series_custom()
#display_frequency_spectra_custom(0.2)
#display_frequency_response()
display_time_series()
display_frequency_spectra()
plt.show()
exit()
#To OD & Hb
#Motion correction
mc = motion_correction(hm)
#Filter frequencies
filtered = filter(mc)

filtered.ch_names

adam_data = np.array(filtered.get_data())


satori_data = np.array(satori.get_data())

raw_data = np.array(raw.get_data())
#Compare them

print("adam shape : ", adam_data.shape)
print("satori shape : ", satori_data.shape)
print("raw shape : ", raw_data.shape)

exit()

raw = light_intensity_to_hemoglobin_concentration(read_raw_snirf(un_preprocessed_filepath).load_data()) #no filter no mc
mc = motion_correction(raw) #no filter w mc
filtered = filter(raw) # filtered no mc
filtered_mc = filter(motion_correction(raw)) #filtered mc

fs = raw.info["sfreq"]

no_filter_no_mc =  np.array(raw.get_data())
no_filter_mc = np.array(mc.get_data())
filter_no_mc =  np.array(filtered.get_data())
filter_mc = np.array(filtered_mc.get_data())

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