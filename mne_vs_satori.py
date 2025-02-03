from preprocessing.fnirs.validation import validate_snirf 
from preprocessing.fnirs.conversion import light_intensity_to_hemoglobin_concentration
from preprocessing.fnirs.motion_correction import motion_correction
from visualization.plotting import plot_raw_channels
from preprocessing.fnirs.filtering import butter_bandpass, butter_bandpass_filter
from analysis.fnirs.frequencies import compute_fft, compute_psd, plot_sos_frequency_response
from mne.io import read_raw_snirf
from mne_nirs.io import write_raw_snirf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#Raw file
un_preprocessed_filepath = "C:/dev/neuro-glial-analysis/data/balance-22-11/2024-11-22_003/2024-11-22_003.snirf"

#File preprocssed from by Satori
satori_filepath = "C:/dev/neuro-glial-analysis/data/balance-22-11/2024-11-22_003/satori-preprocessed/2024-11-22_003_CV10p_SRi10l5t3p5I0p5MON_TDDRHF_THPBW0p01s_TDTS0p4s_ZNORM.snirf"
#satori = read_raw_snirf(satori_filepath)
#validate_snirf(satori_filepath)

adam_preprocssed_filepath = "C:/dev/neuro-glial-analysis/data/balance-22-11/2024-11-22_003/adam-preprocessed/2024-11-22_003_ADAM-PREPROCSSED.snirf"

raw = read_raw_snirf(un_preprocessed_filepath).load_data()
#raw_data = np.array(raw.get_data())
fs = raw.info["sfreq"]

hb = light_intensity_to_hemoglobin_concentration(raw)
#hb = filter_snirf(hb)
hb_data = np.array(hb.get_data())

#Satori Preprocessed S1-D1
s1d1_satori = np.array(pd.read_csv(
    "C:/dev/neuro-glial-analysis\data/balance-22-11/2024-11-22_003/satori-preprocessed\s1d1_satori_cc_001_007_znorm.csv")
                       ).transpose()[0] #0th channel is s1d1 oxy
#Custom Preprocessing
s1d1_data = np.array(hb_data[0][:-1]) 

# Z-Normalization
#s1d1_data = (s1d1_data - np.mean(s1d1_data)) / np.std(s1d1_data)


fs = hb.info["sfreq"]
filter_order = 30
f_low = 0.01
f_high = 0.14
#CUSTOM BW BP FILTER
custom_filtered_data = butter_bandpass_filter(s1d1_data, f_low, f_high, fs, filter_order)
   
def display_frequency_response():
    # Plot the frequency response for a few different orders.
    fig, axs = plt.subplots()
    fig.suptitle(f"Bandpass Butterworth {filter_order}th order  \n@ fs={fs} Hz, lowcut={f_low} Hz highcut={f_high}")
    fig.supxlabel('Frequency (Hz)')
    fig.supylabel('Gain (dB)')
    sos, w, h = butter_bandpass(f_low, f_high, fs, filter_order)
    axs.plot(w, abs(h))
    

def display_time_series_custom():
    fig, axs = plt.subplots(1, 2)
    axs[0].plot(custom_filtered_data, color="green")
    axs[0].set_title("S1-D1 HbO (Adam)")
    axs[1].plot(s1d1_satori, color="red")
    axs[1].set_title("S1-D1 HbO (Satori)")
    fig.supxlabel('Time (seconds)')
    fig.supylabel('Amplitude (Hb)')
    fig.legend(["Adam", "Satori"])
    fig.suptitle(f"Hemoglobin Concentration : \nAdam Custom BP @ {f_low} - {f_high} Hz n={filter_order} Z-Normalized vs Pre-Processed Satori BP @ 0.01 - 0.07 Hz Z-Normalized")

def display_frequency_spectra_custom(freq_limit):
    adam_freqs, adam_spectrum = compute_psd(custom_filtered_data, fs, freq_limit=freq_limit)
    satori_freqs, satori_spectrum = compute_psd(s1d1_satori, fs, freq_limit=freq_limit)

    fig, axs = plt.subplots(1, 2)
    axs[0].plot(adam_freqs, adam_spectrum, color="green")
    axs[0].set_title("S1-D1 HbO PSD (Adam)")
    axs[1].plot(satori_freqs,satori_spectrum, color="red")
    axs[1].set_title("S1-D1 HbO PSD (Satori)")
    fig.supxlabel('Frequency (Hz)')
    fig.supylabel('Gain (dB)')
    fig.legend(["Adam", "Satori"])
    fig.suptitle(f"Power Spectral Density : \nAdam Custom BP @ {f_low} - {f_high} Hz n={filter_order} Z-Normalized vs Pre-Processed Satori BP @ 0.01 - 0.07 Hz Z-Normalized ")


def display_time_series():
    fig, axs = plt.subplots(1, 2)
    axs[0].plot(s1d1_data, color="green")
    axs[0].set_title("S1-D1 HbO (MNE)")
    axs[1].plot(s1d1_satori, color="red")
    axs[1].set_title("S1-D1 HbO (Satori)")
    fig.legend(["MNE", "Satori"])
    fig.suptitle("Hemoglobin Concentration : \nPre-Processed MNE BP @ 0.01 - 0.07 Hz Z-Normalized vs Pre-Processed Satori BP @ 0.01 - 0.07 Hz Z-Normalized")

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

display_time_series_custom()
display_frequency_spectra_custom(0.2)
display_frequency_response()
#display_time_series()
#display_frequency_spectra()
plt.show()
exit()