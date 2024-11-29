

from preprocessing.fnirs.filtering import butter_bandpass, butter_bandpass_filter
from analysis.frequencies import compute_fft, compute_psd, plot_sos_frequency_response
import matplotlib.pyplot as plt 
fs = 100
filter_order = 15
f_low = 0.01
f_high = 0.1
sos, w, h = butter_bandpass(f_low, f_high, fs, freqs=2048, order= filter_order)

fig, axs = plt.subplots()
fig.suptitle(f"Bandpass Butterworth {filter_order}th order  \n@ fs={fs} Hz, lowcut={f_low} Hz highcut={f_high}")
sos, w, h = butter_bandpass(f_low, f_high, fs, filter_order)
axs.plot(w, abs(h))

plt.show()