from wrappers.eeg import EEG
from preprocessing.eeg import preprocess
import numpy as np
import scipy.io
from scipy import signal
import matplotlib.pyplot as plt
from visualization.eeg import inspect_channels



data_folder = "data/omid_2"

eeg = EEG("data/omid_2/TwoHandOmid2025.03.13_13.02.35.hdf5")
eeg.print()

eeg = preprocess(eeg, ica=False)

#scipy.io.savemat('data/omid_2/omid_eeg_channels_trial_1.mat', {'channels': eeg.channel_data})

x = eeg.channel_data[0]
y = eeg.channel_data[31]
#inspect_channels(np.array((x,)), s_freq=eeg.sampling_frequency)
#inspect_channels(np.array((y,)), s_freq=eeg.sampling_frequency)
f, Cxy = signal.coherence(x, y, eeg.sampling_frequency, nperseg=1024)
plt.semilogy(f, Cxy)
plt.xlabel('frequency [Hz]')
plt.ylabel('Coherence')
plt.show()