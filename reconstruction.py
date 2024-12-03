#testing reconstruction

import os
import numpy as np
from preprocessing.fnirs.conversion import light_intensity_to_hemoglobin_concentration

from preprocessing.fnirs.filtering import butter_bandpass_filter
from mne.io import read_raw_snirf, RawArray
from mne_nirs.io import write_raw_snirf
from mne.preprocessing.nirs import optical_density, beer_lambert_law

from mne.io.snirf._snirf import RawSNIRF
from mne import create_info
import matplotlib.pyplot as plt

from snirf import Snirf
new_path = os.path.join("data", "balance-15-11", "2024-11-15_001", "2024-11-15_001_hb.snirf")
snirf = Snirf(r"data\balance-22-11\2024-11-22_003\satori-preprocessed\2024-11-22_003_CV10p_SRi10l5t3p5I0p5MON_TDDRHF_THPBW0p01s_TDTS0p4s_ZNORM.snirf", 'r+')
snirf.save(r"data\balance-22-11\2024-11-22_003\satori-preprocessed\2024-11-22_003_resaved.snirf")

satori = read_raw_snirf("data/balance-22-11/2024-11-22_003/satori-preprocessed/2024-11-22_003_resaved.snirf")
satori.plot()
plt.show()
exit()
def copy_snirf_info(snirf): #
    info = snirf.info
    montage = snirf.get_montage()
    data = snirf.get_data()

    return snirf.info

def create_new_snirf():


    return
#original_path = os.path.join("data", "balance-15-11", "2024-11-15_001", "2024-11-15_001.snirf")
#original = read_raw_snirf(original_path)
#hb = light_intensity_to_hemoglobin_concentration(original)
new_path = os.path.join("data", "balance-15-11", "2024-11-15_001", "2024-11-15_001_hb.snirf")
hb = read_raw_snirf(new_path)
info = hb.info
ch_names = info["ch_names"] 
#ch_types = info["ch_types"]
sfreq = info["sfreq"]
montage = hb.get_montage()
data = np.array(hb.get_data())
#filter data
for i in range(len(data)):
    filtered = butter_bandpass_filter(data[i], 0.01, 0.08, sfreq, 10)
    data[i] = filtered

copy = RawArray(data=data, info=info, verbose=True)
copy.set_montage(montage)

filtered_path = os.path.join("data", "balance-15-11", "2024-11-15_001", "2024-11-15_001_hb_filtered.snirf")
write_raw_snirf(copy, filtered_path)
exit()

new_path = os.path.join("data", "balance-15-11", "2024-11-15_001", "2024-11-15_001_hb.snirf")
hb = read_raw_snirf(new_path)
filtered_path = os.path.join("data", "balance-15-11", "2024-11-15_001", "2024-11-15_001_hb_filtered.snirf")
filtered = read_raw_snirf(filtered_path)

#numpy comparison
original_data = np.array(hb.get_data())
filtered_data = np.array(copy.get_data())

fig, axs = plt.subplots(1,2)
axs[0].plot(original_data[0])
axs[1].plot(filtered_data[0])
plt.show()
exit()
print("Original Info : ")
print(original.info)


