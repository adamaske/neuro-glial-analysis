import numpy as np #
import pandas as pd #
import mne #
import mne_nirs
import snirf
import matplotlib.pyplot as plt

#VALIDATE DATA ->
#folder_path = "balance test final"
#light_intensity = mne.io.read_raw_nirx(folder_path).load_data()


#CONVERT TO SNIRF
#mne_nirs.io.write_raw_snirf(light_intensity, "data/balance.snirf")

#LOAD THE SNIRF
#snirf_intensity = mne.io.read_raw_snirf("data/balance.snirf").load_data()

#VALIDATE THE SNIRF FILE
#result = snirf.validateSnirf("data/balance.snirf")
#assert(result.is_valid())
#result.display()

#CONVERT TO OPTICAL DENSITY
#optical_density = mne.preprocessing.nirs.optical_density(snirf_intensity)
#mne_nirs.io.write_raw_snirf(optical_density, "data/optical_density.snirf")

#VALIDATE THE OPTICAL DENSITY
#result = snirf.validateSnirf("data/optical_density.snirf")
#assert(result.is_valid())
#result.display()

#LOAD OPTICAL DENSITY
#optical_density = mne.io.read_raw_snirf("data/optical_density.snirf")

#CONVERT TO HEMOGLOBIN HbO & HbR
#hemoglobin = mne.preprocessing.nirs.beer_lambert_law(optical_density)
#mne_nirs.io.write_raw_snirf(hemoglobin, "data/hemoglobin.snirf")

#VALIDATE HEMOGLOBIN
#result = snirf.validateSnirf("data/hemoglobin.snirf")
#assert(result.is_valid())
#result.display()

#LOAD HEMOGLOBIN 
#hb = mne.io.read_raw_snirf("data/hemoglobin.snirf")
#hb.plot()
#plt.show()