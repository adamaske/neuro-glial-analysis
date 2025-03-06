import numpy as np
import matplotlib.pyplot as plt
from mne.io import read_raw_snirf
from mne.preprocessing.nirs import optical_density, beer_lambert_law,  temporal_derivative_distribution_repair
from mne.io.snirf._snirf import RawSNIRF
from preprocessing.filter import butter_bandpass_filter, notch_filter

from sklearn.decomposition import PCA
# Wrappper for mne.io.snirf file
fnirs_types = {0:"WL", 1:"OD", 2:"CC"}



class fNIRS:
    def __init__(self, snirf_filepath:str=""):

        if snirf_filepath != "":
            self.read(snirf_filepath)
    
    def read(self, filepath:str):
        
        self.filepath = filepath
        self.snirf = read_raw_snirf(self.filepath)
        self.s_freq = self.snirf.info["sfreq"]
        return self
    
    def write(self, filepath:str):

        if filepath == self.filepath:
            print("Overwriting files is not allowed!")
            
        pass
    
    def print(self):
        print(f"fNIRS : {self.filepath}")
        info = self.snirf.info
        print(info)
        
        pass
    
    def get_data(self):
        return self.snirf.get_data()
    
    def inspect(self):
        
        data = self.snirf.get_data()
        
        fig, axs = plt.subplots(2, 2)
        fig.suptitle(f" fNIRS : {self.filepath}")
        
        
        axs[0, 0] # HbO Time series
        axs[0, 1] # HbO Frequency Domain
        axs[1, 0] # HbR Time series
        axs[1, 1] # HbR Frequency Domain        
        pass
    
    def preprocess(self):
        
        # Optical Density Conversion
        od = optical_density(self.snirf)
        
        od_data = od.get_data()
        
        od_filtered = np.zeros_like(od_data)
        # Bandpass filter
        for idx, channel_data in enumerate(od_data):
            od_filtered[idx] = butter_bandpass_filter(channel_data, 0.01, 0.07, od.info["sfreq"], 200)
        
        # Normalize
        
        mean_vals = np.mean(od_filtered, axis=1, keepdims=True)  # Compute mean per channel
        std_vals = np.std(od_filtered, axis=1, keepdims=True)  # Compute standard deviation per channel
        # Avoid division by zero
        std_vals[std_vals == 0] = 1

        od_normalized = (od_filtered - mean_vals) / std_vals  # Apply Z-Normalization

        # Spike Removal# Apply PCA
        n_components = od_normalized.shape[0]
        pca = PCA()
        transformed = pca.fit_transform(od_normalized.T)  
        transformed[:, n_components:] = 0 # Zero out nosie components
        od_spike_removed = pca.inverse_transform(transformed).T 
    
        
        # Motion Correction
        snirf_copy = od.copy()
        snirf_copy._data = od_spike_removed
        od_motion_corrected = temporal_derivative_distribution_repair(snirf_copy)
        
        cc = beer_lambert_law(od_motion_corrected)
        
        self.snirf = cc
        pass


    