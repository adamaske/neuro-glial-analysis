
import numpy as np
import matplotlib.pyplot as plt


from enum import Enum

from mne.io import read_raw_snirf
from mne_nirs.io import write_raw_snirf
from snirf import validateSnirf
from mne.preprocessing.nirs import optical_density, beer_lambert_law,  temporal_derivative_distribution_repair

from scipy.signal import butter, lfilter
from scipy.signal import butter, freqz, sosfreqz, sosfilt, sosfiltfilt, iirnotch, filtfilt

def butter_bandpass(lowcut, highcut, fs, freqs=512, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    w, h = sosfreqz(sos, worN=None, whole=True, fs=fs)
    return sos, w, h

def butter_bandpass_filter(time_series, lowcut, highcut, fs, order):
    sos, w, h = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfiltfilt(sos, time_series)
    return np.array(y)

def notch_filter(data, sfreq, freqs=[50, 60]):
    """Apply notch filters at specified frequencies."""
    for freq in freqs:
        b, a = iirnotch(freq, 30, sfreq)
        data = filtfilt(b, a, data, axis=-1)
    return data

class fnirs_data_type(Enum):
    Wavelength = "Wavelength"
    OpticalDensity = "Optical Density"
    HemoglobinConcentration = "Hemoglobin Concentration"

WL = fnirs_data_type.Wavelength
OD = fnirs_data_type.OpticalDensity
CC = fnirs_data_type.HemoglobinConcentration

class fNIRS():
    def __init__(self): 
        self.type = WL
        self.snirf = None 
        
        self.sampling_frequency = None
        self.channel_names = None
        self.channel_data = None
        self.channel_num = None
        
        self.feature_onsets = None
        self.feature_descriptions = None
    
    def print(self):
        print("sampling_frequency : ", self.sampling_frequency, " Hz")
        print("channel_num : ", self.channel_num)
        print("channel_data : ", self.channel_data.shape)
        print("channel_names : ", self.channel_names)
        print("feature_onsets : ", self.feature_onsets)
        print("feature_descriptions : ", self.feature_descriptions)
        
    def read_snirf(self, filepath):
        print(f"Reading SNIRF from {filepath}")
        result = validateSnirf(filepath)
        print("valid : ", result.is_valid())
        self.snirf = read_raw_snirf(filepath)
        
        # fNIRS info
        info = self.snirf.info
        self.sampling_frequency = float(info["sfreq"])
        self.channel_names = info["ch_names"]
        self.channel_data = np.array(self.snirf.get_data())
        self.channel_num = int(info["nchan"])
        # Features
        annotations = self.snirf._annotations
        self.feature_onsets = np.array(annotations.onset, dtype=float)
        self.feature_descriptions = np.array(annotations.description, dtype=int)
                
        
    def write_snirf(self, filepath):
        write_raw_snirf(self.snirf, filepath)
        print(f"Wrote SNIRF to {filepath}")
        result = validateSnirf(filepath)
        print("valid : ", result.is_valid())
        
        
    def wl_to_od(self):
        if self.type != WL:
            print(f"sNIRF type is {self.type}, cannot convert to {OD}!")
            return
        
        
        self.snirf._data = self.channel_data.astype(np.float32)
        self.snirf = optical_density(self.snirf)
        
        self.channel_data = self.snirf.get_data()
        self.type = OD

    def od_to_hb(self):
        if self.type != OD:
            print(f"sNIRF type is {self.type}, cannot convert to {CC}!")
            return 
        
        # refresh snirf
        self.snirf._data = self.channel_data.astype(np.float32)
        self.snirf = beer_lambert_law(self.snirf)
        
        self.channel_data = self.snirf.get_data()
        self.type = CC

    def get_channel_dict(self):

        self.channel_dict = {}
        for i, channel_name in enumerate(self.channel_names):
            
            source_detector = channel_name.split()[0]
            wavelength = channel_name.split()[1]

            if source_detector not in self.channel_dict:
                self.channel_dict[source_detector] = {"HbO" : None, 
                                                 "HbR" : None
                                                 }
            
            channel_data = self.channel_data[i] 
            
            if wavelength == "HbR".lower() or wavelength == "760":
                self.channel_dict[source_detector]["HbR"] = channel_data
                
            if wavelength == "HbO".lower() or wavelength == "850":
                self.channel_dict[source_detector]["HbO"] = channel_data
        return self.channel_dict

    def trim_from_features(self, cut_before_first_feature, cut_after_last_feature):


        pass

    def bandpass_filter(self, f_low, f_high, order):
        filtered_channels = np.zeros_like(self.channel_data)  # Ensures matching shape and type

        for i, channel in enumerate(self.channel_data):  # Iterate over each channel
            filtered_channels[i] = butter_bandpass_filter(channel, f_low, f_high, self.sampling_frequency, order)

        self.channel_data = filtered_channels

    def normalize(self,):
        mean = np.mean(self.channel_data, axis=0, keepdims=True)
        std = np.std(self.channel_data, axis=0, keepdims=True)

        # Handle cases where std is zero (constant channels)
        std[std == 0] = 1  # Avoid division by zero

        self.channel_data = (self.channel_data - mean) / std

    
    def tddr(self):

        pass
    def preprocess(self, normalization, tddr, bandpass_low, bandpass_high, bandpass_order):

        if self.type == WL:
            self.wl_to_od()

        if tddr:
            self.tddr()
        self.bandpass_filter(bandpass_low, bandpass_high, bandpass_order)

        if normalization:
            self.normalize()

        if self.type == OD:
            self.od_to_hb()

    def get_hbo_hbr_channels(self):
        dict = self.get_channel_dict()
        hbo = []
        hbr = []
        for i, channel in enumerate(dict):
            hbo.append(self.channel_dict[channel]["HbO"])
            hbr.append(self.channel_dict[channel]["HbR"])
        return np.array(hbo), np.array(hbr)
    
    def feature_epochs(self, description, tmin, tmax):
        
        epochs = []
        for i, onset in enumerate(self.feature_onsets):
            if description != self.feature_descriptions[i]: 
                continue
            
            start = onset + tmin
            end = onset + tmax
            start_frame = int(start * self.sampling_frequency)
            end_frame = int(end * self.sampling_frequency)

            if start_frame < 0 or end_frame >= self.channel_data.shape[1]:
                print(f"Epoch of out range (0, {self.channel_data.shape[1]}) : {start_frame} - {end_frame}")
                continue

            epoch_data = self.channel_data[:, start_frame:end_frame]
            epochs.append(epoch_data)
        
        return epochs

    def get_feature_onsets(self, desc):
        onsets = []
        for i, onset in enumerate(self.feature_onsets):
            if desc != self.feature_descriptions[i]: 
                continue

            onsets.append(onset)

        return onsets
    def inspect_snirf(self):

        hbo_data, hbr_data = self.get_hbo_hbr_channels()

        duration = int(self.channel_data.shape[1] / self.sampling_frequency)
        t = np.linspace(0, duration, self.channel_data.shape[1])

        plt.subplot(2, 2, 1)
        for channel in hbo_data:
            plt.plot(t, channel)
        plt.title("HbO")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude (qM)")
        
        plt.subplot(2, 2, 3)
        for channel in hbr_data:
            plt.plot(t, channel)
        plt.title("HbR")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude (qM)")
        plt.show()


#snirf = fNIRS()
#snirf.read_snirf("C:/dev/neuro-glial-analysis/data/Subject01/Trial 3 - Supination/2025-03-24_003.snirf")
#
#snirf.preprocess(normalization=False, tddr=False, bandpass_low=0.01, bandpass_high=0.1, bandpass_order=10)
#
#snirf.inspect_snirf()
#epochs = snirf.feature_epochs(4, 0, 10)
#
#duration = np.abs(0) + 10
#
#for i, epoch in enumerate(epochs):
#    print("epoch : ", epoch.shape)
#
#    plt.title(f"Epoch {i+1}")
#    t = np.linspace(0, duration, epoch.shape[1])
#    for channel in epoch:
#        plt.plot(t, channel)
#    
#    plt.show()
#exit()
#snirf.wl_to_od()
#snirf.od_to_hb()
#snirf.write_snirf("C:/dev/test_snirf.snirf")