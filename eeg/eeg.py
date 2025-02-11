import numpy as np
import matplotlib.pyplot as plt

from preprocessing.filter import butter_bandpass_filter

def bandpass_filter_eeg(data):



    return|

def common_average_reference_filter(data):
    average_reference = np.mean(data, axis=0)
    filtered = data - average_reference

    return filtered
