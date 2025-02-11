import numpy as np
import matplotlib.pyplot as plt

def common_average_reference_filter(data):
    average_reference = np.mean(data, axis=0)
    filtered = data - average_reference

    return filtered
