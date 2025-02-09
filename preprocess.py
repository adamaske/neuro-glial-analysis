from datasets.fnirs import read_snirf, find_snirf_in_folder
from preprocessing.fnirs.conversion import light_intensity_to_optical_density, optical_density_to_hemoglobin_concentration
from preprocessing.fnirs.filtering import bandpass_filter_snirf

from visualization.fnirs import plot_snirf, plot_psd_snirf

def preprocess_snirf(snirf):
    
    # Optical density
    od = light_intensity_to_optical_density(snirf.copy())
    
    # Bandpass Filtering
    lowcut = 0.01 
    highcut = 0.07
    order = 20
    filtered = bandpass_filter_snirf(od, lowcut, highcut, order)
    
    # Normalization -> Z-Normalization
    
    # Spike Removal -> PCA 
    
    # Hemoglobin
    hb = optical_density_to_hemoglobin_concentration(filtered)
    
    return hb
