from datasets.fnirs import read_snirf, find_snirf_in_folder
from preprocessing.fnirs.conversion import light_intensity_to_optical_density, optical_density_to_hemoglobin_concentration
from preprocessing.fnirs.filtering import bandpass_filter_snirf, z_normalize_snirf, spike_removal_snirf
from preprocessing.fnirs.motion_correction import motion_correction, channel_rejection
from visualization.fnirs import plot_snirf, plot_psd_snirf

def preprocess_snirf(snirf, od=True, hb=True, filter=True, cv=False, tddr=False, z_norm=False, spike_removal=False):
    current = snirf.copy()
    
    # Optical density
    if od:
        current = light_intensity_to_optical_density(current)
    
    # Channel rejection  
    if cv:
        current = channel_rejection(current, 0.1) # 10% CV
    
    # Normalization -> Z-Normalization
    if z_norm:
        current = z_normalize_snirf(current)
        pass

    # Spike Removal -> PCA 
    if spike_removal:
        current = spike_removal_snirf(current, n_components=3)

    # TDDR
    if tddr:
        current = motion_correction(current)


    # Bandpass Filtering
    if filter:
        lowcut = 0.01 
        highcut = 0.07
        order = 10
        current = bandpass_filter_snirf(current, lowcut, highcut, order)
        
    # Hemoglobin
    if hb:
        current = optical_density_to_hemoglobin_concentration(current)
    
    return current
