
import numpy as np
import os
import matplotlib.pyplot as plt
from preprocessing.fnirs.conversion import light_intensity_to_hemoglobin_concentration
from mne.io import read_raw_snirf
from preprocessing.fnirs.filtering import butter_bandpass_filter
from analysis.fnirs.frequencies import compute_fft, compute_psd, plot_sos_frequency_response
from mne_nirs.io.snirf import write_raw_snirf
# Omid
omid_roi_channels_right = ["S14_D12", "S7_D6", "S14_D6", "S7_D15", "S10_D6", "S14_D15", "S10_D12"]
omid_roi_channels_left  = ["S12_D12", "S10_D2", "S3_D11", "S12_D2", "S10_D12", "S3_D2", "S12_D11"]

# Daniel
daniel_roi_channels_right = ["S10_D12", "S10_D10", "S12_D12", "S14_D14", "S12_D10"]
daniel_roi_channels_left  = ["S8_D6", "S4_D6", "S6_D6", "S8_D8", "S6_D8"]

#omid_folder = "data/OMID-13-12-024"
#omid_filepaths = [#"2024-12-13_001/2024-12-13_001.snirf", #both feet on ground
#                  "2024-12-13_002/2024-12-13_002.snirf", #one feet off ground
#                  #"2024-12-13_003/2024-12-13_003.snirf", #easy mat
#                  "2024-12-13_004/2024-12-13_004.snirf", #hard mat
#                ]
#daniel_folder = "data/DANIEL-28-01-2025"
#daniel_filepaths = ["2025-01-28_001/2025-01-28_001.snirf",
#                    "2025-01-28_002/2025-01-28_002.snirf"
#                    ]
#
#omid_snirfs = []
#for file in omid_filepaths:
#    snirf = read_raw_snirf(os.path.join(omid_folder, file))
#    
#    # CROP TO ANNOTATIONS
#    duration = len(snirf.get_data()[0]) / 5.1
#    start = snirf.annotations[0]["onset"]
#    last = snirf.annotations[len(snirf.annotations)-1]
#    end = last["onset"] + last["duration"]
#    end = np.clip(end, 0, duration) # NOTE : The "END" annotation has a 10 sec duration, however sometimes we stop the reocrding right away thus an error occurs
#    
#    omid_snirfs.append(snirf.crop(tmin=start, tmax=end))
#    
#daniel_snirfs = []
#for file in daniel_filepaths:
#    snirf = read_raw_snirf(os.path.join(daniel_folder, file))
#    
#    # CROP TO ANNOTATIONS
#    duration = len(snirf.get_data()[0]) / 5.1
#    start = snirf.annotations[0]["onset"]
#    last = snirf.annotations[len(snirf.annotations)-1]
#    end = last["onset"] + last["duration"]
#    end = np.clip(end, 0, duration)
#    
#    daniel_snirfs.append(snirf.crop(tmin=start, tmax=end))
#    
#omid_1   =  light_intensity_to_hemoglobin_concentration(omid_snirfs[0]) #Feet on ground
#omid_2   =  light_intensity_to_hemoglobin_concentration(omid_snirfs[1]) #Difficult mat
#daniel_1 =  light_intensity_to_hemoglobin_concentration(daniel_snirfs[0]) #Hands
#daniel_2 =  light_intensity_to_hemoglobin_concentration(daniel_snirfs[1]) #Hands
#
#write_raw_snirf(omid_1  , "data/trsd/omid_1_hb.snirf")
#write_raw_snirf(omid_2  , "data/trsd/omid_2_hb.snirf")
#write_raw_snirf(daniel_1, "data/trsd/daniel_1_hb.snirf")
#write_raw_snirf(daniel_2, "data/trsd/daniel_2_hb.snirf")

omid_1   = read_raw_snirf("data/trsd/omid_1_hb.snirf")
omid_2   = read_raw_snirf("data/trsd/omid_2_hb.snirf")


daniel_1 = read_raw_snirf("data/trsd/daniel_1_hb.snirf")
daniel_2 = read_raw_snirf("data/trsd/daniel_2_hb.snirf")

def get_indices_from_channels(wanted_channels, ch_names):
    hbo = []
    hbr = []
    for idx in range(len(ch_names)):
        ch_name = ch_names[idx].split(" ")[0]
        wl = ch_names[idx].split(" ")[1]
        for ch in wanted_channels:
            if ch == ch_name: # It this correct channel ? 
                assert(wl == "hbo" or wl  == "hbr")
                if wl == "hbo":
                    hbo.append(idx)
                if wl == "hbr":
                    hbr.append(idx)
    return hbo, hbr

omid_left_hemisphere_hbo_indices, omid_left_hemisphere_hbr_indices = get_indices_from_channels(omid_roi_channels_left, omid_1.info["ch_names"])
omid_right_hemisphere_hbo_indices, omid_right_hemisphere_hbr_indices = get_indices_from_channels(omid_roi_channels_right, omid_1.info["ch_names"])
daniel_left_hemisphere_hbo_indices, daniel_left_hemisphere_hbr_indices = get_indices_from_channels(daniel_roi_channels_left, daniel_1.info["ch_names"])
daniel_right_hemisphere_hbo_indices, daniel_right_hemisphere_hbr_indices = get_indices_from_channels(daniel_roi_channels_right, daniel_1.info["ch_names"])

print(" [ HbO ] : ")
print("Daniel : ", daniel_left_hemisphere_hbo_indices, ", ", daniel_right_hemisphere_hbo_indices)
print("Omid   : ",   omid_left_hemisphere_hbo_indices, ", ",   omid_right_hemisphere_hbo_indices)

print(" [ HbR ] : ")
print("Daniel : ", daniel_left_hemisphere_hbr_indices, ", ", daniel_right_hemisphere_hbr_indices)
print("Omid   : ",   omid_left_hemisphere_hbr_indices, ", ",   omid_right_hemisphere_hbr_indices)

# We now have our indices -> How do we get these channels

def get_channels_by_indices(indices, data):
    channels = []
    for idx in indices:
        channels.append(np.divide(data[idx], np.max(data[idx])))
    return channels

# HBO
daniel_left_hbo_channels = get_channels_by_indices(daniel_left_hemisphere_hbo_indices, daniel_1.get_data())
daniel_right_hbo_channels = get_channels_by_indices(daniel_right_hemisphere_hbo_indices, daniel_1.get_data())

omid_left_hbo_channels = get_channels_by_indices(omid_left_hemisphere_hbo_indices, omid_1.get_data())
omid_right_hbo_channels = get_channels_by_indices(omid_right_hemisphere_hbo_indices, omid_1.get_data())

# HBR
daniel_left_hbr_channels = get_channels_by_indices(daniel_left_hemisphere_hbr_indices, daniel_1.get_data())
daniel_right_hbr_channels = get_channels_by_indices(daniel_right_hemisphere_hbr_indices, daniel_1.get_data())

omid_left_hbr_channels = get_channels_by_indices(omid_left_hemisphere_hbr_indices, omid_1.get_data())
omid_right_hbr_channels = get_channels_by_indices(omid_right_hemisphere_hbr_indices, omid_1.get_data())

print(f"Daniel HbO -> Left:{len(daniel_left_hbo_channels)}, Right:{len(daniel_right_hbo_channels)}")
print(f"Daniel HbR -> Left:{len(daniel_left_hbr_channels)}, Right:{len(daniel_right_hbr_channels)}")
print(f"Omid   HbO -> Left:{len(omid_left_hbo_channels)}, Right:{len(omid_right_hbo_channels)}")
print(f"Omid   HbR -> Left:{len(omid_left_hbr_channels)}, Right:{len(omid_right_hbr_channels)}")


#Remove the "right foot" from Omid time series
start_frames = []
end_frames = []
for idx in range(len(omid_1.annotations.description)):
    desc = omid_1.annotations.description[idx]
    if desc != "1":
        continue
    # We found the right foot
    onset = omid_1.annotations.description[idx] # When does right foot start
    duration = omid_1.annotations.description[idx] # How long does it last

    start_frame = float(onset) * float(omid_1.info["sfreq"])
    end_frame = start_frame + (float(duration) * omid_1.info["sfreq"])# + (float(omid_1.annotations.description[idx + 1]) * float(omid_1.info["sfreq"]))

    start_frames.append(start_frame)
    end_frames.append(end_frame)

def remove_right_foot(channels):
    new_channels = []
    for ch in channels:
        new_channel = ch
        for start, end in zip(start_frames, end_frames):
            indices_to_remove = np.arange(int(start), int(end))
            new_channel = ch[int(start):int(end)]
        new_channels.append(new_channel)
    return new_channels
        
#omid_left_hbo_channels = remove_right_foot(omid_left_hbo_channels)
#omid_right_hbo_channels = remove_right_foot(omid_right_hbo_channels)
#omid_left_hbr_channels = remove_right_foot(omid_left_hbr_channels)
#omid_right_hbr_channels = remove_right_foot(omid_right_hbr_channels)

#
fs = daniel_1.info["sfreq"]
filter_order = 10
f_low = 0.01
f_high = 0.14

def filter_channel(channels):
    filtered_channels = []
    for ch in channels:
        filtered = butter_bandpass_filter(ch, f_low, f_high, fs, filter_order)
        normalized = np.divide(filtered, np.max(filtered))
        filtered_channels.append(normalized)
    return filtered_channels

# HbO
filtered_daniel_left_hbo_channels = filter_channel(daniel_left_hbo_channels)
filtered_daniel_right_hbo_channels = filter_channel(daniel_right_hbo_channels)

filtered_omid_left_hbo_channels = filter_channel(omid_left_hbo_channels)
filtered_omid_right_hbo_channels = filter_channel(omid_right_hbo_channels)

# HbR
filtered_daniel_left_hbr_channels = filter_channel(daniel_left_hbr_channels)
filtered_daniel_right_hbr_channels = filter_channel(daniel_right_hbr_channels)

filtered_omid_left_hbr_channels = filter_channel(omid_left_hbr_channels)
filtered_omid_right_hbr_channels = filter_channel(omid_right_hbr_channels)

def display_hemisphere_channels(channels, filtered_chs, title):
    channel_amount = len(channels) #

    fig, axs = plt.subplots(channel_amount, 4)
    fig.suptitle(title)
    
    col_titles = ['Original Signal', 'Original PSD', 'Fileterd Signal', 'Filtered PSD']
    for col in range(4):
        axs[0, col].set_title(col_titles[col], fontsize=12, fontweight='bold')
    for row in range(channel_amount):
        axs[row, 0].set_ylabel("Channel " + str(row+1), fontsize=12, fontweight='bold', rotation=90, labelpad=15)
        #axs[row, 3].set_ylim(-0.01, 0.01)

    for idx in range(channel_amount):
        channel = np.array(channels[idx])
        
        axs[idx][0].plot(channel) # raw time series
        freqs, spectra = compute_fft(time_series=channel, 
                                     fs=5.1, 
                                     freq_limit=0.2)
        axs[idx][1].plot(freqs, spectra) # psd

        axs[idx][2].plot(filtered_chs[idx])

        filtered_freqs, filtered_spectra = compute_fft(time_series=filtered_chs[idx], 
                                     fs=5.1, 
                                     freq_limit=0.2)
        axs[idx][3].plot(filtered_freqs, filtered_spectra)
        


display_hemisphere_channels(daniel_left_hbo_channels, 
                            filtered_daniel_left_hbo_channels,
                            f"Daniel Lateral Sensory Cortex : Left Hemisphere\nHbO @ 5.1 Hz BP Filtered @ {f_low}-{f_high} Hz n={filter_order}")

display_hemisphere_channels(daniel_right_hbo_channels, 
                            filtered_daniel_right_hbo_channels,
                            f"Daniel Lateral Sensory Cortex : Right Hemisphere\nHbO @ 5.1 Hz BP Filtered @ {f_low}-{f_high} Hz n={filter_order}")

display_hemisphere_channels(omid_left_hbo_channels, 
                            filtered_omid_left_hbo_channels,
                            f"Omid Medial Sensory Cortex : Left Hemisphere\nHbO @ 5.1 Hz BP Filtered @ {f_low}-{f_high} Hz n={filter_order}")
display_hemisphere_channels(omid_right_hbo_channels, 
                            filtered_omid_right_hbo_channels,
                            f"Omid Medial Sensory Cortex : Right Hemisphere\nHbO @ 5.1 Hz BP Filtered @ {f_low}-{f_high} Hz n={filter_order}")

def task_related_spectral_density_of_time_series(time_series, stimulation_frequency, bandwidth, resolution, ):
    
    # Get psd of time series
    
    #filtered = butter_bandpass_filter(time_series, f_low, f_high, fs, filter_order)
    
    freqs, spectra = compute_fft(time_series=time_series, fs=5.1, freq_limit=0.2)
    lowcut = stimulation_frequency - (bandwidth/2.0)
    highcut = stimulation_frequency + (bandwidth/2.0)

    closest_low_index = np.argmin(np.abs(freqs - lowcut))
    closest_high_index = np.argmin(np.abs(freqs - highcut))
    
    #print(f"lowcut:{lowcut}")
    #print(f"closest_low_index: {closest_low_index}, Closest Value: {freqs[closest_low_index]}")
    #print(f"highcut:{highcut}")
    #print(f"closest_high_index: {closest_high_index}, Closest Value: {freqs[closest_high_index]}")
    
    d_x = (highcut - lowcut) / resolution #delta X
    
    tr_spectrum = spectra[closest_low_index:closest_high_index]
    return np.sum(np.array(tr_spectrum))

def average_trsd_of_channels(channels, stim_freq, w, res):
    trsd_sum = 0
    for ch in channels:
        trsd_sum += task_related_spectral_density_of_time_series(ch, stim_freq, w, res)
    
    avg_trsd = trsd_sum / len(channels)
    return avg_trsd
    
daniel_S_f = 0.04
omid_S_f = 0.0333

bandwidths = [0.02, 0.04]
resolutions = [1000]

# omid
for w in bandwidths:
    for n in resolutions:
        
        omid_left_trsd = average_trsd_of_channels(filtered_omid_left_hbo_channels, omid_S_f, w, n)
        omid_right_trsd = average_trsd_of_channels(filtered_omid_right_hbo_channels, omid_S_f, w, n)
        
        
        print(f"\n\nOmid Left TRSD @ S_f:{omid_S_f}, Bandwidth:{w}, Resolution:{ n} --> {omid_left_trsd}")
        print(f"Omid Right TRSD @ S_f:{omid_S_f}, Bandwidth:{w}, Resolution:{ n} --> {omid_right_trsd}")
        omid_li = (omid_left_trsd - omid_right_trsd) / (omid_left_trsd + omid_right_trsd)
        print(f"Omid TRSD Laterality @ S_f:{omid_S_f}, Bandwidth:{w}, Resolution:{ n} --> {omid_li}")
        
        
        daniel_left_trsd = average_trsd_of_channels(filtered_daniel_left_hbo_channels, omid_S_f, w, n)
        daniel_right_trsd = average_trsd_of_channels(filtered_daniel_right_hbo_channels, omid_S_f, w, n)
        print(f"\nDaniel Left TRSD @ S_f:{omid_S_f}, Bandwidth:{w}, Resolution:{ n} --> {daniel_left_trsd}")
        print(f"Daniel Right TRSD @ S_f:{omid_S_f}, Bandwidth:{w}, Resolution:{ n} --> {daniel_right_trsd}")
        daniel_li = (daniel_left_trsd - daniel_right_trsd) / (daniel_left_trsd + daniel_right_trsd)
        print(f"Daniel TRSD Laterality @ S_f:{omid_S_f}, Bandwidth:{w}, Resolution:{ n} --> {daniel_li}")
        
plt.show()
