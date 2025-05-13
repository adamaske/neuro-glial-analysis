import numpy as np
import matplotlib.pyplot as plt
from neuropipeline.fnirs import fNIRS
from functional import composite_correlation
from functional import merge_matrices_diagonally, threshold_matrix, plot_r_matrix
import threading
from data import get_fnirs_data, get_eeg_data
from preprocessing.eeg import band_ranges_spec
import pandas as pd

fnirs_data = get_fnirs_data("C:/Users/Adam/Desktop/Heel Stimulation/data", load=False)
#eeg_data = get_eeg_data("C:/Users/Adam/Desktop/Heel Stimulation/data")


ROIs = {
    'Both Hemispheres': {
        'S1' : ['S5_D5', 'S5_D6', 'S5_D7', 'S6_D5', 'S6_D7', 'S7_D7', 'S7_D14', 'S9_D6', 'S9_D14' 'S9_D12', 'S9_D14', 'S13_D12', 'S13_D13', 'S13_D15', 'S14_D14', 'S14_D15', 'S15_D13', 'S15_D15'],
        'S2' : ['S7_D8', 'S8_D7', 'S8_D8','S14_D16', 'S16_D16', 'S16_D15'],
        'M1' : ['S3_D3', 'S3_D5', 'S4_D3', 'S4_D4', 'S4_D6', 'S5_D3', 'S9_D4', 'S11_D4', 'S11_D11', 'S11_D12', 'S12_D11', 'S12_D13', 'S13_D11'],
        'PMA': ['S1_D1', 'S1_D2', 'S1_D3', 'S3_D1', 'S3_D3', 'S4_D3','S10_D9', 'S10_D10', 'S10_D11', 'S11_D11', 'S12_D10', 'S12_D11'],
        'SMA': ['S2_D2', 'S2_D4', 'S4_D2', 'S4_D4', 'S2_D9', 'S11_D4', 'S11_D9']
    },
    'Left Hemisphere': {
        'S1' : ['S5_D5', 'S5_D6', 'S5_D7', 'S6_D5', 'S6_D7', 'S7_D7', 'S7_D14', 'S9_D6', 'S9_D14'],
        'S2' : ['S7_D8', 'S8_D7', 'S8_D8'],
        'M1' : ['S3_D3', 'S3_D5', 'S4_D3', 'S4_D4', 'S4_D6', 'S5_D3', 'S9_D4'],
        'PMA': ['S1_D1', 'S1_D2', 'S1_D3', 'S3_D1', 'S3_D3', 'S4_D3'],
        'SMA': ['S2_D2', 'S2_D4', 'S4_D2', 'S4_D4']
    },
    'Right Hemisphere': {
        'S1' : ['S9_D12', 'S9_D14', 'S13_D12', 'S13_D13', 'S13_D15', 'S14_D14', 'S14_D15', 'S15_D13', 'S15_D15'],
        'S2' : ['S14_D16', 'S16_D16', 'S16_D15'],
        'M1' : ['S9_D4', 'S11_D4', 'S11_D11', 'S11_D12', 'S12_D11', 'S12_D13', 'S13_D11'],
        'PMA': ['S10_D9', 'S10_D10', 'S10_D11', 'S11_D11', 'S12_D10', 'S12_D11'],
        'SMA': ['S2_D4', 'S2_D9', 'S11_D4', 'S11_D9']
    }
}
supination_trials = [0, 2, 4]
pronation_trials = [1, 3, 5]

markers = {3 : "Pronation", 
           4 : "Supination"}

def process_fnirs(i, j, hbo_data, hbr_data, ch_names, sampling_frequency):
        hbo_data, ch_names, hbr_data, _ = fnirs_data[subject][trial].split()
        hbo_comp = composite_correlation(hbo_data, ch_names, sampling_frequency, 150, 0.01, 0.1, False)
        hbr_comp = composite_correlation(hbr_data, ch_names, sampling_frequency, 150, 0.01, 0.1, False)
        r_comp = merge_matrices_diagonally(hbo_comp, hbr_comp)

        np.save(f"C:/Users/Adam/Desktop/Heel Stimulation/connectivity/fNIRS/fnirs_sub{i+1}_trial{j+1}_rcomp", r_comp)
threads = []

hbo_data, ch_names, hbr_data, _ = fNIRS("C:/Users/Adam/Desktop/Heel Stimulation/data/Subject01/Trial 1/sub01_trial01.snirf").split()
del hbo_data
del hbr_data
del _

r_supination = []
r_pronation = []

for i, subject in enumerate(fnirs_data.keys()):
    
    for j, trial in enumerate(fnirs_data[subject]):

        if j in pronation_trials:
            trial_type = "Pronation"
        elif j in supination_trials:
            trial_type = "Supination"
        
        #hbo_data, ch_names, hbr_data, _ = fnirs_data[subject][trial].split()
        
        # Load data
        r_comp = np.load(f"C:/Users/Adam/Desktop/Heel Stimulation/connectivity/fNIRS/fnirs_sub{i+1}_trial{j+1}_rcomp.npy")
        
        
        if j in pronation_trials:
            r_pronation.append(r_comp)
        elif j in supination_trials:
            r_supination.append(r_comp)
            
        #plot_r_matrix(r_comp, ch_names, title=f"Subject {i+1} - Trial {j+1} : {trial_type}")
        #plt.savefig(f"C:/Users/Adam/Desktop/Heel Stimulation/connectivity/fNIRS/fnirs_sub{i+1}_trial{j+1}")
        #plt.close()
        #thead = threading.Thread(target=process_fnirs,
        #                         args=(i, j, hbo_data, hbr_data, ch_names, fnirs_data[subject][trial].sampling_frequency))
        #threads.append(thead)
        #thead.start()
    
    #for thread in threads:
    #    thread.join()
    #threads = []

# Both Hemispheres
BH_s1_indices =  [ i for i, ch in enumerate(ch_names) if ch in ROIs['Both Hemispheres']['S1']]
BH_s2_indices =  [ i for i, ch in enumerate(ch_names) if ch in ROIs['Both Hemispheres']['S2']]
BH_m1_indices =  [ i for i, ch in enumerate(ch_names) if ch in ROIs['Both Hemispheres']['M1']]
BH_pma_indices = [ i for i, ch in enumerate(ch_names) if ch in ROIs['Both Hemispheres']['PMA']]
BH_sma_indices = [ i for i, ch in enumerate(ch_names) if ch in ROIs['Both Hemispheres']['SMA']]

# Left Hemisphere
LH_s1_indices =  [ i for i, ch in enumerate(ch_names) if ch in ROIs['Left Hemisphere']['S1']]
LH_s2_indices =  [ i for i, ch in enumerate(ch_names) if ch in ROIs['Left Hemisphere']['S2']]
LH_m1_indices =  [ i for i, ch in enumerate(ch_names) if ch in ROIs['Left Hemisphere']['M1']]
LH_pma_indices = [ i for i, ch in enumerate(ch_names) if ch in ROIs['Left Hemisphere']['PMA']]
LH_sma_indices = [ i for i, ch in enumerate(ch_names) if ch in ROIs['Left Hemisphere']['SMA']]

# Right Hemispphere
RH_s1_indices =  [ i for i, ch in enumerate(ch_names) if ch in ROIs['Right Hemisphere']['S1']]
RH_s2_indices =  [ i for i, ch in enumerate(ch_names) if ch in ROIs['Right Hemisphere']['S2']]
RH_m1_indices =  [ i for i, ch in enumerate(ch_names) if ch in ROIs['Right Hemisphere']['M1']]
RH_pma_indices = [ i for i, ch in enumerate(ch_names) if ch in ROIs['Right Hemisphere']['PMA']]
RH_sma_indices = [ i for i, ch in enumerate(ch_names) if ch in ROIs['Right Hemisphere']['SMA']]

for i, (s, p) in enumerate(zip(r_supination, r_pronation)):
    pass
    # Supination
    # Primary Sensory Cortex
#S1 = s[BH_s1_indices][:, BH_s1_indices]
#print("S1 Supination : ", S1.shape)
#
s = np.mean(r_supination, axis=0)
p = np.mean(r_pronation, axis=0)
plt.subplot(1, 2, 1)
plot_r_matrix(s, ch_names, "Average Supination R")
plt.subplot(1, 2, 2)
plot_r_matrix(p, ch_names, "Average Pronation R")
plt.show()
diff = s - p
plt.subplot(1, 1,1)
plot_r_matrix(diff, ch_names, "Difference R", -0.2, 0.2)
plt.show()

#diff = s - p
#plt.subplot(1, 3, 3)
#plot_r_matrix(d, ROIs['Both Hemispheres']['S1'], "S1")
plt.show()


exit()
        #hbo_comp = composite_correlation(hbo_data, ch_names, fnirs_data[subject][trial].sampling_frequency, 150, 0.01, 0.1, False)
        #hbr_comp = composite_correlation(hbr_data, ch_names, fnirs_data[subject][trial].sampling_frequency, 150, 0.01, 0.1, False)
        #r_comp = merge_matrices_diagonally(hbo_comp, hbr_comp)
#
        #np.save(f"C:/Users/Adam/Desktop/Heel Stimulation/connectivity/fNIRS/fnirs_sub{i+1}_trial{j+1}_rcomp", r_comp)
        #
        ##plot_r_matrix(r_comp, ch_names, title=f"Subject {i+1} - Trial {j+1} : {trial_type}")
        ##plt.savefig(f"C:/Users/Adam/Desktop/Heel Stimulation/connectivity/fNIRS/fnirs_sub{i+1}_trial{j+1}")
        
def process_eeg_trial(subject_index, trial_index, trial_data, ch_names, sfreq, trial_type):

    r_comp = composite_correlation(trial_data, ch_names, sfreq, None, 8, 30, False)
    np.save(f"C:/Users/Adam/Desktop/Heel Stimulation/connectivity/eeg/eeg_sub{subject_index+1}_trial{trial_index+1}_rcomp", r_comp)
# EEG Connectivity
#r_pronation = []
#r_supination = []
#threads = []
#for i, subject in enumerate(eeg_data.keys()):
#    
#    for j, trial in enumerate(eeg_data[subject]):
#
#        if j in pronation_trials:
#            trial_type = "Pronation"
#        elif j in supination_trials:
#            trial_type = "Supination"
#        
#        eeg = eeg_data[subject][trial]
#        data = eeg.channel_data
#        ch_names = [f"Ch {k+1}" for k in range(data.shape[0])]
#        
#        r_comp = np.load(f"C:/Users/Adam/Desktop/Heel Stimulation/connectivity/eeg/eeg_sub{i+1}_trial{j+1}_rcomp.npy")
#        #plt.figure()
#        #plot_r_matrix(r_comp, ch_names, f"Subject {i+1} - Trial {j+1} : {trial_type}")
#        #plt.savefig(f"C:/Users/Adam/Desktop/Heel Stimulation/connectivity/eeg/eeg_sub{i+1}_trial{j+1}")
#        #plt.close()
#        
#        if j in pronation_trials:
#            r_pronation.append(r_comp)
#        if j in supination_trials:
#            r_supination.append(r_comp)
#           
#        
#        #thread = threading.Thread(target=process_eeg_trial,
#        #                          args=(i, j, data, ch_names, eeg.sampling_frequency, trial_type))
#        #threads.append(thread)
#        #thread.start()
#    
#    #for thread in threads:
#    #    thread.join()
#        
#    if i == 1:
#        break
#avg_supination_r = np.mean(r_supination, axis=0)
#avg_pronation_r = np.mean(r_pronation, axis=0)
#
#for i, (s, p) in enumerate(zip(r_supination, r_pronation)):
#    plt.title(f"Trial {i+1}\n")
#    plt.subplot(1, 3, 1)
#    plot_r_matrix(s, ch_names, "Supination")
#    plt.subplot(1, 3, 2)
#    plot_r_matrix(p, ch_names, "Pronation")
#    diff = s - p
#    plt.subplot(1, 3, 3)
#    plot_r_matrix(diff, ch_names, "Difference")
#    plt.show()
#
#df_supination = pd.DataFrame(avg_supination_r, index=ch_names, columns=ch_names)
#df_pronation = pd.DataFrame(avg_pronation_r, index=ch_names, columns=ch_names)
#
#
#excel_filename = "average_connectivity.xlsx"
#writer = pd.ExcelWriter(excel_filename, engine='openpyxl')
#
## Save each DataFrame to a separate sheet in the Excel file
#df_supination.to_excel(writer, sheet_name='Average Supination R')
#df_pronation.to_excel(writer, sheet_name='Average Pronation R')
#
## Save the Excel file
#writer.close()
#
#plt.subplot(1, 2, 1)
#plot_r_matrix(avg_supination_r, ch_names, "Average Supination R")
#plt.subplot(1, 2, 2)
#plot_r_matrix(avg_pronation_r, ch_names, "Average Pronation R")
#plt.show()
#
#print("Supination : ")
#print(avg_supination_r)
#
#print("Pronation : ")
#print(avg_pronation_r)
#
#exit()
#diff = avg_pronation_r -  avg_supination_r
#plt.subplot(1, 3, 3)
#plot_r_matrix(diff, ch_names, "Difference R")
#
#
#exit()

if True:
    fnirs = fNIRS("C:/Users/Adam/Desktop/Heel Stimulation/Subject01/Trial 1 - Supination/2025-03-24_001.snirf")
    hbo_data, ch_names, hbr_data, _ = fnirs.split()

    del hbo_data
    del hbr_data
    del _
    
    # Load matrices 
    subject_01_supination_r_mean = np.load("C:/Users/Adam/Desktop/Heel Stimulation/connectivity/subject_01_supination_r_mean.npy")
    subject_01_pronation_r_mean  = np.load("C:/Users/Adam/Desktop/Heel Stimulation/connectivity/subject_01_pronation_r_mean.npy")
    subject_02_supination_r_mean = np.load("C:/Users/Adam/Desktop/Heel Stimulation/connectivity/subject_02_supination_r_mean.npy")
    subject_02_pronation_r_mean  = np.load("C:/Users/Adam/Desktop/Heel Stimulation/connectivity/subject_02_pronation_r_mean.npy")
    subject_03_supination_r_mean = np.load("C:/Users/Adam/Desktop/Heel Stimulation/connectivity/subject_03_supination_r_mean.npy")
    subject_03_pronation_r_mean  = np.load("C:/Users/Adam/Desktop/Heel Stimulation/connectivity/subject_03_pronation_r_mean.npy")
    subject_04_supination_r_mean = np.load("C:/Users/Adam/Desktop/Heel Stimulation/connectivity/subject_04_supination_r_mean.npy")
    subject_04_pronation_r_mean  = np.load("C:/Users/Adam/Desktop/Heel Stimulation/connectivity/subject_04_pronation_r_mean.npy")
    subject_05_supination_r_mean = np.load("C:/Users/Adam/Desktop/Heel Stimulation/connectivity/subject_05_supination_r_mean.npy")
    subject_05_pronation_r_mean  = np.load("C:/Users/Adam/Desktop/Heel Stimulation/connectivity/subject_05_pronation_r_mean.npy")

    
    # thresholding
    threshold = 0
    subject_01_supination_r_mean = threshold_matrix(subject_01_supination_r_mean, threshold)
    subject_02_supination_r_mean = threshold_matrix(subject_02_supination_r_mean, threshold)
    subject_03_supination_r_mean = threshold_matrix(subject_03_supination_r_mean, threshold)
    subject_04_supination_r_mean = threshold_matrix(subject_04_supination_r_mean, threshold)
    subject_05_supination_r_mean = threshold_matrix(subject_05_supination_r_mean, threshold)

    subject_01_pronation_r_mean = threshold_matrix(subject_01_pronation_r_mean, threshold)
    subject_02_pronation_r_mean = threshold_matrix(subject_02_pronation_r_mean, threshold)
    subject_03_pronation_r_mean = threshold_matrix(subject_03_pronation_r_mean, threshold)
    subject_04_pronation_r_mean = threshold_matrix(subject_04_pronation_r_mean, threshold)
    subject_05_pronation_r_mean = threshold_matrix(subject_05_pronation_r_mean, threshold)

    plt.figure()
    plt.title(f"Composite Correlation [r > {threshold}] - HbO / HbR\n", fontsize=16)
    plt.subplot(2, 5, 1)
    plot_r_matrix(subject_01_supination_r_mean, ch_names, title="Subject 01")

    plt.subplot(2, 5, 2)
    plot_r_matrix(subject_02_supination_r_mean, ch_names, title="Subject 02")

    plt.subplot(2, 5, 3)
    plot_r_matrix(subject_03_supination_r_mean, ch_names, title="Subject 03")

    plt.subplot(2, 5, 4)
    plot_r_matrix(subject_04_supination_r_mean, ch_names, title="Subject 04")

    plt.subplot(2, 5, 5)
    plot_r_matrix(subject_05_supination_r_mean, ch_names, title="Subject 05")

    plt.subplot(2, 5, 6)
    plot_r_matrix(subject_01_pronation_r_mean, ch_names, title="Subject 01")

    plt.subplot(2, 5, 7)
    plot_r_matrix(subject_02_pronation_r_mean, ch_names, title="Subject 02")

    plt.subplot(2, 5, 8)
    plot_r_matrix(subject_03_pronation_r_mean, ch_names, title="Subject 03")

    plt.subplot(2, 5, 9)
    plot_r_matrix(subject_04_pronation_r_mean, ch_names, title="Subject 04")

    plt.subplot(2, 5, 10)
    plot_r_matrix(subject_05_pronation_r_mean, ch_names, title="Subject 05")

    plt.figure()
    mean_supiation = np.mean([subject_01_supination_r_mean, subject_02_supination_r_mean, subject_03_supination_r_mean, subject_04_supination_r_mean, subject_05_supination_r_mean], axis=0)
    mean_pronation = np.mean([subject_01_pronation_r_mean, subject_02_pronation_r_mean, subject_03_pronation_r_mean, subject_04_pronation_r_mean, subject_05_pronation_r_mean], axis=0)
    plt.title(f"Mean Composite Correlation [r > {threshold}] - HbO / HbR\n", fontsize=16)
    plt.subplot(1, 2, 1)
    plot_r_matrix(mean_supiation, ch_names, title="Mean Supination")
    plt.subplot(1, 2, 2)
    plot_r_matrix(mean_pronation, ch_names, title="Mean Pronation")
    plt.show()

    exit()
ROIs = {
    'Both Hemispheres': {
        'S1' : ['S5_D5', 'S5_D6', 'S5_D7', 'S6_D5', 'S6_D7', 'S7_D7', 'S7_D14', 'S9_D6', 'S9_D14' 'S9_D12', 'S9_D14', 'S13_D12', 'S13_D13', 'S13_D15', 'S14_D14', 'S14_D15', 'S15_D13', 'S15_D15'],
        'S2' : ['S7_D8', 'S8_D7', 'S8_D8','S14_D16', 'S16_D16', 'S16_D15'],
        'M1' : ['S3_D3', 'S3_D5', 'S4_D3', 'S4_D4', 'S4_D6', 'S5_D3', 'S9_D4', 'S11_D4', 'S11_D11', 'S11_D12', 'S12_D11', 'S12_D13', 'S13_D11'],
        'PMA': ['S1_D1', 'S1_D2', 'S1_D3', 'S3_D1', 'S3_D3', 'S4_D3','S10_D9', 'S10_D10', 'S10_D11', 'S11_D11', 'S12_D10', 'S12_D11'],
        'SMA': ['S2_D2', 'S2_D4', 'S4_D2', 'S4_D4', 'S2_D9', 'S11_D4', 'S11_D9']
    },
    'Left Hemisphere': {
        'S1' : ['S5_D5', 'S5_D6', 'S5_D7', 'S6_D5', 'S6_D7', 'S7_D7', 'S7_D14', 'S9_D6', 'S9_D14'],
        'S2' : ['S7_D8', 'S8_D7', 'S8_D8'],
        'M1' : ['S3_D3', 'S3_D5', 'S4_D3', 'S4_D4', 'S4_D6', 'S5_D3', 'S9_D4'],
        'PMA': ['S1_D1', 'S1_D2', 'S1_D3', 'S3_D1', 'S3_D3', 'S4_D3'],
        'SMA': ['S2_D2', 'S2_D4', 'S4_D2', 'S4_D4']
    },
    'Right Hemisphere': {
        'S1' : ['S9_D12', 'S9_D14', 'S13_D12', 'S13_D13', 'S13_D15', 'S14_D14', 'S14_D15', 'S15_D13', 'S15_D15'],
        'S2' : ['S14_D16', 'S16_D16', 'S16_D15'],
        'M1' : ['S9_D4', 'S11_D4', 'S11_D11', 'S11_D12', 'S12_D11', 'S12_D13', 'S13_D11'],
        'PMA': ['S10_D9', 'S10_D10', 'S10_D11', 'S11_D11', 'S12_D10', 'S12_D11'],
        'SMA': ['S2_D4', 'S2_D9', 'S11_D4', 'S11_D9']
    }
}


markers = {3 : "Pronation", 
           4 : "Supination"}

def preprocess_fnirs(fnirs):
    fnirs.remove_features([2, 5, 6])
    fnirs.trim_from_features(cut_from_first_feature=5, cut_from_last_feature=16)
    fnirs.preprocess(optical_density=True, 
                      hemoglobin_concentration=True, 
                      temporal_filtering=True, 
                      normalization=True)
    return fnirs

# subject 01
subject_01_supination = [fNIRS("C:/Users/Adam/Desktop/Heel Stimulation/Subject01/Trial 1 - Supination/2025-03-24_001.snirf"),
                      fNIRS("C:/Users/Adam/Desktop/Heel Stimulation/Subject01/Trial 3 - Supination/2025-03-24_003.snirf"),
                      fNIRS("C:/Users/Adam/Desktop/Heel Stimulation/Subject01/Trial 5 - Supination/2025-03-24_005.snirf"),]

subject_01_pronation =  [fNIRS("C:/Users/Adam/Desktop/Heel Stimulation/Subject01/Trial 2 - Pronation/2025-03-24_002.snirf"),
                      fNIRS("C:/Users/Adam/Desktop/Heel Stimulation/Subject01/Trial 4 - Pronation/2025-03-24_004.snirf"),
                      fNIRS("C:/Users/Adam/Desktop/Heel Stimulation/Subject01/Trial 6 - Pronation/2025-03-24_006.snirf"),]
subject_01_supination = [preprocess_fnirs(fnirs) for fnirs in subject_01_supination]
subject_01_pronation = [preprocess_fnirs(fnirs) for fnirs in subject_01_pronation]
# subject 02
subject_02_supination = [fNIRS("C:/Users/Adam/Desktop/Heel Stimulation/Subject02/Trial 1/2025-03-27_002.snirf"),
                      fNIRS("C:/Users/Adam/Desktop/Heel Stimulation/Subject02/Trial 3/2025-03-27_004.snirf"),
                      fNIRS("C:/Users/Adam/Desktop/Heel Stimulation/Subject02/Trial 5/2025-03-27_006.snirf"),]
subject_02_pronation =  [fNIRS("C:/Users/Adam/Desktop/Heel Stimulation/Subject02/Trial 2/2025-03-27_003.snirf"),
                      fNIRS("C:/Users/Adam/Desktop/Heel Stimulation/Subject02/Trial 4/2025-03-27_005.snirf"),
                      fNIRS("C:/Users/Adam/Desktop/Heel Stimulation/Subject02/Trial 6/2025-03-27_007.snirf"),]
subject_02_supination = [preprocess_fnirs(fnirs) for fnirs in subject_02_supination]
subject_02_pronation = [preprocess_fnirs(fnirs) for fnirs in subject_02_pronation]
# subject 03
subject_03_supination = [fNIRS("C:/Users/Adam/Desktop/Heel Stimulation/Subject03 -Only fNIRS/Trial 1/2025-04-01_002.snirf"),
                      fNIRS("C:/Users/Adam/Desktop/Heel Stimulation/Subject03 -Only fNIRS/Trial 3/2025-04-01_005.snirf"),
                      fNIRS("C:/Users/Adam/Desktop/Heel Stimulation/Subject03 -Only fNIRS/Trial 5/2025-04-01_008.snirf"),]
subject_03_pronation =  [fNIRS("C:/Users/Adam/Desktop/Heel Stimulation/Subject03 -Only fNIRS/Trial 2/2025-04-01_003.snirf"),
                      fNIRS("C:/Users/Adam/Desktop/Heel Stimulation/Subject03 -Only fNIRS/Trial 4/2025-04-01_007.snirf"),
                      fNIRS("C:/Users/Adam/Desktop/Heel Stimulation/Subject03 -Only fNIRS/Trial 6/2025-04-01_009.snirf"),]
subject_03_supination = [preprocess_fnirs(fnirs) for fnirs in subject_03_supination]
subject_03_pronation = [preprocess_fnirs(fnirs) for fnirs in subject_03_pronation]
# subject 04
subject_04_supination = [fNIRS("C:/Users/Adam/Desktop/Heel Stimulation/Subject04 - Only fNIRS/Trial 1 - Supination/2025-04-01_017.snirf"),
                      fNIRS("C:/Users/Adam/Desktop/Heel Stimulation/Subject04 - Only fNIRS/Trial 3 - Supination/2025-04-01_019.snirf"),
                      fNIRS("C:/Users/Adam/Desktop/Heel Stimulation/Subject04 - Only fNIRS/Trial 5 - Supination/2025-04-01_021.snirf"),]
subject_04_pronation =  [fNIRS("C:/Users/Adam/Desktop/Heel Stimulation/Subject04 - Only fNIRS/Trial 2 - Pronation/2025-04-01_018.snirf"),
                      fNIRS("C:/Users/Adam/Desktop/Heel Stimulation/Subject04 - Only fNIRS/Trial 4 - Pronation/2025-04-01_020.snirf"),
                      fNIRS("C:/Users/Adam/Desktop/Heel Stimulation/Subject04 - Only fNIRS/Trial 6 - Pronation/2025-04-01_023.snirf"),]
subject_04_supination = [preprocess_fnirs(fnirs) for fnirs in subject_04_supination]
subject_04_pronation = [preprocess_fnirs(fnirs) for fnirs in subject_04_pronation]
# subject 05
subject_05_supination = [fNIRS("C:/Users/Adam/Desktop/Heel Stimulation/Subject05 - Only fNIRS/Trial 1 - Supination/2025-04-01_025.snirf"),
                      fNIRS("C:/Users/Adam/Desktop/Heel Stimulation/Subject05 - Only fNIRS/Trial 3 - Supination/2025-04-01_027.snirf"),
                      fNIRS("C:/Users/Adam/Desktop/Heel Stimulation/Subject05 - Only fNIRS/Trial 5 - Supination/2025-04-01_029.snirf"),]
subject_05_pronation =  [fNIRS("C:/Users/Adam/Desktop/Heel Stimulation/Subject05 - Only fNIRS/Trial 2 - Pronation/2025-04-01_026.snirf"),
                      fNIRS("C:/Users/Adam/Desktop/Heel Stimulation/Subject05 - Only fNIRS/Trial 4 - Pronation/2025-04-01_028.snirf"),
                      fNIRS("C:/Users/Adam/Desktop/Heel Stimulation/Subject05 - Only fNIRS/Trial 6 - Pronation/2025-04-01_030.snirf"),]
subject_05_supination = [preprocess_fnirs(fnirs) for fnirs in subject_05_supination]
subject_05_pronation = [preprocess_fnirs(fnirs) for fnirs in subject_05_pronation]

plt.title("Composite Correlation - HbO / HbR\n")

def get_subject_mean(subject_supination, subject_pronation):
    
    subject_supination_r = []
    for i, trial in enumerate(subject_supination):
        hbo_data, ch_names, hbr_data, _ = trial.split()

        hbo_comp = composite_correlation(hbo_data, hbr_data, trial.sampling_frequency, 150, 0.01, 0.1, False)
        hbo_comp = np.nan_to_num(hbo_comp, nan=0.0, posinf=0.0, neginf=0.0)

        hbr_comp = composite_correlation(hbr_data, hbo_data, trial.sampling_frequency, 150, 0.01, 0.1, False)
        hbr_comp = np.nan_to_num(hbr_comp, nan=0.0, posinf=0.0, neginf=0.0)

        r_comp = merge_matrices_diagonally(hbo_comp, hbr_comp)
        subject_supination_r.append(r_comp)


    subject_pronation_r = []
    for i, trial in enumerate(subject_pronation):

        hbo_data, ch_names, hbr_data, _ = trial.split()

        hbo_comp = composite_correlation(hbo_data, ch_names, trial.sampling_frequency, 150, 0.01, 0.1, False)
        hbr_comp = composite_correlation(hbr_data, ch_names, trial.sampling_frequency, 150, 0.01, 0.1, False)

        r_comp = merge_matrices_diagonally(hbo_comp, hbr_comp)
        subject_pronation_r.append(r_comp)

    subject_supination_r_mean = np.mean(subject_supination_r, axis=0)
    subject_pronation_r_mean = np.mean(subject_pronation_r, axis=0)

    return subject_supination_r_mean, subject_pronation_r_mean, ch_names

# Do each of these functions in a thread

subject_01_supination_r_mean, subject_01_pronation_r_mean, ch_names = get_subject_mean(subject_01_supination, subject_01_pronation)
subject_02_supination_r_mean, subject_02_pronation_r_mean, ch_names = get_subject_mean(subject_02_supination, subject_02_pronation)
subject_03_supination_r_mean, subject_03_pronation_r_mean, ch_names = get_subject_mean(subject_03_supination, subject_03_pronation)
subject_04_supination_r_mean, subject_04_pronation_r_mean, ch_names = get_subject_mean(subject_04_supination, subject_04_pronation)
subject_05_supination_r_mean, subject_05_pronation_r_mean, ch_names = get_subject_mean(subject_05_supination, subject_05_pronation)

# Numpy Save Matrices
np.save("C:/Users/Adam/Desktop/Heel Stimulation/connectivity/subject_01_supination_r_mean.npy", subject_01_supination_r_mean)
np.save("C:/Users/Adam/Desktop/Heel Stimulation/connectivity/subject_01_pronation_r_mean.npy", subject_01_pronation_r_mean)
np.save("C:/Users/Adam/Desktop/Heel Stimulation/connectivity/subject_02_supination_r_mean.npy", subject_02_supination_r_mean)
np.save("C:/Users/Adam/Desktop/Heel Stimulation/connectivity/subject_02_pronation_r_mean.npy", subject_02_pronation_r_mean)
np.save("C:/Users/Adam/Desktop/Heel Stimulation/connectivity/subject_03_supination_r_mean.npy", subject_03_supination_r_mean)
np.save("C:/Users/Adam/Desktop/Heel Stimulation/connectivity/subject_03_pronation_r_mean.npy", subject_03_pronation_r_mean)
np.save("C:/Users/Adam/Desktop/Heel Stimulation/connectivity/subject_04_supination_r_mean.npy", subject_04_supination_r_mean)
np.save("C:/Users/Adam/Desktop/Heel Stimulation/connectivity/subject_04_pronation_r_mean.npy", subject_04_pronation_r_mean)
np.save("C:/Users/Adam/Desktop/Heel Stimulation/connectivity/subject_05_supination_r_mean.npy", subject_05_supination_r_mean)
np.save("C:/Users/Adam/Desktop/Heel Stimulation/connectivity/subject_05_pronation_r_mean.npy", subject_05_pronation_r_mean)
