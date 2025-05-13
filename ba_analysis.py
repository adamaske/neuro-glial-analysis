import numpy as np
import matplotlib.pyplot as plt
from hrf import double_gamma_chrf
from beta_values import glm
from neuropipeline.fnirs import fNIRS
from data import get_fnirs_data, get_eeg_data


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

fnirs_data = get_fnirs_data("C:/Users/Adam/Desktop/Heel Stimulation/data")

supination_trials = [0, 2, 4]
pronation_trials = [1, 3, 5]

markers = {3 : "Pronation", 
           4 : "Supination"}

tmin = 0
tmax = 20
channel_names = []

supination_epochs = []
pronation_epochs = []

for i, subject in enumerate(fnirs_data.keys()):
    
    for j, trial in enumerate(fnirs_data[subject]):

        if j in pronation_trials:
            trial_type = "Pronation"
        elif j in supination_trials:
            trial_type = "Supination"
        
        #hbo_data, ch_names, hbr_data, _ = fnirs_data[subject][trial].split()
        
        fnirs = fnirs_data[subject][trial]
        hbo_data, ch_names, hbr_data, _ = fnirs.split()
        
        for k, onset in enumerate(fnirs.feature_onsets):
            desc = fnirs.feature_descriptions[k]
            
            marker = markers[desc]
            
            start = int((onset + tmin) * fnirs.sampling_frequency)
            end = int((onset + tmax) * fnirs.sampling_frequency)
            
            epoch_hbo = hbo_data[:, start:end]
            epoch_hbr = hbr_data[:, start:end]
            
            # Normalize the epochs to the range [-1, 1]
            epoch_hbo_normalized = 2 * (epoch_hbo - np.min(epoch_hbo)) / (np.max(epoch_hbo) - np.min(epoch_hbo)) - 1
            epoch_hbr_normalized = 2 * (epoch_hbr - np.min(epoch_hbr)) / (np.max(epoch_hbr) - np.min(epoch_hbr)) - 1

            if marker == "Pronation":
                pronation_epochs.append(epoch_hbo_normalized)
            elif marker == "Supination":
                supination_epochs.append(epoch_hbo_normalized)

avg_supination = []
avg_pronation = []

B_supination = []
B_pronation = []

for i, (s, p) in enumerate(zip(supination_epochs, pronation_epochs)):
     
    time = np.linspace(tmin, tmax, s.shape[1])
    chrf = double_gamma_chrf(time, 6, 16, 1, 1, 1/6)

    avg_s = np.mean(s, axis=0)
    avg_p = np.mean(p, axis=0)
    
    avg_supination.append(avg_s)
    avg_pronation.append(avg_p)
    
    B_s = glm(avg_s, chrf)
    B_p = glm(avg_p, chrf)
    
    B_supination.append(B_s)
    B_pronation.append(B_p)
    
    #plt.title(f"Subject {i+1} - {trial_type}\n")
    #plt.subplot(1, 2, 1)
    #plt.title(f"Supination : {B_s}")
    #for j, channel in enumerate(s):
    #    plt.plot(time, s[j], color="red")
    #plt.plot(time, avg_s, color="black", linewidth=2)
    #plt.plot(time, chrf, color="black", linewidth=2, label="cHRF")
    #plt.legend()
    #plt.subplot(1, 2, 2)
    #plt.title(f"Pronation : {B_p}")
    #for j, channel in enumerate(p):
    #    plt.plot(time, p[j], color="blue")
    #plt.plot(time, avg_p, color="black", linewidth=2)
    #plt.plot(time, chrf, color="black", linewidth=2, label="cHRF")
    #plt.legend()
    #plt.show()

print("Supination Beta Values: ", np.mean(B_supination))
print("Pronation Beta Values: ", np.mean(B_pronation))
exit()
# Block Average Analysis
for i, subject in enumerate(fnirs_data.keys()):
    
    for j, trial in enumerate(fnirs_data[subject]):
        
        
        if j in pronation_trials:
            trial_type = "Pronation"
        elif j in supination_trials:
            trial_type = "Supination"
        
        
        fnirs = fnirs_data[subject][trial]
        hbo_data, ch_names, hbr_data, _ = fnirs.split()
        
        if i == 0 and j == 0:
            channel_names = ch_names
        
        for k, onset in enumerate(fnirs.feature_onsets):
            desc = fnirs.feature_descriptions[k]

            marker = markers[desc]
            
            start = int((onset + tmin) * fnirs.sampling_frequency)
            end = int((onset + tmax) * fnirs.sampling_frequency)
    
            channel_names = ch_names
            
            epoch_hbo = hbo_data[:, start:end]
            epoch_hbr = hbr_data[:, start:end]
            
            # Normalize the epochs to the range [-1, 1]
            epoch_hbo_normalized = 2 * (epoch_hbo - np.min(epoch_hbo)) / (np.max(epoch_hbo) - np.min(epoch_hbo)) - 1
            epoch_hbr_normalized = 2 * (epoch_hbr - np.min(epoch_hbr)) / (np.max(epoch_hbr) - np.min(epoch_hbr)) - 1

            
            time = np.linspace(tmin, tmax, hbo.shape[1])
            chrf = double_gamma_chrf(time, 6, 16, 1, 1, 1/6)

            B_hbo = glm(epoch_hbo_normalized, chrf)
            B_hbr = glm(epoch_hbr_normalized, chrf)
            