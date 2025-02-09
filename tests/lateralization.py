import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mne.io import read_raw_snirf
from preprocessing.fnirs.conversion import light_intensity_to_hemoglobin_concentration
from preprocessing.fnirs.filtering import butter_bandpass_filter
left_hemisphere_channels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 31, 34, 38, 39, 40, 41, 42, 43, 44]
right_hemisphere_channels = [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 35, 45, 45, 46, 47, 48, 49, 51]
motor_roi = ( #Motor cortex
        ["S1-D1", "S2-D1", "S1-D3", "S2-D3", "S9-D1", "S1-D9", "S10-D2", "S3-D3", "S3-D2" ] ,  #LEFT HEMISPHEre
        ["S5-D5", "S5-D6", "S5-D7", "S5-D9", "S9-D5", "S6-D5", "S6-D7", "S6-D7", "S10-D6" ] #RIGHT HEMISPHERES
)

#Load GLM results
omid_path = "data/glm_results/omid"
omid_filepaths = []
runs = []
run_names = []
for entry in os.listdir(omid_path):
        file_path = os.path.join(omid_path, entry)
        runs.append(pd.read_excel(file_path))
        run_names.append("Run " + str(len(runs)))
        omid_filepaths.append(file_path)

#Identify channel indices
channels = {}
for channel in left_hemisphere_channels + right_hemisphere_channels:
        ch_name = str(runs[0].iloc[channel, 1])
        channels[ch_name] = channel

#load snirfs
snirf_paths = [
        "data/OMID-13-12-024/2024-12-13_001/2024-12-13_001.snirf", 
        "data/OMID-13-12-024/2024-12-13_002/2024-12-13_002.snirf",
        "data/OMID-13-12-024/2024-12-13_003/2024-12-13_003.snirf",
        "data/OMID-13-12-024/2024-12-13_004/2024-12-13_004.snirf",
        ]

snirfs = []
for path in snirf_paths:
        snirf = read_raw_snirf(path).load_data()
        snirfs.append(snirf)

def extract_blocks(snirf, pre_onset=5, post_onset=15):
        snirf = light_intensity_to_hemoglobin_concentration(snirf)
        data = np.array(snirf.get_data())
        # Compute the mean and standard deviation
        mean = np.mean(data)
        std = np.std(data)

        # Z-normalize the array
        data = (data - mean) / std
        sfreq = snirf.info["sfreq"]

        #filter all data
        for channel in range(len(data)):
                data[channel] = butter_bandpass_filter(data[channel], 0.01, 0.09, sfreq, 20)


        annotations = snirf._annotations
        descriptions = annotations.description
        onsets = annotations.onset
        
        task_data = [] #now we have samples for each task in this snirf
        for i in range(len(descriptions)): #task index
                description = int(descriptions[i])
                if description == 0 or description == 3: #dont collect "REST" or "END"
                        continue

                start_seconds = onsets[i] - pre_onset #prior to onset
                duration = pre_onset + post_onset #after onset
                end_seconds = start_seconds + duration #

                slice_length = int((pre_onset + post_onset) * sfreq)
                onset_frame = int(start_seconds * sfreq)
                end_frame = onset_frame + int(duration * sfreq)

                task = data[:, onset_frame:end_frame] #all of these tasks

                if task.shape[1] != slice_length:#pad it
                        remainder = slice_length - task.shape[1] 
                        if remainder > 0:
                                task = np.pad(task[:], (0, slice_length - task.shape[1]), constant_values=0)
                        if remainder < 0: #only get the 50
                                task = task[:, :slice_length]


                task_data.append(task)
                if False:
                        print("Task : ", description)
                        print("Onset : ", start_seconds, " (", onset_frame, ")")
                        print("End : ", end_seconds, " (", end_frame, ")")
                        print("Shape : ", task.shape)     

        return task_data

def get_rois_from_block(block, roi):
        lh = []
        rh = []
        for left, right in zip(roi[0], roi[1]):
                if left not in channels:
                        print("Channels : KEY ", left ," NOT FOUND")
                        continue
                if right not in channels:
                        print("Channels : KEY ", right ," NOT FOUND")
                        continue

                lh.append(block[channels[left] -1]) #get the correct channel
                rh.append(block[channels[right]-1])

        lh = np.array(lh) #left hemisphere channels during the block
        rh = np.array(rh)
        return lh, rh

def get_mean_responses_from_blocks(blocks):
        lh_means = []
        rh_means = []
        for block in blocks:
                lh, rh = get_rois_from_block(block, motor_roi) #filter from ROI

                lh_mean = np.mean(lh, axis=0, keepdims=True)
                rh_mean = np.mean(rh, axis=0, keepdims=True)
                
                lh_means.append(lh_mean[0])
                rh_means.append(rh_mean[0])
        return lh_means, rh_means
        
run1_blocks = extract_blocks(snirfs[2], pre_onset=10, post_onset=20) #Gets all the blocks
r1_blocks = np.array(run1_blocks)
r1_ll = r1_blocks[[1, 3, 5]]
r1_rl = r1_blocks[[0, 2, 4]]

print("r1_ll : ", r1_ll.shape)
print("r1_rl : ", r1_rl.shape)

#RUN 1 LEFT LEG 
ll_lh, ll_rh = get_mean_responses_from_blocks(r1_ll)
ll_lh = np.array(ll_lh)
ll_rh = np.array(ll_rh)

fig, axs = plt.subplots(2, len(ll_lh))
for block in range(len(ll_lh)):
        axs[0, block].plot(ll_lh[block])
for block in range(len(ll_rh)):
        axs[1, block].plot(ll_rh[block])

fig.suptitle("LEFT LEG STANCE : LEFT HEMISPHERE vs RIGHT HEMISPHERE")

#RUN 1 RIGHT LEG
rl_lh, rl_rh = get_mean_responses_from_blocks(r1_rl)
rl_lh = np.array(rl_lh)
rl_rh = np.array(rl_rh)

fig, axs = plt.subplots(2, len(rl_lh))
for block in range(len(rl_lh)):
        axs[0, block].plot(rl_lh[block])
for block in range(len(rl_rh)):
        axs[1, block].plot(rl_rh[block])

fig.suptitle("RIGHT LEG STANCE : LEFT HEMISPHERE vs RIGHT HEMISPHERE")

plt.show()
exit()
#RUN 1 RIGHT LEG
rl_lh, rl_rh = get_mean_responses_from_blocks(r1_rl)

fig, axs = plt.subplots(2,3)

fig.suptitle("RUN 1 LEFT LEG")
plt.show()

exit()
run2 = extract_blocks(snirfs[1], 5, 15)
run3 = extract_blocks(snirfs[2], 5, 15)
run4 = extract_blocks(snirfs[3], 5, 15)
exit()

#load snirf for channel names -> THIS CAN BE DONE FROM THE GLM XLSL FILE INSTEAD
#snirf_path = "data/OMID-13-12-024/2024-12-13_001/2024-12-13_001.snirf"
#snirf = read_raw_snirf(snirf_path)
#ch_names = snirf.info["ch_names"]
#middle = int(len(ch_names) / 2)
#step_size = 2
#channels = {}
#for channel in left_hemisphere_channels + right_hemisphere_channels:
#    corresponding = (ch_names[channel-1], ch_names[middle+(channel-1)])
#    channels[channel] = corresponding

#pair left hemispehre channels with acutal channel names

#based on "sensori" fnirs montage


   
print("CHANNELS :\n", channels)

def calculate_roi_li(roi):
        ll_lis = []
        rl_lis = []
        for run in range(len(runs)):
                
                left_beta_values = [] #collect all left hemisphere beta values (ll, rl)
                right_beta_values = [] #right hemisphere(ll, rl)
                # go through all runs
                for left, right in zip(roi[0], roi[1]):
                        if left not in channels:
                                print("Channels : KEY ", left ," NOT FOUND")
                                continue
                        if right not in channels:
                                print("Channels : KEY ", right ," NOT FOUND")
                                continue
                
                        left_ch = channels[left]
                        right_ch = channels[right]

                        #LEFT HEMISPHERE
                        lh_rl_beta = float(runs[run].iloc[left_ch, 2])
                        lh_ll_beta = float(runs[run].iloc[left_ch, 6])
                        left_beta_values.append((lh_ll_beta, lh_rl_beta))

                        #RIGHT HEMISPHERE
                        rh_rl_beta = float(runs[run].iloc[right_ch, 2])
                        rh_ll_beta = float(runs[run].iloc[right_ch, 6])
                        right_beta_values.append((rh_ll_beta, rh_rl_beta))
                
                lh_ll = np.array([tup[0] for tup in left_beta_values])
                lh_rl = np.array([tup[1] for tup in left_beta_values])

                rh_ll = np.array([tup[0] for tup in right_beta_values])
                rh_rl = np.array([tup[1] for tup in right_beta_values])

                lh_ll_avg = np.average(lh_ll)
                lh_rl_avg = np.average(lh_rl)           
                rh_ll_avg = np.average(rh_ll)
                rh_rl_avg = np.average(rh_rl)
                
                ll_li = float((lh_ll_avg - rh_ll_avg) / (lh_ll_avg + rh_ll_avg))
                rl_li = float((lh_rl_avg - rh_rl_avg) / (lh_rl_avg + rh_rl_avg))

                ll_lis.append(ll_li)
                rl_lis.append(rl_li)
                #print("Channels : ", left, ", ", right)
                #print("Channel indicies : ", left_ch, ", ", right_ch)
                #print("Run ", run + 1, " : " )
                #print("LEFT HEMISPHERE AVERAGES : Left Leg= ", lh_ll_avg, ", Right Leg= ", lh_rl_avg)
                #print("RIGHT HEMISPHERE AVERAGES : Left Leg= ", rh_ll_avg, ", Right Leg= ", rh_rl_avg)

        return ll_lis, rl_lis
  
ll, rl = calculate_roi_li(motor_roi)

fig, axs = plt.subplots(1, 2)


axs[0].set_title("Left Leg")
axs[1].set_title("Right Leg")

axs[0].plot(ll, color="blue", marker = "o", linestyle= "--", linewidth = 3,
        markersize = 8)
axs[1].plot(rl, color="green", marker = "o", linestyle= "--", linewidth = 3,
        markersize = 8)

axs[0].set_xlim([0, len(runs)-1])
axs[0].set_ylim([-1, 1])
axs[1].set_xlim([0, len(runs)-1])
axs[1].set_ylim([-1, 1])

axs[0].grid(True)
axs[1].grid(True)
fig.suptitle("Formula 3 :\nMotor Cortex ROI Lateralization Index")

plt.show()
exit()
wh_f1_rl = []
wh_f2_rl = []
wh_f3_rl = []
wh_f1_ll = []
wh_f2_ll = []
wh_f3_ll = []

cc_f1_rl = []
cc_f2_rl = []
cc_f3_rl = []
cc_f1_ll = []
cc_f2_ll = []
cc_f3_ll = []

idx = 0
for run in runs:
    #LEFT HEMISPHERE
    rl_lh_beta = []
    ll_lh_beta = []
    #RIGHT HEMISPHERE
    rl_rh_beta = []
    ll_rh_beta = []

    for channel in left_hemisphere_channels:
        rl_channel_beta_value = float(run.iloc[channel, 2])
        rl_lh_beta.append(rl_channel_beta_value)

        ll_channel_beta_value = float(run.iloc[channel, 6])
        ll_lh_beta.append(ll_channel_beta_value)

    for channel in right_hemisphere_channels:
        rl_channel_beta_value = float(run.iloc[channel, 2])
        rl_rh_beta.append(rl_channel_beta_value)

        ll_channel_beta_value = float(run.iloc[channel, 6])
        ll_rh_beta.append(ll_channel_beta_value)

    #right hemisphere
    rl_lh = np.array(rl_lh_beta)
    ll_lh = np.array(ll_lh_beta)
    #left hemisphere
    rl_rh = np.array(rl_rh_beta)
    ll_rh = np.array(ll_rh_beta)

    #normalize beta values
    rl_lh = rl_lh / np.linalg.norm(rl_lh)
    ll_lh = ll_lh / np.linalg.norm(ll_lh)

    rl_rh = rl_rh / np.linalg.norm(rl_rh)
    ll_rh = ll_rh / np.linalg.norm(ll_rh)

    #channel by channel
    #Formula 1
    f1_li_ll = np.divide(np.subtract(ll_lh, ll_rh), np.add(ll_lh, ll_rh))
    f1_li_rl = np.divide(np.subtract(rl_lh, rl_rh), np.add(rl_lh, rl_rh))
    #Formula 2
    f2_li_ll = np.divide(np.subtract(ll_lh, ll_rh), np.add(np.abs(ll_lh), np.abs(ll_rh)))
    f2_li_rl = np.divide(np.subtract(rl_lh, rl_rh), np.add(np.abs(rl_lh), np.abs(rl_rh)))
    #Formula 3
    f3_li_ll = np.divide(np.subtract(np.abs(ll_lh), np.abs(ll_rh)), np.add(np.abs(ll_lh), np.abs(ll_rh)))
    f3_li_rl = np.divide(np.subtract(np.abs(rl_lh), np.abs(rl_rh)), np.add(np.abs(rl_lh), np.abs(rl_rh)))

    cc_f1_rl.append(f1_li_rl)
    cc_f2_rl.append(f2_li_rl)
    cc_f3_rl.append(f3_li_rl)
    cc_f1_ll.append(f1_li_ll)
    cc_f2_ll.append(f2_li_ll)
    cc_f3_ll.append(f3_li_ll)

    #whole hemisphere
    mean_rl_lh = np.mean(rl_lh)
    mean_rl_rh = np.mean(rl_rh)

    mean_ll_lh = np.mean(ll_lh)
    mean_ll_rh = np.mean(ll_rh)

    abs_mean_rl_lh = np.abs(mean_rl_lh)
    abs_mean_rl_rh = np.abs(mean_rl_rh)
    
    abs_mean_ll_lh = np.abs(mean_ll_lh)
    abs_mean_ll_rh = np.abs(mean_ll_rh)

    #FORMULA 1
    li_mean_rl = (mean_rl_lh-mean_rl_rh) / (mean_rl_lh+mean_rl_rh)
    li_mean_ll = (mean_ll_lh-mean_ll_rh) / (mean_ll_lh+mean_ll_rh)
    #FORMULA 2
    li_abs_mean_rl = (mean_rl_lh-mean_rl_rh) / (abs_mean_rl_lh+abs_mean_rl_rh)
    li_abs_mean_ll = (mean_ll_lh-mean_ll_rh) / (abs_mean_ll_lh+abs_mean_ll_rh)
    #FORMULA 3
    li_abs_abs_mean_rl = (abs_mean_rl_lh-abs_mean_rl_rh) / (abs_mean_rl_lh+abs_mean_rl_rh)
    li_abs_abs_mean_ll = (abs_mean_ll_lh-abs_mean_ll_rh) / (abs_mean_ll_lh+abs_mean_ll_rh)

    wh_f1_rl.append(li_mean_rl)
    wh_f1_ll.append(li_mean_ll)
    wh_f2_rl.append(li_abs_mean_rl)
    wh_f2_ll.append(li_abs_mean_ll)
    wh_f3_rl.append(li_abs_abs_mean_rl)
    wh_f3_ll.append(li_abs_abs_mean_ll)

fig, axs = plt.subplots(1, 3)
fig.suptitle("Channel by Channel Lateralization Index")
axs[0].set_title("Formula 1")
axs[0].plot(cc_f1_ll[3], color="blue", marker = "o", linestyle= "--", linewidth = 1,
        markersize = 3)
axs[0].plot(cc_f1_rl[2], color="green", marker = "o", linestyle= "--", linewidth = 1,
        markersize = 5)
axs[1].set_title("Formula 2")
axs[1].plot(cc_f2_ll[3], color="blue", marker = "o", linestyle= "--", linewidth = 1,
        markersize = 5)
axs[1].plot(cc_f2_rl[3], color="green", marker = "o", linestyle= "--", linewidth = 1,
        markersize = 5)
axs[2].set_title("Formula 3")
axs[2].plot(cc_f3_ll[3], color="blue", marker = "o", linestyle= "--", linewidth = 1,
        markersize = 5)
axs[2].plot(cc_f3_rl[3], color="green", marker = "o", linestyle= "--", linewidth = 1,
        markersize = 5)

fig.legend(["Left Leg", "Right Leg"])
plt.show()

wh_f1_rl = np.array(wh_f1_rl)
wh_f2_rl = np.array(wh_f2_rl)
wh_f3_rl = np.array(wh_f3_rl)
wh_f1_ll = np.array(wh_f1_ll)
wh_f2_ll = np.array(wh_f2_ll)
wh_f3_ll = np.array(wh_f3_ll)

wh_f1 = np.stack([wh_f1_ll, wh_f1_rl], axis=0)
wh_f2 = np.stack([wh_f2_ll, wh_f2_rl], axis=0)
wh_f3 = np.stack([wh_f3_ll, wh_f3_rl], axis=0)

columns = ["Run 1", "Run 2", "Run 3", "Run 4"]
rows = ["Left Leg", "Right Leg"]
f1 = pd.DataFrame(wh_f1, columns=columns, index=rows)
f2 = pd.DataFrame(wh_f2, columns=columns, index=rows)
f3 = pd.DataFrame(wh_f3, columns=columns, index=rows)

print("Formula 1 : Whole Hemisphere")
print(f1)
print("\n")
print("Formula 2 : Whole Hemisphere")
print(f2)
print("\n")
print("Formula 3 : Whole Hemisphere")
print(f3)
print("\n")

fig, axs = plt.subplots(1, 3)
fig.suptitle("Whole Hemisphere Lateralization Index")

axs[0].set_title("Formula 1")
axs[0].plot(wh_f1[0], color="blue", marker = "o", linestyle= "--", linewidth = 3,
        markersize = 10)
axs[0].plot(wh_f1[1], color="green",marker = "o", linestyle= "--", linewidth = 3,
        markersize = 10)
axs[0].set_xlim([0, 3])
axs[0].set_ylim([-1, 1])

axs[1].set_title("Formula 2")
axs[1].plot(wh_f2[0], color="blue",marker = "o", linestyle= "--", linewidth = 3,
        markersize = 10)
axs[1].plot(wh_f2[1], color="green",marker = "o", linestyle= "--", linewidth = 3,
        markersize = 10)
axs[1].set_xlim([0, 3])
axs[1].set_ylim([-1, 1])

axs[2].set_title("Formula 3")
axs[2].plot(wh_f3[0], color="blue",marker = "o", linestyle= "--", linewidth = 3,
        markersize = 10)
axs[2].plot(wh_f3[1], color="green",marker = "o", linestyle= "--", linewidth = 3,
        markersize = 10)

fig.legend(["Left Leg", "Right Leg"])
axs[2].set_xlim([0, 3])
axs[2].set_ylim([-1, 1])
axs[0].grid(True)
axs[1].grid(True)
axs[2].grid(True)
plt.show()

def find_li():
    

    omid_data = []
    for path in omid_filepaths:
        df = pd.read_excel(path)

        rl_lh = []
        ll_lh = []

        rl_rh = []
        ll_rh = []
        
        rl_lh = np.array(rl_lh)
        ll_lh = np.array(ll_lh)
        rl_rh = np.array(rl_rh)
        ll_rh = np.array(ll_rh)

        abs_rl_lh = np.abs(rl_lh)
        abs_ll_lh = np.abs(ll_lh)
        abs_rl_rh = np.abs(rl_rh)
        abs_ll_rh = np.abs(ll_rh)
        
        if False:
            print("path : ", path)
            print("rl_lh : ", rl_lh)
            print("ll_lh : ", ll_lh)
            print("\n")
            print("rl_rh : ", rl_rh)
            print("ll_rh : ", ll_rh)
            print("\n")
        
        print("path : ", path)
        print("mean rl lh : ", np.mean(rl_lh))
        print("mean ll lh : ", np.mean(ll_lh))
        print("mean rl rh : ", np.mean(rl_rh))
        print("mean ll rh : ", np.mean(ll_rh))
        #FORMULA 1
        print("FORMULA 1 : ")
        rl_li = (np.mean(rl_lh) - np.mean(rl_rh)) / (np.mean(rl_lh)  + np.mean(rl_rh))
        ll_li = (np.mean(ll_lh) - np.mean(ll_rh)) / (np.mean(ll_lh) + np.mean(ll_rh))
        print("RIGHT LEG LATERALITY : ", rl_li)
        print("LEFT LEG LATERALITY : ", ll_li)

        #FORMULA 2
        print("FORMULA 2 : ")
        rl_li = (np.mean(rl_lh) - np.mean(rl_rh)) / (np.abs(np.mean(rl_lh))  + np.abs(np.mean(rl_rh)))
        ll_li = (np.mean(ll_lh) - np.mean(ll_rh)) / (np.abs(np.mean(ll_lh)) + np.abs(np.mean(ll_rh)))
        print("RIGHT LEG LATERALITY : ", rl_li)
        print("LEFT LEG LATERALITY : ", ll_li)

        
        #FORMULA 2
        print("FORMULA 3 : ")
        rl_li = (np.mean(np.abs(rl_lh)) - np.mean(np.abs(rl_rh)) )/ (np.mean(np.abs(rl_lh))  + np.mean(np.abs(rl_rh)))
        ll_li = (np.mean(np.abs(ll_lh)) - np.mean(np.abs(ll_rh)) )/ (np.mean(np.abs(ll_lh)) +  np.mean(np.abs(ll_rh)))
        print("RIGHT LEG LATERALITY : ", rl_li)
        print("LEFT LEG LATERALITY : ", ll_li)
    return
   
    right_leg_left_hemisphere = []
    left_leg_left_hemisphere = []
    right_leg_right_hemisphere = []
    left_leg_right_hemisphere = []
    print(omid_data[0].loc["0:15", 'a'])
    return 
    for run in omid_data:
        rl_lh_data = []
        #right leg -> left hemipshere
        rl_lh = run.loc[3:18, "c"]
        print(rl_lh)
    #find what channels are left and right hemisphere

    # calculate using block averaging?

    pass

#find_li()


def lateralization_index(x, y):
    if (x + y) == 0: #divide by zero 
        return 0
    
    li = (x - y) / (x + y)

    return li