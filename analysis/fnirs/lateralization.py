import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#based on "sensori" fnirs montage
left_hemisphere_channels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 31, 34, 38, 39, 40, 41, 42, 43, 44]
right_hemisphere_channels = [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 35, 45, 45, 46, 47, 48, 49, 51]
omid_path = "data/glm_results/omid"
omid_filepaths = []
runs = []

for entry in os.listdir(omid_path):
        file_path = os.path.join(omid_path, entry)
        runs.append(pd.read_excel(file_path))
        omid_filepaths.append(file_path)
run1 = pd.read_excel(omid_filepaths[0])
run2 = pd.read_excel(omid_filepaths[1])
run3 = pd.read_excel(omid_filepaths[2])
run4 = pd.read_excel(omid_filepaths[3])

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