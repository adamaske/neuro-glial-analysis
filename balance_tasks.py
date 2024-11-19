import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os



#LEFT HEMISPHERE CHANNELS
left_channels = ["s1d1", "s1d2", "s1d3", "s2d1", "s2d3", "s2d4", "s3d2", "s3d3", "s4d3", "s4d4"]
#RIGHT HEMISPHERE CHANNELS
right_channels = ["s5d5", "s5d6", "s5d7", "s6d5", "s6d7", "s6d8", "s7d6", "s7d7", "s8d7", "s8d8"]
channels = left_channels + right_channels
print("Channels : \n", channels)
l_idx = [0, 1, 2, 3, 4, 5, 6, 7, 9]
r_idx = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

left = []
right = []

for channel in left_channels:
    data = np.array(pd.read_csv("data/balance-8-11/left_hemisphere/" + channel + ".csv")).transpose()
    left.append(data)

for channel in right_channels:
    data = np.array(pd.read_csv("data/balance-8-11/right_hemisphere/" + channel + ".csv")).transpose()
    right.append(data)

left = np.array(left) #left
right = np.array(right)


l_mean_oxy  = np.mean([left[0][0], 
                     left[1][0],
                     left[2][0],
                     left[3][0],
                     left[4][0],
                     left[5][0],
                     left[6][0],
                     left[7][0],
                     left[8][0],
                     left[9][0]], axis=0)
l_mean_glm_oxy  = np.mean([left[0][1], 
                           left[1][1],
                           left[2][1],
                           left[3][1],
                           left[4][1],
                           left[5][1],
                           left[6][1],
                           left[7][1],
                           left[8][1],
                           left[9][1]], axis=0)
l_mean_deoxy  = np.mean([left[0][2], 
                         left[1][2],
                         left[2][2],
                         left[3][2],
                         left[4][2],
                         left[5][2],
                         left[6][2],
                         left[7][2],
                         left[8][2],
                         left[9][2]], axis=0)
l_mean_glm_deoxy  = np.mean([left[0][3], 
                             left[1][3],
                             left[2][3],
                             left[3][3],
                             left[4][3],
                             left[5][3],
                             left[6][3],
                             left[7][3],
                             left[8][3],
                             left[9][3]], axis=0)
r_mean_oxy  = np.mean([right[0][0], 
                       right[1][0],
                       right[2][0],
                       right[3][0],
                       right[4][0],
                       right[5][0],
                       right[6][0],
                       right[7][0],
                       right[8][0],
                       right[9][0]], axis=0)
r_mean_glm_oxy  = np.mean([right[0][1], 
                           right[1][1],
                           right[2][1],
                           right[3][1],
                           right[4][1],
                           right[5][1],
                           right[6][1],
                           right[7][1],
                           right[8][1],
                           right[9][1]], axis=0)
r_mean_deoxy  = np.mean([right[0][2], 
                         right[1][2],
                         right[2][2],
                         right[3][2],
                         right[4][2],
                         right[5][2],
                         right[6][2],
                         right[7][2],
                         right[8][2],
                         right[9][2]], axis=0)
r_mean_glm_deoxy  = np.mean([right[0][3], 
                             right[1][3],
                             right[2][3],
                             right[3][3],
                             right[4][3],
                             right[5][3],
                             right[6][3],
                             right[7][3],
                             right[8][3],
                             right[9][3]], axis=0)

onset = np.array([190, 500, 584, 745, 848, 1000, 1075, 1230, 1306, 1618, 1709, 1871, 1969]) #what frame did each stimuli occur
onset_seconds = np.divide(onset, 7.81) #turn it into seconds (thats what both mne and satori uses for markers)
durations = [40, 10, 20, 10, 20, 10, 20, 10, 40, 10, 20, 10, 40] #duration of each stimuli
event_description = ["rest", "right", "rest", "left", "rest", "right", "rest", "left", "rest" , "right", "rest", "left", "rest"] #order of stimulis / markers


def plot_mean_respones():
    fig, axs =  plt.subplots(1, 1)
    axs.plot(l_mean_glm_oxy, color="red")
    axs.plot(l_mean_glm_deoxy, color="blue")
    axs.legend(["HbO", "HbR", ])
    
    fig, axs =  plt.subplots(1, 1)
    axs.plot(r_mean_glm_oxy, color="red")
    axs.plot(r_mean_glm_deoxy, color="blue")
    axs.legend(["HbO", "HbR"])

    #maxs[0].set_ylim(0, 4)
    #maxs[1].set_ylim(0, 4)

plot_mean_respones()



plt.show()