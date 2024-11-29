import os
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

from mne_nirs.io import write_raw_snirf
from mne.io import read_raw_snirf
from mne import Annotations
from experiments.experiment import Experiment
from datasets.fnirs_data import find_snirf_in_folder
from preprocessing import preprocess
#CREATE EXPERIEMNT
#exp = Experiment(name="balance with mats", #because
#                 time=datetime(day=15, month=11, year=2024),
#                 description="5 trails: barefoot, easy mat, easy mat with visual marker, and difficult mat",
#                 participants="Adam",
#                 data_folder="data/balance-15-11",
#                 filepath="data/balance-15-11/experiment.exp").write_json("data/balance-15-11/experiment.exp"
#                 )

#LOAD EXPERIMENT
exp = Experiment()
exp.read_json("data/balance-15-11/experiment.exp")

data_folder = exp.data_folder


#FIND ALL SNIRFS IN THE DATA FOLDER
snirfs = find_snirf_in_folder(data_folder)

for entry in snirfs:
    print("snirf : ", entry)

# PUTTING MARKERS ONTO OUR DATA -> CALLED ANNOTATIONS IN MNE
barefoot_thickmat = read_raw_snirf(snirfs[4])
sampling_frequency = barefoot_thickmat.info["sfreq"]

onset = np.array([102, 309, 375, 491, 543, 750, 801, 911, 971, 1164, 1220, 1320, 1370]) #what frame did each stimuli occur
onset_seconds = np.divide(onset, sampling_frequency) #turn it into seconds (thats what both mne and satori uses for markers)
durations = [40, 10, 20, 10, 40, 10, 20, 10, 40, 10, 20, 10, 40] #duration of each stimuli
event_description = ["rest", "right", "rest", "left", "rest", "right", "rest", "left", "rest" , "right", "rest", "left", "rest"] #order of stimulis / markers

new_annotations = Annotations(onset=onset_seconds,
                              duration=durations,
                              description=event_description)
barefoot_thickmat.set_annotations(new_annotations)

#THE SNRIF HAS BEEN RE-ANNOTATED
write_raw_snirf(barefoot_thickmat, "data/balance-15-11/barefoot_thick_mat.snirf")

exit()
#WE WROTE DOWN THE ONSET OF EACH STIMULI WITH A DATAFRAME
#WE NEED TO TURN THIS INTO SECONDS
event_times_framecount = ["231", "453", "etc..."]
#DIVIDE BY THE SAMPLING RATE TO TURN THE FRAME COUNT INTO SECONDS
events_secs = np.divide(event_times_framecount, run1.info["sfreq"])
br = 40 #big rest
sr = 20 #small rest
r = 10 #right
l = 10 #left
event_description = ["rest", "right", "rest", "left", "rest", "right", "rest", "left", "rest", "right", "rest", "left", "rest"]

new_annotations = Annotations(onset=events_secs, 
                              duration = [40, 10, 20, 10, 40, 10, 20, 10, 40 , 10, 20, 10, 40], 
                              description =["rest", "right", "rest", "left", "rest", "right", "rest", "left", "rest", "right", "rest", "left", "rest"],
                              )

exit()


channel_data = np.array(pd.read_csv("data/balance-15-11/2_channel_for_correlation.csv")).transpose()

ch1_oxy = channel_data[0]
ch1_deoxy = channel_data[1]

ch2_oxy = channel_data[2]
ch2_deoxy = channel_data[3]

plt.plot(np.subtract(ch1_oxy, ch2_oxy))
plt.show()
exit()
fig, axs =  plt.subplots(1, 2)
axs[0].plot(ch1_oxy, color="red")
#axs[0].plot(ch1_deoxy, color="blue")
axs[0].legend(["HbO", "HbR", ])

axs[1].plot(ch2_oxy, color="red")
#axs[1].plot(ch2_deoxy, color="blue")
axs[1].legend(["HbO", "HbR"])



plt.show()
exit()
print(run1.info)

run1.plot()
plt.show()

#CROSS CORRELATION
#time series f

#time series g



np.correlate()


exit()
run2 = read_raw_snirf(snirfs[1])
run3 = read_raw_snirf(snirfs[2])
run4 = read_raw_snirf(snirfs[3])
run5 = read_raw_snirf(snirfs[4])

data = [run1, run2, run3, run4, run5]
#WE WROTE DOWN THE ONSET OF EACH STIMULI WITH A DATAFRAME
#WE NEED TO TURN THIS INTO SECONDS
event_times_framecount = ["231", "453", "etc..."]
#DIVIDE BY THE SAMPLING RATE TO TURN THE FRAME COUNT INTO SECONDS
events_secs = np.divide(event_times_framecount, run1.info["sfreq"])
br = 40 #big rest
sr = 20 #small rest
r = 10 #right
l = 10 #left
event_description = ["rest", "right", "rest", "left", "rest", "right", "rest", "left", "rest", "right", "rest", "left", "rest"]

new_annotations = Annotations(onset=events_secs, 
                              duration = [40, 10, 20, 10, 40, 10, 20, 10, 40 , 10, 20, 10, 40], 
                              description =["rest", "right", "rest", "left", "rest", "right", "rest", "left", "rest", "right", "rest", "left", "rest"],
                              )
run1.set_annotations(new_annotations)
run1.plot()
plt.show()

runs = ["2024-11-15_001", "2024-11-15_002", "2024-11-15_003", "2024-11-15_004", "2024-11-15_005"]
conditions = ["barefoot", "easy mat", "easy mat w marker", "difficult mat"]


#set annotations