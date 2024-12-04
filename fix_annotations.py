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

#THE SNRIF HAS BEEN RE-ANNOTATED
balance_task = read_raw_snirf( "data/balance-8-11/balance_task_with_triggers.snirf")
annos = balance_task._annotations
desc = balance_task.annotations.description
onset = balance_task.annotations.onset
durations = balance_task.annotations.duration
print(onset)
print(desc)
print(durations)
exit()
#FIXING TRIGGERS FOR THE BALANCE RECORDING ON 8.11.24
#MOTOR 8x8 Montage was used
#sample freq = 7.8

# PUTTING MARKERS ONTO OUR DATA -> CALLED ANNOTATIONS IN MNE
balance_task = read_raw_snirf("data/balance-8-11/balance.snirf")
sampling_frequency = balance_task.info["sfreq"]
print(sampling_frequency)
# 40 * 7.8 = 300
#20 * 7.8 = 150
#10 * 7.8 = 75, 584 + 150 = 684 + 50 = 734
#                   40    10   20   10   20    10    20 
onset = np.array([190, 500, 584, 745, 848, 1000, 1075, 1230, 1306, 1618, 1709, 1871, 1969]) #what frame did each stimuli occur
onset_seconds = np.divide(onset, sampling_frequency) #turn it into seconds (thats what both mne and satori uses for markers)
durations = [40, 10, 20, 10, 20, 10, 20, 10, 40, 10, 20, 10, 40] #duration of each stimuli
event_description = ["rest", "right", "rest", "left", "rest", "right", "rest", "left", "rest" , "right", "rest", "left", "rest"] #order of stimulis / markers

new_annotations = Annotations(onset=onset_seconds,
                              duration=durations,
                              description=event_description)
balance_task.set_annotations(new_annotations)

#THE SNRIF HAS BEEN RE-ANNOTATED
write_raw_snirf(balance_task, "data/balance-8-11/balance_task_with_triggers.snirf")

exit()