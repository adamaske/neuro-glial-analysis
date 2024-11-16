from datasets.fnirs_data import find_snirf_in_folder
from experiments.experiment import Experiment, load_experiments
from mne.io import read_raw_snirf
import matplotlib.pyplot as plt
import datetime
from fnirs_preprocessing import preprocess

experiments = load_experiments()
experiment = Experiment(data_folder="C:/dev/neuro-glial-analysis/data/balance-15-11/2024-11-15_001")
#what experiments do we have?
print("Experiments : ", experiments)

snirfs = find_snirf_in_folder(experiments.data_folder)
print(snirfs)


r1 = read_raw_snirf(snirfs[0])
r2 = read_raw_snirf(snirfs[1])
r3 = read_raw_snirf(snirfs[2])
r4 = read_raw_snirf(snirfs[3])
r5 = read_raw_snirf(snirfs[4])

r1pp = preprocess(r1)
r2pp = preprocess(r2)
r3pp = preprocess(r3)
r4pp = preprocess(r4)
r5pp = preprocess(r5)

