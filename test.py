import matplotlib.pyplot as plt
import datetime

from mne.io import read_raw_snirf
from mne.annotations import Annotations
from fnirs_preprocessing import preprocess, annotations
from datasets.fnirs_data import find_snirf_in_folder
from experiments.experiment import Experiment, load_experiments


#experiments = load_experiments("C:/dev/neuro-glial-analysis/experiments")
experiment = Experiment(filepath="C:/dev/neuro-glial-analysis/experiments/exp01.exp")
snirfs = find_snirf_in_folder(experiment.data_folder)

raw = read_raw_snirf(snirfs[0])
new_annotations = Annotations(
    [40, 70, 100], 
    [8, 8, 8], 
    ["Right", "Left", "Rest"]
    )
raw.set_annotations(new_annotations)
#annotations.add_annotations(raw, None)

print(raw.subject)
#raw.plot()
#plt.show()

