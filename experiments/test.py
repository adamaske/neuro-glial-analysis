from experiment import Experiment
from datasets.fnirs_data import find_snirf_in_folder 

exp = Experiment("path to experiment")

snirf_paths = find_snirf_in_folder(exp.data_folder)

for file in snirf_paths:


